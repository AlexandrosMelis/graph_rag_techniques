import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np
import pandas as pd


class TextEncoder(Protocol):
    model_name: str

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the `k` largest scores, sorted by descending score."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k == scores.shape[0]:
        return np.argsort(-scores, kind="stable")
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


@dataclass
class CorpusIndex:
    """
    Chunked corpus with L2-normalized embeddings, held in memory. Retrieval runs
    against this object; Neo4j is only an export target for exploration.

    `graph_embeddings` is an optional second vector space (GNN node embeddings)
    aligned row-by-row with `chunks`.
    """

    chunks: pd.DataFrame
    embeddings: np.ndarray
    embedding_model: str
    graph_embeddings: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.chunks = self.chunks.reset_index(drop=True)
        if len(self.chunks) != len(self.embeddings):
            raise ValueError("chunks and embeddings must have the same number of rows")
        self.embeddings = l2_normalize(self.embeddings)
        if self.graph_embeddings is not None:
            if len(self.graph_embeddings) != len(self.chunks):
                raise ValueError("graph_embeddings must align with chunks")
            self.graph_embeddings = l2_normalize(self.graph_embeddings)
        self.pmids = self.chunks["pmid"].astype(str).to_numpy()
        self.chunk_ids = self.chunks["chunk_id"].astype(str).to_numpy()
        self.texts = self.chunks["text"].astype(str).to_numpy()

    @property
    def n(self) -> int:
        return len(self.chunks)

    @property
    def dimension(self) -> int:
        return self.embeddings.shape[1]

    @classmethod
    def build(
        cls, chunks: pd.DataFrame, encoder: TextEncoder, show_progress: bool = True
    ) -> "CorpusIndex":
        embeddings = encoder.encode_documents(chunks["text"].tolist(), show_progress=show_progress)
        return cls(chunks=chunks, embeddings=embeddings, embedding_model=encoder.model_name)

    def matrix(self, space: str = "semantic") -> np.ndarray:
        if space == "semantic":
            return self.embeddings
        if space == "graph":
            if self.graph_embeddings is None:
                raise ValueError(
                    "No graph embeddings; run `python -m graph_rag.pipelines.train_gnn`."
                )
            return self.graph_embeddings
        raise ValueError(f"Unknown space {space!r}")

    def search(
        self, query_vector: np.ndarray, top_k: int, space: str = "semantic"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exhaustive cosine search. Returns (chunk indices, scores)."""
        scores = self.matrix(space) @ np.asarray(query_vector, dtype=np.float32)
        idx = top_k_indices(scores, top_k)
        return idx, scores[idx]

    def pmid_set(self) -> set[str]:
        return set(self.pmids.tolist())

    def gold_coverage(self, gold_pmid_lists: Iterable[Iterable[str]]) -> float:
        """Fraction of gold PMIDs that exist in the index: the recall ceiling."""
        present = self.pmid_set()
        gold = [str(p) for pmids in gold_pmid_lists for p in pmids]
        return float(np.mean([p in present for p in gold])) if gold else 1.0

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.chunks.to_parquet(directory / "chunks.parquet", index=False)
        np.save(directory / "embeddings.npy", self.embeddings)
        if self.graph_embeddings is not None:
            np.save(directory / "graph_embeddings.npy", self.graph_embeddings)
        meta = {"embedding_model": self.embedding_model, "n_chunks": self.n, **self.metadata}
        (directory / "index.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "CorpusIndex":
        directory = Path(directory)
        meta = json.loads((directory / "index.json").read_text())
        graph_path = directory / "graph_embeddings.npy"
        return cls(
            chunks=pd.read_parquet(directory / "chunks.parquet"),
            embeddings=np.load(directory / "embeddings.npy"),
            embedding_model=meta.pop("embedding_model"),
            graph_embeddings=np.load(graph_path) if graph_path.exists() else None,
            metadata={k: v for k, v in meta.items() if k != "n_chunks"},
        )
