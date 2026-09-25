"""
Sparse corpus graph aligned with a `CorpusIndex`.

Edge types, chosen so that each carries information the chunk embedding does not:
- MENTIONS: chunk -> entity (MeSH heading or NER span), weighted by entity IDF.
- NEXT: consecutive chunks of the same passage (document structure).
- SIMILAR_TO: embedding k-nearest neighbours. Optional and meant as an ablation: these
  edges are a function of the node features, so a GNN cannot learn anything from them
  that cosine similarity does not already give.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from graph_rag.index.corpus_index import TextEncoder, top_k_indices

EDGE_TYPES = ("entity", "next", "knn")


def top_k_per_row(matrix: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep the `k` largest entries of every row of a CSR matrix."""
    matrix = matrix.tocsr()
    rows, cols, vals = [], [], []
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        if start == end:
            continue
        data, idx = matrix.data[start:end], matrix.indices[start:end]
        keep = top_k_indices(data, k)
        rows.append(np.full(len(keep), i))
        cols.append(idx[keep])
        vals.append(data[keep])
    if not rows:
        return sp.csr_matrix(matrix.shape, dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=matrix.shape,
        dtype=np.float32,
    )


def symmetrize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Undirected version of a weighted graph (elementwise max of A and A^T), no self-loops."""
    sym = matrix.maximum(matrix.T).tocsr()
    sym.setdiag(0)
    sym.eliminate_zeros()
    return sym


def knn_edges(
    embeddings: np.ndarray, k: int, block_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbours of every row (self excluded), computed in row blocks."""
    n = embeddings.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    src, dst, weights = [], [], []
    for start in range(0, n, block_size):
        block = embeddings[start : start + block_size] @ embeddings.T
        rows = np.arange(block.shape[0])
        block[rows, start + rows] = -np.inf
        nbrs = np.argpartition(-block, k - 1, axis=1)[:, :k]
        src.append(np.repeat(start + rows, k))
        dst.append(nbrs.ravel())
        weights.append(np.take_along_axis(block, nbrs, axis=1).ravel())
    edge_index = np.vstack([np.concatenate(src), np.concatenate(dst)]).astype(np.int64)
    return edge_index, np.concatenate(weights).astype(np.float32)


@dataclass
class CorpusGraph:
    n_chunks: int
    entity_names: np.ndarray
    entity_labels: np.ndarray
    mentions: sp.csr_matrix  # (n_chunks, n_entities), IDF-weighted, one entry per pair
    next_edges: np.ndarray  # (2, E), chunk i -> chunk i+1 of the same passage
    knn_edge_index: Optional[np.ndarray] = None  # (2, E)
    knn_weights: Optional[np.ndarray] = None
    entity_embeddings: Optional[np.ndarray] = None

    @property
    def n_entities(self) -> int:
        return len(self.entity_names)

    def entity_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.entity_names.tolist())}

    def entity_document_frequency(self) -> np.ndarray:
        return np.asarray((self.mentions > 0).sum(axis=0)).ravel()

    def _normalized_mentions(self) -> sp.csr_matrix:
        mentions = self.mentions.tocsr().astype(np.float32)
        if mentions.nnz:
            mentions = mentions / mentions.data.max()
        return sp.csr_matrix(mentions)

    def next_adjacency(self) -> sp.csr_matrix:
        src, dst = self.next_edges
        adj = sp.csr_matrix(
            (np.ones(len(src), dtype=np.float32), (src, dst)), shape=(self.n_chunks, self.n_chunks)
        )
        return symmetrize(adj)

    def knn_adjacency(self) -> sp.csr_matrix:
        if self.knn_edge_index is None:
            raise ValueError("Graph was built without kNN edges (use --knn-k > 0).")
        src, dst = self.knn_edge_index
        adj = sp.csr_matrix(
            (np.clip(self.knn_weights, 0, None), (src, dst)), shape=(self.n_chunks, self.n_chunks)
        )
        return symmetrize(adj)

    def entity_cooccurrence(self, top_k: int = 20, block_size: int = 512) -> sp.csr_matrix:
        """Chunk-chunk edges weighted by the IDF mass of shared entities, top-k per chunk."""
        mentions = self._normalized_mentions()
        mentions_t = mentions.T.tocsr()
        blocks = []
        for start in range(0, self.n_chunks, block_size):
            block = (mentions[start : start + block_size] @ mentions_t).tocoo()
            off_diagonal = block.row + start != block.col
            block = sp.csr_matrix(
                (block.data[off_diagonal], (block.row[off_diagonal], block.col[off_diagonal])),
                shape=block.shape,
            )
            blocks.append(top_k_per_row(block, top_k))
        cooc = sp.vstack(blocks).tocsr() if blocks else sp.csr_matrix((0, 0))
        if cooc.nnz:
            cooc = cooc / cooc.data.max()
        return symmetrize(sp.csr_matrix(cooc))

    def chunk_adjacency(
        self, edge_types: Sequence[str] = ("entity", "next"), entity_top_k: int = 20
    ) -> sp.csr_matrix:
        """Undirected, weighted chunk-chunk adjacency (weights in [0, 1]) for the given edge types."""
        unknown = set(edge_types) - set(EDGE_TYPES)
        if unknown:
            raise ValueError(f"Unknown edge types: {sorted(unknown)}")
        adj = sp.csr_matrix((self.n_chunks, self.n_chunks), dtype=np.float32)
        if "entity" in edge_types:
            adj = adj.maximum(self.entity_cooccurrence(top_k=entity_top_k))
        if "next" in edge_types:
            adj = adj.maximum(self.next_adjacency())
        if "knn" in edge_types:
            adj = adj.maximum(self.knn_adjacency())
        return symmetrize(sp.csr_matrix(adj))

    def bipartite_adjacency(self, include_next: bool = True) -> sp.csr_matrix:
        """
        Square adjacency over chunk nodes [0, n_chunks) followed by entity nodes, used for
        personalized PageRank.
        """
        mentions = self._normalized_mentions()
        chunk_block = self.next_adjacency() if include_next else None
        if chunk_block is None:
            chunk_block = sp.csr_matrix((self.n_chunks, self.n_chunks), dtype=np.float32)
        entity_block = sp.csr_matrix((self.n_entities, self.n_entities), dtype=np.float32)
        return sp.bmat([[chunk_block, mentions], [mentions.T, entity_block]], format="csr")

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sp.save_npz(directory / "mentions.npz", self.mentions)
        np.save(directory / "next_edges.npy", self.next_edges)
        pd.DataFrame({"entity": self.entity_names, "label": self.entity_labels}).to_parquet(
            directory / "entities.parquet", index=False
        )
        if self.knn_edge_index is not None:
            np.save(directory / "knn_edge_index.npy", self.knn_edge_index)
            np.save(directory / "knn_weights.npy", self.knn_weights)
        if self.entity_embeddings is not None:
            np.save(directory / "entity_embeddings.npy", self.entity_embeddings)
        stats = {
            "n_chunks": self.n_chunks,
            "n_entities": self.n_entities,
            "n_mentions": int(self.mentions.nnz),
            "n_next_edges": int(self.next_edges.shape[1]),
            "n_knn_edges": 0 if self.knn_edge_index is None else int(self.knn_edge_index.shape[1]),
        }
        (directory / "graph.json").write_text(json.dumps(stats, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "CorpusGraph":
        directory = Path(directory)
        stats = json.loads((directory / "graph.json").read_text())
        entities = pd.read_parquet(directory / "entities.parquet")
        optional = {
            name: np.load(directory / f"{name}.npy")
            if (directory / f"{name}.npy").exists()
            else None
            for name in ("knn_edge_index", "knn_weights", "entity_embeddings")
        }
        return cls(
            n_chunks=stats["n_chunks"],
            entity_names=entities["entity"].to_numpy(dtype=object),
            entity_labels=entities["label"].to_numpy(dtype=object),
            mentions=sp.load_npz(directory / "mentions.npz").tocsr(),
            next_edges=np.load(directory / "next_edges.npy"),
            **optional,
        )


def build_next_edges(chunks: pd.DataFrame) -> np.ndarray:
    ordered = chunks.reset_index().sort_values(["pmid", "chunk_index"])
    src, dst = [], []
    for _, group in ordered.groupby("pmid", sort=False):
        positions = group["index"].to_numpy()
        src.extend(positions[:-1])
        dst.extend(positions[1:])
    return np.array([src, dst], dtype=np.int64).reshape(2, -1)


def build_corpus_graph(
    chunks: pd.DataFrame,
    entity_rows: pd.DataFrame,
    embeddings: Optional[np.ndarray] = None,
    max_entity_df: float = 0.05,
    knn_k: int = 0,
    entity_encoder: Optional[TextEncoder] = None,
) -> CorpusGraph:
    """
    Build the graph for `chunks` (row order defines node ids).

    `max_entity_df` drops entities that occur in more than this fraction of chunks (or
    this many chunks when > 1); hub entities connect everything and carry no signal.
    """
    chunks = chunks.reset_index(drop=True)
    n_chunks = len(chunks)
    chunk_pos = pd.Series(np.arange(n_chunks), index=chunks["chunk_id"].astype(str))

    rows = entity_rows[["chunk_id", "entity", "label"]].copy()
    rows["chunk_id"] = rows["chunk_id"].astype(str)
    rows = rows[rows["chunk_id"].isin(chunk_pos.index) & (rows["entity"] != "")]
    # One edge per (chunk, entity) pair, however many times it was extracted.
    rows = rows.drop_duplicates(["chunk_id", "entity"])

    limit = max_entity_df * n_chunks if max_entity_df <= 1 else max_entity_df
    df_counts = rows.groupby("entity")["chunk_id"].nunique()
    keep = df_counts[df_counts <= max(limit, 1)].index
    rows = rows[rows["entity"].isin(keep)]

    labels = rows.groupby("entity")["label"].agg(lambda s: s.mode().iat[0])
    entity_names = np.array(sorted(labels.index), dtype=object)
    entity_pos = pd.Series(np.arange(len(entity_names)), index=entity_names)
    idf = np.log1p(n_chunks / df_counts.reindex(entity_names).to_numpy(dtype=np.float64))

    r = chunk_pos.loc[rows["chunk_id"]].to_numpy()
    c = entity_pos.loc[rows["entity"]].to_numpy()
    mentions = sp.csr_matrix(
        (idf[c].astype(np.float32), (r, c)), shape=(n_chunks, len(entity_names))
    )

    knn_edge_index = knn_weights = None
    if knn_k > 0:
        if embeddings is None:
            raise ValueError("kNN edges need chunk embeddings")
        knn_edge_index, knn_weights = knn_edges(embeddings, knn_k)

    entity_embeddings = None
    if entity_encoder is not None and len(entity_names):
        entity_embeddings = entity_encoder.encode_documents(entity_names.tolist())

    return CorpusGraph(
        n_chunks=n_chunks,
        entity_names=entity_names,
        entity_labels=labels.reindex(entity_names).to_numpy(dtype=object),
        mentions=mentions,
        next_edges=build_next_edges(chunks),
        knn_edge_index=knn_edge_index,
        knn_weights=knn_weights,
        entity_embeddings=entity_embeddings,
    )
