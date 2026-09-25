from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from graph_rag.index.corpus_index import CorpusIndex


@dataclass(frozen=True)
class Hit:
    chunk_idx: int
    chunk_id: str
    pmid: str
    score: float


class BaseRetriever(ABC):
    """Returns ranked chunks for a query. Relevance is judged per PMID downstream."""

    name: str = "retriever"

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        """Return up to `top_k` chunks, best first."""


def make_hits(index: CorpusIndex, indices: Iterable[int], scores: Iterable[float]) -> list[Hit]:
    return [
        Hit(
            chunk_idx=int(i),
            chunk_id=str(index.chunk_ids[i]),
            pmid=str(index.pmids[i]),
            score=float(s),
        )
        for i, s in zip(indices, scores)
    ]


def hits_to_pmids(hits: Iterable[Hit], top_k: Optional[int] = None) -> list[str]:
    """Collapse chunk hits to passages, keeping each PMID at its best rank."""
    pmids = list(dict.fromkeys(hit.pmid for hit in hits))
    return pmids[:top_k] if top_k is not None else pmids


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    span = values.max() - values.min()
    return np.ones_like(values) if span == 0 else (values - values.min()) / span
