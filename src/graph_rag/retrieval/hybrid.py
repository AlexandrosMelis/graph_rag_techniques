from collections import defaultdict
from typing import Optional, Sequence

from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.retrieval.base import BaseRetriever, Hit, make_hits


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = 60, weights: Optional[Sequence[float]] = None
) -> dict[int, float]:
    """RRF: score(d) = sum_r w_r / (k + rank_r(d)), ranks starting at 1."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            fused[item] += weight / (k + rank)
    return dict(fused)


class HybridRetriever(BaseRetriever):
    """Fuses several retrievers (typically BM25 + dense) with reciprocal rank fusion."""

    def __init__(
        self,
        index: CorpusIndex,
        retrievers: Sequence[BaseRetriever],
        candidate_k: int = 100,
        rrf_k: int = 60,
        weights: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
    ):
        self.index = index
        self.retrievers = list(retrievers)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.weights = weights
        self.name = name or "hybrid(" + "+".join(r.name for r in self.retrievers) + ")"

    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        depth = max(self.candidate_k, top_k)
        rankings = [[h.chunk_idx for h in r.retrieve(query, depth)] for r in self.retrievers]
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=self.weights)
        ranked = sorted(fused.items(), key=lambda item: -item[1])[:top_k]
        return make_hits(self.index, [i for i, _ in ranked], [s for _, s in ranked])
