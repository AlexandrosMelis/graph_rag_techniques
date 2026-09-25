from typing import Optional, Protocol, Sequence

import numpy as np

from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.llm.reranker import CrossEncoderReranker
from graph_rag.retrieval.base import BaseRetriever, Hit


class HitScorer(Protocol):
    name: str

    def score_hits(self, query: str, hits: Sequence[Hit]) -> np.ndarray: ...


class CrossEncoderScorer:
    def __init__(self, index: CorpusIndex, model: CrossEncoderReranker):
        self.index = index
        self.model = model
        self.name = "cross-encoder"

    def score_hits(self, query: str, hits: Sequence[Hit]) -> np.ndarray:
        return self.model.score(query, [self.index.texts[h.chunk_idx] for h in hits])


class RerankingRetriever(BaseRetriever):
    """Re-scores the first stage's top `candidate_k` chunks with a second-stage scorer."""

    def __init__(
        self,
        first_stage: BaseRetriever,
        scorer: HitScorer,
        candidate_k: int = 50,
        name: Optional[str] = None,
    ):
        self.first_stage = first_stage
        self.scorer = scorer
        self.candidate_k = candidate_k
        self.name = name or f"{first_stage.name}>>{scorer.name}"

    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        candidates = self.first_stage.retrieve(query, max(self.candidate_k, top_k))
        if not candidates:
            return []
        scores = self.scorer.score_hits(query, candidates)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            Hit(
                candidates[i].chunk_idx,
                candidates[i].chunk_id,
                candidates[i].pmid,
                float(scores[i]),
            )
            for i in order
        ]
