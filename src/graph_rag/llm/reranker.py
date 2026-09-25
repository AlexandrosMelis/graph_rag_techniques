from typing import Optional, Sequence

import numpy as np

from graph_rag.llm.embeddings import resolve_device

# PubMedBERT cross-encoder trained on PubMed search logs; use BAAI/bge-reranker-v2-m3 or
# Qwen/Qwen3-Reranker-0.6B-style models for general-domain corpora.
DEFAULT_CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly; the quality upper bound for re-ranking."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: Optional[str] = "auto",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=resolve_device(device), max_length=max_length)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, passage) for passage in passages]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32).reshape(-1)
