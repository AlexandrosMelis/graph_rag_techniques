from functools import lru_cache
from typing import List, Optional

import numpy as np
import torch
from langchain_core.embeddings import Embeddings

DEFAULT_EMBEDDING_MODEL = "neuml/pubmedbert-base-embeddings"


def resolve_device(device: Optional[str] = None) -> str:
    """Use the requested device if it is available, otherwise fall back to CPU."""
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU")
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS is not available, using CPU")
        return "cpu"
    return device


class EmbeddingModel(Embeddings):
    """
    Sentence-transformers embedder that returns L2-normalized vectors, so a dot
    product is the cosine similarity.

    Asymmetric models need a query instruction: pass `query_prompt_name="query"` for
    Qwen3-Embedding / EmbeddingGemma, or `query_prefix="query: "` for E5-style models.
    Implements the LangChain `Embeddings` interface for RAGAS.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = "auto",
        batch_size: int = 64,
        query_prompt_name: Optional[str] = None,
        query_prefix: str = "",
        document_prefix: str = "",
        max_seq_length: Optional[int] = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.query_prompt_name = query_prompt_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.model = SentenceTransformer(model_name, device=self.device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._encode_query_cached = lru_cache(maxsize=8192)(self._encode_query)
        print(f"Embedding model initialized: {model_name} on {self.device}")

    def encode_documents(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        texts = [self.document_prefix + t for t in texts]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def _encode_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            [self.query_prefix + text],
            prompt_name=self.query_prompt_name,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        vector.setflags(write=False)
        return vector

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode_query_cached(text)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self.encode_query(t) for t in texts])

    # LangChain Embeddings interface
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encode_documents(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.encode_query(text).tolist()
