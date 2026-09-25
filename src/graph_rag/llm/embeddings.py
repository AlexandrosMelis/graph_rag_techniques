from typing import List

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel(Embeddings):
    """
    Embedding model class for embedding documents and queries.
    Utilizes HuggingFaceEmbeddings from langchain_huggingface.
    Initialization example:
        model_name = "neuml/pubmedbert-base-embeddings"
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding_model = EmbeddingModel(model_name, model_kwargs, encode_kwargs)

    Other models:
    - `sentence-transformers/all-MiniLM-L6-v2`
    """

    def __init__(
        self,
        model_name: str = None,
        model_kwargs: dict = None,
        encode_kwargs: dict = None,
    ):
        self.model_name = (
            model_name if model_name is not None else "neuml/pubmedbert-base-embeddings"
        )
        self.model_kwargs = (
            model_kwargs if model_kwargs is not None else {"device": "cuda"}
        )
        self.encode_kwargs = (
            encode_kwargs
            if encode_kwargs is not None
            else {"normalize_embeddings": False}
        )
        self._check_device(self.model_kwargs.get("device"))
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
        )
        print(f"Embedding model initialized: {self.model_name}")

    def _check_device(self, device: str):
        if not device:
            print("No device specified, using CPU")
        elif device == "cuda":
            if torch.cuda.is_available():
                print("CUDA is available, using GPU")
            else:
                print("CUDA is not available, using CPU")
        elif device == "cpu":
            print("Using CPU")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.embed_documents(texts)

    def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.aembed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)

    def aembed_query(self, text: str) -> List[float]:
        return self.embedding_model.aembed_query(text)
