from typing import Optional, Protocol

import numpy as np

from graph_rag.index.corpus_index import CorpusIndex, TextEncoder
from graph_rag.retrieval.base import BaseRetriever, Hit, make_hits


class QueryTransform(Protocol):
    def transform(self, vectors: np.ndarray) -> np.ndarray: ...


class DenseRetriever(BaseRetriever):
    """
    Cosine search over chunk embeddings. An optional query adapter maps the query
    vector before search; `space="graph"` searches GNN node embeddings instead, which
    only makes sense with an adapter trained for that space.
    """

    def __init__(
        self,
        index: CorpusIndex,
        encoder: TextEncoder,
        adapter: Optional[QueryTransform] = None,
        space: str = "semantic",
        name: Optional[str] = None,
    ):
        if space == "graph" and adapter is None:
            raise ValueError("Searching the graph space needs a query adapter trained for it.")
        self.index = index
        self.encoder = encoder
        self.adapter = adapter
        self.space = space
        self.name = name or ("dense" if adapter is None else f"dense+adapter[{space}]")

    def query_vector(self, query: str) -> np.ndarray:
        vector = self.encoder.encode_query(query)
        if self.adapter is not None:
            vector = self.adapter.transform(vector[None, :])[0]
        return vector

    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        indices, scores = self.index.search(self.query_vector(query), top_k, space=self.space)
        return make_hits(self.index, indices, scores)
