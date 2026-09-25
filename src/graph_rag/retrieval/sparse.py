from typing import Optional

from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.retrieval.base import BaseRetriever, Hit, make_hits


class BM25Retriever(BaseRetriever):
    """Lexical BM25 over chunk texts (bm25s, sparse-matrix scoring)."""

    def __init__(
        self,
        index: CorpusIndex,
        k1: float = 1.2,
        b: float = 0.75,
        stopwords: Optional[str] = "en",
        name: str = "bm25",
    ):
        import bm25s

        self._bm25s = bm25s
        self.index = index
        self.stopwords = stopwords
        self.name = name
        self.model = bm25s.BM25(k1=k1, b=b)
        tokens = bm25s.tokenize(index.texts.tolist(), stopwords=stopwords, show_progress=False)
        self.model.index(tokens, show_progress=False)

    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        tokens = self._bm25s.tokenize([query], stopwords=self.stopwords, show_progress=False)
        k = min(top_k, self.index.n)
        indices, scores = self.model.retrieve(tokens, k=k, show_progress=False)
        matched = scores[0] > 0
        return make_hits(self.index, indices[0][matched], scores[0][matched])
