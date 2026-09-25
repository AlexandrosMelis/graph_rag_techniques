from typing import Any

from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.retrieval.base import BaseRetriever

ANSWER_PROMPT = """You are a biomedical expert. Answer the question using only the numbered \
context passages. If they do not contain the answer, say so. Be concise.

Context:
{context}

Question: {question}
Answer:"""


class RAGAnswerer:
    """Retrieve top passages, then ask an LLM to answer from them."""

    def __init__(self, retriever: BaseRetriever, llm: Any, index: CorpusIndex, top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.index = index
        self.top_k = top_k

    def answer(self, question: str) -> dict:
        hits = self.retriever.retrieve(question, top_k=self.top_k)
        contexts = [str(self.index.texts[h.chunk_idx]) for h in hits]
        context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
        reply = self.llm.invoke(ANSWER_PROMPT.format(context=context, question=question))
        response = getattr(reply, "content", reply)
        return {
            "response": str(response).strip(),
            "retrieved_contexts": contexts,
            "retrieved_pmids": [h.pmid for h in hits],
        }
