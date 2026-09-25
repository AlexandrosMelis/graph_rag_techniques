from typing import Any

import mlflow
from mlflow.entities import SpanType

from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.retrieval.base import BaseRetriever

ANSWER_PROMPT = """You are a biomedical expert. Answer the question using only the numbered \
context passages. If they do not contain the answer, say so. Be concise.

Context:
{context}

Question: {question}
Answer:"""


class RAGAnswerer:
    """
    Retrieve top passages, then ask an LLM to answer from them. Both steps are MLflow
    trace spans (the retrieval span records the passages themselves); LangChain LLM calls
    appear as child spans when `mlflow.langchain.autolog()` is on.
    """

    def __init__(self, retriever: BaseRetriever, llm: Any, index: CorpusIndex, top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.index = index
        self.top_k = top_k

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def retrieve(self, question: str) -> list[dict]:
        hits = self.retriever.retrieve(question, top_k=self.top_k)
        return [
            {
                "id": h.chunk_id,
                "page_content": str(self.index.texts[h.chunk_idx]),
                "metadata": {"pmid": h.pmid, "score": h.score, "retriever": self.retriever.name},
            }
            for h in hits
        ]

    @mlflow.trace(span_type=SpanType.CHAIN)
    def answer(self, question: str) -> dict:
        documents = self.retrieve(question)
        contexts = [d["page_content"] for d in documents]
        context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
        reply = self.llm.invoke(ANSWER_PROMPT.format(context=context, question=question))
        response = getattr(reply, "content", reply)
        return {
            "response": str(response).strip(),
            "retrieved_contexts": contexts,
            "retrieved_pmids": [d["metadata"]["pmid"] for d in documents],
        }
