"""
Interactive helpers for poking at the dataset, the MeSH layer of the graph, RAGAS
scoring and PubMed parsing.

Usage:
    python -m graph_rag.experiments.exploration preview --split train --n 3
    python -m graph_rag.experiments.exploration mesh --question "..." --k 5
    python -m graph_rag.experiments.exploration article --pmid 20007090
    python -m graph_rag.experiments.exploration ragas --question "..." --contexts-file ctx.json \
        --response "..." --reference "..."
"""

import argparse
import asyncio
import json
import os

from Bio import Entrez

from graph_rag.config import ConfigEnv, ConfigPath
from graph_rag.data.bioasq import BioASQDataReader
from graph_rag.data.pubmed import PubMedArticleFetcher


def preview_questions(split: str = "train", n: int = 3) -> list[dict]:
    """Return the first `n` BioASQ question records of a local parquet split."""
    reader = BioASQDataReader(samples_end=n)
    path = os.path.join(ConfigPath.RAW_DATA_DIR, f"bioasq_{split}.parquet")
    return reader.read_parquet_file(file_path=path)


def top_mesh_terms(question: str, k: int = 5) -> list[dict]:
    """Rank MESH nodes by cosine similarity between the question and the MeSH definition."""
    from graph_rag.graph.connection import Neo4jConnection
    from graph_rag.llm.embeddings import EmbeddingModel

    connection = Neo4jConnection(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
    )
    embedding = EmbeddingModel().embed_query(question)
    cypher = """
    MATCH (m:MESH) WHERE m.embedding IS NOT NULL
    WITH m, vector.similarity.cosine($embedding, m.embedding) AS score
    ORDER BY score DESC
    LIMIT $k
    RETURN m.name AS term, m.definition AS definition, score
    """
    try:
        return connection.execute_query(cypher, {"embedding": embedding, "k": k})
    finally:
        connection.close()


def fetch_article(pmid: str) -> dict:
    """Fetch and parse a single MEDLINE record without touching the cached dataset files."""
    Entrez.email = ConfigEnv.ENTREZ_EMAIL
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
    try:
        text = handle.read()
    finally:
        handle.close()
    return PubMedArticleFetcher().extract_pubmed_data(text)


def ragas_single_sample(
    question: str,
    contexts: list[str],
    response: str,
    reference: str,
    provider: str = "google",
    model_name: str = "gemini-2.0-flash-lite",
) -> dict:
    """Score one (question, contexts, response, reference) sample with the RAGAS metric suite."""
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import ContextRelevance

    from graph_rag.evaluation.ragas_eval import LLMBasedEvaluator
    from graph_rag.llm.chat import ChatModel
    from graph_rag.llm.embeddings import EmbeddingModel

    llm = ChatModel(provider=provider, model_name=model_name).initialize_model()
    sample = SingleTurnSample(user_input=question, retrieved_contexts=contexts)
    relevance = asyncio.run(
        ContextRelevance(llm=LangchainLLMWrapper(llm)).single_turn_ascore(sample)
    )

    evaluator = LLMBasedEvaluator(llm=llm, embedding_model=EmbeddingModel())
    suite = evaluator.evaluate_answers(
        [
            {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": response,
                "reference": reference,
            }
        ]
    )
    return {"context_relevance": relevance, "suite": suite}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preview = sub.add_parser("preview")
    preview.add_argument("--split", default="train", choices=["train", "test"])
    preview.add_argument("--n", type=int, default=3)
    mesh = sub.add_parser("mesh")
    mesh.add_argument("--question", required=True)
    mesh.add_argument("--k", type=int, default=5)
    article = sub.add_parser("article")
    article.add_argument("--pmid", required=True)
    ragas = sub.add_parser("ragas")
    ragas.add_argument("--question", required=True)
    ragas.add_argument("--contexts-file", required=True, help="JSON list of context strings")
    ragas.add_argument("--response", required=True)
    ragas.add_argument("--reference", required=True)
    args = parser.parse_args()

    if args.command == "preview":
        result = preview_questions(args.split, args.n)
    elif args.command == "mesh":
        result = top_mesh_terms(args.question, args.k)
    elif args.command == "article":
        result = fetch_article(args.pmid)
    else:
        with open(args.contexts_file, encoding="utf-8") as f:
            contexts = json.load(f)
        result = ragas_single_sample(args.question, contexts, args.response, args.reference)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
