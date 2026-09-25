"""
Interactive helpers for poking at the dataset, the entity layer of the corpus graph,
RAGAS scoring and PubMed parsing.

Usage:
    python -m graph_rag.experiments.exploration preview --split train --n 3
    python -m graph_rag.experiments.exploration entities --question "..."
    python -m graph_rag.experiments.exploration article --pmid 20007090
    python -m graph_rag.experiments.exploration ragas --question "..." --contexts-file ctx.json \
        --response "..." --reference "..."
"""

import argparse
import asyncio
import json
from pathlib import Path

from graph_rag.config import settings
from graph_rag.data.bioasq import load_questions
from graph_rag.data.pubmed import PubMedClient


def preview_questions(split: str = "train", n: int = 3) -> list[dict]:
    """Return the first `n` question records of an official split."""
    return [q.to_dict() for q in load_questions(split)[:n]]


def link_query_entities(question: str, index_dir: str = settings.index_dir) -> list[dict]:
    """Entities of the corpus graph that the linker attaches to a question, with weights."""
    from graph_rag.index.corpus_index import CorpusIndex
    from graph_rag.index.graph import CorpusGraph
    from graph_rag.index.linking import EntityLinker
    from graph_rag.retrieval.factory import encoder_from_index

    index = CorpusIndex.load(index_dir)
    graph = CorpusGraph.load(Path(index_dir) / "graph")
    links = EntityLinker(graph, encoder_from_index(index)).link(question)
    df = graph.entity_document_frequency()
    return [
        {"entity": graph.entity_names[e], "weight": w, "chunks": int(df[e])}
        for e, w in sorted(links.items(), key=lambda item: -item[1])
    ]


def fetch_article(pmid: str) -> dict:
    """Fetch and parse a single MEDLINE record."""
    records = PubMedClient().fetch_records([pmid])
    return records[0] if records else {}


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
    entities = sub.add_parser("entities")
    entities.add_argument("--question", required=True)
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
    elif args.command == "entities":
        result = link_query_entities(args.question)
    elif args.command == "article":
        result = fetch_article(args.pmid)
    else:
        with open(args.contexts_file, encoding="utf-8") as f:
            contexts = json.load(f)
        result = ragas_single_sample(args.question, contexts, args.response, args.reference)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
