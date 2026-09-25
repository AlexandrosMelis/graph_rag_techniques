"""
End-to-end RAG evaluation: retrieve, generate an answer with an LLM, score with RAGAS.

LLM calls cost money and time, so a seeded sample of the split is used by default.

Usage:
    python -m graph_rag.pipelines.evaluate_rag --retriever hybrid --n 50 \
        --provider google --model gemini-2.0-flash-lite
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

from graph_rag.config import ConfigPath
from graph_rag.data.splits import load_splits
from graph_rag.evaluation.rag import RAGAnswerer
from graph_rag.evaluation.ragas_eval import run_evaluation_on_generated_answers
from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.index.graph import CorpusGraph
from graph_rag.llm.chat import ChatModel
from graph_rag.retrieval.factory import RETRIEVERS, RetrieverFactory, encoder_from_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=sorted(RETRIEVERS), default="hybrid")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--provider", default="google")
    parser.add_argument("--model", default="gemini-2.0-flash-lite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-dir", default=ConfigPath.INDEX_DIR)
    args = parser.parse_args()

    questions = load_splits()[args.split]
    questions = random.Random(args.seed).sample(questions, min(args.n, len(questions)))
    index = CorpusIndex.load(args.index_dir)
    graph_dir = Path(args.index_dir) / "graph"
    encoder = encoder_from_index(index)
    factory = RetrieverFactory(
        index, encoder, CorpusGraph.load(graph_dir) if graph_dir.exists() else None
    )
    llm = ChatModel(provider=args.provider, model_name=args.model).initialize_model()
    answerer = RAGAnswerer(factory.build(args.retriever), llm, index, top_k=args.top_k)

    generated = []
    for q in questions:
        output = answerer.answer(q.question)
        generated.append(
            {
                "id": q.id,
                "user_input": q.question,
                "reference": q.answer,
                "response": output["response"],
                "retrieved_contexts": output["retrieved_contexts"],
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        ConfigPath.RESULTS_DIR, f"rag_{args.retriever}_{args.split}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    Path(output_dir, "generated.json").write_text(json.dumps(generated, indent=2))
    scores = run_evaluation_on_generated_answers(
        [{k: v for k, v in g.items() if k != "id"} for g in generated],
        llm=llm,
        embedding_model=encoder,
    )
    Path(output_dir, "ragas_scores.json").write_text(str(scores))
    Path(output_dir, "config.json").write_text(json.dumps(vars(args), indent=2))
    print(scores)
    print(f"results written to {output_dir}")


if __name__ == "__main__":
    main()
