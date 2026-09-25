"""
Train the query-conditioned graph re-ranker over hybrid (BM25 + dense) candidates.

Usage:
    python -m graph_rag.pipelines.train_graph_reranker [--edge-types entity,next] [--candidate-k 50]
"""

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

from graph_rag.config import ConfigPath
from graph_rag.data.splits import load_splits
from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.index.graph import CorpusGraph
from graph_rag.models.graph_reranker import (
    CandidateGraphBuilder,
    RerankerTrainingConfig,
    train_graph_reranker,
)
from graph_rag.retrieval.factory import DEFAULT_RERANKER_DIR, RetrieverFactory, encoder_from_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default=ConfigPath.INDEX_DIR)
    parser.add_argument("--edge-types", default="entity,next")
    parser.add_argument("--output-dir", default=str(DEFAULT_RERANKER_DIR))
    parser.add_argument("--max-train-queries", type=int, default=None)
    for f in fields(RerankerTrainingConfig):
        parser.add_argument(
            f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default
        )
    args = parser.parse_args()

    config = RerankerTrainingConfig(
        **{f.name: getattr(args, f.name) for f in fields(RerankerTrainingConfig)}
    )
    splits = load_splits()
    index = CorpusIndex.load(args.index_dir)
    graph = CorpusGraph.load(Path(args.index_dir) / "graph")
    encoder = encoder_from_index(index)
    factory = RetrieverFactory(index, encoder, graph)
    builder = CandidateGraphBuilder(
        index, graph, encoder, factory.linker(), edge_types=tuple(args.edge_types.split(","))
    )

    train_questions = (
        splits.train[: args.max_train_queries] if args.max_train_queries else splits.train
    )
    model, history = train_graph_reranker(
        train_questions, splits.dev, factory.hybrid(), builder, config
    )
    model.save(
        args.output_dir,
        extra={"candidate_k": config.candidate_k, "edge_types": list(builder.edge_types)},
    )
    (Path(args.output_dir) / "history.json").write_text(json.dumps(history, indent=2))
    print(
        f"re-ranker saved to {os.path.abspath(args.output_dir)} "
        f"(dev recall@{config.eval_k}: first stage {history['first_stage_dev_recall']:.4f}, "
        f"re-ranked {history['best_dev_recall']:.4f})"
    )


if __name__ == "__main__":
    main()
