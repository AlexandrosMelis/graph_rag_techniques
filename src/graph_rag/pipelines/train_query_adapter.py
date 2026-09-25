"""
Train the low-rank query adapter on train-split questions, selecting on dev.

Usage:
    python -m graph_rag.pipelines.train_query_adapter                 # semantic space
    python -m graph_rag.pipelines.train_query_adapter --space graph   # GNN space (RQ0)
"""

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

from graph_rag.config import ConfigPath
from graph_rag.data.splits import load_splits
from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.models.query_adapter import AdapterTrainingConfig, train_query_adapter
from graph_rag.retrieval.factory import encoder_from_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default=ConfigPath.INDEX_DIR)
    parser.add_argument("--space", choices=["semantic", "graph"], default="semantic")
    parser.add_argument("--output-dir", default=None)
    for f in fields(AdapterTrainingConfig):
        parser.add_argument(
            f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default
        )
    args = parser.parse_args()

    config = AdapterTrainingConfig(
        **{f.name: getattr(args, f.name) for f in fields(AdapterTrainingConfig)}
    )
    output_dir = args.output_dir or os.path.join(
        ConfigPath.MODELS_DIR, f"query_adapter_{args.space}"
    )
    splits = load_splits()
    index = CorpusIndex.load(args.index_dir)
    encoder = encoder_from_index(index)

    adapter, history = train_query_adapter(
        splits.train, splits.dev, index, encoder, config, space=args.space
    )
    adapter.save(output_dir, extra={"space": args.space, "embedding_model": index.embedding_model})
    (Path(output_dir) / "history.json").write_text(json.dumps(history, indent=2))
    print(
        f"adapter saved to {output_dir} (best dev recall@{config.eval_k}={history['best_dev_recall']:.4f})"
    )


if __name__ == "__main__":
    main()
