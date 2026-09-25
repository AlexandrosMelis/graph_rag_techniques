"""
Train the GNN node encoder by link prediction and store node embeddings in the index.

The default pretext edges are entity co-mentions, which the text embedding does not
encode. `--edge-types knn` reproduces the embedding-similarity graph as an ablation;
its cosine-baseline AUC will be ~1.0.

Usage:
    python -m graph_rag.pipelines.train_gnn [--edge-types entity,next] [--epochs 300]
"""

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

from graph_rag.config import ConfigPath
from graph_rag.gnn.data import build_node_graph, split_edges
from graph_rag.gnn.inference import compute_node_embeddings
from graph_rag.gnn.training import GNNTrainingConfig, train_link_prediction
from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.index.graph import CorpusGraph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default=ConfigPath.INDEX_DIR)
    parser.add_argument("--edge-types", default="entity")
    parser.add_argument("--output-dir", default=os.path.join(ConfigPath.MODELS_DIR, "gnn"))
    for f in fields(GNNTrainingConfig):
        parser.add_argument(
            f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default
        )
    args = parser.parse_args()

    config = GNNTrainingConfig(**{f.name: getattr(args, f.name) for f in fields(GNNTrainingConfig)})
    edge_types = tuple(args.edge_types.split(","))
    index = CorpusIndex.load(args.index_dir)
    graph = CorpusGraph.load(Path(args.index_dir) / "graph")

    data = build_node_graph(index, graph, edge_types)
    print(f"graph for GNN: {data.num_nodes} nodes, {data.edge_index.shape[1]} directed edges")
    train, val, test = split_edges(data)
    encoder, history = train_link_prediction(train, val, test, config)

    index.graph_embeddings = compute_node_embeddings(encoder, data)
    index.metadata["graph_embedding_edge_types"] = list(edge_types)
    index.save(args.index_dir)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    history["edge_types"] = list(edge_types)
    (Path(args.output_dir) / "history.json").write_text(json.dumps(history, indent=2))
    print(f"graph embeddings written to {args.index_dir}; history in {args.output_dir}")


if __name__ == "__main__":
    main()
