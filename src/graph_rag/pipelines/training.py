"""
Training entry points for the learned components: query adapter (semantic or GNN space),
graph re-ranker and GNN node encoder. Each run is tracked in MLflow and returns a
JSON-serializable summary.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from graph_rag.config import settings
from graph_rag.data.splits import load_splits
from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.index.graph import CorpusGraph
from graph_rag.retrieval.factory import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_GRAPH_ADAPTER_DIR,
    DEFAULT_RERANKER_DIR,
)
from graph_rag.tracking import numeric_metrics, tracked_run

Progress = Optional[Callable[[str], None]]


def _epoch_callback(tracker, progress: Progress, label: str):
    def on_epoch(epoch: int, metrics: dict) -> None:
        tracker.log_metrics(metrics, step=epoch)
        if progress:
            progress(f"{label} epoch {epoch}: {metrics}")

    return on_epoch


def train_adapter(
    space: str = "semantic",
    config=None,
    index_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
    progress: Progress = None,
) -> dict:
    from graph_rag.models.query_adapter import AdapterTrainingConfig, train_query_adapter
    from graph_rag.retrieval.factory import encoder_from_index

    config = config or AdapterTrainingConfig()
    index_dir = Path(index_dir or settings.index_dir)
    default_dir = DEFAULT_ADAPTER_DIR if space == "semantic" else DEFAULT_GRAPH_ADAPTER_DIR
    output_dir = Path(output_dir or default_dir)
    splits = load_splits()
    index = CorpusIndex.load(index_dir)

    params = {"space": space, "embedding_model": index.embedding_model, **asdict(config)}
    with tracked_run(f"train_query_adapter[{space}]", params=params) as tracker:
        adapter, history = train_query_adapter(
            splits.train,
            splits.dev,
            index,
            encoder_from_index(index),
            config,
            space=space,
            log=log,
            on_epoch=_epoch_callback(tracker, progress, "adapter"),
        )
        adapter.save(output_dir, extra={"space": space, "embedding_model": index.embedding_model})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        summary = {
            "output_dir": str(output_dir),
            "identity_dev_recall": history["dev_recall"][0],
            "best_dev_recall": history["best_dev_recall"],
        }
        tracker.log_metrics(numeric_metrics(summary))
        tracker.log_artifacts(output_dir, "model")
        log(f"adapter saved to {output_dir} (best dev recall={summary['best_dev_recall']:.4f})")
        return summary


def train_reranker(
    edge_types: str = "entity,next",
    config=None,
    max_train_queries: Optional[int] = None,
    index_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
    progress: Progress = None,
) -> dict:
    from graph_rag.models.graph_reranker import (
        CandidateGraphBuilder,
        RerankerTrainingConfig,
        train_graph_reranker,
    )
    from graph_rag.retrieval.factory import RetrieverFactory, encoder_from_index

    config = config or RerankerTrainingConfig()
    index_dir = Path(index_dir or settings.index_dir)
    output_dir = Path(output_dir or DEFAULT_RERANKER_DIR)
    splits = load_splits()
    index = CorpusIndex.load(index_dir)
    graph = CorpusGraph.load(index_dir / "graph")
    encoder = encoder_from_index(index)
    factory = RetrieverFactory(index, encoder, graph)
    builder = CandidateGraphBuilder(
        index, graph, encoder, factory.linker(), edge_types=tuple(edge_types.split(","))
    )
    train_questions = splits.train[:max_train_queries] if max_train_queries else splits.train

    params = {"edge_types": edge_types, "train_queries": len(train_questions), **asdict(config)}
    with tracked_run("train_graph_reranker", params=params) as tracker:
        if progress:
            progress("building candidate graphs")
        model, history = train_graph_reranker(
            train_questions,
            splits.dev,
            factory.hybrid(),
            builder,
            config,
            log=log,
            on_epoch=_epoch_callback(tracker, progress, "reranker"),
        )
        model.save(
            output_dir,
            extra={
                "candidate_k": config.candidate_k,
                "edge_types": list(builder.edge_types),
                "embedding_model": index.embedding_model,
            },
        )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        summary = {
            "output_dir": str(output_dir),
            "first_stage_dev_recall": history["first_stage_dev_recall"],
            "best_dev_recall": history["best_dev_recall"],
        }
        tracker.log_metrics(numeric_metrics(summary))
        tracker.log_artifacts(output_dir, "model")
        log(
            f"re-ranker saved to {output_dir} (dev recall: first stage "
            f"{summary['first_stage_dev_recall']:.4f}, re-ranked {summary['best_dev_recall']:.4f})"
        )
        return summary


def train_gnn(
    edge_types: str = "entity",
    config=None,
    index_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
    progress: Progress = None,
) -> dict:
    """
    Link prediction on the chosen edge types, then GNN node embeddings are written into
    the index as a second vector space. `edge_types="knn"` is the RQ0 ablation.
    """
    from graph_rag.gnn.data import build_node_graph, split_edges
    from graph_rag.gnn.inference import compute_node_embeddings
    from graph_rag.gnn.training import GNNTrainingConfig, train_link_prediction

    config = config or GNNTrainingConfig()
    index_dir = Path(index_dir or settings.index_dir)
    output_dir = Path(output_dir or settings.models_dir / "gnn")
    types = tuple(edge_types.split(","))
    index = CorpusIndex.load(index_dir)
    graph = CorpusGraph.load(index_dir / "graph")

    with tracked_run("train_gnn", params={"edge_types": edge_types, **asdict(config)}) as tracker:
        data = build_node_graph(index, graph, types)
        log(f"graph for GNN: {data.num_nodes} nodes, {data.edge_index.shape[1]} directed edges")
        train, val, test = split_edges(data)
        encoder, history = train_link_prediction(
            train, val, test, config, log=log, on_epoch=_epoch_callback(tracker, progress, "gnn")
        )
        index.graph_embeddings = compute_node_embeddings(encoder, data)
        index.metadata["graph_embedding_edge_types"] = list(types)
        index.save(index_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        history["edge_types"] = list(types)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        summary = {
            key: history[key]
            for key in (
                "best_val_auc",
                "test_auc",
                "feature_cosine_val_auc",
                "feature_cosine_test_auc",
            )
        }
        tracker.log_metrics(summary)
        tracker.log_artifacts(output_dir, "history")
        return {"output_dir": str(output_dir), **summary}
