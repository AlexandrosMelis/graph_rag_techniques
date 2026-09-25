"""
Temporal activities: thin wrappers around the pipeline `run` functions.

They are synchronous (embedding, training and NCBI calls block), so the worker runs them
in a thread pool. Pipelines report progress through callbacks, which heartbeat from the
activity thread; configuration and missing-artifact errors are marked non-retryable
because retrying cannot fix them.
"""

import functools
from dataclasses import fields

from temporalio import activity
from temporalio.exceptions import ApplicationError

from graph_rag.config import MissingSettingError
from graph_rag.orchestration.types import (
    EvaluateInput,
    IndexInput,
    PrepareDataInput,
    TrainInput,
)

PERMANENT_ERRORS = (FileNotFoundError, MissingSettingError, ValueError, KeyError, TypeError)


def _non_retryable_on_permanent_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except PERMANENT_ERRORS as exc:
            raise ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True) from exc

    return wrapper


def _heartbeat(message: str) -> None:
    activity.heartbeat(message)


def _log(message: str) -> None:
    activity.logger.info(message)


def _config(cls, overrides: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**overrides)


@activity.defn
@_non_retryable_on_permanent_errors
def prepare_data(params: PrepareDataInput) -> dict:
    from graph_rag.pipelines import prepare_data as pipeline

    return pipeline.run(params.dev_fraction, params.seed, params.force_download, log=_log)


@activity.defn
@_non_retryable_on_permanent_errors
def fetch_mesh_headings(params: IndexInput) -> dict:
    from graph_rag.pipelines.build_index import IndexConfig, fetch_mesh

    return fetch_mesh(_config(IndexConfig, params.overrides), progress=_heartbeat)


@activity.defn
@_non_retryable_on_permanent_errors
def build_index(params: IndexInput) -> dict:
    from graph_rag.pipelines import build_index as pipeline

    config = _config(pipeline.IndexConfig, params.overrides)
    return pipeline.run(config, log=_log, progress=_heartbeat)


@activity.defn
@_non_retryable_on_permanent_errors
def train_component(params: TrainInput) -> dict:
    from graph_rag.pipelines import training

    if params.component in ("adapter", "graph_adapter"):
        from graph_rag.models.query_adapter import AdapterTrainingConfig

        space = "semantic" if params.component == "adapter" else "graph"
        config = _config(AdapterTrainingConfig, params.overrides)
        return training.train_adapter(space, config, log=_log, progress=_heartbeat)
    if params.component == "reranker":
        from graph_rag.models.graph_reranker import RerankerTrainingConfig

        config = _config(RerankerTrainingConfig, params.overrides)
        return training.train_reranker(
            params.edge_types or "entity,next", config, log=_log, progress=_heartbeat
        )
    if params.component == "gnn":
        from graph_rag.gnn.training import GNNTrainingConfig

        config = _config(GNNTrainingConfig, params.overrides)
        return training.train_gnn(
            params.edge_types or "entity", config, log=_log, progress=_heartbeat
        )
    raise ValueError(f"Unknown component {params.component!r}")


@activity.defn
@_non_retryable_on_permanent_errors
def evaluate_retrieval(params: EvaluateInput) -> dict:
    from graph_rag.pipelines.evaluation import evaluate_retrieval as pipeline

    result = pipeline(
        retrievers=params.retrievers,
        split=params.split,
        baseline=params.baseline,
        k_values=params.k_values,
        tune_first=params.tune,
        log=_log,
        progress=_heartbeat,
    )
    return {"output_dir": result["output_dir"], "table": result["table"]}


ALL_ACTIVITIES = [
    prepare_data,
    fetch_mesh_headings,
    build_index,
    train_component,
    evaluate_retrieval,
]
