"""
Durable workflows for the long-running parts of the project.

IndexCorpusWorkflow: download + splits, MeSH fetch (hours of rate-limited NCBI calls,
resumable), then chunk/embed/graph. ExperimentWorkflow: GNN first (it rewrites the index
files the others read), then the adapters and the re-ranker in parallel, then evaluation.
A crashed worker resumes at the failed step instead of starting over.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from graph_rag.orchestration import activities
    from graph_rag.orchestration.types import (
        TRAINABLE,
        ExperimentInput,
        IndexCorpusInput,
    )

# Training and indexing failures are mostly deterministic and expensive to repeat.
EXPENSIVE_RETRY = RetryPolicy(maximum_attempts=3)


@workflow.defn
class IndexCorpusWorkflow:
    def __init__(self) -> None:
        self._stage = "pending"

    @workflow.query
    def stage(self) -> str:
        return self._stage

    @workflow.run
    async def run(self, params: IndexCorpusInput) -> dict:
        self._stage = "prepare_data"
        data = await workflow.execute_activity(
            activities.prepare_data,
            params.prepare,
            start_to_close_timeout=timedelta(minutes=30),
        )
        result = {"data": data}

        if params.index.overrides.get("entities", "mesh") == "mesh":
            self._stage = "fetch_mesh_headings"
            result["mesh"] = await workflow.execute_activity(
                activities.fetch_mesh_headings,
                params.index,
                start_to_close_timeout=timedelta(hours=6),
                heartbeat_timeout=timedelta(minutes=5),
            )

        self._stage = "build_index"
        result["index"] = await workflow.execute_activity(
            activities.build_index,
            params.index,
            start_to_close_timeout=timedelta(hours=12),
            heartbeat_timeout=timedelta(minutes=15),
            retry_policy=EXPENSIVE_RETRY,
        )
        self._stage = "done"
        return result


@workflow.defn
class ExperimentWorkflow:
    def __init__(self) -> None:
        self._stage = "pending"

    @workflow.query
    def stage(self) -> str:
        return self._stage

    async def _train(self, spec) -> dict:
        return await workflow.execute_activity(
            activities.train_component,
            spec,
            start_to_close_timeout=timedelta(hours=12),
            heartbeat_timeout=timedelta(minutes=15),
            retry_policy=EXPENSIVE_RETRY,
        )

    @workflow.run
    async def run(self, params: ExperimentInput) -> dict:
        unknown = set(params.train) - set(TRAINABLE)
        if unknown:
            # ApplicationError fails the workflow; other exceptions would retry the task forever.
            raise ApplicationError(f"Unknown components {sorted(unknown)}", type="ValidationError")
        results: dict = {}

        if "gnn" in params.train:
            self._stage = "train:gnn"
            results["gnn"] = await self._train(params.gnn)

        parallel = [c for c in ("adapter", "graph_adapter", "reranker") if c in params.train]
        if parallel:
            self._stage = "train:" + ",".join(parallel)
            outputs = await asyncio.gather(*(self._train(getattr(params, c)) for c in parallel))
            results.update(zip(parallel, outputs))

        self._stage = "evaluate"
        results["evaluation"] = await workflow.execute_activity(
            activities.evaluate_retrieval,
            params.evaluate,
            start_to_close_timeout=timedelta(hours=6),
            heartbeat_timeout=timedelta(minutes=15),
            retry_policy=EXPENSIVE_RETRY,
        )
        self._stage = "done"
        return results
