import uuid

import pytest
import pytest_asyncio
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from graph_rag.orchestration.types import (
    EvaluateInput,
    ExperimentInput,
    IndexCorpusInput,
    IndexInput,
    PrepareDataInput,
    TrainInput,
)
from graph_rag.orchestration.workflows import ExperimentWorkflow, IndexCorpusWorkflow

pytestmark = [pytest.mark.temporal, pytest.mark.asyncio]


def recording_activities(calls: list):
    """Stand-ins with the real activity names that record the order they ran in."""

    @activity.defn(name="prepare_data")
    async def prepare_data(params: PrepareDataInput) -> dict:
        calls.append("prepare_data")
        return {"seed": params.seed}

    @activity.defn(name="fetch_mesh_headings")
    async def fetch_mesh_headings(params: IndexInput) -> dict:
        calls.append("fetch_mesh_headings")
        return {"pmids": 1}

    @activity.defn(name="build_index")
    async def build_index(params: IndexInput) -> dict:
        calls.append("build_index")
        return {"chunks": 1, **params.overrides}

    @activity.defn(name="train_component")
    async def train_component(params: TrainInput) -> dict:
        calls.append(f"train:{params.component}")
        return {"component": params.component, "edge_types": params.edge_types}

    @activity.defn(name="evaluate_retrieval")
    async def evaluate_retrieval(params: EvaluateInput) -> dict:
        calls.append("evaluate_retrieval")
        return {"split": params.split}

    return [prepare_data, fetch_mesh_headings, build_index, train_component, evaluate_retrieval]


@pytest_asyncio.fixture
async def env():
    async with await WorkflowEnvironment.start_local() as environment:
        yield environment


async def run_workflow(env, workflow_run, params, calls):
    task_queue = str(uuid.uuid4())
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[IndexCorpusWorkflow, ExperimentWorkflow],
        activities=recording_activities(calls),
    ):
        return await env.client.execute_workflow(
            workflow_run, params, id=str(uuid.uuid4()), task_queue=task_queue
        )


async def test_index_workflow_fetches_mesh_only_when_needed(env):
    calls: list[str] = []
    result = await run_workflow(
        env, IndexCorpusWorkflow.run, IndexCorpusInput(index=IndexInput({"entities": "mesh"})), calls
    )
    assert calls == ["prepare_data", "fetch_mesh_headings", "build_index"]
    assert result["index"]["entities"] == "mesh"

    calls.clear()
    await run_workflow(
        env, IndexCorpusWorkflow.run, IndexCorpusInput(index=IndexInput({"entities": "none"})), calls
    )
    assert calls == ["prepare_data", "build_index"]


async def test_experiment_trains_gnn_first_and_evaluates_last(env):
    calls: list[str] = []
    params = ExperimentInput(train=["reranker", "gnn", "adapter"], evaluate=EvaluateInput(split="test"))
    result = await run_workflow(env, ExperimentWorkflow.run, params, calls)
    assert calls[0] == "train:gnn"
    assert set(calls[1:3]) == {"train:adapter", "train:reranker"}
    assert calls[-1] == "evaluate_retrieval"
    assert result["reranker"]["edge_types"] == "entity,next"
    assert result["evaluation"] == {"split": "test"}


async def test_experiment_rejects_unknown_components(env):
    with pytest.raises(WorkflowFailureError):
        await run_workflow(env, ExperimentWorkflow.run, ExperimentInput(train=["bogus"]), [])
