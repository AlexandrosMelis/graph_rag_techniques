import uuid
from typing import Optional

from graph_rag.config import Settings, get_settings
from graph_rag.orchestration.types import ExperimentInput, IndexCorpusInput
from graph_rag.orchestration.worker import connect
from graph_rag.orchestration.workflows import ExperimentWorkflow, IndexCorpusWorkflow


async def start_workflow(
    params: IndexCorpusInput | ExperimentInput,
    wait: bool = False,
    workflow_id: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> dict:
    """Start the workflow matching `params`; with `wait`, block until it finishes."""
    settings = settings or get_settings()
    client = await connect(settings)
    if isinstance(params, IndexCorpusInput):
        run, prefix = IndexCorpusWorkflow.run, "index-corpus"
    else:
        run, prefix = ExperimentWorkflow.run, "experiment"
    handle = await client.start_workflow(
        run,
        params,
        id=workflow_id or f"{prefix}-{uuid.uuid4().hex[:12]}",
        task_queue=settings.temporal_task_queue,
    )
    info = {"workflow_id": handle.id, "run_id": handle.result_run_id}
    if wait:
        info["result"] = await handle.result()
    return info
