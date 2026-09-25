import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from temporalio.client import Client
from temporalio.worker import Worker

from graph_rag.config import Settings, get_settings
from graph_rag.orchestration.activities import ALL_ACTIVITIES
from graph_rag.orchestration.workflows import ExperimentWorkflow, IndexCorpusWorkflow

WORKFLOWS = [IndexCorpusWorkflow, ExperimentWorkflow]


async def connect(settings: Optional[Settings] = None) -> Client:
    settings = settings or get_settings()
    return await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)


async def run_worker(max_concurrent_activities: int = 2, settings: Optional[Settings] = None) -> None:
    """
    Poll the task queue until interrupted. Activities are GPU/CPU heavy, so only a couple
    run at once per worker; start more workers (on more machines) to scale out.
    """
    settings = settings or get_settings()
    client = await connect(settings)
    with ThreadPoolExecutor(max_workers=max_concurrent_activities) as executor:
        worker = Worker(
            client,
            task_queue=settings.temporal_task_queue,
            workflows=WORKFLOWS,
            activities=ALL_ACTIVITIES,
            activity_executor=executor,
            max_concurrent_activities=max_concurrent_activities,
        )
        await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
