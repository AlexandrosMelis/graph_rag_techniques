"""`graph-rag` command line interface."""

import asyncio
import dataclasses
import json
import typing
from pathlib import Path
from typing import Optional, Union

import typer

from graph_rag.retrieval.factory import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_GRAPH_ADAPTER_DIR,
    DEFAULT_RERANKER_DIR,
    RETRIEVERS,
)

app = typer.Typer(
    name="graph-rag",
    help="Graph-augmented retrieval experiments for biomedical RAG.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Dataset download and splits.", no_args_is_help=True)
index_app = typer.Typer(help="Corpus index and graph.", no_args_is_help=True)
train_app = typer.Typer(help="Train learned components.", no_args_is_help=True)
eval_app = typer.Typer(help="Retrieval and RAG evaluation.", no_args_is_help=True)
hub_app = typer.Typer(help="Share artifacts on the Hugging Face Hub.", no_args_is_help=True)
workflow_app = typer.Typer(help="Durable runs on Temporal.", no_args_is_help=True)
for sub, name in [
    (data_app, "data"),
    (index_app, "index"),
    (train_app, "train"),
    (eval_app, "evaluate"),
    (hub_app, "hub"),
    (workflow_app, "workflow"),
]:
    app.add_typer(sub, name=name)

PARAM_HELP = "Config override as key=value, repeatable (e.g. -p epochs=10 -p lr=5e-4)."
TRUE, FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}


def _coerce(raw: str, annotation):
    if typing.get_origin(annotation) is Union:
        if raw.lower() in ("none", "null"):
            return None
        annotation = next(a for a in typing.get_args(annotation) if a is not type(None))
    if annotation is bool:
        if raw.lower() not in TRUE | FALSE:
            raise ValueError(f"expected a boolean, got {raw!r}")
        return raw.lower() in TRUE
    if annotation in (int, float, str):
        return annotation(raw)
    return raw


def parse_overrides(pairs: list[str], config_cls) -> dict:
    """Parse `key=value` strings into typed keyword arguments for a config dataclass."""
    hints = typing.get_type_hints(config_cls)
    known = {f.name for f in dataclasses.fields(config_cls)}
    overrides = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        key = key.strip().replace("-", "_")
        if not sep or key not in known:
            raise typer.BadParameter(
                f"{pair!r}: expected key=value with key in {sorted(known)}", param_hint="--param"
            )
        try:
            overrides[key] = _coerce(raw.strip(), hints[key])
        except ValueError as exc:
            raise typer.BadParameter(f"{pair!r}: {exc}", param_hint="--param") from exc
    return overrides


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _echo_json(data) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


@app.command("retrievers")
def list_retrievers() -> None:
    """List the retrievers `evaluate retrieval` can build."""
    width = max(map(len, RETRIEVERS))
    for name, description in RETRIEVERS.items():
        typer.echo(f"{name:<{width}}  {description}")


@data_app.command("prepare")
def data_prepare(
    dev_fraction: float = 0.1, seed: int = 42, force_download: bool = False
) -> None:
    """Download the dataset at a pinned revision and write train/dev/test splits."""
    from graph_rag.pipelines import prepare_data

    prepare_data.run(dev_fraction, seed, force_download)


@index_app.command("build")
def index_build(
    param: list[str] = typer.Option([], "--param", "-p", help=PARAM_HELP),
    entities: str = typer.Option("mesh", help="Entity source: mesh | gliner | none."),
    embedding_model: Optional[str] = typer.Option(None, help="Sentence-transformers model id."),
    knn_k: int = typer.Option(0, help="Embedding kNN edges per chunk (RQ0 ablation only)."),
    limit: Optional[int] = typer.Option(None, help="Only index the first N passages."),
    export_neo4j: bool = typer.Option(False, help="Also export the graph to Neo4j."),
) -> None:
    """Chunk and embed the corpus, extract entities and build the corpus graph."""
    from graph_rag.pipelines import build_index

    overrides = parse_overrides(param, build_index.IndexConfig)
    overrides.update(entities=entities, knn_k=knn_k, limit=limit, export_neo4j=export_neo4j)
    if embedding_model:
        overrides["embedding_model"] = embedding_model
    build_index.run(build_index.IndexConfig(**overrides))


@index_app.command("fetch-mesh")
def index_fetch_mesh(limit: Optional[int] = None) -> None:
    """Fetch MeSH headings for the corpus into the resumable cache (needs ENTREZ_EMAIL)."""
    from graph_rag.pipelines.build_index import IndexConfig, fetch_mesh

    _echo_json(fetch_mesh(IndexConfig(limit=limit), progress=typer.echo))


@train_app.command("adapter")
def train_adapter(
    space: str = typer.Option("semantic", help="semantic, or graph for the RQ0 ablation."),
    param: list[str] = typer.Option([], "--param", "-p", help=PARAM_HELP),
    output_dir: Optional[Path] = None,
) -> None:
    """Train the low-rank query adapter (train split, selected on dev)."""
    from graph_rag.models.query_adapter import AdapterTrainingConfig
    from graph_rag.pipelines import training

    config = AdapterTrainingConfig(**parse_overrides(param, AdapterTrainingConfig))
    _echo_json(training.train_adapter(space, config, output_dir=output_dir))


@train_app.command("reranker")
def train_reranker(
    edge_types: str = "entity,next",
    max_train_queries: Optional[int] = None,
    param: list[str] = typer.Option([], "--param", "-p", help=PARAM_HELP),
    output_dir: Optional[Path] = None,
) -> None:
    """Train the query-conditioned graph re-ranker over hybrid candidates."""
    from graph_rag.models.graph_reranker import RerankerTrainingConfig
    from graph_rag.pipelines import training

    config = RerankerTrainingConfig(**parse_overrides(param, RerankerTrainingConfig))
    _echo_json(training.train_reranker(edge_types, config, max_train_queries, output_dir=output_dir))


@train_app.command("gnn")
def train_gnn(
    edge_types: str = typer.Option("entity", help="entity, next, knn (comma-separated)."),
    param: list[str] = typer.Option([], "--param", "-p", help=PARAM_HELP),
) -> None:
    """Train the GNN node encoder and store node embeddings in the index."""
    from graph_rag.gnn.training import GNNTrainingConfig
    from graph_rag.pipelines import training

    config = GNNTrainingConfig(**parse_overrides(param, GNNTrainingConfig))
    _echo_json(training.train_gnn(edge_types, config))


@eval_app.command("retrieval")
def evaluate_retrieval(
    retrievers: str = typer.Option("bm25,dense,hybrid", help="Comma-separated retriever names."),
    split: str = typer.Option("dev", help="dev while iterating, test for final numbers."),
    baseline: str = "dense",
    k: str = typer.Option("1,5,10", help="Cutoffs, comma-separated."),
    tune: bool = typer.Option(False, help="Tune ppr/expand/entity on dev first."),
    tune_queries: int = 200,
    max_queries: Optional[int] = None,
    adapter: str = typer.Option(str(DEFAULT_ADAPTER_DIR), help="Directory or hf://user/repo."),
    graph_adapter: str = str(DEFAULT_GRAPH_ADAPTER_DIR),
    reranker: str = typer.Option(str(DEFAULT_RERANKER_DIR), help="Directory or hf://user/repo."),
    cross_encoder: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Evaluate retrievers on a split with paired tests against a baseline."""
    from graph_rag.pipelines.evaluation import evaluate_retrieval as run

    run(
        retrievers=_csv(retrievers),
        split=split,
        baseline=baseline,
        k_values=[int(v) for v in _csv(k)],
        tune_first=tune,
        tune_queries=tune_queries,
        max_queries=max_queries,
        adapter_dir=adapter,
        graph_adapter_dir=graph_adapter,
        reranker_dir=reranker,
        cross_encoder=cross_encoder,
        output_dir=output_dir,
    )


@eval_app.command("rag")
def evaluate_rag(
    retriever: str = "hybrid",
    split: str = "test",
    n: int = typer.Option(50, help="Questions sampled (LLM calls cost money)."),
    top_k: int = 5,
    provider: str = "google",
    model: str = "gemini-2.0-flash-lite",
    seed: int = 42,
) -> None:
    """Retrieve, answer with an LLM and score with RAGAS (traced in MLflow)."""
    from graph_rag.pipelines.evaluation import evaluate_rag as run

    run(retriever, split, n, top_k, provider, model, seed)


@hub_app.command("push")
def hub_push(local_dir: Path, repo_id: str, public: bool = False) -> None:
    """Upload a trained adapter or re-ranker directory (a model card is generated)."""
    from graph_rag.hub import push_artifact

    typer.echo(push_artifact(local_dir, repo_id, private=not public))


@hub_app.command("pull")
def hub_pull(ref: str = typer.Argument(..., help="hf://user/repo[@revision]")) -> None:
    """Download an artifact from the Hub and print its local path."""
    from graph_rag.hub import resolve_artifact

    typer.echo(resolve_artifact(ref))


@workflow_app.command("worker")
def workflow_worker(max_concurrent_activities: int = 2) -> None:
    """Run a Temporal worker for the graph-rag task queue."""
    from graph_rag.orchestration.worker import run_worker

    asyncio.run(run_worker(max_concurrent_activities))


@workflow_app.command("index")
def workflow_index(
    param: list[str] = typer.Option([], "--param", "-p", help=PARAM_HELP),
    dev_fraction: float = 0.1,
    seed: int = 42,
    wait: bool = False,
) -> None:
    """Start IndexCorpusWorkflow: data, MeSH fetch (if entities=mesh), index build."""
    from graph_rag.orchestration.client import start_workflow
    from graph_rag.orchestration.types import IndexCorpusInput, IndexInput, PrepareDataInput
    from graph_rag.pipelines.build_index import IndexConfig

    params = IndexCorpusInput(
        prepare=PrepareDataInput(dev_fraction=dev_fraction, seed=seed),
        index=IndexInput(overrides=parse_overrides(param, IndexConfig)),
    )
    _echo_json(asyncio.run(start_workflow(params, wait=wait)))


@workflow_app.command("experiment")
def workflow_experiment(
    train: str = typer.Option("adapter,reranker", help="gnn, adapter, graph_adapter, reranker."),
    retrievers: str = "bm25,dense,hybrid,dense_adapter,graph_reranker",
    split: str = "dev",
    baseline: str = "dense",
    tune: bool = False,
    gnn_edge_types: str = "entity",
    reranker_edge_types: str = "entity,next",
    wait: bool = False,
) -> None:
    """Start ExperimentWorkflow: train components, then evaluate."""
    from graph_rag.orchestration.client import start_workflow
    from graph_rag.orchestration.types import (
        TRAINABLE,
        EvaluateInput,
        ExperimentInput,
        TrainInput,
    )

    components = _csv(train)
    unknown = set(components) - set(TRAINABLE)
    if unknown:
        raise typer.BadParameter(f"unknown components {sorted(unknown)}", param_hint="--train")
    params = ExperimentInput(
        train=components,
        gnn=TrainInput("gnn", edge_types=gnn_edge_types),
        reranker=TrainInput("reranker", edge_types=reranker_edge_types),
        evaluate=EvaluateInput(
            split=split, retrievers=_csv(retrievers), baseline=baseline, tune=tune
        ),
    )
    _echo_json(asyncio.run(start_workflow(params, wait=wait)))


if __name__ == "__main__":
    app()
