import json

import mlflow
import pytest
import typer
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from graph_rag.cli import app, parse_overrides
from graph_rag.config import MissingSettingError, Settings
from graph_rag.evaluation.rag import RAGAnswerer
from graph_rag.hub import is_hub_ref, model_card, parse_hub_ref, resolve_artifact
from graph_rag.models.query_adapter import AdapterTrainingConfig, QueryAdapter
from graph_rag.pipelines.build_index import IndexConfig
from graph_rag.retrieval.dense import DenseRetriever
from graph_rag.tracking import NullTracker, flatten, numeric_metrics, tracked_run


def test_settings_aliases_and_required_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("NEO4J_PUBMED_DATABASE", "bioasq")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.neo4j_database == "bioasq"
    assert s.index_dir == tmp_path / "index"
    with pytest.raises(MissingSettingError, match="NEO4J_URI"):
        s.neo4j_connection_kwargs()


def test_default_tracking_store_is_local_sqlite(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.tracking_uri == f"sqlite:///{tmp_path / 'mlflow' / 'mlflow.db'}"


def test_flatten_and_numeric_metrics():
    data = {"metrics": {"10": {"recall": 0.5}}, "name": "x", "flag": True}
    assert flatten(data) == {"metrics.10.recall": 0.5, "name": "x", "flag": True}
    assert numeric_metrics(data) == {"metrics.10.recall": 0.5}


def test_tracking_disabled_uses_null_tracker(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path, tracking_enabled=False)
    with tracked_run("x", params={"a": 1}, settings=s) as tracker:
        assert isinstance(tracker, NullTracker)


def test_tracked_run_logs_params_and_metrics(tmp_path):
    s = Settings(
        _env_file=None,
        data_dir=tmp_path,
        tracking_enabled=True,
        mlflow_tracking_uri=None,
        mlflow_experiment="tests",
    )
    with tracked_run("train", params={"config": {"lr": 0.1}}, settings=s) as tracker:
        tracker.log_metrics({"dev_recall": 0.3}, step=1)
        run_id = tracker.run_id
    run = MlflowClient(tracking_uri=s.tracking_uri).get_run(run_id)
    assert run.data.params["config.lr"] == "0.1"
    assert run.data.metrics["dev_recall"] == pytest.approx(0.3)
    assert run.data.tags["pipeline"] == "train"


def test_rag_answer_is_traced_with_retrieved_documents(index, encoder, tmp_path):
    class FakeLLM:
        def invoke(self, prompt):
            return "aspirin reduces inflammation"

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/traces.db")
    # Artifacts go under tmp_path too; the default would write ./mlruns into the repo.
    experiment_id = mlflow.create_experiment(
        "rag-tracing", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    experiment = mlflow.set_experiment(experiment_id=experiment_id)
    answer = RAGAnswerer(DenseRetriever(index, encoder), FakeLLM(), index, top_k=2).answer(
        "does aspirin reduce inflammation"
    )
    assert answer["response"] == "aspirin reduces inflammation"
    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(locations=[experiment.experiment_id], return_type="list")
    assert len(traces) == 1
    spans = {span.name: span for span in traces[0].data.spans}
    assert spans["answer"].span_type == "CHAIN"
    retrieval = spans["retrieve"]
    assert retrieval.span_type == "RETRIEVER"
    assert retrieval.outputs[0]["page_content"] == answer["retrieved_contexts"][0]


def test_hub_references(tmp_path, monkeypatch):
    assert parse_hub_ref("hf://user/adapter@v1") == ("user/adapter", "v1")
    assert not is_hub_ref(tmp_path)
    assert resolve_artifact(tmp_path) == tmp_path
    with pytest.raises(ValueError):
        parse_hub_ref("hf://no-namespace")

    calls = {}

    def fake_snapshot_download(repo_id, revision, repo_type):
        calls.update(repo_id=repo_id, revision=revision, repo_type=repo_type)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    assert resolve_artifact("hf://user/adapter") == tmp_path
    assert calls == {"repo_id": "user/adapter", "revision": None, "repo_type": "model"}


def test_model_card_from_adapter_directory(tmp_path):
    QueryAdapter(dim=8, rank=2).save(
        tmp_path, extra={"embedding_model": "neuml/pubmedbert-base-embeddings"}
    )
    (tmp_path / "history.json").write_text(
        json.dumps({"best_dev_recall": 0.42, "config": {"eval_k": 10}})
    )
    card = model_card(tmp_path)
    assert "base_model: neuml/pubmedbert-base-embeddings" in card
    assert "recall@10: 0.4200" in card
    assert "enelpol/rag-mini-bioasq" in card


def test_parse_overrides_types_values():
    parsed = parse_overrides(["epochs=5", "lr=5e-4", "rank=16"], AdapterTrainingConfig)
    assert parsed == {"epochs": 5, "lr": 5e-4, "rank": 16}
    assert parse_overrides(["limit=none", "export-neo4j=true"], IndexConfig) == {
        "limit": None,
        "export_neo4j": True,
    }
    with pytest.raises(typer.BadParameter):
        parse_overrides(["bogus=1"], AdapterTrainingConfig)
    with pytest.raises(typer.BadParameter):
        parse_overrides(["epochs"], AdapterTrainingConfig)


def test_cli_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["retrievers"])
    assert result.exit_code == 0 and "graph_reranker" in result.output
    for command in (["train", "adapter", "--help"], ["workflow", "experiment", "--help"]):
        assert runner.invoke(app, command).exit_code == 0
    bad = runner.invoke(app, ["workflow", "experiment", "--train", "bogus"])
    assert bad.exit_code != 0
