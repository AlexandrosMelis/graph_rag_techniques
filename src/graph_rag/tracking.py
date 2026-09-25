"""
Experiment tracking with MLflow.

Every pipeline run opens an MLflow run: parameters, per-epoch metrics, final metrics
and output directories as artifacts. Without MLFLOW_TRACKING_URI the store is a local
SQLite database under the data directory (`mlflow ui --backend-store-uri <that uri>`).
Set GRAPH_RAG_TRACKING=false to turn tracking off; the pipelines then use a no-op tracker.
"""

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Protocol

from graph_rag.config import Settings, get_settings

MAX_PARAM_LENGTH = 6000


class Tracker(Protocol):
    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None: ...

    def log_dict(self, data: dict, filename: str) -> None: ...

    def log_artifacts(self, directory: str | Path, artifact_path: Optional[str] = None) -> None: ...

    def set_tags(self, tags: Mapping[str, Any]) -> None: ...


class NullTracker:
    run_id = None

    def log_params(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_dict(self, data, filename):
        pass

    def log_artifacts(self, directory, artifact_path=None):
        pass

    def set_tags(self, tags):
        pass


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into dotted keys: {"a": {"b": 1}} -> {"a.b": 1}."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def numeric_metrics(data: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """The numeric leaves of a nested mapping, for `log_metrics`."""
    return {
        key: float(value)
        for key, value in flatten(data, prefix).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


class MlflowTracker:
    def __init__(self, mlflow_module, run):
        self._mlflow = mlflow_module
        self.run_id = run.info.run_id

    def log_params(self, params):
        clean = {}
        for key, value in flatten(params).items():
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            clean[key] = text[:MAX_PARAM_LENGTH]
        if clean:
            self._mlflow.log_params(clean)

    def log_metrics(self, metrics, step=None):
        values = {k: float(v) for k, v in metrics.items() if v is not None}
        if values:
            self._mlflow.log_metrics(values, step=step)

    def log_dict(self, data, filename):
        self._mlflow.log_dict(data, filename)

    def log_artifacts(self, directory, artifact_path=None):
        if Path(directory).exists():
            self._mlflow.log_artifacts(str(directory), artifact_path=artifact_path)

    def set_tags(self, tags):
        self._mlflow.set_tags({k: str(v) for k, v in tags.items()})


def configure_mlflow(settings: Settings):
    """Point MLflow at the configured store and experiment; returns the mlflow module."""
    import mlflow

    mlflow.set_tracking_uri(settings.tracking_uri)
    if settings.tracking_uri.startswith("sqlite:///") and not settings.mlflow_tracking_uri:
        settings.mlflow_dir.mkdir(parents=True, exist_ok=True)
        if mlflow.get_experiment_by_name(settings.mlflow_experiment) is None:
            artifacts = settings.mlflow_dir / "artifacts"
            mlflow.create_experiment(
                settings.mlflow_experiment, artifact_location=artifacts.as_uri()
            )
    mlflow.set_experiment(settings.mlflow_experiment)
    return mlflow


@contextmanager
def tracked_run(
    name: str,
    params: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, Any]] = None,
    nested: bool = False,
    settings: Optional[Settings] = None,
) -> Iterator[Tracker]:
    settings = settings or get_settings()
    if not settings.tracking_enabled:
        yield NullTracker()
        return
    mlflow = configure_mlflow(settings)
    with mlflow.start_run(run_name=name, nested=nested) as run:
        tracker = MlflowTracker(mlflow, run)
        tracker.set_tags({"pipeline": name, **(tags or {})})
        if params:
            tracker.log_params(params)
        yield tracker
