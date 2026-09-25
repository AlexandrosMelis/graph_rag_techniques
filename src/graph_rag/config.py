"""Typed settings read from environment variables and `.env` (pydantic-settings)."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[2]


class MissingSettingError(EnvironmentError):
    """A setting required by the service being used is not configured."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    data_dir: Path = Field(PROJECT_DIR / "data", validation_alias="GRAPH_RAG_DATA_DIR")

    entrez_email: str | None = Field(None, validation_alias="ENTREZ_EMAIL")
    entrez_api_key: SecretStr | None = Field(None, validation_alias="ENTREZ_API_KEY")

    neo4j_uri: str | None = Field(None, validation_alias="NEO4J_URI")
    neo4j_user: str | None = Field(None, validation_alias="NEO4J_USER")
    neo4j_password: SecretStr | None = Field(None, validation_alias="NEO4J_PASSWORD")
    neo4j_database: str | None = Field(
        None, validation_alias=AliasChoices("NEO4J_DATABASE", "NEO4J_PUBMED_DATABASE")
    )

    tracking_enabled: bool = Field(True, validation_alias="GRAPH_RAG_TRACKING")
    mlflow_tracking_uri: str | None = Field(None, validation_alias="MLFLOW_TRACKING_URI")
    mlflow_experiment: str = Field("graph-rag", validation_alias="GRAPH_RAG_MLFLOW_EXPERIMENT")

    temporal_address: str = Field("localhost:7233", validation_alias="TEMPORAL_ADDRESS")
    temporal_namespace: str = Field("default", validation_alias="TEMPORAL_NAMESPACE")
    temporal_task_queue: str = Field("graph-rag", validation_alias="TEMPORAL_TASK_QUEUE")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def external_dir(self) -> Path:
        return self.data_dir / "external"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def mlflow_dir(self) -> Path:
        return self.data_dir / "mlflow"

    @property
    def tracking_uri(self) -> str:
        """MLFLOW_TRACKING_URI, or a local SQLite store under the data directory."""
        return self.mlflow_tracking_uri or f"sqlite:///{self.mlflow_dir / 'mlflow.db'}"

    def require(self, *fields: str) -> None:
        """Raise if a setting is unset. Checked where a service is used, not at import."""
        missing = [name for name in fields if not getattr(self, name)]
        if missing:
            env_names = []
            for name in missing:
                alias = type(self).model_fields[name].validation_alias
                env_names.append(alias.choices[0] if isinstance(alias, AliasChoices) else alias)
            raise MissingSettingError(f"Missing required setting(s): {', '.join(env_names)}")

    def neo4j_connection_kwargs(self) -> dict:
        self.require("neo4j_uri", "neo4j_user", "neo4j_password", "neo4j_database")
        return {
            "uri": self.neo4j_uri,
            "user": self.neo4j_user,
            "password": self.neo4j_password.get_secret_value(),
            "database": self.neo4j_database,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
