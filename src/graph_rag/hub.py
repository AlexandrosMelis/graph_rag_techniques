"""
Share trained artifacts (query adapters, graph re-rankers) through the Hugging Face Hub.

Anywhere a model directory is expected, `hf://<user>/<repo>[@revision]` works too: the
snapshot is downloaded into the local HF cache and its path is used.
"""

import json
from pathlib import Path
from typing import Optional

from graph_rag.data.bioasq import DATASET_ID

HUB_PREFIX = "hf://"


def is_hub_ref(ref: str | Path) -> bool:
    return str(ref).startswith(HUB_PREFIX)


def parse_hub_ref(ref: str) -> tuple[str, Optional[str]]:
    """`hf://user/repo@rev` -> ("user/repo", "rev")."""
    repo = str(ref)[len(HUB_PREFIX) :]
    repo_id, _, revision = repo.partition("@")
    if repo_id.count("/") != 1:
        raise ValueError(f"Expected hf://<user>/<repo>[@revision], got {ref!r}")
    return repo_id, revision or None


def resolve_artifact(ref: str | Path) -> Path:
    if not is_hub_ref(ref):
        return Path(ref)
    from huggingface_hub import snapshot_download

    repo_id, revision = parse_hub_ref(str(ref))
    return Path(snapshot_download(repo_id=repo_id, revision=revision, repo_type="model"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def model_card(local_dir: str | Path) -> str:
    """A README with Hub metadata, built from the artifact's config and training history."""
    local_dir = Path(local_dir)
    config = _read_json(local_dir / "adapter.json") or _read_json(local_dir / "reranker.json")
    history = _read_json(local_dir / "history.json")
    kind = "query adapter" if (local_dir / "adapter.json").exists() else "graph re-ranker"
    base_model = config.get("embedding_model")
    front_matter = ["---", "library_name: pytorch", f"datasets:\n- {DATASET_ID}"]
    if base_model:
        front_matter.append(f"base_model: {base_model}")
    front_matter += ["tags:\n- retrieval\n- rag\n- graph-rag\n- biomedical", "---"]
    lines = [
        *front_matter,
        "",
        f"# graph-rag {kind}",
        "",
        f"Trained with [graph_rag_techniques](https://github.com/AlexandrosMelis/graph_rag_techniques) "
        f"on the train split of `{DATASET_ID}`; the checkpoint was selected on the dev split.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
    ]
    if "best_dev_recall" in history:
        k = history.get("config", {}).get("eval_k", 10)
        lines += ["", "## Dev results", "", f"- recall@{k}: {history['best_dev_recall']:.4f}"]
        if "first_stage_dev_recall" in history:
            lines.append(f"- first stage recall@{k}: {history['first_stage_dev_recall']:.4f}")
    return "\n".join(lines) + "\n"


def push_artifact(
    local_dir: str | Path,
    repo_id: str,
    private: bool = True,
    commit_message: str = "Upload graph-rag artifact",
) -> str:
    """Create the repo if needed, write a model card when missing, and upload the folder."""
    from huggingface_hub import HfApi

    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(local_dir)
    readme = local_dir / "README.md"
    if not readme.exists():
        readme.write_text(model_card(local_dir))
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(local_dir), repo_id=repo_id, repo_type="model", commit_message=commit_message
    )
    return str(commit.commit_url if hasattr(commit, "commit_url") else commit)
