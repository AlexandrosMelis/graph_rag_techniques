"""
BioASQ questions and passage corpus from `enelpol/rag-mini-bioasq`.

The corpus is the dataset's own `text-corpus` (40,181 PubMed passages keyed by PMID),
so it is fixed and independent of which questions are evaluated. Questions keep the
official train/test split; the dev split is carved out of train in `splits.py`.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from graph_rag.config import ConfigPath

DATASET_ID = "enelpol/rag-mini-bioasq"
# Pinned so every run reads byte-identical data.
DATASET_REVISION = "8a845907dc1cff31d42fa6f7bb9c6eef5f3ae6f6"
DATASET_FILES = {
    "train": "question-answer-passages/train-00000-of-00001.parquet",
    "test": "question-answer-passages/test-00000-of-00001.parquet",
    "corpus": "text-corpus/test-00000-of-00001.parquet",
}


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    relevant_pmids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "relevant_pmids": list(self.relevant_pmids),
        }

    @classmethod
    def from_dict(cls, record: dict) -> "Question":
        return cls(
            id=str(record["id"]),
            question=record["question"],
            answer=record.get("answer") or "",
            relevant_pmids=tuple(str(p) for p in record.get("relevant_pmids", [])),
        )


def local_path(name: str, raw_dir: str | Path = ConfigPath.RAW_DATA_DIR) -> Path:
    prefix = "bioasq_corpus" if name == "corpus" else f"bioasq_{name}"
    return Path(raw_dir) / f"{prefix}.parquet"


def download_dataset(
    raw_dir: str | Path = ConfigPath.RAW_DATA_DIR,
    revision: str = DATASET_REVISION,
    force: bool = False,
) -> dict[str, Path]:
    """Download the question splits and the corpus at a pinned revision into `raw_dir`."""
    from huggingface_hub import hf_hub_download

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, filename in DATASET_FILES.items():
        target = local_path(name, raw_dir)
        if force or not target.exists():
            cached = hf_hub_download(
                repo_id=DATASET_ID, filename=filename, repo_type="dataset", revision=revision
            )
            shutil.copyfile(cached, target)
        paths[name] = target
    return paths


def load_questions(split: str, raw_dir: str | Path = ConfigPath.RAW_DATA_DIR) -> list[Question]:
    """Load the official `train` or `test` questions."""
    if split not in ("train", "test"):
        raise ValueError(f"Unknown split {split!r}; the dataset ships 'train' and 'test'.")
    df = pd.read_parquet(local_path(split, raw_dir))
    return [
        Question(
            id=str(row.id),
            question=row.question,
            answer=row.answer or "",
            relevant_pmids=tuple(str(p) for p in row.relevant_passage_ids),
        )
        for row in df.itertuples(index=False)
    ]


def load_corpus(raw_dir: str | Path = ConfigPath.RAW_DATA_DIR) -> pd.DataFrame:
    """Load the passage corpus as a DataFrame with `pmid` and `text` columns."""
    df = pd.read_parquet(local_path("corpus", raw_dir))
    corpus = pd.DataFrame({"pmid": df["id"].astype(str), "text": df["passage"].astype(str)})
    corpus["text"] = corpus["text"].str.strip()
    corpus = corpus[corpus["text"] != ""]
    return corpus.drop_duplicates("pmid").reset_index(drop=True)
