import json
import random
from dataclasses import dataclass
from pathlib import Path

from graph_rag.config import settings
from graph_rag.data.bioasq import Question

SPLIT_NAMES = ("train", "dev", "test")


@dataclass
class Splits:
    """
    Question-level splits. `train` fits models, `dev` selects checkpoints and tunes
    hyperparameters, `test` is only read for the final numbers.
    """

    train: list[Question]
    dev: list[Question]
    test: list[Question]

    def __getitem__(self, name: str) -> list[Question]:
        if name not in SPLIT_NAMES:
            raise KeyError(name)
        return getattr(self, name)


def make_splits(
    train_questions: list[Question],
    test_questions: list[Question],
    dev_fraction: float = 0.1,
    seed: int = 42,
) -> Splits:
    """Carve a deterministic dev split out of the official train split."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1)")
    ordered = sorted(train_questions, key=lambda q: q.id)
    random.Random(seed).shuffle(ordered)
    n_dev = max(1, round(len(ordered) * dev_fraction))
    splits = Splits(train=ordered[n_dev:], dev=ordered[:n_dev], test=list(test_questions))
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: Splits) -> None:
    """Fail if any question id or question text appears in more than one split."""
    for attr in ("id", "question"):
        seen: dict[str, str] = {}
        for name in SPLIT_NAMES:
            for q in splits[name]:
                key = getattr(q, attr)
                if key in seen and seen[key] != name:
                    raise ValueError(f"Question {attr} {key!r} is in both {seen[key]} and {name}")
                seen[key] = name


def save_splits(splits: Splits, splits_dir: str | Path = settings.splits_dir) -> None:
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        with open(Path(splits_dir) / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for q in splits[name]:
                f.write(json.dumps(q.to_dict()) + "\n")


def load_splits(splits_dir: str | Path = settings.splits_dir) -> Splits:
    loaded = {}
    for name in SPLIT_NAMES:
        path = Path(splits_dir) / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run `python -m graph_rag.pipelines.prepare_data`."
            )
        with open(path, encoding="utf-8") as f:
            loaded[name] = [Question.from_dict(json.loads(line)) for line in f if line.strip()]
    splits = Splits(**loaded)
    assert_disjoint(splits)
    return splits
