"""Download rag-mini-bioasq at a pinned revision and write the train/dev/test splits."""

from typing import Callable

import numpy as np

from graph_rag.data.bioasq import (
    DATASET_ID,
    DATASET_REVISION,
    download_dataset,
    load_corpus,
    load_questions,
)
from graph_rag.data.splits import make_splits, save_splits
from graph_rag.tracking import numeric_metrics, tracked_run


def run(
    dev_fraction: float = 0.1,
    seed: int = 42,
    force_download: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    params = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "dev_fraction": dev_fraction,
        "seed": seed,
    }
    with tracked_run("prepare_data", params=params) as tracker:
        download_dataset(force=force_download)
        splits = make_splits(load_questions("train"), load_questions("test"), dev_fraction, seed)
        save_splits(splits)

        corpus_pmids = set(load_corpus()["pmid"])
        stats: dict = {"corpus_passages": len(corpus_pmids), "splits": {}}
        for name in ("train", "dev", "test"):
            questions = splits[name]
            gold = [p for q in questions for p in q.relevant_pmids]
            stats["splits"][name] = {
                "questions": len(questions),
                "gold_per_question": float(np.mean([len(q.relevant_pmids) for q in questions])),
                "gold_coverage": float(np.mean([p in corpus_pmids for p in gold])) if gold else 1.0,
            }
            s = stats["splits"][name]
            log(
                f"{name}: {s['questions']} questions, {s['gold_per_question']:.2f} gold "
                f"passages/question, {s['gold_coverage']:.2%} of gold passages in the corpus"
            )
        log(f"corpus: {stats['corpus_passages']} passages")
        tracker.log_metrics(numeric_metrics(stats))
        tracker.log_dict(stats, "data_stats.json")
        return stats
