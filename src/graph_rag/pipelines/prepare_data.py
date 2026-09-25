"""
Download rag-mini-bioasq at a pinned revision and write the train/dev/test splits.

Usage:
    python -m graph_rag.pipelines.prepare_data [--dev-fraction 0.1] [--seed 42]
"""

import argparse

import numpy as np

from graph_rag.data.bioasq import download_dataset, load_corpus, load_questions
from graph_rag.data.splits import make_splits, save_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    paths = download_dataset(force=args.force_download)
    print("Downloaded:", {name: str(path) for name, path in paths.items()})

    splits = make_splits(
        load_questions("train"), load_questions("test"), args.dev_fraction, args.seed
    )
    save_splits(splits)

    corpus_pmids = set(load_corpus()["pmid"])
    for name in ("train", "dev", "test"):
        questions = splits[name]
        gold = [p for q in questions for p in q.relevant_pmids]
        coverage = np.mean([p in corpus_pmids for p in gold]) if gold else 1.0
        per_query = np.mean([len(q.relevant_pmids) for q in questions])
        print(
            f"{name}: {len(questions)} questions, {per_query:.2f} gold passages/question, "
            f"{coverage:.2%} of gold passages in the corpus"
        )
    print(f"corpus: {len(corpus_pmids)} passages")


if __name__ == "__main__":
    main()
