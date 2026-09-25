import json
import os
import time
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from graph_rag.data.bioasq import Question
from graph_rag.evaluation.metrics import (
    NonLLMRetrievalEvaluator,
    paired_bootstrap_test,
    per_query_scores,
)
from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.retrieval.base import BaseRetriever, hits_to_pmids

# Chunks requested per wanted passage, so collapsing chunks to PMIDs still fills top_k.
CHUNK_OVERSAMPLING = 3


def run_retrieval(
    questions: Sequence[Question],
    retriever: BaseRetriever,
    top_k: int = 10,
    output_dir: Optional[str] = None,
    show_progress: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """
    Run the retriever on every question and collapse chunk hits to ranked PMIDs.
    Retriever errors propagate: a crash must never be scored as an empty ranking.
    """
    results = []
    for i, q in enumerate(
        tqdm(questions, desc=f"Retrieving [{retriever.name}]", disable=not show_progress), start=1
    ):
        if progress and i % 25 == 0:
            progress(f"{retriever.name}: {i}/{len(questions)} queries")
        start = time.perf_counter()
        hits = retriever.retrieve(q.question, top_k=top_k * CHUNK_OVERSAMPLING)
        latency_ms = (time.perf_counter() - start) * 1000
        pmids = hits_to_pmids(hits, top_k)
        best_score = {}
        for hit in hits:
            best_score.setdefault(hit.pmid, hit.score)
        results.append(
            {
                "id": q.id,
                "query": q.question,
                "true_pmids": list(q.relevant_pmids),
                "retrieved_pmids": pmids,
                "retrieved_scores": [best_score[p] for p in pmids],
                "latency_ms": latency_ms,
            }
        )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(
            os.path.join(output_dir, f"{safe_name(retriever.name)}_retrieval_results.csv"),
            index=False,
        )
    return results


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def summarize_run(
    results: list[dict],
    k_values: Sequence[int] = (1, 5, 10),
    index: Optional[CorpusIndex] = None,
    ci_k: int = 10,
) -> dict:
    """Metrics per k, bootstrap intervals at `ci_k`, latency percentiles and the recall ceiling."""
    evaluator = NonLLMRetrievalEvaluator()
    metrics = evaluator.calculate_evaluation_metrics(results, list(k_values))
    latency = np.array([r["latency_ms"] for r in results])
    summary = {
        "n_queries": len(results),
        "metrics": {str(k): v for k, v in metrics.items()},
        "confidence_intervals": evaluator.calculate_confidence_intervals(results, k=ci_k),
        "latency_ms": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "mean": float(latency.mean()),
        },
    }
    if index is not None:
        summary["recall_ceiling"] = index.gold_coverage(r["true_pmids"] for r in results)
    return summary


def compare_runs(
    baseline: list[dict],
    candidate: list[dict],
    metrics: Sequence[str] = ("recall", "ndcg", "mrr"),
    k: int = 10,
) -> dict:
    """Paired bootstrap tests of candidate vs baseline on the same queries."""
    if [r["id"] for r in baseline] != [r["id"] for r in candidate]:
        raise ValueError("Runs must cover the same queries in the same order")
    true_lists = [r["true_pmids"] for r in baseline]
    comparison = {}
    for metric in metrics:
        a = per_query_scores(metric, true_lists, [r["retrieved_pmids"] for r in baseline], k)
        b = per_query_scores(metric, true_lists, [r["retrieved_pmids"] for r in candidate], k)
        comparison[f"{metric}@{k}"] = paired_bootstrap_test(a, b)
    return comparison


def save_summary(summary: dict, output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{safe_name(name)}_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path
