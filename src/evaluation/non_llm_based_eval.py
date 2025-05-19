import math
import os
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.utils import save_json_file


def precision_at_k(
    true_lists: List[List[str]], pred_lists: List[List[str]], k: int
) -> float:
    precisions = []
    for true, pred in zip(true_lists, pred_lists):
        top_k = pred[:k]
        if not top_k:
            precisions.append(0.0)
        else:
            relevant = sum(1 for p in top_k if p in true)
            precisions.append(relevant / k)
    return float(np.mean(precisions))


def recall_at_k(
    true_lists: List[List[str]], pred_lists: List[List[str]], k: int
) -> float:
    recalls = []
    for true, pred in zip(true_lists, pred_lists):
        top_k = pred[:k]
        recalls.append(
            (sum(1 for p in top_k if p in true) / len(true)) if true else 0.0
        )
    return float(np.mean(recalls))


def f1_at_k(true_lists: List[List[str]], pred_lists: List[List[str]], k: int) -> float:
    f1s = []
    for true, pred in zip(true_lists, pred_lists):
        top_k = pred[:k]
        tp = sum(1 for p in top_k if p in true)
        precision = tp / k if k > 0 else 0.0
        recall = tp / len(true) if true else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
    return float(np.mean(f1s))


def mean_reciprocal_rank(
    true_lists: List[List[str]], pred_lists: List[List[str]], k: int
) -> float:
    rr = []
    for true, pred in zip(true_lists, pred_lists):
        score = 0.0
        for idx, p in enumerate(pred[:k], start=1):
            if p in true:
                score = 1.0 / idx
                break
        rr.append(score)
    return float(np.mean(rr))


def ndcg_at_k(
    true_lists: List[List[str]], pred_lists: List[List[str]], k: int
) -> float:
    ndcgs = []
    for true, pred in zip(true_lists, pred_lists):
        dcg = 0.0
        for i, p in enumerate(pred[:k], start=1):
            rel = 1 if p in true else 0
            dcg += (2**rel - 1) / math.log2(i + 1)
        ideal_dcg = sum(
            (2**1 - 1) / math.log2(i + 1) for i in range(1, min(len(true), k) + 1)
        )
        ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    return float(np.mean(ndcgs))


def success_at_k(
    true_lists: List[List[str]], pred_lists: List[List[str]], k: int
) -> float:
    successes = []
    for true, pred in zip(true_lists, pred_lists):
        successes.append(1.0 if any(p in true for p in pred[:k]) else 0.0)
    return float(np.mean(successes))


def coverage_at_k(
    true_lists: List[List[str]], pred_lists: List[List[str]], k: int
) -> float:
    all_true = set(pmid for true in true_lists for pmid in true)
    retrieved = set(
        p
        for pred in pred_lists
        for p in pred[:k]
        if any(p in true for true in true_lists)
    )
    return len(retrieved) / len(all_true) if all_true else 0.0


def average_precision_at_k(true: List[str], pred: List[str], k: int) -> float:
    hits = 0
    sum_prec = 0.0
    for i, p in enumerate(pred[:k], start=1):
        if p in true:
            hits += 1
            sum_prec += hits / i
    return sum_prec / hits if hits > 0 else 0.0


def map_at_k(true_lists: List[List[str]], pred_lists: List[List[str]], k: int) -> float:
    aps = [
        average_precision_at_k(true, pred, k)
        for true, pred in zip(true_lists, pred_lists)
    ]
    return float(np.mean(aps))


def bootstrap_confidence_interval(
    metric_fn: Callable[[List[List[str]], List[List[str]], int], float],
    true_lists: List[List[str]],
    pred_lists: List[List[str]],
    k: int,
    num_rounds: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    n = len(true_lists)
    scores = []
    for _ in range(num_rounds):
        idxs = np.random.randint(0, n, n)
        bs_true = [true_lists[i] for i in idxs]
        bs_pred = [pred_lists[i] for i in idxs]
        scores.append(metric_fn(bs_true, bs_pred, k))
    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return lower, upper


class NonLLMRetrievalEvaluator:
    """
    Evaluator for assessing the retrieval technique ability to correctly retrieve relevant chunks.
    Calculates various metrics relevant with chunk retrieval.

    ** Core Metrics **
    - Precision@k: Measures the percentage of retrieved chunks that are in the ground truth relevant set. It answers "Of the chunks I retrieved, how many were actually relevant?"
    - Recall@k: Measures the percentage of ground truth relevant chunks that were successfully retrieved. It answers "Of all the relevant chunks, how many did I manage to retrieve?"
    - F1 Score@k: The harmonic mean between precision and recall, providing a balanced measure of retrieval performance.
    - Mean Reciprocal Rank (MRR): Focuses on the position of the first relevant chunk in the ranking. Higher is better.
    - Normalized Discounted Cumulative Gain (nDCG@k): Considers both relevance and position in the results. It penalizes relevant chunks that appear lower in the ranking.
    - Success@k: Binary measure indicating whether at least one relevant chunk was found in the top-k results.
    - Coverage@k: Measures what proportion of all relevant passages across the entire dataset were retrieved.
    - Mean Average Precision (MAP@k): Average of precision values calculated at each position where a relevant document is found.

    Where k parameter indicates the number of top results to consider.

    ** Confidence Intervals **
    The calculate_confidence_intervals method estimates how reliable your metrics are by computing the variability across your test samples.
    """

    def __init__(self):
        self.metrics_per_k: Dict[int, Dict[str, float]] = {}

    def calculate_evaluation_metrics(
        self,
        retrieval_results: List[Dict[str, Any]],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[int, Dict[str, float]]:
        true_lists = [ex.get("true_pmids", []) for ex in retrieval_results]
        pred_lists = [ex.get("retrieved_pmids", []) for ex in retrieval_results]
        results: Dict[int, Dict[str, float]] = {}
        for k in k_values:
            results[k] = {
                "precision": precision_at_k(true_lists, pred_lists, k),
                "recall": recall_at_k(true_lists, pred_lists, k),
                "f1": f1_at_k(true_lists, pred_lists, k),
                "mrr": mean_reciprocal_rank(true_lists, pred_lists, k),
                "ndcg": ndcg_at_k(true_lists, pred_lists, k),
                "success": success_at_k(true_lists, pred_lists, k),
                "map": map_at_k(true_lists, pred_lists, k),
                "coverage": coverage_at_k(true_lists, pred_lists, k),
            }
        self.metrics_per_k = results
        return results

    def generate_summary_report(self, output_path: str = None) -> pd.DataFrame:
        if not self.metrics_per_k:
            raise ValueError(
                "Run calculate_evaluation_metrics first to populate metrics_per_k."
            )
        df = pd.DataFrame(self.metrics_per_k).T
        df.index.name = "k"
        if output_path:
            df.to_csv(output_path)
        return df

    def plot_metrics(
        self,
        metrics_to_plot: List[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        output_path: str = None,
        retriever_name: str = "test_retriever",
    ) -> None:
        if not self.metrics_per_k:
            raise ValueError(
                "Run calculate_evaluation_metrics first to populate metrics_per_k."
            )
        df = pd.DataFrame(self.metrics_per_k).T
        ks = df.index.values
        to_plot = metrics_to_plot or [c for c in df.columns if c != "coverage"]
        plt.figure(figsize=figsize)
        for metric in to_plot:
            plt.plot(ks, df[metric], marker="o", label=metric)
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.title(f"Retrieval Metrics - {retriever_name}")
        plt.xticks(ks)
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        plt.show()

    def compare_models(
        self,
        model_results: Dict[str, Dict[int, Dict[str, float]]],
        metric: str = "precision",
        k: int = 10,
        figsize: Tuple[int, int] = (12, 8),
        output_file: str = None,
    ) -> pd.DataFrame:
        data = []
        for model, res in model_results.items():
            val = res.get(k, {}).get(metric, 0.0)
            data.append({"model": model, metric: val})
        df = pd.DataFrame(data)
        plt.figure(figsize=figsize)
        plt.bar(df["model"], df[metric])
        plt.ylabel(metric)
        plt.title(f"Model Comparison on {metric}@{k}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, bbox_inches="tight")
        plt.show()
        return df

    def calculate_confidence_intervals(
        self,
        retrieval_results: List[Dict[str, Any]],
        k: int = 10,
        confidence: float = 0.95,
    ) -> Dict[str, Dict[str, float]]:
        true_lists = [ex.get("true_pmids", []) for ex in retrieval_results]
        pred_lists = [ex.get("retrieved_pmids", []) for ex in retrieval_results]
        metric_fns: Dict[str, Callable] = {
            "precision": precision_at_k,
            "recall": recall_at_k,
            "f1": f1_at_k,
            "mrr": mean_reciprocal_rank,
            "ndcg": ndcg_at_k,
            "success": success_at_k,
            "map": map_at_k,
        }
        ci: Dict[str, Dict[str, float]] = {}
        alpha = 1 - confidence
        for name, fn in metric_fns.items():
            mean = fn(true_lists, pred_lists, k)
            lower, upper = bootstrap_confidence_interval(
                fn, true_lists, pred_lists, k, alpha=alpha
            )
            ci[f"{name}@{k}"] = {
                "mean": mean,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        return ci


def run_evaluation(
    retrieval_results: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
    output_dir: str = None,
    retriever_name: str = "test_retriever",
) -> Tuple[Dict[int, Dict[str, float]], NonLLMRetrievalEvaluator]:
    evaluator = NonLLMRetrievalEvaluator()
    metrics = evaluator.calculate_evaluation_metrics(retrieval_results, k_values)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_json_file(
            file_path=os.path.join(
                output_dir, f"{retriever_name}_retrieval_metrics.json"
            ),
            data=metrics,
        )
        evaluator.generate_summary_report(
            output_path=os.path.join(
                output_dir, f"{retriever_name}_retrieval_metrics_summary.csv"
            )
        )
        evaluator.plot_metrics(
            output_path=os.path.join(
                output_dir, f"{retriever_name}_retrieval_metrics_plot.png"
            ),
            retriever_name=retriever_name,
        )

    return metrics, evaluator
