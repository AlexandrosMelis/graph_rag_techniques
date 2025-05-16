import math
import os
from typing import Any, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

from utils.utils import save_json_file


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
        """
        Evaluate retrieval results for each k in k_values.

        Args:
            retrieval_results: List of dicts each with:
                {
                    'id': unique identifier,
                    'true_pmids': List[str],
                    'retrieved_pmids': List[str],
                    'retrieved_scores': List[float]  # optional
                }
            k_values: List of k cutoffs.

        Returns:
            Dictionary mapping k -> dict of metric values.
        """
        # Total relevant across dataset (for coverage)
        total_relevant = sum(
            len(example.get("true_pmids", [])) for example in retrieval_results
        )

        results: Dict[int, Dict[str, float]] = {}
        for k in k_values:
            precisions, recalls, f1s, mrrs, ndcgs, successes, aps = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            total_found = 0

            for example in retrieval_results:
                true_set: Set[str] = set(example.get("true_pmids", []))
                retrieved_list: List[str] = example.get("retrieved_pmids", [])[:k]
                retrieved_set: Set[str] = set(retrieved_list)

                # Precision@k
                precision = (
                    len(retrieved_set & true_set) / len(retrieved_list)
                    if retrieved_list
                    else 0.0
                )
                precisions.append(precision)

                # Recall@k
                recall = (
                    len(retrieved_set & true_set) / len(true_set) if true_set else 1.0
                )
                recalls.append(recall)

                # F1@k
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                f1s.append(f1)

                # MRR@k
                mrr = self._calculate_mrr(retrieved_list, true_set)
                mrrs.append(mrr)

                # nDCG@k
                ndcg = self._calculate_ndcg(retrieved_list, true_set, k)
                ndcgs.append(ndcg)

                # Success@k
                success = 1.0 if precision > 0 else 0.0
                successes.append(success)

                # AP@k (for MAP)
                ap = self._calculate_average_precision(retrieved_list, true_set)
                aps.append(ap)

                # Coverage counters
                total_found += len(retrieved_set & true_set)

            # Aggregated metrics
            coverage = total_found / total_relevant if total_relevant > 0 else 0.0
            results[k] = {
                "precision": np.mean(precisions) if precisions else 0.0,
                "recall": np.mean(recalls) if recalls else 0.0,
                "f1": np.mean(f1s) if f1s else 0.0,
                "mrr": np.mean(mrrs) if mrrs else 0.0,
                "ndcg": np.mean(ndcgs) if ndcgs else 0.0,
                "success": np.mean(successes) if successes else 0.0,
                "map": np.mean(aps) if aps else 0.0,
                "coverage": coverage,
            }

        self.metrics_per_k = results
        return results

    def _calculate_mrr(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Mean Reciprocal Rank."""
        for idx, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1.0 / idx
        return 0.0

    def _calculate_ndcg(
        self, retrieved: List[str], relevant: Set[str], k: int
    ) -> float:
        """Normalized Discounted Cumulative Gain."""
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant:
                dcg += 1.0 / math.log2(i + 2)
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def _calculate_average_precision(
        self, retrieved: List[str], relevant: Set[str]
    ) -> float:
        """Average Precision for MAP."""
        hits = 0
        sum_prec = 0.0
        for idx, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                hits += 1
                sum_prec += hits / idx
        return sum_prec / len(relevant) if relevant else 0.0

    def generate_summary_report(self, output_path: str = None) -> pd.DataFrame:
        """
        Generate a pivot table report of metrics_per_k.

        Args:
            output_path: if provided, save CSV to this path.

        Returns:
            Pandas DataFrame (metrics x k values).
        """
        if not self.metrics_per_k:
            raise ValueError("Run evaluate_retrieval first to populate metrics_per_k.")

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
    ) -> None:
        """
        Plot selected metrics across k values.
        """
        if not self.metrics_per_k:
            raise ValueError("Run evaluate_retrieval first to populate metrics_per_k.")

        df = pd.DataFrame(self.metrics_per_k).T
        ks = df.index.values
        to_plot = metrics_to_plot or [c for c in df.columns if c != "coverage"]

        plt.figure(figsize=figsize)
        for metric in to_plot:
            plt.plot(ks, df[metric], marker="o", label=metric)
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.title("Retrieval Metrics")
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
        """
        Compare a single metric at a given k across models.

        Args:
            model_results: {model_name: {k: {metric: value}}}
            metric: which metric to compare
            k: k value
        """
        data = []
        for model, res in model_results.items():
            val = res.get(k, {}).get(metric, 0.0)
            data.append({"model": model, metric: val})
        df = pd.DataFrame(data)

        # Plot
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
        """
        Calculate approximate confidence intervals for each metric at specified k.

        Args:
            retrieval_results: same format as evaluate_retrieval
            k: cutoff
            confidence: 0.95, 0.90, or 0.99
        """
        # Collect per-example metrics
        sample_metrics: Dict[str, List[float]] = {
            "precision": [],
            "recall": [],
            "f1": [],
            "mrr": [],
            "ndcg": [],
            "success": [],
            "ap": [],
        }
        for example in retrieval_results:
            true_set = set(example.get("true_pmids", []))
            retrieved = example.get("retrieved_pmids", [])[:k]
            retrieved_set = set(retrieved)
            p = len(retrieved_set & true_set) / len(retrieved) if retrieved else 0.0
            r = len(retrieved_set & true_set) / len(true_set) if true_set else 1.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            mrr = self._calculate_mrr(retrieved, true_set)
            ndcg = self._calculate_ndcg(retrieved, true_set, k)
            success = 1.0 if p > 0 else 0.0
            ap = self._calculate_average_precision(retrieved, true_set)

            sample_metrics["precision"].append(p)
            sample_metrics["recall"].append(r)
            sample_metrics["f1"].append(f1)
            sample_metrics["mrr"].append(mrr)
            sample_metrics["ndcg"].append(ndcg)
            sample_metrics["success"].append(success)
            sample_metrics["ap"].append(ap)

        # z-scores
        z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_map.get(confidence, 1.96)

        ci: Dict[str, Dict[str, float]] = {}
        for metric_name, values in sample_metrics.items():
            arr = np.array(values)
            mean = arr.mean()
            se = sem(arr) if len(arr) > 1 else 0.0
            moe = z * se
            ci[f"{metric_name}@{k}"] = {
                "mean": mean,
                "lower_bound": max(0.0, mean - moe),
                "upper_bound": min(1.0, mean + moe),
            }
        return ci


def run_evaluation(
    retrieval_results: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
    output_dir: str = None,
) -> Tuple[Dict[int, Dict[str, float]], NonLLMRetrievalEvaluator]:
    """
    Function to run evaluation and optionally save results.
    """
    evaluator = NonLLMRetrievalEvaluator()
    metrics = evaluator.calculate_evaluation_metrics(retrieval_results, k_values)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # save metrics JSON
        save_json_file(
            file_path=os.path.join(output_dir, "retrieval_metrics.json"),
            data=metrics,
        )
        # save CSV summary
        evaluator.generate_summary_report(
            output_path=os.path.join(output_dir, "retrieval_metrics_summary.csv")
        )
        # save plot
        evaluator.plot_metrics(
            output_path=os.path.join(output_dir, "retrieval_metrics_plot.png")
        )

    return metrics, evaluator
