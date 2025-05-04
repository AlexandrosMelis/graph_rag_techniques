import math
import os
from collections import defaultdict
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

    def __init__(self, benchmark_data: List[Dict[str, Any]]):
        """
        Initialize evaluator with benchmark dataset.

        Args:
            benchmark_data: List of dictionaries, each containing:
                - question: str
                - answer: str
                - relevant_passage_ids: list of expected chunks that should be retrieved
        """
        self.benchmark_data = benchmark_data
        self.metrics_results = {}

    def evaluate_retrieval(
        self,
        retrieval_results: List[Dict[str, Any]],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval results against benchmark data for various k values.

        Args:
            retrieval_results: List of dictionaries, each containing:
                - question_id: ID matching a question in benchmark_data
                - retrieved_chunks: List of tuples (passage_id, score) ordered by relevance
            k_values: List of k values to evaluate metrics at

        Returns:
            Dictionary of metrics at each k value
        """
        # Create a mapping of questions to relevant passage IDs for quick lookup
        benchmark_samples = {
            sample["id"]: set(sample["relevant_passage_ids"])
            for sample in self.benchmark_data
        }

        # Create a mapping of question IDs to retrieval results
        retrieval_map = {
            result["question_id"]: result["retrieved_chunks"]
            for result in retrieval_results
        }

        # Initialize metrics dict
        metrics = defaultdict(dict)

        # Calculate metrics for each k value
        for k in k_values:
            precision_values = []
            recall_values = []
            f1_values = []
            mrr_values = []
            ndcg_values = []
            success_values = []
            ap_values = []  # For MAP calculation

            for question_id, relevant_passages in benchmark_samples.items():

                if question_id not in retrieval_map:
                    continue

                retrieved = [chunk_id for chunk_id, _ in retrieval_map[question_id][:k]]
                retrieved_set = set(retrieved)
                relevant_set = set(relevant_passages)

                # Precision@k
                precision = (
                    len(retrieved_set.intersection(relevant_set)) / len(retrieved)
                    if retrieved
                    else 0
                )
                precision_values.append(precision)

                # Recall@k
                recall = (
                    len(retrieved_set.intersection(relevant_set)) / len(relevant_set)
                    if relevant_set
                    else 1.0
                )
                recall_values.append(recall)

                # F1 Score@k
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
                f1_values.append(f1)

                # MRR
                mrr = self._calculate_mrr(retrieved, relevant_set)
                mrr_values.append(mrr)

                # nDCG@k
                ndcg = self._calculate_ndcg(retrieved, relevant_set, k)
                ndcg_values.append(ndcg)

                # Success@k
                success = 1 if len(retrieved_set.intersection(relevant_set)) > 0 else 0
                success_values.append(success)

                # Average Precision for MAP
                ap = self._calculate_average_precision(retrieved, relevant_set)
                ap_values.append(ap)

            # Store metrics for this k value
            metrics[f"precision@{k}"] = (
                np.mean(precision_values) if precision_values else 0
            )
            metrics[f"recall@{k}"] = np.mean(recall_values) if recall_values else 0
            metrics[f"f1@{k}"] = np.mean(f1_values) if f1_values else 0
            metrics[f"mrr@{k}"] = np.mean(mrr_values) if mrr_values else 0
            metrics[f"ndcg@{k}"] = np.mean(ndcg_values) if ndcg_values else 0
            metrics[f"success@{k}"] = np.mean(success_values) if success_values else 0
            metrics[f"map@{k}"] = np.mean(ap_values) if ap_values else 0

            # Calculate coverage across all samples
            total_relevant = sum(
                len(relevant) for relevant in benchmark_samples.values()
            )
            total_found = sum(
                len(
                    set(
                        [chunk_id for chunk_id, _ in retrieval_map.get(qid, [])[:k]]
                    ).intersection(relevant)
                )
                for qid, relevant in benchmark_samples.items()
            )
            metrics[f"coverage@{k}"] = (
                total_found / total_relevant if total_relevant > 0 else 0
            )

        self.metrics_results = metrics
        return metrics

    def _calculate_mrr(self, retrieved: List[str], relevant_set: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, chunk_id in enumerate(retrieved):
            if chunk_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_ndcg(
        self, retrieved: List[str], relevant_set: Set[str], k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved):
            if chunk_id in relevant_set:
                # Using binary relevance (1 if relevant, 0 if not)
                dcg += 1.0 / np.log2(i + 2)  # +2 because log_2(1) = 0

        # Calculate ideal DCG
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))

        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def _calculate_average_precision(
        self, retrieved: List[str], relevant_set: Set[str]
    ) -> float:
        """Calculate Average Precision for MAP."""
        hits = 0
        sum_precisions = 0.0

        for i, chunk_id in enumerate(retrieved):
            if chunk_id in relevant_set:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i

        return sum_precisions / len(relevant_set) if relevant_set else 0.0

    def generate_summary_report(self, output_path: str = None) -> pd.DataFrame:
        """
        Generate a summary report of all metrics.

        Args:
            output_path: Optional path to save CSV file

        Returns:
            DataFrame with metrics summary
        """
        if not self.metrics_results:
            raise ValueError(
                "No evaluation results available. Run evaluate_retrieval first."
            )

        # Convert metrics to DataFrame for easier visualization
        metrics_df = pd.DataFrame([self.metrics_results])
        metrics_df = metrics_df.T.reset_index()
        metrics_df.columns = ["Metric", "Value"]

        # Extract k value and base metric name
        metrics_df[["Metric", "k"]] = metrics_df["Metric"].str.split("@", expand=True)
        metrics_df["k"] = metrics_df["k"].astype(int)

        # Pivot to get metrics by k value
        pivot_df = metrics_df.pivot(index="Metric", columns="k", values="Value")

        if output_path:
            pivot_df.to_csv(output_path)

        return pivot_df

    def plot_metrics(
        self,
        metrics_to_plot: List[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        output_path: str = None,
    ) -> None:
        """
        Plot metrics against k values.

        Args:
            metrics_to_plot: List of metrics to plot (e.g., ['precision', 'recall'])
            figsize: Figure size as (width, height)
            output_file: Optional path to save plot
        """
        if not self.metrics_results:
            raise ValueError(
                "No evaluation results available. Run evaluate_retrieval first."
            )

        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame([self.metrics_results])
        metrics_df = metrics_df.T.reset_index()
        metrics_df.columns = ["Metric", "Value"]

        # Extract k value and base metric name
        metrics_df[["Base", "k"]] = metrics_df["Metric"].str.split("@", expand=True)
        metrics_df["k"] = metrics_df["k"].astype(int)

        # Filter metrics to plot if specified
        if metrics_to_plot:
            metrics_df = metrics_df[metrics_df["Base"].isin(metrics_to_plot)]

        # Get unique k values and base metrics
        k_values = sorted(metrics_df["k"].unique())
        base_metrics = metrics_df["Base"].unique()

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        for metric in base_metrics:
            metric_data = metrics_df[metrics_df["Base"] == metric]
            ax.plot(metric_data["k"], metric_data["Value"], marker="o", label=metric)

        ax.set_xlabel("k value")
        ax.set_ylabel("Score")
        ax.set_title("Retrieval Metrics at Different k Values")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Set x-ticks to k values
        ax.set_xticks(k_values)

        if output_path:
            plt.savefig(output_path, bbox_inches="tight")

        plt.tight_layout()
        plt.show()

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics_to_compare: List[str] = None,
        k_value: int = 10,
        figsize: Tuple[int, int] = (12, 8),
        output_file: str = None,
    ) -> pd.DataFrame:
        """
        Compare different models based on their metrics.

        Args:
            model_results: Dictionary mapping model names to their metric results
            metrics_to_compare: List of metrics to compare (e.g., ['precision@10', 'recall@10'])
            k_value: k value to use for comparison
            figsize: Figure size as (width, height)
            output_file: Optional path to save plot

        Returns:
            DataFrame with model comparison
        """
        # Create comparison DataFrame
        models = list(model_results.keys())

        if not metrics_to_compare:
            # Use all metrics with specified k value
            metrics_to_compare = [
                key for key in model_results[models[0]].keys() if f"@{k_value}" in key
            ]

        # Create comparison data
        comparison_data = []
        for metric in metrics_to_compare:
            metric_values = [model_results[model].get(metric, 0) for model in models]
            comparison_data.append(
                {
                    "Metric": metric,
                    **{model: value for model, value in zip(models, metric_values)},
                }
            )

        comparison_df = pd.DataFrame(comparison_data)

        # Plot comparison
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(metrics_to_compare))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            values = [row[model] for row in comparison_data]
            ax.bar(x + i * width - 0.4 + width / 2, values, width, label=model)

        ax.set_ylabel("Score")
        ax.set_title(f"Model Comparison at k={k_value}")
        ax.set_xticks(x)
        ax.set_xticklabels([metric.split("@")[0] for metric in metrics_to_compare])
        ax.legend()

        if output_file:
            plt.savefig(output_file, bbox_inches="tight")

        plt.tight_layout()
        plt.show()

        return comparison_df

    def calculate_confidence_intervals(
        self,
        retrieval_results: List[Dict[str, Any]],
        k_value: int = 10,
        confidence: float = 0.95,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals for metrics using bootstrap resampling.

        Args:
            retrieval_results: List of dictionaries with retrieval results
            k_value: k value to use for metrics
            confidence: Confidence level (e.g., 0.95 for 95% confidence)

        Returns:
            Dictionary of metrics with their confidence intervals
        """
        # Create a mapping of questions to relevant passage IDs
        question_to_relevant = {
            i: set(sample["relevant_passage_ids"])
            for i, sample in enumerate(self.benchmark_data)
        }

        # Create a mapping of question IDs to retrieval results
        retrieval_map = {
            result["question_id"]: result["retrieved_chunks"]
            for result in retrieval_results
        }

        # Calculate metrics for each sample
        sample_metrics = []

        for question_id, relevant_passages in question_to_relevant.items():
            if question_id not in retrieval_map:
                continue

            retrieved = [
                chunk_id for chunk_id, _ in retrieval_map[question_id][:k_value]
            ]
            retrieved_set = set(retrieved)
            relevant_set = set(relevant_passages)

            # Calculate sample-level metrics
            sample_metric = {}

            # Precision@k
            sample_metric["precision"] = (
                len(retrieved_set.intersection(relevant_set)) / len(retrieved)
                if retrieved
                else 0
            )

            # Recall@k
            sample_metric["recall"] = (
                len(retrieved_set.intersection(relevant_set)) / len(relevant_set)
                if relevant_set
                else 1.0
            )

            # F1 Score@k
            sample_metric["f1"] = (
                2
                * sample_metric["precision"]
                * sample_metric["recall"]
                / (sample_metric["precision"] + sample_metric["recall"])
                if (sample_metric["precision"] + sample_metric["recall"]) > 0
                else 0
            )

            # MRR
            sample_metric["mrr"] = self._calculate_mrr(retrieved, relevant_set)

            # nDCG@k
            sample_metric["ndcg"] = self._calculate_ndcg(
                retrieved, relevant_set, k_value
            )

            # Success@k
            sample_metric["success"] = (
                1 if len(retrieved_set.intersection(relevant_set)) > 0 else 0
            )

            # AP for MAP
            sample_metric["ap"] = self._calculate_average_precision(
                retrieved, relevant_set
            )

            sample_metrics.append(sample_metric)

        # Convert to DataFrame for easier calculation
        metrics_df = pd.DataFrame(sample_metrics)

        # Calculate mean and standard error for confidence intervals
        confidence_intervals = {}

        for metric in metrics_df.columns:
            mean_value = metrics_df[metric].mean()
            std_error = sem(metrics_df[metric])

            # t-value for the given confidence level and degrees of freedom
            # Using z-score approximation for large sample sizes
            z_score = 1.96  # For 95% confidence
            if confidence != 0.95:
                # Approximate z-score for other confidence levels
                z_score = -math.inf
                if confidence == 0.90:
                    z_score = 1.645
                elif confidence == 0.99:
                    z_score = 2.576

            margin_of_error = z_score * std_error

            confidence_intervals[f"{metric}@{k_value}"] = {
                "mean": mean_value,
                "lower_bound": max(0, mean_value - margin_of_error),
                "upper_bound": min(1, mean_value + margin_of_error),
            }

        return confidence_intervals


def format_retrieval_results(
    raw_results: Dict[str, List[Tuple[str, float]]],
) -> List[Dict[str, Any]]:
    """
    Format raw retrieval results into the structure expected by the evaluator.

    Args:
        raw_results: Dictionary mapping question IDs to lists of (passage_id, score) tuples

    Returns:
        List of formatted retrieval results
    """
    formatted_results = []
    for question_id, retrieved_chunks in raw_results.items():
        formatted_results.append(
            {"question_id": question_id, "retrieved_chunks": retrieved_chunks}
        )
    return formatted_results


def run_evaluation_on_retrieved_chunks(
    benchmark_data: List[Dict[str, Any]],
    retrieval_results: Dict[str, List[Tuple[str, float]]],
    k_values: List[int] = [1, 3, 5, 10],
    output_dir: str = None,
) -> Tuple[Dict[str, float], NonLLMRetrievalEvaluator]:
    """
    Convenience function to run evaluation in one step.

    Args:
        benchmark_data: List of benchmark data samples
        retrieval_results: Dictionary mapping question IDs to lists of (passage_id, score) tuples
        k_values: List of k values to evaluate

    Returns:
        Dictionary of metrics
    """
    # Format results
    formatted_results = format_retrieval_results(retrieval_results)

    # Initialize evaluator
    evaluator = NonLLMRetrievalEvaluator(benchmark_data)

    # Run evaluation
    metrics = evaluator.evaluate_retrieval(formatted_results, k_values)

    # Save metrics
    if output_dir:
        # save metrics json
        metrics_file_name = "non_llm_based_metrics.json"
        save_json_file(
            file_path=os.path.join(output_dir, metrics_file_name), data=metrics
        )

        # save summary report csv
        report_file_name = "non_llm_based_metrics_summary_report.csv"
        output_path = os.path.join(output_dir, report_file_name)
        evaluator.generate_summary_report(output_path=output_path)

        # save plots
        plot_file_name = "non_llm_based_plot.png"
        output_path = os.path.join(output_dir, plot_file_name)
        evaluator.plot_metrics(output_path=output_path)

    return metrics, evaluator


# Example usage
if __name__ == "__main__":
    # Example benchmark data
    benchmark_data = [
        {
            "id": 0,
            "question": "What is GraphRAG?",
            "answer": "GraphRAG is a technique that combines graph databases with retrieval augmented generation.",
            "relevant_passage_ids": ["p1", "p2", "p3"],
        },
        {
            "id": 1,
            "question": "How does GraphRAG improve over traditional RAG?",
            "answer": "It leverages graph structure for better context retrieval.",
            "relevant_passage_ids": ["p4", "p5"],
        },
    ]

    # Example retrieval results (would come from your GraphRAG system)
    retrieval_results = {
        0: [
            {"pmid": "p1", "score": 0.95},
            {"pmid": "p3", "score": 0.85},
            {"pmid": "p2", "score": 0.75},
            {"pmid": "p6", "score": 0.65},
            {"pmid": "p7", "score": 0.55},
        ],
        1: [
            {"pmid": "p6", "score": 0.90},
            {"pmid": "p4", "score": 0.85},
            {"pmid": "p8", "score": 0.70},
            {"pmid": "p5", "score": 0.65},
            {"pmid": "p9", "score": 0.60},
        ],
    }

    # Run evaluation
    metrics, evaluator = run_evaluation_on_retrieved_chunks(
        benchmark_data, retrieval_results
    )

    # Generate summary report
    report = evaluator.generate_summary_report()
    print(report)

    # Plot metrics
    evaluator.plot_metrics()

    # Compare different models (example)
    model1_results = metrics
    # Assume model2_results would be from another run
    model2_results = {k: v * 0.9 for k, v in metrics.items()}  # Simulated second model

    evaluator.compare_models(
        {"GraphRAG Model 1": model1_results, "GraphRAG Model 2": model2_results}
    )

    # Calculate confidence intervals
    confidence_intervals = evaluator.calculate_confidence_intervals(
        format_retrieval_results(retrieval_results)
    )
    print("\nConfidence Intervals:")
    for metric, values in confidence_intervals.items():
        print(
            f"{metric}: {values['mean']:.3f} [{values['lower_bound']:.3f}, {values['upper_bound']:.3f}]"
        )
