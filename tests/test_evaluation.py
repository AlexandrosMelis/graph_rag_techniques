import numpy as np
import pytest

from graph_rag.evaluation.executor import compare_runs, run_retrieval, summarize_run
from graph_rag.evaluation.metrics import average_precision_at_k, map_at_k, paired_bootstrap_test
from graph_rag.retrieval.base import BaseRetriever
from graph_rag.retrieval.dense import DenseRetriever


def test_average_precision_counts_missed_relevant_documents():
    # One hit at rank 1 out of 4 relevant documents: AP@10 = 1/4, not 1.
    assert average_precision_at_k(["a", "b", "c", "d"], ["a", "x"], 10) == pytest.approx(0.25)
    assert average_precision_at_k(["a", "b"], ["a", "b"], 10) == pytest.approx(1.0)
    assert map_at_k([["a"], []], [["a"], ["x"]], 5) == pytest.approx(0.5)


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 0.5, 200)
    result = paired_bootstrap_test(a, a + 0.2)
    assert result["mean_diff"] == pytest.approx(0.2)
    assert result["p_value"] < 0.01
    assert paired_bootstrap_test(a, a)["p_value"] == pytest.approx(1.0)


def test_retriever_errors_propagate(questions):
    class Broken(BaseRetriever):
        name = "broken"

        def retrieve(self, query, top_k=10):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_retrieval(questions, Broken(), show_progress=False)


def test_run_and_summarize(index, encoder, questions, tmp_path):
    results = run_retrieval(
        questions, DenseRetriever(index, encoder), top_k=3, output_dir=tmp_path, show_progress=False
    )
    assert len(results) == len(questions)
    assert all(len(r["retrieved_pmids"]) == len(set(r["retrieved_pmids"])) <= 3 for r in results)
    summary = summarize_run(results, k_values=(1, 3), index=index, ci_k=3)
    assert summary["recall_ceiling"] == pytest.approx(1.0)
    assert set(summary["metrics"]) == {"1", "3"}
    assert summary["latency_ms"]["p95"] >= summary["latency_ms"]["p50"]
    comparison = compare_runs(results, results, k=3)
    assert comparison["recall@3"]["mean_diff"] == 0
