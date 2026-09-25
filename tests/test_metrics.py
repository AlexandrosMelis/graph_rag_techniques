import math

import pytest

from graph_rag.evaluation.metrics import (
    coverage_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    success_at_k,
)

TRUE = [["a", "b"], ["c"]]
PRED = [["x", "a", "b"], ["y", "z", "w"]]


def test_precision_divides_by_k():
    assert precision_at_k(TRUE, PRED, 3) == pytest.approx((2 / 3 + 0) / 2)


def test_recall():
    assert recall_at_k(TRUE, PRED, 2) == pytest.approx((1 / 2 + 0) / 2)
    assert recall_at_k(TRUE, PRED, 3) == pytest.approx((1 + 0) / 2)


def test_mrr_uses_first_hit():
    assert mean_reciprocal_rank(TRUE, PRED, 3) == pytest.approx((1 / 2 + 0) / 2)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([["a", "b"]], [["a", "b", "x"]], 3) == pytest.approx(1.0)


def test_ndcg_single_query():
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k([["a", "b"]], [["x", "a", "b"]], 3) == pytest.approx(dcg / idcg)


def test_success_and_coverage():
    assert success_at_k(TRUE, PRED, 3) == pytest.approx(0.5)
    assert coverage_at_k(TRUE, PRED, 3) == pytest.approx(2 / 3)
