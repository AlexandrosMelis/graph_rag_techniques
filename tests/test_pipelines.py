import pytest

from graph_rag.config import settings
from graph_rag.data.splits import Splits, save_splits
from graph_rag.pipelines import evaluation


@pytest.fixture
def toy_artifacts(index, graph, questions, encoder, monkeypatch):
    """The toy index, graph and splits written where the pipelines look for them."""
    index.save(settings.index_dir)
    graph.save(settings.index_dir / "graph")
    save_splits(Splits(train=questions[:2], dev=questions[2:3], test=questions[3:]))
    monkeypatch.setattr(evaluation, "encoder_from_index", lambda index: encoder)


def test_evaluate_retrieval_end_to_end(toy_artifacts, tmp_path):
    result = evaluation.evaluate_retrieval(
        retrievers=["bm25", "hybrid", "ppr", "expand", "entity"],
        split="dev",
        baseline="dense",
        k_values=[1, 3],
        tune_first=True,
        output_dir=tmp_path,
        log=lambda _: None,
    )
    assert set(result["metrics"]) == {"dense", "bm25", "hybrid", "ppr", "expand", "entity"}
    assert (tmp_path / "summary.md").read_text().startswith("| retriever |")
    assert (tmp_path / "ppr_summary.json").exists()


def test_unknown_retriever_is_rejected(toy_artifacts):
    with pytest.raises(ValueError, match="Unknown retrievers"):
        evaluation.evaluate_retrieval(retrievers=["nope"], log=lambda _: None)
