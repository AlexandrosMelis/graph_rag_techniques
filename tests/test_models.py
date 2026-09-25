import numpy as np
import pytest
import torch

from graph_rag.index.linking import EntityLinker
from graph_rag.models.graph_reranker import (
    SCALAR_FEATURES,
    CandidateGraphBuilder,
    GraphReranker,
    GraphRerankerScorer,
    RerankerTrainingConfig,
    grouped_info_nce,
    train_graph_reranker,
)
from graph_rag.models.losses import multi_positive_info_nce
from graph_rag.models.query_adapter import (
    AdapterTrainingConfig,
    QueryAdapter,
    mine_hard_negatives,
    positive_chunks,
    train_query_adapter,
)
from graph_rag.retrieval.dense import DenseRetriever


def test_multi_positive_info_nce():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [False, False, False]])
    expected = -np.log((np.e**2 + np.e) / (np.e**2 + np.e + 1))
    assert multi_positive_info_nce(logits, mask).item() == pytest.approx(expected, rel=1e-5)


def test_adapter_is_identity_at_init():
    adapter = QueryAdapter(dim=16, rank=4)
    vectors = np.random.default_rng(0).normal(size=(3, 16)).astype(np.float32)
    expected = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    np.testing.assert_allclose(adapter.transform(vectors), expected, rtol=1e-5, atol=1e-6)
    assert sum(p.numel() for p in adapter.parameters()) == 2 * 16 * 4


def test_hard_negatives_exclude_every_positive(index, questions, encoder):
    positives = positive_chunks(questions, index)
    vectors = np.stack([encoder.encode_query(q.question) for q in questions])
    pools = mine_hard_negatives(vectors, index.embeddings, positives, pool_size=5)
    for pool, pos in zip(pools, positives):
        assert not set(pool.tolist()) & set(pos.tolist())


def test_adapter_training_keeps_the_best_dev_checkpoint(index, questions, encoder, tmp_path):
    config = AdapterTrainingConfig(
        rank=4, epochs=3, batch_size=2, hard_negatives=2, hard_negative_pool=3, eval_k=2
    )
    adapter, history = train_query_adapter(
        questions, questions, index, encoder, config, log=lambda _: None
    )
    assert history["best_dev_recall"] >= history["dev_recall"][0]
    adapter.save(tmp_path)
    np.testing.assert_allclose(
        QueryAdapter.load(tmp_path).transform(index.embeddings[:2]),
        adapter.transform(index.embeddings[:2]),
    )


def test_candidate_graph_features_never_see_labels(index, graph, encoder):
    builder = CandidateGraphBuilder(index, graph, encoder, EntityLinker(graph, encoder))
    hits = DenseRetriever(index, encoder).retrieve("brca1 tamoxifen", top_k=5)
    unlabeled = builder.build("brca1 tamoxifen", hits)
    labeled = builder.build("brca1 tamoxifen", hits, gold_pmids={"6"})
    assert torch.equal(unlabeled.x, labeled.x)
    assert not hasattr(unlabeled, "y") or unlabeled.y is None
    assert labeled.y.sum() == 1
    assert unlabeled.x.shape == (5, index.dimension + len(SCALAR_FEATURES))
    assert int(unlabeled.edge_index.max()) < 5


def test_graph_reranker_trains_and_scores(index, graph, encoder, questions, tmp_path):
    builder = CandidateGraphBuilder(index, graph, encoder, EntityLinker(graph, encoder))
    first_stage = DenseRetriever(index, encoder)
    config = RerankerTrainingConfig(
        candidate_k=5, hidden=16, heads=2, epochs=2, batch_size=2, eval_k=2
    )
    model, history = train_graph_reranker(
        questions, questions, first_stage, builder, config, log=lambda _: None
    )
    assert len(history["dev_recall"]) >= 1
    hits = first_stage.retrieve("glucose", top_k=5)
    scores = GraphRerankerScorer(model, builder).score_hits("glucose", hits)
    assert scores.shape == (5,) and np.isfinite(scores).all()
    model.save(tmp_path, extra={"candidate_k": 5})
    loaded, config_dict = GraphReranker.load(tmp_path)
    assert config_dict["candidate_k"] == 5


def test_grouped_info_nce_handles_ragged_batches():
    scores = torch.tensor([1.0, 0.0, 0.5, 2.0, 0.1])
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
    batch = torch.tensor([0, 0, 1, 1, 1])
    loss = grouped_info_nce(scores, labels, batch)
    assert torch.isfinite(loss) and loss > 0
