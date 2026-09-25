import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graph_rag.gnn.data import build_node_graph, feature_cosine_auc, split_edges
from graph_rag.gnn.inference import compute_node_embeddings
from graph_rag.gnn.training import GNNTrainingConfig, train_link_prediction


def threshold_graph(n: int = 300, dim: int = 16, threshold: float = 0.6, seed: int = 0) -> Data:
    """Edges wherever cosine(x_i, x_j) >= threshold: the old IS_SIMILAR_TO construction."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(10, dim))
    x = centers[rng.integers(0, 10, n)] + 0.6 * rng.normal(size=(n, dim))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    sim = x @ x.T
    np.fill_diagonal(sim, -1)
    src, dst = np.nonzero(sim >= threshold)
    return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(np.vstack([src, dst])))


def test_similarity_threshold_edges_are_solved_by_cosine():
    torch.manual_seed(0)
    _, val, test = split_edges(threshold_graph())
    assert feature_cosine_auc(val) == pytest.approx(1.0)
    assert feature_cosine_auc(test) == pytest.approx(1.0)


def test_link_prediction_training_and_embeddings():
    torch.manual_seed(0)
    data = threshold_graph()
    train, val, test = split_edges(data)
    config = GNNTrainingConfig(hidden=16, epochs=10, eval_every=2, patience=3)
    encoder, history = train_link_prediction(train, val, test, config, log=lambda _: None)
    assert {"test_auc", "feature_cosine_test_auc", "best_val_auc"} <= set(history)
    z = compute_node_embeddings(encoder, data)
    assert z.shape == tuple(data.x.shape)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, rtol=1e-5)


def test_node_graph_uses_index_embeddings(index, graph):
    data = build_node_graph(index, graph, ("entity", "next"))
    assert data.num_nodes == index.n
    assert torch.equal(data.x, torch.from_numpy(index.embeddings))
