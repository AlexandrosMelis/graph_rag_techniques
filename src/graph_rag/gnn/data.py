from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.index.graph import CorpusGraph


def build_node_graph(
    index: CorpusIndex, graph: CorpusGraph, edge_types: Sequence[str] = ("entity",)
) -> Data:
    """PyG graph over chunks: node features are chunk embeddings, edges the chosen types."""
    adjacency = graph.chunk_adjacency(edge_types).tocoo()
    return Data(
        x=torch.from_numpy(index.embeddings),
        edge_index=torch.from_numpy(np.vstack([adjacency.row, adjacency.col]).astype(np.int64)),
        edge_weight=torch.from_numpy(adjacency.data.astype(np.float32)),
    )


def split_edges(data: Data, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """Edge-level split with 1:1 negatives; validation/test edges are hidden from message passing."""
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    return transform(Data(x=data.x, edge_index=data.edge_index, num_nodes=data.num_nodes))


def feature_cosine_auc(split: Data) -> float:
    """
    AUC of plain cosine(x_i, x_j) on a split's positive/negative edges. It is the
    zero-parameter baseline a GNN has to beat; ~1.0 means the edges are a function of
    the features (e.g. embedding kNN edges) and link prediction teaches nothing.
    """
    x = torch.nn.functional.normalize(split.x, dim=-1)
    pos, neg = split.pos_edge_label_index, split.neg_edge_label_index
    scores = torch.cat([(x[pos[0]] * x[pos[1]]).sum(-1), (x[neg[0]] * x[neg[1]]).sum(-1)])
    labels = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
    return float(roc_auc_score(labels.numpy(), scores.numpy()))
