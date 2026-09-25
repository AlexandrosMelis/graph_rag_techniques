import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATQueryProjector(nn.Module):
    """
    GAT-based encoder that ingests a small subgraph
    (including the query as node 0) and outputs
    a single graph‐pooled embedding for the query node.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
        heads: int = 4,
        p_dropout: float = 0.2,
    ):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=p_dropout)
        # after heads concat: hidden_dim*heads
        self.gat2 = GATConv(
            hidden_dim * heads, out_dim, heads=1, concat=False, dropout=p_dropout
        )
        self.dropout = nn.Dropout(p_dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, batch=None):
        # 1st layer
        h = self.gat1(x, edge_index)
        h = self.act(h)
        h = self.dropout(h)
        # 2nd layer
        h = self.gat2(h, edge_index)
        # global mean pooling per connected component,
        # but here we want the query node’s embedding:
        #   batch can be all zeros => whole graph is one component
        return h[0]  # index 0 is the query node
