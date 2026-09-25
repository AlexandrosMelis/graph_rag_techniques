import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GraphEncoder(nn.Module):
    """
    Residual GNN whose output lives in the input embedding space (z = x + f(x, A)),
    so node embeddings stay comparable with the original chunk embeddings.
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.2,
        conv: str = "sage",
    ):
        super().__init__()
        self.config = dict(dim=dim, hidden=hidden, layers=layers, dropout=dropout, conv=conv)
        conv_cls = {"sage": SAGEConv, "gcn": GCNConv}[conv]
        dims = [dim] + [hidden] * (layers - 1) + [dim]
        self.convs = nn.ModuleList(conv_cls(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return x + h


def link_logits(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
    """Symmetric inner-product decoder."""
    return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
