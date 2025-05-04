import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Identity, Linear, ModuleList
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GAE, GCNConv, SAGEConv

# ----------------------------------
# Model Definition
# ----------------------------------


class GraphEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        p_dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.res_projs = ModuleList()

        # compute dimensions per layer
        dims_in = [in_channels] + [hidden_channels] * (num_layers - 2)
        dims_out = [hidden_channels] * (num_layers - 1)
        dims_in += [hidden_channels]
        dims_out += [out_channels]

        # build layers
        for in_dim, out_dim in zip(dims_in, dims_out):
            self.convs.append(SAGEConv(in_dim, out_dim))
            self.bns.append(BatchNorm1d(out_dim))
            # projection for residual if dims differ
            if in_dim != out_dim:
                self.res_projs.append(Linear(in_dim, out_dim))
            else:
                self.res_projs.append(Identity())

        self.drop = Dropout(p_dropout)

    def forward(self, x, edge_index):
        h = x
        # apply all but final layer with activation + residual
        for i in range(len(self.convs) - 1):
            h0 = h
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = self.drop(h)
            # project residual if needed
            res = self.res_projs[i](h0)
            h = h + res

        # final layer: no activation or residual
        z = self.convs[-1](h, edge_index)
        z = self.bns[-1](z)
        return z


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            Linear(2 * in_dim, in_dim), torch.nn.ReLU(), Linear(in_dim, 1)
        )

    def forward(self, z, edge_index):
        # z: [N, F], edge_index: [2, E]
        zi = z[edge_index[0]]
        zj = z[edge_index[1]]
        logits = self.mlp(torch.cat([zi, zj], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)


def build_model(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
):
    """Instantiate encoder+decoder, optimizer, scheduler."""
    encoder = GraphEncoder(in_channels, hidden_channels, out_channels).to(device)
    predictor = LinkPredictor(out_channels).to(device)
    optimizer = AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )
    print(
        f"Model built: in={in_channels}, hidden={hidden_channels}, out={out_channels}, lr={lr}, wd={weight_decay}"
    )
    return encoder, predictor, optimizer, scheduler
