import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Identity, Linear, ModuleList, MultiheadAttention
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GAE, GCNConv, SAGEConv, TransformerConv, GATv2Conv

# ----------------------------------
# Model Definition
# ----------------------------------


class AttentiveFeatureFusion(torch.nn.Module):
    """Attention mechanism to fuse original BERT features with graph-learned features."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = Dropout(0.1)
        
    def forward(self, original_features, graph_features):
        # original_features: [N, D] - BERT embeddings
        # graph_features: [N, D] - Graph-learned embeddings
        
        # Add sequence dimension for attention
        orig = original_features.unsqueeze(1)  # [N, 1, D]
        graph = graph_features.unsqueeze(1)    # [N, 1, D]
        
        # Use graph features as query, original as key/value
        fused, _ = self.attention(graph, orig, orig)  # [N, 1, D]
        fused = fused.squeeze(1)  # [N, D]
        
        # Residual connection and layer norm
        fused = self.layer_norm(fused + graph_features)
        return self.dropout(fused)


class GraphEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        p_dropout: float = 0.2,
        use_attention: bool = True,
        heads: int = 4,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.res_projs = ModuleList()

        # Compute dimensions per layer
        dims_in = [in_channels] + [hidden_channels] * (num_layers - 2)
        dims_out = [hidden_channels] * (num_layers - 1)
        dims_in += [hidden_channels]
        dims_out += [out_channels]

        # Build layers with mixed architectures for better expressiveness
        for i, (in_dim, out_dim) in enumerate(zip(dims_in, dims_out)):
            # Alternate between different layer types
            if i == 0:
                # First layer: SAGEConv for neighborhood aggregation
                self.convs.append(SAGEConv(in_dim, out_dim))
            elif i < len(dims_in) - 1:
                # Middle layers: GATv2Conv for attention-based aggregation
                self.convs.append(GATv2Conv(in_dim, out_dim // heads, heads=heads, dropout=p_dropout))
            else:
                # Final layer: TransformerConv for global attention
                self.convs.append(TransformerConv(in_dim, out_dim, heads=1, dropout=p_dropout))
            
            self.bns.append(BatchNorm1d(out_dim))
            
            # Projection for residual connections
            if in_dim != out_dim:
                self.res_projs.append(Linear(in_dim, out_dim))
            else:
                self.res_projs.append(Identity())

        self.dropout = Dropout(p_dropout)
        
        # Feature fusion mechanism
        if use_attention and out_channels == in_channels:
            self.feature_fusion = AttentiveFeatureFusion(out_channels)
        else:
            self.feature_fusion = None

    def forward(self, x, edge_index):
        original_x = x  # Store original BERT features
        h = x
        
        # Apply all but final layer with activation + residual
        for i in range(len(self.convs) - 1):
            h0 = h
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = self.dropout(h)
            
            # Residual connection
            res = self.res_projs[i](h0)
            h = h + res

        # Final layer: no activation for embedding space
        z = self.convs[-1](h, edge_index)
        z = self.bns[-1](z)
        
        # Apply feature fusion if available
        if self.feature_fusion is not None:
            z = self.feature_fusion(original_x, z)
        
        return z


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim // 2
            
        self.mlp = torch.nn.Sequential(
            Linear(2 * in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            Linear(hidden_dim // 2, 1)
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
    num_layers: int = 4,
    use_attention: bool = True,
):
    """Instantiate encoder+decoder, optimizer, scheduler."""
    encoder = GraphEncoder(
        in_channels, 
        hidden_channels, 
        out_channels, 
        num_layers=num_layers,
        use_attention=use_attention
    ).to(device)
    
    predictor = LinkPredictor(out_channels).to(device)
    
    optimizer = AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.7, patience=8, verbose=True, min_lr=1e-6
    )
    
    print(
        f"Enhanced Model built: in={in_channels}, hidden={hidden_channels}, out={out_channels}, "
        f"layers={num_layers}, attention={use_attention}, lr={lr}, wd={weight_decay}"
    )
    return encoder, predictor, optimizer, scheduler
