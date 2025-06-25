import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Identity, Linear, ModuleList, ModuleDict
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATv2Conv, TransformerConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional

# ----------------------------------
# Heterogeneous Model Definition
# ----------------------------------


class HeteroGraphEncoder(torch.nn.Module):
    """
    Heterogeneous Graph Neural Network Encoder that handles multiple node types and edge types.
    Uses HeteroConv to apply different convolutions for different edge types.
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_attention: bool = True,
        heads: int = 4
    ):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # Create heterogeneous convolution layers
        self.convs = ModuleList()
        self.norms = ModuleDict()
        self.dropouts = ModuleDict()
        
        # Input projections to make all node types have the same hidden dimension
        self.input_projections = ModuleDict()
        for node_type in node_types:
            if in_channels_dict[node_type] != hidden_channels:
                self.input_projections[node_type] = Linear(in_channels_dict[node_type], hidden_channels)
            else:
                self.input_projections[node_type] = Identity()
        
        # Build heterogeneous convolution layers
        for i in range(num_layers):
            conv_dict = {}
            
            for edge_type in edge_types:
                source_type, relation, target_type = edge_type
                
                # Choose different convolution types for different layers
                if i == 0:
                    # First layer: SAGEConv for robust neighborhood aggregation
                    conv_dict[edge_type] = SAGEConv(
                        (hidden_channels, hidden_channels), 
                        hidden_channels
                    )
                elif i < num_layers - 1 and use_attention:
                    # Middle layers: GAT for attention-based aggregation
                    out_dim = hidden_channels // heads
                    conv_dict[edge_type] = GATv2Conv(
                        (hidden_channels, hidden_channels),
                        out_dim,
                        heads=heads,
                        dropout=dropout,
                        concat=True,
                        add_self_loops=False
                    )
                else:
                    # Final layer: SAGE for stable final representations (supports heterogeneous graphs)
                    conv_dict[edge_type] = SAGEConv(
                        (hidden_channels, hidden_channels),
                        out_channels
                    )
            
            # Create HeteroConv layer
            hetero_conv = HeteroConv(conv_dict, aggr='mean')
            
            # Ensure all conv modules have add_self_loops=False for heterogeneous graphs
            for edge_type, conv_module in hetero_conv.convs.items():
                if hasattr(conv_module, 'add_self_loops'):
                    conv_module.add_self_loops = False
            
            self.convs.append(hetero_conv)
            
            # Add normalization and dropout for each node type
            if i < num_layers - 1:  # No norm/dropout on final layer
                for node_type in node_types:
                    layer_key = f"{node_type}_{i}"
                    self.norms[layer_key] = BatchNorm1d(hidden_channels)
                    self.dropouts[layer_key] = Dropout(dropout)
        
        # Type-specific output projections to ensure all node types have correct output dimension
        self.output_projections = ModuleDict()
        self.use_output_projections = True
        
        for node_type in node_types:
            # For nodes that don't get updated by conv layers (like qa_pair), 
            # they need projection from hidden_channels to out_channels
            self.output_projections[node_type] = Linear(hidden_channels, out_channels)
        
        # Cross-type attention for feature fusion
        if use_attention:
            # Use out_channels since that's what the conv layers actually output
            self.cross_attention = HeteroCrossAttention(
                node_types, out_channels, heads=4
            )
        else:
            self.cross_attention = None
    
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the heterogeneous graph encoder.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            
        Returns:
            Dictionary of node embeddings for each node type
        """
        
        # Debug: Check input data
        for node_type, x in x_dict.items():
            if x is None or x.size(0) == 0:
                raise ValueError(f"Node type '{node_type}' has no data or is None")
        
        # Debug: Check edge data
        valid_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index is not None and edge_index.size(1) > 0:
                valid_edge_index_dict[edge_type] = edge_index
        
        # Apply input projections
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.input_projections[node_type](x)
        
        # Apply heterogeneous convolution layers
        for i, conv in enumerate(self.convs):
            # Store previous state for nodes not updated by this layer
            prev_h_dict = h_dict.copy()
            h_dict = conv(h_dict, valid_edge_index_dict)
            
            # Handle missing node types (nodes not involved in edges for this layer)
            for node_type in prev_h_dict:
                if node_type not in h_dict or h_dict[node_type] is None:
                    h_dict[node_type] = prev_h_dict[node_type]
            
            # Apply normalization and dropout (except for final layer)
            if i < self.num_layers - 1:
                new_h_dict = {}
                for node_type, h in h_dict.items():
                    if h is not None:  # Safety check
                        layer_key = f"{node_type}_{i}"
                        h = self.norms[layer_key](h)
                        h = F.relu(h)
                        h = self.dropouts[layer_key](h)
                        new_h_dict[node_type] = h
                    else:
                        # Use previous state if current is None
                        new_h_dict[node_type] = prev_h_dict[node_type]
                h_dict = new_h_dict
        
        # Apply output projections (selective based on whether nodes were updated by conv layers)
        if self.use_output_projections:
            out_dict = {}
            for node_type, h in h_dict.items():
                # Check if this node type appears as target in any edge type
                is_target_node = any(edge_type[2] == node_type for edge_type in self.edge_types)
                
                if is_target_node:
                    # Node was updated by conv layers, should already have correct dimension
                    out_dict[node_type] = h
                else:
                    # Node was not updated by conv layers, needs projection
                    out_dict[node_type] = self.output_projections[node_type](h)
                    
                
        else:
            out_dict = h_dict
        

        # Apply cross-type attention if enabled
        if self.cross_attention is not None:
            out_dict = self.cross_attention(out_dict)
        
        return out_dict


class HeteroCrossAttention(torch.nn.Module):
    """Cross-attention mechanism between different node types."""
    
    def __init__(self, node_types: List[str], embed_dim: int, heads: int = 4):
        super().__init__()
        self.node_types = node_types
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        
        # Linear projections for queries, keys, values
        self.query_projections = ModuleDict()
        self.key_projections = ModuleDict()
        self.value_projections = ModuleDict()
        self.output_projections = ModuleDict()
        
        for node_type in node_types:
            self.query_projections[node_type] = Linear(embed_dim, embed_dim)
            self.key_projections[node_type] = Linear(embed_dim, embed_dim)
            self.value_projections[node_type] = Linear(embed_dim, embed_dim)
            self.output_projections[node_type] = Linear(embed_dim, embed_dim)
        
        # Layer normalization
        self.layer_norms = ModuleDict()
        for node_type in node_types:
            self.layer_norms[node_type] = torch.nn.LayerNorm(embed_dim)
            
        self.dropout = torch.nn.Dropout(0.1)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply cross-attention between node types."""
        
        out_dict = {}
        
        for query_type in self.node_types:
            query_nodes = x_dict[query_type]  # [num_query_nodes, embed_dim]
            batch_size = query_nodes.size(0)
            
            # Project queries
            Q = self.query_projections[query_type](query_nodes)  # [num_query_nodes, embed_dim]
            Q = Q.view(batch_size, self.heads, self.head_dim)  # [num_query_nodes, heads, head_dim]
            
            # Gather keys and values from other node types
            all_keys = []
            all_values = []
            
            for key_type in self.node_types:
                if key_type != query_type and key_type in x_dict:
                    key_nodes = x_dict[key_type]
                    K = self.key_projections[key_type](key_nodes)
                    V = self.value_projections[key_type](key_nodes)
                    all_keys.append(K)
                    all_values.append(V)
            
            if all_keys:
                # Concatenate all keys and values
                K = torch.cat(all_keys, dim=0)  # [total_key_nodes, embed_dim]
                V = torch.cat(all_values, dim=0)  # [total_key_nodes, embed_dim]
                
                seq_len = K.size(0)
                K = K.view(seq_len, self.heads, self.head_dim)  # [total_key_nodes, heads, head_dim]
                V = V.view(seq_len, self.heads, self.head_dim)  # [total_key_nodes, heads, head_dim]
                
                # Compute attention scores
                scores = torch.einsum('qhd,khd->qhk', Q, K) * self.scale  # [num_query_nodes, heads, total_key_nodes]
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                # Apply attention to values
                attended = torch.einsum('qhk,khd->qhd', attn_weights, V)  # [num_query_nodes, heads, head_dim]
                attended = attended.contiguous().view(batch_size, self.embed_dim)  # [num_query_nodes, embed_dim]
                
                # Output projection
                attended = self.output_projections[query_type](attended)
                
                # Residual connection and layer norm
                out_dict[query_type] = self.layer_norms[query_type](
                    attended + query_nodes
                )
            else:
                # No other node types to attend to
                out_dict[query_type] = query_nodes
        
        return out_dict


class HeteroLinkPredictor(torch.nn.Module):
    """
    Heterogeneous link predictor that can predict links between different node types.
    """
    
    def __init__(
        self, 
        in_channels_dict: Dict[str, int],
        hidden_dim: int = 128,
        edge_types: List[Tuple[str, str, str]] = None
    ):
        super().__init__()
        
        self.edge_types = edge_types or []
        
        # Create MLP for each edge type
        self.predictors = ModuleDict()
        
        for edge_type in self.edge_types:
            source_type, relation, target_type = edge_type
            source_dim = in_channels_dict[source_type]
            target_dim = in_channels_dict[target_type]
            
            # MLP for this specific edge type
            self.predictors[f"{source_type}_{relation}_{target_type}"] = torch.nn.Sequential(
                Linear(source_dim + target_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                Linear(hidden_dim, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                Linear(hidden_dim // 2, 1)
            )
    
    def forward(
        self, 
        z_dict: Dict[str, torch.Tensor], 
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_type: Tuple[str, str, str]
    ) -> torch.Tensor:
        """
        Predict links for a specific edge type.
        
        Args:
            z_dict: Node embeddings for each node type
            edge_index_dict: Edge indices (used to get node pairs)
            edge_type: The specific edge type to predict
            
        Returns:
            Link prediction scores
        """
        source_type, relation, target_type = edge_type
        edge_index = edge_index_dict[edge_type]
        
        # Get embeddings for source and target nodes
        source_emb = z_dict[source_type][edge_index[0]]
        target_emb = z_dict[target_type][edge_index[1]]
        
        # Concatenate and predict
        edge_features = torch.cat([source_emb, target_emb], dim=-1)
        
        predictor_key = f"{source_type}_{relation}_{target_type}"
        logits = self.predictors[predictor_key](edge_features).squeeze(-1)
        
        return torch.sigmoid(logits)


def build_hetero_model(
    node_types: List[str],
    edge_types: List[Tuple[str, str, str]],
    in_channels_dict: Dict[str, int],
    hidden_channels: int,
    out_channels: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    num_layers: int = 3,
    use_attention: bool = True
) -> Tuple[HeteroGraphEncoder, HeteroLinkPredictor, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Build heterogeneous graph encoder with predictor, optimizer, and scheduler."""
    
    # Create encoder
    encoder = HeteroGraphEncoder(
        node_types=node_types,
        edge_types=edge_types,
        in_channels_dict=in_channels_dict,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        use_attention=use_attention
    ).to(device)
    
    # Create predictor with output dimensions
    out_channels_dict = {node_type: out_channels for node_type in node_types}
    predictor = HeteroLinkPredictor(
        in_channels_dict=out_channels_dict,
        edge_types=edge_types
    ).to(device)
    
    # Create optimizer
    optimizer = AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.7, patience=8, verbose=True, min_lr=1e-6
    )
    
    print(f"Built heterogeneous model:")
    print(f"  Node types: {node_types}")
    print(f"  Edge types: {len(edge_types)}")
    print(f"  Hidden dim: {hidden_channels}, Output dim: {out_channels}")
    print(f"  Layers: {num_layers}, Attention: {use_attention}")
    print(f"  Learning rate: {lr}, Weight decay: {weight_decay}")
    
    return encoder, predictor, optimizer, scheduler


class HeteroFeatureFusion(torch.nn.Module):
    """
    Feature fusion module that combines information from different node types
    to create enhanced representations.
    """
    
    def __init__(
        self, 
        node_types: List[str],
        embed_dim: int,
        fusion_method: str = "attention"  # "attention", "concat", "average"
    ):
        super().__init__()
        self.node_types = node_types
        self.embed_dim = embed_dim
        self.fusion_method = fusion_method
        
        if fusion_method == "attention":
            # Learnable attention weights for each node type
            self.attention_weights = torch.nn.Parameter(
                torch.randn(len(node_types), embed_dim)
            )
            self.attention_mlp = torch.nn.Sequential(
                Linear(embed_dim, embed_dim // 2),
                torch.nn.ReLU(),
                Linear(embed_dim // 2, 1)
            )
        elif fusion_method == "concat":
            # Projection after concatenation
            self.fusion_proj = Linear(len(node_types) * embed_dim, embed_dim)
    
    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from different node types.
        
        Args:
            x_dict: Dictionary of node embeddings for each type
            
        Returns:
            Fused embedding tensor
        """
        
        if self.fusion_method == "attention":
            # Attention-weighted fusion
            embeddings = []
            weights = []
            
            for i, node_type in enumerate(self.node_types):
                if node_type in x_dict:
                    emb = x_dict[node_type].mean(dim=0)  # Average pool nodes of this type
                    attention_score = self.attention_mlp(emb * self.attention_weights[i])
                    embeddings.append(emb)
                    weights.append(attention_score)
            
            if embeddings:
                embeddings = torch.stack(embeddings)
                weights = torch.softmax(torch.stack(weights), dim=0)
                fused = (embeddings * weights).sum(dim=0)
            else:
                fused = torch.zeros(self.embed_dim, device=next(self.parameters()).device)
                
        elif self.fusion_method == "concat":
            # Concatenation fusion
            embeddings = []
            for node_type in self.node_types:
                if node_type in x_dict:
                    emb = x_dict[node_type].mean(dim=0)
                    embeddings.append(emb)
                else:
                    embeddings.append(torch.zeros(self.embed_dim, device=next(self.parameters()).device))
            
            concatenated = torch.cat(embeddings, dim=0)
            fused = self.fusion_proj(concatenated)
            
        else:  # average
            # Simple averaging
            embeddings = []
            for node_type in self.node_types:
                if node_type in x_dict:
                    embeddings.append(x_dict[node_type].mean(dim=0))
            
            if embeddings:
                fused = torch.stack(embeddings).mean(dim=0)
            else:
                fused = torch.zeros(self.embed_dim, device=next(self.parameters()).device)
        
        return fused 