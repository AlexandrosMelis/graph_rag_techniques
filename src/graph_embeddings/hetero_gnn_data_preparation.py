from typing import Optional, Tuple, Dict, List
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.utils import to_undirected
import numpy as np

# ----------------------------------
# Heterogeneous PyG Data Preparation
# ----------------------------------


def build_hetero_data(
    node_features: Dict[str, torch.FloatTensor],
    edge_indices: Dict[str, torch.LongTensor],
    edge_attrs: Dict[str, torch.FloatTensor],
    edge_type_mappings: Dict[str, Tuple[str, str]]
) -> HeteroData:
    """Create a HeteroData object from heterogeneous graph components."""
    
    print("Building HeteroData object...")
    
    # Create HeteroData object
    data = HeteroData()
    
    # Add node features
    for node_type, features in node_features.items():
        data[node_type].x = features
        print(f"Added {node_type} nodes: {features.shape}")
    
    # Add edges
    for edge_type, edge_index in edge_indices.items():
        source_type, target_type = edge_type_mappings[edge_type]
        
        # Create edge type tuple for PyG
        if source_type == target_type:
            # Homogeneous edge (e.g., context-context)
            edge_key = (source_type, edge_type, target_type)
        else:
            # Heterogeneous edge
            edge_key = (source_type, edge_type, target_type)
        
        data[edge_key].edge_index = edge_index
        
        # Add edge attributes if available
        if edge_type in edge_attrs:
            data[edge_key].edge_attr = edge_attrs[edge_type].unsqueeze(1)
            print(f"Added {edge_key} edges: {edge_index.shape} with attributes")
        else:
            print(f"Added {edge_key} edges: {edge_index.shape}")
    
    # Add reverse edges for undirected relationships
    add_reverse_edges(data, edge_type_mappings)
    
    return data


def add_reverse_edges(data: HeteroData, edge_type_mappings: Dict[str, Tuple[str, str]]):
    """Add reverse edges for undirected relationships."""
    
    reverse_mappings = {
        'has_context': 'rev_has_context',
        'has_mesh_term': 'rev_has_mesh_term'
    }
    
    for edge_type, reverse_type in reverse_mappings.items():
        if edge_type in edge_type_mappings:
            source_type, target_type = edge_type_mappings[edge_type]
            
            # Create forward edge key
            forward_key = (source_type, edge_type, target_type)
            
            if forward_key in data.edge_types:
                # Create reverse edge key
                reverse_key = (target_type, reverse_type, source_type)
                
                # Add reverse edges
                forward_edge_index = data[forward_key].edge_index
                reverse_edge_index = torch.stack([forward_edge_index[1], forward_edge_index[0]])
                data[reverse_key].edge_index = reverse_edge_index
                
                # Copy edge attributes if they exist
                if hasattr(data[forward_key], 'edge_attr'):
                    data[reverse_key].edge_attr = data[forward_key].edge_attr.clone()
                
                print(f"Added reverse edges: {reverse_key}")


def split_hetero_data(
    data: HeteroData,
    target_edge_type: Tuple[str, str, str] = ('context', 'is_similar_to', 'context'),
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    add_negative_train_samples: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[HeteroData, HeteroData, HeteroData]:
    """Split heterogeneous data for link prediction on a specific edge type."""
    
    print(f"Splitting heterogeneous data on edge type: {target_edge_type}")
    print(f"Original {target_edge_type} edges: {data[target_edge_type].edge_index.shape[1]}")
    
    # Create a transform for the specific edge type
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,  # Assume context similarity is undirected
        split_labels=True,
        add_negative_train_samples=add_negative_train_samples,
        neg_sampling_ratio=1.0,
        edge_types=[target_edge_type],
        rev_edge_types=None  # We'll handle reverse edges separately
    )
    
    # Apply the transform
    train_data, val_data, test_data = transform(data)
    
    # Validate splits
    print("Validating heterogeneous data splits...")
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        if hasattr(split_data[target_edge_type], 'pos_edge_label_index'):
            pos_edges = split_data[target_edge_type].pos_edge_label_index.shape[1]
            neg_edges = split_data[target_edge_type].neg_edge_label_index.shape[1]
            print(f"  {split_name}: pos_edges={pos_edges}, neg_edges={neg_edges}")
        else:
            print(f"  {split_name}: no edge labels found")
    
    if device is not None:
        print(f"Moving heterogeneous data to device: {device}")
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)
    
    # Enhanced logging
    print("\n=== HETEROGENEOUS DATA SPLIT SUMMARY ===")
    for node_type in data.node_types:
        num_nodes = data[node_type].x.shape[0]
        num_features = data[node_type].x.shape[1]
        print(f"{node_type} nodes: {num_nodes} (features: {num_features})")
    
    for edge_type in data.edge_types:
        num_edges = data[edge_type].edge_index.shape[1]
        print(f"{edge_type} edges: {num_edges}")
    
    print("=" * 40)
    
    return train_data, val_data, test_data


def validate_hetero_data(data: HeteroData) -> bool:
    """Validate heterogeneous graph data quality."""
    
    print("=== HETEROGENEOUS DATA VALIDATION ===")
    
    valid = True
    
    # Check node types
    print("Node types:")
    for node_type in data.node_types:
        num_nodes = data[node_type].x.shape[0]
        num_features = data[node_type].x.shape[1]
        print(f"  {node_type}: {num_nodes} nodes, {num_features} features")
        
        if num_nodes == 0 or num_features == 0:
            print(f"  ERROR: {node_type} has invalid dimensions")
            valid = False
    
    # Check edge types
    print("\nEdge types:")
    for edge_type in data.edge_types:
        num_edges = data[edge_type].edge_index.shape[1]
        print(f"  {edge_type}: {num_edges} edges")
        
        if num_edges == 0:
            print(f"  WARNING: {edge_type} has no edges")
        
        # Check edge index validity
        source_type, relation, target_type = edge_type
        
        if source_type in data.node_types and target_type in data.node_types:
            max_source = data[edge_type].edge_index[0].max().item() if num_edges > 0 else -1
            max_target = data[edge_type].edge_index[1].max().item() if num_edges > 0 else -1
            
            source_nodes = data[source_type].x.shape[0]
            target_nodes = data[target_type].x.shape[0]
            
            if max_source >= source_nodes or max_target >= target_nodes:
                print(f"  ERROR: {edge_type} has invalid edge indices")
                print(f"    Max source: {max_source}, available: {source_nodes}")
                print(f"    Max target: {max_target}, available: {target_nodes}")
                valid = False
    
    # Check connectivity
    print("\nConnectivity analysis:")
    total_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    total_nodes = sum(data[nt].x.shape[0] for nt in data.node_types)
    
    print(f"Total nodes: {total_nodes}")
    print(f"Total edges: {total_edges}")
    
    if total_edges == 0:
        print("ERROR: No edges in the graph")
        valid = False
    
    print(f"Data validation: {'PASSED' if valid else 'FAILED'}")
    print("=" * 40)
    
    return valid


def create_heterogeneous_node_embeddings(
    data: HeteroData,
    embedding_dim: int = 128
) -> Dict[str, torch.Tensor]:
    """Create initial node embeddings for heterogeneous nodes."""
    
    node_embeddings = {}
    
    for node_type in data.node_types:
        num_nodes = data[node_type].x.shape[0]
        
        # Use existing features if available, otherwise create random embeddings
        if data[node_type].x.shape[1] >= embedding_dim:
            # Use first 'embedding_dim' features
            node_embeddings[node_type] = data[node_type].x[:, :embedding_dim]
        else:
            # Create random embeddings
            node_embeddings[node_type] = torch.randn(num_nodes, embedding_dim)
        
        print(f"Created embeddings for {node_type}: {node_embeddings[node_type].shape}")
    
    return node_embeddings


def create_metapath_edges(
    data: HeteroData,
    metapaths: List[List[Tuple[str, str, str]]]
) -> HeteroData:
    """Create metapath-based edges for enhanced heterogeneous learning."""
    
    print("Creating metapath edges...")
    
    # Example metapaths:
    # QA -> Context -> MESH -> Context -> QA
    # Context -> MESH -> Context (via shared MESH terms)
    
    for i, metapath in enumerate(metapaths):
        print(f"Processing metapath {i+1}: {' -> '.join([f'{s}-{r}-{t}' for s, r, t in metapath])}")
        
        # Implement metapath traversal logic here
        # This is a simplified version - in practice, you'd want more sophisticated metapath computation
        
    return data


def prepare_heterogeneous_training_data(
    raw_data: Dict,
    embedding_dim: int = 768,
    target_edge_type: str = 'is_similar_to',
    device: Optional[torch.device] = None
) -> Tuple[HeteroData, HeteroData, HeteroData]:
    """Complete pipeline for preparing heterogeneous training data."""
    
    print("=== HETEROGENEOUS DATA PREPARATION PIPELINE ===")
    
    # Build HeteroData object
    data = build_hetero_data(
        raw_data['node_features'],
        raw_data['edge_indices'],
        raw_data['edge_attrs'],
        raw_data['edge_type_mappings']
    )
    
    # Validate data
    if not validate_hetero_data(data):
        raise ValueError("Invalid heterogeneous data")
    
    # Split data
    target_edge_tuple = ('context', target_edge_type, 'context')
    train_data, val_data, test_data = split_hetero_data(
        data,
        target_edge_type=target_edge_tuple,
        device=device
    )
    
    print("Heterogeneous data preparation completed successfully!")
    print("=" * 50)
    
    return train_data, val_data, test_data 