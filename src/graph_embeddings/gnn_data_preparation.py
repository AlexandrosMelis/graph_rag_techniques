from typing import Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.utils import to_undirected

# ----------------------------------
# PyG Data Preparation
# ----------------------------------


def build_pyg_data(x: torch.FloatTensor, edge_index: torch.LongTensor) -> Data:
    """Create a PyG Data object from features and edge index with validation."""
    # Validate input dimensions
    print(f"Building PyG data: nodes={x.size(0)}, features={x.size(1)}, edges={edge_index.size(1)}")
    
    # Ensure edge_index is undirected for better learning
    edge_index = to_undirected(edge_index)
    print(f"After making undirected: edges={edge_index.size(1)}")
    
    data = Data(x=x, edge_index=edge_index)
    
    # Validate the created data
    assert data.is_undirected(), "Graph should be undirected"
    # assert not data.has_isolated_nodes(), "Graph should not have isolated nodes"
    
    return data


def split_data(
    data: Data,
    val_ratio: float = 0.15,  # Increased validation set
    test_ratio: float = 0.15,  # Increased test set
    undirected: bool = True,
    device: Optional[torch.device] = None,
    add_negative_train_samples: bool = True,
) -> Tuple[Data, Data, Data]:
    """Enhanced data splitting with better negative sampling and validation."""
    
    print(f"Splitting data: val_ratio={val_ratio}, test_ratio={test_ratio}")
    print(f"Original edges: {data.edge_index.size(1)}")
    
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=undirected,
        split_labels=True,
        add_negative_train_samples=add_negative_train_samples,
        neg_sampling_ratio=1.0,  # 1:1 ratio of positive to negative edges
    )
    
    train, val, test = transform(data)

    # Validate splits
    print("Validating data splits...")
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        pos_edges = split_data.pos_edge_label_index.size(1)
        neg_edges = split_data.neg_edge_label_index.size(1)
        print(f"  {split_name}: pos_edges={pos_edges}, neg_edges={neg_edges}")
        
        # Ensure we have balanced positive/negative samples
        assert pos_edges > 0, f"{split_name} split has no positive edges"
        assert neg_edges > 0, f"{split_name} split has no negative edges"

    if device is not None:
        print(f"Moving data to device: {device}")
        train = train.to(device)
        val = val.to(device)
        test = test.to(device)

    # Enhanced logging
    print("\n=== DATA SPLIT SUMMARY ===")
    print(f"Training edges: {train.pos_edge_label_index.size(1)} pos + {train.neg_edge_label_index.size(1)} neg")
    print(f"Validation edges: {val.pos_edge_label_index.size(1)} pos + {val.neg_edge_label_index.size(1)} neg")
    print(f"Test edges: {test.pos_edge_label_index.size(1)} pos + {test.neg_edge_label_index.size(1)} neg")
    print(f"Total nodes: {train.x.size(0)}")
    print(f"Node features: {train.x.size(1)}")
    print("========================\n")
    
    return train, val, test


def validate_graph_data(data: Data) -> bool:
    """Validate graph data quality and provide diagnostics."""
    print("=== GRAPH DATA VALIDATION ===")
    
    # Basic checks
    num_nodes = data.x.size(0) if data.x is not None else 0
    num_edges = data.edge_index.size(1) if data.edge_index is not None else 0
    feature_dim = data.x.size(1) if data.x is not None else 0
    
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Features: {feature_dim}")
    
    # Connectivity checks
    if num_edges > 0:
        avg_degree = (2 * num_edges) / num_nodes  # For undirected graphs
        print(f"Average degree: {avg_degree:.2f}")
        
        # Check for isolated nodes
        unique_nodes = torch.unique(data.edge_index).size(0)
        isolated_nodes = num_nodes - unique_nodes
        print(f"Isolated nodes: {isolated_nodes} ({isolated_nodes/num_nodes*100:.1f}%)")
        
        # Check edge index validity
        max_node_id = torch.max(data.edge_index).item()
        print(f"Max node ID: {max_node_id} (should be < {num_nodes})")
        
        valid = (
            num_nodes > 0 and 
            num_edges > 0 and 
            feature_dim > 0 and
            max_node_id < num_nodes and
            isolated_nodes / num_nodes < 0.1  # Less than 10% isolated nodes
        )
    else:
        valid = False
    
    print(f"Data validation: {'PASSED' if valid else 'FAILED'}")
    print("============================\n")
    
    return valid
