import os
import json
from typing import Dict, List, Tuple, Optional

import torch
from graphdatascience import GraphDataScience

from configs.config import ConfigEnv, ConfigPath

# your existing imports
from graph_embeddings.gnn_data_extraction import (
    connect_to_neo4j,
    create_gds_graph,
    fetch_node_features,
    fetch_topology,
)
from graph_embeddings.graph_encoder_model import GraphEncoder


def load_model_with_config(
    model_path: str,
    device: torch.device,
    in_channels: int,
    config: Optional[Dict] = None,
) -> GraphEncoder:
    """
    Load a trained GraphEncoder model with configuration support.
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on
        in_channels: Input feature dimensions
        config: Optional model configuration dict. If None, will try to load from model directory
    """
    # Try to load configuration from model directory
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "training_metrics.json")
    
    if config is None and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                training_data = json.load(f)
                config = training_data.get('config', {}).get('model', {})
                print(f"Loaded model configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            config = {}
    
    if config is None:
        config = {}
    
    # Set default parameters for enhanced model
    model_params = {
        'in_channels': in_channels,
        'hidden_channels': config.get('hid_dim', 512),  # Enhanced default
        'out_channels': config.get('out_dim', 768),
        'num_layers': config.get('num_layers', 4),
        'p_dropout': 0.2,  # Default from GraphEncoder
        'use_attention': config.get('use_attention', True),
        'heads': 4,  # Default from GraphEncoder
    }
    
    print(f"Creating GraphEncoder with parameters: {model_params}")
    
    # Create encoder with enhanced parameters
    encoder = GraphEncoder(**model_params).to(device)
    
    # Load checkpoint
    try:
        ckpt = torch.load(model_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        raise
    
    return encoder


def load_model(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    model_path: str,
    device: torch.device,
) -> GraphEncoder:
    """
    Legacy function for backward compatibility.
    Instantiate the encoder and load saved weights with basic parameters.
    """
    config = {
        'hid_dim': hidden_channels,
        'out_dim': out_channels,
        'num_layers': 3,  # Legacy default
        'use_attention': False,  # Legacy default
    }
    return load_model_with_config(model_path, device, in_channels, config)


def compute_embeddings(
    encoder: GraphEncoder,
    edge_index: torch.LongTensor,
    x: torch.FloatTensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run a forward pass to get z ∈ ℝ^(N×F).
    """
    with torch.no_grad():
        x = x.to(device)
        edge_index = edge_index.to(device)
        z = encoder(x, edge_index)  # [N, out_channels]
    return z.cpu()


def write_back_embeddings(
    gds: GraphDataScience,
    node_ids: List[int],
    embeddings: torch.Tensor,
    batch_size: int = 200,
) -> None:
    """
    UNWIND a list of {nodeId, embedding} maps to set each node.graph_embeddings.
    Batches it in case of large graphs.
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (n:CONTEXT)
      WHERE id(n) = row.nodeId
    CALL db.create.setNodeVectorProperty(n, 'graph_embedding', row.embedding)
    """

    # convert torch.Tensor to python lists
    emb_list = embeddings.tolist()
    rows = [{"nodeId": nid, "embedding": emb} for nid, emb in zip(node_ids, emb_list)]

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        gds.run_cypher(cypher, params={"batch": batch})
        print(f"Wrote batch {i} – {i+len(batch)} of {len(rows)}")


def find_latest_model(models_dir: str, model_prefix: str = "gnn_enhanced") -> Optional[str]:
    """
    Find the latest model directory based on timestamp.
    
    Args:
        models_dir: Directory containing model subdirectories
        model_prefix: Prefix to filter model directories
    
    Returns:
        Path to the latest model file or None if not found
    """
    try:
        model_dirs = [d for d in os.listdir(models_dir) 
                     if d.startswith(model_prefix) and os.path.isdir(os.path.join(models_dir, d))]
        
        if not model_dirs:
            return None
        
        # Sort by timestamp (assuming format: gnn_enhanced_YYYYMMDD_HHMMSS)
        model_dirs.sort(reverse=True)
        latest_dir = model_dirs[0]
        
        model_path = os.path.join(models_dir, latest_dir, "graphsage_encoder_pred.pt")
        
        if os.path.exists(model_path):
            return model_path
        else:
            return None
            
    except Exception as e:
        print(f"Error finding latest model: {e}")
        return None


def write_graph_embeddings_to_neo4j(
    model_path: Optional[str] = None,
    graph_name: str = "contexts",
    use_auto_device: bool = True,
    batch_size: int = 200,
) -> None:
    """
    Enhanced function to write GNN embeddings back to Neo4j.
    
    Args:
        model_path: Path to model checkpoint. If None, will find latest enhanced model.
        graph_name: Name of the GDS graph projection
        use_auto_device: Whether to automatically detect best device (GPU/CPU)
        batch_size: Batch size for writing embeddings to Neo4j
    """
    print("=== Starting GNN Embedding Computation ===")
    
    # 1) Device setup
    if use_auto_device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Auto-detected device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    # 2) Model path resolution
    if model_path is None:
        model_path = find_latest_model(ConfigPath.MODELS_DIR)
        if model_path is None:
            raise FileNotFoundError(
                f"No enhanced model found in {ConfigPath.MODELS_DIR}. "
                "Please train a model first or specify a model_path."
            )
        print(f"Using latest model: {model_path}")
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        print(f"Using specified model: {model_path}")

    # 3) Connect to Neo4j and create GDS graph projection
    print("Connecting to Neo4j...")
    gds = connect_to_neo4j(
        ConfigEnv.NEO4J_URI,
        ConfigEnv.NEO4J_USER,
        ConfigEnv.NEO4J_PASSWORD,
        ConfigEnv.NEO4J_DB,
    )

    print(f"Creating GDS graph projection: {graph_name}")
    gds_graph = create_gds_graph(gds, graph_name)

    # 4) Fetch topology & raw node embeddings
    print("Fetching graph topology and node features...")
    edge_index, node_df = fetch_topology(gds, gds_graph)
    x = fetch_node_features(node_df)
    
    print(f"Graph stats: {len(node_df)} nodes, {edge_index.size(1)} edges")
    print(f"Feature dimensions: {x.shape}")

    # 5) Load trained encoder with enhanced configuration
    print("Loading trained GNN model...")
    in_channels = x.size(1)  # e.g. 768 (BERT dimensions)
    
    encoder = load_model_with_config(
        model_path=model_path,
        device=device,
        in_channels=in_channels,
    )

    # 6) Compute graph embeddings
    print("Computing graph embeddings...")
    z = compute_embeddings(encoder, edge_index, x, device)
    print(f"Generated embeddings shape: {z.shape}")

    # 7) Write embeddings back to Neo4j
    print(f"Writing embeddings to Neo4j (batch_size={batch_size})...")
    write_back_embeddings(gds, node_df["nodeId"].tolist(), z, batch_size=batch_size)

    print("=== GNN Embedding Computation Complete ===")
    print(f"Successfully wrote {z.shape[0]} node embeddings to Neo4j")
    print(f"Each embedding has {z.shape[1]} dimensions")
