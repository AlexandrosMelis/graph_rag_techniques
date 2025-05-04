import logging
from typing import List, Tuple

import numpy as np
import torch
from graphdatascience import GraphDataScience
from torch_geometric.data import Data
from torch_geometric.nn import GAE


def load_model(model_path: str, model: GAE, device: torch.device) -> GAE:
    """Load state dict into provided GAE instance."""
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def predict_edges(
    model: GAE, data: Data, edge_label_index: torch.Tensor, device: torch.device
) -> np.ndarray:
    """
    Given a set of edges (edge_label_index), predict link probabilities.
    Returns a NumPy array of shape [num_edges].
    """
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        # use inner-product decoder
        preds = model.decoder(z, edge_label_index).flatten()
        probs = preds.sigmoid().cpu().numpy()
    print(f"Predicted {probs.shape[0]} edges.")
    return probs


def compute_full_embeddings(model: GAE, data: Data, device: torch.device) -> np.ndarray:
    """
    Compute and return full-graph node embeddings.
    Returns an array of shape [num_nodes, embedding_dim].
    """
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    embeddings = z.cpu().numpy()
    print(f"Computed embeddings of shape {embeddings.shape}.")
    return embeddings


def write_embeddings(
    gds: GraphDataScience,
    graph_name: str,
    node_ids: List[int],
    embeddings: np.ndarray,
    property_key: str,
) -> None:
    """
    Write back embeddings as node properties using GDS writeNodeProperties.
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "nodeId": node_ids,
            property_key: embeddings.tolist(),
        }
    )
    gds.graph.writeNodeProperties(graph_name, df, ["nodeId"], [property_key])
    print(f"Wrote embeddings under '{property_key}' to graph '{graph_name}'.")
