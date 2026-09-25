import numpy as np
import torch
from torch_geometric.data import Data

from graph_rag.gnn.encoder import GraphEncoder


@torch.no_grad()
def compute_node_embeddings(encoder: GraphEncoder, data: Data) -> np.ndarray:
    """Encode every chunk with the full graph (all edges) and L2-normalize."""
    encoder.eval()
    z = encoder(data.x, data.edge_index)
    return torch.nn.functional.normalize(z, dim=-1).numpy().astype(np.float32)
