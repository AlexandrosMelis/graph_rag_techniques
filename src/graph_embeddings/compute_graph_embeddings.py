import argparse
import os
from typing import Dict, List, Tuple

import torch
from graphdatascience import GraphDataScience

from configs.config import ConfigEnv, ConfigPath

# your existing imports
from graph_embeddings.data_extraction import (
    connect_to_neo4j,
    create_gds_graph,
    fetch_node_features,
    fetch_topology,
)
from graph_embeddings.graph_encoder_model import GraphEncoder


def load_model(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    model_path: str,
    device: torch.device,
) -> GraphEncoder:
    """
    Instantiate the encoder and load saved weights.
    """
    encoder = GraphEncoder(in_channels, hidden_channels, out_channels).to(device)
    ckpt = torch.load(model_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    return encoder


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


def write_graph_embeddings_to_neo4j():

    # 1) connect & project (or fetch) GDS graph
    gds = connect_to_neo4j(
        ConfigEnv.NEO4J_URI,
        ConfigEnv.NEO4J_USER,
        ConfigEnv.NEO4J_PASSWORD,
        ConfigEnv.NEO4J_DB,
    )

    graph_name = "contexts"
    gds_graph = create_gds_graph(gds, graph_name)

    # 2) fetch topology & raw node embeddings
    edge_index, node_df = fetch_topology(gds, gds_graph)
    x = fetch_node_features(node_df)

    # 3) load your trained encoder
    device = "cuda"
    device = torch.device(device)

    in_channels = x.size(1)  # e.g. 768
    hidden_channels = 256
    out_channels = 768
    model_path = os.path.join(
        ConfigPath.MODELS_DIR, "gnn_20250516_223235", "graphsage_encoder_pred.pt"
    )
    encoder = load_model(
        in_channels,
        hidden_channels,
        out_channels,
        model_path,
        device,
    )

    # 4) compute new graph embeddings
    z = compute_embeddings(encoder, edge_index, x, device)

    # 5) write back to Neo4j
    write_back_embeddings(gds, node_df["nodeId"].tolist(), z)

    print("Done writing graph_embeddings to Neo4j.")
