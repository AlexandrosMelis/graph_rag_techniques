"""
Graph autoencoder (GCN + GAE) trained by link prediction on a random-walk sample of
the CONTEXT similarity graph, plus a helper that runs a trained query projection
model through the graph-embedding retriever.

Usage:
    python -m graph_rag.experiments.gae_link_prediction train
    python -m graph_rag.experiments.gae_link_prediction project --model-path <path> --question "..."
"""

import argparse

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.transforms import RandomLinkSplit

from graph_rag.config import ConfigEnv
from graph_rag.gnn.extraction import (
    connect_to_neo4j,
    create_gds_graph,
    fetch_node_features,
    fetch_topology,
    sample_graph,
)
from graph_rag.utils import set_seed


class GCNEncoder(torch.nn.Module):
    """Two-layer GCN that compresses 768-d BERT features into a small structural latent."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def load_sampled_graph(graph_name: str = "contexts", seed: int = 42) -> tuple[Data, pd.DataFrame]:
    """Project the CONTEXT graph in GDS, sample it with random walks, and export it to PyG."""
    gds = connect_to_neo4j(
        ConfigEnv.NEO4J_URI, ConfigEnv.NEO4J_USER, ConfigEnv.NEO4J_PASSWORD, ConfigEnv.NEO4J_DB
    )
    create_gds_graph(gds=gds, graph_name=graph_name)
    sampled = sample_graph(gds, graph_name, f"{graph_name}_sample", seed=seed)
    edge_index, node_df = fetch_topology(gds, sampled)
    x = fetch_node_features(node_df)
    return Data(x=x, edge_index=edge_index), node_df


def train_gae(
    data: Data,
    hidden_dim: int = 128,
    out_dim: int = 64,
    lr: float = 0.01,
    weight_decay: float = 1e-5,
    epochs: int = 300,
    eval_every: int = 10,
) -> tuple[GAE, dict]:
    """Train a GAE on a 85/10/5 edge split and report validation/test AUC and AP."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = RandomLinkSplit(num_val=0.10, num_test=0.05, is_undirected=True, split_labels=True)
    train_data, val_data, test_data = (split.to(device) for split in transform(data))

    model = GAE(GCNEncoder(data.num_node_features, hidden_dim, out_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def evaluate(split: Data) -> tuple[float, float]:
        model.eval()
        z = model.encode(split.x, split.edge_index)
        return model.test(z, split.pos_edge_label_index, split.neg_edge_label_index)

    history = {"epoch": [], "loss": [], "val_auc": [], "val_ap": []}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(
            z,
            pos_edge_index=train_data.pos_edge_label_index,
            neg_edge_index=train_data.neg_edge_label_index,
        )
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0:
            val_auc, val_ap = evaluate(val_data)
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["val_auc"].append(val_auc)
            history["val_ap"].append(val_ap)
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, AP: {val_ap:.4f}")

    history["test_auc"], history["test_ap"] = evaluate(test_data)
    print(f"Test AUC: {history['test_auc']:.4f}, AP: {history['test_ap']:.4f}")
    return model, history


@torch.no_grad()
def encode_nodes(model: GAE, data: Data, node_df: pd.DataFrame) -> pd.DataFrame:
    """Encode the full sampled graph and pair each GDS nodeId with its graph embedding."""
    device = next(model.parameters()).device
    model.eval()
    z = model.encode(data.x.to(device), data.edge_index.to(device)).cpu()
    out = node_df[["nodeId"]].copy()
    out["graph_embedding"] = z.tolist()
    return out


def project_and_retrieve(model_path: str, question: str, top_k: int = 5) -> list[dict]:
    """Run a trained domain-adversarial projection model through the graph-embedding retriever."""
    from graph_rag.graph.connection import Neo4jConnection
    from graph_rag.llm.embeddings import EmbeddingModel
    from graph_rag.projection.domain_adversarial import QueryProjectionEncoderModel
    from graph_rag.retrieval.graph_embedding import GraphEmbeddingSimilarityRetriever

    projection = QueryProjectionEncoderModel(dim_sem=768, dim_graph=768)
    projection.load_state_dict(torch.load(model_path, map_location="cpu"))
    projection.eval()

    connection = Neo4jConnection(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
    )
    retriever = GraphEmbeddingSimilarityRetriever(
        embedding_model=EmbeddingModel(),
        neo4j_driver=connection.get_driver(),
        projection_model=projection,
        device="cpu",
    )
    return retriever.retrieve(question, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train the GAE on a sampled context graph")
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--lr", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=42)
    project = sub.add_parser("project", help="retrieve with a trained projection model")
    project.add_argument("--model-path", required=True)
    project.add_argument("--question", required=True)
    project.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "train":
        set_seed(args.seed)
        data, node_df = load_sampled_graph(seed=args.seed)
        print(data)
        model, _ = train_gae(data, lr=args.lr, epochs=args.epochs)
        print(encode_nodes(model, data, node_df).head())
    else:
        for hit in project_and_retrieve(args.model_path, args.question, args.top_k):
            print(hit)


if __name__ == "__main__":
    main()
