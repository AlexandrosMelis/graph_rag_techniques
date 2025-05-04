import os
from datetime import datetime

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from configs.config import ConfigEnv, ConfigPath
from graph_embeddings.data_extraction import (
    connect_to_neo4j,
    create_gds_graph,
    fetch_node_features,
    fetch_topology,
    sample_graph,
)
from graph_embeddings.data_preparation import build_pyg_data, split_data
from graph_embeddings.model import build_model
from graph_embeddings.train import evaluate, save_metrics, save_model, train_model
from graph_embeddings.utils import set_seed


def run_gnn_training(apply_sampling: bool = False) -> None:
    seed = 42
    set_seed(seed)

    # Neo4j connection
    gds = connect_to_neo4j(
        ConfigEnv.NEO4J_URI,
        ConfigEnv.NEO4J_USER,
        ConfigEnv.NEO4J_PASSWORD,
        ConfigEnv.NEO4J_DB,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. (Optional) Graph sampling
    graph_name = "contexts"
    G = create_gds_graph(gds=gds, graph_name=graph_name)
    if apply_sampling:
        G = sample_graph(gds, graph_name, f"{graph_name}_sample", seed=seed)

    # 2. Fetch topology & features
    edge_index, node_df = fetch_topology(gds, G)
    x = fetch_node_features(node_df)

    # 3. Build PyG Data & Split onto device
    data = build_pyg_data(x, edge_index)
    train_data, val_data, test_data = split_data(data, device=device)

    # 4. Build model, optimizer & scheduler
    in_dim = x.size(1)  # e.g. 768
    hid_dim = 256
    out_dim = 768
    lr = 1e-3
    weight_decay = 1e-4

    encoder, predictor, optimizer, scheduler = build_model(
        in_channels=in_dim,
        hidden_channels=hid_dim,
        out_channels=out_dim,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    # 5. Train (multi‐task + early stopping + LR scheduling)
    history = train_model(
        encoder=encoder,
        predictor=predictor,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data=train_data,
        val_data=val_data,
        epochs=300,
        λ_feat=0.5,
        patience=10,
    )

    # 6. Final evaluation on test split
    test_auc, test_ap = evaluate(encoder, predictor, test_data)
    print(f"Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}")

    # 7. Save model & metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ConfigPath.MODELS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "graphsage_encoder_pred.pt")
    metrics_path = os.path.join(run_dir, "training_metrics.json")

    save_model(encoder, predictor, model_path)

    history["test_auc"] = test_auc
    history["test_ap"] = test_ap
    save_metrics(history, metrics_path)


if __name__ == "__main__":
    run_gnn_training(apply_sampling=False)
