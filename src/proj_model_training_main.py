import json
import os
from datetime import datetime

import torch

from configs.config import ConfigEnv, ConfigPath
from graph_embeddings.query_projection_model import (
    DataProcessor,
    QGDataset,
    train_query_proj,
)

if __name__ == "__main__":

    # 1) load tabular data
    loader = DataProcessor(
        ConfigEnv.NEO4J_URI,
        ConfigEnv.NEO4J_USER,
        ConfigEnv.NEO4J_PASSWORD,
        ConfigEnv.NEO4J_DB,
    )
    df = loader.fetch_pairs()

    # 2) build dataset
    ds = QGDataset(df)

    # 3) train mapping
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ConfigPath.MODELS_DIR, f"proj_model_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    bert_dim = 768
    graph_dim = 768
    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 16
    epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, history = train_query_proj(
        ds,
        model_path=run_dir,
        dim_sem=bert_dim,
        dim_graph=graph_dim,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )

    # 4) save weights
    model_path = os.path.join(run_dir, "projection_model.pt")

    torch.save(model.state_dict(), model_path)
    print("Trained QueryProj saved to", model_path)

    metrics_path = os.path.join(run_dir, "training_metrics.json")
    try:
        with open(metrics_path, "w") as f:
            json.dump(history, f)
        print("Training metrics saved to", metrics_path)
    except Exception as e:
        print(f"Error saving training metrics: {e}")

    print("Training losses:", history)
