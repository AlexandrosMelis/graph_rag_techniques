import os
from datetime import datetime

import torch

from configs.config import ConfigEnv, ConfigPath
from graph_embeddings.projection_model_v2 import train_projection


def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ConfigPath.MODELS_DIR, f"proj_model_v2_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 16
    epochs = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = train_projection(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
        model_dir=run_dir,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        val_ratio=0.2,
    )

    print(f"Training complete. Best model and plot saved in '{run_dir}'")


if __name__ == "__main__":
    main()
