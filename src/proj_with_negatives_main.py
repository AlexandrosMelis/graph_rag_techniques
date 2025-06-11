import os
from datetime import datetime

import torch

from configs.config import ConfigEnv, ConfigPath
from projection_models.proj_model_with_triplets_ import train_projection

"""
Main file for training the query projection model with triplets.

Model details:
- input: triplet (q, c_pos, c_neg), where:
    - Query: Question's BERT embedding (q_emb)
    - Positive: Graph embedding of a context connected to this question (c_pos)
    - Negative: Graph embedding of a context NOT connected to this question (c_neg)
- hard negative sampling: Sample random contexts NOT connected to it (negatives)
- output: projection of q, c_pos, c_neg into a joint embedding space
- loss: triplet margin loss
- architecture: MLP with 3 hidden layers [1024, 512, 256]
"""

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
