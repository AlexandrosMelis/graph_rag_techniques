import os
from datetime import datetime

import torch

from configs.config import ConfigEnv, ConfigPath
from graph_embeddings.proj_model_with_attentive_pooling import train

"""
Main file for training the query projection model with triplets and attentive positives.

Model details:
- input: triplet (q, c_pos, c_neg), where:
    - Query: Question's BERT embedding (q_emb)
    - Positive: Graph embedding of a context connected to this question (c_pos)
    - Negative: Graph embedding of a context NOT connected to this question (c_neg)
- hard negative sampling: Sample random contexts NOT connected to it (negatives)
- attentive positives: Use attention to select the most relevant context from the question's context list
- loss: triplet margin loss
- architecture: MLP with 3 hidden layers [1024, 512, 256]
"""


def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ConfigPath.MODELS_DIR, f"proj_model_v3_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    batch_size = 64
    epochs = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
        model_dir=run_dir,
        batch_size=batch_size,
        epochs=epochs,
    )

    print(f"Training complete. Best model and plot saved in '{run_dir}'")


if __name__ == "__main__":
    main()
