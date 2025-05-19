import json
import os
from datetime import datetime

import torch

from configs.config import ConfigEnv, ConfigPath
from graph_embeddings.query_proj_model_with_domain_classifier import (
    DataProcessor,
    QGDataset,
    train_query_proj_with_domain_classifier,
)
from llms.embedding_model import EmbeddingModel


def create_and_store_question_embeddings():

    embedding_model = EmbeddingModel()
    data_processor = DataProcessor(
        ConfigEnv.NEO4J_URI,
        ConfigEnv.NEO4J_USER,
        ConfigEnv.NEO4J_PASSWORD,
        ConfigEnv.NEO4J_DB,
    )

    data_processor.embed_questions_and_store(embedding_model=embedding_model)


def run_query_projection_model_with_discriminator_training():
    # 1) load QA–context pairs from Neo4j
    loader = DataProcessor(
        ConfigEnv.NEO4J_URI,
        ConfigEnv.NEO4J_USER,
        ConfigEnv.NEO4J_PASSWORD,
        ConfigEnv.NEO4J_DB,
    )
    df = loader.fetch_pairs()

    # 2) build dataset
    ds = QGDataset(df)

    # 3) prepare run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ConfigPath.MODELS_DIR, f"proj_model_da_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 4) hyperparameters
    bert_dim = 768
    graph_dim = 768
    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 16
    epochs = 500
    val_ratio = 0.1
    patience = 10
    lambda_ctr = 1.0
    lambda_da = 0.5
    lambda_grl = 1.0
    temperature = 0.07
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 5) train with contrastive + domain-adversarial
    model = train_query_proj_with_domain_classifier(
        dataset=ds,
        dim_sem=bert_dim,
        dim_graph=graph_dim,
        model_path=run_dir,
        epochs=epochs,
    )

    # 6) save model weights
    model_path = os.path.join(run_dir, "projection_model.pt")
    torch.save(model.state_dict(), model_path)
    print("Trained QueryProj with discriminator saved to", model_path)


if __name__ == "__main__":
    # run_query_projection_model_with_discriminator_training()
    # create_and_store_question_embeddings()
    pass
