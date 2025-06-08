import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from neo4j import GraphDatabase
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


# -----------------------------------------------------------------------------
# 1) DATA LOADING & HARD-NEGATIVE SAMPLING
# -----------------------------------------------------------------------------
class DataProcessor:
    """
    Loads positive (question, context) pairs and all contexts from Neo4j.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password), database=database
        )

    def fetch_positive_pairs(self) -> pd.DataFrame:
        """
        Returns:
          qa_id: int
          q_emb: List[float]   (BERT embedding)
          c_emb: List[float]   (graph embedding of connected context)
        """
        query = """
        MATCH (qa:QA_PAIR)-[:HAS_CONTEXT]->(ctx:CONTEXT)
        RETURN qa.id AS qa_id, qa.embedding AS q_emb, ctx.graph_embedding AS c_emb
        """.strip()
        rows = []
        with self.driver.session() as sess:
            for rec in sess.run(query):
                rows.append(
                    {
                        "qa_id": rec["qa_id"],
                        "q_emb": rec["q_emb"],
                        "c_emb": rec["c_emb"],
                    }
                )
        return pd.DataFrame(rows)

    def fetch_all_contexts(self) -> pd.DataFrame:
        """
        Returns:
          ctx_id: int (internal ID)
          c_emb: List[float]
        """
        query = """
        MATCH (ctx:CONTEXT)
        RETURN id(ctx) AS ctx_id, ctx.graph_embedding AS c_emb
        """
        rows = []
        with self.driver.session() as sess:
            for rec in sess.run(query):
                rows.append(
                    {
                        "ctx_id": rec["ctx_id"],
                        "c_emb": rec["c_emb"],
                    }
                )
        return pd.DataFrame(rows)


def build_triplet_dataframe(
    pos_df: pd.DataFrame, all_ctx_df: pd.DataFrame, neg_per_pos: int = 1, seed: int = 42
) -> pd.DataFrame:
    """
    Builds a DataFrame of triples (q_emb, c_pos, c_neg).
    For each positive pair, samples `neg_per_pos` contexts
    not connected to that question.
    Columns: q_emb, c_pos, c_neg
    """
    random.seed(seed)

    # map each question to its positive context embeddings
    qa2pos = {}
    for _, row in pos_df.iterrows():
        qa2pos.setdefault(row["qa_id"], []).append(row["c_emb"])

    all_embeddings = all_ctx_df["c_emb"].tolist()
    triplets = []

    from tqdm import tqdm

    # sample negatives for each positive
    for _, row in tqdm(
        pos_df.iterrows(), total=pos_df.shape[0], desc="Building triplets"
    ):
        q_emb = row["q_emb"]
        pos_emb = row["c_emb"]
        forbidden = set(tuple(x) for x in qa2pos[row["qa_id"]])
        candidates = [emb for emb in all_embeddings if tuple(emb) not in forbidden]
        sampled_negs = random.sample(candidates, k=min(neg_per_pos, len(candidates)))
        for neg_emb in sampled_negs:
            triplets.append({"q_emb": q_emb, "c_pos": pos_emb, "c_neg": neg_emb})

    return pd.DataFrame(triplets)


class TripletDataset(Dataset):
    """
    Dataset yielding (q_emb, pos_emb, neg_emb) for triplet loss.
    """

    def __init__(self, df: pd.DataFrame):
        self.q = torch.tensor(df["q_emb"].tolist(), dtype=torch.float32)
        self.pos = torch.tensor(df["c_pos"].tolist(), dtype=torch.float32)
        self.neg = torch.tensor(df["c_neg"].tolist(), dtype=torch.float32)

    def __len__(self):
        return self.q.size(0)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q[idx], self.pos[idx], self.neg[idx]


# -----------------------------------------------------------------------------
# 2) PROJECTION MODEL (DEEPER MLP)
# -----------------------------------------------------------------------------
class QueryGraphProjectionModel(nn.Module):
    """
    Deep MLP mapping BERT embeddings to graph embedding space.
    hidden_dims: list of hidden layer sizes.
    """

    def __init__(
        self,
        dim_sem: int,
        dim_graph: int,
        hidden_dims: List[int] = [1024, 512, 256],
        p_dropout: float = 0.2,
    ):
        super().__init__()
        # build full layer list: input -> *hidden_dims -> output
        dims = [dim_sem] + hidden_dims + [dim_graph]
        layers = []
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            # add nonlinearity and dropout only on hidden layers
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(out_d))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p_dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# 3) TRAINING WITH TRIPLET LOSS + HISTORY PLOTTING
# -----------------------------------------------------------------------------
def plot_training_history(
    train_losses: List[float], val_losses: List[float], save_path: str
) -> None:
    """
    Plots and saves training & validation loss curves.
    """
    plt.figure()
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_projection(
    uri: str,
    user: str,
    password: str,
    database: str,
    model_dir: str,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    val_ratio: float = 0.1,
    neg_per_pos: int = 1,
    margin: float = 1.0,
    patience: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    hidden_dims: List[int] = [1024, 512, 256],
) -> QueryGraphProjectionModel:
    # 1) Load data & build triplets
    dp = DataProcessor(uri, user, password, database=database)
    pos_df = dp.fetch_positive_pairs()
    all_ctx_df = dp.fetch_all_contexts()
    triplet_df = build_triplet_dataframe(pos_df, all_ctx_df, neg_per_pos)
    print(f"Triplet dataset size: {len(triplet_df)}")
    print(f"Positive pairs: {len(pos_df)}")
    print(f"All contexts: {len(all_ctx_df)}")
    print(f"Negative pairs: {len(triplet_df) - len(pos_df)}")

    dataset = TripletDataset(triplet_df)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2) Model, optimizer, loss
    dim_sem = len(pos_df["q_emb"].iloc[0])
    dim_graph = len(all_ctx_df["c_emb"].iloc[0])
    model = QueryGraphProjectionModel(dim_sem, dim_graph, hidden_dims=hidden_dims).to(
        device
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    best_val = float("inf")
    no_improve = 0

    # history
    train_hist, val_hist = [], []

    print(f"Training on {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for q, p, n in tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            q, p, n = q.to(device), p.to(device), n.to(device)
            q_proj = model(q)
            loss = criterion(q_proj, p, n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_train = float(np.mean(batch_losses))
        train_hist.append(avg_train)

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for q, p, n in val_loader:
                q, p, n = q.to(device), p.to(device), n.to(device)
                q_proj = model(q)
                val_losses.append(criterion(q_proj, p, n).item())
        avg_val = float(np.mean(val_losses))
        val_hist.append(avg_val)

        print(f"[Epoch {epoch:02d}] Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # checkpoint
        if avg_val < best_val:
            best_val = avg_val
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best_proj.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs. Early stopping.")
                break

    # plot and save curves
    plot_path = os.path.join(model_dir, "training_curve.png")
    plot_training_history(train_hist, val_hist, plot_path)

    # load best weights
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "best_proj.pt"), map_location=device)
    )
    return model


def project_query(
    q_emb: List[float], model: QueryGraphProjectionModel, device: str = "cpu"
) -> torch.Tensor:
    """
    Projects a single BERT embedding into the graph embedding space.
    """
    model.eval()
    x = torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        proj = model(x)
    return proj.squeeze(0).cpu()
