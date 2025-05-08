# graph_rag/query_proj.py

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class DataProcessor:
    """Load QA_PAIR → CONTEXT embedding pairs from Neo4j."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def embed_questions_and_store(self, embedding_model) -> None:
        query = """MATCH (qa:QA_PAIR) WHERE qa.embedding IS NULL RETURN qa.id as qa_id, qa.question as question"""
        with self.driver.session() as session:
            result = session.run(query)
            questions_df = pd.DataFrame([dict(record) for record in result])
        if not questions_df.empty:
            question_embeddings = embedding_model.embed_documents(
                questions_df["question"].tolist()
            )
            questions_df["embeddings"] = question_embeddings
            with self.driver.session() as session:
                for index, row in tqdm(questions_df.iterrows()):
                    query = """
                    MATCH (qa:QA_PAIR {id: $qa_id})
                    CALL db.create.setNodeVectorProperty(qa, 'embedding', $embedding)
                    """
                    session.run(query, qa_id=row["qa_id"], embedding=row["embeddings"])
            print("Question embeddings stored in the graph database!")
        else:
            print("No questions found for embedding.")

    def fetch_pairs(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          - qid: question node internal id
          - q_emb: list[float] (BERT)
          - c_emb: list[float] (graph)
        """
        query = """
        MATCH (qa:QA_PAIR)-[:HAS_CONTEXT]->(context:CONTEXT)
        RETURN qa.id AS qa_id, qa.embedding AS question_embedding, context.graph_embedding AS context_graph_embedding
        """
        with self.driver.session(database=self.database) as sess:
            result = sess.run(query)
            rows = []
            for rec in result:
                rows.append(
                    {
                        "qid": rec["qa_id"],
                        "q_emb": rec["question_embedding"],
                        "c_emb": rec["context_graph_embedding"],
                    }
                )
        return pd.DataFrame(rows)


class QGDataset(Dataset):
    """
    Dataset of (q_sem, avg_c_graph) pairs for regression.
    Aggregates multiple contexts per question by mean.
    """

    def __init__(self, df: pd.DataFrame):
        # group contexts per question
        agg = (
            df.groupby("qid")
            .agg(
                {
                    "q_emb": lambda L: L.iloc[0],  # same per group
                    "c_emb": lambda L: list(L),  # list of lists
                }
            )
            .reset_index()
        )
        self.q_sem = torch.tensor(agg["q_emb"].tolist(), dtype=torch.float)
        # average graph embeddings per question
        avg_c = [
            torch.tensor(ces, dtype=torch.float).mean(dim=0).tolist()
            for ces in agg["c_emb"]
        ]
        self.c_graph = torch.tensor(avg_c, dtype=torch.float)
        assert self.q_sem.shape[0] == self.c_graph.shape[0]

    def __len__(self):
        return self.q_sem.size(0)

    def __getitem__(self, idx):
        return self.q_sem[idx], self.c_graph[idx]


class QueryProj(nn.Module):
    """
    A small MLP to map BERT embeddings → graph embeddings.
    - hidden layer with residual & LayerNorm
    - dropout for regularization
    """

    def __init__(
        self,
        dim_sem: int,
        dim_graph: int,
        hidden_dim: int = 256,
        p_dropout: float = 0.2,
    ):
        super().__init__()
        self.lin1 = nn.Linear(dim_sem, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(p_dropout)
        self.lin2 = nn.Linear(hidden_dim, dim_graph)

        # residual if dims match
        if dim_sem == hidden_dim:
            self.res1 = nn.Identity()
        else:
            self.res1 = nn.Linear(dim_sem, hidden_dim)

    def forward(self, x):
        # first block
        h0 = x
        h = self.lin1(x)
        h = self.norm1(h)
        h = F.relu(h + self.res1(h0))
        h = self.drop(h)
        # output
        return self.lin2(h)


def train_query_proj(
    dataset: QGDataset,
    dim_sem: int,
    dim_graph: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 100,
    val_ratio: float = 0.1,
    patience: int = 10,
    device: str = "cuda",
):
    # 1) split train/val
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2) model, optimizer, scheduler, loss
    model = QueryProj(dim_sem, dim_graph).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    no_improve = 0
    history = {
        "epoch": [],
        "train_mse": [],
        "val_mse": [],
        "train_cos": [],
        "val_cos": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses, train_cos = [], []
        for q_sem, c_graph in train_loader:
            q_sem = q_sem.to(device)
            c_graph = c_graph.to(device)

            pred = model(q_sem)
            loss = criterion(pred, c_graph)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            # cosine similarity per batch
            cos = (
                cosine_similarity(pred.detach().cpu(), c_graph.cpu()).diagonal().mean()
            )
            train_cos.append(cos)

        # eval
        model.eval()
        with torch.no_grad():
            val_losses, val_cos = [], []
            for q_sem, c_graph in val_loader:
                q_sem = q_sem.to(device)
                c_graph = c_graph.to(device)
                pred = model(q_sem)
                val_losses.append(criterion(pred, c_graph).item())
                cos = cosine_similarity(pred.cpu(), c_graph.cpu()).diagonal().mean()
                val_cos.append(cos)

        # aggregate
        train_mse = np.mean(train_losses)
        val_mse = np.mean(val_losses)
        train_c = np.mean(train_cos)
        val_c = np.mean(val_cos)

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["train_cos"].append(train_c)
        history["val_cos"].append(val_c)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} || "
            f"Train Cos: {train_c:.4f} | Val Cos: {val_c:.4f}"
        )

        # lr scheduling & early stopping
        scheduler.step(val_mse)
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            no_improve = 0
            torch.save(model.state_dict(), "best_query_proj.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs → stopping early.")
                break

    # load best
    model.load_state_dict(torch.load("best_query_proj.pt"))
    return model, history


def project_query(
    q_emb: List[float],
    model: QueryProj,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Given a new question's BERT embedding, returns its projected graph-space vector.
    """
    model.eval()
    with torch.no_grad():
        q = (
            torch.tensor(q_emb, dtype=torch.float).to(device).unsqueeze(0)
        )  # [1, dim_sem]
        z = model(q)  # [1, dim_graph]
    return z.squeeze(0).cpu()
