# graph_rag/query_proj.py

from typing import List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from neo4j import GraphDatabase
from torch.utils.data import DataLoader, Dataset


class Neo4jLoader:
    """Load QA_PAIR → CONTEXT embedding pairs from Neo4j."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def fetch_pairs(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          - qid: question node internal id
          - q_emb: list[float] (BERT)
          - c_emb: list[float] (graph)
        """
        query = """
        // adjust property names if yours differ
        MATCH (q:QA_PAIR)-[:HAS_CONTEXT]->(c:CONTEXT)
        RETURN id(q) AS qid, q.bert_embedding AS q_emb, c.graph_embeddings AS c_emb
        """
        with self.driver.session(database=self.database) as sess:
            result = sess.run(query)
            rows = []
            for rec in result:
                rows.append(
                    {
                        "qid": rec["qid"],
                        "q_emb": rec["q_emb"],
                        "c_emb": rec["c_emb"],
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
    """Maps BERT-space → graph-embedding-space with a single linear layer."""

    def __init__(self, dim_sem: int, dim_graph: int):
        super().__init__()
        self.linear = nn.Linear(dim_sem, dim_graph)

    def forward(self, x):
        return self.linear(x)


def train_query_proj(
    dataset: QGDataset,
    dim_sem: int,
    dim_graph: int,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 50,
    device: str = "cpu",
) -> Tuple[QueryProj, List[float]]:
    """
    Trains QueryProj to minimize MSE(project(q), avg_context_graph).
    Returns the trained model and list of training losses.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = QueryProj(dim_sem, dim_graph).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for q_sem, c_graph in loader:
            q_sem = q_sem.to(device)
            c_graph = c_graph.to(device)

            pred = model(q_sem)
            loss = criterion(pred, c_graph)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * q_sem.size(0)

        epoch_loss = running / len(dataset)
        losses.append(epoch_loss)
        print(f"[Epoch {epoch:02d}] Loss: {epoch_loss:.4f}")
    return model, losses


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


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--uri", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--database", default="neo4j")
    p.add_argument("--dim-sem", type=int, required=True)
    p.add_argument("--dim-graph", type=int, required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # 1) load tabular data
    loader = Neo4jLoader(args.uri, args.user, args.password, args.database)
    df = loader.fetch_pairs()

    # 2) build dataset
    ds = QGDataset(df)

    # 3) train mapping
    model, losses = train_query_proj(
        ds,
        dim_sem=args.dim_sem,
        dim_graph=args.dim_graph,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
    )

    # 4) save weights
    torch.save(model.state_dict(), "query_proj.pt")
    print("Trained QueryProj saved to query_proj.pt")
