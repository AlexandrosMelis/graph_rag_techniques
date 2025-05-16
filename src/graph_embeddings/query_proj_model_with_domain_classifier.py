# graph_rag/query_proj_regularized.py

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neo4j import GraphDatabase
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


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, λ):
        ctx.λ = λ
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.λ, None


class QueryProjectionEncoderModel(nn.Module):
    """
    Maps BERT embeddings → graph embeddings, while exposing a hidden feature
    for the domain classifier.
    """

    def __init__(self, dim_sem, dim_graph, hidden_dim=512, p_dropout=0.2):
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Hidden feature before final projection."""
        h0 = x
        h = self.lin1(x)
        h = self.norm1(h)
        h = F.relu(h + self.res1(h0))
        h = self.drop(h)
        return h

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        h = self.encode(x)
        out = self.lin2(h)
        if return_hidden:
            return out, h
        return out


class DomainClassifier(nn.Module):
    """
    A lightweight head that sees both hidden query features (reversed) and
    a linear projection of the true graph embeddings, to force them to share
    the same hidden distribution.
    """

    def __init__(self, dim_graph: int, hidden_dim: int = 256):
        super().__init__()
        self.graph_proj = nn.Linear(dim_graph, hidden_dim)
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, h_query: torch.Tensor, c_graph: torch.Tensor, λ_grl: float):
        # h_query: [B, H], c_graph: [B, G]
        # 1) Reverse-gradient on query features
        h_rev = GradientReversal.apply(h_query, λ_grl)  # [B, H]
        # 2) Project graph embeddings into same H-dim space
        g_feat = self.graph_proj(c_graph)  # [B, H]
        # 3) Stack domain examples
        combined = torch.cat([h_rev, g_feat], dim=0)  # [2B, H]
        logits = self.clf(combined)  # [2B, 2]
        return logits


def nt_xent_loss(
    z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """
    Compute NT-Xent (InfoNCE) loss between batches of projections z_i and positives z_j.
    """
    B = z_i.size(0)
    # Normalize
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    # Cosine similarity matrix [B, B]
    sim = torch.mm(z_i, z_j.t()) / temperature
    labels = torch.arange(B, device=z_i.device)
    return F.cross_entropy(sim, labels)


def train_query_proj_with_domain_classifier(
    dataset: QGDataset,
    dim_sem: int,
    dim_graph: int,
    model_path: str,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 100,
    val_ratio: float = 0.1,
    patience: int = 10,
    device: str = "cuda",
    λ_ctr: float = 1.0,
    λ_da: float = 0.5,
    λ_grl: float = 1.0,
    temperature: float = 0.07,
):
    # 1) split train/val
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2) model + domain classifier + optimizer + losses + scheduler
    model = QueryProjectionEncoderModel(dim_sem, dim_graph).to(device)
    domain_clf = DomainClassifier(dim_graph, hidden_dim=model.lin1.out_features).to(
        device
    )
    optimizer = optim.AdamW(
        list(model.parameters()) + list(domain_clf.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    mse_criterion = nn.MSELoss()

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # —— TRAINING ——
        model.train()
        domain_clf.train()
        train_losses = []

        for q_sem, c_graph in train_loader:
            q_sem = q_sem.to(device)
            c_graph = c_graph.to(device)

            # forward pass
            proj, hidden = model(q_sem, return_hidden=True)
            # MSE
            loss_mse = mse_criterion(proj, c_graph)
            # contrastive
            loss_ctr = nt_xent_loss(proj, c_graph, temperature)
            # domain-adversarial
            domain_logits = domain_clf(hidden, c_graph, λ_grl)
            # labels: first B = queries → 0, next B = real graph → 1
            B = q_sem.size(0)
            domain_labels = torch.cat(
                [torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.long)],
                dim=0,
            ).to(device)
            loss_da = F.cross_entropy(domain_logits, domain_labels)

            loss = loss_mse + λ_ctr * loss_ctr + λ_da * loss_da
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # —— VALIDATION ——
        model.eval()
        val_losses = []
        with torch.no_grad():
            for q_sem, c_graph in val_loader:
                q_sem = q_sem.to(device)
                c_graph = c_graph.to(device)
                proj = model(q_sem)
                val_losses.append(mse_criterion(proj, c_graph).item())
        val_mse = float(np.mean(val_losses))

        print(
            f"[Epoch {epoch:03d}] Train Loss: {np.mean(train_losses):.4f} | Val MSE: {val_mse:.4f}"
        )

        scheduler.step(val_mse)
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            no_improve = 0
            torch.save(
                model.state_dict(), os.path.join(model_path, "best_query_proj.pt")
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs → stopping early.")
                break

    # load best and return
    model.load_state_dict(torch.load(os.path.join(model_path, "best_query_proj.pt")))
    return model


def project_query(
    q_emb: List[float],
    model: QueryProjectionEncoderModel,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Given a new question's BERT embedding, returns its projected graph-space vector.
    """
    model.eval()
    with torch.no_grad():
        q = torch.tensor(q_emb, dtype=torch.float).to(device).unsqueeze(0)
        z = model(q)
    return z.squeeze(0).cpu()
