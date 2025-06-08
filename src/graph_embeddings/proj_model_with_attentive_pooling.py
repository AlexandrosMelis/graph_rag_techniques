import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from neo4j import GraphDatabase
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


class DataProcessor:
    def __init__(self, uri: str, user: str, password: str, database: str):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password), database=database
        )

    def fetch_positive_pairs(self) -> pd.DataFrame:
        query = """
        MATCH (qa:QA_PAIR)-[:HAS_CONTEXT]->(ctx:CONTEXT)
        RETURN qa.id AS qa_id,
               qa.embedding AS q_emb,
               id(ctx) AS ctx_id,
               ctx.graph_embedding AS c_emb
        """
        rows = []
        with self.driver.session() as sess:
            for r in sess.run(query):
                rows.append(dict(r))
        return pd.DataFrame(rows)

    def fetch_all_contexts(self) -> pd.DataFrame:
        query = """
        MATCH (ctx:CONTEXT)
        OPTIONAL MATCH (ctx)-[:IS_SIMILAR_TO]->(nbr:CONTEXT)
        RETURN id(ctx) AS ctx_id,
               ctx.graph_embedding    AS c_emb,
               collect(nbr.graph_embedding) AS nbr_embs
        """
        rows = []
        with self.driver.session() as sess:
            for r in sess.run(query):
                rows.append(dict(r))
        return pd.DataFrame(rows)


def aggregate_graphsage(
    emb: List[float], nbrs: List[List[float]], alpha: float = 0.5
) -> List[float]:
    self_v = np.array(emb, dtype=np.float32)
    if nbrs:
        nbr_v = np.mean(np.array(nbrs, dtype=np.float32), axis=0)
        return (alpha * self_v + (1 - alpha) * nbr_v).tolist()
    return emb


def attentive_pooling(
    q_emb: np.ndarray, c_embs: np.ndarray, temp: float = 0.1
) -> np.ndarray:
    q = q_emb.flatten()  # shape (D,)
    assert c_embs.ndim == 2  # shape (N, D)
    sims = np.sum(c_embs * q[None, :], axis=1) / temp  # (N,)
    sims = sims - np.max(sims)
    weights = np.exp(sims)
    weights /= weights.sum()
    return (weights[:, None] * c_embs).sum(axis=0)


# -----------------------------------------------------------------------------
# 2) BUILD TRIPLETS W/ ATTENTIVE POSITIVES & HARD NEGATIVES
# -----------------------------------------------------------------------------
def build_triplets(
    pos_df: pd.DataFrame,
    all_ctx: pd.DataFrame,
    neg_per_pos: int = 2,
    agg_alpha: float = 0.5,
    attn_temp: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[int, List[int]]]:
    random.seed(seed)
    np.random.seed(seed)

    # map ctx_id -> (emb, nbrs)
    ctx_map = {r.ctx_id: (r.c_emb, r.nbr_embs) for r in all_ctx.itertuples()}

    # group contexts per question
    q2ctxs: Dict[int, List[int]] = {}
    q2qemb: Dict[int, np.ndarray] = {}
    for r in pos_df.itertuples():
        q2ctxs.setdefault(r.qa_id, []).append(r.ctx_id)
        q2qemb[r.qa_id] = np.array(r.q_emb, dtype=np.float32)

    triplets = []
    for qid, pos_ctxs in q2ctxs.items():
        q_emb = q2qemb[qid]
        # attentive positive pooling
        pos_embs = np.vstack([ctx_map[cid][0] for cid in pos_ctxs])  # (M, D)
        pos_target = attentive_pooling(q_emb, pos_embs, temp=attn_temp)
        # hard negatives
        all_ids = list(ctx_map.keys())
        neg_cands = list(set(all_ids) - set(pos_ctxs))
        sampled = random.sample(neg_cands, k=min(neg_per_pos, len(neg_cands)))
        for neg_id in sampled:
            neg_emb, nbrs = ctx_map[neg_id]
            neg_agg = aggregate_graphsage(neg_emb, nbrs, alpha=agg_alpha)
            triplets.append(
                {
                    "qa_id": qid,
                    "q_emb": q_emb.tolist(),
                    "c_pos": pos_target.tolist(),
                    "c_neg": neg_agg,
                }
            )
    return pd.DataFrame(triplets), q2qemb, q2ctxs


# -----------------------------------------------------------------------------
# 3) DATASET & RESIDUAL‑STACK MLP
# -----------------------------------------------------------------------------
class TripletDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.q = torch.tensor(df["q_emb"].tolist(), dtype=torch.float32)
        self.pos = torch.tensor(df["c_pos"].tolist(), dtype=torch.float32)
        self.neg = torch.tensor(df["c_neg"].tolist(), dtype=torch.float32)
        self.qid = df["qa_id"].tolist()

    def __len__(self) -> int:
        return len(self.q)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.q[idx], self.pos[idx], self.neg[idx], self.qid[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.lin(x))
        return self.act(x + h)


class QueryProjectionModel(nn.Module):
    def __init__(
        self,
        dim_sem: int,
        dim_graph: int,
        hidden_dims: List[int] = [512, 2048, 1024],
        p_dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        prev = dim_sem
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p_dropout),
                ResidualBlock(h),
            ]
            prev = h
        layers.append(nn.Linear(prev, dim_graph))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# 4) METRICS, TRAIN LOOP & EARLY STOPPING
# -----------------------------------------------------------------------------
def recall_at_k(
    model: nn.Module,
    q2qemb: Dict[int, np.ndarray],
    q2ctxs: Dict[int, List[int]],
    all_ctx: pd.DataFrame,
    k: int,
    device: str,
) -> float:
    ctx_ids = all_ctx["ctx_id"].tolist()
    C = []
    for r in all_ctx.itertuples():
        C.append(aggregate_graphsage(r.c_emb, r.nbr_embs))
    C = torch.tensor(C, dtype=torch.float32).to(device)
    recalls = []
    for qid, pos_ids in q2ctxs.items():
        q = torch.tensor(q2qemb[qid], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            p = model(q)  # (1,D)
            sims = F.cosine_similarity(p, C)  # (N,)
        topk = {ctx_ids[i] for i in sims.topk(k).indices.cpu().tolist()}
        recalls.append(int(bool(topk & set(pos_ids))))
    return float(np.mean(recalls))


def plot_history(train_l, val_l, path: str) -> None:
    plt.figure()
    x = range(1, len(train_l) + 1)
    plt.plot(x, train_l, label="Train")
    plt.plot(x, val_l, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train(
    uri: str,
    user: str,
    password: str,
    database: str,
    model_dir: str,
    batch_size: int = 64,
    lr: float = 1e-3,
    wd: float = 1e-4,
    epochs: int = 50,
    val_ratio: float = 0.1,
    neg_per_pos: int = 2,
    patience: int = 10,
    hidden_dims: List[int] = [512, 2048, 1024],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> QueryProjectionModel:
    dp = DataProcessor(uri, user, password, database)
    pos_df = dp.fetch_positive_pairs()
    all_ctx = dp.fetch_all_contexts()

    trip_df, q2qemb, q2ctxs = build_triplets(pos_df, all_ctx, neg_per_pos=neg_per_pos)
    ds = TripletDataset(trip_df)
    print("\n\nTriplets created!\n")

    # split QA IDs
    qids = list(q2qemb.keys())
    random.shuffle(qids)
    n_val = int(len(qids) * val_ratio)
    val_q = set(qids[:n_val])

    idx_tr = trip_df[~trip_df["qa_id"].isin(val_q)].index.tolist()
    idx_va = trip_df[trip_df["qa_id"].isin(val_q)].index.tolist()
    train_ds = Subset(ds, idx_tr)
    val_ds = Subset(ds, idx_va)
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    tr_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    va_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    dim_sem = len(pos_df["q_emb"].iloc[0])
    dim_graph = len(all_ctx["c_emb"].iloc[0])
    model = QueryProjectionModel(dim_sem, dim_graph, hidden_dims).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    os.makedirs(model_dir, exist_ok=True)
    best_val, no_imp = float("inf"), 0
    hist_tr, hist_va = [], []

    for ep in range(1, epochs + 1):
        model.train()
        tloss = []
        for q, p, n, _ in tqdm(tr_ld, desc=f"Epoch {ep:02d}"):
            q, p, n = q.to(device), p.to(device), n.to(device)
            l = loss_fn(model(q), p, n)
            opt.zero_grad()
            l.backward()
            opt.step()
            tloss.append(l.item())
        avg_tr = float(np.mean(tloss))
        hist_tr.append(avg_tr)

        model.eval()
        vloss = []
        with torch.no_grad():
            for q, p, n, _ in va_ld:
                q, p, n = q.to(device), p.to(device), n.to(device)
                vloss.append(loss_fn(model(q), p, n).item())
        avg_va = float(np.mean(vloss))
        hist_va.append(avg_va)

        # retrieval metrics
        r5 = recall_at_k(
            model, q2qemb, {q: q2ctxs[q] for q in val_q}, all_ctx, 5, device
        )
        r10 = recall_at_k(
            model, q2qemb, {q: q2ctxs[q] for q in val_q}, all_ctx, 10, device
        )
        print(
            f"[Ep {ep:02d}] Train={avg_tr:.4f} Validation={avg_va:.4f} R@5={r5:.3f} R@10={r10:.3f}"
        )

        # early stopping
        if avg_va < best_val - 1e-6:
            best_val, no_imp = avg_va, 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best.pt"))
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"No imp for {patience} epochs → stopping.")
                break

    plot_history(hist_tr, hist_va, os.path.join(model_dir, "loss_curve.png"))
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "best.pt"), map_location=device)
    )
    return model


def project_query(
    q_emb: List[float], model: nn.Module, device: str = "cpu"
) -> torch.Tensor:
    model.eval()
    x = torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    return out.squeeze(0).cpu()
