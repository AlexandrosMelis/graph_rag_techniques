import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from graph_embeddings.gat_projection_model import GATQueryProjector
from graph_embeddings.query_gat_loader import QueryGATLoader


class QGPairDataset(Dataset):
    """
    Wraps your existing df of (q_emb, avg_c_graph) into a Dataset.
    Each item is a tuple (q_emb: Tensor, target: Tensor).
    """

    def __init__(self, df):
        self.qs = torch.tensor(df["q_emb"].tolist(), dtype=torch.float)
        self.tg = torch.tensor(df["c_emb"].tolist(), dtype=torch.float)

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, idx):
        return self.qs[idx], self.tg[idx]


def train_gat_projection(
    df,
    neo4j_params: dict,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    epochs: int = 50,
    val_split: float = 0.1,
    patience: int = 10,
    device: str = "cuda",
):
    # Dataset + split
    full_ds = QGPairDataset(df)
    n_val = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True
    )  # 1 per batch due to per‐sample subgraph
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    loader = QueryGATLoader(**neo4j_params)
    model = GATQueryProjector(in_dim, hidden_dim, out_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=5, factor=0.5, verbose=True
    )
    mse = MSELoss()

    best_val = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses, train_cos = [], []
        for q_emb, tgt in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            q_emb = q_emb.squeeze(0).to(device)  # [d_sem]
            tgt = tgt.squeeze(0).to(device)  # [d_graph]

            # build subgraph and move to device
            data = loader.build_subgraph(q_emb, top_k=10).to(device)

            opt.zero_grad()
            pred = model(data.x, data.edge_index)  # [out_dim]
            loss = mse(pred, tgt)
            loss.backward()
            opt.step()

            train_losses.append(loss.item())
            train_cos.append(
                cosine_similarity(pred.unsqueeze(0).cpu(), tgt.unsqueeze(0).cpu())[0, 0]
            )

        # validation
        model.eval()
        val_losses, val_cos = [], []
        with torch.no_grad():
            for q_emb, tgt in tqdm(val_loader, desc="Validate"):
                q_emb = q_emb.squeeze(0).to(device)
                tgt = tgt.squeeze(0).to(device)
                data = loader.build_subgraph(q_emb, top_k=10).to(device)
                pred = model(data.x, data.edge_index)
                val_losses.append(mse(pred, tgt).item())
                val_cos.append(
                    cosine_similarity(pred.unsqueeze(0).cpu(), tgt.unsqueeze(0).cpu())[
                        0, 0
                    ]
                )

        train_mse = np.mean(train_losses)
        val_mse = np.mean(val_losses)
        train_c = np.mean(train_cos)
        val_c = np.mean(val_cos)

        print(
            f"[Epoch {epoch:02d}] Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} || Train Cos: {train_c:.4f} | Val Cos: {val_c:.4f}"
        )
        history.append((train_mse, val_mse, train_c, val_c))

        sched.step(val_mse)
        if val_mse + 1e-6 < best_val:
            best_val = val_mse
            no_improve = 0
            torch.save(model.state_dict(), "best_gat_proj.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No validation improvement for {patience} epochs → stopping.")
                break

    # load best weights
    model.load_state_dict(torch.load("best_gat_proj.pt"))
    return model, history
