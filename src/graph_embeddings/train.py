import json
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GAE


@torch.no_grad()
def evaluate(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    data: Data,
) -> Tuple[float, float]:
    encoder.eval()
    predictor.eval()
    z = encoder(data.x, data.edge_index)
    pos_score = predictor(z, data.pos_edge_label_index)
    neg_score = predictor(z, data.neg_edge_label_index)
    scores = torch.cat([pos_score, neg_score], dim=0).cpu()
    labels = torch.cat(
        [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
    ).cpu()
    from sklearn.metrics import average_precision_score, roc_auc_score

    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def train_model(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_data: Data,
    val_data: Data,
    epochs: int = 200,
    λ_feat: float = 0.5,
    patience: int = 10,
) -> dict:
    history = {"epoch": [], "loss": [], "val_auc": [], "val_ap": []}
    best_auc = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        encoder.train()
        predictor.train()
        optimizer.zero_grad()

        # forward
        z = encoder(train_data.x, train_data.edge_index)
        pos_score = predictor(z, train_data.pos_edge_label_index)
        neg_score = predictor(z, train_data.neg_edge_label_index)

        # losses
        edge_labels = torch.cat(
            [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
        )
        edge_preds = torch.cat([pos_score, neg_score], dim=0)
        recon_loss = F.binary_cross_entropy(edge_preds, edge_labels)
        feat_loss = F.mse_loss(z, train_data.x)
        loss = recon_loss + λ_feat * feat_loss

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
        )
        optimizer.step()

        # eval & logging
        if epoch % 10 == 0 or epoch == 1:
            val_auc, val_ap = evaluate(encoder, predictor, val_data)
            scheduler.step(val_auc)
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["val_auc"].append(val_auc)
            history["val_ap"].append(val_ap)
            print(
                f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}"
            )

            # early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} evaluations, stopping early.")
                break

    return history


def save_model(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"encoder": encoder.state_dict(), "predictor": predictor.state_dict()}, path
    )
    print(f"Model saved to {path}.")


def save_metrics(metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {path}.")
