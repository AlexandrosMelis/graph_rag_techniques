from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

from graph_rag.gnn.data import feature_cosine_auc
from graph_rag.gnn.encoder import GraphEncoder, link_logits


@dataclass
class GNNTrainingConfig:
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.2
    conv: str = "sage"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 300
    eval_every: int = 5
    patience: int = 10
    feature_weight: float = 0.1
    seed: int = 42


@torch.no_grad()
def evaluate_split(encoder: GraphEncoder, split: Data) -> tuple[float, float]:
    encoder.eval()
    z = encoder(split.x, split.edge_index)
    pos = link_logits(z, split.pos_edge_label_index)
    neg = link_logits(z, split.neg_edge_label_index)
    scores = torch.cat([pos, neg]).cpu().numpy()
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).cpu().numpy()
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def train_link_prediction(
    train: Data,
    val: Data,
    test: Data,
    config: GNNTrainingConfig = GNNTrainingConfig(),
    log: Callable[[str], None] = print,
) -> tuple[GraphEncoder, dict]:
    """
    Link prediction with a feature-preservation term (1 - cos(z, x)). The checkpoint
    with the best validation AUC is restored, and the cosine-of-features baseline AUC is
    reported next to the GNN so a tautological edge type is visible immediately.
    """
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = train.to(device), val.to(device), test.to(device)
    encoder = GraphEncoder(
        train.x.shape[1], config.hidden, config.layers, config.dropout, config.conv
    ).to(device)
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    history = {
        "epoch": [],
        "loss": [],
        "val_auc": [],
        "val_ap": [],
        "feature_cosine_val_auc": feature_cosine_auc(val.cpu()),
        "feature_cosine_test_auc": feature_cosine_auc(test.cpu()),
    }
    log(f"cosine(x_i, x_j) baseline: val AUC={history['feature_cosine_val_auc']:.4f}")
    best_auc, best_state, stale = -1.0, None, 0

    for epoch in range(1, config.epochs + 1):
        encoder.train()
        optimizer.zero_grad()
        z = encoder(train.x, train.edge_index)
        pos = link_logits(z, train.pos_edge_label_index)
        neg = link_logits(z, train.neg_edge_label_index)
        link_loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        )
        feature_loss = 1 - F.cosine_similarity(z, train.x, dim=-1).mean()
        loss = link_loss + config.feature_weight * feature_loss
        loss.backward()
        optimizer.step()

        if epoch % config.eval_every == 0 or epoch == 1:
            val_auc, val_ap = evaluate_split(encoder, val)
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["val_auc"].append(val_auc)
            history["val_ap"].append(val_ap)
            log(f"epoch {epoch:04d} loss={loss.item():.4f} val AUC={val_auc:.4f} AP={val_ap:.4f}")
            if val_auc > best_auc:
                best_auc, stale = val_auc, 0
                best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            else:
                stale += 1
                if stale >= config.patience:
                    log(f"early stopping at epoch {epoch}")
                    break

    encoder.load_state_dict(best_state)
    history["test_auc"], history["test_ap"] = evaluate_split(encoder, test)
    history["best_val_auc"] = best_auc
    history["config"] = asdict(config)
    log(
        f"test AUC={history['test_auc']:.4f} vs cosine baseline "
        f"{history['feature_cosine_test_auc']:.4f}"
    )
    return encoder.cpu(), history
