"""
Low-rank residual query adapter.

q' = normalize(q + U D q), with U initialised to zero so the adapter starts as the
identity: training can only move away from the pretrained embedding geometry when
the supervision justifies it. With rank 64 on 768-d vectors it has ~98k parameters.
The document index is never re-embedded.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_rag.data.bioasq import Question
from graph_rag.evaluation.metrics import recall_at_k
from graph_rag.index.corpus_index import CorpusIndex, TextEncoder, top_k_indices
from graph_rag.models.losses import multi_positive_info_nce


class QueryAdapter(nn.Module):
    def __init__(self, dim: int, rank: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dim, self.rank, self.dropout_p = dim, rank, dropout
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return F.normalize(queries + self.up(self.dropout(self.down(queries))), dim=-1)

    @torch.no_grad()
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        self.eval()
        tensor = torch.as_tensor(np.asarray(vectors, dtype=np.float32))
        return self(tensor).numpy()

    def save(self, directory: str | Path, extra: Optional[dict] = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "adapter.pt")
        config = {"dim": self.dim, "rank": self.rank, "dropout": self.dropout_p, **(extra or {})}
        (directory / "adapter.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "QueryAdapter":
        directory = Path(directory)
        config = json.loads((directory / "adapter.json").read_text())
        adapter = cls(config["dim"], config["rank"], config["dropout"])
        adapter.load_state_dict(torch.load(directory / "adapter.pt", map_location="cpu"))
        adapter.eval()
        return adapter


@dataclass
class AdapterTrainingConfig:
    rank: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 64
    temperature: float = 0.05
    hard_negatives: int = 8
    hard_negative_pool: int = 50
    patience: int = 3
    eval_k: int = 10
    seed: int = 42


def positive_chunks(questions: list[Question], index: CorpusIndex) -> list[np.ndarray]:
    """Chunk indices of every gold passage of each question."""
    by_pmid: dict[str, list[int]] = {}
    for i, pmid in enumerate(index.pmids.tolist()):
        by_pmid.setdefault(pmid, []).append(i)
    return [
        np.array(sorted({c for p in q.relevant_pmids for c in by_pmid.get(p, [])}), dtype=np.int64)
        for q in questions
    ]


def mine_hard_negatives(
    query_vectors: np.ndarray,
    documents: np.ndarray,
    positives: list[np.ndarray],
    pool_size: int,
) -> list[np.ndarray]:
    """Top-scoring chunks under the base embeddings, excluding every gold chunk of the query."""
    pools = []
    for vector, pos in zip(query_vectors, positives):
        scores = documents @ vector
        scores[pos] = -np.inf
        pools.append(top_k_indices(scores, pool_size))
    return pools


def recall_with_transform(
    transform: Callable[[np.ndarray], np.ndarray],
    query_vectors: np.ndarray,
    questions: list[Question],
    index: CorpusIndex,
    k: int,
    space: str = "semantic",
) -> float:
    mapped = transform(query_vectors)
    retrieved = []
    for vector in mapped:
        chunk_idx, _ = index.search(vector, top_k=k * 3, space=space)
        retrieved.append(list(dict.fromkeys(index.pmids[chunk_idx].tolist()))[:k])
    return recall_at_k([list(q.relevant_pmids) for q in questions], retrieved, k)


def train_query_adapter(
    train_questions: list[Question],
    dev_questions: list[Question],
    index: CorpusIndex,
    encoder: TextEncoder,
    config: AdapterTrainingConfig = AdapterTrainingConfig(),
    space: str = "semantic",
    log: Callable[[str], None] = print,
) -> tuple[QueryAdapter, dict]:
    """
    Multi-positive InfoNCE over in-batch documents plus mined hard negatives. The
    checkpoint with the best dev Recall@k is returned.
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    documents = index.matrix(space)
    positives = positive_chunks(train_questions, index)
    usable = [i for i, pos in enumerate(positives) if len(pos)]
    train_questions = [train_questions[i] for i in usable]
    positives = [positives[i] for i in usable]
    positive_sets = [set(pos.tolist()) for pos in positives]

    train_vectors = np.stack([encoder.encode_query(q.question) for q in train_questions])
    dev_vectors = np.stack([encoder.encode_query(q.question) for q in dev_questions])
    pools = mine_hard_negatives(train_vectors, documents, positives, config.hard_negative_pool)

    adapter = QueryAdapter(train_vectors.shape[1], config.rank, config.dropout)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    doc_tensor = torch.as_tensor(documents)
    query_tensor = torch.as_tensor(train_vectors)

    baseline = recall_with_transform(
        lambda v: v, dev_vectors, dev_questions, index, config.eval_k, space
    )
    history = {"epoch": [0], "loss": [None], "dev_recall": [baseline]}
    log(f"epoch 0 (identity adapter): dev recall@{config.eval_k}={baseline:.4f}")
    best_recall, best_state, stale = (
        baseline,
        {k: v.clone() for k, v in adapter.state_dict().items()},
        0,
    )

    for epoch in range(1, config.epochs + 1):
        adapter.train()
        order = np.random.permutation(len(train_questions))
        losses = []
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            doc_ids = []
            for i in batch:
                doc_ids.append(int(np.random.choice(positives[i])))
                pool = pools[i]
                take = min(config.hard_negatives, len(pool))
                doc_ids.extend(np.random.choice(pool, size=take, replace=False).tolist())
            doc_ids = np.array(sorted(set(doc_ids)))
            mask = torch.tensor([[d in positive_sets[i] for d in doc_ids] for i in batch])
            logits = adapter(query_tensor[batch]) @ doc_tensor[doc_ids].T / config.temperature
            loss = multi_positive_info_nce(logits, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        recall = recall_with_transform(
            adapter.transform, dev_vectors, dev_questions, index, config.eval_k, space
        )
        history["epoch"].append(epoch)
        history["loss"].append(float(np.mean(losses)))
        history["dev_recall"].append(recall)
        log(f"epoch {epoch}: loss={np.mean(losses):.4f} dev recall@{config.eval_k}={recall:.4f}")
        if recall > best_recall:
            best_recall, stale = recall, 0
            best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                log(f"early stopping after {epoch} epochs")
                break

    adapter.load_state_dict(best_state)
    adapter.eval()
    history.update(best_dev_recall=best_recall, config=asdict(config), space=space)
    return adapter, history
