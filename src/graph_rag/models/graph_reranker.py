"""
Query-conditioned graph re-ranker.

For each query the first stage returns K candidate chunks. They become the nodes of a
small graph whose edges come from the corpus graph (shared entities, same-passage
adjacency, optionally kNN). Node inputs depend on the query (q * d, dense score,
first-stage rank and score, linked-entity coverage), so message passing spreads
query relevance between related candidates instead of smoothing static content.

The candidate graph is built the same way at training and inference time, and gold
labels are only ever training targets, never node inputs.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch

from graph_rag.data.bioasq import Question
from graph_rag.evaluation.metrics import recall_at_k
from graph_rag.index.corpus_index import CorpusIndex, TextEncoder
from graph_rag.index.graph import CorpusGraph
from graph_rag.index.linking import EntityLinker
from graph_rag.models.losses import multi_positive_info_nce
from graph_rag.retrieval.base import BaseRetriever, Hit, minmax

SCALAR_FEATURES = ("dense_score", "reciprocal_rank", "first_stage_score", "entity_coverage")


class CandidateGraphBuilder:
    def __init__(
        self,
        index: CorpusIndex,
        graph: CorpusGraph,
        encoder: TextEncoder,
        linker: Optional[EntityLinker] = None,
        edge_types: Sequence[str] = ("entity", "next"),
    ):
        self.index = index
        self.encoder = encoder
        self.linker = linker
        self.edge_types = tuple(edge_types)
        self.adjacency = graph.chunk_adjacency(edge_types)
        self.mentions = graph.mentions.tocsr()

    @property
    def feature_dim(self) -> int:
        return self.index.dimension + len(SCALAR_FEATURES)

    def build(self, query: str, hits: Sequence[Hit], gold_pmids: Optional[set[str]] = None) -> Data:
        indices = np.array([h.chunk_idx for h in hits], dtype=np.int64)
        query_vector = self.encoder.encode_query(query)
        documents = self.index.embeddings[indices]

        dense = documents @ query_vector
        reciprocal_rank = 1.0 / np.arange(1, len(hits) + 1)
        first_stage = minmax(np.array([h.score for h in hits]))
        coverage = np.zeros(len(hits))
        if self.linker is not None:
            links = self.linker.link(query, query_vector)
            if links:
                weights = np.zeros(self.mentions.shape[1])
                for entity, weight in links.items():
                    weights[entity] = weight
                coverage = (self.mentions[indices] > 0).astype(np.float64) @ weights / weights.sum()
        scalars = np.stack([dense, reciprocal_rank, first_stage, coverage], axis=1)
        x = np.concatenate([documents * query_vector, scalars], axis=1).astype(np.float32)

        sub = self.adjacency[indices][:, indices].tocoo()
        data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(np.vstack([sub.row, sub.col]).astype(np.int64)),
            edge_attr=torch.from_numpy(sub.data.astype(np.float32)).unsqueeze(-1),
            chunk_idx=torch.from_numpy(indices),
            num_nodes=len(hits),
        )
        if gold_pmids is not None:
            data.y = torch.tensor([h.pmid in gold_pmids for h in hits], dtype=torch.float32)
        return data


class GraphReranker(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_scalars: int = len(SCALAR_FEATURES),
        hidden: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = dict(
            embedding_dim=embedding_dim,
            n_scalars=n_scalars,
            hidden=hidden,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )
        self.embedding_dim = embedding_dim
        self.interaction = nn.Linear(embedding_dim, hidden)
        self.scalars = nn.Linear(n_scalars, hidden)
        # Self-loops keep every candidate's own evidence in its update.
        self.convs = nn.ModuleList(
            GATv2Conv(
                hidden,
                hidden,
                heads=heads,
                concat=False,
                edge_dim=1,
                add_self_loops=True,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # Linear path on the scalar features so the untrained model can follow the first stage.
        self.skip = nn.Linear(n_scalars, 1)

    def forward(self, data: Data) -> torch.Tensor:
        interaction, scalars = data.x[:, : self.embedding_dim], data.x[:, self.embedding_dim :]
        h = F.relu(self.interaction(interaction) + self.scalars(scalars))
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + self.dropout(F.relu(conv(h, data.edge_index, data.edge_attr))))
        return self.head(h).squeeze(-1) + self.skip(scalars).squeeze(-1)

    def save(self, directory: str | Path, extra: Optional[dict] = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "reranker.pt")
        (directory / "reranker.json").write_text(
            json.dumps({**self.config, **(extra or {})}, indent=2)
        )

    @classmethod
    def load(cls, directory: str | Path) -> tuple["GraphReranker", dict]:
        directory = Path(directory)
        config = json.loads((directory / "reranker.json").read_text())
        model = cls(
            **{
                k: config[k]
                for k in ("embedding_dim", "n_scalars", "hidden", "heads", "layers", "dropout")
            }
        )
        model.load_state_dict(torch.load(directory / "reranker.pt", map_location="cpu"))
        model.eval()
        return model, config


class GraphRerankerScorer:
    """Adapts a trained `GraphReranker` to `RerankingRetriever`."""

    def __init__(self, model: GraphReranker, builder: CandidateGraphBuilder):
        self.model = model.eval()
        self.builder = builder
        self.name = "graph-reranker"

    @torch.no_grad()
    def score_hits(self, query: str, hits: Sequence[Hit]) -> np.ndarray:
        return self.model(self.builder.build(query, hits)).numpy()


def grouped_info_nce(
    scores: torch.Tensor, labels: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    dense_scores, mask = to_dense_batch(scores.unsqueeze(-1), batch, fill_value=float("-inf"))
    dense_labels, _ = to_dense_batch(labels.unsqueeze(-1), batch, fill_value=0.0)
    return multi_positive_info_nce(dense_scores.squeeze(-1), dense_labels.squeeze(-1) > 0)


@dataclass
class RerankerTrainingConfig:
    candidate_k: int = 50
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 32
    patience: int = 5
    eval_k: int = 10
    seed: int = 42


def build_examples(
    questions: list[Question],
    first_stage: BaseRetriever,
    builder: CandidateGraphBuilder,
    candidate_k: int,
) -> list[Data]:
    return [
        builder.build(
            q.question, first_stage.retrieve(q.question, candidate_k), set(q.relevant_pmids)
        )
        for q in questions
    ]


@torch.no_grad()
def reranked_recall(
    model: GraphReranker,
    examples: list[Data],
    questions: list[Question],
    index: CorpusIndex,
    k: int,
) -> float:
    model.eval()
    retrieved = []
    for data in examples:
        order = torch.argsort(model(data), descending=True).numpy()
        chunk_idx = data.chunk_idx.numpy()[order]
        retrieved.append(list(dict.fromkeys(index.pmids[chunk_idx].tolist()))[:k])
    return recall_at_k([list(q.relevant_pmids) for q in questions], retrieved, k)


def train_graph_reranker(
    train_questions: list[Question],
    dev_questions: list[Question],
    first_stage: BaseRetriever,
    builder: CandidateGraphBuilder,
    config: RerankerTrainingConfig = RerankerTrainingConfig(),
    log: Callable[[str], None] = print,
) -> tuple[GraphReranker, dict]:
    """Listwise training over each query's candidate graph; best dev Recall@k is kept."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_examples = build_examples(train_questions, first_stage, builder, config.candidate_k)
    learnable = [d for d in train_examples if d.y.sum() > 0]
    dev_examples = build_examples(dev_questions, first_stage, builder, config.candidate_k)
    log(
        f"{len(learnable)}/{len(train_examples)} train queries have a gold chunk among the candidates"
    )

    model = GraphReranker(
        builder.index.dimension,
        hidden=config.hidden,
        heads=config.heads,
        layers=config.layers,
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loader = DataLoader(learnable, batch_size=config.batch_size, shuffle=True)

    first_stage_recall = recall_at_k(
        [list(q.relevant_pmids) for q in dev_questions],
        [
            list(dict.fromkeys(builder.index.pmids[d.chunk_idx.numpy()].tolist()))[: config.eval_k]
            for d in dev_examples
        ],
        config.eval_k,
    )
    history = {
        "epoch": [],
        "loss": [],
        "dev_recall": [],
        "first_stage_dev_recall": first_stage_recall,
    }
    log(f"first stage dev recall@{config.eval_k}={first_stage_recall:.4f}")
    best_recall, best_state, stale = -1.0, None, 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            loss = grouped_info_nce(model(batch), batch.y, batch.batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        recall = reranked_recall(model, dev_examples, dev_questions, builder.index, config.eval_k)
        history["epoch"].append(epoch)
        history["loss"].append(float(np.mean(losses)))
        history["dev_recall"].append(recall)
        log(f"epoch {epoch}: loss={np.mean(losses):.4f} dev recall@{config.eval_k}={recall:.4f}")
        if recall > best_recall:
            best_recall, stale = recall, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                log(f"early stopping after {epoch} epochs")
                break

    model.load_state_dict(best_state)
    model.eval()
    history.update(
        best_dev_recall=best_recall, config=asdict(config), edge_types=list(builder.edge_types)
    )
    return model, history
