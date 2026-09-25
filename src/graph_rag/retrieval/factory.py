"""Builds named retrievers from saved artifacts, and tunes their knobs on the dev split."""

import itertools
from pathlib import Path
from typing import Any, Optional

from graph_rag.config import settings
from graph_rag.data.bioasq import Question
from graph_rag.evaluation.metrics import recall_at_k
from graph_rag.hub import resolve_artifact
from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.index.graph import CorpusGraph
from graph_rag.index.linking import EntityLinker
from graph_rag.retrieval.base import BaseRetriever, hits_to_pmids

DEFAULT_ADAPTER_DIR = Path(settings.models_dir) / "query_adapter_semantic"
DEFAULT_GRAPH_ADAPTER_DIR = Path(settings.models_dir) / "query_adapter_graph"
DEFAULT_RERANKER_DIR = Path(settings.models_dir) / "graph_reranker"

RETRIEVERS = {
    "bm25": "BM25 over chunk text",
    "dense": "cosine search over chunk embeddings",
    "dense_adapter": "dense with the trained query adapter",
    "hybrid": "BM25 + dense fused with RRF",
    "hybrid_ce": "hybrid, re-ranked by a cross-encoder",
    "ppr": "personalized PageRank over the chunk-entity graph, seeded by hybrid + query entities",
    "expand": "hybrid, then relevance propagation to graph neighbours",
    "entity": "dense search restricted to chunks mentioning query entities",
    "graph_reranker": "hybrid, re-ranked by the query-conditioned graph re-ranker",
    "graph_space": "search in GNN embedding space with a graph-space adapter (RQ0 ablation)",
}

# Grids searched on the dev split by `tune`.
TUNING_GRIDS: dict[str, dict[str, list]] = {
    "ppr": {"damping": [0.3, 0.5, 0.7], "entity_seed_share": [0.0, 0.3, 0.6]},
    "expand": {"decay": [0.3, 0.5, 0.8], "n_hops": [1, 2]},
    "entity": {"entity_weight": [0.1, 0.3, 0.5]},
}


def encoder_from_index(index: CorpusIndex, device: str = "auto"):
    """The encoder used to build the index, with the same query/document prompts."""
    from graph_rag.llm.embeddings import EmbeddingModel

    meta = index.metadata
    return EmbeddingModel(
        model_name=index.embedding_model,
        device=device,
        query_prompt_name=meta.get("query_prompt_name"),
        query_prefix=meta.get("query_prefix", ""),
        document_prefix=meta.get("document_prefix", ""),
    )


class RetrieverFactory:
    def __init__(
        self,
        index: CorpusIndex,
        encoder: Any,
        graph: Optional[CorpusGraph] = None,
        adapter_dir: str | Path = DEFAULT_ADAPTER_DIR,
        graph_adapter_dir: str | Path = DEFAULT_GRAPH_ADAPTER_DIR,
        reranker_dir: str | Path = DEFAULT_RERANKER_DIR,
        cross_encoder: Optional[str] = None,
    ):
        self.index = index
        self.encoder = encoder
        self.graph = graph
        # Local directories or hf://<user>/<repo>[@revision] references.
        self.adapter_dir = adapter_dir
        self.graph_adapter_dir = graph_adapter_dir
        self.reranker_dir = reranker_dir
        self.cross_encoder = cross_encoder
        self._cache: dict[str, Any] = {}

    def _cached(self, key: str, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def _require_graph(self) -> CorpusGraph:
        if self.graph is None:
            raise ValueError("This retriever needs the corpus graph (build_index writes it).")
        return self.graph

    def linker(self) -> EntityLinker:
        return self._cached("linker", lambda: EntityLinker(self._require_graph(), self.encoder))

    def bm25(self):
        from graph_rag.retrieval.sparse import BM25Retriever

        return self._cached("bm25", lambda: BM25Retriever(self.index))

    def dense(self):
        from graph_rag.retrieval.dense import DenseRetriever

        return self._cached("dense", lambda: DenseRetriever(self.index, self.encoder))

    def hybrid(self):
        from graph_rag.retrieval.hybrid import HybridRetriever

        return self._cached(
            "hybrid",
            lambda: HybridRetriever(self.index, [self.bm25(), self.dense()], name="hybrid"),
        )

    def build(self, name: str, **params) -> BaseRetriever:
        if name not in RETRIEVERS:
            raise ValueError(f"Unknown retriever {name!r}; choose from {sorted(RETRIEVERS)}")
        if name == "bm25":
            return self.bm25()
        if name == "dense":
            return self.dense()
        if name == "hybrid":
            return self.hybrid()
        if name == "dense_adapter":
            from graph_rag.models.query_adapter import QueryAdapter
            from graph_rag.retrieval.dense import DenseRetriever

            adapter = QueryAdapter.load(resolve_artifact(self.adapter_dir))
            return DenseRetriever(self.index, self.encoder, adapter=adapter, name="dense_adapter")
        if name == "graph_space":
            from graph_rag.models.query_adapter import QueryAdapter
            from graph_rag.retrieval.dense import DenseRetriever

            adapter = QueryAdapter.load(resolve_artifact(self.graph_adapter_dir))
            return DenseRetriever(
                self.index, self.encoder, adapter=adapter, space="graph", name="graph_space"
            )
        if name == "hybrid_ce":
            from graph_rag.llm.reranker import DEFAULT_CROSS_ENCODER, CrossEncoderReranker
            from graph_rag.retrieval.rerank import CrossEncoderScorer, RerankingRetriever

            model = CrossEncoderReranker(self.cross_encoder or DEFAULT_CROSS_ENCODER)
            return RerankingRetriever(
                self.hybrid(),
                CrossEncoderScorer(self.index, model),
                candidate_k=params.get("candidate_k", 50),
                name="hybrid_ce",
            )
        if name == "ppr":
            from graph_rag.retrieval.pagerank import PersonalizedPageRankRetriever

            return PersonalizedPageRankRetriever(
                self.index, self._require_graph(), self.hybrid(), linker=self.linker(), **params
            )
        if name == "expand":
            from graph_rag.retrieval.graph_expansion import NeighborhoodExpansionRetriever

            return NeighborhoodExpansionRetriever(
                self.index, self._require_graph(), self.hybrid(), name="expand", **params
            )
        if name == "entity":
            from graph_rag.retrieval.graph_expansion import EntityAnchoredRetriever

            return EntityAnchoredRetriever(
                self.index, self._require_graph(), self.encoder, self.linker(), **params
            )
        # graph_reranker
        from graph_rag.models.graph_reranker import (
            CandidateGraphBuilder,
            GraphReranker,
            GraphRerankerScorer,
        )
        from graph_rag.retrieval.rerank import RerankingRetriever

        model, config = GraphReranker.load(resolve_artifact(self.reranker_dir))
        builder = CandidateGraphBuilder(
            self.index,
            self._require_graph(),
            self.encoder,
            self.linker(),
            edge_types=config.get("edge_types", ("entity", "next")),
        )
        return RerankingRetriever(
            self.hybrid(),
            GraphRerankerScorer(model, builder),
            candidate_k=config.get("candidate_k", 50),
            name="graph_reranker",
        )


def dev_recall(retriever: BaseRetriever, questions: list[Question], k: int = 10) -> float:
    retrieved = [hits_to_pmids(retriever.retrieve(q.question, top_k=k * 3), k) for q in questions]
    return recall_at_k([list(q.relevant_pmids) for q in questions], retrieved, k)


def tune(
    factory: RetrieverFactory,
    name: str,
    dev_questions: list[Question],
    k: int = 10,
    grid: Optional[dict[str, list]] = None,
    log=print,
) -> tuple[dict, float]:
    """Grid search on dev Recall@k. Returns (best params, best dev recall)."""
    grid = grid if grid is not None else TUNING_GRIDS.get(name, {})
    if not grid:
        return {}, dev_recall(factory.build(name), dev_questions, k)
    keys = sorted(grid)
    best_params, best_score = {}, -1.0
    for values in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        score = dev_recall(factory.build(name, **params), dev_questions, k)
        log(f"  {name} {params}: dev recall@{k}={score:.4f}")
        if score > best_score:
            best_params, best_score = params, score
    return best_params, best_score
