import numpy as np
import pytest
import scipy.sparse as sp

from graph_rag.index.linking import EntityLinker
from graph_rag.retrieval.base import BaseRetriever, Hit, hits_to_pmids, make_hits
from graph_rag.retrieval.dense import DenseRetriever
from graph_rag.retrieval.graph_expansion import (
    EntityAnchoredRetriever,
    NeighborhoodExpansionRetriever,
)
from graph_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from graph_rag.retrieval.pagerank import (
    PersonalizedPageRankRetriever,
    column_stochastic,
    personalized_pagerank,
)
from graph_rag.retrieval.rerank import RerankingRetriever
from graph_rag.retrieval.sparse import BM25Retriever


class FixedRetriever(BaseRetriever):
    def __init__(self, index, pmids, name="fixed"):
        self.index, self.pmids, self.name = index, pmids, name

    def retrieve(self, query, top_k=10):
        ids = [int(np.where(self.index.pmids == p)[0][0]) for p in self.pmids][:top_k]
        return make_hits(self.index, ids, np.linspace(1.0, 0.5, len(ids)))


def ranked_pmids(retriever, query, k=3):
    return hits_to_pmids(retriever.retrieve(query, top_k=k))


def test_dense_retriever(index, encoder):
    assert ranked_pmids(DenseRetriever(index, encoder), "aspirin inflammation")[0] == "1"


def test_graph_space_requires_adapter(index, encoder):
    with pytest.raises(ValueError):
        DenseRetriever(index, encoder, space="graph")


def test_bm25_retriever(index):
    bm25 = BM25Retriever(index)
    assert ranked_pmids(bm25, "metformin glucose")[0] == "4"
    assert bm25.retrieve("zebra giraffe", top_k=3) == []


def test_reciprocal_rank_fusion():
    fused = reciprocal_rank_fusion([[1, 2], [2, 3]], k=60)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
    assert max(fused, key=fused.get) == 2


def test_hybrid_retriever(index, encoder):
    hybrid = HybridRetriever(index, [BM25Retriever(index), DenseRetriever(index, encoder)])
    assert ranked_pmids(hybrid, "oseltamivir influenza")[0] == "8"


def test_reranking_retriever_reorders(index):
    class Reverse:
        name = "reverse"

        def score_hits(self, query, hits):
            return np.arange(len(hits), dtype=float)

    reranked = RerankingRetriever(FixedRetriever(index, ["1", "2", "3"]), Reverse(), candidate_k=3)
    assert ranked_pmids(reranked, "q") == ["3", "2", "1"]


def test_hits_to_pmids_keeps_best_rank():
    hits = [Hit(0, "a#0", "a", 1.0), Hit(1, "b#0", "b", 0.9), Hit(2, "a#1", "a", 0.8)]
    assert hits_to_pmids(hits) == ["a", "b"]


def test_personalized_pagerank_on_a_path():
    adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    transition, dangling = column_stochastic(adj)
    r = personalized_pagerank(transition, dangling, np.array([1.0, 0, 0]), damping=0.5)
    assert r.sum() == pytest.approx(1.0)
    assert r[0] > r[1] > r[2] > 0


def test_ppr_reaches_the_bridge_passage(index, graph):
    ppr = PersonalizedPageRankRetriever(index, graph, FixedRetriever(index, ["5"]), seed_k=1)
    ranked = ranked_pmids(ppr, "brca1", k=3)
    assert ranked[0] == "5" and "6" in ranked
    assert "7" not in ranked


def test_ppr_uses_query_entities(index, graph, encoder):
    linker = EntityLinker(graph, encoder, embedding_top_k=0)
    ppr = PersonalizedPageRankRetriever(
        index, graph, FixedRetriever(index, ["7"]), linker=linker, seed_k=1, entity_seed_share=0.9
    )
    assert "6" in ranked_pmids(ppr, "what is known about tamoxifen", k=2)


def test_neighborhood_expansion_adds_neighbours(index, graph):
    expand = NeighborhoodExpansionRetriever(
        index, graph, FixedRetriever(index, ["5", "1"]), edge_types=("entity",), seed_k=1
    )
    assert "6" in ranked_pmids(expand, "brca1", k=3)


def test_entity_anchored_retriever(index, graph, encoder):
    linker = EntityLinker(graph, encoder, embedding_top_k=0)
    entity = EntityAnchoredRetriever(index, graph, encoder, linker)
    assert ranked_pmids(entity, "tamoxifen") == ["6"]
    # No linked entity: falls back to dense search over the whole corpus.
    assert len(entity.retrieve("zebra", top_k=3)) == 3
