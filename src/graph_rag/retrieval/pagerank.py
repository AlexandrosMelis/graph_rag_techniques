from typing import Optional

import numpy as np
import scipy.sparse as sp

from graph_rag.index.corpus_index import CorpusIndex, top_k_indices
from graph_rag.index.graph import CorpusGraph
from graph_rag.index.linking import EntityLinker
from graph_rag.retrieval.base import BaseRetriever, Hit, make_hits


def column_stochastic(adjacency: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """Transition matrix T[i, j] = A[i, j] / deg(j), plus a mask of dangling columns."""
    degree = np.asarray(adjacency.sum(axis=0)).ravel()
    dangling = degree == 0
    inverse = np.where(dangling, 0.0, 1.0 / np.where(dangling, 1.0, degree))
    return (adjacency @ sp.diags(inverse)).tocsr(), dangling


def personalized_pagerank(
    transition: sp.csr_matrix,
    dangling: np.ndarray,
    personalization: np.ndarray,
    damping: float = 0.5,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Power iteration for r = (1 - d) p + d (T r). `damping` is the probability of following
    an edge rather than restarting at the seeds; mass on dangling nodes restarts at the seeds.
    """
    p = personalization / personalization.sum()
    r = p.copy()
    for _ in range(max_iter):
        spread = transition @ r + r[dangling].sum() * p
        r_next = (1 - damping) * p + damping * spread
        if np.abs(r_next - r).sum() < tol:
            return r_next
        r = r_next
    return r


class PersonalizedPageRankRetriever(BaseRetriever):
    """
    HippoRAG-style propagation over the chunk-entity graph. Seeds are the first stage's
    top chunks (reciprocal-rank weights) and the entities linked from the query; the
    ranking is the stationary PPR mass on chunk nodes. The graph and its transition
    matrix are built once, so a query costs a few sparse mat-vecs.
    """

    def __init__(
        self,
        index: CorpusIndex,
        graph: CorpusGraph,
        first_stage: BaseRetriever,
        linker: Optional[EntityLinker] = None,
        seed_k: int = 20,
        damping: float = 0.5,
        entity_seed_share: float = 0.5,
        include_next: bool = True,
        max_iter: int = 50,
        name: str = "ppr",
    ):
        if graph.n_chunks != index.n:
            raise ValueError("graph and index are not aligned")
        self.index = index
        self.graph = graph
        self.first_stage = first_stage
        self.linker = linker
        self.seed_k = seed_k
        self.damping = damping
        self.entity_seed_share = entity_seed_share
        self.max_iter = max_iter
        self.name = name
        self.transition, self.dangling = column_stochastic(graph.bipartite_adjacency(include_next))

    def personalization(self, query: str) -> np.ndarray:
        n_chunks = self.graph.n_chunks
        p = np.zeros(n_chunks + self.graph.n_entities)
        seeds = self.first_stage.retrieve(query, self.seed_k)
        for rank, hit in enumerate(seeds, start=1):
            p[hit.chunk_idx] += 1.0 / rank
        chunk_mass = p[:n_chunks].sum()
        if chunk_mass:
            p[:n_chunks] /= chunk_mass

        links = self.linker.link(query) if self.linker is not None else {}
        if links and self.entity_seed_share > 0:
            entity_p = np.zeros(self.graph.n_entities)
            for entity, weight in links.items():
                entity_p[entity] = weight
            entity_p /= entity_p.sum()
            share = self.entity_seed_share if chunk_mass else 1.0
            p[:n_chunks] *= 1 - share
            p[n_chunks:] = share * entity_p
        return p

    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        p = self.personalization(query)
        if p.sum() == 0:
            return []
        scores = personalized_pagerank(
            self.transition, self.dangling, p, damping=self.damping, max_iter=self.max_iter
        )[: self.graph.n_chunks]
        indices = top_k_indices(scores, top_k)
        indices = indices[scores[indices] > 0]
        return make_hits(self.index, indices, scores[indices])
