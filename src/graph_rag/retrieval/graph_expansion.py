"""Training-free graph baselines: neighbourhood expansion and entity-anchored search."""

from typing import Optional, Sequence

import numpy as np

from graph_rag.index.corpus_index import CorpusIndex, TextEncoder, top_k_indices
from graph_rag.index.graph import CorpusGraph
from graph_rag.index.linking import EntityLinker
from graph_rag.retrieval.base import BaseRetriever, Hit, make_hits, minmax


class NeighborhoodExpansionRetriever(BaseRetriever):
    """
    Seeds from the first stage, then propagates relevance along undirected graph edges:
    a neighbour at hop h scores seed_score * edge_weight * decay^h (max over paths), and
    each chunk keeps the larger of its own and its propagated score.
    """

    def __init__(
        self,
        index: CorpusIndex,
        graph: CorpusGraph,
        first_stage: BaseRetriever,
        edge_types: Sequence[str] = ("entity", "next"),
        seed_k: int = 10,
        n_hops: int = 1,
        decay: float = 0.5,
        name: Optional[str] = None,
    ):
        self.index = index
        self.first_stage = first_stage
        self.seed_k = seed_k
        self.n_hops = n_hops
        self.decay = decay
        self.adjacency = graph.chunk_adjacency(edge_types)
        self.name = name or f"expand[{'+'.join(edge_types)},hops={n_hops}]"

    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        seeds = self.first_stage.retrieve(query, max(self.seed_k, top_k))
        if not seeds:
            return []
        seed_scores = minmax(np.array([h.score for h in seeds]))
        scores = {h.chunk_idx: float(s) for h, s in zip(seeds[: self.seed_k], seed_scores)}
        final = {h.chunk_idx: float(s) for h, s in zip(seeds, seed_scores)}
        frontier = dict(scores)
        for _ in range(self.n_hops):
            reached: dict[int, float] = {}
            for node, score in frontier.items():
                row = self.adjacency.getrow(node)
                for nbr, weight in zip(row.indices, row.data):
                    value = score * weight * self.decay
                    if value > reached.get(nbr, 0.0):
                        reached[nbr] = value
            for node, value in reached.items():
                final[node] = max(final.get(node, 0.0), value)
            frontier = reached
        ranked = sorted(final.items(), key=lambda item: -item[1])[:top_k]
        return make_hits(self.index, [i for i, _ in ranked], [s for _, s in ranked])


class EntityAnchoredRetriever(BaseRetriever):
    """
    Restricts dense search to chunks that mention entities linked from the query and
    blends in how much of the query's linked-entity mass each chunk covers. Falls back
    to plain dense search when the query links to nothing.
    """

    def __init__(
        self,
        index: CorpusIndex,
        graph: CorpusGraph,
        encoder: TextEncoder,
        linker: EntityLinker,
        entity_weight: float = 0.3,
        name: str = "entity-anchored",
    ):
        self.index = index
        self.encoder = encoder
        self.linker = linker
        self.entity_weight = entity_weight
        self.name = name
        self.mentions_by_entity = graph.mentions.tocsc()

    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        query_vector = self.encoder.encode_query(query)
        links = self.linker.link(query, query_vector)
        candidates = set()
        for entity in links:
            column = self.mentions_by_entity.getcol(entity)
            candidates.update(column.indices.tolist())
        if not candidates:
            indices, scores = self.index.search(query_vector, top_k)
            return make_hits(self.index, indices, scores)

        candidates = np.array(sorted(candidates))
        dense = self.index.embeddings[candidates] @ query_vector
        link_weights = np.zeros(self.mentions_by_entity.shape[1])
        for entity, weight in links.items():
            link_weights[entity] = weight
        covered = (self.mentions_by_entity[candidates] > 0).astype(np.float64) @ link_weights
        coverage = covered / link_weights.sum()
        scores = (1 - self.entity_weight) * dense + self.entity_weight * coverage
        order = top_k_indices(scores, top_k)
        return make_hits(self.index, candidates[order], scores[order])
