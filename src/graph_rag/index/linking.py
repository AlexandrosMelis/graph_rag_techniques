import re
from typing import Optional

import numpy as np

from graph_rag.data.entities import normalize_entity
from graph_rag.index.corpus_index import TextEncoder, top_k_indices
from graph_rag.index.graph import CorpusGraph

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-']*")


class EntityLinker:
    """
    Maps a query to entity nodes of the corpus graph: exact n-gram matches against the
    entity vocabulary (weight 1.0) plus the nearest entity names in embedding space
    (weight = cosine similarity), when entity embeddings are available.
    """

    def __init__(
        self,
        graph: CorpusGraph,
        encoder: Optional[TextEncoder] = None,
        max_ngram: int = 6,
        embedding_top_k: int = 3,
        embedding_threshold: float = 0.5,
    ):
        self.graph = graph
        self.encoder = encoder
        self.vocabulary = graph.entity_index()
        self.max_ngram = max_ngram
        self.embedding_top_k = embedding_top_k
        self.embedding_threshold = embedding_threshold

    def dictionary_matches(self, query: str) -> list[int]:
        tokens = TOKEN_PATTERN.findall(query.lower())
        found = []
        for n in range(min(self.max_ngram, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                candidate = normalize_entity(" ".join(tokens[i : i + n]))
                entity = self.vocabulary.get(candidate)
                if entity is not None and entity not in found:
                    found.append(entity)
        return found

    def link(self, query: str, query_vector: Optional[np.ndarray] = None) -> dict[int, float]:
        links = {entity: 1.0 for entity in self.dictionary_matches(query)}
        embeddings = self.graph.entity_embeddings
        if embeddings is not None and self.embedding_top_k > 0:
            if query_vector is None and self.encoder is not None:
                query_vector = self.encoder.encode_query(query)
            if query_vector is not None:
                scores = embeddings @ query_vector
                for entity in top_k_indices(scores, self.embedding_top_k):
                    if scores[entity] >= self.embedding_threshold:
                        links[int(entity)] = max(links.get(int(entity), 0.0), float(scores[entity]))
        return links
