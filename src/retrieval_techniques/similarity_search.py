from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from neo4j import Driver

from retrieval_techniques.base_retriever import BaseRetriever

# ======================================================================
# 1. Baseline similarity retriever
# ======================================================================


class BaselineSimilarityRetriever(BaseRetriever):
    """Cosine similarity over all CONTEXT nodes' BERT embeddings (baseline)."""

    name = "baseline_similarity"

    def retrieve(self, query: str, top_k: int = 10, **kwargs):
        query_embedding = self.embedding_model.embed_query(query)
        cypher = """
        MATCH (context:CONTEXT)
        WITH context, vector.similarity.cosine($embedding, context.embedding) AS score
        ORDER BY score DESC
        LIMIT $k
        RETURN id(context) AS id,
               context.pmid AS pmid,
               score AS score
        """
        with self.neo4j_driver.session() as session:
            result = session.run(cypher, embedding=query_embedding, k=top_k)
            results = [dict(record) for record in result]

        return results


# ======================================================================
# 2. N‑hop expansion retriever
# ======================================================================


class ExpandNHopsSimilarityRetriever(BaseRetriever):
    """Retrieve then expand via *IS_SIMILAR_TO* up to *n_hops* hops."""

    name = "expand_n_hops_similarity"

    def __init__(self, embedding_model: Any, neo4j_driver: Driver):
        pass

    def retrieve(self, query: str, top_k: int = 10, **kwargs):
        # the n_hops expansion should be provided in retrieve
        pass


# ======================================================================
# 3. MeSH‑anchored sub‑graph retriever
# ======================================================================


class MeshSubgraphSimilarityRetriever(BaseRetriever):
    """Restrict search to CONTEXTs attached to MeSH terms similar to the query."""

    name = "mesh_subgraph_similarity"

    def __init__(self, embedding_model: Any, neo4j_driver: Driver, mesh_k: int = 25):
        pass

    def retrieve(self, query: str, top_k: int = 10, **kwargs):
        pass


# ======================================================================
# Registry for convenience
# ======================================================================

RETRIEVER_REGISTRY = {
    cls.name: cls
    for cls in [
        BaselineSimilarityRetriever,
        ExpandNHopsSimilarityRetriever,
        MeshSubgraphSimilarityRetriever,
    ]
}
