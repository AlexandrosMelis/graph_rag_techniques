from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from neo4j import Driver

from retrieval_techniques.base_retriever import BaseRetriever

# ======================================================================
# 1. Baseline similarity retriever
# ======================================================================


class BaselineBERTSimilarityRetriever(BaseRetriever):
    """Cosine similarity over all CONTEXT nodes' BERT embeddings (baseline)."""

    name = "Baseline BERT Embedding Similarity Search"

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
        super().__init__(embedding_model, neo4j_driver)

    def retrieve(self, query: str, top_k: int = 10, n_hops: int = 2, initial_k: int = None, **kwargs):
        """
        Retrieve contexts by first finding similar ones, then expanding via IS_SIMILAR_TO relationships.
        
        Args:
            query: Input query string
            top_k: Final number of results to return
            n_hops: Number of hops to expand (default: 2)
            initial_k: Number of initial contexts to start expansion from (default: top_k//2)
        """
        if initial_k is None:
            initial_k = max(1, top_k // 2)
        
        query_embedding = self.embedding_model.embed_query(query)
        
        # Step 1: Get initial similar contexts
        initial_cypher = """
        MATCH (context:CONTEXT)
        WITH context, vector.similarity.cosine($embedding, context.embedding) AS score
        ORDER BY score DESC
        LIMIT $k
        RETURN id(context) AS context_id, context.pmid AS pmid, score AS initial_score
        """
        
        with self.neo4j_driver.session() as session:
            initial_result = session.run(initial_cypher, embedding=query_embedding, k=initial_k)
            initial_contexts = [dict(record) for record in initial_result]
        
        if not initial_contexts:
            return []
        
        # Step 2: Expand via IS_SIMILAR_TO relationships up to n_hops
        initial_ids = [ctx['context_id'] for ctx in initial_contexts]
        
        # Build dynamic cypher query for n-hops expansion
        hop_patterns = []
        for i in range(1, n_hops + 1):
            if i == 1:
                hop_patterns.append(f"(start)-[:IS_SIMILAR_TO]->(hop{i}:CONTEXT)")
            else:
                hop_patterns.append(f"(hop{i-1})-[:IS_SIMILAR_TO]->(hop{i}:CONTEXT)")
        
        # Create UNION query for all hop levels
        union_parts = []
        for hop_num in range(1, n_hops + 1):
            hop_path = "->".join([f"(start)" if i == 0 else f"(hop{i})" for i in range(hop_num + 1)])
            pattern = "-[:IS_SIMILAR_TO]->".join([f"(start)" if i == 0 else f"(hop{i}:CONTEXT)" for i in range(hop_num + 1)])
            
            union_parts.append(f"""
                MATCH (start:CONTEXT)
                WHERE id(start) IN $initial_ids
                MATCH {pattern}
                WITH hop{hop_num} AS expanded_context, {hop_num} AS hop_distance
                RETURN DISTINCT id(expanded_context) AS context_id, expanded_context.pmid AS pmid, hop_distance
            """)
        
        expansion_cypher = " UNION ".join(union_parts)
        
        with self.neo4j_driver.session() as session:
            expansion_result = session.run(expansion_cypher, initial_ids=initial_ids)
            expanded_contexts = [dict(record) for record in expansion_result]
        
        # Step 3: Combine initial and expanded contexts, compute final scores
        all_context_ids = set(initial_ids)
        context_scores = {}
        
        # Add initial contexts with their scores
        for ctx in initial_contexts:
            context_scores[ctx['context_id']] = {
                'pmid': ctx['pmid'],
                'score': ctx['initial_score'],
                'hop_distance': 0
            }
        
        # Add expanded contexts with distance-based score decay
        for ctx in expanded_contexts:
            ctx_id = ctx['context_id']
            if ctx_id not in all_context_ids:
                all_context_ids.add(ctx_id)
                # Score decays with hop distance
                base_score = 0.5  # Base score for expanded contexts
                decay_factor = 0.8 ** ctx['hop_distance']
                context_scores[ctx_id] = {
                    'pmid': ctx['pmid'],
                    'score': base_score * decay_factor,
                    'hop_distance': ctx['hop_distance']
                }
        
        # Step 4: Rank and return top_k results
        final_results = []
        for ctx_id, ctx_data in context_scores.items():
            final_results.append({
                'id': ctx_id,
                'pmid': ctx_data['pmid'],
                'score': ctx_data['score']
            })
        
        # Sort by score and return top_k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:top_k]


# ======================================================================
# 3. MeSH‑anchored sub‑graph retriever
# ======================================================================


class MeshSubgraphSimilarityRetriever(BaseRetriever):
    """Restrict search to CONTEXTs attached to MeSH terms similar to the query."""

    name = "mesh_subgraph_similarity"

    def __init__(self, embedding_model: Any, neo4j_driver: Driver, mesh_k: int = 25):
        super().__init__(embedding_model, neo4j_driver)
        self.mesh_k = mesh_k

    def retrieve(self, query: str, top_k: int = 10, mesh_k: int = None, **kwargs):
        """
        Retrieve contexts by first finding similar MeSH terms, then searching only contexts 
        connected to those MeSH terms.
        
        Args:
            query: Input query string
            top_k: Number of final results to return
            mesh_k: Number of similar MeSH terms to use for filtering (default: self.mesh_k)
        """
        if mesh_k is None:
            mesh_k = self.mesh_k
        
        query_embedding = self.embedding_model.embed_query(query)
        
        # Step 1: Find similar MeSH terms
        mesh_cypher = """
        MATCH (mesh:MESH)
        WHERE mesh.embedding IS NOT NULL
        WITH mesh, vector.similarity.cosine($embedding, mesh.embedding) AS mesh_score
        ORDER BY mesh_score DESC
        LIMIT $mesh_k
        RETURN mesh.name AS mesh_name, mesh_score
        """
        
        with self.neo4j_driver.session() as session:
            mesh_result = session.run(mesh_cypher, embedding=query_embedding, mesh_k=mesh_k)
            similar_mesh_terms = [dict(record) for record in mesh_result]
        
        if not similar_mesh_terms:
            # Fallback to baseline search if no MeSH terms found
            baseline_cypher = """
            MATCH (context:CONTEXT)
            WITH context, vector.similarity.cosine($embedding, context.embedding) AS score
            ORDER BY score DESC
            LIMIT $k
            RETURN id(context) AS id, context.pmid AS pmid, score AS score
            """
            with self.neo4j_driver.session() as session:
                result = session.run(baseline_cypher, embedding=query_embedding, k=top_k)
                return [dict(record) for record in result]
        
        # Step 2: Find contexts connected to the similar MeSH terms
        mesh_names = [mesh['mesh_name'] for mesh in similar_mesh_terms]
        
        # Create a weighted score based on MeSH term similarity
        mesh_weights = {mesh['mesh_name']: mesh['mesh_score'] for mesh in similar_mesh_terms}
        
        context_cypher = """
        MATCH (context:CONTEXT)-[:HAS_MESH_TERM]-(mesh:MESH)
        WHERE mesh.name IN $mesh_names
        WITH DISTINCT context, mesh, vector.similarity.cosine($embedding, context.embedding) AS context_score
        RETURN id(context) AS id, 
               context.pmid AS pmid, 
               context_score,
               collect(mesh.name) AS mesh_terms
        ORDER BY context_score DESC
        LIMIT $k
        """
        
        with self.neo4j_driver.session() as session:
            context_result = session.run(
                context_cypher, 
                embedding=query_embedding, 
                mesh_names=mesh_names, 
                k=top_k * 2  # Get more results for better ranking
            )
            contexts = [dict(record) for record in context_result]
        
        # Step 3: Combine context similarity with MeSH term similarity
        final_results = []
        for ctx in contexts:
            context_score = ctx['context_score']
            mesh_terms = ctx['mesh_terms']
            
            # Calculate weighted MeSH score
            mesh_score_sum = sum(mesh_weights.get(term, 0) for term in mesh_terms)
            avg_mesh_score = mesh_score_sum / len(mesh_terms) if mesh_terms else 0
            
            # Combine scores (70% context similarity, 30% MeSH similarity)
            combined_score = 0.7 * context_score + 0.3 * avg_mesh_score
            
            final_results.append({
                'id': ctx['id'],
                'pmid': ctx['pmid'],
                'score': combined_score
            })
        
        # Sort by combined score and return top_k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:top_k]


# ======================================================================
# Registry for convenience
# ======================================================================

RETRIEVER_REGISTRY = {
    cls.name: cls
    for cls in [
        BaselineBERTSimilarityRetriever,
        ExpandNHopsSimilarityRetriever,
        MeshSubgraphSimilarityRetriever,
    ]
}
