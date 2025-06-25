from typing import Any, Dict, List
import uuid

from neo4j import Driver
from llms.embedding_model import EmbeddingModel
from retrieval_techniques.base_retriever import BaseRetriever

class PersonalizedPageRankRetriever(BaseRetriever):
    """
    A retriever that uses Personalized PageRank to find structurally important
    CONTEXT nodes relative to a query, and then combines this structural
    importance with semantic similarity for a final ranking.
    """

    name: str = "Personalized PageRank Retriever"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        neo4j_driver: Driver,
        alpha: float = 0.6,
        k_mesh: int = 5,
        k_context: int = 5,
    ):
        """
        Args:
            embedding_model: The embedding model instance.
            neo4j_driver: The Neo4j driver instance.
            alpha: Weight for combining PageRank and semantic scores.
                   final_score = (alpha * pr_score) + ((1 - alpha) * sem_score)
            k_mesh: The number of top MeSH nodes to use as seeds.
            k_context: The number of top CONTEXT nodes to use as seeds.
        """
        super().__init__(embedding_model, neo4j_driver)
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.k_mesh = k_mesh
        self.k_context = k_context
        self.gds_graph_name = f"pr-graph-{uuid.uuid4()}"

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.embed_query(query)

        with self.neo4j_driver.session() as session:
            try:
                # Stage 1: Identify Seed Nodes
                seed_node_ids = self._get_seed_nodes(session, query_embedding)
                if not seed_node_ids:
                    print("No seed nodes found. Aborting PageRank.")
                    return []

                # Stage 2: Execute Personalized PageRank
                pr_scores = self._run_pagerank(session, seed_node_ids)
                if not pr_scores:
                    print("PageRank did not return any scores. Aborting.")
                    return []

                # Stage 3: Re-score and Rank
                final_results = self._rescore_with_semantic_similarity(
                    session, pr_scores, query_embedding, top_k
                )

                return final_results

            finally:
                # Cleanup the in-memory GDS graph
                self._drop_gds_graph(session)

    def _get_seed_nodes(self, session, query_embedding: List[float]) -> List[int]:
        """Get the internal Neo4j IDs for the seed nodes."""
        mesh_seeds_query = """
            MATCH (m:MESH)
            WHERE m.embedding IS NOT NULL
            WITH m, vector.similarity.cosine($embedding, m.embedding) as score
            ORDER BY score DESC
            LIMIT $k
            RETURN id(m) as id
        """
        context_seeds_query = """
            MATCH (c:CONTEXT)
            WHERE c.embedding IS NOT NULL
            WITH c, vector.similarity.cosine($embedding, c.embedding) as score
            ORDER BY score DESC
            LIMIT $k
            RETURN id(c) as id
        """
        
        mesh_ids = [
            r["id"] for r in session.run(mesh_seeds_query, embedding=query_embedding, k=self.k_mesh)
        ]
        context_ids = [
            r["id"] for r in session.run(context_seeds_query, embedding=query_embedding, k=self.k_context)
        ]
        
        seed_ids = list(set(mesh_ids + context_ids))
        print(f"Found {len(seed_ids)} unique seed nodes ({len(mesh_ids)} MeSH, {len(context_ids)} CONTEXT).")
        return seed_ids

    def _run_pagerank(self, session, seed_node_ids: List[int]) -> Dict[int, float]:
        """Project graph and run Personalized PageRank."""
        # 1. Project graph
        project_query = """
            CALL gds.graph.project(
              $graphName,
              ['CONTEXT', 'MESH'],
              {
                IS_SIMILAR_TO: { orientation: 'UNDIRECTED' },
                HAS_MESH_TERM: { orientation: 'UNDIRECTED' }
              }
            )
        """
        session.run(project_query, graphName=self.gds_graph_name)

        # 2. Run PageRank
        pagerank_query = """
            CALL gds.pageRank.stream($graphName, {
              maxIterations: 20,
              dampingFactor: 0.85,
              sourceNodes: $seedNodeIds
            })
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).pmid AS pmid, score
        """
        
        results = session.run(pagerank_query, graphName=self.gds_graph_name, seedNodeIds=seed_node_ids)
        
        # We only care about CONTEXT nodes with pmids
        pr_scores = {r["pmid"]: r["score"] for r in results if r["pmid"]}
        print(f"Personalized PageRank returned scores for {len(pr_scores)} CONTEXT nodes.")
        return pr_scores

    def _rescore_with_semantic_similarity(
        self, session, pr_scores: Dict[str, float], query_embedding: List[float], top_k: int
    ) -> List[Dict[str, Any]]:
        """Combine PageRank scores with semantic similarity."""
        
        pmids = list(pr_scores.keys())

        # Normalize PageRank scores to be between 0 and 1
        max_pr_score = max(pr_scores.values()) if pr_scores else 1.0
        normalized_pr_scores = {
            pmid: score / max_pr_score for pmid, score in pr_scores.items()
        }

        # Get semantic scores for the same set of nodes
        semantic_query = """
            MATCH (c:CONTEXT)
            WHERE c.pmid IN $pmids
            RETURN c.pmid as pmid,
                   vector.similarity.cosine($embedding, c.embedding) as score,
                   id(c) as id
        """
        
        results = session.run(semantic_query, pmids=pmids, embedding=query_embedding)
        
        final_results = []
        for record in results:
            pmid = record["pmid"]
            semantic_score = record["score"]
            pr_score = normalized_pr_scores.get(pmid, 0)
            
            # Combine scores
            final_score = (self.alpha * pr_score) + ((1 - self.alpha) * semantic_score)
            
            final_results.append(
                {"id": record["id"], "pmid": pmid, "score": final_score}
            )
            
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]

    def _drop_gds_graph(self, session):
        """Clean up the in-memory GDS graph."""
        drop_query = "CALL gds.graph.drop($graphName, false) YIELD graphName"
        # The result might be empty if the graph didn't exist, so we consume it.
        list(session.run(drop_query, graphName=self.gds_graph_name))
        print(f"Dropped GDS graph: {self.gds_graph_name}")

    def __del__(self):
        # Ensure graph is dropped when the object is destroyed,
        # though explicit session management in retrieve is better.
        with self.neo4j_driver.session() as session:
            self._drop_gds_graph(session) 