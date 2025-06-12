"""
Dual Projection Retriever for Contrastive Learning in Semantic and Graph Spaces

This module implements a retriever that uses a trained dual projection model to 
project queries into both semantic (SBERT) and graph (GNN) embedding spaces,
then performs similarity search in both spaces and combines the results.

The retriever supports:
1. Dual-space projection (semantic + graph)
2. Weighted combination of similarity scores
3. Flexible similarity search strategies
"""

from typing import Any, List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from neo4j import GraphDatabase

from llms.embedding_model import EmbeddingModel
from retrieval_techniques.base_retriever import BaseRetriever
from projection_models.dual_projection_model import DualProjectionModel, load_dual_projection_model


class DualProjectionRetriever(BaseRetriever):
    """
    Retriever that uses dual projection model to search in both semantic and graph spaces.
    
    The retriever:
    1. Takes a query string and converts it to BERT embeddings
    2. Projects the embeddings into both semantic and graph spaces using the dual projection model
    3. Performs similarity search in both spaces
    4. Combines the results using weighted scoring
    """

    name: str = "Dual Projection Similarity Search"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        neo4j_driver: GraphDatabase.driver,
        dual_projection_model: DualProjectionModel,
        device: str = "cpu",
        semantic_weight: float = 0.5,
        graph_weight: float = 0.5,
        combination_strategy: str = "weighted_sum",  # "weighted_sum", "max", "semantic_only", "graph_only"
        top_k_contexts: int = 10
    ):
        super().__init__(embedding_model, neo4j_driver)
        self.dual_projection_model = dual_projection_model.to(device)
        self.dual_projection_model.eval()
        self.device = device
        self.semantic_weight = semantic_weight
        self.graph_weight = graph_weight
        self.combination_strategy = combination_strategy
        self.top_k_contexts = top_k_contexts
        
        # Validate weights
        if abs(semantic_weight + graph_weight - 1.0) > 1e-6:
            print(f"⚠️ Warning: Semantic weight ({semantic_weight}) + Graph weight ({graph_weight}) != 1.0")
        
        # Update name based on strategy
        self.name = f"Dual Projection Similarity Search ({combination_strategy})"
        
        print(f"Initialized {self.name}")
        print(f"Device: {self.device}")
        print(f"Semantic weight: {self.semantic_weight}, Graph weight: {self.graph_weight}")
        print(f"Combination strategy: {self.combination_strategy}")

    def _project_query(self, query_embedding: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project query embedding into both semantic and graph spaces.
        
        Args:
            query_embedding: Raw BERT embedding of the query
            
        Returns:
            Tuple of (semantic_projection, graph_projection)
        """
        with torch.no_grad():
            # Convert to tensor
            query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get dual projections
            semantic_proj, graph_proj = self.dual_projection_model(query_tensor)
            
            # Convert back to numpy
            semantic_proj = semantic_proj.cpu().numpy().flatten()
            graph_proj = graph_proj.cpu().numpy().flatten()
            
        return semantic_proj, graph_proj

    def _semantic_similarity_search(self, semantic_projection: np.ndarray, top_k: int) -> List[Dict]:
        """
        Perform similarity search in semantic space.
        
        Args:
            semantic_projection: Projected query in semantic space
            top_k: Number of results to return
            
        Returns:
            List of results with semantic scores
        """
        # Convert to list for Neo4j query
        semantic_vec = semantic_projection.tolist()
        
        cypher = """
        WITH $sem_vec AS q_emb
        MATCH (context:CONTEXT)
        WHERE context.sbert_embedding IS NOT NULL
        WITH context, vector.similarity.cosine(q_emb, context.sbert_embedding) AS semantic_score
        ORDER BY semantic_score DESC
        LIMIT $k
        RETURN
          id(context)          AS id,
          context.pmid         AS pmid,
          context.content      AS content,
          semantic_score       AS semantic_score
        """
        
        with self.neo4j_driver.session() as session:
            result = session.run(cypher, sem_vec=semantic_vec, k=top_k)
            results = [dict(record) for record in result]
        
        return results

    def _graph_similarity_search(self, graph_projection: np.ndarray, top_k: int) -> List[Dict]:
        """
        Perform similarity search in graph space.
        
        Args:
            graph_projection: Projected query in graph space
            top_k: Number of results to return
            
        Returns:
            List of results with graph scores
        """
        # Convert to list for Neo4j query
        graph_vec = graph_projection.tolist()
        
        cypher = """
        WITH $graph_vec AS q_emb
        MATCH (context:CONTEXT)
        WHERE context.graph_embedding IS NOT NULL
        WITH context, vector.similarity.cosine(q_emb, context.graph_embedding) AS graph_score
        ORDER BY graph_score DESC
        LIMIT $k
        RETURN
          id(context)          AS id,  
          context.pmid         AS pmid,
          context.content      AS content,
          graph_score          AS graph_score
        """
        
        with self.neo4j_driver.session() as session:
            result = session.run(cypher, graph_vec=graph_vec, k=top_k)
            results = [dict(record) for record in result]
        
        return results

    def _combine_results(
        self, 
        semantic_results: List[Dict], 
        graph_results: List[Dict], 
        top_k: int
    ) -> List[Dict]:
        """
        Combine results from both semantic and graph searches.
        
        Args:
            semantic_results: Results from semantic space search
            graph_results: Results from graph space search
            top_k: Number of final results to return
            
        Returns:
            Combined and ranked results
        """
        # Create dictionaries for easy lookup
        semantic_scores = {r['pmid']: r.get('semantic_score', 0.0) for r in semantic_results}
        graph_scores = {r['pmid']: r.get('graph_score', 0.0) for r in graph_results}
        
        # Get all unique PMIDs and their context info
        all_pmids = set(semantic_scores.keys()) | set(graph_scores.keys())
        context_info = {}
        
        # Collect context information
        for result in semantic_results + graph_results:
            pmid = result['pmid']
            if pmid not in context_info:
                context_info[pmid] = {
                    'id': result['id'],
                    'pmid': pmid,
                    'content': result['content']
                }
        
        # Combine scores based on strategy
        combined_results = []
        
        for pmid in all_pmids:
            semantic_score = semantic_scores.get(pmid, 0.0)
            graph_score = graph_scores.get(pmid, 0.0)
            
            if self.combination_strategy == "weighted_sum":
                combined_score = (self.semantic_weight * semantic_score + 
                                self.graph_weight * graph_score)
            elif self.combination_strategy == "max":
                combined_score = max(semantic_score, graph_score)
            elif self.combination_strategy == "semantic_only":
                combined_score = semantic_score
            elif self.combination_strategy == "graph_only":
                combined_score = graph_score
            else:
                # Default to weighted sum
                combined_score = (self.semantic_weight * semantic_score + 
                                self.graph_weight * graph_score)
            
            result_item = context_info[pmid].copy()
            result_item.update({
                'score': combined_score,
                'semantic_score': semantic_score,
                'graph_score': graph_score
            })
            combined_results.append(result_item)
        
        # Sort by combined score and return top-k
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:top_k]

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve contexts using dual projection model.
        
        Args:
            query: Input query string
            top_k: Number of top contexts to retrieve
            
        Returns:
            List of retrieved contexts with combined scores
        """
        try:
            # 1. Get raw BERT embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # 2. Project into both spaces
            semantic_proj, graph_proj = self._project_query(query_embedding)
            
            # 3. Perform searches in both spaces
            # We retrieve more results initially to have better combination options
            search_k = min(top_k * 3, 50)  # Get 3x results for better combination
            
            semantic_results = self._semantic_similarity_search(semantic_proj, search_k)
            graph_results = self._graph_similarity_search(graph_proj, search_k)
            
            # 4. Combine results
            combined_results = self._combine_results(semantic_results, graph_results, top_k)
            
            print(f"Retrieved {len(combined_results)} contexts using {self.name}")
            print(f"Semantic results: {len(semantic_results)}, Graph results: {len(graph_results)}")
            
            return combined_results
            
        except Exception as e:
            print(f"Error in {self.name} retrieval: {e}")
            return []

    def get_retrieval_stats(self) -> Dict:
        """Get statistics about the retrieval setup."""
        stats = {
            "retriever_name": self.name,
            "device": self.device,
            "semantic_weight": self.semantic_weight,
            "graph_weight": self.graph_weight,
            "combination_strategy": self.combination_strategy,
            "top_k_contexts": self.top_k_contexts,
            "model_info": {
                "dim_sem": self.dual_projection_model.dim_sem,
                "dim_graph": self.dual_projection_model.dim_graph,
                "hidden_dims": self.dual_projection_model.hidden_dims,
                "p_dropout": self.dual_projection_model.p_dropout
            }
        }
        
        # Add model parameter count
        total_params = sum(p.numel() for p in self.dual_projection_model.parameters())
        stats["model_info"]["total_parameters"] = total_params
        
        return stats

    def test_projection(self, test_query: str = "What is the role of genes in cancer?") -> Dict:
        """
        Test the dual projection functionality.
        
        Args:
            test_query: Query to test with
            
        Returns:
            Dictionary with projection results and statistics
        """
        try:
            print(f"🧪 Testing dual projection with query: '{test_query}'")
            
            # Get embedding and projections
            query_embedding = self.embedding_model.embed_query(test_query)
            semantic_proj, graph_proj = self._project_query(query_embedding)
            
            # Compute some statistics
            semantic_norm = np.linalg.norm(semantic_proj)
            graph_norm = np.linalg.norm(graph_proj)
            projection_similarity = np.dot(semantic_proj, graph_proj) / (semantic_norm * graph_norm)
            
            test_results = {
                "test_query": test_query,
                "input_embedding_shape": len(query_embedding),
                "semantic_projection_shape": semantic_proj.shape,
                "graph_projection_shape": graph_proj.shape,
                "semantic_norm": float(semantic_norm),
                "graph_norm": float(graph_norm),
                "projection_similarity": float(projection_similarity),
                "semantic_projection_sample": semantic_proj[:5].tolist(),
                "graph_projection_sample": graph_proj[:5].tolist()
            }
            
            print(f"✅ Projection test successful:")
            print(f"   Input shape: {test_results['input_embedding_shape']}")
            print(f"   Semantic projection shape: {test_results['semantic_projection_shape']}")
            print(f"   Graph projection shape: {test_results['graph_projection_shape']}")
            print(f"   Projection similarity: {test_results['projection_similarity']:.4f}")
            
            return test_results
            
        except Exception as e:
            print(f"❌ Projection test failed: {e}")
            return {"error": str(e)}


def create_dual_projection_retriever(
    model_path: str,
    embedding_model: EmbeddingModel,
    neo4j_driver: GraphDatabase.driver,
    device: str = "auto",
    semantic_weight: float = 0.5,
    graph_weight: float = 0.5,
    combination_strategy: str = "weighted_sum"
) -> DualProjectionRetriever:
    """
    Factory function to create a dual projection retriever from saved model.
    
    Args:
        model_path: Path to saved dual projection model
        embedding_model: Embedding model instance
        neo4j_driver: Neo4j driver instance
        device: Device to use ("auto", "cuda", or "cpu")
        semantic_weight: Weight for semantic space scores
        graph_weight: Weight for graph space scores
        combination_strategy: Strategy for combining scores
        
    Returns:
        Configured DualProjectionRetriever instance
    """
    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    dual_projection_model, checkpoint = load_dual_projection_model(model_path, device)
    
    # Create retriever
    retriever = DualProjectionRetriever(
        embedding_model=embedding_model,
        neo4j_driver=neo4j_driver,
        dual_projection_model=dual_projection_model,
        device=device,
        semantic_weight=semantic_weight,
        graph_weight=graph_weight,
        combination_strategy=combination_strategy
    )
    
    return retriever 