from typing import Any, Optional

from neo4j import GraphDatabase

from projection_models.proj_model_with_attentive_pooling import project_query
from projection_models.projection_gat_model import project_query_gat, GATDataProcessor
from llms.embedding_model import EmbeddingModel
from retrieval_techniques.base_retriever import BaseRetriever


class GraphEmbeddingSimilarityRetriever(BaseRetriever):
    """
    Enhanced retriever that supports both traditional projection models and GAT models.
    
    Uses:
      1) a BERT embedding model to turn `query:str` → `q_emb: List[float]`
      2) your trained `projection_model` to map `q_emb` → graph-space vector
         - For traditional models: uses simple projection
         - For GAT models: constructs subgraphs and applies GAT attention
      3) a GDS cosine similarity call in Neo4j to rank all CONTEXT nodes by their
         `context.graph_embedding` vs. this projected query vector
    """

    name: str = "Graph Embedding Similarity Search"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        neo4j_driver: GraphDatabase.driver,
        projection_model: Any,
        device: str = "cpu",
        gat_data_processor: Optional[GATDataProcessor] = None,
        use_gat_projection: bool = False,
        top_k_contexts: int = 10
    ):
        super().__init__(embedding_model, neo4j_driver)
        self.projection_model = projection_model
        self.device = device
        self.use_gat_projection = use_gat_projection
        self.gat_data_processor = gat_data_processor
        self.top_k_contexts = top_k_contexts
        
        # Validate GAT configuration
        if self.use_gat_projection and self.gat_data_processor is None:
            raise ValueError("GAT data processor is required when use_gat_projection=True")
        
        # Update name based on model type
        if self.use_gat_projection:
            self.name = "GAT Graph Embedding Similarity Search"
        
        print(f"Initialized {self.name}")
        print(f"Device: {self.device}")
        print(f"GAT projection: {self.use_gat_projection}")

    def retrieve(self, query: str, top_k: int = 10):
        """
        Retrieve contexts using either traditional or GAT-based projection.
        
        Args:
            query: Input query string
            top_k: Number of top contexts to retrieve
            
        Returns:
            List of retrieved contexts with scores
        """
        try:
            # 1) Get raw BERT embedding
            q_emb = self.embedding_model.embed_query(query)
            
            # 2) Project into graph embedding space
            if self.use_gat_projection:
                # Use GAT-based projection with subgraph construction
                q_graph_vec = project_query_gat(
                    query_embedding=q_emb,
                    model=self.projection_model,
                    data_processor=self.gat_data_processor,
                    device=self.device,
                    top_k_contexts=self.top_k_contexts
                )
            else:
                # Use traditional projection
                q_graph_vec = project_query(q_emb, self.projection_model, device=self.device)
            
            # Convert to list for Neo4j query
            q_graph_list = q_graph_vec.tolist()
            
            # 3) Run cosine similarity ranking in Neo4j
            cypher = """
            WITH $q_vec AS q_emb
            MATCH (context:CONTEXT)
            WHERE context.graph_embedding IS NOT NULL
            WITH context, vector.similarity.cosine(q_emb, context.graph_embedding) AS score
            ORDER BY score DESC
            LIMIT $k
            RETURN
              id(context)          AS id,
              context.pmid         AS pmid,
              context.content      AS content,
              score                AS score
            """
            
            with self.neo4j_driver.session() as session:
                result = session.run(cypher, q_vec=q_graph_list, k=top_k)
                results = [dict(record) for record in result]
            
            print(f"Retrieved {len(results)} contexts using {self.name}")
            return results
            
        except Exception as e:
            print(f"Error in {self.name} retrieval: {e}")
            # Return empty results on error
            return []

    def get_retrieval_stats(self) -> dict:
        """Get statistics about the retrieval setup."""
        stats = {
            "retriever_name": self.name,
            "device": self.device,
            "use_gat_projection": self.use_gat_projection,
            "top_k_contexts": self.top_k_contexts if self.use_gat_projection else "N/A"
        }
        
        if self.use_gat_projection and hasattr(self.projection_model, 'hidden_dim'):
            stats.update({
                "gat_hidden_dim": self.projection_model.hidden_dim,
                "gat_n_layers": self.projection_model.n_layers,
                "gat_input_dim": self.projection_model.input_dim,
                "gat_output_dim": self.projection_model.output_dim
            })
        
        return stats
