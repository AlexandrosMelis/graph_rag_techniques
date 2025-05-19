from typing import Any

from neo4j import GraphDatabase

from graph_embeddings.query_projection_model import project_query
from llms.embedding_model import EmbeddingModel
from retrieval_techniques.base_retriever import BaseRetriever


class GraphEmbeddingSimilarityRetriever(BaseRetriever):
    """
    Uses:
      1) a BERT embedding model to turn `query:str` → `q_emb: List[float]`
      2) your trained `projection_model` to map `q_emb` → graph-space vector
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
    ):
        super().__init__(embedding_model, neo4j_driver)
        self.projection_model = projection_model
        self.device = device

    def retrieve(self, query: str, top_k: int = 10):
        # 1) get raw BERT embedding
        #    assumes embed_documents returns List[List[float]]
        q_emb = self.embedding_model.embed_query(query)

        # 2) project into graph embedding space
        q_graph_vec = project_query(q_emb, self.projection_model, device=self.device)
        q_graph_list = q_graph_vec.tolist()

        # 3) run a cosine‐similarity ranking in Neo4j
        cypher = """
        WITH $q_vec AS q_emb
        MATCH (context:CONTEXT)
        WITH context, vector.similarity.cosine(q_emb, context.graph_embedding) AS score
        ORDER BY score DESC
        LIMIT $k
        RETURN
          id(context)          AS id,
          context.pmid         AS pmid,
          score            AS score
        """
        with self.neo4j_driver.session() as session:
            result = session.run(cypher, q_vec=q_graph_list, k=top_k)
            results = [dict(record) for record in result]

        return results
