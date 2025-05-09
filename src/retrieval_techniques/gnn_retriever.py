from abc import ABC, abstractmethod
from typing import Any

from graph_embeddings.query_projection_model import project_query


class Retriever(ABC):
    """
    Abstract class for retrieving relevant contexts for a given query.
    """

    def __init__(self, embedding_model, neo4j_driver):
        self.embedding_model = embedding_model
        self.neo4j_driver = neo4j_driver

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10):
        """
        Retrieve the top_k most relevant contexts for a given query.
        """
        pass


class GNNRetriever(Retriever):
    """
    Uses:
      1) a BERT embedder to turn `query:str` → `q_emb: List[float]`
      2) your trained `projection_model` to map `q_emb` → graph-space vector
      3) a GDS cosine similarity call in Neo4j to rank all CONTEXT nodes by their
         `context.graph_embedding` vs. this projected query vector
    """

    def __init__(
        self,
        embedding_model: Any,
        neo4j_driver: Any,
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
        MATCH (c:CONTEXT)
        WITH c, vector.similarity.cosine(q_emb, c.graph_embedding) AS sim
        ORDER BY sim DESC
        LIMIT $k
        RETURN
          id(c)          AS context_node_id,
          c.pmid         AS context_pmid,
          c.text_content AS context_text,
          sim            AS score
        """
        with self.neo4j_driver.session() as session:
            result = session.run(cypher, q_vec=q_graph_list, k=top_k)
            results = [dict(record) for record in result]

        return results
