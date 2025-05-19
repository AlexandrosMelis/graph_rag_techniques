import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm


class DataProcessor:
    """Load QA_PAIR → CONTEXT embedding pairs from Neo4j."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password), database=database
        )
        self.database = database

    def embed_questions_and_store(self, embedding_model) -> None:
        query = """MATCH (qa:QA_PAIR) WHERE qa.embedding IS NULL RETURN qa.id as qa_id, qa.question as question"""
        with self.driver.session() as session:
            result = session.run(query)
            questions_df = pd.DataFrame([dict(record) for record in result])
        if not questions_df.empty:
            question_embeddings = embedding_model.embed_documents(
                questions_df["question"].tolist()
            )
            questions_df["embeddings"] = question_embeddings
            with self.driver.session() as session:
                for index, row in tqdm(questions_df.iterrows()):
                    query = """
                    MATCH (qa:QA_PAIR {id: $qa_id})
                    CALL db.create.setNodeVectorProperty(qa, 'embedding', $embedding)
                    """
                    session.run(query, qa_id=row["qa_id"], embedding=row["embeddings"])
            print("Question embeddings stored in the graph database!")
        else:
            print("No questions found for embedding.")

    def fetch_pairs(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          - qid: question node internal id
          - q_emb: list[float] (BERT)
          - c_emb: list[float] (graph)
        """
        query = """
        MATCH (qa:QA_PAIR)-[:HAS_CONTEXT]->(context:CONTEXT)
        RETURN qa.id AS qa_id, qa.embedding AS question_embedding, context.graph_embedding AS context_graph_embedding
        """
        with self.driver.session(database=self.database) as sess:
            result = sess.run(query)
            rows = []
            for rec in result:
                rows.append(
                    {
                        "qid": rec["qa_id"],
                        "q_emb": rec["question_embedding"],
                        "c_emb": rec["context_graph_embedding"],
                    }
                )
        return pd.DataFrame(rows)
