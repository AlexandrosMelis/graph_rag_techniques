from neo4j import GraphDatabase


class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str, database: str):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password), database=database
        )
        print(f"Using database: {database}")
        try:
            self.driver.verify_connectivity()
            print("Connection established!")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise e

    def get_driver(self) -> GraphDatabase:
        return self.driver

    def execute_query(self, query: str, params: dict) -> list:
        records, _, _ = self.driver.execute_query(query, params)
        if records:
            return [record.data() for record in records]
        else:
            return []

    def close(self):
        """Close the Neo4j driver."""
        print("Closing Neo4j connection...")
        self.driver.close()
