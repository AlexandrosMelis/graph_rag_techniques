from abc import ABC, abstractmethod
from typing import Any

from neo4j import GraphDatabase

from llms.embedding_model import EmbeddingModel


class BaseRetriever(ABC):
    """
    Abstract class for retrieving relevant contexts for a given query.
    """

    name: str

    def __init__(
        self, embedding_model: EmbeddingModel, neo4j_driver: GraphDatabase.driver
    ):
        self.embedding_model = embedding_model
        self.neo4j_driver = neo4j_driver

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10):
        """
        Retrieve the top_k most relevant contexts for a given query.
        """
        pass
