import os
import re
import zlib

import numpy as np
import pandas as pd
import pytest

# graph_rag.config reads these at import time; tests never talk to real services.
for key, value in {
    "ENTREZ_EMAIL": "tests@example.com",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "unused",
    "NEO4J_PUBMED_DATABASE": "unused",
}.items():
    os.environ.setdefault(key, value)

from graph_rag.data.bioasq import Question  # noqa: E402
from graph_rag.data.chunking import chunk_corpus  # noqa: E402
from graph_rag.data.entities import MeshEntityExtractor  # noqa: E402
from graph_rag.index.corpus_index import CorpusIndex  # noqa: E402
from graph_rag.index.graph import build_corpus_graph  # noqa: E402

PASSAGES = {
    "1": "aspirin inhibits cyclooxygenase and reduces inflammation",
    "2": "cyclooxygenase enzymes produce prostaglandins during inflammation",
    "3": "insulin regulates glucose metabolism in diabetes",
    "4": "metformin lowers glucose in type 2 diabetes patients",
    "5": "brca1 mutations increase breast cancer risk",
    "6": "tamoxifen treats estrogen receptor positive tumours",
    "7": "influenza virus infects the respiratory tract",
    "8": "oseltamivir inhibits influenza neuraminidase",
}

MESH = {
    "1": ["Aspirin", "Cyclooxygenase*", "Inflammation/metabolism", "Humans"],
    "2": ["Cyclooxygenase", "Prostaglandins", "Inflammation", "Humans"],
    "3": ["Insulin", "Glucose", "Diabetes Mellitus", "Humans"],
    "4": ["Metformin", "Glucose", "Diabetes Mellitus", "Humans"],
    "5": ["BRCA1 Protein", "Breast Neoplasms", "Humans"],
    "6": ["Tamoxifen", "Breast Neoplasms", "Humans"],
    "7": ["Influenza, Human", "Respiratory Tract", "Humans"],
    "8": ["Oseltamivir", "Influenza, Human", "Humans"],
}


class FakeEncoder:
    """Deterministic hashed bag-of-words embedder: shared words mean higher cosine."""

    model_name = "fake-bow"

    def __init__(self, dim: int = 64):
        self.dimension = dim

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(token.encode()) % self.dimension] += 1.0
        if not v.any():
            v[0] = 1.0
        return v / np.linalg.norm(v)

    def encode_documents(self, texts, show_progress=False):
        return np.stack([self._vector(t) for t in texts]).astype(np.float32)

    def encode_query(self, text):
        return self._vector(text)


@pytest.fixture
def encoder():
    return FakeEncoder()


@pytest.fixture
def corpus():
    return pd.DataFrame({"pmid": list(PASSAGES), "text": list(PASSAGES.values())})


@pytest.fixture
def chunks(corpus):
    return chunk_corpus(corpus, chunk_size=64, chunk_overlap=8)


@pytest.fixture
def index(chunks, encoder):
    return CorpusIndex.build(chunks, encoder, show_progress=False)


@pytest.fixture
def graph(chunks, index, encoder):
    rows = MeshEntityExtractor(MESH).extract(chunks)
    return build_corpus_graph(
        chunks,
        rows,
        embeddings=index.embeddings,
        max_entity_df=0.5,
        knn_k=2,
        entity_encoder=encoder,
    )


@pytest.fixture
def questions():
    return [
        Question("q1", "does aspirin reduce inflammation", "yes", ("1", "2")),
        Question("q2", "which drug lowers glucose in diabetes", "metformin", ("4",)),
        Question(
            "q3", "what treats tumours in carriers of brca1 mutations", "tamoxifen", ("5", "6")
        ),
        Question("q4", "which antiviral inhibits influenza", "oseltamivir", ("8",)),
    ]
