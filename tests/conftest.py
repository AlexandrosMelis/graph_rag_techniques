import os

# graph_rag.config validates these at import time; tests never talk to real services.
for key, value in {
    "ENTREZ_EMAIL": "tests@example.com",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "unused",
    "NEO4J_PUBMED_DATABASE": "unused",
}.items():
    os.environ.setdefault(key, value)
