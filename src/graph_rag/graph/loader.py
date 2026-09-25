"""
Export the corpus index and graph to Neo4j for exploration and visualization.

Schema:
    (:CHUNK {chunk_id, pmid, chunk_index, text, embedding})
    (:ENTITY {name, label})
    (:CHUNK)-[:MENTIONS {weight}]->(:ENTITY)
    (:CHUNK)-[:NEXT]->(:CHUNK)
    (:CHUNK)-[:SIMILAR_TO {score}]->(:CHUNK)   only when kNN edges were built

Questions and relevance judgments are never written: they are evaluation labels, and
anything stored in the graph can end up as a model input.
"""

import numpy as np

from graph_rag.graph.crud import GraphCrud
from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.index.graph import CorpusGraph


class GraphExporter:
    CHUNK_VECTOR_INDEX = "chunk_vector_index"

    def __init__(self, crud: GraphCrud, batch_size: int = 1000):
        self.crud = crud
        self.batch_size = batch_size

    def _write(self, cypher: str, rows: list[dict]) -> None:
        self.crud.run_batched_write(cypher, rows, batch_size=self.batch_size)

    def create_constraints(self) -> None:
        for cypher in (
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:CHUNK) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:ENTITY) REQUIRE e.name IS UNIQUE",
        ):
            self.crud._execute_write(lambda tx, q=cypher: tx.run(q).consume())

    def export_chunks(self, index: CorpusIndex) -> None:
        rows = [
            {
                "chunk_id": str(row.chunk_id),
                "pmid": str(row.pmid),
                "chunk_index": int(row.chunk_index),
                "text": row.text,
                "embedding": index.embeddings[i].tolist(),
            }
            for i, row in enumerate(index.chunks.itertuples(index=False))
        ]
        self._write(
            """
            UNWIND $rows AS row
            MERGE (c:CHUNK {chunk_id: row.chunk_id})
            SET c.pmid = row.pmid, c.chunk_index = row.chunk_index, c.text = row.text
            WITH c, row
            CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
            """,
            rows,
        )
        self.crud.ensure_vector_index(
            index_name=self.CHUNK_VECTOR_INDEX,
            label="CHUNK",
            property_name="embedding",
            dimensions=index.dimension,
            similarity_function="cosine",
        )

    def export_entities(self, index: CorpusIndex, graph: CorpusGraph) -> None:
        self._write(
            "UNWIND $rows AS row MERGE (e:ENTITY {name: row.name}) SET e.label = row.label",
            [
                {"name": str(n), "label": str(lbl)}
                for n, lbl in zip(graph.entity_names, graph.entity_labels)
            ],
        )
        mentions = graph.mentions.tocoo()
        self._write(
            """
            UNWIND $rows AS row
            MATCH (c:CHUNK {chunk_id: row.chunk_id}), (e:ENTITY {name: row.entity})
            MERGE (c)-[m:MENTIONS]->(e)
            SET m.weight = row.weight
            """,
            [
                {
                    "chunk_id": str(index.chunk_ids[r]),
                    "entity": str(graph.entity_names[c]),
                    "weight": float(w),
                }
                for r, c, w in zip(mentions.row, mentions.col, mentions.data)
            ],
        )

    def export_chunk_edges(self, index: CorpusIndex, graph: CorpusGraph, include_knn: bool) -> None:
        src, dst = graph.next_edges
        self._write(
            """
            UNWIND $rows AS row
            MATCH (a:CHUNK {chunk_id: row.src}), (b:CHUNK {chunk_id: row.dst})
            MERGE (a)-[:NEXT]->(b)
            """,
            [
                {"src": str(index.chunk_ids[a]), "dst": str(index.chunk_ids[b])}
                for a, b in zip(src, dst)
            ],
        )
        if include_knn and graph.knn_edge_index is not None:
            src, dst = graph.knn_edge_index
            # Store each undirected pair once.
            keep = src < dst
            self._write(
                """
                UNWIND $rows AS row
                MATCH (a:CHUNK {chunk_id: row.src}), (b:CHUNK {chunk_id: row.dst})
                MERGE (a)-[s:SIMILAR_TO]->(b)
                SET s.score = row.score
                """,
                [
                    {
                        "src": str(index.chunk_ids[a]),
                        "dst": str(index.chunk_ids[b]),
                        "score": float(w),
                    }
                    for a, b, w in zip(src[keep], dst[keep], np.asarray(graph.knn_weights)[keep])
                ],
            )

    def export(self, index: CorpusIndex, graph: CorpusGraph, include_knn: bool = False) -> None:
        self.create_constraints()
        self.export_chunks(index)
        self.export_entities(index, graph)
        self.export_chunk_edges(index, graph, include_knn)
