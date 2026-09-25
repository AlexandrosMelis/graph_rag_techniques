"""
Chunk and embed the corpus, extract entities and build the corpus graph.

Usage:
    python -m graph_rag.pipelines.build_index --entities mesh
    python -m graph_rag.pipelines.build_index --entities gliner --knn-k 10 --export-neo4j
    python -m graph_rag.pipelines.build_index --limit 2000   # quick smoke run on a subset
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from graph_rag.config import ConfigEnv, ConfigPath
from graph_rag.data.bioasq import load_corpus
from graph_rag.data.chunking import chunk_corpus
from graph_rag.data.entities import GlinerEntityExtractor, MeshEntityExtractor
from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.index.graph import build_corpus_graph
from graph_rag.llm.embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingModel


def extract_entities(kind: str, chunks: pd.DataFrame) -> pd.DataFrame:
    if kind == "mesh":
        from graph_rag.data.pubmed import PubMedClient

        cache = os.path.join(ConfigPath.EXTERNAL_DATA_DIR, "mesh_headings.jsonl")
        headings = PubMedClient().fetch_mesh_headings(chunks["pmid"].unique(), cache_path=cache)
        return MeshEntityExtractor(headings).extract(chunks)
    if kind == "gliner":
        return GlinerEntityExtractor().extract(chunks)
    return pd.DataFrame(columns=["chunk_id", "entity", "label"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default=ConfigPath.INDEX_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--query-prompt-name", default=None, help='e.g. "query" for Qwen3-Embedding'
    )
    parser.add_argument("--query-prefix", default="", help='e.g. "query: " for E5 models')
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--entities", choices=["mesh", "gliner", "none"], default="mesh")
    parser.add_argument("--max-entity-df", type=float, default=0.05)
    parser.add_argument("--knn-k", type=int, default=0, help="embedding kNN edges (ablation only)")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N passages")
    parser.add_argument("--export-neo4j", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    corpus = load_corpus()
    if args.limit:
        corpus = corpus.head(args.limit)
    chunks = chunk_corpus(corpus, args.chunk_size, args.chunk_overlap)
    print(f"{len(corpus)} passages -> {len(chunks)} chunks")

    encoder = EmbeddingModel(
        model_name=args.embedding_model,
        device=args.device,
        query_prompt_name=args.query_prompt_name,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
    )
    index = CorpusIndex.build(chunks, encoder)
    index.metadata.update(
        query_prompt_name=args.query_prompt_name,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_passages=len(corpus),
        entities=args.entities,
    )
    index.save(args.index_dir)

    entity_rows = extract_entities(args.entities, chunks)
    graph = build_corpus_graph(
        chunks,
        entity_rows,
        embeddings=index.embeddings,
        max_entity_df=args.max_entity_df,
        knn_k=args.knn_k,
        entity_encoder=encoder,
    )
    graph.save(Path(args.index_dir) / "graph")
    print(
        f"graph: {graph.n_entities} entities, {graph.mentions.nnz} mentions, "
        f"{graph.next_edges.shape[1]} next edges, knn_k={args.knn_k}"
    )

    if args.export_neo4j:
        from graph_rag.graph.connection import Neo4jConnection
        from graph_rag.graph.crud import GraphCrud
        from graph_rag.graph.loader import GraphExporter

        ConfigEnv.require(*ConfigEnv.NEO4J_VARS)
        connection = Neo4jConnection(
            uri=ConfigEnv.NEO4J_URI,
            user=ConfigEnv.NEO4J_USER,
            password=ConfigEnv.NEO4J_PASSWORD,
            database=ConfigEnv.NEO4J_DB,
        )
        try:
            GraphExporter(GraphCrud(connection)).export(index, graph, include_knn=args.knn_k > 0)
        finally:
            connection.close()

    print(f"index written to {args.index_dir} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
