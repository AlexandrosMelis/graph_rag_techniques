"""Chunk and embed the corpus, extract entities and build the corpus graph."""

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from graph_rag.config import settings
from graph_rag.data.bioasq import load_corpus
from graph_rag.data.chunking import chunk_corpus
from graph_rag.data.entities import GlinerEntityExtractor, MeshEntityExtractor
from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.index.graph import build_corpus_graph
from graph_rag.llm.embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingModel
from graph_rag.tracking import numeric_metrics, tracked_run

ENTITY_SOURCES = ("mesh", "gliner", "none")


@dataclass
class IndexConfig:
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    query_prompt_name: Optional[str] = None
    query_prefix: str = ""
    document_prefix: str = ""
    device: str = "auto"
    chunk_size: int = 384
    chunk_overlap: int = 64
    entities: str = "mesh"
    max_entity_df: float = 0.05
    knn_k: int = 0
    limit: Optional[int] = None
    export_neo4j: bool = False
    index_dir: str = field(default_factory=lambda: str(settings.index_dir))

    def __post_init__(self):
        if self.entities not in ENTITY_SOURCES:
            raise ValueError(f"entities must be one of {ENTITY_SOURCES}, got {self.entities!r}")


def mesh_cache_path() -> Path:
    return settings.external_dir / "mesh_headings.jsonl"


def load_passages(config: IndexConfig) -> pd.DataFrame:
    corpus = load_corpus()
    return corpus.head(config.limit) if config.limit else corpus


def fetch_mesh(config: IndexConfig, progress: Optional[Callable[[str], None]] = None) -> dict:
    """Fetch (or resume fetching) MeSH headings for every passage into the on-disk cache."""
    from graph_rag.data.pubmed import PubMedClient

    pmids = load_passages(config)["pmid"].tolist()
    headings = PubMedClient().fetch_mesh_headings(
        pmids,
        cache_path=mesh_cache_path(),
        progress=(lambda done, total: progress(f"MeSH {done}/{total}")) if progress else None,
    )
    return {"pmids": len(pmids), "with_mesh": sum(1 for p in pmids if headings.get(p))}


def extract_entities(
    config: IndexConfig, chunks: pd.DataFrame, progress: Optional[Callable[[str], None]]
) -> pd.DataFrame:
    if config.entities == "mesh":
        from graph_rag.data.pubmed import PubMedClient

        headings = PubMedClient().fetch_mesh_headings(
            chunks["pmid"].unique(), cache_path=mesh_cache_path()
        )
        return MeshEntityExtractor(headings).extract(chunks)
    if config.entities == "gliner":
        if progress:
            progress("running GLiNER")
        return GlinerEntityExtractor().extract(chunks)
    return pd.DataFrame(columns=["chunk_id", "entity", "label"])


def export_to_neo4j(index: CorpusIndex, graph, include_knn: bool) -> None:
    from graph_rag.graph.connection import Neo4jConnection
    from graph_rag.graph.crud import GraphCrud
    from graph_rag.graph.loader import GraphExporter

    connection = Neo4jConnection(**settings.neo4j_connection_kwargs())
    try:
        GraphExporter(GraphCrud(connection)).export(index, graph, include_knn=include_knn)
    finally:
        connection.close()


def run(
    config: IndexConfig = IndexConfig(),
    log: Callable[[str], None] = print,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    started = time.perf_counter()
    with tracked_run("build_index", params=asdict(config)) as tracker:
        corpus = load_passages(config)
        chunks = chunk_corpus(corpus, config.chunk_size, config.chunk_overlap)
        log(f"{len(corpus)} passages -> {len(chunks)} chunks")

        encoder = EmbeddingModel(
            model_name=config.embedding_model,
            device=config.device,
            query_prompt_name=config.query_prompt_name,
            query_prefix=config.query_prefix,
            document_prefix=config.document_prefix,
        )
        index = CorpusIndex.build(
            chunks,
            encoder,
            show_progress=progress is None,
            progress=(lambda done, total: progress(f"embedded {done}/{total}"))
            if progress
            else None,
        )
        index.metadata.update(
            query_prompt_name=config.query_prompt_name,
            query_prefix=config.query_prefix,
            document_prefix=config.document_prefix,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            n_passages=len(corpus),
            entities=config.entities,
        )
        index.save(config.index_dir)

        entity_rows = extract_entities(config, chunks, progress)
        graph = build_corpus_graph(
            chunks,
            entity_rows,
            embeddings=index.embeddings,
            max_entity_df=config.max_entity_df,
            knn_k=config.knn_k,
            entity_encoder=encoder,
        )
        graph.save(Path(config.index_dir) / "graph")
        stats = {
            "passages": len(corpus),
            "chunks": index.n,
            "entities": graph.n_entities,
            "mentions": int(graph.mentions.nnz),
            "next_edges": int(graph.next_edges.shape[1]),
            "knn_edges": 0 if graph.knn_edge_index is None else int(graph.knn_edge_index.shape[1]),
        }
        log(
            f"graph: {stats['entities']} entities, {stats['mentions']} mentions, "
            f"{stats['next_edges']} next edges, {stats['knn_edges']} kNN edges"
        )

        if config.export_neo4j:
            export_to_neo4j(index, graph, include_knn=config.knn_k > 0)

        stats["seconds"] = time.perf_counter() - started
        stats["index_dir"] = config.index_dir
        tracker.log_metrics(numeric_metrics(stats))
        tracker.log_dict(stats, "index_stats.json")
        log(f"index written to {config.index_dir} in {stats['seconds']:.0f}s")
        return stats
