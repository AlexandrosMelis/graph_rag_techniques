import argparse
import os
from datetime import datetime

from graph_rag.config import ConfigEnv, ConfigPath
from graph_rag.data.dataset_builder import DatasetConstructor
from graph_rag.data.pubmed import MeshTermFetcher, PubMedArticleFetcher
from graph_rag.data.bioasq import BioASQDataReader
from graph_rag.evaluation.executor import run_retrieval
from graph_rag.evaluation.metrics import run_evaluation
from graph_rag.graph.connection import Neo4jConnection
from graph_rag.graph.crud import GraphCrud
from graph_rag.graph.loader import GraphLoader
from graph_rag.llm.embeddings import EmbeddingModel
from graph_rag.retrieval.base import BaseRetriever
from graph_rag.retrieval.non_ml import BaselineBERTSimilarityRetriever
from graph_rag.utils import read_json_file, save_json_file


def construct_graph_dataset(
    asq_reader: BioASQDataReader, file_name: str = "bioasq_test.parquet"
):
    """
    The function aims to construct the dataset for the graph database.
    The following steps are performed:
    1. Read the BIOASQ data from the parquet file.
    2. Fetch the articles from PubMed for the distinct PMIDs mentioned in the BIOASQ data.
    3. Fetch the Mesh Term Definitions for the Mesh Terms mentioned in the PubMed articles.
    4. Combine the BIOASQ, PubMed, and Mesh Term Definitions to create the graph data for loading into Neo4j.
    """

    article_fetcher = PubMedArticleFetcher()

    # 1. Read the BIOASQ parquet data file
    asq_data_file_path = os.path.join(ConfigPath.RAW_DATA_DIR, file_name)
    asq_data = asq_reader.read_parquet_file(file_path=asq_data_file_path)
    pmids_for_fetch = asq_reader.get_distinct_pmids()

    # 2. Fetch articles from PubMed
    pubmed_data = article_fetcher.fetch_articles(pmids=pmids_for_fetch)
    mesh_terms = article_fetcher.get_mesh_terms()

    # 3. Fetch mesh term definitions
    mesh_fetcher = MeshTermFetcher()
    mesh_term_definitions = mesh_fetcher.fetch_definitions(mesh_terms=mesh_terms)
    print(f"Total Mesh Term Definitions: {len(mesh_term_definitions)}")

    # 4. Combine BIOASQ and PubMed to create the graph data for loading into Neo4j
    dataset_constructor = DatasetConstructor(
        bioasq_data=asq_data, pubmed_data=pubmed_data
    )
    dataset_constructor.create_graph_data()


def load_graph_data(embedding_model, graph_crud):
    """
    The function aims to load the graph data into Neo4j.
    1. Initialize the EmbeddingModel, Neo4jConnection, GraphCrud, TextSplitter, and GraphLoader.
    2. Load the Mesh Nodes into the Neo4j graph.
    3. Load the QA Pairs, Articles, and Context Nodes into the Neo4j graph.
    4. Load the Similarity Relationships between Context Nodes into the Neo4j graph.
    """
    graph_data = read_json_file(
        file_path=os.path.join(ConfigPath.RAW_DATA_DIR, "bioasq_graph_data.json")
    )
    print(f"Created graph data len: {len(graph_data)}")

    graph_loader = GraphLoader(
        data=graph_data,
        embedding_model=embedding_model,
        crud=graph_crud,
    )
    graph_loader.load_all()


def evaluate_retriever_without_llm(
    source_data: list,
    retriever: BaseRetriever,
    retriever_args: dict,
    output_dir_path: str = None,
    k_eval_values: list = [1, 3, 5, 10],
):

    save_json_file(
        file_path=os.path.join(output_dir_path, "retriever_args.json"),
        data=retriever_args,
    )

    retrieval_results = run_retrieval(
        source_data=source_data,
        retriever=retriever,
        retriever_args=retriever_args,
        output_dir=output_dir_path,
    )

    metrics, _ = run_evaluation(
        retrieval_results=retrieval_results,
        k_values=k_eval_values,
        output_dir=output_dir_path,
        retriever_name=retriever.name,
    )
    print(f"Evaluation metrics for {retriever.name}:\n{metrics}\n")
    print("\n\nEvaluation without LLM completed successfully!")


def evaluate_baseline(output_root: str = ConfigPath.RESULTS_DIR) -> None:
    """Evaluate the BERT similarity baseline on the local BioASQ test split."""
    asq_reader = BioASQDataReader(samples_start=0)
    data = asq_reader.read_parquet_file(
        file_path=os.path.join(ConfigPath.RAW_DATA_DIR, "bioasq_test.parquet")
    )
    neo4j_connection = Neo4jConnection(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
    )
    retriever = BaselineBERTSimilarityRetriever(
        embedding_model=EmbeddingModel(),
        neo4j_driver=neo4j_connection.get_driver(),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = os.path.join(output_root, f"BERT_Similarity_Search_{timestamp}")
    os.makedirs(output_dir_path, exist_ok=True)
    evaluate_retriever_without_llm(
        source_data=data,
        retriever=retriever,
        retriever_args={"top_k": 10},
        output_dir_path=output_dir_path,
        k_eval_values=[1, 5, 10],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BioASQ graph and run the baseline.")
    parser.add_argument(
        "step",
        choices=["construct", "load", "evaluate-baseline"],
        help="construct: fetch PubMed/MeSH data; load: write it to Neo4j; "
        "evaluate-baseline: score the BERT similarity retriever",
    )
    parser.add_argument("--file-name", default="bioasq_test.parquet")
    args = parser.parse_args()

    if args.step == "construct":
        construct_graph_dataset(asq_reader=BioASQDataReader(), file_name=args.file_name)
    elif args.step == "load":
        neo4j_connection = Neo4jConnection(
            uri=ConfigEnv.NEO4J_URI,
            user=ConfigEnv.NEO4J_USER,
            password=ConfigEnv.NEO4J_PASSWORD,
            database=ConfigEnv.NEO4J_DB,
        )
        load_graph_data(
            embedding_model=EmbeddingModel(),
            graph_crud=GraphCrud(neo4j_connection=neo4j_connection),
        )
    else:
        evaluate_baseline()


if __name__ == "__main__":
    main()
