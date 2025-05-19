import os
from datetime import datetime
from typing import Any

from configs import ConfigEnv, ConfigPath
from data_collection.dataset_constructor import DatasetConstructor
from data_collection.fetcher import MeshTermFetcher, PubMedArticleFetcher
from data_collection.reader import BioASQDataReader
from evaluation.executor import collect_generated_answers, run_retrieval
from evaluation.llm_based_eval import run_evaluation_on_generated_answers
from evaluation.non_llm_based_eval import run_evaluation
from knowledge_graph.connection import Neo4jConnection
from knowledge_graph.crud import GraphCrud
from knowledge_graph.loader import GraphLoader
from llms.embedding_model import EmbeddingModel
from retrieval_techniques.base_retriever import BaseRetriever
from retrieval_techniques.similarity_search import BaselineBERTSimilarityRetriever
from utils.utils import read_json_file, save_json_file


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


def evaluate_retriever_with_llm(
    source_data: list,
    retriever: Any,
    output_dir_path: str = None,
    llm: Any = None,
    embedding_model: Any = None,
):
    retrieved_answers = collect_generated_answers(
        source_data=source_data, retriever=retriever, output_dir=output_dir_path
    )
    # TODO: add evaluation function
    run_evaluation_on_generated_answers(
        generated_data=retrieved_answers,
        llm=llm,
        embedding_model=embedding_model,
        output_dir=output_dir_path,
    )
    print("\n\nEvaluation with LLM completed successfully!")


if __name__ == "__main__":
    # required initializations
    samples_start = 0
    # samples_end = 700
    asq_reader = BioASQDataReader(samples_start=samples_start)
    asq_data_file_path = os.path.join(ConfigPath.RAW_DATA_DIR, "bioasq_test.parquet")
    data = asq_reader.read_parquet_file(file_path=asq_data_file_path)
    # llm = ChatModel(
    #     provider="google", model_name="gemini-2.0-flash-lite"
    # ).initialize_model()
    embedding_model = EmbeddingModel()
    neo4j_connection = Neo4jConnection(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
    )
    # graph_crud = GraphCrud(neo4j_connection=neo4j_connection)

    #############################################
    # GRAPH PREPARATION AND LOADING
    #############################################
    # 1 step: construct the graph dataset
    # construct_graph_dataset(asq_reader=asq_reader, file_name="bioasq_test.parquet")
    # 2 step: load the dataset to Neo4j db
    # load_graph_data(embedding_model=embedding_model, graph_crud=graph_crud)

    #############################################
    # RETRIEVAL EXECUTION AND EVALUATION
    #############################################

    # initialize retriever

    # gnn retriever

    # import torch

    # from graph_embeddings.query_proj_model_with_domain_classifier import (
    #     QueryProjectionEncoderModel,
    # )
    # from retrieval_techniques.gnn_retriever import GraphEmbeddingSimilarityRetriever

    # proj_model = QueryProjectionEncoderModel(dim_sem=768, dim_graph=768)
    # model_path = os.path.join(
    #     ConfigPath.MODELS_DIR, "proj_model_da_20250516_230821", "best_query_proj.pt"
    # )
    # proj_model.load_state_dict(torch.load(model_path))
    # proj_model.eval()
    # retriever = GraphEmbeddingSimilarityRetriever(
    #     embedding_model=embedding_model,
    #     neo4j_driver=neo4j_connection.get_driver(),
    #     projection_model=proj_model,
    #     device="cpu",
    # )

    retriever = BaselineBERTSimilarityRetriever(
        embedding_model=embedding_model,
        neo4j_driver=neo4j_connection.get_driver(),
    )

    # # Evaluate similarity search retriever without LLM
    # func_args = {
    #     "retrieval_type": "1_hop_similar_contexts",
    #     "k": 10,
    #     "n_similar_contexts": 10,
    # }
    retriever_args = {"top_k": 10}

    # create results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = os.path.join(
        ConfigPath.RESULTS_DIR, f"{retriever.name}_{timestamp}"
    )
    os.makedirs(output_dir_path, exist_ok=True)

    evaluate_retriever_without_llm(
        source_data=data,
        retriever=retriever,
        retriever_args=retriever_args,
        output_dir_path=output_dir_path,
        k_eval_values=[1, 3, 5, 10],
    )

    # Evaluate similarity search retriever with LLM
    # evaluate_retriever_with_llm(
    #     source_data=data,
    #     retriever=vector_tool_with_answers,
    #     output_dir_path=output_dir_path,
    # )
