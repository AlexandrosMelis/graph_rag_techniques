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
from retrieval_techniques.non_ml_retrievers import BaselineBERTSimilarityRetriever
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


# def evaluate_retriever_with_llm(
#     source_data: list,
#     retriever: Any,
#     output_dir_path: str = None,
#     llm: Any = None,
#     embedding_model: Any = None,
# ):
#     retrieved_answers = collect_generated_answers(
#         source_data=source_data, retriever=retriever, output_dir=output_dir_path
#     )
#     # TODO: add evaluation function
#     run_evaluation_on_generated_answers(
#         generated_data=retrieved_answers,
#         llm=llm,
#         embedding_model=embedding_model,
#         output_dir=output_dir_path,
#     )
#     print("\n\nEvaluation with LLM completed successfully!")


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

    # GAT-based retriever

    # import torch
    # import glob

    # from projection_models.projection_gat_model import load_gat_model, GATDataProcessor
    # from retrieval_techniques.gnn_retriever import GraphEmbeddingSimilarityRetriever

    # # Find the latest GAT model directory
    # gat_model_pattern = os.path.join(ConfigPath.MODELS_DIR, "gat_proj_model_*")
    # gat_model_dirs = glob.glob(gat_model_pattern)
    
    # if not gat_model_dirs:
    #     raise ValueError(f"No GAT model directories found matching pattern: {gat_model_pattern}")
    
    # # Get the most recent model directory
    # latest_gat_dir = max(gat_model_dirs, key=os.path.getctime)
    # best_model_path = os.path.join(latest_gat_dir, "best_gat_model.pt")
    
    # if not os.path.exists(best_model_path):
    #     raise ValueError(f"Best GAT model not found at: {best_model_path}")
    
    # print(f"🔍 Loading GAT model from: {latest_gat_dir}")
    # print(f"📁 Model file: {best_model_path}")
    
    # # Load GAT projection model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"💻 Using device: {device}")
    
    # try:
    #     gat_model = load_gat_model(best_model_path, device=device)
    #     print(f"✅ GAT model loaded successfully")
        
    #     # Print model architecture info
    #     if hasattr(gat_model, 'input_dim'):
    #         print(f"🏗️ Model Architecture:")
    #         print(f"   Input dimension: {gat_model.input_dim}")
    #         print(f"   Hidden dimension: {gat_model.hidden_dim}")
    #         print(f"   Output dimension: {gat_model.output_dim}")
    #         print(f"   GAT layers: {gat_model.n_layers}")
    #         total_params = sum(p.numel() for p in gat_model.parameters())
    #         print(f"   Total parameters: {total_params:,}")
            
    # except Exception as e:
    #     print(f"❌ Failed to load GAT model: {e}")
    #     raise
    
    # Initialize GAT data processor for subgraph construction
    # gat_data_processor = GATDataProcessor(
    #     uri=ConfigEnv.NEO4J_URI,
    #     user=ConfigEnv.NEO4J_USER,
    #     password=ConfigEnv.NEO4J_PASSWORD,
    #     database=ConfigEnv.NEO4J_DB
    # )
    
    # Create GAT-based retriever
    # retriever = GraphEmbeddingSimilarityRetriever(
    #     embedding_model=embedding_model,
    #     neo4j_driver=neo4j_connection.get_driver(),
    #     projection_model=gat_model,
    #     device=device,
    #     gat_data_processor=gat_data_processor,  # Add GAT-specific processor
    #     use_gat_projection=True  # Flag to indicate GAT-based projection
    # )

    retriever = BaselineBERTSimilarityRetriever(
        embedding_model=embedding_model,
        neo4j_driver=neo4j_connection.get_driver(),
    )

    # # Evaluate similarity search retriever without LLM
   
    # Get retriever statistics and save them
    # retriever_stats = retriever.get_retrieval_stats()
    # print(f"\n📊 Retriever Configuration:")
    # for key, value in retriever_stats.items():
    #     print(f"   {key}: {value}")
    
    # Test GAT retrieval with a sample query
    print(f"\n🧪 Testing GAT retrieval with sample query...")
    try:
        sample_query = data[0]["question"]
        print(f"📝 Sample query: {sample_query[:100]}...")
        
        test_results = retriever.retrieve(query=sample_query, top_k=3)
        print(f"✅ Test retrieval successful - got {len(test_results)} results")
        
        if test_results:
            print(f"🎯 Top result score: {test_results[0]['score']:.4f}")
            print(f"📄 Top result PMID: {test_results[0]['pmid']}")
    except Exception as e:
        print(f"❌ Test retrieval failed: {e}")
        print("⚠️ Continuing with evaluation anyway...")
    
    # Evaluation configuration
    retriever_args = {"top_k": 10}
    k_eval_values = [1, 5, 10]
    
    # Create results folder with GAT model info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_name = os.path.basename(latest_gat_dir)
    output_dir_path = os.path.join(
        ConfigPath.RESULTS_DIR, f"BERT_Similarity_Search_{timestamp}"
    )
    os.makedirs(output_dir_path, exist_ok=True)
    
    # print(f"\n🔍 Starting GAT Model Evaluation")
    # print(f"📁 Results will be saved to: {output_dir_path}")
    # print(f"🎯 Evaluation metrics: Recall@{k_eval_values}")
    # print(f"📊 Retrieval top-k: {retriever_args['top_k']}")
    # print(f"📝 Test data samples: {len(data)}")
    
    # # Save GAT model info to results
    # model_info = {
    #     "gat_model_directory": latest_gat_dir,
    #     "gat_model_file": best_model_path,
    #     "retriever_stats": retriever_stats,
    #     "evaluation_config": {
    #         "top_k": retriever_args["top_k"],
    #         "k_eval_values": k_eval_values,
    #         "test_samples": len(data)
    #     }
    # }
    # save_json_file(
    #     file_path=os.path.join(output_dir_path, "gat_model_info.json"),
    #     data=model_info
    # )

    evaluate_retriever_without_llm(
        source_data=data,
        retriever=retriever,
        retriever_args=retriever_args,
        output_dir_path=output_dir_path,
        k_eval_values=k_eval_values,
    )

    # Evaluate similarity search retriever with LLM
    # evaluate_retriever_with_llm(
    #     source_data=data,
    #     retriever=vector_tool_with_answers,
    #     output_dir_path=output_dir_path,
    # )
