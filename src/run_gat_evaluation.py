
"""
Dedicated script to evaluate GAT projection model performance on test data.

This script:
1. Loads the latest trained GAT model
2. Sets up GAT-based retriever
3. Evaluates retrieval performance on BioASQ test data
4. Saves results and metrics to output directory

Usage:
    python run_gat_evaluation.py
"""

import os
import glob
import torch
from datetime import datetime

# Import required modules
from configs import ConfigEnv, ConfigPath
from data_collection.reader import BioASQDataReader
from evaluation.executor import run_retrieval
from evaluation.non_llm_based_eval import run_evaluation
from knowledge_graph.connection import Neo4jConnection
from llms.embedding_model import EmbeddingModel
from projection_models.projection_gat_model import load_gat_model, GATDataProcessor
from retrieval_techniques.gnn_retriever import GraphEmbeddingSimilarityRetriever
from utils.utils import save_json_file


def find_latest_gat_model():
    """Find the most recently trained GAT model."""
    gat_model_pattern = os.path.join(ConfigPath.MODELS_DIR, "gat_proj_model_*")
    gat_model_dirs = glob.glob(gat_model_pattern)
    
    if not gat_model_dirs:
        raise ValueError(f"No GAT model directories found matching pattern: {gat_model_pattern}")
    
    # Get the most recent model directory
    latest_gat_dir = max(gat_model_dirs, key=os.path.getctime)
    best_model_path = os.path.join(latest_gat_dir, "best_gat_model.pt")
    
    if not os.path.exists(best_model_path):
        raise ValueError(f"Best GAT model not found at: {best_model_path}")
    
    return latest_gat_dir, best_model_path


def load_test_data(samples_start=0, samples_end=None):
    """Load BioASQ test data."""
    asq_reader = BioASQDataReader(samples_start=samples_start)
    asq_data_file_path = os.path.join(ConfigPath.RAW_DATA_DIR, "bioasq_test.parquet")
    data = asq_reader.read_parquet_file(file_path=asq_data_file_path)
    
    if samples_end:
        data = data[:samples_end]
    
    return data


def setup_gat_retriever(gat_model_path, device="auto"):
    """Set up GAT-based retriever with all required components."""
    
    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"💻 Using device: {device}")
    
    # Initialize required components
    embedding_model = EmbeddingModel()
    neo4j_connection = Neo4jConnection(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
    )
    
    # Load GAT model
    print(f"🔍 Loading GAT model from: {gat_model_path}")
    try:
        gat_model = load_gat_model(gat_model_path, device=device)
        print(f"✅ GAT model loaded successfully")
        
        # Print model info
        if hasattr(gat_model, 'input_dim'):
            print(f"🏗️ Model Architecture:")
            print(f"   Input dimension: {gat_model.input_dim}")
            print(f"   Hidden dimension: {gat_model.hidden_dim}")
            print(f"   Output dimension: {gat_model.output_dim}")
            print(f"   GAT layers: {gat_model.n_layers}")
            total_params = sum(p.numel() for p in gat_model.parameters())
            print(f"   Total parameters: {total_params:,}")
            
    except Exception as e:
        print(f"❌ Failed to load GAT model: {e}")
        raise
    
    # Initialize GAT data processor
    gat_data_processor = GATDataProcessor(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB
    )
    
    # Create GAT-based retriever
    retriever = GraphEmbeddingSimilarityRetriever(
        embedding_model=embedding_model,
        neo4j_driver=neo4j_connection.get_driver(),
        projection_model=gat_model,
        device=device,
        gat_data_processor=gat_data_processor,
        use_gat_projection=False,
        top_k_contexts=10,
    )
    
    return retriever


def test_retrieval(retriever, test_data):
    """Test retrieval with a sample query."""
    print(f"\n🧪 Testing GAT retrieval with sample query...")
    try:
        sample_query = test_data[0]["question"]
        print(f"📝 Sample query: {sample_query[:100]}...")
        
        test_results = retriever.retrieve(query=sample_query, top_k=3)
        print(f"✅ Test retrieval successful - got {len(test_results)} results")
        
        if test_results:
            print(f"🎯 Top result score: {test_results[0]['score']:.4f}")
            print(f"📄 Top result PMID: {test_results[0]['pmid']}")
            return True
    except Exception as e:
        print(f"❌ Test retrieval failed: {e}")
        return False


def run_gat_evaluation(
    max_samples=None,
    top_k_retrieval=10,
    k_eval_values=[1,3, 5, 10],
    device="auto"
):
    """
    Main function to run GAT model evaluation.
    
    Args:
        max_samples: Maximum number of test samples to evaluate (None for all)
        top_k_retrieval: Number of contexts to retrieve per query
        k_eval_values: K values for recall@k evaluation
        device: Device to use ("auto", "cuda", or "cpu")
    """
    
    print("🚀 Starting GAT Model Evaluation")
    print("=" * 50)
    
    # 1. Find latest GAT model
    latest_gat_dir, best_model_path = find_latest_gat_model()
    model_name = os.path.basename(latest_gat_dir)
    print(f"📁 Using GAT model: {model_name}")
    
    # 2. Load test data
    test_data = load_test_data(samples_end=max_samples)
    print(f"📊 Loaded {len(test_data)} test samples")
    
    # 3. Setup retriever
    retriever = setup_gat_retriever(best_model_path, device=device)
    
    # 4. Get retriever stats
    retriever_stats = retriever.get_retrieval_stats()
    print(f"\n📊 Retriever Configuration:")
    for key, value in retriever_stats.items():
        print(f"   {key}: {value}")
    
    # 5. Test retrieval
    test_success = test_retrieval(retriever, test_data)
    if not test_success:
        print("⚠️ Test retrieval failed, but continuing with evaluation...")
    
    # 6. Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = os.path.join(
        ConfigPath.RESULTS_DIR, f"GAT_Eval_{model_name}_{timestamp}"
    )
    os.makedirs(output_dir_path, exist_ok=True)
    
    print(f"\n🔍 Starting Full Evaluation")
    print(f"📁 Results will be saved to: {output_dir_path}")
    print(f"🎯 Evaluation metrics: Recall@{k_eval_values}")
    print(f"📊 Retrieval top-k: {top_k_retrieval}")
    print(f"📝 Test data samples: {len(test_data)}")
    
    # 7. Save evaluation configuration
    evaluation_info = {
        "gat_model_directory": latest_gat_dir,
        "gat_model_file": best_model_path,
        "retriever_stats": retriever_stats,
        "evaluation_config": {
            "top_k_retrieval": top_k_retrieval,
            "k_eval_values": k_eval_values,
            "test_samples": len(test_data),
            "max_samples": max_samples,
            "device": device
        },
        "test_retrieval_success": test_success
    }
    save_json_file(
        file_path=os.path.join(output_dir_path, "gat_evaluation_config.json"),
        data=evaluation_info
    )
    
    # 8. Run retrieval
    retriever_args = {"top_k": top_k_retrieval}
    save_json_file(
        file_path=os.path.join(output_dir_path, "retriever_args.json"),
        data=retriever_args,
    )
    
    print(f"\n⏳ Running retrieval on {len(test_data)} samples...")
    retrieval_results = run_retrieval(
        source_data=test_data,
        retriever=retriever,
        retriever_args=retriever_args,
        output_dir=output_dir_path,
    )
    
    # 9. Run evaluation
    print(f"\n📈 Computing evaluation metrics...")
    metrics, detailed_results = run_evaluation(
        retrieval_results=retrieval_results,
        k_values=k_eval_values,
        output_dir=output_dir_path,
        retriever_name=retriever.name,
    )
    
    # Debug: Print metrics structure
    print(f"📋 Computed {len(metrics)} metrics:")
    for key in metrics.keys():
        print(f"   - {key}: {type(metrics[key])}")
    
    # 10. Print results
    print(f"\n🎉 GAT Model Evaluation Results:")
    print("=" * 40)
    for metric, value in metrics.items():
        if isinstance(value, dict):
            print(f"📊 {metric}:")
            for sub_metric, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    print(f"   {sub_metric}: {sub_value:.4f}")
                else:
                    print(f"   {sub_metric}: {sub_value}")
        elif isinstance(value, (int, float)):
            print(f"📊 {metric}: {value:.4f}")
        else:
            print(f"📊 {metric}: {value}")
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📁 All results saved in: {output_dir_path}")
    print(f"📋 Key files:")
    print(f"   - gat_evaluation_config.json: Evaluation configuration")
    print(f"   - {retriever.name}_retrieval_results.csv: Raw retrieval results")
    print(f"   - evaluation_metrics.json: Performance metrics")
    
    return metrics, output_dir_path


if __name__ == "__main__":
    # Configuration
    MAX_SAMPLES = None  # Set to None for full evaluation, or number for subset (using 10 for quick test)
    TOP_K_RETRIEVAL = 10
    K_EVAL_VALUES = [1, 5, 10]
    DEVICE = "auto"  # "auto", "cuda", or "cpu"
    
    print(f"🔧 Configuration:")
    print(f"   Max samples: {MAX_SAMPLES}")
    print(f"   Top-k retrieval: {TOP_K_RETRIEVAL}")
    print(f"   Evaluation K values: {K_EVAL_VALUES}")
    print(f"   Device: {DEVICE}")
    
    try:
        metrics, results_dir = run_gat_evaluation(
            max_samples=MAX_SAMPLES,
            top_k_retrieval=TOP_K_RETRIEVAL,
            k_eval_values=K_EVAL_VALUES,
            device=DEVICE
        )
        
        print(f"\n🎊 Evaluation Summary:")
        
        # Find best numeric metric
        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric_metrics:
            best_metric = max(numeric_metrics, key=numeric_metrics.get)
            best_value = numeric_metrics[best_metric]
            print(f"📈 Best metric: {best_metric} = {best_value:.4f}")
        else:
            print(f"📈 Metrics: {len(metrics)} evaluation results computed")
        
        print(f"📁 Results directory: {results_dir}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        raise 