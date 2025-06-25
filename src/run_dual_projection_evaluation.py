"""
Dedicated script to evaluate Dual Projection model performance on test data.

This script:
1. Loads the latest trained Dual Projection model
2. Sets up Dual Projection-based retriever
3. Evaluates retrieval performance on BioASQ test data
4. Saves results and metrics to output directory

Usage:
    python run_dual_projection_evaluation.py
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
from projection_models.dual_projection_model import load_dual_projection_model
from retrieval_techniques.dual_projection_retriever import DualProjectionRetriever, create_dual_projection_retriever
from utils.utils import save_json_file


def find_latest_dual_projection_model():
    """Find the most recently trained Dual Projection model."""
    # First try the specific directory mentioned by user
    specific_model_dir = os.path.join(ConfigPath.MODELS_DIR, "dual_proj_neo4j_20250614_214744")
    specific_model_path = os.path.join(specific_model_dir, "best_dual_projection_model.pt")
    
    if os.path.exists(specific_model_path):
        print(f"✅ Found specific dual projection model: {specific_model_dir}")
        return specific_model_dir, specific_model_path
    
    # Fall back to finding latest model
    dual_proj_model_pattern = os.path.join(ConfigPath.MODELS_DIR, "dual_proj_model_*")
    dual_proj_model_dirs = glob.glob(dual_proj_model_pattern)
    
    if not dual_proj_model_dirs:
        raise ValueError(f"No Dual Projection model directories found matching pattern: {dual_proj_model_pattern}")
    
    # Get the most recent model directory
    latest_dual_proj_dir = max(dual_proj_model_dirs, key=os.path.getctime)
    best_model_path = os.path.join(latest_dual_proj_dir, "best_dual_projection_model.pt")
    
    if not os.path.exists(best_model_path):
        raise ValueError(f"Best Dual Projection model not found at: {best_model_path}")
    
    return latest_dual_proj_dir, best_model_path


def load_test_data(samples_start=0, samples_end=None):
    """Load BioASQ test data."""
    asq_reader = BioASQDataReader(samples_start=samples_start)
    asq_data_file_path = os.path.join(ConfigPath.RAW_DATA_DIR, "bioasq_test.parquet")
    data = asq_reader.read_parquet_file(file_path=asq_data_file_path)
    
    if samples_end:
        data = data[:samples_end]
    
    return data


def setup_dual_projection_retriever(
    dual_proj_model_path, 
    device="auto",
    semantic_weight=0.5,
    graph_weight=0.5,
    combination_strategy="weighted_sum"
):
    """Set up Dual Projection-based retriever with all required components."""
    
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
    
    # Load Dual Projection model
    print(f"🔍 Loading Dual Projection model from: {dual_proj_model_path}")
    try:
        dual_proj_model, checkpoint = load_dual_projection_model(dual_proj_model_path, device=device)
        print(f"✅ Dual Projection model loaded successfully")
        
        # Print model info
        print(f"🏗️ Model Architecture:")
        print(f"   Semantic dimension: {dual_proj_model.dim_sem}")
        print(f"   Graph dimension: {dual_proj_model.dim_graph}")
        print(f"   Hidden dimensions: {dual_proj_model.hidden_dims}")
        print(f"   Dropout: {dual_proj_model.p_dropout}")
        total_params = sum(p.numel() for p in dual_proj_model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        # Print training info if available
        if 'training_config' in checkpoint:
            training_config = checkpoint['training_config']
            print(f"🎯 Training Configuration:")
            print(f"   Semantic loss weight: {training_config.get('semantic_loss_weight', 'N/A')}")
            print(f"   Graph loss weight: {training_config.get('graph_loss_weight', 'N/A')}")
            print(f"   Temperature: {training_config.get('temperature', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Failed to load Dual Projection model: {e}")
        raise
    
    # Create Dual Projection-based retriever
    retriever = DualProjectionRetriever(
        embedding_model=embedding_model,
        neo4j_driver=neo4j_connection.get_driver(),
        dual_projection_model=dual_proj_model,
        device=device,
        semantic_weight=semantic_weight,
        graph_weight=graph_weight,
        combination_strategy=combination_strategy
    )
    
    return retriever


def test_retrieval(retriever, test_data):
    """Test retrieval with a sample query."""
    print(f"\n🧪 Testing Dual Projection retrieval with sample query...")
    try:
        sample_query = test_data[0]["question"]
        print(f"📝 Sample query: {sample_query[:100]}...")
        
        # Test projection functionality first
        projection_test = retriever.test_projection(sample_query)
        if "error" in projection_test:
            print(f"❌ Projection test failed: {projection_test['error']}")
            return False
        
        # Test actual retrieval
        test_results = retriever.retrieve(query=sample_query, top_k=3)
        print(f"✅ Test retrieval successful - got {len(test_results)} results")
        
        if test_results:
            top_result = test_results[0]
            print(f"🎯 Top result combined score: {top_result['score']:.4f}")
            print(f"   Semantic score: {top_result.get('semantic_score', 'N/A')}")
            print(f"   Graph score: {top_result.get('graph_score', 'N/A')}")
            print(f"📄 Top result PMID: {top_result['pmid']}")
            return True
        else:
            print("⚠️ No results returned from test retrieval")
            return False
    except Exception as e:
        print(f"❌ Test retrieval failed: {e}")
        return False


def run_dual_projection_evaluation(
    max_samples=None,
    top_k_retrieval=10,
    k_eval_values=[1, 5, 10],
    device="auto",
    semantic_weight=0.5,
    graph_weight=0.5,
    combination_strategy="weighted_sum"
):
    """
    Main function to run Dual Projection model evaluation.
    
    Args:
        max_samples: Maximum number of test samples to evaluate (None for all)
        top_k_retrieval: Number of contexts to retrieve per query
        k_eval_values: K values for recall@k evaluation
        device: Device to use ("auto", "cuda", or "cpu")
        semantic_weight: Weight for semantic space in combination
        graph_weight: Weight for graph space in combination
        combination_strategy: Strategy for combining dual space results
    """
    
    print("🚀 Starting Dual Projection Model Evaluation")
    print("=" * 50)
    
    # 1. Find latest Dual Projection model
    latest_dual_proj_dir, best_model_path = find_latest_dual_projection_model()
    model_name = os.path.basename(latest_dual_proj_dir)
    print(f"📁 Using Dual Projection model: {model_name}")
    
    # 2. Load test data
    test_data = load_test_data(samples_end=max_samples)
    print(f"📊 Loaded {len(test_data)} test samples")
    
    # 3. Setup retriever
    retriever = setup_dual_projection_retriever(
        dual_proj_model_path=best_model_path, 
        device=device,
        semantic_weight=semantic_weight,
        graph_weight=graph_weight,
        combination_strategy=combination_strategy
    )
    
    # 4. Get retriever stats
    retriever_stats = retriever.get_retrieval_stats()
    print(f"\n📊 Retriever Configuration:")
    for key, value in retriever_stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"      {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    # 5. Test retrieval
    test_success = test_retrieval(retriever, test_data)
    if not test_success:
        print("⚠️ Test retrieval failed, but continuing with evaluation...")
    
    # 6. Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_suffix = f"_{combination_strategy}" if combination_strategy != "weighted_sum" else ""
    weights_suffix = f"_s{semantic_weight}_g{graph_weight}" if (semantic_weight != 0.5 or graph_weight != 0.5) else ""
    output_dir_path = os.path.join(
        ConfigPath.RESULTS_DIR, f"DualProj_Eval_{model_name}{strategy_suffix}{weights_suffix}_{timestamp}"
    )
    os.makedirs(output_dir_path, exist_ok=True)
    
    print(f"\n🔍 Starting Full Evaluation")
    print(f"📁 Results will be saved to: {output_dir_path}")
    print(f"🎯 Evaluation metrics: Recall@{k_eval_values}")
    print(f"📊 Retrieval top-k: {top_k_retrieval}")
    print(f"📝 Test data samples: {len(test_data)}")
    print(f"⚖️ Combination strategy: {combination_strategy}")
    print(f"🔀 Weights: Semantic={semantic_weight}, Graph={graph_weight}")
    
    # 7. Save evaluation configuration
    evaluation_info = {
        "dual_projection_model_directory": latest_dual_proj_dir,
        "dual_projection_model_file": best_model_path,
        "retriever_stats": retriever_stats,
        "evaluation_config": {
            "top_k_retrieval": top_k_retrieval,
            "k_eval_values": k_eval_values,
            "test_samples": len(test_data),
            "max_samples": max_samples,
            "device": device,
            "semantic_weight": semantic_weight,
            "graph_weight": graph_weight,
            "combination_strategy": combination_strategy
        },
        "test_retrieval_success": test_success
    }
    save_json_file(
        file_path=os.path.join(output_dir_path, "dual_projection_evaluation_config.json"),
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
    print(f"\n🎉 Dual Projection Model Evaluation Results:")
    print("=" * 50)
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
    
    # 11. Additional analysis
    print(f"\n📊 Detailed Analysis:")
    if retrieval_results:
        # Analyze dual-space performance
        semantic_scores = []
        graph_scores = []
        combined_scores = []
        
        for result in retrieval_results:
            if 'retrieved_scores' in result and result['retrieved_scores']:
                # Note: The combined score is what's returned as 'score' by our retriever
                combined_scores.extend(result['retrieved_scores'])
        
        if combined_scores:
            avg_combined_score = sum(combined_scores) / len(combined_scores)
            max_combined_score = max(combined_scores)
            min_combined_score = min(combined_scores)
            
            print(f"   Average combined score: {avg_combined_score:.4f}")
            print(f"   Max combined score: {max_combined_score:.4f}")
            print(f"   Min combined score: {min_combined_score:.4f}")
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📁 All results saved in: {output_dir_path}")
    print(f"📋 Key files:")
    print(f"   - dual_projection_evaluation_config.json: Evaluation configuration")
    print(f"   - {retriever.name}_retrieval_results.csv: Raw retrieval results")
    print(f"   - evaluation_metrics.json: Performance metrics")
    
    return metrics, output_dir_path


def run_comparative_evaluation(
    max_samples=None,
    top_k_retrieval=10,
    k_eval_values=[1, 5, 10],
    device="auto"
):
    """
    Run evaluation with different combination strategies for comparison.
    
    Args:
        max_samples: Maximum number of test samples to evaluate
        top_k_retrieval: Number of contexts to retrieve per query
        k_eval_values: K values for evaluation metrics
        device: Device to use
    """
    
    print("🚀 Starting Comparative Dual Projection Evaluation")
    print("=" * 60)
    
    # Different strategies to compare
    strategies = {
        "weighted_sum": {"semantic_weight": 0.5, "graph_weight": 0.5},
        "semantic_only": {"semantic_weight": 1.0, "graph_weight": 0.0},
        "graph_only": {"semantic_weight": 0.0, "graph_weight": 1.0},
        "semantic_heavy": {"semantic_weight": 0.7, "graph_weight": 0.3},
        "graph_heavy": {"semantic_weight": 0.3, "graph_weight": 0.7},
        "max": {"semantic_weight": 0.5, "graph_weight": 0.5}  # max strategy ignores weights
    }
    
    all_results = {}
    
    for strategy_name, weights in strategies.items():
        print(f"\n{'='*20} {strategy_name.upper()} {'='*20}")
        
        combination_strategy = strategy_name if strategy_name in ["weighted_sum", "max", "semantic_only", "graph_only"] else "weighted_sum"
        
        try:
            metrics, output_dir = run_dual_projection_evaluation(
                max_samples=max_samples,
                top_k_retrieval=top_k_retrieval,
                k_eval_values=k_eval_values,
                device=device,
                semantic_weight=weights["semantic_weight"],
                graph_weight=weights["graph_weight"],
                combination_strategy=combination_strategy
            )
            
            all_results[strategy_name] = {
                "metrics": metrics,
                "output_dir": output_dir,
                "config": {
                    "semantic_weight": weights["semantic_weight"],
                    "graph_weight": weights["graph_weight"],
                    "combination_strategy": combination_strategy
                }
            }
            
        except Exception as e:
            print(f"❌ Strategy {strategy_name} failed: {e}")
            all_results[strategy_name] = {"error": str(e)}
    
    # Save comparative results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparative_output_dir = os.path.join(
        ConfigPath.RESULTS_DIR, f"DualProj_Comparative_{timestamp}"
    )
    os.makedirs(comparative_output_dir, exist_ok=True)
    
    save_json_file(
        file_path=os.path.join(comparative_output_dir, "comparative_results.json"),
        data=all_results
    )
    
    # Print comparison summary
    print(f"\n🏆 COMPARATIVE RESULTS SUMMARY")
    print("=" * 60)
    
    # Compare precision@10 across strategies
    precision_10_results = {}
    for strategy_name, result in all_results.items():
        if "error" not in result and "metrics" in result:
            metrics = result["metrics"]
            if 10 in metrics and "precision" in metrics[10]:
                precision_10_results[strategy_name] = metrics[10]["precision"]
    
    if precision_10_results:
        print(f"📊 Precision@10 Comparison:")
        sorted_strategies = sorted(precision_10_results.items(), key=lambda x: x[1], reverse=True)
        for i, (strategy, precision) in enumerate(sorted_strategies, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📍"
            print(f"   {emoji} {strategy}: {precision:.4f}")
    
    print(f"\n📁 Comparative results saved in: {comparative_output_dir}")
    
    return all_results


if __name__ == "__main__":
    # Configuration
    MAX_SAMPLES = None  # Set to None for full evaluation, or number for subset
    TOP_K_RETRIEVAL = 10
    K_EVAL_VALUES = [1, 5, 10]
    DEVICE = "auto"  # "auto", "cuda", or "cpu"
    
    # Single evaluation configuration
    SEMANTIC_WEIGHT = 0.5
    GRAPH_WEIGHT = 0.5
    COMBINATION_STRATEGY = "weighted_sum"  # "weighted_sum", "max", "semantic_only", "graph_only"
    
    # Set to True to run comparative evaluation with multiple strategies
    RUN_COMPARATIVE = False
    
    print(f"🔧 Configuration:")
    print(f"   Max samples: {MAX_SAMPLES}")
    print(f"   Top-k retrieval: {TOP_K_RETRIEVAL}")
    print(f"   Evaluation K values: {K_EVAL_VALUES}")
    print(f"   Device: {DEVICE}")
    if not RUN_COMPARATIVE:
        print(f"   Semantic weight: {SEMANTIC_WEIGHT}")
        print(f"   Graph weight: {GRAPH_WEIGHT}")
        print(f"   Combination strategy: {COMBINATION_STRATEGY}")
    print(f"   Run comparative: {RUN_COMPARATIVE}")
    
    try:
        if RUN_COMPARATIVE:
            # Run comparative evaluation
            all_results = run_comparative_evaluation(
                max_samples=MAX_SAMPLES,
                top_k_retrieval=TOP_K_RETRIEVAL,
                k_eval_values=K_EVAL_VALUES,
                device=DEVICE
            )
            
            print(f"\n🎊 Comparative Evaluation Complete!")
            successful_runs = sum(1 for r in all_results.values() if "error" not in r)
            print(f"📈 Successful strategy evaluations: {successful_runs}/{len(all_results)}")
            
        else:
            # Run single evaluation
            metrics, results_dir = run_dual_projection_evaluation(
                max_samples=MAX_SAMPLES,
                top_k_retrieval=TOP_K_RETRIEVAL,
                k_eval_values=K_EVAL_VALUES,
                device=DEVICE,
                semantic_weight=SEMANTIC_WEIGHT,
                graph_weight=GRAPH_WEIGHT,
                combination_strategy=COMBINATION_STRATEGY
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