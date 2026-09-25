import os
from datetime import datetime
from typing import Dict, List, Any

from graph_rag.config import ConfigEnv, ConfigPath
from graph_rag.data.bioasq import BioASQDataReader
from graph_rag.evaluation.executor import run_retrieval
from graph_rag.evaluation.metrics import run_evaluation
from graph_rag.graph.connection import Neo4jConnection
from graph_rag.llm.embeddings import EmbeddingModel
from graph_rag.retrieval.non_ml import (
    BaselineBERTSimilarityRetriever,
    ExpandNHopsSimilarityRetriever,
    MeshSubgraphSimilarityRetriever,
    RETRIEVER_REGISTRY
)
from graph_rag.retrieval.pagerank import PersonalizedPageRankRetriever
from graph_rag.utils import save_json_file


def evaluate_single_retriever(
    source_data: List[Dict[str, Any]],
    retriever_class,
    retriever_name: str,
    retriever_init_args: Dict[str, Any],
    retriever_args: Dict[str, Any],
    embedding_model: Any,
    neo4j_driver: Any,
    output_base_dir: str,
    k_eval_values: List[int] = [1, 3, 5, 10],
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate a single retriever and save results.
    
    Args:
        source_data: Test data for evaluation
        retriever_class: Class of the retriever to instantiate
        retriever_name: Name for output files
        retriever_init_args: Arguments to pass to retriever initialization
        retriever_args: Arguments to pass to retrieve method
        embedding_model: Embedding model instance
        neo4j_driver: Neo4j driver instance
        output_base_dir: Base directory for outputs
        k_eval_values: K values for evaluation metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"🔍 Evaluating: {retriever_name}")
    print(f"{'='*60}")
    
    # Create output directory for this retriever
    safe_name = retriever_name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"{safe_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize retriever
    retriever = retriever_class(
        embedding_model=embedding_model,
        neo4j_driver=neo4j_driver,
        **retriever_init_args
    )
    
    print(f"📊 Retriever initialized: {retriever.name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️ Init args: {retriever_init_args}")
    print(f"⚙️ Retrieve args: {retriever_args}")
    print(f"📝 Test samples: {len(source_data)}")
    print(f"🎯 K values for evaluation: {k_eval_values}")
    
    try:
        # Test retrieval with a sample query
        if source_data:
            sample_query = source_data[0]["question"]
            print(f"\n🧪 Testing with sample query: {sample_query[:100]}...")
            test_results = retriever.retrieve(query=sample_query, **retriever_args)
            print(f"✅ Test successful - got {len(test_results)} results")
            if test_results:
                print(f"🎯 Top result score: {test_results[0]['score']:.4f}")
        
        # Save retriever arguments
        all_args = {"init_args": retriever_init_args, "retrieve_args": retriever_args}
        save_json_file(
            file_path=os.path.join(output_dir, "retriever_args.json"),
            data=all_args,
        )
        
        # Run retrieval on all test data
        print(f"\n🚀 Starting retrieval on {len(source_data)} samples...")
        retrieval_results = run_retrieval(
            source_data=source_data,
            retriever=retriever,
            retriever_args=retriever_args,
            output_dir=output_dir,
        )
        
        print(f"✅ Retrieval completed - {len(retrieval_results)} results collected")
        
        # Run evaluation
        print(f"📊 Running evaluation with k_values: {k_eval_values}")
        metrics, evaluator = run_evaluation(
            retrieval_results=retrieval_results,
            k_values=k_eval_values,
            output_dir=output_dir,
            retriever_name=retriever.name,
        )
        
        print(f"✅ Evaluation completed for {retriever_name}")
        print(f"📈 Results saved to: {output_dir}")
        
        # Print summary metrics
        print(f"\n📊 Summary Metrics for {retriever_name}:")
        if metrics:
            for k, k_metrics in metrics.items():
                print(f"  📍 k={k}:")
                for metric_name, value in k_metrics.items():
                    print(f"    {metric_name}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error evaluating {retriever_name}: {e}")
        import traceback
        traceback.print_exc()
        print(f"📁 Check output directory: {output_dir}")
        return {}


def compare_retrievers(
    all_metrics: Dict[str, Dict[int, Dict[str, float]]],
    output_dir: str,
    k_values: List[int] = [1, 3, 5, 10]
) -> None:
    """
    Compare all retrievers and generate comparison report.
    
    Args:
        all_metrics: Dictionary mapping retriever names to their metrics
        output_dir: Directory to save comparison results
        k_values: K values to include in comparison
    """
    print(f"\n{'='*60}")
    print("📊 RETRIEVER COMPARISON REPORT")
    print(f"{'='*60}")
    
    # Create comparison tables for each k value
    comparison_data = {}
    
    for k in k_values:
        comparison_data[k] = {}
        print(f"\n📍 Comparison at k={k}:")
        print("-" * 50)
        
        for retriever_name, metrics in all_metrics.items():
            k_metrics = metrics.get(k, {})
            comparison_data[k][retriever_name] = k_metrics
            
            print(f"🔍 {retriever_name}:")
            if k_metrics:
                for metric_name, value in k_metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
            else:
                print("  No metrics available.")
        
        # Find best performer for each metric at this k
        print(f"\n🏆 Best performers at k={k}:")
        if any(comparison_data[k].values()):
            metric_names = set()
            for metrics in comparison_data[k].values():
                metric_names.update(metrics.keys())
            
            for metric in sorted(list(metric_names)):
                best_score = -1.0
                best_retriever = "N/A"
                for retriever_name, metrics in comparison_data[k].items():
                    score = metrics.get(metric, -1.0)
                    if score > best_score:
                        best_score = score
                        best_retriever = retriever_name
                
                print(f"  {metric:<12}: {best_retriever} ({best_score:.4f})")
    
    # Save comparison results
    comparison_file = os.path.join(output_dir, "retriever_comparison_summary.json")
    save_json_file(comparison_file, comparison_data)
    print(f"\n💾 Comparison results saved to: {comparison_file}")


def run_comprehensive_evaluation():
    """
    Run comprehensive evaluation of all non-ML retrievers.
    """
    print("🚀 Starting Comprehensive Non-ML Retriever Evaluation")
    print("=" * 60)
    
    # Initialize components
    print("🔧 Initializing components...")
    
    # Load test data
    asq_reader = BioASQDataReader(samples_start=0)
    asq_data_file_path = os.path.join(ConfigPath.RAW_DATA_DIR, "bioasq_test.parquet")
    data = asq_reader.read_parquet_file(file_path=asq_data_file_path)
    print(f"📚 Loaded {len(data)} test samples")
    
    # Initialize embedding model
    embedding_model = EmbeddingModel()
    print("✅ Embedding model initialized")
    
    # Initialize Neo4j connection
    neo4j_connection = Neo4jConnection(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
    )
    print("✅ Neo4j connection initialized")
    
    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(ConfigPath.RESULTS_DIR, f"non_ml_retrievers_eval_{timestamp}")
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {output_base_dir}")
    
    # Define retrievers and their configurations
    retriever_configs = [
        {
            "class": BaselineBERTSimilarityRetriever,
            "name": "Baseline BERT Similarity",
            "init_args": {},
            "retrieve_args": {"top_k": 10}
        },
        {
            "class": ExpandNHopsSimilarityRetriever,
            "name": "N-Hops Expansion (2 hops)",
            "init_args": {},
            "retrieve_args": {"top_k": 10, "n_hops": 2}
        },
        {
            "class": ExpandNHopsSimilarityRetriever,
            "name": "N-Hops Expansion (3 hops)",
            "init_args": {},
            "retrieve_args": {"top_k": 10, "n_hops": 3}
        },
        {
            "class": MeshSubgraphSimilarityRetriever,
            "name": "MeSH Subgraph (mesh_k=15)",
            "init_args": {"mesh_k": 15},
            "retrieve_args": {"top_k": 10}
        },
        {
            "class": MeshSubgraphSimilarityRetriever,
            "name": "MeSH Subgraph (mesh_k=25)",
            "init_args": {"mesh_k": 25},
            "retrieve_args": {"top_k": 10}
        },
        {
            "class": PersonalizedPageRankRetriever,
            "name": "Personalized PageRank (alpha=0.6)",
            "init_args": {"alpha": 0.6, "k_mesh": 5, "k_context": 5},
            "retrieve_args": {"top_k": 10}
        },
        {
            "class": PersonalizedPageRankRetriever,
            "name": "Personalized PageRank (alpha=0.8)",
            "init_args": {"alpha": 0.8, "k_mesh": 10, "k_context": 3},
            "retrieve_args": {"top_k": 10}
        }
    ]
    
    # Evaluation configuration
    k_eval_values = [1, 3, 5, 10]
    all_metrics = {}
    
    print(f"\n🎯 Will evaluate {len(retriever_configs)} retriever configurations")
    print(f"📊 Evaluation metrics: Precision, Recall, F1, MRR, NDCG, Success, MAP, Coverage")
    print(f"📈 K values: {k_eval_values}")
    
    # Evaluate each retriever
    for i, config in enumerate(retriever_configs, 1):
        print(f"\n🔄 [{i}/{len(retriever_configs)}] Evaluating: {config['name']}")
        
        try:
            metrics = evaluate_single_retriever(
                source_data=data,
                retriever_class=config["class"],
                retriever_name=config["name"],
                retriever_init_args=config["init_args"],
                retriever_args=config["retrieve_args"],
                embedding_model=embedding_model,
                neo4j_driver=neo4j_connection.get_driver(),
                output_base_dir=output_base_dir,
                k_eval_values=k_eval_values,
            )
            
            if metrics:
                all_metrics[config["name"]] = metrics
                print(f"✅ [{i}/{len(retriever_configs)}] Completed: {config['name']}")
            else:
                print(f"❌ [{i}/{len(retriever_configs)}] Failed: {config['name']}")
                
        except Exception as e:
            print(f"❌ [{i}/{len(retriever_configs)}] Error with {config['name']}: {e}")
            continue
    
    # Generate comparison report
    if all_metrics:
        print(f"\n🎉 Evaluation completed! {len(all_metrics)} retrievers successful")
        compare_retrievers(all_metrics, output_base_dir, k_eval_values)
    else:
        print("❌ No retrievers were successfully evaluated")
    
    # Final summary
    print(f"\n🏁 EVALUATION COMPLETE")
    print(f"📁 All results saved to: {output_base_dir}")
    print(f"✅ Successfully evaluated: {len(all_metrics)} retrievers")
    print(f"❌ Failed evaluations: {len(retriever_configs) - len(all_metrics)}")
    
    # Close Neo4j connection
    neo4j_connection.close()
    print("🔐 Neo4j connection closed")
    
    return output_base_dir, all_metrics


if __name__ == "__main__":
    try:
        output_dir, metrics = run_comprehensive_evaluation()
        print(f"\n🎊 SUCCESS! Evaluation results available at: {output_dir}")
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n💥 Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc() 