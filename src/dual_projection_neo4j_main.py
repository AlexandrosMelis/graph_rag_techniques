"""
Main Training Script for Dual Projection Model with Neo4j Data

This script trains a dual projection model using real data from Neo4j database:
- Question embeddings from QA_PAIR.embedding
- Semantic context embeddings from CONTEXT.embedding
- Graph structure embeddings from CONTEXT.graph_embedding
- Uses HAS_CONTEXT relationships for positive pairs
- Implements hard negative mining using Neo4j vector similarity

Usage:
    python dual_projection_neo4j_main.py

Author: AI Assistant
"""

import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

# Import required modules
from configs import ConfigEnv, ConfigPath
from llms.embedding_model import EmbeddingModel
from knowledge_graph.connection import Neo4jConnection
from projection_models.dual_projection_model import (
    DualProjectionModel,
    DualProjectionTrainer,
    load_dual_projection_model
)
from projection_models.dual_projection_neo4j_data import (
    Neo4jDualProjectionDataLoader,
    Neo4jDualProjectionDataset,
    collate_neo4j_dual_projection_batch,
    validate_neo4j_data,
    get_embedding_statistics
)
from utils.utils import save_json_file


def setup_neo4j_connection():
    """Setup Neo4j connection and validate data."""
    print("🔌 Setting up Neo4j connection...")
    
    neo4j_connection = Neo4jConnection(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
    )
    
    # Validate data
    validation_results = validate_neo4j_data(neo4j_connection)
    
    if not validation_results['valid']:
        print("❌ Neo4j data validation failed!")
        for error in validation_results['errors']:
            print(f"   - {error}")
        raise ValueError("Neo4j database does not contain required data for training")
    
    print("✅ Neo4j connection established and data validated")
    return neo4j_connection, validation_results


def load_and_analyze_data(
    neo4j_connection: Neo4jConnection,
    max_samples: int = None,
    use_hard_negatives: bool = True,
    num_hard_negatives: int = 5,
    train_split: float = 0.8
):
    """Load and analyze training data from Neo4j."""
    print(f"\n🔍 Loading training data from Neo4j...")
    
    # Initialize data loader
    data_loader = Neo4jDualProjectionDataLoader(neo4j_connection)
    
    # Create training data
    train_data, val_data = data_loader.create_training_data(
        max_samples=max_samples,
        use_hard_negatives=use_hard_negatives,
        num_hard_negatives=num_hard_negatives,
        train_split=train_split
    )
    
    # Analyze data statistics
    print(f"\n📊 Data Analysis:")
    
    # Analyze question embeddings
    question_embeddings = [sample['question_embedding'] for sample in train_data['samples']]
    question_stats = get_embedding_statistics(question_embeddings)
    print(f"   Question embeddings: {question_stats}")
    
    # Analyze semantic context embeddings
    semantic_embeddings = [sample['positive_context']['semantic_embedding'] for sample in train_data['samples']]
    semantic_stats = get_embedding_statistics(semantic_embeddings)
    print(f"   Semantic context embeddings: {semantic_stats}")
    
    # Analyze graph context embeddings
    graph_embeddings = [sample['positive_context']['graph_embedding'] for sample in train_data['samples']]
    graph_stats = get_embedding_statistics(graph_embeddings)
    print(f"   Graph context embeddings: {graph_stats}")
    
    # Analyze hard negatives
    if use_hard_negatives:
        hard_neg_counts = [len(sample.get('hard_negatives', [])) for sample in train_data['samples']]
        avg_hard_negs = np.mean(hard_neg_counts) if hard_neg_counts else 0
        print(f"   Average hard negatives per sample: {avg_hard_negs:.2f}")
    
    return train_data, val_data, {
        'question_stats': question_stats,
        'semantic_stats': semantic_stats,
        'graph_stats': graph_stats
    }


def create_data_loaders(
    train_data: dict,
    val_data: dict,
    batch_size: int = 16,
    use_hard_negatives: bool = True
):
    """Create PyTorch data loaders for training and validation."""
    print(f"\n🔧 Creating data loaders...")
    
    # Create datasets
    train_dataset = Neo4jDualProjectionDataset(
        training_data=train_data,
        use_hard_negatives=use_hard_negatives
    )
    
    val_dataset = Neo4jDualProjectionDataset(
        training_data=val_data,
        use_hard_negatives=False  # No hard negatives for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_neo4j_dual_projection_batch,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_neo4j_dual_projection_batch,
        num_workers=0
    )
    
    print(f"✅ Created data loaders:")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def evaluate_dual_projections_neo4j(model, val_loader, device="cuda"):
    """Evaluate the quality of dual projections using Neo4j data."""
    print("📊 Evaluating dual projection quality...")
    
    model.eval()
    semantic_similarities = []
    graph_similarities = []
    cross_similarities = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            question_embeddings = batch['question_embeddings'].to(device)
            semantic_contexts = batch['positive_semantic_embeddings'].to(device)
            graph_contexts = batch['positive_graph_embeddings'].to(device)
            
            # Get projections
            semantic_proj, graph_proj = model(question_embeddings)
            
            # Compute similarities (diagonal elements for positive pairs)
            semantic_sim = torch.cosine_similarity(semantic_proj, semantic_contexts).cpu().numpy()
            graph_sim = torch.cosine_similarity(graph_proj, graph_contexts).cpu().numpy()
            cross_sim = torch.cosine_similarity(semantic_proj, graph_contexts).cpu().numpy()
            
            semantic_similarities.extend(semantic_sim)
            graph_similarities.extend(graph_sim)
            cross_similarities.extend(cross_sim)
    
    # Compute statistics
    eval_metrics = {
        'semantic_similarity': {
            'mean': float(np.mean(semantic_similarities)),
            'std': float(np.std(semantic_similarities)),
            'min': float(np.min(semantic_similarities)),
            'max': float(np.max(semantic_similarities))
        },
        'graph_similarity': {
            'mean': float(np.mean(graph_similarities)),
            'std': float(np.std(graph_similarities)),
            'min': float(np.min(graph_similarities)),
            'max': float(np.max(graph_similarities))
        },
        'cross_similarity': {
            'mean': float(np.mean(cross_similarities)),
            'std': float(np.std(cross_similarities)),
            'min': float(np.min(cross_similarities)),
            'max': float(np.max(cross_similarities))
        }
    }
    
    print(f"📊 Evaluation Results:")
    print(f"   Semantic similarity: {eval_metrics['semantic_similarity']['mean']:.4f} ± {eval_metrics['semantic_similarity']['std']:.4f}")
    print(f"   Graph similarity: {eval_metrics['graph_similarity']['mean']:.4f} ± {eval_metrics['graph_similarity']['std']:.4f}")
    print(f"   Cross similarity: {eval_metrics['cross_similarity']['mean']:.4f} ± {eval_metrics['cross_similarity']['std']:.4f}")
    
    return eval_metrics


def main():
    """Main training function with Neo4j data."""
    print("🚀 Starting Dual Projection Model Training with Neo4j Data")
    print("=" * 70)
    
    # Configuration
    CONFIG = {
        'max_samples': None,  # Set to None for full dataset
        'train_split': 0.8,
        'batch_size': 16,
        'num_epochs': 400,
        'learning_rate': 1e-4,
        'hidden_dims': [512, 2048, 1024],
        'dropout': 0.2,
        'semantic_loss_weight': 0.5,
        'graph_loss_weight': 0.5,
        'temperature': 0.1,
        'num_hard_negatives': 5,
        'use_hard_negative_mining': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'patience': 10,
        'min_delta': 1e-4
    }
    
    print(f"🔧 Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(ConfigPath.MODELS_DIR, f"dual_proj_neo4j_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    save_json_file(config_path, CONFIG)
    
    # 1. Setup Neo4j connection and validate data
    neo4j_connection, validation_results = setup_neo4j_connection()
    
    # Save validation results
    validation_path = os.path.join(output_dir, "data_validation.json")
    save_json_file(validation_path, validation_results)
    
    # 2. Load and analyze data
    train_data, val_data, data_stats = load_and_analyze_data(
        neo4j_connection=neo4j_connection,
        max_samples=CONFIG['max_samples'],
        use_hard_negatives=CONFIG['use_hard_negative_mining'],
        num_hard_negatives=CONFIG['num_hard_negatives'],
        train_split=CONFIG['train_split']
    )
    
    # Save data statistics
    data_stats_path = os.path.join(output_dir, "data_statistics.json")
    save_json_file(data_stats_path, data_stats)
    
    # 3. Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        batch_size=CONFIG['batch_size'],
        use_hard_negatives=CONFIG['use_hard_negative_mining']
    )
    
    # 4. Determine embedding dimensions from data
    sample_question_emb = train_data['samples'][0]['question_embedding']
    sample_semantic_emb = train_data['samples'][0]['positive_context']['semantic_embedding']
    sample_graph_emb = train_data['samples'][0]['positive_context']['graph_embedding']
    
    input_dim = len(sample_question_emb)
    semantic_dim = len(sample_semantic_emb)
    graph_dim = len(sample_graph_emb)
    
    print(f"\n🏗️ Model Architecture:")
    print(f"   Input dimension: {input_dim}")
    print(f"   Semantic output dimension: {semantic_dim}")
    print(f"   Graph output dimension: {graph_dim}")
    
    # 5. Create model
    model = DualProjectionModel(
        input_dim=input_dim,
        dim_sem=semantic_dim,
        dim_graph=graph_dim,
        hidden_dims=CONFIG['hidden_dims'],
        p_dropout=CONFIG['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Created model with {total_params:,} parameters")
    
    # 6. Setup trainer
    trainer = DualProjectionTrainer(
        model=model,
        semantic_loss_weight=CONFIG['semantic_loss_weight'],
        graph_loss_weight=CONFIG['graph_loss_weight'],
        temperature=CONFIG['temperature'],
        learning_rate=CONFIG['learning_rate'],
        device=CONFIG['device'],
        patience=CONFIG['patience'],
        min_delta=CONFIG['min_delta']
    )
    print("✅ Initialized trainer")
    
    # 7. Train model
    print(f"\n🚀 Starting training for {CONFIG['num_epochs']} epochs...")
    train_history = trainer.train(
        train_dataloader=train_loader,
        num_epochs=CONFIG['num_epochs'],
        validation_dataloader=val_loader,
        save_dir=output_dir
    )
    
    # 8. Load best model for evaluation
    best_model_path = os.path.join(output_dir, "best_dual_projection_model.pt")
    if os.path.exists(best_model_path):
        print(f"\n📂 Loading best model for evaluation...")
        best_model, checkpoint = load_dual_projection_model(best_model_path, CONFIG['device'])
        
        # 9. Evaluate model
        eval_metrics = evaluate_dual_projections_neo4j(best_model, val_loader, CONFIG['device'])
        
        # Save evaluation results
        eval_path = os.path.join(output_dir, "evaluation_metrics.json")
        save_json_file(eval_path, eval_metrics)
        
    else:
        print("⚠️ Best model not found, using last epoch model for evaluation")
        eval_metrics = evaluate_dual_projections_neo4j(model, val_loader, CONFIG['device'])
    
    # 10. Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    save_json_file(history_path, train_history)
    
    # 11. Create summary report
    summary = {
        'training_config': CONFIG,
        'data_validation': validation_results,
        'data_statistics': data_stats,
        'model_architecture': {
            'input_dim': input_dim,
            'semantic_dim': semantic_dim,
            'graph_dim': graph_dim,
            'total_parameters': total_params
        },
        'training_results': {
            'final_epoch': len(train_history.get('total_loss', [])),
            'best_validation_loss': trainer.best_val_loss,
            'early_stopped': trainer.early_stopped
        },
        'evaluation_metrics': eval_metrics,
        'output_directory': output_dir
    }
    
    summary_path = os.path.join(output_dir, "training_summary.json")
    save_json_file(summary_path, summary)
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📋 Key files:")
    print(f"   - config.json: Training configuration")
    print(f"   - best_dual_projection_model.pt: Best trained model")
    print(f"   - training_history.json: Training metrics")
    print(f"   - evaluation_metrics.json: Final evaluation results")
    print(f"   - data_validation.json: Neo4j data validation")
    print(f"   - training_summary.json: Complete training summary")
    
    # Close Neo4j connection
    neo4j_connection.close()
    print("🔌 Closed Neo4j connection")
    
    return output_dir, summary


if __name__ == "__main__":
    try:
        output_dir, summary = main()
        
        print(f"\n🎊 Training Summary:")
        print(f"📈 Final validation loss: {summary['training_results']['best_validation_loss']:.4f}")
        print(f"📊 Semantic similarity: {summary['evaluation_metrics']['semantic_similarity']['mean']:.4f}")
        print(f"📊 Graph similarity: {summary['evaluation_metrics']['graph_similarity']['mean']:.4f}")
        print(f"📁 Model directory: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise 