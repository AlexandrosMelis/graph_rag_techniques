"""
Main Training Script for Dual Projection Model

This script demonstrates how to train a dual projection model that learns to project
questions into both semantic (SBERT) and graph (GNN) embedding spaces using 
contrastive learning with hard negative mining.

Usage:
    python dual_projection_main.py

Author: AI Assistant
"""

import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

# Import required modules
from configs import ConfigEnv, ConfigPath
from data_collection.reader import BioASQDataReader
from llms.embedding_model import EmbeddingModel
from knowledge_graph.connection import Neo4jConnection
from projection_models.dual_projection_model import (
    DualProjectionModel,
    DualProjectionTrainer,
    HardNegativeMiner,
    DualSpaceDataset,
    collate_dual_space_batch,
    load_dual_projection_model
)
from utils.utils import save_json_file


def prepare_dual_space_data(
    data_samples, 
    max_samples=1000,
    train_split=0.8
):
    """
    Prepare data for dual-space contrastive learning.
    
    For this example, we'll use the same contexts for both semantic and graph spaces.
    In practice, you might want to:
    1. Use SBERT embeddings for semantic contexts
    2. Use GNN node embeddings for graph contexts
    """
    print(f"🔧 Preparing dual-space data from {len(data_samples)} samples...")
    
    # Limit samples if specified
    if max_samples and len(data_samples) > max_samples:
        data_samples = data_samples[:max_samples]
        print(f"📊 Limited to {max_samples} samples for faster training")
    
    # Extract questions and contexts
    questions = []
    semantic_contexts = []
    graph_contexts = []
    
    for sample in data_samples:
        question = sample['question']
        
        # For semantic contexts, use the first context or the question itself if no contexts
        if sample.get('contexts') and len(sample['contexts']) > 0:
            semantic_context = sample['contexts'][0]
        else:
            # Fallback: use a modified version of the question
            semantic_context = f"Context for: {question}"
        
        # For graph contexts, we'll use the same for this demo
        # In practice, you'd retrieve actual graph node embeddings
        graph_context = semantic_context
        
        questions.append(question)
        semantic_contexts.append(semantic_context)
        graph_contexts.append(graph_context)
    
    # Train/validation split
    split_idx = int(len(questions) * train_split)
    
    train_data = {
        'questions': questions[:split_idx],
        'semantic_contexts': semantic_contexts[:split_idx],
        'graph_contexts': graph_contexts[:split_idx]
    }
    
    val_data = {
        'questions': questions[split_idx:],
        'semantic_contexts': semantic_contexts[split_idx:],
        'graph_contexts': graph_contexts[split_idx:]
    }
    
    print(f"✅ Prepared data: {len(train_data['questions'])} train, {len(val_data['questions'])} validation")
    
    return train_data, val_data


def setup_hard_negative_mining(
    train_data, 
    embedding_model, 
    neo4j_connection,
    use_precomputed=True,
    precomputed_path=None
):
    """Setup hard negative mining with precomputed embeddings."""
    print("🔧 Setting up hard negative mining...")
    
    hard_negative_miner = HardNegativeMiner(
        embedding_model=embedding_model,
        neo4j_connection=neo4j_connection,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if precomputed_path is None:
        precomputed_path = os.path.join(ConfigPath.MODELS_DIR, "dual_projection_negatives.pkl")
    
    if use_precomputed and os.path.exists(precomputed_path):
        print(f"📂 Loading precomputed hard negatives from {precomputed_path}")
        hard_negative_miner.load_precomputed_data(precomputed_path)
    else:
        print("🔧 Computing hard negatives (this may take a while)...")
        
        # Combine all contexts for negative mining
        all_contexts = list(set(
            train_data['semantic_contexts'] + train_data['graph_contexts']
        ))
        
        hard_negative_miner.precompute_embeddings(
            questions=train_data['questions'],
            contexts=all_contexts,
            batch_size=32
        )
        
        # Save for future use
        hard_negative_miner.save_precomputed_data(precomputed_path)
    
    return hard_negative_miner


def create_data_loaders(
    train_data, 
    val_data, 
    embedding_model,
    hard_negative_miner=None,
    batch_size=16,
    num_hard_negatives=5
):
    """Create data loaders for training and validation."""
    print("🔧 Creating data loaders...")
    
    # Create datasets
    train_dataset = DualSpaceDataset(
        questions=train_data['questions'],
        semantic_contexts=train_data['semantic_contexts'],
        graph_contexts=train_data['graph_contexts'],
        embedding_model=embedding_model,
        hard_negative_miner=hard_negative_miner,
        num_hard_negatives=num_hard_negatives,
        use_hard_negatives=(hard_negative_miner is not None)
    )
    
    val_dataset = DualSpaceDataset(
        questions=val_data['questions'],
        semantic_contexts=val_data['semantic_contexts'],
        graph_contexts=val_data['graph_contexts'],
        embedding_model=embedding_model,
        hard_negative_miner=None,  # No hard negatives for validation
        use_hard_negatives=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_dual_space_batch,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dual_space_batch,
        num_workers=0
    )
    
    print(f"✅ Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader


def evaluate_dual_projections(model, val_loader, device="cuda"):
    """Evaluate the quality of dual projections."""
    print("📊 Evaluating dual projection quality...")
    
    model.eval()
    semantic_similarities = []
    graph_similarities = []
    cross_similarities = []  # Semantic proj vs graph contexts
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            question_embeddings = batch['question_embeddings'].to(device)
            semantic_contexts = batch['semantic_context_embeddings'].to(device)
            graph_contexts = batch['graph_context_embeddings'].to(device)
            
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
    """Main training function."""
    print("🚀 Starting Dual Projection Model Training")
    print("=" * 60)
    
    # Configuration
    CONFIG = {
        'max_samples': None,  # Limit for demo (set to None for full dataset)
        'train_split': 0.8,
        'batch_size': 16,
        'num_epochs': 300,
        'learning_rate': 1e-4,
        'hidden_dims': [512, 2048, 1024],
        'dropout': 0.2,
        'semantic_loss_weight': 0.5,
        'graph_loss_weight': 0.5,
        'temperature': 0.1,
        'num_hard_negatives': 5,
        'use_hard_negative_mining': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"🔧 Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(ConfigPath.MODELS_DIR, f"dual_proj_model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    save_json_file(config_path, CONFIG)
    
    # 1. Load data
    print("\n📊 Loading BioASQ data...")
    reader = BioASQDataReader()
    train_file = os.path.join(ConfigPath.RAW_DATA_DIR, "bioasq_train.parquet")
    
    if not os.path.exists(train_file):
        print(f"❌ Training file not found: {train_file}")
        print("Please ensure BioASQ training data is available.")
        return
    
    data_samples = reader.read_parquet_file(train_file)
    print(f"✅ Loaded {len(data_samples)} data samples")
    
    # 2. Prepare dual-space data
    train_data, val_data = prepare_dual_space_data(
        data_samples,
        max_samples=CONFIG['max_samples'],
        train_split=CONFIG['train_split']
    )
    
    # 3. Initialize components
    print("\n🔧 Initializing components...")
    
    # Embedding model
    embedding_model = EmbeddingModel()
    print("✅ Initialized embedding model")
    
    # Neo4j connection
    neo4j_connection = Neo4jConnection(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
    )
    print("✅ Initialized Neo4j connection")
    
    # 4. Setup hard negative mining
    hard_negative_miner = None
    if CONFIG['use_hard_negative_mining']:
        try:
            hard_negative_miner = setup_hard_negative_mining(
                train_data=train_data,
                embedding_model=embedding_model,
                neo4j_connection=neo4j_connection,
                use_precomputed=True,
                precomputed_path=os.path.join(output_dir, "hard_negatives.pkl")
            )
            print("✅ Setup hard negative mining")
        except Exception as e:
            print(f"⚠️ Warning: Failed to setup hard negative mining: {e}")
            print("   Continuing without hard negatives...")
            CONFIG['use_hard_negative_mining'] = False
    
    # 5. Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        embedding_model=embedding_model,
        hard_negative_miner=hard_negative_miner,
        batch_size=CONFIG['batch_size'],
        num_hard_negatives=CONFIG['num_hard_negatives']
    )
    
    # 6. Create model
    print(f"\n🏗️ Creating Dual Projection Model...")
    model = DualProjectionModel(
        dim_sem=768,
        dim_graph=768,
        hidden_dims=CONFIG['hidden_dims'],
        p_dropout=CONFIG['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Created model with {total_params:,} parameters")
    
    # 7. Setup trainer
    trainer = DualProjectionTrainer(
        model=model,
        semantic_loss_weight=CONFIG['semantic_loss_weight'],
        graph_loss_weight=CONFIG['graph_loss_weight'],
        temperature=CONFIG['temperature'],
        learning_rate=CONFIG['learning_rate'],
        device=CONFIG['device']
    )
    print("✅ Initialized trainer")
    
    # 8. Train model
    print(f"\n🚀 Starting training for {CONFIG['num_epochs']} epochs...")
    train_history = trainer.train(
        train_dataloader=train_loader,
        num_epochs=CONFIG['num_epochs'],
        validation_dataloader=val_loader,
        save_dir=output_dir
    )
    
    # 9. Load best model for evaluation
    best_model_path = os.path.join(output_dir, "best_dual_projection_model.pt")
    if os.path.exists(best_model_path):
        print(f"\n📂 Loading best model for evaluation...")
        best_model, checkpoint = load_dual_projection_model(best_model_path, CONFIG['device'])
        
        # 10. Evaluate model
        eval_metrics = evaluate_dual_projections(best_model, val_loader, CONFIG['device'])
        
        # Save evaluation results
        eval_path = os.path.join(output_dir, "evaluation_metrics.json")
        save_json_file(eval_path, eval_metrics)
        
    else:
        print("⚠️ Best model not found, using last epoch model for evaluation")
        eval_metrics = evaluate_dual_projections(model, val_loader, CONFIG['device'])
    
    # 11. Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    save_json_file(history_path, train_history)
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📋 Key files:")
    print(f"   - config.json: Training configuration")
    print(f"   - best_dual_projection_model.pt: Best trained model")
    print(f"   - training_history.json: Training metrics")
    print(f"   - evaluation_metrics.json: Final evaluation results")
    
    # Close Neo4j connection
    neo4j_connection.close()
    print("🔌 Closed Neo4j connection")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise 