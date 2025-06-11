import os
from datetime import datetime

import torch

from configs.config import ConfigEnv, ConfigPath
from projection_models.projection_gat_model import train

"""
Main file for training the GAT-based query projection model.

Model details:
- Architecture: Graph Attention Network (GAT) with multi-layer attention
- Input: Query BERT embedding + subgraph of related CONTEXT nodes
- Subgraph includes: 
  * Query node connected to relevant CONTEXT nodes
  * CONTEXT-CONTEXT connections via IS_SIMILAR_TO relationships
  * Node features: BERT embeddings for all nodes
- Output: Projected query embedding in graph embedding space
- Training objective: Combined cosine similarity + MSE loss
- Target: Average of connected CONTEXT graph embeddings

Key features:
- Query-aware attention pooling for graph-level readout
- Multi-head attention in GAT layers
- Automatic subgraph construction from Neo4j
- Comprehensive evaluation with recall@k and MRR
- Embedding visualization (t-SNE comparison)
"""


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ConfigPath.MODELS_DIR, f"gat_proj_model_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # GAT model hyperparameters
    batch_size = 16  # Smaller batch size due to subgraph memory requirements
    epochs = 500
    lr = 1e-3  # Lower learning rate for GAT stability
    hidden_dim = 512  # Hidden dimension for GAT layers
    n_layers = 2  # Number of GAT layers
    n_heads = 6  # Number of attention heads
    dropout = 0.3  # Dropout rate
    max_neighbors = 10  # Maximum neighbors per context node
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model directory: {run_dir}")

    # Training configuration
    training_config = {
        "lr": lr,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": dropout,
        "max_neighbors": max_neighbors,
        "patience": 10,  # Early stopping patience
        "val_ratio": 0.15,  # Validation split ratio
        "device": device
    }

    print("Training GAT-based Query Projection Model")
    print("=" * 50)
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of GAT layers: {n_layers}")
    print(f"Attention heads: {n_heads}")
    print(f"Dropout: {dropout}")
    print(f"Max neighbors per context: {max_neighbors}")
    print("=" * 50)

    try:
        model = train(
            uri=ConfigEnv.NEO4J_URI,
            user=ConfigEnv.NEO4J_USER,
            password=ConfigEnv.NEO4J_PASSWORD,
            database=ConfigEnv.NEO4J_DB,
            model_dir=run_dir,
            batch_size=batch_size,
            epochs=epochs,
            evaluate_during_training=True,  # Enable evaluation during training
            eval_frequency=15,  # Evaluate every 15 epochs for more data points
            **training_config
        )

        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved in: {run_dir}")
        print(f"\n📊 Generated Plots & Reports:")
        print(f"   📈 training_progression_report.png - Comprehensive 6-panel training report")
        print(f"   📉 detailed_loss_curves.png - Publication-ready loss curves")
        print(f"   📊 evaluation_metrics.png - Evaluation metrics progression (if enabled)")
        print(f"   📈 recall_progression.png - Dedicated recall@k metrics during training")
        print(f"   📊 training_history.png - Basic loss curves (backward compatibility)")
        print(f"   🎨 embedding_visualization.png - t-SNE comparison plots")
        print(f"   📋 evaluation_results.txt - Final performance metrics")
        print(f"   📄 recall_evaluation_summary.txt - Detailed recall metrics history")
        
        # Print model summary
        print(f"\n📋 Model Summary:")
        print(f"   - Input dimension: {model.input_dim}")
        print(f"   - Hidden dimension: {model.hidden_dim}")
        print(f"   - Output dimension: {model.output_dim}")
        print(f"   - GAT layers: {model.n_layers}")
        print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 