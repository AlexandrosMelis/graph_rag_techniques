import os
from datetime import datetime

import torch

from configs.config import ConfigEnv, ConfigPath
from graph_embeddings.compute_gnn_embeddings import write_graph_embeddings_to_neo4j
from graph_embeddings.gnn_data_extraction import (
    connect_to_neo4j,
    create_gds_graph,
    fetch_node_features,
    fetch_topology,
    sample_graph,
)
from graph_embeddings.gnn_data_preparation import build_pyg_data, split_data
from graph_embeddings.graph_encoder_model import build_model
from graph_embeddings.gnn_train import evaluate, save_model, train_model, save_training_artifacts
from graph_embeddings.utils import set_seed


def run_gnn_training(apply_sampling: bool = False, use_enhanced_model: bool = True) -> None:
    seed = 42
    set_seed(seed)

    # Neo4j connection
    gds = connect_to_neo4j(
        ConfigEnv.NEO4J_URI,
        ConfigEnv.NEO4J_USER,
        ConfigEnv.NEO4J_PASSWORD,
        ConfigEnv.NEO4J_DB,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. (Optional) Graph sampling
    graph_name = "contexts"
    G = create_gds_graph(gds=gds, graph_name=graph_name)
    if apply_sampling:
        G = sample_graph(gds, graph_name, f"{graph_name}_sample", seed=seed)

    # 2. Fetch topology & features
    edge_index, node_df = fetch_topology(gds, G)
    x = fetch_node_features(node_df)

    # 3. Build PyG Data & Split onto device
    data = build_pyg_data(x, edge_index)
    train_data, val_data, test_data = split_data(data, device=device)

    # 4. Enhanced model configuration
    in_dim = x.size(1)  # e.g. 768 (BERT dimensions)
    
    if use_enhanced_model:
        # Enhanced configuration for better BERT+graph fusion
        hid_dim = 512  # Increased hidden dimension
        out_dim = 768  # Keep same as input for better feature fusion
        lr = 5e-4      # Lower learning rate for more stable training
        weight_decay = 1e-5  # Reduced weight decay
        num_layers = 4
        use_attention = True
        
        print(f"Using enhanced model configuration:")
        print(f"  - Hidden dim: {hid_dim}")
        print(f"  - Output dim: {out_dim}")
        print(f"  - Layers: {num_layers}")
        print(f"  - Attention: {use_attention}")
        print(f"  - Learning rate: {lr}")
    else:
        # Original configuration
        hid_dim = 256
        out_dim = 768
        lr = 1e-3
        weight_decay = 1e-4
        num_layers = 3
        use_attention = False

    encoder, predictor, optimizer, scheduler = build_model(
        in_channels=in_dim,
        hidden_channels=hid_dim,
        out_channels=out_dim,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        num_layers=num_layers,
        use_attention=use_attention,
    )

    # 5. Enhanced training configuration
    training_config = {
        "epochs": 800,
        "λ_feat": 0.3,  # Lower initial feature weight (will decay adaptively)
        "patience": 30,  # Increased patience
        "eval_freq": 3,  # More frequent evaluation
        "use_adaptive_weights": True,
        "use_cosine_loss": True,
        "min_improvement": 5e-5,  # Lower improvement threshold
    }
    
    print(f"Training configuration: {training_config}")
    
    history = train_model(
        encoder=encoder,
        predictor=predictor,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data=train_data,
        val_data=val_data,
        **training_config
    )

    # 6. Final evaluation on test split
    test_auc, test_ap = evaluate(encoder, predictor, test_data)
    print(f"\n=== FINAL TEST RESULTS ===")
    print(f"Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}")

    # 7. Save model, metrics, and plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = "enhanced" if use_enhanced_model else "basic"
    run_dir = os.path.join(ConfigPath.MODELS_DIR, f"gnn_{model_suffix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "graphsage_encoder_pred.pt")

    # Save model
    save_model(encoder, predictor, model_path)

    # Enhanced metrics with configuration
    history["test_auc"] = test_auc
    history["test_ap"] = test_ap
    history["config"] = {
        "model": {
            "in_dim": in_dim,
            "hid_dim": hid_dim,
            "out_dim": out_dim,
            "num_layers": num_layers,
            "use_attention": use_attention,
        },
        "training": training_config,
        "hyperparams": {
            "lr": lr,
            "weight_decay": weight_decay,
        },
        "data": {
            "num_nodes": train_data.x.size(0),
            "num_features": train_data.x.size(1),
            "train_pos_edges": train_data.pos_edge_label_index.size(1),
            "train_neg_edges": train_data.neg_edge_label_index.size(1),
            "val_pos_edges": val_data.pos_edge_label_index.size(1),
            "val_neg_edges": val_data.neg_edge_label_index.size(1),
            "test_pos_edges": test_data.pos_edge_label_index.size(1),
            "test_neg_edges": test_data.neg_edge_label_index.size(1),
        }
    }
    
    # Save metrics and create comprehensive training plots
    save_training_artifacts(history, run_dir, model_path)
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Model saved to: {model_path}")
    print(f"Run directory: {run_dir}")
    print(f"\nGenerated files:")
    print(f"  📊 training_progress.png - Main training dashboard")
    print(f"  📈 detailed_loss_analysis.png - Loss component analysis")
    print(f"  📋 validation_analysis.png - Validation metrics breakdown")
    print(f"  📋 training_metrics.json - Complete training history")
    
    # Performance summary
    if history["val_auc"]:
        max_auc = max(history["val_auc"])
        max_ap = max(history["val_ap"])
        final_auc = history["val_auc"][-1]
        final_ap = history["val_ap"][-1]
        total_epochs = len(history["epoch"])
        
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Training epochs: {total_epochs}")
        print(f"   Best validation AUC: {max_auc:.4f}")
        print(f"   Best validation AP: {max_ap:.4f}")
        print(f"   Final validation AUC: {final_auc:.4f}")
        print(f"   Final validation AP: {final_ap:.4f}")
        print(f"   Test AUC: {test_auc:.4f}")
        print(f"   Test AP: {test_ap:.4f}")


def generate_embeddings_for_model(model_path: str = None) -> None:
    """
    Generate and write GNN embeddings to Neo4j for a specific model.
    
    Args:
        model_path: Path to the trained model. If None, uses the latest enhanced model.
    """
    try:
        write_graph_embeddings_to_neo4j(
            model_path=model_path,
            graph_name="contexts",
            use_auto_device=True,
            batch_size=200
        )
        print("✅ Successfully generated and wrote embeddings to Neo4j!")
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    # Command line usage examples:
    # python graph_autoencoder_training_main.py train         # Train new model
    # python graph_autoencoder_training_main.py embeddings   # Generate embeddings with latest model
    # python graph_autoencoder_training_main.py both         # Train then generate embeddings
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "train":
            print("🚀 Starting GNN training...")
            run_gnn_training(apply_sampling=False, use_enhanced_model=True)
            
        elif mode == "embeddings":
            print("📊 Generating embeddings with latest model...")
            generate_embeddings_for_model()
            
        elif mode == "both":
            print("🚀 Training model and generating embeddings...")
            run_gnn_training(apply_sampling=False, use_enhanced_model=True)
            print("\n" + "="*50)
            print("Training complete! Now generating embeddings...")
            generate_embeddings_for_model()
            
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python graph_autoencoder_training_main.py [train|embeddings|both]")
    else:
        # Default behavior: generate embeddings with latest model
        print("📊 Generating embeddings with latest enhanced model...")
        print("   Use 'python graph_autoencoder_training_main.py train' to train a new model")
        generate_embeddings_for_model()
