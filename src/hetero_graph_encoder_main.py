#!/usr/bin/env python3
"""
Heterogeneous Graph Encoder Training Main Script

This script orchestrates the training of a heterogeneous graph neural network
that leverages QA_PAIR, CONTEXT, and MESH nodes with their relationships
from the Neo4j knowledge graph.
"""

import os
import time
import torch
from datetime import datetime
from typing import Dict, Tuple, List

# Import our heterogeneous modules
from graph_embeddings.hetero_gnn_data_extraction import (
    connect_to_neo4j, build_heterogeneous_data
)
from graph_embeddings.hetero_gnn_data_preparation import (
    prepare_heterogeneous_training_data
)
from graph_embeddings.hetero_graph_encoder_model import (
    build_hetero_model
)
from graph_embeddings.hetero_gnn_train import (
    train_hetero_model, save_hetero_model, save_hetero_training_artifacts
)
from graph_embeddings.utils import set_seed
from configs.config import ConfigPath, ConfigEnv


def main():
    """Main training pipeline for heterogeneous graph encoder."""
    
    # Configuration
    config = {
        # Data settings
        'sample_subgraph': True,
        'max_qa_pairs': 2000,  # Start smaller for testing
        'max_contexts_per_qa': 8,
        'max_mesh_per_context': 3,
        
        # Model settings
        'hidden_channels': 512,
        'out_channels': 768,
        'num_layers': 3,
        'use_attention': True,
        'dropout': 0.2,
        
        # Training settings
        'epochs': 200,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'patience': 20,
        'eval_freq': 5,
        'λ_semantic': 0.3,
        
        # Other settings
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'run_name': f"hetero_gnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    print("=" * 60)
    print("HETEROGENEOUS GRAPH ENCODER TRAINING")
    print("=" * 60)
    print(f"Run name: {config['run_name']}")
    print(f"Device: {config['device']}")
    print(f"Sample subgraph: {config['sample_subgraph']}")
    print(f"Max QA pairs: {config['max_qa_pairs']}")
    print("=" * 60)
    
    # Set random seed
    set_seed(config['seed'])
    device = torch.device(config['device'])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ConfigPath.MODELS_DIR, f"hetero_gnn_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "hetero_model.pt")
    
    try:
        # Step 1: Connect to Neo4j and extract heterogeneous data
        print("\n1. CONNECTING TO NEO4J AND EXTRACTING DATA")
        print("-" * 50)
        
        start_time = time.time()
        gds = connect_to_neo4j(ConfigEnv.NEO4J_URI, ConfigEnv.NEO4J_USER, ConfigEnv.NEO4J_PASSWORD, ConfigEnv.NEO4J_DB)
        
        raw_data = build_heterogeneous_data(
            gds,
            sample_subgraph=config['sample_subgraph'],
            max_qa_pairs=config['max_qa_pairs'],
            max_contexts_per_qa=config['max_contexts_per_qa'],
            max_mesh_per_context=config['max_mesh_per_context'],
            seed=config['seed']
        )
        
        print(f"Data extraction completed in {time.time() - start_time:.2f}s")
        
        # Step 2: Prepare training data
        print("\n2. PREPARING HETEROGENEOUS TRAINING DATA")
        print("-" * 50)
        
        start_time = time.time()
        train_data, val_data, test_data = prepare_heterogeneous_training_data(
            raw_data,
            device=device
        )
        
        print(f"Data preparation completed in {time.time() - start_time:.2f}s")
        
        # Step 3: Build heterogeneous model
        print("\n3. BUILDING HETEROGENEOUS MODEL")
        print("-" * 50)
        
        # Extract model parameters
        node_types = list(raw_data['node_features'].keys())
        edge_types = list(raw_data['edge_indices'].keys())
        
        # Convert edge types to tuples
        edge_type_tuples = []
        for edge_type in edge_types:
            source_type, target_type = raw_data['edge_type_mappings'][edge_type]
            edge_type_tuples.append((source_type, edge_type, target_type))
        
        # Get input dimensions
        in_channels_dict = {
            node_type: features.shape[1] 
            for node_type, features in raw_data['node_features'].items()
        }
        
        print(f"Node types: {node_types}")
        print(f"Edge types: {edge_type_tuples}")
        print(f"Input dimensions: {in_channels_dict}")
        
        encoder, predictor, optimizer, scheduler = build_hetero_model(
            node_types=node_types,
            edge_types=edge_type_tuples,
            in_channels_dict=in_channels_dict,
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            device=device,
            num_layers=config['num_layers'],
            use_attention=config['use_attention']
        )
        
        # Step 4: Train the model
        print("\n4. TRAINING HETEROGENEOUS MODEL")
        print("-" * 50)
        
        start_time = time.time()
        target_edge_type = ('context', 'is_similar_to', 'context')
        
        history = train_hetero_model(
            encoder=encoder,
            predictor=predictor,
            optimizer=optimizer,
            scheduler=scheduler,
            train_data=train_data,
            val_data=val_data,
            target_edge_type=target_edge_type,
            epochs=config['epochs'],
            λ_semantic=config['λ_semantic'],
            patience=config['patience'],
            eval_freq=config['eval_freq'],
            device=device
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f}s")
        
        # Step 5: Save model and results
        print("\n5. SAVING MODEL AND RESULTS")
        print("-" * 50)
        
        # Prepare metadata
        metadata = {
            'config': config,
            'node_types': node_types,
            'edge_types': [list(et) for et in edge_type_tuples],  # Convert tuples to lists for JSON
            'in_channels_dict': in_channels_dict,
            'training_time': training_time,
            'final_auc': history['val_auc'][-1] if history['val_auc'] else 0,
            'final_ap': history['val_ap'][-1] if history['val_ap'] else 0,
            'best_auc': max(history['val_auc']) if history['val_auc'] else 0,
            'best_ap': max(history['val_ap']) if history['val_ap'] else 0,
        }
        
        # Save model
        save_hetero_model(encoder, predictor, model_path, metadata)
        
        # Save training artifacts
        save_hetero_training_artifacts(history, run_dir, model_path, metadata)
        
        # Step 6: Final evaluation on test set
        print("\n6. FINAL EVALUATION ON TEST SET")
        print("-" * 50)
        
        from graph_embeddings.hetero_gnn_train import evaluate_hetero_model
        
        test_auc, test_ap = evaluate_hetero_model(
            encoder, predictor, test_data, target_edge_type
        )
        
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test AP: {test_ap:.4f}")
        
        # Update metadata with test results
        metadata['test_auc'] = test_auc
        metadata['test_ap'] = test_ap
        
        # Save updated metadata
        import json
        metadata_path = os.path.join(run_dir, "final_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Run directory: {run_dir}")
        print(f"Final validation AUC: {metadata['final_auc']:.4f}")
        print(f"Final validation AP: {metadata['final_ap']:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test AP: {test_ap:.4f}")
        print(f"Training time: {training_time:.2f}s")
        print("=" * 60)
        
        return metadata
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Close Neo4j connection
        if 'gds' in locals():
            gds.close()


if __name__ == "__main__":
    main() 