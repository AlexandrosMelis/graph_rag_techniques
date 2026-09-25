import json
import os
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# ----------------------------------
# Heterogeneous Training Functions
# ----------------------------------


def create_hetero_training_plots(history: dict, save_dir: str) -> None:
    """Create comprehensive training visualization plots for heterogeneous training."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Heterogeneous Graph Encoder Training Progress', fontsize=16, fontweight='bold')
    
    epochs = history['epoch']
    
    # 1. Total Loss Progression
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Loss Components
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['recon_loss'], 'r-', linewidth=2, label='Link Prediction Loss')
    if 'type_loss' in history:
        ax2.plot(epochs, history['type_loss'], 'g-', linewidth=2, label='Node Type Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    # 3. Validation Metrics
    ax3 = axes[0, 2]
    ax3.plot(epochs, history['val_auc'], 'purple', linewidth=2, marker='o', markersize=4, label='AUC')
    ax3.plot(epochs, history['val_ap'], 'orange', linewidth=2, marker='s', markersize=4, label='AP')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Validation Metrics')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # 4. Node Type Performance (if available)
    ax4 = axes[1, 0]
    if 'node_type_metrics' in history and history['node_type_metrics']:
        for node_type, metrics in history['node_type_metrics'][-1].items():
            if 'f1' in metrics:
                ax4.bar(node_type, metrics['f1'], alpha=0.7)
        ax4.set_title('Node Type Classification F1 Scores')
        ax4.set_ylabel('F1 Score')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No node type metrics available', 
                transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('Node Type Performance')
    
    # 5. Edge Type Performance
    ax5 = axes[1, 1]
    if 'edge_type_metrics' in history and history['edge_type_metrics']:
        edge_types = []
        aucs = []
        for edge_type, auc in history['edge_type_metrics'][-1].items():
            edge_types.append(f"{edge_type[0]}-{edge_type[1]}")
            aucs.append(auc)
        ax5.bar(edge_types, aucs, alpha=0.7)
        ax5.set_title('Edge Type Prediction AUC')
        ax5.set_ylabel('AUC')
        ax5.tick_params(axis='x', rotation=45)
        ax5.set_ylim(0, 1)
    else:
        ax5.text(0.5, 0.5, 'No edge type metrics available', 
                transform=ax5.transAxes, ha='center', va='center')
        ax5.set_title('Edge Type Performance')
    
    # 6. Training Summary Statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate summary statistics
    final_auc = history['val_auc'][-1] if history['val_auc'] else 0
    final_ap = history['val_ap'][-1] if history['val_ap'] else 0
    best_auc = max(history['val_auc']) if history['val_auc'] else 0
    best_ap = max(history['val_ap']) if history['val_ap'] else 0
    final_loss = history['loss'][-1] if history['loss'] else 0
    total_epochs = len(epochs)
    
    summary_text = f"""Training Summary:
    
Total Epochs: {total_epochs}
    
Final Metrics:
• AUC: {final_auc:.4f}
• AP: {final_ap:.4f}
• Loss: {final_loss:.4f}

Best Metrics:
• Best AUC: {best_auc:.4f}
• Best AP: {best_ap:.4f}

Model Type: Heterogeneous GNN
Node Types: {len(set(history.get('node_types', ['context'])))}
Edge Types: {len(set(history.get('edge_types', [('context', 'is_similar_to', 'context')])))}"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_path = os.path.join(save_dir, 'hetero_training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heterogeneous training plots saved to: {save_dir}")


@torch.no_grad()
def evaluate_hetero_model(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    data: HeteroData,
    target_edge_type: Tuple[str, str, str] = ('context', 'is_similar_to', 'context')
) -> Tuple[float, float]:
    """Evaluate heterogeneous model on link prediction task."""
    encoder.eval()
    predictor.eval()
    
    # Get node embeddings
    z_dict = encoder(data.x_dict, data.edge_index_dict)
    
    # Get positive and negative edges
    pos_edge_index = data[target_edge_type].pos_edge_label_index
    neg_edge_index = data[target_edge_type].neg_edge_label_index
    
    # Create temporary edge index dict for prediction
    temp_edge_dict = {target_edge_type: pos_edge_index}
    pos_scores = predictor(z_dict, temp_edge_dict, target_edge_type)
    
    temp_edge_dict = {target_edge_type: neg_edge_index}
    neg_scores = predictor(z_dict, temp_edge_dict, target_edge_type)
    
    # Combine scores and labels
    scores = torch.cat([pos_scores, neg_scores], dim=0).cpu()
    labels = torch.cat([
        torch.ones_like(pos_scores), 
        torch.zeros_like(neg_scores)
    ], dim=0).cpu()
    
    # Calculate metrics
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    
    return auc, ap


@torch.no_grad()
def evaluate_edge_types(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    data: HeteroData,
    edge_types: List[Tuple[str, str, str]]
) -> Dict[Tuple[str, str, str], float]:
    """Evaluate model performance on different edge types."""
    encoder.eval()
    predictor.eval()
    
    edge_type_metrics = {}
    z_dict = encoder(data.x_dict, data.edge_index_dict)
    
    for edge_type in edge_types:
        try:
            # Check if this edge type has evaluation data
            if (hasattr(data[edge_type], 'pos_edge_label_index') and 
                hasattr(data[edge_type], 'neg_edge_label_index')):
                
                auc, ap = evaluate_hetero_model(encoder, predictor, data, edge_type)
                edge_type_metrics[edge_type] = auc
            else:
                # Skip if no evaluation data available
                continue
        except Exception as e:
            print(f"Could not evaluate edge type {edge_type}: {e}")
            continue
    
    return edge_type_metrics


def hetero_cosine_similarity_loss(
    z_dict: Dict[str, torch.Tensor], 
    x_dict: Dict[str, torch.Tensor],
    node_type: str = 'context'
) -> torch.Tensor:
    """
    Pairwise cosine similarity loss to preserve semantic relationships.
    Instead of comparing embeddings directly, we compare pairwise similarities
    between embeddings to preserve the relative semantic structure.
    """
    if node_type not in z_dict or node_type not in x_dict:
        return torch.tensor(0.0, device=next(iter(z_dict.values())).device)
    
    z = z_dict[node_type]
    x = x_dict[node_type]
    
    # Normalize embeddings
    z_norm = F.normalize(z, p=2, dim=1)
    x_norm = F.normalize(x, p=2, dim=1)
    
    # Sample a subset of nodes for efficiency (to avoid O(n^2) computation)
    n_samples = min(32, z.size(0))  # Sample up to 32 nodes
    if z.size(0) > n_samples:
        indices = torch.randperm(z.size(0), device=z.device)[:n_samples]
        z_sample = z_norm[indices]
        x_sample = x_norm[indices]
    else:
        z_sample = z_norm
        x_sample = x_norm
    
    # Compute pairwise cosine similarities
    z_sim = torch.mm(z_sample, z_sample.t())  # [n_samples, n_samples]
    x_sim = torch.mm(x_sample, x_sample.t())  # [n_samples, n_samples]
    
    # Only consider off-diagonal elements (avoid self-similarity)
    mask = ~torch.eye(z_sim.size(0), dtype=torch.bool, device=z_sim.device)
    z_sim_flat = z_sim[mask]
    x_sim_flat = x_sim[mask]
    
    # Mean squared error between similarity matrices
    loss = F.mse_loss(z_sim_flat, x_sim_flat)
    
    return loss


def train_hetero_model(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_data: HeteroData,
    val_data: HeteroData,
    target_edge_type: Tuple[str, str, str] = ('context', 'is_similar_to', 'context'),
    epochs: int = 500,
    λ_semantic: float = 0.3,
    patience: int = 25,
    eval_freq: int = 5,
    use_cosine_loss: bool = True,
    min_improvement: float = 1e-4,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    Enhanced training for heterogeneous models with:
    - Multiple edge type evaluation
    - Semantic preservation loss
    - Better early stopping
    """
    
    history = {
        "epoch": [], "loss": [], "recon_loss": [], "semantic_loss": [],
        "val_auc": [], "val_ap": [], "edge_type_metrics": [],
        "node_types": list(train_data.node_types),
        "edge_types": list(train_data.edge_types)
    }
    
    best_auc = 0.0
    best_ap = 0.0
    no_improve = 0
    improvement_threshold = min_improvement
    
    print(f"Training heterogeneous model for {epochs} epochs")
    print(f"Target edge type: {target_edge_type}")
    print(f"Node types: {train_data.node_types}")
    print(f"Edge types: {train_data.edge_types}")
    
    for epoch in range(1, epochs + 1):
        encoder.train()
        predictor.train()
        optimizer.zero_grad()
        
        # Forward pass
        z_dict = encoder(train_data.x_dict, train_data.edge_index_dict)
        
        # Link prediction loss on target edge type
        pos_edge_index = train_data[target_edge_type].pos_edge_label_index
        neg_edge_index = train_data[target_edge_type].neg_edge_label_index
        
        # Positive predictions
        temp_edge_dict = {target_edge_type: pos_edge_index}
        pos_scores = predictor(z_dict, temp_edge_dict, target_edge_type)
        
        # Negative predictions  
        temp_edge_dict = {target_edge_type: neg_edge_index}
        neg_scores = predictor(z_dict, temp_edge_dict, target_edge_type)
        
        # Binary cross entropy loss
        pos_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        recon_loss = pos_loss + neg_loss
        
        # Semantic preservation loss (focus on CONTEXT nodes)
        semantic_loss = torch.tensor(0.0, device=device)
        if use_cosine_loss and 'context' in z_dict:
            semantic_loss = hetero_cosine_similarity_loss(z_dict, train_data.x_dict, 'context')
        
        # Total loss
        total_loss = recon_loss + λ_semantic * semantic_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
        )
        optimizer.step()
        
        # Evaluation and logging
        if epoch % eval_freq == 0 or epoch == 1:
            val_auc, val_ap = evaluate_hetero_model(encoder, predictor, val_data, target_edge_type)
            
            # Evaluate different edge types
            edge_type_metrics = evaluate_edge_types(
                encoder, predictor, val_data, 
                [et for et in train_data.edge_types if et == target_edge_type]
            )
            
            # Step scheduler
            if hasattr(scheduler, 'step'):
                if 'ReduceLROnPlateau' in str(type(scheduler)):
                    scheduler.step(val_auc)
                else:
                    scheduler.step()
            
            # Log metrics
            history["epoch"].append(epoch)
            history["loss"].append(total_loss.item())
            history["recon_loss"].append(recon_loss.item())
            history["semantic_loss"].append(semantic_loss.item())
            history["val_auc"].append(val_auc)
            history["val_ap"].append(val_ap)
            history["edge_type_metrics"].append(edge_type_metrics)
            
            print(
                f"Epoch {epoch:04d} | Loss: {total_loss:.4f} "
                f"(recon: {recon_loss:.4f}, semantic: {semantic_loss:.4f}) | "
                f"Val AUC: {val_auc:.4f} | AP: {val_ap:.4f}"
            )
            
            # Early stopping
            current_score = 0.7 * val_auc + 0.3 * val_ap
            improvement = current_score - (0.7 * best_auc + 0.3 * best_ap)
            
            if improvement > improvement_threshold:
                best_auc = val_auc
                best_ap = val_ap
                no_improve = 0
                print(f"  → New best score: {current_score:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping after {epoch} epochs")
                    break
    
    return history


def save_hetero_model(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    path: str,
    metadata: Dict = None
) -> None:
    """Save heterogeneous model with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "metadata": metadata or {}
    }
    
    torch.save(save_dict, path)
    print(f"Heterogeneous model saved to {path}")


def save_hetero_training_artifacts(
    history: Dict, 
    run_dir: str, 
    model_path: str,
    metadata: Dict = None
) -> None:
    """Save training artifacts for heterogeneous model."""
    
    # Save metrics
    metrics_path = os.path.join(run_dir, "hetero_training_metrics.json")
    with open(metrics_path, "w") as f:
        # Convert tensor values to regular Python types for JSON serialization
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, list):
                serializable_list = []
                for item in value:
                    if hasattr(item, 'tolist'):
                        serializable_list.append(item.tolist())
                    elif isinstance(item, dict):
                        # Handle dictionaries with tuple keys (like edge_type_metrics)
                        serializable_dict = {}
                        for k, v in item.items():
                            # Convert tuple keys to strings
                            str_key = str(k) if isinstance(k, tuple) else k
                            serializable_dict[str_key] = v.tolist() if hasattr(v, 'tolist') else v
                        serializable_list.append(serializable_dict)
                    else:
                        serializable_list.append(item)
                serializable_history[key] = serializable_list
            else:
                serializable_history[key] = value
        json.dump(serializable_history, f, indent=4)
    
    # Create plots
    create_hetero_training_plots(history, run_dir)
    
    # Save metadata
    if metadata:
        metadata_path = os.path.join(run_dir, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
    
    print(f"Heterogeneous training artifacts saved to: {run_dir}")
    print(f"  - Model: {os.path.basename(model_path)}")
    print(f"  - Metrics: hetero_training_metrics.json")
    print(f"  - Plots: hetero_training_progress.png")
    if metadata:
        print(f"  - Metadata: model_metadata.json") 