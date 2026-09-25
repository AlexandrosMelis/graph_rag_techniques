import json
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GAE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_training_plots(history: dict, save_dir: str) -> None:
    """Create comprehensive training visualization plots."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Graph Encoder Training Progress', fontsize=16, fontweight='bold')
    
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
    ax2.plot(epochs, history['recon_loss'], 'r-', linewidth=2, label='Reconstruction Loss')
    ax2.plot(epochs, history['feat_loss'], 'g-', linewidth=2, label='Feature Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')  # Log scale for better visualization
    
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
    
    # 4. Feature Weight Progression
    ax4 = axes[1, 0]
    ax4.plot(epochs, history['feat_weight'], 'brown', linewidth=2, marker='d', markersize=3)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Feature Weight (λ)')
    ax4.set_title('Adaptive Feature Weight')
    ax4.grid(True, alpha=0.3)
    
    # 5. Combined Score (AUC + AP weighted)
    ax5 = axes[1, 1]
    combined_score = [0.7 * auc + 0.3 * ap for auc, ap in zip(history['val_auc'], history['val_ap'])]
    ax5.plot(epochs, combined_score, 'darkgreen', linewidth=2, marker='v', markersize=4)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Combined Score')
    ax5.set_title('Combined Validation Score (0.7×AUC + 0.3×AP)')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)
    
    # 6. Training Summary Statistics
    ax6 = axes[1, 2]
    ax6.axis('off')  # Turn off axis for text display
    
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

Feature Weight:
• Initial: {history['feat_weight'][0]:.3f}
• Final: {history['feat_weight'][-1]:.3f}"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_path = os.path.join(save_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate detailed loss plot
    create_detailed_loss_plot(history, save_dir)
    
    # Create validation metrics comparison plot
    create_validation_comparison_plot(history, save_dir)
    
    print(f"Training plots saved to: {save_dir}")


def create_detailed_loss_plot(history: dict, save_dir: str) -> None:
    """Create detailed loss analysis plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Detailed Loss Analysis', fontsize=14, fontweight='bold')
    
    epochs = history['epoch']
    
    # Left plot: All losses together
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Total Loss', alpha=0.8)
    ax1.plot(epochs, history['recon_loss'], 'r-', linewidth=2, label='Reconstruction Loss', alpha=0.8)
    
    # Calculate weighted feature loss for visualization
    weighted_feat_loss = [f_loss * f_weight for f_loss, f_weight in 
                         zip(history['feat_loss'], history['feat_weight'])]
    ax1.plot(epochs, weighted_feat_loss, 'g-', linewidth=2, label='Weighted Feature Loss', alpha=0.8)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Components Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Loss ratios
    recon_ratio = [r_loss / total_loss if total_loss > 0 else 0 
                   for r_loss, total_loss in zip(history['recon_loss'], history['loss'])]
    feat_ratio = [w_f_loss / total_loss if total_loss > 0 else 0 
                  for w_f_loss, total_loss in zip(weighted_feat_loss, history['loss'])]
    
    ax2.plot(epochs, recon_ratio, 'r-', linewidth=2, label='Reconstruction %', alpha=0.8)
    ax2.plot(epochs, feat_ratio, 'g-', linewidth=2, label='Feature %', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Contribution Ratio')
    ax2.set_title('Loss Component Ratios')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save detailed loss plot
    loss_plot_path = os.path.join(save_dir, 'detailed_loss_analysis.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_validation_comparison_plot(history: dict, save_dir: str) -> None:
    """Create validation metrics comparison and improvement tracking."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Validation Metrics Analysis', fontsize=14, fontweight='bold')
    
    epochs = history['epoch']
    
    # Left plot: Validation metrics with trend lines
    ax1.plot(epochs, history['val_auc'], 'purple', linewidth=2, marker='o', 
             markersize=4, label='AUC', alpha=0.8)
    ax1.plot(epochs, history['val_ap'], 'orange', linewidth=2, marker='s', 
             markersize=4, label='AP', alpha=0.8)
    
    # Add trend lines
    if len(epochs) > 1:
        auc_trend = np.polyfit(epochs, history['val_auc'], 1)
        ap_trend = np.polyfit(epochs, history['val_ap'], 1)
        
        ax1.plot(epochs, np.polyval(auc_trend, epochs), 'purple', 
                linestyle='--', alpha=0.5, label='AUC Trend')
        ax1.plot(epochs, np.polyval(ap_trend, epochs), 'orange', 
                linestyle='--', alpha=0.5, label='AP Trend')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title('Validation Metrics with Trends')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Right plot: Improvement tracking
    auc_improvements = [0] + [curr - prev for curr, prev in 
                             zip(history['val_auc'][1:], history['val_auc'][:-1])]
    ap_improvements = [0] + [curr - prev for curr, prev in 
                            zip(history['val_ap'][1:], history['val_ap'][:-1])]
    
    ax2.bar([e - 0.2 for e in epochs], auc_improvements, width=0.4, 
            label='AUC Improvement', alpha=0.7, color='purple')
    ax2.bar([e + 0.2 for e in epochs], ap_improvements, width=0.4, 
            label='AP Improvement', alpha=0.7, color='orange')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score Improvement')
    ax2.set_title('Per-Epoch Metric Improvements')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save validation comparison plot
    val_plot_path = os.path.join(save_dir, 'validation_analysis.png')
    plt.savefig(val_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def evaluate(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    data: Data,
) -> Tuple[float, float]:
    encoder.eval()
    predictor.eval()
    z = encoder(data.x, data.edge_index)
    pos_score = predictor(z, data.pos_edge_label_index)
    neg_score = predictor(z, data.neg_edge_label_index)
    scores = torch.cat([pos_score, neg_score], dim=0).cpu()
    labels = torch.cat(
        [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
    ).cpu()
    from sklearn.metrics import average_precision_score, roc_auc_score

    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def adaptive_loss_weights(epoch: int, initial_feat_weight: float = 0.5) -> float:
    """Adaptively adjust feature reconstruction weight during training."""
    # Start with higher feature preservation, gradually reduce to allow more graph learning
    decay_factor = 0.95 ** (epoch // 10)  # Decay every 10 epochs
    min_weight = 0.1  # Minimum weight to maintain some feature preservation
    return max(initial_feat_weight * decay_factor, min_weight)


def cosine_similarity_loss(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss to preserve semantic relationships while allowing adaptation."""
    # Normalize embeddings
    z_norm = F.normalize(z, p=2, dim=1)
    x_norm = F.normalize(x, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(z_norm * x_norm, dim=1)
    
    # Loss is 1 - average cosine similarity (encourages high similarity)
    return 1 - cosine_sim.mean()


def train_model(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_data: Data,
    val_data: Data,
    epochs: int = 500,
    λ_feat: float = 0.5,
    patience: int = 25,
    eval_freq: int = 5,
    use_adaptive_weights: bool = True,
    use_cosine_loss: bool = True,
    min_improvement: float = 1e-4,
) -> dict:
    """
    Enhanced training with:
    - More frequent evaluation
    - Adaptive loss balancing
    - Better early stopping criteria
    - Cosine similarity loss for semantic preservation
    """
    history = {
        "epoch": [], "loss": [], "recon_loss": [], "feat_loss": [], 
        "val_auc": [], "val_ap": [], "feat_weight": []
    }
    best_auc = 0.0
    best_ap = 0.0
    no_improve = 0
    improvement_threshold = min_improvement

    print(f"Training for {epochs} epochs with eval every {eval_freq} epochs, patience={patience}")
    
    for epoch in range(1, epochs + 1):
        encoder.train()
        predictor.train()
        optimizer.zero_grad()

        # Forward pass
        z = encoder(train_data.x, train_data.edge_index)
        pos_score = predictor(z, train_data.pos_edge_label_index)
        neg_score = predictor(z, train_data.neg_edge_label_index)

        # Reconstruction loss (link prediction)
        edge_labels = torch.cat(
            [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
        )
        edge_preds = torch.cat([pos_score, neg_score], dim=0)
        recon_loss = F.binary_cross_entropy(edge_preds, edge_labels)
        
        # Feature preservation loss
        if use_cosine_loss:
            feat_loss = cosine_similarity_loss(z, train_data.x)
        else:
            feat_loss = F.mse_loss(z, train_data.x)
        
        # Adaptive loss weighting
        if use_adaptive_weights:
            current_feat_weight = adaptive_loss_weights(epoch, λ_feat)
        else:
            current_feat_weight = λ_feat
            
        total_loss = recon_loss + current_feat_weight * feat_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
        )
        optimizer.step()

        # Evaluation and logging
        if epoch % eval_freq == 0 or epoch == 1:
            val_auc, val_ap = evaluate(encoder, predictor, val_data)
            
            # Step scheduler with validation AUC
            if hasattr(scheduler, 'step'):
                if 'ReduceLROnPlateau' in str(type(scheduler)):
                    scheduler.step(val_auc)
                else:
                    scheduler.step()
            
            # Log metrics
            history["epoch"].append(epoch)
            history["loss"].append(total_loss.item())
            history["recon_loss"].append(recon_loss.item())
            history["feat_loss"].append(feat_loss.item())
            history["val_auc"].append(val_auc)
            history["val_ap"].append(val_ap)
            history["feat_weight"].append(current_feat_weight)
            
            print(
                f"Epoch {epoch:04d} | Loss: {total_loss:.4f} (recon: {recon_loss:.4f}, "
                f"feat: {feat_loss:.4f}, λ: {current_feat_weight:.3f}) | "
                f"Val AUC: {val_auc:.4f} | VAP: {val_ap:.4f}"
            )

            # Enhanced early stopping with multiple criteria
            current_score = 0.7 * val_auc + 0.3 * val_ap  # Combined metric
            improvement = current_score - (0.7 * best_auc + 0.3 * best_ap)
            
            if improvement > improvement_threshold:
                best_auc = val_auc
                best_ap = val_ap
                no_improve = 0
                print(f"  → New best score: {current_score:.4f} (improvement: {improvement:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"No significant improvement for {patience} evaluations ({no_improve * eval_freq} epochs), stopping early.")
                    print(f"Best AUC: {best_auc:.4f}, Best AP: {best_ap:.4f}")
                    break
            
            # Adaptive patience: reduce patience requirement if we're making slow progress
            if epoch > 100 and val_auc > 0.6:
                improvement_threshold = max(min_improvement * 0.5, 1e-5)

    return history


def save_model(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"encoder": encoder.state_dict(), "predictor": predictor.state_dict()}, path
    )
    print(f"Model saved to {path}.")


def save_metrics(metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {path}.")


def save_training_artifacts(history: dict, run_dir: str, model_path: str) -> None:
    """Save both metrics and training plots."""
    # Save metrics
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    save_metrics(history, metrics_path)
    
    # Create and save plots
    create_training_plots(history, run_dir)
    
    print(f"Training artifacts saved to: {run_dir}")
    print(f"  - Model: {os.path.basename(model_path)}")
    print(f"  - Metrics: training_metrics.json")
    print(f"  - Plots: training_progress.png, detailed_loss_analysis.png, validation_analysis.png")
