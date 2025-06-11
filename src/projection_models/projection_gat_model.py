import os
import random
import time
from typing import List, Tuple, Dict, Any, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from neo4j import GraphDatabase
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1) DATA PROCESSING FOR GAT SUBGRAPHS
# -----------------------------------------------------------------------------

class GATDataProcessor:
    """
    Data processor for GAT model that retrieves subgraphs around each question.
    For each QA_PAIR, retrieves:
    1. Connected CONTEXT nodes (positive contexts)
    2. Similar CONTEXT nodes via IS_SIMILAR_TO relationships  
    3. Builds subgraph with connectivity information
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password), database=database
        )
        self.database = database
    
    def fetch_qa_with_contexts(self) -> pd.DataFrame:
        """
        Fetch QA_PAIR nodes with their connected CONTEXT nodes and graph embeddings.
        Returns DataFrame with columns: qa_id, q_emb, ctx_ids, ctx_embs, ctx_graph_embs
        """
        query = """
        MATCH (qa:QA_PAIR)-[:HAS_CONTEXT]->(ctx:CONTEXT)
        WITH qa, collect(id(ctx)) AS ctx_ids, 
             collect(ctx.embedding) AS ctx_embs,
             collect(ctx.graph_embedding) AS ctx_graph_embs
        RETURN qa.id AS qa_id,
               qa.embedding AS q_emb,
               ctx_ids,
               ctx_embs, 
               ctx_graph_embs
        """
        
        rows = []
        with self.driver.session() as sess:
            for rec in sess.run(query):
                rows.append(dict(rec))
        
        return pd.DataFrame(rows)
    
    def fetch_context_neighborhoods(self, ctx_ids: List[int], max_neighbors: int = 20) -> Dict[int, Dict]:
        """
        For given context IDs, fetch their neighborhoods via IS_SIMILAR_TO relationships.
        Returns dict mapping ctx_id -> {neighbors: List[int], neighbor_embs: List[List[float]], 
                                       scores: List[float]}
        """
        if not ctx_ids:
            return {}
            
        query = """
        UNWIND $ctx_ids AS ctx_id
        MATCH (ctx:CONTEXT)-[r:IS_SIMILAR_TO]->(nbr:CONTEXT)
        WHERE id(ctx) = ctx_id
        WITH ctx_id, collect({id: id(nbr), emb: nbr.embedding, score: r.score}) AS neighbors
        RETURN ctx_id, neighbors[0..coalesce($max_neighbors, 20)] AS limited_neighbors
        """
        
        neighborhoods = {}
        with self.driver.session() as sess:
            for rec in sess.run(query, ctx_ids=ctx_ids, max_neighbors=max_neighbors):
                ctx_id = rec["ctx_id"]
                neighbors_data = rec["limited_neighbors"] or []
                
                neighborhoods[ctx_id] = {
                    "neighbors": [n["id"] for n in neighbors_data],
                    "neighbor_embs": [n["emb"] for n in neighbors_data],
                    "scores": [n["score"] for n in neighbors_data]
                }
        
        return neighborhoods
    
    def build_subgraph_data(self, qa_row: pd.Series, max_neighbors: int = 20) -> Dict[str, Any]:
        """
        Build subgraph data for a single QA pair.
        Returns dict with node features, adjacency info, and target embeddings.
        """
        qa_id = qa_row["qa_id"]
        q_emb = qa_row["q_emb"]
        ctx_ids = qa_row["ctx_ids"]
        ctx_embs = qa_row["ctx_embs"]
        ctx_graph_embs = qa_row["ctx_graph_embs"]
        
        # Get neighborhoods for context nodes
        neighborhoods = self.fetch_context_neighborhoods(ctx_ids, max_neighbors)
        
        # Build node list: query + contexts + neighbors
        node_features = [q_emb]  # Query node is first
        node_ids = [-1]  # Query has special ID -1
        
        # Add context nodes
        for i, (ctx_id, ctx_emb) in enumerate(zip(ctx_ids, ctx_embs)):
            node_features.append(ctx_emb)
            node_ids.append(ctx_id)
        
        # Add neighbor nodes (avoiding duplicates)
        seen_neighbors = set(ctx_ids)
        neighbor_map = {}  # Maps neighbor_id -> index in node_features
        
        for ctx_id in ctx_ids:
            if ctx_id in neighborhoods:
                for nbr_id, nbr_emb in zip(neighborhoods[ctx_id]["neighbors"], 
                                          neighborhoods[ctx_id]["neighbor_embs"]):
                    if nbr_id not in seen_neighbors:
                        node_features.append(nbr_emb)
                        node_ids.append(nbr_id)
                        neighbor_map[nbr_id] = len(node_features) - 1
                        seen_neighbors.add(nbr_id)
        
        # Build adjacency matrix (edges between contexts and their neighbors)
        num_nodes = len(node_features)
        adjacency = torch.zeros(num_nodes, num_nodes)
        edge_indices = []
        edge_weights = []
        
        # Add edges: query -> contexts (bidirectional)
        for i in range(1, len(ctx_ids) + 1):  # Context nodes start at index 1
            edge_indices.extend([[0, i], [i, 0]])
            edge_weights.extend([1.0, 1.0])  # Query-context edges have weight 1.0
            adjacency[0, i] = 1.0
            adjacency[i, 0] = 1.0
        
        # Add edges: contexts -> neighbors
        for i, ctx_id in enumerate(ctx_ids, 1):  # Start from index 1
            if ctx_id in neighborhoods:
                for nbr_id, score in zip(neighborhoods[ctx_id]["neighbors"], 
                                       neighborhoods[ctx_id]["scores"]):
                    if nbr_id in neighbor_map:
                        nbr_idx = neighbor_map[nbr_id]
                        edge_indices.extend([[i, nbr_idx], [nbr_idx, i]])
                        edge_weights.extend([score, score])
                        adjacency[i, nbr_idx] = score
                        adjacency[nbr_idx, i] = score
        
        # Target: average of context graph embeddings (what we want to predict)
        target_emb = np.mean(ctx_graph_embs, axis=0) if ctx_graph_embs else np.zeros(len(ctx_graph_embs[0]))
        
        return {
            "qa_id": qa_id,
            "node_features": torch.tensor(node_features, dtype=torch.float32),
            "adjacency": adjacency,
            "edge_indices": torch.tensor(edge_indices, dtype=torch.long).t() if edge_indices else torch.empty((2, 0), dtype=torch.long),
            "edge_weights": torch.tensor(edge_weights, dtype=torch.float32),
            "target_embedding": torch.tensor(target_emb, dtype=torch.float32),
            "context_indices": list(range(1, len(ctx_ids) + 1))  # Indices of context nodes
        }


class GATSubgraphDataset(Dataset):
    """
    Dataset for GAT training. Each item is a subgraph around a question.
    """
    
    def __init__(self, qa_df: pd.DataFrame, data_processor: GATDataProcessor, max_neighbors: int = 20):
        self.qa_df = qa_df
        self.data_processor = data_processor
        self.max_neighbors = max_neighbors
        
        # Pre-build all subgraphs (could be memory intensive for large datasets)
        print("Building subgraph data...")
        self.subgraphs = []
        for _, row in tqdm(qa_df.iterrows(), total=len(qa_df)):
            try:
                subgraph = data_processor.build_subgraph_data(row, max_neighbors)
                self.subgraphs.append(subgraph)
            except Exception as e:
                print(f"Error building subgraph for QA {row['qa_id']}: {e}")
                continue
    
    def __len__(self):
        return len(self.subgraphs)
    
    def __getitem__(self, idx):
        return self.subgraphs[idx]


# -----------------------------------------------------------------------------
# 2) GAT MODEL ARCHITECTURE
# -----------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    Single Graph Attention Layer.
    """
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 8, 
                 dropout: float = 0.6, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Linear transformations for each head
        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False) for _ in range(n_heads)
        ])
        
        # Attention mechanism
        self.a = nn.ModuleList([
            nn.Linear(2 * out_features, 1, bias=False) for _ in range(n_heads)
        ])
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        Returns:
            Updated node features [N, out_features]
        """
        batch_size, n_nodes = x.size(0), x.size(1)
        
        # Multi-head attention
        head_outputs = []
        
        for i in range(self.n_heads):
            # Linear transformation
            h = self.W[i](x)  # [batch, N, out_features]
            
            # Compute attention coefficients
            a_input = self._prepare_attentional_mechanism_input(h)  # [batch, N, N, 2*out_features]
            e = self.leakyrelu(self.a[i](a_input).squeeze(-1))  # [batch, N, N]
            
            # Mask with adjacency matrix
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=-1)
            attention = self.dropout_layer(attention)
            
            # Apply attention to features
            h_prime = torch.bmm(attention, h)  # [batch, N, out_features]
            head_outputs.append(h_prime)
        
        # Concatenate or average heads
        if len(head_outputs) > 1:
            output = torch.cat(head_outputs, dim=-1)  # [batch, N, n_heads*out_features]
        else:
            output = head_outputs[0]
            
        return output
    
    def _prepare_attentional_mechanism_input(self, h: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for attention mechanism by creating all pairwise combinations.
        """
        batch_size, n_nodes, features = h.size()
        
        # Repeat for all pairs
        h_i = h.unsqueeze(2).repeat(1, 1, n_nodes, 1)  # [batch, N, N, features]
        h_j = h.unsqueeze(1).repeat(1, n_nodes, 1, 1)  # [batch, N, N, features]
        
        # Concatenate
        return torch.cat([h_i, h_j], dim=-1)  # [batch, N, N, 2*features]


class GraphQueryProjectionGAT(nn.Module):
    """
    GAT-based model for projecting queries into graph embedding space.
    
    Architecture:
    1. Multi-layer GAT to encode the subgraph
    2. Query-aware attention pooling
    3. Final projection to graph embedding space
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 n_layers: int = 2,
                 n_heads: int = 8,
                 dropout: float = 0.6):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATLayer(input_dim, hidden_dim, n_heads, dropout)
        )
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.gat_layers.append(
                GATLayer(hidden_dim * n_heads, hidden_dim, n_heads, dropout)
            )
        
        # Final GAT layer (single head for clean output)
        if n_layers > 1:
            self.gat_layers.append(
                GATLayer(hidden_dim * n_heads, hidden_dim, 1, dropout)
            )
        
        # Query-aware attention pooling
        self.query_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection layers
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor, 
                context_indices: List[List[int]]) -> torch.Tensor:
        """
        Args:
            node_features: [batch, max_nodes, input_dim]
            adjacency: [batch, max_nodes, max_nodes]
            context_indices: List of lists containing context node indices for each sample
            
        Returns:
            projected_embeddings: [batch, output_dim]
        """
        batch_size = node_features.size(0)
        
        # Apply GAT layers
        h = node_features
        for gat_layer in self.gat_layers:
            h = gat_layer(h, adjacency)
            h = F.elu(h)
        
        # Query-aware pooling
        # Use query node (index 0) as query for attention over context nodes
        pooled_embeddings = []
        
        for i in range(batch_size):
            query_emb = h[i, 0:1, :]  # Query node embedding [1, hidden_dim]
            
            # Get context node embeddings
            ctx_indices = context_indices[i]
            if ctx_indices:
                context_embs = h[i, ctx_indices, :]  # [n_contexts, hidden_dim]
                
                # Apply attention with query as query, contexts as key/value
                attended_emb, _ = self.query_attention(
                    query=query_emb,
                    key=context_embs,
                    value=context_embs
                )
                pooled_embeddings.append(attended_emb.squeeze(0))  # [hidden_dim]
            else:
                # Fallback: use query embedding directly
                pooled_embeddings.append(query_emb.squeeze(0))
        
        pooled_embeddings = torch.stack(pooled_embeddings)  # [batch, hidden_dim]
        
        # Final projection
        output = self.projection(pooled_embeddings)
        
        return output


# -----------------------------------------------------------------------------
# 3) LOSS FUNCTIONS
# -----------------------------------------------------------------------------

def cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss for directional alignment."""
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    return 1 - cos_sim.mean()  # Minimize distance, maximize similarity

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return F.mse_loss(pred, target)

def combined_loss(pred: torch.Tensor, target: torch.Tensor, 
                  alpha: float = 0.5, beta: float = 0.5) -> torch.Tensor:
    """Combined cosine similarity and MSE loss."""
    cos_loss = cosine_similarity_loss(pred, target)
    mse_loss_val = mse_loss(pred, target)
    return alpha * cos_loss + beta * mse_loss_val


# -----------------------------------------------------------------------------
# 4) TRAINING UTILITIES
# -----------------------------------------------------------------------------

def collate_subgraphs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader to handle variable-size subgraphs.
    Pads to maximum size in batch.
    """
    batch_size = len(batch)
    
    # Find maximum dimensions
    max_nodes = max(item["node_features"].size(0) for item in batch)
    feature_dim = batch[0]["node_features"].size(1)
    output_dim = batch[0]["target_embedding"].size(0)
    
    # Initialize padded tensors
    padded_features = torch.zeros(batch_size, max_nodes, feature_dim)
    padded_adjacency = torch.zeros(batch_size, max_nodes, max_nodes)
    targets = torch.zeros(batch_size, output_dim)
    context_indices_batch = []
    qa_ids = []
    
    # Fill padded tensors
    for i, item in enumerate(batch):
        n_nodes = item["node_features"].size(0)
        padded_features[i, :n_nodes, :] = item["node_features"]
        padded_adjacency[i, :n_nodes, :n_nodes] = item["adjacency"]
        targets[i] = item["target_embedding"]
        context_indices_batch.append(item["context_indices"])
        qa_ids.append(item["qa_id"])
    
    return {
        "node_features": padded_features,
        "adjacency": padded_adjacency,
        "targets": targets,
        "context_indices": context_indices_batch,
        "qa_ids": qa_ids
    }


def plot_training_history(train_losses: List[float], val_losses: List[float], 
                         save_path: str) -> None:
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAT Model Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comprehensive_training_plots(
    train_losses: List[float],
    val_losses: List[float],
    learning_rates: List[float],
    model_dir: str,
    training_config: Dict[str, Any],
    eval_metrics: Optional[Dict[str, List[float]]] = None,
    training_times: Optional[List[float]] = None,
    best_epoch: Optional[int] = None
) -> None:
    """
    Create comprehensive training progression plots and save them to model directory.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch  
        learning_rates: List of learning rates per epoch
        model_dir: Directory to save plots
        training_config: Training configuration dictionary
        eval_metrics: Optional dictionary of evaluation metrics over epochs
        training_times: Optional list of training times per epoch
        best_epoch: Optional epoch number with best validation loss
    """
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5
    })
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Define colors
    colors = {
        'train': '#1f77b4',
        'val': '#ff7f0e', 
        'lr': '#2ca02c',
        'best': '#d62728',
        'grid': '#cccccc'
    }
    
    # 1. Loss Curves (Main plot)
    ax1 = plt.subplot(2, 3, (1, 2))
    ax1.plot(epochs, train_losses, color=colors['train'], label='Training Loss', linewidth=3)
    ax1.plot(epochs, val_losses, color=colors['val'], label='Validation Loss', linewidth=3)
    
    # Mark best epoch
    if best_epoch is not None and best_epoch <= len(val_losses):
        ax1.axvline(x=best_epoch, color=colors['best'], linestyle='--', 
                   label=f'Best Epoch ({best_epoch})', alpha=0.8)
        ax1.scatter([best_epoch], [val_losses[best_epoch-1]], 
                   color=colors['best'], s=100, zorder=5)
    
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Training & Validation Loss Progression', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(epochs))
    
    # Add loss statistics
    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    
    stats_text = f'Min Train: {min_train_loss:.4f}\nMin Val: {min_val_loss:.4f}\n'
    stats_text += f'Final Train: {final_train_loss:.4f}\nFinal Val: {final_val_loss:.4f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Learning Rate Schedule
    ax2 = plt.subplot(2, 3, 3)
    ax2.plot(epochs, learning_rates, color=colors['lr'], linewidth=3)
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
    ax2.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(epochs))
    ax2.set_yscale('log')
    
    # Add LR statistics
    initial_lr = learning_rates[0]
    final_lr = learning_rates[-1]
    min_lr = min(learning_rates)
    
    lr_text = f'Initial: {initial_lr:.2e}\nFinal: {final_lr:.2e}\nMin: {min_lr:.2e}'
    ax2.text(0.02, 0.98, lr_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Loss Convergence Analysis
    ax3 = plt.subplot(2, 3, 4)
    
    # Calculate moving averages for smoothing
    window_size = max(1, len(train_losses) // 10)
    if len(train_losses) >= window_size:
        train_smooth = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
        smooth_epochs = range(window_size, len(train_losses) + 1)
        
        ax3.plot(smooth_epochs, train_smooth, color=colors['train'], label='Train (Smoothed)', alpha=0.8)
        ax3.plot(smooth_epochs, val_smooth, color=colors['val'], label='Val (Smoothed)', alpha=0.8)
    
    ax3.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Smoothed Loss', fontsize=14, fontweight='bold')
    ax3.set_title('Loss Convergence (Smoothed)', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Training Efficiency Metrics
    ax4 = plt.subplot(2, 3, 5)
    
    if training_times is not None and len(training_times) > 0:
        # Plot training time per epoch
        ax4.plot(epochs[:len(training_times)], training_times, color='purple', linewidth=3)
        ax4.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Training Time (seconds)', fontsize=14, fontweight='bold')
        ax4.set_title('Training Time per Epoch', fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add timing statistics
        avg_time = np.mean(training_times)
        total_time = sum(training_times)
        time_text = f'Avg: {avg_time:.1f}s\nTotal: {total_time/3600:.1f}h'
        ax4.text(0.02, 0.98, time_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Plot loss difference (overfitting indicator)
        loss_diff = [val - train for val, train in zip(val_losses, train_losses)]
        ax4.plot(epochs, loss_diff, color='red', linewidth=3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Val Loss - Train Loss', fontsize=14, fontweight='bold')
        ax4.set_title('Overfitting Indicator', fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add overfitting analysis
        final_diff = loss_diff[-1]
        max_diff = max(loss_diff)
        diff_text = f'Final Diff: {final_diff:.4f}\nMax Diff: {max_diff:.4f}'
        ax4.text(0.02, 0.98, diff_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Model Configuration & Summary
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')
    
    # Create configuration summary
    config_text = "Training Configuration:\n"
    config_text += "=" * 25 + "\n"
    
    key_configs = [
        ('Batch Size', training_config.get('batch_size', 'N/A')),
        ('Learning Rate', f"{training_config.get('lr', 'N/A'):.1e}"),
        ('Hidden Dim', training_config.get('hidden_dim', 'N/A')),
        ('GAT Layers', training_config.get('n_layers', 'N/A')),
        ('Attention Heads', training_config.get('n_heads', 'N/A')),
        ('Dropout', training_config.get('dropout', 'N/A')),
        ('Max Neighbors', training_config.get('max_neighbors', 'N/A')),
        ('Patience', training_config.get('patience', 'N/A'))
    ]
    
    for key, value in key_configs:
        config_text += f"{key:<15}: {value}\n"
    
    config_text += "\n" + "Training Summary:\n"
    config_text += "=" * 25 + "\n"
    config_text += f"{'Total Epochs':<15}: {len(train_losses)}\n"
    config_text += f"{'Best Epoch':<15}: {best_epoch or 'N/A'}\n"
    config_text += f"{'Final Train Loss':<15}: {train_losses[-1]:.4f}\n"
    config_text += f"{'Final Val Loss':<15}: {val_losses[-1]:.4f}\n"
    config_text += f"{'Best Val Loss':<15}: {min(val_losses):.4f}\n"
    
    # Calculate improvement
    if len(train_losses) > 1:
        improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100
        config_text += f"{'Loss Reduction':<15}: {improvement:.1f}%\n"
    
    ax5.text(0.05, 0.95, config_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Add title to the entire figure
    fig.suptitle('GAT Model Training Progression Report', fontsize=20, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # Save the comprehensive plot
    comprehensive_path = os.path.join(model_dir, "training_progression_report.png")
    plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create a separate detailed loss plot
    create_detailed_loss_plot(train_losses, val_losses, model_dir, best_epoch)
    
    # Create evaluation metrics plot if available
    if eval_metrics:
        create_evaluation_metrics_plot(eval_metrics, model_dir)
    
    print(f"📊 Training progression plots saved:")
    print(f"   - Comprehensive report: {comprehensive_path}")
    print(f"   - Detailed loss plot: {os.path.join(model_dir, 'detailed_loss_curves.png')}")
    if eval_metrics:
        print(f"   - Evaluation metrics: {os.path.join(model_dir, 'evaluation_metrics.png')}")


def create_detailed_loss_plot(
    train_losses: List[float], 
    val_losses: List[float], 
    model_dir: str,
    best_epoch: Optional[int] = None
) -> None:
    """Create a detailed, publication-ready loss plot."""
    
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses with different styles
    plt.plot(epochs, train_losses, 'o-', color='#1f77b4', label='Training Loss', 
             linewidth=3, markersize=4, alpha=0.8)
    plt.plot(epochs, val_losses, 's-', color='#ff7f0e', label='Validation Loss', 
             linewidth=3, markersize=4, alpha=0.8)
    
    # Mark best epoch
    if best_epoch is not None and best_epoch <= len(val_losses):
        plt.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2,
                   label=f'Best Model (Epoch {best_epoch})', alpha=0.7)
        plt.scatter([best_epoch], [val_losses[best_epoch-1]], 
                   color='red', s=150, zorder=5, marker='*')
    
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('Loss', fontsize=16, fontweight='bold')
    plt.title('Training and Validation Loss Curves', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add statistics box
    stats_text = f'Training Loss:\n  Initial: {train_losses[0]:.4f}\n  Final: {train_losses[-1]:.4f}\n  Min: {min(train_losses):.4f}\n\n'
    stats_text += f'Validation Loss:\n  Initial: {val_losses[0]:.4f}\n  Final: {val_losses[-1]:.4f}\n  Min: {min(val_losses):.4f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Set axis limits with some padding
    plt.xlim(0.5, len(epochs) + 0.5)
    y_min = min(min(train_losses), min(val_losses)) * 0.95
    y_max = max(max(train_losses), max(val_losses)) * 1.05
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save detailed plot
    detailed_path = os.path.join(model_dir, "detailed_loss_curves.png")
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_evaluation_metrics_plot(eval_metrics: Dict[str, List[float]], model_dir: str) -> None:
    """Create evaluation metrics progression plot."""
    
    if not eval_metrics:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (metric, values) in enumerate(eval_metrics.items()):
        if i >= 4:  # Max 4 metrics in subplot grid
            break
            
        ax = axes[i]
        epochs = range(1, len(values) + 1)
        
        ax.plot(epochs, values, 'o-', color=colors[i % len(colors)], 
               linewidth=3, markersize=6, label=metric)
        
        ax.set_xlabel('Evaluation Point', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Progression', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add final value annotation
        if values:
            final_val = values[-1]
            ax.annotate(f'Final: {final_val:.4f}', 
                       xy=(len(values), final_val), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       fontsize=10)
    
    # Hide unused subplots
    for i in range(len(eval_metrics), 4):
        axes[i].set_visible(False)
    
    plt.suptitle('Evaluation Metrics Progression', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save evaluation plot
    eval_path = os.path.join(model_dir, "evaluation_metrics.png")
    plt.savefig(eval_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# -----------------------------------------------------------------------------
# 5) TRAINING FUNCTION
# -----------------------------------------------------------------------------

def create_optimized_val_loader(
    val_dataset: List[Dict],
    batch_size: int = 8,  # Smaller batches for validation
    num_workers: int = 2,
    fast_mode: bool = True
) -> DataLoader:
    """Create optimized validation DataLoader with performance improvements."""
    
    if fast_mode:
        # Limit subgraph size for faster validation
        optimized_dataset = []
        for item in val_dataset:
            # Limit nodes to reduce computational complexity
            max_val_nodes = min(30, item["node_features"].size(0))
            optimized_item = {
                "qa_id": item["qa_id"],
                "node_features": item["node_features"][:max_val_nodes],
                "adjacency": item["adjacency"][:max_val_nodes, :max_val_nodes],
                "target_embedding": item["target_embedding"],
                "context_indices": [i for i in item["context_indices"] if i < max_val_nodes]
            }
            optimized_dataset.append(optimized_item)
        val_dataset = optimized_dataset
    
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_subgraphs_optimized,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )


def collate_subgraphs_optimized(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Optimized collate function with better memory efficiency.
    Uses smaller padding and sparse adjacency when beneficial.
    """
    batch_size = len(batch)
    
    # Find dimensions with adaptive padding
    node_counts = [item["node_features"].size(0) for item in batch]
    max_nodes = max(node_counts)
    avg_nodes = sum(node_counts) / len(node_counts)
    
    # Use adaptive padding: if variance is high, use smaller max size
    if max_nodes > avg_nodes * 2:
        # Cap maximum padding to reduce memory waste
        max_nodes = min(max_nodes, int(avg_nodes * 1.5) + 10)
    
    feature_dim = batch[0]["node_features"].size(1)
    output_dim = batch[0]["target_embedding"].size(0)
    
    # Initialize padded tensors
    padded_features = torch.zeros(batch_size, max_nodes, feature_dim)
    padded_adjacency = torch.zeros(batch_size, max_nodes, max_nodes)
    targets = torch.zeros(batch_size, output_dim)
    context_indices_batch = []
    qa_ids = []
    
    # Fill padded tensors
    for i, item in enumerate(batch):
        n_nodes = min(item["node_features"].size(0), max_nodes)
        padded_features[i, :n_nodes, :] = item["node_features"][:n_nodes]
        padded_adjacency[i, :n_nodes, :n_nodes] = item["adjacency"][:n_nodes, :n_nodes]
        targets[i] = item["target_embedding"]
        # Adjust context indices for truncated graphs
        adjusted_indices = [idx for idx in item["context_indices"] if idx < n_nodes]
        context_indices_batch.append(adjusted_indices)
        qa_ids.append(item["qa_id"])
    
    return {
        "node_features": padded_features,
        "adjacency": padded_adjacency,
        "targets": targets,
        "context_indices": context_indices_batch,
        "qa_ids": qa_ids
    }


def fast_validation_forward(
    model: 'GraphQueryProjectionGAT',
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    context_indices: List[List[int]]
) -> torch.Tensor:
    """
    Optimized forward pass for validation with reduced computational overhead.
    """
    batch_size = node_features.size(0)
    device = node_features.device
    
    # Apply GAT layers (same as original)
    h = node_features
    for gat_layer in model.gat_layers:
        h = gat_layer(h, adjacency)
        h = F.elu(h)
    
    # Optimized query-aware pooling - batch processing when possible
    pooled_embeddings = []
    
    # Group samples with similar context sizes for efficient batch processing
    query_embeddings = h[:, 0, :]  # All query embeddings [batch, hidden_dim]
    
    for i in range(batch_size):
        ctx_indices = context_indices[i]
        if ctx_indices and len(ctx_indices) > 0:
            try:
                context_embs = h[i, ctx_indices, :]  # [n_contexts, hidden_dim]
                query_emb = query_embeddings[i:i+1, :]  # [1, hidden_dim]
                
                # Ensure tensors are on same device
                context_embs = context_embs.to(device)
                query_emb = query_emb.to(device)
                
                # Use faster attention computation for validation
                attended_emb, _ = model.query_attention(
                    query=query_emb,
                    key=context_embs,
                    value=context_embs
                )
                pooled_embeddings.append(attended_emb.squeeze(0))
            except Exception as e:
                # Fallback on error: use query embedding directly
                pooled_embeddings.append(query_embeddings[i])
        else:
            # Fallback: use query embedding directly
            pooled_embeddings.append(query_embeddings[i])
    
    pooled_embeddings = torch.stack(pooled_embeddings)  # [batch, hidden_dim]
    
    # Final projection
    output = model.projection(pooled_embeddings)
    return output


def train_gat_projection(
    uri: str,
    user: str, 
    password: str,
    database: str,
    model_dir: str,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 300,
    val_ratio: float = 0.1,
    max_neighbors: int = 20,
    hidden_dim: int = 512,
    n_layers: int = 2,
    n_heads: int = 8,
    dropout: float = 0.6,
    patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_callback: Optional[callable] = None,
    eval_frequency: int = 10,
    eval_data: Optional[Tuple[Any, pd.DataFrame, Dict[str, List[float]]]] = None,
    fast_validation: bool = True  # New parameter for fast validation
) -> GraphQueryProjectionGAT:
    """
    Train the GAT-based projection model with optimized validation.
    """
    
    print(f"Training GAT model on {device}")
    print(f"Model will be saved in: {model_dir}")
    if fast_validation:
        print("🚀 Fast validation mode enabled")
    
    # 1) Load data
    data_processor = GATDataProcessor(uri, user, password, database)
    qa_df = data_processor.fetch_qa_with_contexts()
    
    if len(qa_df) == 0:
        raise ValueError("No QA pairs found in database!")
    
    print(f"Found {len(qa_df)} QA pairs")
    
    # 2) Create dataset and split
    dataset = GATSubgraphDataset(qa_df, data_processor, max_neighbors)
    
    if len(dataset) == 0:
        raise ValueError("No valid subgraphs created!")
    
    # Split dataset
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    
    # Create data loaders with optimization
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_subgraphs,
        num_workers=2,  # Enable parallel data loading
        pin_memory=True
    )
    
    # Optimized validation loader
    if fast_validation:
        val_loader = create_optimized_val_loader(
            val_dataset, 
            batch_size=max(4, batch_size // 4),  # Smaller validation batches
            num_workers=2,
            fast_mode=True
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_subgraphs,
            num_workers=2,
            pin_memory=True
        )
    
    # 3) Initialize model
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["node_features"].size(-1)
    output_dim = sample_batch["targets"].size(-1)
    
    model = GraphQueryProjectionGAT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout
    ).to(device)
    
    # 4) Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # 5) Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    learning_rates = []
    training_times = []
    best_epoch = None
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        # Training phase
        model.train()
        train_epoch_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]"):
            node_features = batch["node_features"].to(device)
            adjacency = batch["adjacency"].to(device)
            targets = batch["targets"].to(device)
            context_indices = batch["context_indices"]
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(node_features, adjacency, context_indices)
            
            # Compute loss
            loss = combined_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_epoch_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_epoch_losses)
        train_losses.append(avg_train_loss)
        
        # OPTIMIZED Validation phase
        model.eval()
        val_epoch_losses = []
        
        validation_start = time.time()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]"):
                node_features = batch["node_features"].to(device)
                adjacency = batch["adjacency"].to(device)
                targets = batch["targets"].to(device)
                context_indices = batch["context_indices"]
                
                # Use fast validation forward pass if enabled
                if fast_validation:
                    predictions = fast_validation_forward(model, node_features, adjacency, context_indices)
                else:
                    predictions = model(node_features, adjacency, context_indices)
                
                loss = combined_loss(predictions, targets)
                val_epoch_losses.append(loss.item())
        
        validation_time = time.time() - validation_start
        
        avg_val_loss = np.mean(val_epoch_losses)
        val_losses.append(avg_val_loss)
        
        # Track additional metrics
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        epoch_time = time.time() - epoch_start_time
        training_times.append(epoch_time)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch:02d}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, LR = {current_lr:.2e}, "
              f"Time = {epoch_time:.1f}s (Val: {validation_time:.1f}s)")
        
        # Evaluation callback
        if eval_callback is not None and eval_data is not None and epoch % eval_frequency == 0:
            model.eval()
            eval_callback(model, epoch, eval_data, device)
            model.train()
        
        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'output_dim': output_dim,
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                    'dropout': dropout
                }
            }, os.path.join(model_dir, "best_gat_model.pt"))
            
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # 6) Create comprehensive training plots
    training_config_dict = {
        'batch_size': batch_size,
        'lr': lr,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'dropout': dropout,
        'max_neighbors': max_neighbors,
        'patience': patience,
        'weight_decay': weight_decay
    }
    
    create_comprehensive_training_plots(
        train_losses=train_losses,
        val_losses=val_losses,
        learning_rates=learning_rates,
        model_dir=model_dir,
        training_config=training_config_dict,
        training_times=training_times,
        best_epoch=best_epoch
    )
    
    # Also save the simple plot for backward compatibility
    plot_path = os.path.join(model_dir, "training_history.png")
    plot_training_history(train_losses, val_losses, plot_path)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'training_times': training_times,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'dropout': dropout
        }
    }, os.path.join(model_dir, "final_gat_model.pt"))
    
    # Load best model
    checkpoint = torch.load(os.path.join(model_dir, "best_gat_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    return model


# -----------------------------------------------------------------------------
# 6) EVALUATION FUNCTIONS
# -----------------------------------------------------------------------------

def evaluate_gat_model(
    model: GraphQueryProjectionGAT,
    data_processor: GATDataProcessor, 
    qa_df: pd.DataFrame,
    device: str = "cpu",
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate the GAT model using recall@k and MRR metrics.
    """
    model.eval()
    
    # Get all context embeddings for retrieval pool
    query = """
    MATCH (ctx:CONTEXT) 
    WHERE ctx.graph_embedding IS NOT NULL
    RETURN id(ctx) AS ctx_id, ctx.graph_embedding AS graph_emb
    """
    
    with data_processor.driver.session() as sess:
        context_records = list(sess.run(query))
    
    context_ids = [rec["ctx_id"] for rec in context_records]
    context_embeddings = np.array([rec["graph_emb"] for rec in context_records])
    
    print(f"Evaluation context pool size: {len(context_ids)}")
    
    # Evaluate on subset of QA pairs
    eval_results = {"recall": {k: [] for k in k_values}, "mrr": []}
    
    with torch.no_grad():
        for _, qa_row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Evaluating"):
            try:
                # Build subgraph for this QA pair
                subgraph = data_processor.build_subgraph_data(qa_row)
                
                # Prepare batch (single item)
                batch = collate_subgraphs([subgraph])
                node_features = batch["node_features"].to(device)
                adjacency = batch["adjacency"].to(device)
                context_indices = batch["context_indices"]
                
                # Get model prediction
                pred_embedding = model(node_features, adjacency, context_indices)
                pred_np = pred_embedding.cpu().numpy()[0]  # Single prediction
                
                # Compute similarities with all contexts
                similarities = cosine_similarity([pred_np], context_embeddings)[0]
                
                # Get top-k predictions
                top_indices = np.argsort(similarities)[::-1]
                
                # True positive context IDs for this question
                true_context_ids = set(qa_row["ctx_ids"])
                
                # Compute recall@k
                for k in k_values:
                    top_k_ids = [context_ids[idx] for idx in top_indices[:k]]
                    recall_k = len(set(top_k_ids) & true_context_ids) / len(true_context_ids)
                    eval_results["recall"][k].append(recall_k)
                
                # Compute MRR
                mrr = 0.0
                for rank, idx in enumerate(top_indices, 1):
                    if context_ids[idx] in true_context_ids:
                        mrr = 1.0 / rank
                        break
                eval_results["mrr"].append(mrr)
                
            except Exception as e:
                print(f"Error evaluating QA {qa_row['qa_id']}: {e}")
                continue
    
    # Aggregate results
    final_results = {}
    for k in k_values:
        final_results[f"recall@{k}"] = np.mean(eval_results["recall"][k])
    final_results["mrr"] = np.mean(eval_results["mrr"])
    
    return final_results


def visualize_embeddings(
    model: GraphQueryProjectionGAT,
    data_processor: GATDataProcessor,
    qa_df: pd.DataFrame,
    save_path: str,
    device: str = "cpu",
    n_samples: int = 100
):
    """
    Create t-SNE visualization comparing semantic and projected embeddings.
    """
    model.eval()
    model.to(device)  # Ensure model is on correct device
    
    # Sample subset for visualization
    sample_df = qa_df.sample(min(n_samples, len(qa_df))).reset_index(drop=True)
    
    semantic_embeddings = []
    projected_embeddings = []
    labels = []
    
    print(f"Processing {len(sample_df)} samples for visualization...")
    
    with torch.no_grad():
        for idx, (_, qa_row) in enumerate(tqdm(sample_df.iterrows(), desc="Generating embeddings")):
            try:
                # Get semantic embedding (original BERT)
                semantic_emb = qa_row["q_emb"]
                if semantic_emb is None or len(semantic_emb) == 0:
                    continue
                    
                # Get projected embedding with proper device handling
                subgraph = data_processor.build_subgraph_data(qa_row)
                batch = collate_subgraphs([subgraph])
                
                # Ensure all tensors are on the same device
                node_features = batch["node_features"].to(device, non_blocking=True)
                adjacency = batch["adjacency"].to(device, non_blocking=True)
                context_indices = batch["context_indices"]
                
                # Forward pass
                proj_emb = model(node_features, adjacency, context_indices)
                
                # Convert to CPU numpy
                proj_emb_np = proj_emb.detach().cpu().numpy()[0]
                
                # Only add if both embeddings are valid
                if len(proj_emb_np) > 0:
                    semantic_embeddings.append(semantic_emb)
                    projected_embeddings.append(proj_emb_np)
                    labels.append(f"QA_{qa_row['qa_id']}")
                
            except Exception as e:
                print(f"Error processing QA {qa_row['qa_id']}: {e}")
                continue
    
    print(f"Successfully processed {len(projected_embeddings)} embeddings for visualization")
    
    # Check if we have enough samples for t-SNE
    if len(projected_embeddings) < 5:
        print(f"⚠️ Warning: Only {len(projected_embeddings)} valid embeddings found. Skipping visualization.")
        # Create a simple fallback plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Insufficient data for visualization\n({len(projected_embeddings)} samples)', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.title("Embedding Visualization - Insufficient Data")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Convert lists to numpy arrays for t-SNE
    semantic_embeddings_array = np.array(semantic_embeddings)
    projected_embeddings_array = np.array(projected_embeddings)
    
    print(f"Semantic embeddings shape: {semantic_embeddings_array.shape}")
    print(f"Projected embeddings shape: {projected_embeddings_array.shape}")
    
    # Create t-SNE visualization with adaptive parameters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate appropriate perplexity (must be less than n_samples)
    n_samples = len(semantic_embeddings_array)
    max_perplexity = min(30, max(5, n_samples - 1))  # Ensure valid range
    
    print(f"Using t-SNE with perplexity={max_perplexity} for {n_samples} samples")
    
    # Semantic embeddings
    if n_samples >= 2:
        semantic_2d = TSNE(n_components=2, random_state=42, perplexity=max_perplexity).fit_transform(semantic_embeddings_array)
        ax1.scatter(semantic_2d[:, 0], semantic_2d[:, 1], alpha=0.6, s=50)
    else:
        ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax1.transAxes)
    
    ax1.set_title(f"Original BERT Embeddings (n={n_samples})")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    
    # Projected embeddings  
    if n_samples >= 2:
        projected_2d = TSNE(n_components=2, random_state=42, perplexity=max_perplexity).fit_transform(projected_embeddings_array)
        ax2.scatter(projected_2d[:, 0], projected_2d[:, 1], alpha=0.6, s=50, color='red')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_title(f"GAT Projected Embeddings (n={n_samples})")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Embedding visualization saved to {save_path}")


# -----------------------------------------------------------------------------
# 7) MAIN TRAINING FUNCTION
# -----------------------------------------------------------------------------

def train(
    uri: str,
    user: str,
    password: str, 
    database: str,
    model_dir: str,
    batch_size: int = 32,
    epochs: int = 300,
    evaluate_during_training: bool = False,
    eval_frequency: int = 10,
    **kwargs
):
    """Main training function with default parameters."""
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize data processor for evaluation during training
    eval_metrics_history = {}
    if evaluate_during_training:
        print("📊 Evaluation during training is enabled")
        data_processor = GATDataProcessor(uri, user, password, database)
        qa_df = data_processor.fetch_qa_with_contexts()
        eval_metrics_history = {"recall@1": [], "recall@5": [], "recall@10": [], "mrr": []}
    
    # Train model with evaluation callbacks and fast validation
    model = train_gat_projection(
        uri=uri,
        user=user,
        password=password,
        database=database,
        model_dir=model_dir,
        batch_size=batch_size,
        epochs=epochs,
        eval_callback=(evaluate_during_training_callback if evaluate_during_training else None),
        eval_frequency=eval_frequency,
        eval_data=(data_processor, qa_df, eval_metrics_history) if evaluate_during_training else None,
        fast_validation=True,  # Enable fast validation by default
        **kwargs
    )
    
    # Final evaluation
    print("\n🔍 Final model evaluation...")
    if not evaluate_during_training:
        data_processor = GATDataProcessor(uri, user, password, database)
        qa_df = data_processor.fetch_qa_with_contexts()
    
    eval_results = evaluate_gat_model(model, data_processor, qa_df)
    
    print("\n📈 Final Evaluation Results:")
    for metric, score in eval_results.items():
        print(f"  {metric}: {score:.4f}")
    
    # Save evaluation results
    results_path = os.path.join(model_dir, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write("Final Evaluation Results:\n")
        f.write("=" * 30 + "\n")
        for metric, score in eval_results.items():
            f.write(f"{metric}: {score:.4f}\n")
        
        if eval_metrics_history and any(eval_metrics_history.values()):
            f.write("\nTraining Evaluation History:\n")
            f.write("=" * 30 + "\n")
            for metric, values in eval_metrics_history.items():
                if values:
                    f.write(f"{metric}: {values}\n")
    
    # Create final comprehensive plots including evaluation metrics
    if eval_metrics_history and any(eval_metrics_history.values()):
        print("📊 Creating comprehensive plots with evaluation metrics...")
        # Update plots with evaluation metrics
        training_config_dict = {
            'batch_size': batch_size,
            'epochs': epochs,
            'eval_frequency': eval_frequency,
            **kwargs
        }
        
        # Create dedicated recall progression plot
        if 'eval_epochs' in eval_metrics_history:
            eval_epochs = eval_metrics_history['eval_epochs']
            recall_data = {k: v for k, v in eval_metrics_history.items() 
                          if k.startswith('recall@') or k == 'mrr'}
            
            if recall_data:
                create_recall_progression_plot(
                    recall_history=recall_data,
                    eval_epochs=eval_epochs,
                    model_dir=model_dir,
                    training_config=training_config_dict
                )
                
                save_recall_metrics_summary(
                    recall_history=recall_data,
                    eval_epochs=eval_epochs,
                    model_dir=model_dir
                )
        
        # Load training history from saved model
        try:
            checkpoint_path = os.path.join(model_dir, "final_gat_model.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
                    create_comprehensive_training_plots(
                        train_losses=checkpoint['train_losses'],
                        val_losses=checkpoint['val_losses'],
                        learning_rates=checkpoint.get('learning_rates', []),
                        model_dir=model_dir,
                        training_config=training_config_dict,
                        eval_metrics=eval_metrics_history,
                        training_times=checkpoint.get('training_times', []),
                        best_epoch=checkpoint.get('best_epoch')
                    )
        except Exception as e:
            print(f"⚠️ Could not create comprehensive plots with evaluation metrics: {e}")
    
    # Create visualization with error handling
    print("🎨 Creating embedding visualization...")
    viz_path = os.path.join(model_dir, "embedding_visualization.png")
    try:
        visualize_embeddings(model, data_processor, qa_df, viz_path, device="cpu")
        print(f"✅ Embedding visualization saved: {viz_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not create embedding visualization: {e}")
        # Create a placeholder plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Visualization failed:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title("Embedding Visualization - Error")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ Training completed! All results saved in: {model_dir}")
    return model


def evaluate_during_training_callback(
    model: 'GraphQueryProjectionGAT',
    epoch: int,
    eval_data: Tuple[Any, pd.DataFrame, Dict[str, List[float]]],
    device: str = "cpu"
) -> None:
    """
    Enhanced callback function to evaluate model during training with detailed recall tracking.
    
    Args:
        model: The GAT model being trained
        epoch: Current epoch number
        eval_data: Tuple of (data_processor, qa_df, eval_metrics_history)
        device: Device for evaluation
    """
    data_processor, qa_df, eval_metrics_history = eval_data
    
    print(f"  📊 Evaluating recall metrics at epoch {epoch}...")
    
    try:
        # Sample a subset for faster evaluation during training
        sample_size = min(100, len(qa_df))  # Increased sample size for better recall estimates
        qa_sample = qa_df.sample(n=sample_size, random_state=42)
        
        # Comprehensive evaluation with multiple k values
        eval_results = evaluate_gat_model(
            model, data_processor, qa_sample, device, k_values=[1, 3, 5, 10]
        )
        
        # Store results in history
        for metric, score in eval_results.items():
            if metric not in eval_metrics_history:
                eval_metrics_history[metric] = []
            eval_metrics_history[metric].append(score)
        
        # Also track evaluation epochs for plotting
        if 'eval_epochs' not in eval_metrics_history:
            eval_metrics_history['eval_epochs'] = []
        eval_metrics_history['eval_epochs'].append(epoch)
        
        # Print evaluation results with better formatting
        eval_str = " | ".join([f"{k}: {v:.3f}" for k, v in eval_results.items()])
        print(f"  📈 Epoch {epoch} metrics: {eval_str}")
        
        # Print improvement tracking for key metrics
        if len(eval_metrics_history.get('recall@1', [])) >= 2:
            r1_history = eval_metrics_history['recall@1']
            r1_improvement = r1_history[-1] - r1_history[-2]
            trend = "↗️" if r1_improvement > 0 else "↘️" if r1_improvement < 0 else "➡️"
            print(f"  📊 R@1 trend: {trend} ({r1_improvement:+.3f} from previous)")
        
    except Exception as e:
        print(f"  ⚠️ Evaluation failed at epoch {epoch}: {e}")
        # Add None values to maintain consistency
        for metric in ['recall@1', 'recall@3', 'recall@5', 'recall@10', 'mrr']:
            if metric not in eval_metrics_history:
                eval_metrics_history[metric] = []
            eval_metrics_history[metric].append(None)
        
        if 'eval_epochs' not in eval_metrics_history:
            eval_metrics_history['eval_epochs'] = []
        eval_metrics_history['eval_epochs'].append(epoch)


# -----------------------------------------------------------------------------
# 8) UTILITY FUNCTIONS FOR INFERENCE
# -----------------------------------------------------------------------------

def load_gat_model(model_path: str, device: str = "cpu") -> GraphQueryProjectionGAT:
    """Load a trained GAT model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    
    model = GraphQueryProjectionGAT(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'], 
        output_dim=config['output_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def project_query_gat(
    query_embedding: List[float],
    model: GraphQueryProjectionGAT,
    data_processor: GATDataProcessor,
    device: str = "cpu",
    top_k_contexts: int = 10
) -> torch.Tensor:
    """
    Project a single query embedding using the trained GAT model.
    
    This function retrieves relevant contexts for the query and builds
    a subgraph for projection.
    """
    model.eval()
    
    # Find similar contexts for the query
    query = """
    WITH $q_emb AS q_emb
    MATCH (ctx:CONTEXT)
    WHERE ctx.embedding IS NOT NULL
    WITH ctx, gds.similarity.cosine(q_emb, ctx.embedding) AS sim
    ORDER BY sim DESC
    LIMIT $k
    RETURN id(ctx) AS ctx_id, ctx.embedding AS ctx_emb, ctx.graph_embedding AS ctx_graph_emb
    """
    
    with data_processor.driver.session() as sess:
        result = list(sess.run(query, q_emb=query_embedding, k=top_k_contexts))
    
    if not result:
        raise ValueError("No contexts found for query projection")
    
    # Build mock QA row for subgraph creation
    ctx_ids = [rec["ctx_id"] for rec in result]
    ctx_embs = [rec["ctx_emb"] for rec in result]
    ctx_graph_embs = [rec["ctx_graph_emb"] for rec in result]
    
    mock_qa_row = pd.Series({
        "qa_id": -1,  # Mock ID
        "q_emb": query_embedding,
        "ctx_ids": ctx_ids,
        "ctx_embs": ctx_embs,
        "ctx_graph_embs": ctx_graph_embs
    })
    
    # Build subgraph and predict
    with torch.no_grad():
        subgraph = data_processor.build_subgraph_data(mock_qa_row)
        batch = collate_subgraphs([subgraph])
        
        node_features = batch["node_features"].to(device)
        adjacency = batch["adjacency"].to(device)
        context_indices = batch["context_indices"]
        
        projected_emb = model(node_features, adjacency, context_indices)
        
    return projected_emb.squeeze(0).cpu()  # Return single embedding


def create_recall_progression_plot(
    recall_history: Dict[str, List[float]], 
    eval_epochs: List[int],
    model_dir: str,
    training_config: Dict[str, Any]
) -> None:
    """
    Create dedicated recall progression plot showing how recall metrics improve during training.
    
    Args:
        recall_history: Dict with keys like 'recall@1', 'recall@5', etc. and lists of values
        eval_epochs: List of epoch numbers when evaluations were performed
        model_dir: Directory to save the plot
        training_config: Training configuration for plot annotations
    """
    
    if not recall_history or not any(recall_history.values()):
        print("⚠️ No recall history data available for plotting")
        return
    
    # Set up professional plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'lines.linewidth': 3,
        'grid.alpha': 0.3
    })
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors for different recall metrics
    colors = {
        'recall@1': '#e74c3c',
        'recall@3': '#f39c12', 
        'recall@5': '#2ecc71',
        'recall@10': '#3498db',
        'mrr': '#9b59b6'
    }
    
    # Plot 1: All Recall Metrics
    ax1.set_title('Recall@K Progression During Training', fontsize=16, fontweight='bold', pad=20)
    
    for metric, values in recall_history.items():
        if values and metric.startswith('recall@'):
            # Only plot non-None values
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_epochs = [eval_epochs[i] for i in valid_indices]
            valid_values = [values[i] for i in valid_indices]
            
            if valid_values:
                color = colors.get(metric, '#7f8c8d')
                ax1.plot(valid_epochs, valid_values, 'o-', 
                        color=color, label=metric.upper(), 
                        linewidth=3, markersize=8, alpha=0.8)
                
                # Add final value annotation
                if valid_values:
                    final_epoch = valid_epochs[-1]
                    final_value = valid_values[-1]
                    ax1.annotate(f'{final_value:.3f}', 
                               xy=(final_epoch, final_value),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                               fontsize=10, color='white', fontweight='bold')
    
    ax1.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Recall Score', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Add improvement annotations
    if 'recall@1' in recall_history and recall_history['recall@1']:
        r1_values = [v for v in recall_history['recall@1'] if v is not None]
        if len(r1_values) >= 2:
            improvement = r1_values[-1] - r1_values[0]
            improvement_text = f"R@1 Improvement: {improvement:+.3f}"
            ax1.text(0.02, 0.98, improvement_text, transform=ax1.transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: MRR and Best Recall Comparison
    ax2.set_title('MRR and Top Recall Metrics', fontsize=16, fontweight='bold', pad=20)
    
    # Plot MRR if available
    if 'mrr' in recall_history and recall_history['mrr']:
        mrr_values = recall_history['mrr']
        valid_indices = [i for i, v in enumerate(mrr_values) if v is not None]
        valid_epochs = [eval_epochs[i] for i in valid_indices]
        valid_mrr = [mrr_values[i] for i in valid_indices]
        
        if valid_mrr:
            ax2.plot(valid_epochs, valid_mrr, 's-', 
                    color=colors['mrr'], label='MRR', 
                    linewidth=3, markersize=8, alpha=0.8)
    
    # Plot best recall metrics
    for metric in ['recall@1', 'recall@5']:
        if metric in recall_history and recall_history[metric]:
            values = recall_history[metric]
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_epochs = [eval_epochs[i] for i in valid_indices]
            valid_values = [values[i] for i in valid_indices]
            
            if valid_values:
                color = colors.get(metric, '#7f8c8d')
                ax2.plot(valid_epochs, valid_values, 'o-', 
                        color=color, label=metric.upper(), 
                        linewidth=3, markersize=8, alpha=0.8)
    
    ax2.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # Add summary statistics
    stats_text = "Training Summary:\n"
    stats_text += f"Evaluation Points: {len([e for e in eval_epochs if e > 0])}\n"
    stats_text += f"Eval Frequency: {training_config.get('eval_frequency', 'N/A')}\n"
    
    # Add best scores
    for metric in ['recall@1', 'recall@5', 'mrr']:
        if metric in recall_history and recall_history[metric]:
            valid_values = [v for v in recall_history[metric] if v is not None]
            if valid_values:
                best_score = max(valid_values)
                stats_text += f"Best {metric.upper()}: {best_score:.3f}\n"
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Overall title
    fig.suptitle('Retrieval Performance Evaluation During Training', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save the plot
    recall_plot_path = os.path.join(model_dir, "recall_progression.png")
    plt.savefig(recall_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📈 Recall progression plot saved: {recall_plot_path}")


def save_recall_metrics_summary(
    recall_history: Dict[str, List[float]], 
    eval_epochs: List[int],
    model_dir: str
) -> None:
    """
    Save detailed recall metrics summary to text file.
    
    Args:
        recall_history: Dict with recall metrics history
        eval_epochs: List of evaluation epochs
        model_dir: Directory to save the summary
    """
    
    summary_path = os.path.join(model_dir, "recall_evaluation_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("GAT Model Recall Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Write evaluation schedule
        f.write(f"Evaluation performed at {len(eval_epochs)} points:\n")
        f.write(f"Epochs: {eval_epochs}\n\n")
        
        # Write metric progression
        for metric, values in recall_history.items():
            if values:
                f.write(f"{metric.upper()} Progression:\n")
                f.write("-" * 30 + "\n")
                
                valid_data = [(eval_epochs[i], values[i]) for i in range(len(values)) 
                             if i < len(eval_epochs) and values[i] is not None]
                
                for epoch, score in valid_data:
                    f.write(f"Epoch {epoch:3d}: {score:.4f}\n")
                
                if valid_data:
                    scores = [score for _, score in valid_data]
                    f.write(f"\nSummary for {metric.upper()}:\n")
                    f.write(f"  Initial: {scores[0]:.4f}\n")
                    f.write(f"  Final:   {scores[-1]:.4f}\n")
                    f.write(f"  Best:    {max(scores):.4f}\n")
                    f.write(f"  Improvement: {scores[-1] - scores[0]:+.4f}\n")
                
                f.write("\n" + "=" * 30 + "\n\n")
        
        # Write best overall performance
        f.write("BEST PERFORMANCE ACHIEVED:\n")
        f.write("=" * 30 + "\n")
        
        for metric, values in recall_history.items():
            if values:
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    best_score = max(valid_values)
                    best_epoch_idx = next(i for i, v in enumerate(values) if v == best_score)
                    best_epoch = eval_epochs[best_epoch_idx] if best_epoch_idx < len(eval_epochs) else "Unknown"
                    f.write(f"{metric.upper():<12}: {best_score:.4f} (Epoch {best_epoch})\n")
    
    print(f"📄 Recall evaluation summary saved: {summary_path}")
