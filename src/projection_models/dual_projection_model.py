"""
Dual Projection Model for Contrastive Learning in Semantic and Graph Spaces

This module implements a dual-head projection model that learns to project questions
into both semantic (SBERT) and graph (GNN) embedding spaces simultaneously using
contrastive learning with hard negative mining strategies.

Key Components:
1. DualProjectionModel: Neural network with separate semantic and graph heads
2. Contrastive loss functions for both spaces
3. Hard negative mining (in-batch and offline strategies)
4. Training utilities with dual-space optimization
5. Data processing for dual-space learning
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime
from tqdm import tqdm
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Import your existing modules
from configs import ConfigEnv, ConfigPath
from llms.embedding_model import EmbeddingModel
from knowledge_graph.connection import Neo4jConnection


class ResidualBlock(nn.Module):
    """Residual block with layer normalization and dropout."""
    
    def __init__(self, dim: int, p_dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Linear(dim, dim),
            nn.Dropout(p_dropout)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.layers(x))


class DualProjectionModel(nn.Module):
    """
    Dual Projection Model that projects questions into both semantic and graph spaces.
    
    This model takes question embeddings and projects them into two spaces:
    1. Semantic space (SBERT-compatible)
    2. Graph space (GNN node embedding compatible)
    """
    
    def __init__(
        self,
        dim_sem: int = 768,
        dim_graph: int = 768,
        hidden_dims: List[int] = [512, 2048, 1024],
        p_dropout: float = 0.2,
    ):
        super().__init__()
        
        self.dim_sem = dim_sem
        self.dim_graph = dim_graph
        self.hidden_dims = hidden_dims
        self.p_dropout = p_dropout
        
        def build_head(output_dim):
            layers = []
            prev = dim_sem  # Input dimension (BERT embedding size)
            for h in hidden_dims:
                layers += [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p_dropout),
                    ResidualBlock(h, p_dropout),
                ]
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            return nn.Sequential(*layers)

        # Two separate heads
        self.semantic_head = build_head(dim_sem)    # Projects into SBERT space
        self.graph_head = build_head(dim_graph)     # Projects into GNN space
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both projection heads.
        
        Args:
            x: Input question embeddings [batch_size, dim_sem]
            
        Returns:
            Tuple of (semantic_projection, graph_projection)
        """
        semantic_proj = self.semantic_head(x)
        graph_proj = self.graph_head(x)
        return semantic_proj, graph_proj


class ContrastiveLoss(nn.Module):
    """Contrastive loss function for dual-space learning."""
    
    def __init__(self, temperature: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self, 
        query_proj: torch.Tensor, 
        context_proj: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss between query and context projections.
        
        Args:
            query_proj: Query projections [batch_size, dim]
            context_proj: Context projections [batch_size, dim] or [batch_size, num_contexts, dim]
            labels: Ground truth labels [batch_size] (if None, assumes diagonal matching)
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        query_proj = F.normalize(query_proj, dim=-1)
        
        if context_proj.dim() == 3:
            # Multiple contexts per query
            context_proj = F.normalize(context_proj, dim=-1)
            # Compute similarity matrix
            logits = torch.bmm(query_proj.unsqueeze(1), context_proj.transpose(1, 2)).squeeze(1)
        else:
            # Single context per query (in-batch negatives)
            context_proj = F.normalize(context_proj, dim=-1)
            # Compute similarity matrix
            logits = query_proj @ context_proj.T
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        # Create labels if not provided (diagonal matching)
        if labels is None:
            labels = torch.arange(len(query_proj), device=query_proj.device)
        
        # Compute cross-entropy loss
        return F.cross_entropy(logits, labels, reduction=self.reduction)


def contrastive_loss(q_proj, c_proj, temperature=0.1):
    """
    Simple contrastive loss function for in-batch hard negatives.
    
    Args:
        q_proj: Query projections [batch_size, dim]
        c_proj: Context projections [batch_size, dim]
        temperature: Temperature scaling factor
        
    Returns:
        Contrastive loss value
    """
    q_proj = F.normalize(q_proj, dim=-1)
    c_proj = F.normalize(c_proj, dim=-1)
    logits = q_proj @ c_proj.T 
    labels = torch.arange(len(q_proj)).to(q_proj.device)
    return F.cross_entropy(logits / temperature, labels)


class HardNegativeMiner:
    """Hard negative mining utilities for dual-space contrastive learning."""
    
    def __init__(
        self, 
        embedding_model: EmbeddingModel,
        neo4j_connection: Neo4jConnection,
        device: str = "cuda"
    ):
        self.embedding_model = embedding_model
        self.neo4j_connection = neo4j_connection
        self.device = device
        
        # Storage for precomputed embeddings and indices
        self.question_embeddings = None
        self.context_embeddings = None
        self.graph_embeddings = None
        self.faiss_index = None
        self.context_to_idx = {}
        self.idx_to_context = {}
        
    def precompute_embeddings(
        self, 
        questions: List[str], 
        contexts: List[str],
        batch_size: int = 32
    ):
        """Precompute all embeddings for offline negative mining."""
        print("🔧 Precomputing embeddings for negative mining...")
        
        # Compute question embeddings
        print("📝 Computing question embeddings...")
        self.question_embeddings = []
        for i in tqdm(range(0, len(questions), batch_size), desc="Questions"):
            batch = questions[i:i+batch_size]
            embeddings = self.embedding_model.embed_documents(batch)
            self.question_embeddings.extend(embeddings)
        self.question_embeddings = np.array(self.question_embeddings)
        
        # Compute context embeddings
        print("📄 Computing context embeddings...")
        self.context_embeddings = []
        for i in tqdm(range(0, len(contexts), batch_size), desc="Contexts"):
            batch = contexts[i:i+batch_size]
            embeddings = self.embedding_model.embed_documents(batch)
            self.context_embeddings.extend(embeddings)
        self.context_embeddings = np.array(self.context_embeddings)
        
        # Build context mapping
        self.context_to_idx = {ctx: idx for idx, ctx in enumerate(contexts)}
        self.idx_to_context = {idx: ctx for idx, ctx in enumerate(contexts)}
        
        # Build FAISS index for fast similarity search
        print("🔍 Building FAISS index...")
        dimension = self.context_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        normalized_contexts = self.context_embeddings / np.linalg.norm(
            self.context_embeddings, axis=1, keepdims=True
        )
        self.faiss_index.add(normalized_contexts.astype(np.float32))
        
        print(f"✅ Precomputed {len(self.question_embeddings)} question and {len(self.context_embeddings)} context embeddings")
    
    def get_hard_negatives(
        self, 
        question_idx: int, 
        positive_contexts: List[str], 
        num_negatives: int = 10,
        min_similarity: float = 0.3
    ) -> List[int]:
        """
        Get hard negatives for a specific question using offline mining.
        
        Args:
            question_idx: Index of the question
            positive_contexts: List of positive contexts to exclude
            num_negatives: Number of hard negatives to return
            min_similarity: Minimum similarity threshold for hard negatives
            
        Returns:
            List of context indices for hard negatives
        """
        if self.faiss_index is None or self.question_embeddings is None:
            raise ValueError("Must call precompute_embeddings() first")
        
        # Get question embedding
        q_embedding = self.question_embeddings[question_idx:question_idx+1]
        q_embedding = q_embedding / np.linalg.norm(q_embedding, axis=1, keepdims=True)
        
        # Search for similar contexts
        scores, indices = self.faiss_index.search(
            q_embedding.astype(np.float32), 
            min(len(self.context_embeddings), num_negatives * 3)  # Get more candidates
        )
        
        # Filter out positive contexts and low-similarity contexts
        positive_indices = set()
        for pos_ctx in positive_contexts:
            if pos_ctx in self.context_to_idx:
                positive_indices.add(self.context_to_idx[pos_ctx])
        
        hard_negatives = []
        for score, idx in zip(scores[0], indices[0]):
            if (idx not in positive_indices and 
                score >= min_similarity and 
                len(hard_negatives) < num_negatives):
                hard_negatives.append(int(idx))
        
        # If we don't have enough hard negatives, add random ones
        if len(hard_negatives) < num_negatives:
            available_indices = set(range(len(self.context_embeddings))) - positive_indices - set(hard_negatives)
            additional_needed = num_negatives - len(hard_negatives)
            if available_indices:
                additional = np.random.choice(
                    list(available_indices), 
                    size=min(additional_needed, len(available_indices)), 
                    replace=False
                )
                hard_negatives.extend(additional.tolist())
        
        return hard_negatives
    
    def save_precomputed_data(self, save_path: str):
        """Save precomputed embeddings and indices."""
        save_data = {
            'question_embeddings': self.question_embeddings,
            'context_embeddings': self.context_embeddings,
            'context_to_idx': self.context_to_idx,
            'idx_to_context': self.idx_to_context
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save FAISS index separately
        if self.faiss_index is not None:
            faiss_path = save_path.replace('.pkl', '_faiss.index')
            faiss.write_index(self.faiss_index, faiss_path)
        
        print(f"💾 Saved precomputed data to {save_path}")
    
    def load_precomputed_data(self, save_path: str):
        """Load precomputed embeddings and indices."""
        with open(save_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.question_embeddings = save_data['question_embeddings']
        self.context_embeddings = save_data['context_embeddings']
        self.context_to_idx = save_data['context_to_idx']
        self.idx_to_context = save_data['idx_to_context']
        
        # Load FAISS index
        faiss_path = save_path.replace('.pkl', '_faiss.index')
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
        
        print(f"📂 Loaded precomputed data from {save_path}")


class DualSpaceDataset(Dataset):
    """Dataset for dual-space contrastive learning."""
    
    def __init__(
        self,
        questions: List[str],
        semantic_contexts: List[str],
        graph_contexts: List[str],
        embedding_model: EmbeddingModel,
        hard_negative_miner: Optional[HardNegativeMiner] = None,
        num_hard_negatives: int = 5,
        use_hard_negatives: bool = True
    ):
        self.questions = questions
        self.semantic_contexts = semantic_contexts
        self.graph_contexts = graph_contexts
        self.embedding_model = embedding_model
        self.hard_negative_miner = hard_negative_miner
        self.num_hard_negatives = num_hard_negatives
        self.use_hard_negatives = use_hard_negatives
        
        # Precompute question embeddings
        print("🔧 Precomputing question embeddings for dataset...")
        self.question_embeddings = self.embedding_model.embed_documents(questions)
        
        # Precompute context embeddings
        print("🔧 Precomputing context embeddings for dataset...")
        self.semantic_context_embeddings = self.embedding_model.embed_documents(semantic_contexts)
        self.graph_context_embeddings = self.embedding_model.embed_documents(graph_contexts)
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        """Get a training sample with hard negatives if available."""
        item = {
            'question_embedding': torch.tensor(self.question_embeddings[idx], dtype=torch.float32),
            'semantic_context_embedding': torch.tensor(self.semantic_context_embeddings[idx], dtype=torch.float32),
            'graph_context_embedding': torch.tensor(self.graph_context_embeddings[idx], dtype=torch.float32),
            'question_text': self.questions[idx],
            'semantic_context_text': self.semantic_contexts[idx],
            'graph_context_text': self.graph_contexts[idx]
        }
        
        # Add hard negatives if available
        if self.use_hard_negatives and self.hard_negative_miner is not None:
            try:
                hard_neg_indices = self.hard_negative_miner.get_hard_negatives(
                    question_idx=idx,
                    positive_contexts=[self.semantic_contexts[idx]],
                    num_negatives=self.num_hard_negatives
                )
                
                if hard_neg_indices:
                    # Get hard negative embeddings
                    hard_neg_semantic = [self.semantic_context_embeddings[i] for i in hard_neg_indices]
                    hard_neg_graph = [self.graph_context_embeddings[i] for i in hard_neg_indices]
                    
                    item['hard_negative_semantic'] = torch.tensor(hard_neg_semantic, dtype=torch.float32)
                    item['hard_negative_graph'] = torch.tensor(hard_neg_graph, dtype=torch.float32)
                    item['hard_negative_indices'] = hard_neg_indices
            except Exception as e:
                print(f"⚠️ Warning: Failed to get hard negatives for sample {idx}: {e}")
        
        return item


def collate_dual_space_batch(batch):
    """Custom collate function for dual-space batches."""
    collated = {
        'question_embeddings': torch.stack([item['question_embedding'] for item in batch]),
        'semantic_context_embeddings': torch.stack([item['semantic_context_embedding'] for item in batch]),
        'graph_context_embeddings': torch.stack([item['graph_context_embedding'] for item in batch]),
        'question_texts': [item['question_text'] for item in batch],
        'semantic_context_texts': [item['semantic_context_text'] for item in batch],
        'graph_context_texts': [item['graph_context_text'] for item in batch]
    }
    
    # Handle hard negatives if present
    if 'hard_negative_semantic' in batch[0]:
        # Stack hard negatives with padding if needed
        max_negatives = max(len(item.get('hard_negative_indices', [])) for item in batch)
        if max_negatives > 0:
            semantic_negatives = []
            graph_negatives = []
            
            for item in batch:
                if 'hard_negative_semantic' in item:
                    sem_neg = item['hard_negative_semantic']
                    graph_neg = item['hard_negative_graph']
                    
                    # Pad if necessary
                    if len(sem_neg) < max_negatives:
                        padding_needed = max_negatives - len(sem_neg)
                        sem_pad = torch.zeros(padding_needed, sem_neg.shape[1])
                        graph_pad = torch.zeros(padding_needed, graph_neg.shape[1])
                        sem_neg = torch.cat([sem_neg, sem_pad], dim=0)
                        graph_neg = torch.cat([graph_neg, graph_pad], dim=0)
                    
                    semantic_negatives.append(sem_neg)
                    graph_negatives.append(graph_neg)
                else:
                    # Create dummy negatives
                    sem_dim = collated['semantic_context_embeddings'].shape[1]
                    graph_dim = collated['graph_context_embeddings'].shape[1]
                    semantic_negatives.append(torch.zeros(max_negatives, sem_dim))
                    graph_negatives.append(torch.zeros(max_negatives, graph_dim))
            
            collated['hard_negative_semantic'] = torch.stack(semantic_negatives)
            collated['hard_negative_graph'] = torch.stack(graph_negatives)
    
    return collated


class DualProjectionTrainer:
    """Trainer for dual projection model with contrastive learning."""
    
    def __init__(
        self,
        model: DualProjectionModel,
        semantic_loss_weight: float = 0.5,
        graph_loss_weight: float = 0.5,
        temperature: float = 0.1,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        patience: int = 10,
        min_delta: float = 1e-4
    ):
        self.model = model.to(device)
        self.device = device
        self.semantic_loss_weight = semantic_loss_weight
        self.graph_loss_weight = graph_loss_weight
        self.patience = patience
        self.min_delta = min_delta
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        
        # Loss functions
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Scheduler
        self.scheduler = None
        
        # Training history
        self.train_history = {
            'total_loss': [],
            'semantic_loss': [],
            'graph_loss': [],
            'learning_rates': []
        }
    
    def setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler."""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_training_steps,
            eta_min=1e-6
        )
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        question_embeddings = batch['question_embeddings'].to(self.device)
        semantic_contexts = batch['semantic_context_embeddings'].to(self.device)
        graph_contexts = batch['graph_context_embeddings'].to(self.device)
        
        # Forward pass
        semantic_proj, graph_proj = self.model(question_embeddings)
        
        # Compute losses
        semantic_loss = self.contrastive_loss(semantic_proj, semantic_contexts)
        graph_loss = self.contrastive_loss(graph_proj, graph_contexts)
        
        # Handle hard negatives if present
        if 'hard_negative_semantic' in batch:
            hard_neg_semantic = batch['hard_negative_semantic'].to(self.device)
            hard_neg_graph = batch['hard_negative_graph'].to(self.device)
            
            # Expand projections to match hard negatives
            batch_size = semantic_proj.shape[0]
            num_negatives = hard_neg_semantic.shape[1]
            
            # Create combined positive + negative contexts
            combined_semantic = torch.cat([
                semantic_contexts.unsqueeze(1),  # [B, 1, D]
                hard_neg_semantic  # [B, N, D]
            ], dim=1)  # [B, 1+N, D]
            
            combined_graph = torch.cat([
                graph_contexts.unsqueeze(1),  # [B, 1, D]
                hard_neg_graph  # [B, N, D]
            ], dim=1)  # [B, 1+N, D]
            
            # Labels are always 0 (first position is positive)
            labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            
            # Recompute losses with hard negatives
            semantic_loss = self.contrastive_loss(
                semantic_proj, combined_semantic, labels
            )
            graph_loss = self.contrastive_loss(
                graph_proj, combined_graph, labels
            )
        
        # Combined loss
        total_loss = (
            self.semantic_loss_weight * semantic_loss +
            self.graph_loss_weight * graph_loss
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Record metrics
        current_lr = self.optimizer.param_groups[0]['lr']
        self.train_history['total_loss'].append(total_loss.item())
        self.train_history['semantic_loss'].append(semantic_loss.item())
        self.train_history['graph_loss'].append(graph_loss.item())
        self.train_history['learning_rates'].append(current_lr)
        
        return {
            'total_loss': total_loss.item(),
            'semantic_loss': semantic_loss.item(),
            'graph_loss': graph_loss.item(),
            'learning_rate': current_lr
        }
    
    def train(
        self, 
        train_dataloader: DataLoader,
        num_epochs: int = 10,
        validation_dataloader: Optional[DataLoader] = None,
        save_dir: Optional[str] = None
    ):
        """Train the dual projection model with early stopping."""
        print(f"🚀 Starting dual projection training for up to {num_epochs} epochs")
        print(f"📋 Early stopping: patience={self.patience}, min_delta={self.min_delta}")
        
        # Setup scheduler
        num_training_steps = len(train_dataloader) * num_epochs
        self.setup_scheduler(num_training_steps)
        
        # Setup save directory
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            config_path = os.path.join(save_dir, "training_config.json")
            self._save_training_config(config_path, num_epochs, len(train_dataloader))
        
        best_epoch = 0
        
        for epoch in range(num_epochs):
            if self.early_stopped:
                print(f"\n⏹️ Early stopping triggered at epoch {epoch}")
                break
                
            print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            epoch_metrics = {
                'total_loss': 0.0,
                'semantic_loss': 0.0,
                'graph_loss': 0.0,
                'learning_rate': 0.0
            }
            
            train_pbar = tqdm(train_dataloader, desc="Training")
            for batch_idx, batch in enumerate(train_pbar):
                step_metrics = self.train_step(batch)
                
                # Update epoch metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += step_metrics[key]
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Total': f"{step_metrics['total_loss']:.4f}",
                    'Sem': f"{step_metrics['semantic_loss']:.4f}",
                    'Graph': f"{step_metrics['graph_loss']:.4f}",
                    'LR': f"{step_metrics['learning_rate']:.2e}"
                })
            
            # Average epoch metrics
            num_batches = len(train_dataloader)
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
            
            print(f"📊 Training - Total: {epoch_metrics['total_loss']:.4f}, "
                  f"Semantic: {epoch_metrics['semantic_loss']:.4f}, "
                  f"Graph: {epoch_metrics['graph_loss']:.4f}")
            
            # Validation phase
            val_metrics = None
            if validation_dataloader is not None:
                val_metrics = self.validate(validation_dataloader)
                val_loss = val_metrics['total_loss']
                
                print(f"📊 Validation - Total: {val_loss:.4f}, "
                      f"Semantic: {val_metrics['semantic_loss']:.4f}, "
                      f"Graph: {val_metrics['graph_loss']:.4f}")
                
                # Early stopping and best model saving
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    best_epoch = epoch
                    self.patience_counter = 0
                    
                    # Save best model
                    if save_dir is not None:
                        best_model_path = os.path.join(save_dir, "best_dual_projection_model.pt")
                        self.save_model(best_model_path)
                        print(f"💾 Saved best model (epoch {epoch + 1}) - Val Loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"⏳ Patience: {self.patience_counter}/{self.patience}")
                    
                    if self.patience_counter >= self.patience:
                        self.early_stopped = True
                        print(f"🛑 Early stopping triggered - no improvement for {self.patience} epochs")
        
        final_message = "✅ Training completed"
        if self.early_stopped:
            final_message += f" (early stopped at epoch {epoch + 1})"
        else:
            final_message += f" (completed all {num_epochs} epochs)"
            
        print(f"\n{final_message}!")
        if validation_dataloader is not None:
            print(f"🏆 Best validation loss: {self.best_val_loss:.4f} (epoch {best_epoch + 1})")
        
        return self.train_history
    
    def validate(self, validation_dataloader: DataLoader):
        """Validate the model."""
        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'semantic_loss': 0.0,
            'graph_loss': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(validation_dataloader, desc="Validation"):
                # Move data to device
                question_embeddings = batch['question_embeddings'].to(self.device)
                semantic_contexts = batch['semantic_context_embeddings'].to(self.device)
                graph_contexts = batch['graph_context_embeddings'].to(self.device)
                
                # Forward pass
                semantic_proj, graph_proj = self.model(question_embeddings)
                
                # Compute losses
                semantic_loss = self.contrastive_loss(semantic_proj, semantic_contexts)
                graph_loss = self.contrastive_loss(graph_proj, graph_contexts)
                total_loss = (
                    self.semantic_loss_weight * semantic_loss +
                    self.graph_loss_weight * graph_loss
                )
                
                val_metrics['total_loss'] += total_loss.item()
                val_metrics['semantic_loss'] += semantic_loss.item()
                val_metrics['graph_loss'] += graph_loss.item()
        
        # Average metrics
        num_batches = len(validation_dataloader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def save_model(self, save_path: str):
        """Save the model and training state."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'model_config': {
                'dim_sem': self.model.dim_sem,
                'dim_graph': self.model.dim_graph,
                'hidden_dims': self.model.hidden_dims,
                'p_dropout': self.model.p_dropout
            },
            'training_config': {
                'semantic_loss_weight': self.semantic_loss_weight,
                'graph_loss_weight': self.graph_loss_weight,
                'temperature': self.contrastive_loss.temperature
            }
        }
        
        if self.scheduler is not None:
            save_dict['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(save_dict, save_path)
    
    def load_model(self, load_path: str):
        """Load the model and training state."""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint.get('train_history', self.train_history)
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"📂 Loaded model from {load_path}")
        return checkpoint
    
    def _save_training_config(self, config_path: str, num_epochs: int, steps_per_epoch: int):
        """Save training configuration."""
        config = {
            'model_config': {
                'dim_sem': self.model.dim_sem,
                'dim_graph': self.model.dim_graph,
                'hidden_dims': self.model.hidden_dims,
                'p_dropout': self.model.p_dropout
            },
            'training_config': {
                'num_epochs': num_epochs,
                'steps_per_epoch': steps_per_epoch,
                'semantic_loss_weight': self.semantic_loss_weight,
                'graph_loss_weight': self.graph_loss_weight,
                'temperature': self.contrastive_loss.temperature,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'device': self.device
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def load_dual_projection_model(
    model_path: str, 
    device: str = "cuda"
) -> Tuple[DualProjectionModel, dict]:
    """
    Load a trained dual projection model.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, checkpoint_info)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    
    # Create model
    model = DualProjectionModel(
        dim_sem=model_config['dim_sem'],
        dim_graph=model_config['dim_graph'],
        hidden_dims=model_config['hidden_dims'],
        p_dropout=model_config['p_dropout']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


# Example usage functions
def create_dual_projection_example():
    """Create an example dual projection model."""
    return DualProjectionModel(
        dim_sem=768,           # BERT embedding dimension
        dim_graph=768,         # Graph embedding dimension
        hidden_dims=[512, 2048, 1024],  # Hidden layer dimensions
        p_dropout=0.2          # Dropout probability
    )


if __name__ == "__main__":
    # Example usage
    print("🔧 Testing Dual Projection Model components...")
    
    # Create model
    model = create_dual_projection_example()
    print(f"✅ Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 4
    input_dim = 768
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        semantic_proj, graph_proj = model(x)
        print(f"✅ Forward pass successful:")
        print(f"   Input shape: {x.shape}")
        print(f"   Semantic projection shape: {semantic_proj.shape}")
        print(f"   Graph projection shape: {graph_proj.shape}")
    
    # Test contrastive loss
    loss_fn = ContrastiveLoss(temperature=0.1)
    query_proj = torch.randn(batch_size, 768)
    context_proj = torch.randn(batch_size, 768)
    
    loss = loss_fn(query_proj, context_proj)
    print(f"✅ Contrastive loss computed: {loss.item():.4f}")
    
    print("\n🎉 All components working correctly!") 