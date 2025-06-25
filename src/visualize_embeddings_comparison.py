import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from graphdatascience import GraphDataScience
from typing import List, Tuple, Optional
import pandas as pd

from configs.config import ConfigEnv


def connect_to_neo4j() -> GraphDataScience:
    """Connect to Neo4j database using GDS client."""
    return GraphDataScience(
        ConfigEnv.NEO4J_URI, 
        auth=(ConfigEnv.NEO4J_USER, ConfigEnv.NEO4J_PASSWORD), 
        database=ConfigEnv.NEO4J_DB
    )


def fetch_sample_embeddings(gds: GraphDataScience, sample_size: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Fetch sample of nodes with both BERT and graph embeddings from Neo4j.
    
    Args:
        gds: GraphDataScience client
        sample_size: Number of samples to fetch
        
    Returns:
        Tuple of (bert_embeddings, graph_embeddings, node_texts)
    """
    print(f"Fetching {sample_size} samples with both BERT and graph embeddings...")
    
    # Cypher query to get nodes with both embedding types
    cypher_query = """
    MATCH (n:CONTEXT)
    WHERE n.embedding IS NOT NULL AND n.graph_embedding IS NOT NULL
    RETURN n.text as text, n.embedding as bert_embedding, n.graph_embedding as graph_embedding
    LIMIT $sample_size
    """
    
    result = gds.run_cypher(cypher_query, params={"sample_size": sample_size})
    
    if result.empty:
        raise ValueError("No nodes found with both BERT and graph embeddings. Please ensure embeddings are computed.")
    
    print(f"Successfully fetched {len(result)} samples")
    
    # Extract embeddings and texts
    bert_embeddings = np.array(result['bert_embedding'].tolist())
    graph_embeddings = np.array(result['graph_embedding'].tolist())
    texts = result['text'].tolist()
    
    print(f"BERT embeddings shape: {bert_embeddings.shape}")
    print(f"Graph embeddings shape: {graph_embeddings.shape}")
    
    return bert_embeddings, graph_embeddings, texts


def apply_tsne_reduction(
    bert_embeddings: np.ndarray, 
    graph_embeddings: np.ndarray,
    perplexity: int = 30,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply t-SNE dimensionality reduction to both embedding types.
    
    Args:
        bert_embeddings: BERT embeddings array
        graph_embeddings: Graph embeddings array
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (bert_tsne, graph_tsne) 2D coordinates
    """
    print("Applying t-SNE dimensionality reduction...")
    
    # Standardize embeddings before t-SNE
    scaler_bert = StandardScaler()
    scaler_graph = StandardScaler()
    
    bert_scaled = scaler_bert.fit_transform(bert_embeddings)
    graph_scaled = scaler_graph.fit_transform(graph_embeddings)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, verbose=1)
    
    print("Computing t-SNE for BERT embeddings...")
    bert_tsne = tsne.fit_transform(bert_scaled)
    
    print("Computing t-SNE for Graph embeddings...")
    # Create new t-SNE instance for graph embeddings
    tsne_graph = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, verbose=1)
    graph_tsne = tsne_graph.fit_transform(graph_scaled)
    
    print("t-SNE reduction completed!")
    return bert_tsne, graph_tsne


def create_comparison_plot(
    bert_tsne: np.ndarray,
    graph_tsne: np.ndarray,
    texts: List[str],
    save_path: str = "embeddings_comparison.png"
) -> None:
    """
    Create side-by-side comparison plot of BERT vs Graph embeddings.
    
    Args:
        bert_tsne: t-SNE reduced BERT embeddings
        graph_tsne: t-SNE reduced Graph embeddings
        texts: Node texts for potential labeling
        save_path: Path to save the plot
    """
    print("Creating comparison visualization...")
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Color points based on cluster-like regions (using y-coordinate for coloring)
    colors_bert = bert_tsne[:, 1]  # Use y-coordinate for coloring
    colors_graph = graph_tsne[:, 1]  # Use y-coordinate for coloring
    
    # Plot BERT embeddings
    scatter1 = ax1.scatter(
        bert_tsne[:, 0], bert_tsne[:, 1], 
        c=colors_bert, 
        cmap='viridis', 
        alpha=0.7, 
        s=60,
        edgecolors='black',
        linewidth=0.5
    )
    ax1.set_title('BERT Embeddings (t-SNE)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot Graph embeddings
    scatter2 = ax2.scatter(
        graph_tsne[:, 0], graph_tsne[:, 1], 
        c=colors_graph, 
        cmap='plasma', 
        alpha=0.7, 
        s=60,
        edgecolors='black',
        linewidth=0.5
    )
    ax2.set_title('Graph Embeddings (t-SNE)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('BERT vs Graph Embeddings Comparison (t-SNE Projection)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Add statistics text
    stats_text = f"Samples: {len(texts)}\nBERT dim: {bert_tsne.shape[1] if len(bert_tsne.shape) > 1 else 'N/A'}\nGraph dim: {graph_tsne.shape[1] if len(graph_tsne.shape) > 1 else 'N/A'}"
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved as: {save_path}")
    
    # Show the plot
    plt.show()


def create_overlay_plot(
    bert_tsne: np.ndarray,
    graph_tsne: np.ndarray,
    save_path: str = "embeddings_overlay_comparison.png"
) -> None:
    """
    Create an overlay plot showing both embeddings on the same axes.
    
    Args:
        bert_tsne: t-SNE reduced BERT embeddings
        graph_tsne: t-SNE reduced Graph embeddings
        save_path: Path to save the plot
    """
    print("Creating overlay comparison visualization...")
    
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot both embedding types with different markers and colors
    ax.scatter(
        bert_tsne[:, 0], bert_tsne[:, 1], 
        c='blue', 
        alpha=0.6, 
        s=80,
        marker='o',
        label='BERT Embeddings',
        edgecolors='darkblue',
        linewidth=0.5
    )
    
    ax.scatter(
        graph_tsne[:, 0], graph_tsne[:, 1], 
        c='red', 
        alpha=0.6, 
        s=80,
        marker='^',
        label='Graph Embeddings',
        edgecolors='darkred',
        linewidth=0.5
    )
    
    # Add connecting lines between corresponding points
    for i in range(min(len(bert_tsne), len(graph_tsne))):
        ax.plot([bert_tsne[i, 0], graph_tsne[i, 0]], 
                [bert_tsne[i, 1], graph_tsne[i, 1]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    ax.set_title('BERT vs Graph Embeddings Overlay (t-SNE Projection)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Overlay plot saved as: {save_path}")
    plt.show()


def analyze_embedding_distances(
    bert_embeddings: np.ndarray,
    graph_embeddings: np.ndarray,
    bert_tsne: np.ndarray,
    graph_tsne: np.ndarray
) -> None:
    """
    Analyze and report statistics about the embeddings.
    
    Args:
        bert_embeddings: Original BERT embeddings
        graph_embeddings: Original Graph embeddings
        bert_tsne: t-SNE reduced BERT embeddings
        graph_tsne: t-SNE reduced Graph embeddings
    """
    print("\n=== EMBEDDING ANALYSIS ===")
    
    # Original space statistics
    from scipy.spatial.distance import pdist
    
    bert_distances = pdist(bert_embeddings)
    graph_distances = pdist(graph_embeddings)
    
    print(f"Original BERT embeddings:")
    print(f"  Mean pairwise distance: {np.mean(bert_distances):.4f}")
    print(f"  Std pairwise distance: {np.std(bert_distances):.4f}")
    
    print(f"Original Graph embeddings:")
    print(f"  Mean pairwise distance: {np.mean(graph_distances):.4f}")
    print(f"  Std pairwise distance: {np.std(graph_distances):.4f}")
    
    # t-SNE space statistics
    bert_tsne_distances = pdist(bert_tsne)
    graph_tsne_distances = pdist(graph_tsne)
    
    print(f"t-SNE BERT embeddings:")
    print(f"  Mean pairwise distance: {np.mean(bert_tsne_distances):.4f}")
    print(f"  Std pairwise distance: {np.std(bert_tsne_distances):.4f}")
    
    print(f"t-SNE Graph embeddings:")
    print(f"  Mean pairwise distance: {np.mean(graph_tsne_distances):.4f}")
    print(f"  Std pairwise distance: {np.std(graph_tsne_distances):.4f}")
    
    # Correlation between original and t-SNE distances
    from scipy.stats import pearsonr
    
    bert_corr, _ = pearsonr(bert_distances, bert_tsne_distances)
    graph_corr, _ = pearsonr(graph_distances, graph_tsne_distances)
    
    print(f"Distance preservation (correlation):")
    print(f"  BERT: {bert_corr:.4f}")
    print(f"  Graph: {graph_corr:.4f}")
    
    print("========================\n")


def main(sample_size: int = 100, perplexity: int = 30):
    """
    Main function to create embedding comparison visualization.
    
    Args:
        sample_size: Number of samples to analyze
        perplexity: t-SNE perplexity parameter
    """
    print("=== BERT vs Graph Embeddings Comparison ===")
    
    try:
        # Connect to Neo4j
        gds = connect_to_neo4j()
        
        # Fetch sample embeddings
        bert_embeddings, graph_embeddings, texts = fetch_sample_embeddings(gds, sample_size)
        
        # Apply t-SNE reduction
        bert_tsne, graph_tsne = apply_tsne_reduction(
            bert_embeddings, graph_embeddings, perplexity=perplexity
        )
        
        # Create visualizations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        comparison_path = os.path.join(current_dir, "bert_vs_graph_embeddings_comparison.png")
        overlay_path = os.path.join(current_dir, "bert_vs_graph_embeddings_overlay.png")
        
        create_comparison_plot(bert_tsne, graph_tsne, texts, comparison_path)
        create_overlay_plot(bert_tsne, graph_tsne, overlay_path)
        
        # Analyze embeddings
        analyze_embedding_distances(bert_embeddings, graph_embeddings, bert_tsne, graph_tsne)
        
        print("=== Analysis Complete ===")
        print(f"Generated plots:")
        print(f"  - {comparison_path}")
        print(f"  - {overlay_path}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    # Run with default parameters
    main(sample_size=100, perplexity=30) 