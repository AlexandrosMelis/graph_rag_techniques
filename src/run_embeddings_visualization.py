#!/usr/bin/env python3
"""
Runner script for BERT vs Graph embeddings visualization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize_embeddings_comparison import main

if __name__ == "__main__":
    try:
        print("Starting BERT vs Graph embeddings visualization...")
        print("This will:")
        print("1. Connect to Neo4j database")
        print("2. Fetch 100 samples with both BERT and graph embeddings")
        print("3. Apply t-SNE dimensionality reduction")
        print("4. Create comparison plots")
        print("5. Save plots in the current directory")
        print("-" * 50)
        
        # Run the main visualization function
        main(sample_size=100, perplexity=30)
        
        print("-" * 50)
        print("Visualization completed successfully!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure Neo4j is running and accessible")
        print("2. Check that both BERT and graph embeddings exist in the database")
        print("3. Verify the database has at least 100 nodes with both embedding types")
        print("4. Make sure all required packages are installed")
        sys.exit(1) 