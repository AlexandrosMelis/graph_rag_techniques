#!/usr/bin/env python3
"""
Test script for the updated GNN embedding generation functionality.
This script demonstrates various ways to use the enhanced write_graph_embeddings_to_neo4j function.
"""

import os
import sys
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from graph_embeddings.compute_gnn_embeddings import (
    write_graph_embeddings_to_neo4j,
    find_latest_model
)
from configs.config import ConfigPath


def test_find_latest_model():
    """Test the find_latest_model function."""
    print("🔍 Testing find_latest_model function...")
    
    try:
        latest_model = find_latest_model(ConfigPath.MODELS_DIR)
        if latest_model:
            print(f"✅ Found latest model: {latest_model}")
            return latest_model
        else:
            print("⚠️  No enhanced models found")
            return None
    except Exception as e:
        print(f"❌ Error finding latest model: {e}")
        return None


def test_embedding_generation_auto():
    """Test embedding generation with automatic model detection."""
    print("\n🧪 Testing embedding generation with auto-detection...")
    
    try:
        write_graph_embeddings_to_neo4j(
            model_path=None,  # Auto-detect latest model
            graph_name="contexts",
            use_auto_device=True,
            batch_size=100  # Smaller batch for testing
        )
        print("✅ Auto-detection test passed!")
        return True
    except Exception as e:
        print(f"❌ Auto-detection test failed: {e}")
        traceback.print_exc()
        return False


def test_embedding_generation_manual(model_path: str):
    """Test embedding generation with manual model path."""
    print(f"\n🧪 Testing embedding generation with manual path: {model_path}")
    
    try:
        write_graph_embeddings_to_neo4j(
            model_path=model_path,
            graph_name="contexts",
            use_auto_device=True,
            batch_size=50  # Small batch for testing
        )
        print("✅ Manual path test passed!")
        return True
    except Exception as e:
        print(f"❌ Manual path test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 GNN Embedding Generation Test Suite")
    print("=" * 60)
    
    # Test 1: Find latest model
    latest_model = test_find_latest_model()
    
    # Test 2: Auto-detection
    auto_success = test_embedding_generation_auto()
    
    # Test 3: Manual path (if model found)
    manual_success = True
    if latest_model:
        manual_success = test_embedding_generation_manual(latest_model)
    else:
        print("\n⚠️  Skipping manual path test (no model found)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Find latest model: {'✅ PASS' if latest_model else '⚠️  SKIP'}")
    print(f"   Auto-detection: {'✅ PASS' if auto_success else '❌ FAIL'}")
    print(f"   Manual path: {'✅ PASS' if manual_success else '❌ FAIL'}")
    
    overall_success = auto_success and manual_success
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if not overall_success:
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure you have trained a GNN model first")
        print("   2. Check your Neo4j connection settings")
        print("   3. Ensure the 'contexts' graph projection exists")
        print("   4. Verify CUDA availability if using GPU")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 