"""
Test script to verify dual projection model setup and loading.
"""

import os
import sys
sys.path.append('src')

def test_dual_projection_setup():
    """Test dual projection model setup."""
    print("🔧 Testing Dual Projection Model Setup")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from configs import ConfigPath
        from run_dual_projection_evaluation import find_latest_dual_projection_model
        from retrieval_techniques.dual_projection_retriever import DualProjectionRetriever
        from projection_models.dual_projection_model import load_dual_projection_model
        print("✅ All imports successful")
        
        # Test finding the model
        print("\n🔍 Testing dual projection model finding...")
        try:
            latest_dir, best_model_path = find_latest_dual_projection_model()
            print(f"✅ Found latest model directory: {os.path.basename(latest_dir)}")
            print(f"✅ Best model path: {best_model_path}")
            print(f"✅ Model file exists: {os.path.exists(best_model_path)}")
            
            if os.path.exists(best_model_path):
                # Test loading the model
                print("\n🏗️ Testing model loading...")
                model, checkpoint = load_dual_projection_model(best_model_path, device="cpu")
                print(f"✅ Model loaded successfully")
                print(f"   Semantic dim: {model.dim_sem}")
                print(f"   Graph dim: {model.dim_graph}")
                print(f"   Hidden dims: {model.hidden_dims}")
                
                # Test checkpoint info
                if 'model_config' in checkpoint:
                    print(f"✅ Model config found in checkpoint")
                if 'training_config' in checkpoint:
                    print(f"✅ Training config found in checkpoint")
                    training_config = checkpoint['training_config']
                    print(f"   Semantic weight: {training_config.get('semantic_loss_weight', 'N/A')}")
                    print(f"   Graph weight: {training_config.get('graph_loss_weight', 'N/A')}")
                
                return True
            else:
                print("❌ Model file does not exist")
                return False
                
        except Exception as e:
            print(f"❌ Error finding/loading model: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_dual_projection_setup()
    if success:
        print("\n🎉 All tests passed! Ready for evaluation.")
    else:
        print("\n💥 Some tests failed. Please check the setup.") 