"""
Deployment Verification Script
Tests if the package is ready for deployment/GitHub upload
"""

import sys
from pathlib import Path
import importlib.util

def check_file(filepath, required=True):
    """Check if a file exists"""
    exists = filepath.exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = "REQUIRED" if required else "optional"
    print(f"{status} {filepath.name:40s} ({req_text})")
    return exists

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        spec = importlib.util.find_spec(module_name)
        exists = spec is not None
        status = "✅" if exists else "❌"
        print(f"{status} {module_name:40s}")
        return exists
    except:
        print(f"❌ {module_name:40s}")
        return False

def main():
    print("\n" + "="*70)
    print("DEPLOYMENT VERIFICATION")
    print("="*70)
    
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    # Check required files
    print("\n📁 Checking Required Files:")
    print("-" * 70)
    
    required_files = [
        SCRIPT_DIR / "model" / "best_model.ckpt",
        SCRIPT_DIR / "fire_prediction" / "__init__.py",
        SCRIPT_DIR / "fire_prediction" / "models" / "physics_informed.py",
        SCRIPT_DIR / "fire_prediction" / "utils" / "physics.py",
        SCRIPT_DIR / "predict.py",
        SCRIPT_DIR / "batch_predict.py",
        SCRIPT_DIR / "requirements.txt",
    ]
    
    optional_files = [
        SCRIPT_DIR / "README.md",
        SCRIPT_DIR / "DEPLOYMENT_README.md",
        SCRIPT_DIR / "Input" / "EXTREME_TEST_5719_hrr.csv",
        SCRIPT_DIR / "Input" / "test_11cm_mesh_hrr.csv",
    ]
    
    required_ok = all(check_file(f, required=True) for f in required_files)
    print()
    for f in optional_files:
        check_file(f, required=False)
    
    # Check Python dependencies
    print("\n📦 Checking Python Dependencies:")
    print("-" * 70)
    
    dependencies = [
        "torch",
        "pytorch_lightning", 
        "numpy",
        "pandas",
        "matplotlib",
    ]
    
    deps_ok = all(check_import(dep) for dep in dependencies)
    
    # Test imports
    print("\n🔧 Testing Module Imports:")
    print("-" * 70)
    
    sys.path.insert(0, str(SCRIPT_DIR))
    
    try:
        from fire_prediction.models.physics_informed import PhysicsInformedLSTM
        print("✅ PhysicsInformedLSTM import successful")
        import_ok = True
    except Exception as e:
        print(f"❌ PhysicsInformedLSTM import failed: {e}")
        import_ok = False
    
    try:
        from fire_prediction.utils.physics import compute_heskestad_features
        print("✅ compute_heskestad_features import successful")
    except Exception as e:
        print(f"❌ compute_heskestad_features import failed: {e}")
        import_ok = False
    
    # Test model loading
    print("\n🧠 Testing Model Loading:")
    print("-" * 70)
    
    model_ok = False
    try:
        import torch
        from fire_prediction.models.physics_informed import PhysicsInformedLSTM
        
        MODEL_PATH = SCRIPT_DIR / "model" / "best_model.ckpt"
        
        if MODEL_PATH.exists():
            checkpoint = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
            print(f"✅ Checkpoint loaded successfully")
            print(f"   Keys: {list(checkpoint.keys())}")
            print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}")
            
            # Try loading model
            model = PhysicsInformedLSTM(
                input_dim=6,
                hidden_dim=128,
                num_layers=2,
                output_dim=3,
                dropout=0.1,
                lr=0.001,
                pred_horizon=10,
            )
            
            state_dict = checkpoint['model_state_dict']
            if 'fc.weight' in state_dict and 'head.weight' not in state_dict:
                state_dict['head.weight'] = state_dict.pop('fc.weight')
                state_dict['head.bias'] = state_dict.pop('fc.bias')
            
            model.load_state_dict(state_dict)
            model.eval()
            print(f"✅ Model initialized and loaded successfully")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            model_ok = True
        else:
            print(f"❌ Checkpoint not found: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
    
    # Test from different directory
    print("\n📍 Testing Portable Paths:")
    print("-" * 70)
    
    import os
    original_dir = Path.cwd()
    try:
        # Change to parent directory
        parent_dir = SCRIPT_DIR.parent
        os.chdir(parent_dir)
        print(f"✅ Changed directory to: {Path.cwd()}")
        
        # Try importing again
        sys.path.insert(0, str(SCRIPT_DIR))
        from fire_prediction.models.physics_informed import PhysicsInformedLSTM
        print(f"✅ Import still works from different directory")
        
        # Check if model path still resolves
        MODEL_PATH = SCRIPT_DIR / "model" / "best_model.ckpt"
        if MODEL_PATH.exists():
            print(f"✅ Model path resolves correctly: {MODEL_PATH}")
            portable_ok = True
        else:
            print(f"❌ Model path broken: {MODEL_PATH}")
            portable_ok = False
            
    except Exception as e:
        print(f"❌ Portability test failed: {e}")
        portable_ok = False
    finally:
        os.chdir(original_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    checks = [
        ("Required Files", required_ok),
        ("Dependencies", deps_ok),
        ("Module Imports", import_ok),
        ("Model Loading", model_ok),
        ("Portable Paths", portable_ok),
    ]
    
    all_ok = all(status for _, status in checks)
    
    for name, status in checks:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {name:30s} {'PASS' if status else 'FAIL'}")
    
    print("="*70)
    
    if all_ok:
        print("\n🎉 ALL CHECKS PASSED - READY FOR DEPLOYMENT!")
        print("\nNext steps:")
        print("1. Review DEPLOYMENT_README.md")
        print("2. Create/update .gitignore")
        print("3. Git add, commit, push to GitHub")
        print("4. Share repository link")
    else:
        print("\n⚠️  SOME CHECKS FAILED - REVIEW ISSUES ABOVE")
        print("\nFix the issues before deploying.")
    
    print("="*70 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
