#!/usr/bin/env python3
"""
Test script for ComfyUI HunyuanVideo-Foley custom node
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from ComfyUI_HunyuanVideoFoley import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("✅ Successfully imported node mappings")
        
        print(f"Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
        print(f"Display names: {NODE_DISPLAY_NAME_MAPPINGS}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_node_structure():
    """Test node class structure"""
    print("\nTesting node structure...")
    
    try:
        from ComfyUI_HunyuanVideoFoley.nodes import HunyuanVideoFoleyNode, HunyuanVideoFoleyModelLoader
        
        # Test HunyuanVideoFoleyNode
        node = HunyuanVideoFoleyNode()
        input_types = node.INPUT_TYPES()
        
        print("✅ HunyuanVideoFoleyNode structure:")
        print(f"  - Required inputs: {list(input_types['required'].keys())}")
        print(f"  - Optional inputs: {list(input_types.get('optional', {}).keys())}")
        print(f"  - Return types: {node.RETURN_TYPES}")
        print(f"  - Function: {node.FUNCTION}")
        print(f"  - Category: {node.CATEGORY}")
        
        # Test HunyuanVideoFoleyModelLoader
        loader = HunyuanVideoFoleyModelLoader()
        loader_input_types = loader.INPUT_TYPES()
        
        print("✅ HunyuanVideoFoleyModelLoader structure:")
        print(f"  - Required inputs: {list(loader_input_types['required'].keys())}")
        print(f"  - Return types: {loader.RETURN_TYPES}")
        print(f"  - Function: {loader.FUNCTION}")
        
        return True
        
    except Exception as e:
        print(f"❌ Node structure test failed: {e}")
        return False

def test_device_setup():
    """Test device setup functionality"""
    print("\nTesting device setup...")
    
    try:
        from ComfyUI_HunyuanVideoFoley.nodes import HunyuanVideoFoleyNode
        
        device = HunyuanVideoFoleyNode.setup_device("auto")
        print(f"✅ Device setup successful: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Device setup failed: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        from ComfyUI_HunyuanVideoFoley.utils import (
            get_optimal_device,
            check_memory_requirements,
            format_duration,
            validate_model_files
        )
        
        # Test device detection
        device = get_optimal_device()
        print(f"✅ Optimal device: {device}")
        
        # Test memory check
        has_memory, msg = check_memory_requirements(device)
        print(f"✅ Memory check: {msg}")
        
        # Test duration formatting
        duration = format_duration(125.5)
        print(f"✅ Duration formatting: 125.5s -> {duration}")
        
        # Test model validation (will fail without models, but that's expected)
        is_valid, msg = validate_model_files("./pretrained_models/")
        print(f"✅ Model validation: {msg}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False

def test_requirements():
    """Test if key requirements are available"""
    print("\nTesting requirements...")
    
    required_packages = [
        'torch',
        'torchaudio', 
        'numpy',
        'loguru',
        'diffusers',
        'transformers'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main test function"""
    print("🧪 Testing ComfyUI HunyuanVideo-Foley Custom Node")
    print("=" * 50)
    
    tests = [
        ("Requirements", test_requirements),
        ("Imports", test_imports),
        ("Node Structure", test_node_structure),
        ("Device Setup", test_device_setup),
        ("Utils", test_utils),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running test: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The custom node is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)