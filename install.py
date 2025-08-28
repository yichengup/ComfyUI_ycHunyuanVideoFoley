"""
Installation script for ComfyUI HunyuanVideo-Foley Custom Node
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_and_install_requirements():
    """Check and install required packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("Requirements file not found!")
        return False
    
    try:
        print("Checking and installing requirements...")
        
        # Read requirements
        with open(requirements_file, 'r') as f:
            requirements = f.read().splitlines()
        
        # Filter out comments and empty lines
        requirements = [line.strip() for line in requirements 
                       if line.strip() and not line.strip().startswith('#')]
        
        # Install packages
        for requirement in requirements:
            try:
                # Skip git+ requirements for now (they need special handling)
                if requirement.startswith('git+'):
                    print(f"Installing git requirement: {requirement}")
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement])
                else:
                    # Check if package is already installed
                    try:
                        pkg_resources.require([requirement])
                        print(f"✓ {requirement} already installed")
                    except pkg_resources.DistributionNotFound:
                        print(f"Installing {requirement}...")
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement])
                    except pkg_resources.VersionConflict:
                        print(f"Updating {requirement}...")
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', requirement])
                        
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {requirement}: {e}")
                return False
        
        print("✅ All requirements installed successfully!")
        return True
        
    except Exception as e:
        print(f"Error installing requirements: {e}")
        return False

def setup_model_directories():
    """Create necessary model directories"""
    base_dir = Path(__file__).parent.parent.parent  # Go up to ComfyUI root
    
    # Create ComfyUI/models/foley directory for automatic downloads
    foley_models_dir = base_dir / "models" / "foley"
    foley_models_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created ComfyUI models directory: {foley_models_dir}")
    
    # Also create local fallback directories
    node_dir = Path(__file__).parent
    local_dirs = [
        node_dir / "pretrained_models",
        node_dir / "configs"
    ]
    
    for dir_path in local_dirs:
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created local directory: {dir_path}")

def main():
    """Main installation function"""
    print("🚀 Installing ComfyUI HunyuanVideo-Foley Custom Node...")
    
    # Install requirements
    if not check_and_install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Setup directories
    setup_model_directories()
    
    print("📁 Directory structure created")
    print("📋 Installation completed!")
    print()
    print("📌 Next steps:")
    print("1. Restart ComfyUI to load the custom nodes")
    print("2. Models will be automatically downloaded when you first use the node")
    print("3. Alternatively, manually download models and place them in ComfyUI/models/foley/")
    print("4. Model URLs are configured in model_urls.py (can be updated as needed)")
    print()
    
    return True

if __name__ == "__main__":
    main()