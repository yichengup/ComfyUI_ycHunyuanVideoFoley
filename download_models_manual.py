#!/usr/bin/env python3
"""
Manual download helper for HunyuanVideo-Foley models
Run this script directly if the automatic download fails in ComfyUI
"""

import os
import sys
from pathlib import Path
import urllib.request
import time

# Model download URLs
MODELS = [
    {
        "url": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/hunyuanvideo_foley.pth",
        "filename": "hunyuanvideo_foley.pth",
        "size_gb": 10.3,
        "description": "Main HunyuanVideo-Foley model"
    },
    {
        "url": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/synchformer_state_dict.pth",
        "filename": "synchformer_state_dict.pth",
        "size_gb": 0.95,
        "description": "Synchformer model weights"
    },
    {
        "url": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/vae_128d_48k.pth",
        "filename": "vae_128d_48k.pth",
        "size_gb": 1.49,
        "description": "VAE model weights"
    }
]

def download_with_progress(url, dest_path):
    """Download file with progress display"""
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, (downloaded * 100) // total_size)
            size_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            
            # Print progress
            bar_len = 40
            filled_len = int(bar_len * percent // 100)
            bar = '█' * filled_len + '-' * (bar_len - filled_len)
            
            sys.stdout.write(f'\r[{bar}] {percent}% ({size_mb:.1f}/{total_mb:.1f} MB)')
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\nError: {e}")
        return False

def main():
    # Determine ComfyUI models directory
    comfyui_root = Path(__file__).parent.parent.parent  # Go up to ComfyUI root
    models_dir = comfyui_root / "models" / "foley" / "hunyuanvideo-foley-xxl"
    
    print("=" * 60)
    print("HunyuanVideo-Foley Model Downloader")
    print("=" * 60)
    print(f"\nModels will be downloaded to:")
    print(f"  {models_dir}")
    
    # Create directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check disk space
    import shutil
    stat = shutil.disk_usage(models_dir)
    available_gb = stat.free / (1024 ** 3)
    required_gb = sum(m["size_gb"] for m in MODELS) + 1  # Add 1GB buffer
    
    print(f"\nDisk space available: {available_gb:.1f} GB")
    print(f"Space required: {required_gb:.1f} GB")
    
    if available_gb < required_gb:
        print(f"\n⚠️ WARNING: Insufficient disk space!")
        print(f"Please free up at least {required_gb - available_gb:.1f} GB before continuing.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\n" + "=" * 60)
    
    # Download each model
    for i, model_info in enumerate(MODELS, 1):
        model_path = models_dir / model_info["filename"]
        
        print(f"\n[{i}/{len(MODELS)}] {model_info['description']}")
        print(f"  File: {model_info['filename']} ({model_info['size_gb']} GB)")
        
        # Check if already downloaded
        if model_path.exists() and model_path.stat().st_size > 100 * 1024 * 1024:
            size_gb = model_path.stat().st_size / (1024 ** 3)
            print(f"  ✓ Already downloaded ({size_gb:.2f} GB)")
            continue
        
        print(f"  Downloading from: {model_info['url']}")
        
        # Try downloading with retries
        max_retries = 3
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"  Retry {attempt}/{max_retries - 1}...")
                time.sleep(5)
            
            success = download_with_progress(model_info["url"], model_path)
            if success:
                size_gb = model_path.stat().st_size / (1024 ** 3)
                print(f"  ✓ Downloaded successfully ({size_gb:.2f} GB)")
                break
            else:
                if attempt == max_retries - 1:
                    print(f"  ✗ Failed to download after {max_retries} attempts")
                    print(f"\n  You can manually download from:")
                    print(f"    {model_info['url']}")
                    print(f"  And place it at:")
                    print(f"    {model_path}")
    
    print("\n" + "=" * 60)
    print("Download process completed!")
    print("\nYou can now use the HunyuanVideo-Foley node in ComfyUI.")
    print("=" * 60)

if __name__ == "__main__":
    main()