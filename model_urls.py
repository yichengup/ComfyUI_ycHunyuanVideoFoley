# HunyuanVideo-Foley Model URLs Configuration
# Update these URLs with the actual download links for the models

MODEL_URLS = {
    "hunyuanvideo-foley-xxl": {
        "models": [
            {
                "url": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/hunyuanvideo_foley.pth",
                "filename": "hunyuanvideo_foley.pth",
                "description": "Main HunyuanVideo-Foley model"
            },
            {
                "url": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/synchformer_state_dict.pth",
                "filename": "synchformer_state_dict.pth",
                "description": "Synchformer model weights"
            },
            {
                "url": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/vae_128d_48k.pth",
                "filename": "vae_128d_48k.pth",
                "description": "VAE model weights"
            }
        ],
        "extracted_dir": "hunyuanvideo-foley-xxl",
        "description": "HunyuanVideo-Foley XXL model for audio generation"
    }
}

# Alternative mirror URLs (if main URLs fail)
MIRROR_URLS = {
    # Add mirror download sources here if needed
}

def get_model_url(model_name: str, use_mirror: bool = False) -> dict:
    """Get model URL configuration"""
    urls_dict = MIRROR_URLS if use_mirror else MODEL_URLS
    return urls_dict.get(model_name, {})

def list_available_models() -> list:
    """List all available model names"""
    return list(MODEL_URLS.keys())