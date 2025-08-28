# Installation Guide for ComfyUI HunyuanVideo-Foley Custom Node

## Overview

This custom node wraps the HunyuanVideo-Foley model for use in ComfyUI, enabling text-video-to-audio synthesis directly within ComfyUI workflows.

## Prerequisites

- ComfyUI installation
- Python 3.8+ 
- CUDA-capable GPU (recommended 16GB+ VRAM)
- At least 32GB system RAM

## Step-by-Step Installation

### 1. Clone the Custom Node

```bash
cd /path/to/ComfyUI/custom_nodes
git clone <repository_url> ComfyUI_HunyuanVideoFoley
cd ComfyUI_HunyuanVideoFoley
```

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or run the installation script
python install.py
```

### 3. Download Models

You need to download the HunyuanVideo-Foley models:

```bash
# Option 1: Manual download from HuggingFace
# Visit: https://huggingface.co/tencent/HunyuanVideo-Foley
# Download and place files in the following structure:

mkdir -p pretrained_models
mkdir -p configs

# Download these files to pretrained_models/:
# - hunyuanvideo_foley.pth
# - vae_128d_48k.pth  
# - synchformer_state_dict.pth

# Download config file to configs/:
# - hunyuanvideo-foley-xxl.yaml
```

Or use the Gradio app's auto-download feature by setting the environment variable:

```bash
export HIFI_FOLEY_MODEL_PATH="/path/to/ComfyUI/custom_nodes/ComfyUI_HunyuanVideoFoley/pretrained_models"
```

### 4. Verify Installation

```bash
# Run the test script to verify everything is working
python test_node.py
```

### 5. Restart ComfyUI

After installation, restart ComfyUI to load the new custom nodes.

## Expected Directory Structure

After installation, your directory should look like this:

```
ComfyUI_HunyuanVideoFoley/
├── __init__.py
├── nodes.py
├── utils.py
├── requirements.txt
├── install.py
├── test_node.py
├── example_workflow.json
├── README.md
├── INSTALLATION_GUIDE.md
└── pyproject.toml

pretrained_models/
├── hunyuanvideo_foley.pth
├── vae_128d_48k.pth
└── synchformer_state_dict.pth

configs/
└── hunyuanvideo-foley-xxl.yaml
```

## Usage

### Nodes Available

1. **HunyuanVideo-Foley Generator**
   - Main node for audio generation
   - Inputs: video, text prompt, generation parameters
   - Outputs: video with audio, audio only, status message

2. **HunyuanVideo-Foley Model Loader** 
   - Separate model loading node
   - Useful for sharing models between multiple generator nodes
   - Inputs: model path, config path
   - Outputs: model handle, status message

### Basic Workflow

1. Load a video using ComfyUI's video input nodes
2. Add the "HunyuanVideo-Foley Generator" node
3. Connect the video to the generator
4. Set your text prompt (e.g., "A person walks on frozen ice")
5. Adjust parameters as needed
6. Run the workflow

### Example Workflow

An example workflow is provided in `example_workflow.json`. Load this file in ComfyUI to see a basic setup.

## Performance Tips

- **VRAM Usage**: The model requires significant GPU memory. If you encounter CUDA out of memory errors:
  - Reduce `sample_nums` parameter
  - Lower `num_inference_steps`
  - Use CPU mode (slower but works with less memory)

- **Generation Time**: Audio generation can take several minutes depending on:
  - Video length
  - Number of inference steps
  - Number of samples generated
  - Hardware specifications

## Troubleshooting

### Common Issues

1. **"Failed to import HunyuanVideo-Foley modules"**
   ```bash
   # Make sure you're in the correct directory and have all dependencies
   pip install -r requirements.txt
   ```

2. **"Model path does not exist"** 
   ```bash
   # Download models from HuggingFace and verify directory structure
   ls pretrained_models/
   # Should show: hunyuanvideo_foley.pth, vae_128d_48k.pth, synchformer_state_dict.pth
   ```

3. **CUDA out of memory**
   ```bash
   # Reduce memory usage by adjusting parameters:
   # - Lower sample_nums to 1
   # - Reduce num_inference_steps to 25
   # - Use shorter videos for testing
   ```

4. **Slow generation**
   ```bash
   # Normal for first run (model loading)
   # Subsequent runs should be faster
   # Consider using fewer inference steps for faster results
   ```

### Getting Help

- Check the test script output: `python test_node.py`
- Review ComfyUI console output for detailed error messages
- Ensure all model files are downloaded correctly
- Verify GPU memory availability

## Model Information

The HunyuanVideo-Foley model consists of several components:

- **Main Foley Model**: Core text-video-to-audio generation
- **DAC VAE**: Audio encoding/decoding  
- **SigLIP2**: Visual feature extraction
- **CLAP**: Text feature extraction
- **Synchformer**: Video-audio synchronization

All components are automatically loaded when using the custom node.

## License & Credits

This custom node is based on the HunyuanVideo-Foley project by Tencent. Please respect the original project's license terms when using this implementation.

Original project: https://github.com/tencent/HunyuanVideo-Foley