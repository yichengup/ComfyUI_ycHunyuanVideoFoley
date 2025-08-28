# ComfyUI HunyuanVideo-Foley Custom Node

This is a ComfyUI custom node wrapper for the HunyuanVideo-Foley model, which generates realistic audio from video and text descriptions.

## Features

- **Text-Video-to-Audio Synthesis**: Generate realistic audio that matches your video content
- **Flexible Text Prompts**: Use optional text descriptions to guide audio generation
- **Multiple Samples**: Generate up to 6 different audio variations per inference
- **Configurable Parameters**: Control guidance scale, inference steps, and sampling
- **Seed Control**: Reproducible results with seed parameter
- **Model Caching**: Efficient model loading and reuse across generations
- **Automatic Model Downloads**: Models are automatically downloaded to `ComfyUI/models/foley/` when needed
- **Multiple Model Variants**: Support for different model sizes (XXL, Base, etc.)

## Installation

1. **Clone this repository** into your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone <repository_url> ComfyUI_HunyuanVideoFoley
   ```

2. **Install dependencies**:
   ```bash
   cd ComfyUI_HunyuanVideoFoley
   pip install -r requirements.txt
   ```

3. **Run the installation script** (recommended):
   ```bash
   python install.py
   ```

4. **Restart ComfyUI** to load the new nodes.

### Model Setup

The models can be obtained in two ways:

#### Option 1: Automatic Download (Recommended)
- Models will be automatically downloaded to `ComfyUI/models/foley/` when you first run the node
- No manual setup required
- Progress will be shown in the ComfyUI console

#### Option 2: Manual Download
- Download models from [HuggingFace](https://huggingface.co/tencent/HunyuanVideo-Foley)
- Place models in `ComfyUI/models/foley/` (recommended) or `./pretrained_models/` directory
- Ensure the config file is at `configs/hunyuanvideo-foley-xxl.yaml`

## Usage

### Node Types

#### 1. HunyuanVideo-Foley Generator
Main node for generating audio from video and text.

**Inputs:**
- **video**: Video input (VIDEO type)
- **text_prompt**: Text description of desired audio (STRING)
- **guidance_scale**: CFG scale for generation control (1.0-10.0, default: 4.5)
- **num_inference_steps**: Number of denoising steps (10-100, default: 50)
- **sample_nums**: Number of audio samples to generate (1-6, default: 1)
- **seed**: Random seed for reproducibility (INT)
- **model_path**: Path to pretrained models (optional, leave empty for auto-download)
- **config_path**: Path to config file (optional, leave empty for default)
- **auto_download**: Enable automatic model downloading (BOOLEAN, default: True)
- **model_variant**: Choose model variant to download (dropdown, default: "hunyuanvideo-foley-xxl")

**Outputs:**
- **video_with_audio**: Video with generated audio merged (VIDEO)
- **audio_only**: Generated audio file (AUDIO) 
- **status_message**: Generation status and info (STRING)

#### 2. HunyuanVideo-Foley Model Loader
Separate node for loading models (useful for sharing models between multiple generator nodes).

**Inputs:**
- **model_path**: Path to pretrained models
- **config_path**: Path to config file

**Outputs:**
- **model**: Model handle for use with other nodes
- **status_message**: Loading status

## Example Workflow

1. **Load Video**: Use a "Load Video" node to input your video file
2. **Add Generator**: Add the "HunyuanVideo-Foley Generator" node
3. **Connect Video**: Connect the video output to the generator's video input
4. **Set Prompt**: Enter a text description (e.g., "A person walks on frozen ice")
5. **Adjust Settings**: Configure guidance scale, steps, and sample count as needed
6. **Generate**: Run the workflow to generate audio

## Model Requirements

The node expects the following model structure:
```
pretrained_models/
├── hunyuanvideo_foley.pth          # Main Foley model
├── vae_128d_48k.pth                # DAC VAE model  
└── synchformer_state_dict.pth      # Synchformer model

configs/
└── hunyuanvideo-foley-xxl.yaml     # Configuration file
```

## Performance Notes

- **VRAM Usage**: The model requires significant GPU memory (recommended 16GB+ VRAM)
- **Generation Time**: Audio generation can take several minutes depending on video length and settings
- **Model Loading**: First run will download and cache models from HuggingFace if not present locally

## Troubleshooting

### Common Issues:

1. **"Failed to import HunyuanVideo-Foley modules"**
   - Ensure the parent HunyuanVideo-Foley package is properly installed
   - Check that all dependencies are installed correctly

2. **"Model path does not exist"**
   - Verify the model files are downloaded and in the correct directory
   - Check the model_path parameter matches your model location

3. **CUDA out of memory**
   - Reduce the number of samples (`sample_nums`)
   - Lower the number of inference steps
   - Use CPU if GPU memory is insufficient

4. **"Config path does not exist"**
   - Ensure the config file is in the expected location
   - Verify the config_path parameter is correct

## License

This custom node is based on the HunyuanVideo-Foley project. Please check the original project's license terms.

## Credits

Based on the HunyuanVideo-Foley project by Tencent. Original paper and code available at:
- Paper: [HunyuanVideo-Foley: Text-Video-to-Audio Synthesis]
- Code: [https://github.com/tencent/HunyuanVideo-Foley]