# ComfyUI HunyuanVideo-Foley Custom Node

This is a ComfyUI custom node wrapper for the HunyuanVideo-Foley model, which generates realistic audio from video and text descriptions.


## ✨大佬的原插件已更新，安装原插件就行 (https://github.com/if-ai/ComfyUI_HunyuanVideoFoley)
## The original plug-in of the boss has been updated, just install the original plug-in
<img width="1006" height="660" alt="image" src="https://github.com/user-attachments/assets/bb7e392b-7832-42c8-ba32-a3183769e276" />
## 模型卸载 Model Unloading
<img width="804" height="542" alt="bd36754d74ffe001890d76cfd0b4211" src="https://github.com/user-attachments/assets/043c74ed-1e8e-42a3-8702-5abb7c73bd23" />




## Features

- **Text-Video-to-Audio Synthesis**: Generate realistic audio that matches your video content
- **Flexible Text Prompts**: Use optional text descriptions to guide audio generation
- **Multiple Samples**: Generate up to 6 different audio variations per inference
- **Configurable Parameters**: Control guidance scale, inference steps, and sampling
- **Seed Control**: Reproducible results with seed parameter
- **Model Caching**: Efficient model loading and reuse across generations
- **Automatic Model Downloads**: Models are automatically downloaded to `ComfyUI/models/foley/` when needed

## Installation

1. **Clone this repository** into your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yichengup/ComfyUI_ycHunyuanVideoFoley.git
   ```

2. **Install dependencies**:
   ```bash
   cd ComfyUI_ycHunyuanVideoFoley
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

**Outputs:**
- **video_with_audio**: Video with generated audio merged (VIDEO)
- **audio_only**: Generated audio file (AUDIO) 
- **status_message**: Generation status and info (STRING)

## ⚠ Important Limitations

### **Frame Count & Duration Limits**
- **Maximum Frames**: 450 frames (hard limit)
- **Maximum Duration**: 15 seconds at 30fps
- **Recommended**: Keep videos ≤15 seconds for best results

### **FPS Recommendations**
- **30fps**: Max 15 seconds (450 frames)
- **24fps**: Max 18.75 seconds (450 frames)  
- **15fps**: Max 30 seconds (450 frames)

### **Long Video Solutions**
For videos longer than 15 seconds:
1. **Reduce FPS**: Lower FPS allows longer duration within frame limit
2. **Segment Processing**: Split long videos into 15s segments
3. **Audio Merging**: Combine generated audio segments in post-processing


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


## License

This custom node is based on the HunyuanVideo-Foley project. Please check the original project's license terms.

## Credits

Based on the HunyuanVideo-Foley project by Tencent. Original paper and code available at:
- Paper: [HunyuanVideo-Foley: Text-Video-to-Audio Synthesis]

- Code: [https://github.com/tencent/HunyuanVideo-Foley]












