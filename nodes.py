import os
import torch
import torchaudio
import tempfile
import numpy as np
from loguru import logger
from typing import Optional, Tuple, Any
import folder_paths
import random
import urllib.request
import zipfile
from pathlib import Path
from datetime import datetime
import shutil
import time
import math
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Import ComfyUI video types
try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    try:
        # Fallback to latest API location
        from comfy_api.latest._input_impl.video_types import VideoFromFile
    except ImportError:
        logger.warning("VideoFromFile not available, will return file paths only")
        VideoFromFile = None

# Import ComfyUI ProgressBar for real-time progress display
try:
    from comfy.utils import ProgressBar
    HAS_PROGRESS_BAR = True
except ImportError:
    HAS_PROGRESS_BAR = False
    logger.warning("ProgressBar not available, progress display will be disabled")

# Add foley models directory to ComfyUI folder paths
foley_models_dir = os.path.join(folder_paths.models_dir, "foley")
if "foley" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["foley"] = ([foley_models_dir], folder_paths.supported_pt_extensions)

# Import model URLs configuration
try:
    from .model_urls import MODEL_URLS, get_model_url, list_available_models
except ImportError:
    logger.warning("model_urls.py not found, using default URLs")
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
    
# Import the HunyuanVideo-Foley modules
try:
    from hunyuanvideo_foley.utils.model_utils import load_model, denoise_process
    from hunyuanvideo_foley.utils.feature_utils import feature_process as original_feature_process
    from hunyuanvideo_foley.utils.media_utils import merge_audio_video
except ImportError as e:
    logger.error(f"Failed to import HunyuanVideo-Foley modules: {e}")
    logger.error("Make sure the HunyuanVideo-Foley package is installed and accessible")
    raise

class YCHunyuanVideoFoley:
    """
    ComfyUI Node for HunyuanVideo-Foley: Generate audio from image frames and text prompts
    """
    
    # Class variables for model caching
    _model_dict = None
    _cfg = None
    _device = None
    _model_path = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Changed from VIDEO to IMAGE to support frame sequences
                "text_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "A person walks on frozen ice",
                    "placeholder": "Describe the audio you want to generate..."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 5
                }),
                "sample_nums": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 6,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Additional negative prompts (optional). Will be combined with built-in quality controls."
                }),
                "fps": (
                    "FLOAT",  # 改为FLOAT类型以支持小数帧率，与VideoHelperSuite兼容
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}
                ),
                "output_folder": ("STRING", {
                    "default": "hunyuan_foley",
                    "multiline": False,
                    "placeholder": "Subfolder name in ComfyUI/output/"
                }),
                "filename_prefix": ("STRING", {
                    "default": "foley_",
                    "multiline": False,
                    "placeholder": "Prefix for output filename"
                }),
            },
            # Keep operational parameters out of the UI
            "hidden": {
                # Force auto-download flow with paired defaults
                "model_path": "",
                "config_path": "",
                "auto_download": True,
                "model_variant": "hunyuanvideo-foley-xxl",
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")  # Changed from STRING to IMAGE for frame output
    RETURN_NAMES = ("video_frames", "audio", "status_message")  # Updated return names
    FUNCTION = "generate_audio"
    CATEGORY = "HunyuanVideo-Foley"
    DESCRIPTION = "Generate realistic audio from image frames and text descriptions using HunyuanVideo-Foley. Supports IMAGE sequence input, custom negative prompts, and outputs frames with audio."

    @classmethod
    def _required_model_filenames(cls, model_variant: str) -> Tuple[str, ...]:
        """Return tuple of filenames required for a given variant."""
        info = MODEL_URLS.get(model_variant, MODEL_URLS.get("hunyuanvideo-foley-xxl", {}))
        if "models" in info:
            return tuple(m["filename"] for m in info["models"])
        # Fallback generic names
        return ("hunyuanvideo_foley.pth", "synchformer_state_dict.pth", "vae_128d_48k.pth")

    @staticmethod
    def _string_looks_like_video_path(value: str) -> bool:
        try:
            value_l = value.lower()
            return value_l.endswith((".mp4", ".mov", ".webm", ".mkv", ".avi")) and len(value_l) > 4
        except Exception:
            return False

    @classmethod
    def _extract_frames_from_image_input(cls, images: Any) -> Tuple[bool, Optional[list], Optional[str], str]:
        """Extract frames from ComfyUI IMAGE input (can be single image or sequence)."""
        try:
            frames = []
            
            # Handle torch tensor input (most common case)
            if isinstance(images, torch.Tensor):
                if images.ndim == 3:
                    # Single image (C, H, W) -> convert to sequence of 1
                    frames = [images]
                elif images.ndim == 4:
                    # Image sequence (B, C, H, W) -> extract each frame
                    frames = [images[i] for i in range(images.shape[0])]
                else:
                    return False, None, None, f"Unsupported tensor shape: {images.shape}"
                
                return True, frames, None, ""
            
            # Handle list/tuple of tensors
            elif isinstance(images, (list, tuple)) and len(images) > 0:
                if all(isinstance(f, torch.Tensor) for f in images):
                    frames = list(images)
                    return True, frames, None, ""
                else:
                    return False, None, None, "Mixed types in frame sequence"
            
            # Handle dict with frames key
            elif isinstance(images, dict):
                for key in ("frames", "images", "video"):
                    val = images.get(key)
                    if val is not None:
                        return cls._extract_frames_from_image_input(val)
            
            # Handle string path (fallback for compatibility)
            elif isinstance(images, str) and cls._string_looks_like_video_path(images):
                return False, None, images, "Video file path detected"
            
            return False, None, None, f"Unsupported IMAGE input type: {type(images)}"
            
        except Exception as e:
            return False, None, None, f"Error extracting frames: {str(e)}"

    @staticmethod
    def _to_uint8_frame(frame: Any) -> Optional[np.ndarray]:
        """Convert a possible torch/numpy image tensor into HxWx3 uint8."""
        try:
            import numpy as _np
            if isinstance(frame, torch.Tensor):
                arr = frame.detach().cpu()
                # Accept shapes: (C,H,W) or (H,W,C)
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = arr.permute(1, 2, 0)
                arr = arr.float()
                # Normalize if in 0..1
                if arr.max() <= 1.5:
                    arr = (arr * 255.0).clamp(0, 255)
                arr = arr.to(torch.uint8).cpu().numpy()
            else:
                arr = _np.asarray(frame)
                if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                    # Assume CHW -> HWC
                    arr = _np.transpose(arr, (1, 2, 0))
                if arr.dtype != _np.uint8:
                    arr = _np.clip(arr, 0, 255).astype(_np.uint8)
            # Ensure 3 channels
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = _np.repeat(arr, 3, axis=2)
            return arr
        except Exception as _:
            return None

    @classmethod
    def _write_temp_video(cls, frames: list, fps: float = 24.0) -> Tuple[bool, Optional[str], str]:
        """Convert IMAGE sequence frames to temporary mp4 video file."""
        try:
            if not frames or len(frames) == 0:
                return False, None, "No frames provided"
            
            # Normalize frames to list of HxWx3 uint8
            norm_frames: list[np.ndarray] = []
            
            for frame in frames:
                arr = cls._to_uint8_frame(frame)
                if arr is None:
                    return False, None, "Failed to convert a frame to uint8 format"
                norm_frames.append(arr)
            
            # Write with imageio
            try:
                import imageio
                temp_dir = tempfile.mkdtemp()
                temp_mp4 = os.path.join(temp_dir, "input_video.mp4")
                
                # Use specified FPS or default to 24fps
                writer = imageio.get_writer(temp_mp4, fps=fps, codec='libx264', quality=8)
                
                for arr in norm_frames:
                    writer.append_data(arr)
                writer.close()
                
                logger.info(f"Created temporary video with {len(norm_frames)} frames at {fps}fps")
                return True, temp_mp4, ""
                
            except ImportError as e:
                return False, None, f"imageio library not available. Please install with: pip install 'imageio[ffmpeg]'"
            except Exception as e:
                return False, None, f"Failed to write temporary video: {e}"
                
        except Exception as e:
            return False, None, f"Frame processing error: {e}"


    @classmethod
    def setup_device(cls, device_str: str = "auto", gpu_id: int = 0) -> torch.device:
        """Setup computing device"""
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{gpu_id}")
                logger.info(f"Using CUDA device: {device}")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(device_str)
            logger.info(f"Using specified device: {device}")
        
        return device

    @classmethod
    def download_with_resume(cls, url: str, dest_path: Path, progress_hook=None) -> bool:
        """Download file with resume support for large files"""
        try:
            # Check if file partially exists
            resume_header = {}
            mode = 'wb'
            resume_pos = 0
            
            partial_path = dest_path.with_suffix('.partial')
            
            if partial_path.exists():
                resume_pos = partial_path.stat().st_size
                resume_header = {'Range': f'bytes={resume_pos}-'}
                mode = 'ab'
                logger.info(f"Resuming download from {resume_pos / (1024**2):.1f}MB")
            
            # Use requests if available for better streaming
            if HAS_REQUESTS:
                response = requests.get(url, headers=resume_header, stream=True, timeout=30)
                response.raise_for_status()
                
                # Get total size
                total_size = int(response.headers.get('content-length', 0))
                if resume_pos > 0:
                    total_size += resume_pos
                
                # Download in chunks
                chunk_size = 8192 * 128  # 1MB chunks
                downloaded = resume_pos
                
                with open(partial_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Call progress hook
                            if progress_hook and total_size > 0:
                                percent = min(100, (downloaded * 100) // total_size)
                                if downloaded % (10 * 1024 * 1024) == 0 or percent >= 100:  # Log every 10MB
                                    progress_hook(downloaded // chunk_size, chunk_size, total_size)
                
                # Move to final location
                partial_path.rename(dest_path)
                return True
                
            else:
                # Fallback to urllib
                urllib.request.urlretrieve(url, partial_path, progress_hook)
                partial_path.rename(dest_path)
                return True
                
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    @classmethod
    def check_disk_space(cls, path: Path, required_gb: float = 15.0) -> Tuple[bool, str]:
        """Check if there's enough disk space for model downloads"""
        try:
            import shutil
            stat = shutil.disk_usage(path)
            available_gb = stat.free / (1024 ** 3)
            
            if available_gb < required_gb:
                return False, f"Insufficient disk space: {available_gb:.1f}GB available, {required_gb:.1f}GB required"
            return True, f"Sufficient disk space: {available_gb:.1f}GB available"
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True, "Could not verify disk space, proceeding anyway"
    
    @classmethod
    def download_models(cls, model_name: str = "hunyuanvideo-foley-xxl") -> Tuple[bool, str, str]:
        """Download models automatically if not found"""
        try:
            if model_name not in MODEL_URLS:
                return False, f"Unknown model: {model_name}", ""
            
            foley_dir = folder_paths.folder_names_and_paths.get("foley", [None])[0]
            if not foley_dir or len(foley_dir) == 0:
                return False, "Foley models directory not configured", ""
            
            models_dir = Path(foley_dir[0])
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Check disk space before downloading (models need ~12GB total)
            has_space, space_msg = cls.check_disk_space(models_dir, required_gb=15.0)
            if not has_space:
                logger.error(space_msg)
                return False, space_msg, ""
            
            model_info = MODEL_URLS[model_name]
            extracted_path = models_dir / model_info["extracted_dir"]
            
            # Create the model directory
            extracted_path.mkdir(parents=True, exist_ok=True)
            
            # Check if models are individual files instead of an archive
            if "models" in model_info:
                # First, check which models are needed and their total size
                models_to_download = []
                total_size_gb = 0
                
                for model_file_info in model_info["models"]:
                    model_file = extracted_path / model_file_info["filename"]
                    
                    # Check if file already exists with reasonable size
                    if model_file.exists() and model_file.stat().st_size > 100 * 1024 * 1024:  # > 100MB
                        logger.info(f"Model file already exists: {model_file} ({model_file.stat().st_size / (1024**3):.1f}GB)")
                        continue
                    
                    models_to_download.append(model_file_info)
                    # Estimate sizes based on known models
                    if "hunyuanvideo_foley" in model_file_info["filename"]:
                        total_size_gb += 10.5
                    elif "synchformer" in model_file_info["filename"]:
                        total_size_gb += 1.0
                    elif "vae" in model_file_info["filename"]:
                        total_size_gb += 1.5
                
                if models_to_download:
                    # Check if there's enough space for remaining downloads
                    has_space, space_msg = cls.check_disk_space(models_dir, required_gb=total_size_gb + 1)
                    if not has_space:
                        missing_models = ", ".join([m["filename"] for m in models_to_download])
                        return False, f"Insufficient disk space for remaining models ({missing_models}): {space_msg}", ""
                
                # Download individual model files
                all_downloaded = True
                for model_file_info in models_to_download:
                    model_file = extracted_path / model_file_info["filename"]
                    
                    logger.info(f"Downloading {model_file_info['description']}...")
                    logger.info(f"URL: {model_file_info['url']}")
                    logger.info(f"Destination: {model_file}")
                    
                    # Download the model file with progress
                    def progress_hook(block_num, block_size, total_size):
                        if total_size > 0:
                            percent = min(100, (block_num * block_size * 100) // total_size)
                            if block_num % 100 == 0 or percent >= 100:  # Log every 100 blocks or at completion
                                size_mb = (block_num * block_size) / (1024 * 1024)
                                total_mb = total_size / (1024 * 1024)
                                logger.info(f"Download progress: {percent}% ({size_mb:.1f} MB / {total_mb:.1f} MB)")
                    
                    # Try downloading with retries and resume support
                    max_retries = 3
                    download_success = False
                    
                    for attempt in range(max_retries):
                        try:
                            logger.info(f"Download attempt {attempt + 1}/{max_retries}")
                            
                            # Use the new download method with resume support
                            success = cls.download_with_resume(
                                model_file_info["url"], 
                                model_file, 
                                progress_hook
                            )
                            
                            if success:
                                # Verify download size
                                downloaded_size = model_file.stat().st_size
                                logger.info(f"Successfully downloaded: {model_file_info['filename']} ({downloaded_size / (1024**3):.2f}GB)")
                                download_success = True
                                break
                            else:
                                raise Exception("Download failed")
                                
                        except Exception as e:
                            logger.error(f"Download attempt {attempt + 1} failed for {model_file_info['filename']}: {e}")
                            
                            if attempt == max_retries - 1:
                                logger.error(f"Failed to download {model_file_info['filename']} after {max_retries} attempts")
                                # Don't delete partial file on final attempt - allow resume later
                            else:
                                # Wait before retry
                                wait_time = (attempt + 1) * 5
                                logger.info(f"Waiting {wait_time} seconds before retry...")
                                time.sleep(wait_time)
                    
                    # Optional fallback via Hugging Face Hub if available
                    if not download_success:
                        try:
                            from huggingface_hub import hf_hub_download
                            logger.info("Falling back to huggingface_hub download API...")
                            repo_id = "tencent/HunyuanVideo-Foley"
                            filename = model_file_info["filename"]
                            local_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(extracted_path))
                            if os.path.exists(local_path) and os.path.getsize(local_path) > 100 * 1024 * 1024:
                                logger.info(f"huggingface_hub downloaded: {local_path}")
                                download_success = True
                            else:
                                logger.error("huggingface_hub download did not produce a valid file")
                        except Exception as hub_e:
                            logger.error(f"huggingface_hub fallback failed for {model_file_info['filename']}: {hub_e}")

                    if not download_success:
                        all_downloaded = False
                
                if all_downloaded:
                    logger.info(f"All models downloaded to: {extracted_path}")
                    return True, "Models downloaded successfully", str(extracted_path)
                else:
                    return False, "Some models failed to download", ""
                    
            else:
                # Original archive download logic (keeping for compatibility)
                model_file = models_dir / model_info["filename"]
                
                # Check if already downloaded and extracted
                if extracted_path.exists() and any(extracted_path.iterdir()):
                    logger.info(f"Models already downloaded at: {extracted_path}")
                    return True, "Models already downloaded", str(extracted_path)
                
                logger.info(f"Downloading {model_name} models...")
                logger.info(f"URL: {model_info['url']}")
                logger.info(f"Destination: {model_file}")
                
                # Download the model file with progress
                def progress_hook(block_num, block_size, total_size):
                    if total_size > 0:
                        percent = min(100, (block_num * block_size * 100) // total_size)
                        if block_num % 100 == 0 or percent >= 100:  # Log every 100 blocks or at completion
                            logger.info(f"Download progress: {percent}% ({block_num * block_size} / {total_size} bytes)")
                
                urllib.request.urlretrieve(model_info["url"], model_file, progress_hook)
                
                logger.info("Download completed. Extracting...")
                
                # Extract the archive
                if model_file.suffix.lower() in ['.zip']:
                    with zipfile.ZipFile(model_file, 'r') as zip_ref:
                        zip_ref.extractall(models_dir)
                elif model_file.suffix.lower() in ['.tar.gz', '.tgz']:
                    import tarfile
                    with tarfile.open(model_file, 'r:gz') as tar_ref:
                        tar_ref.extractall(models_dir)
                else:
                    return False, f"Unsupported archive format: {model_file.suffix}", ""
                
                # Clean up downloaded archive
                model_file.unlink()
                
                logger.info(f"Models extracted to: {extracted_path}")
                return True, "Models downloaded and extracted successfully", str(extracted_path)
            
        except Exception as e:
            error_msg = f"Failed to download models: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, ""

    @classmethod
    def load_models(cls, model_path: str, config_path: str, auto_download: bool = True, model_variant: str = "hunyuanvideo-foley-xxl") -> Tuple[bool, str]:
        """Load models if not already loaded or if path changed"""
        try:
            # Set default paths if empty
            if not model_path.strip():
                # Try ComfyUI foley models directory first
                foley_models_dir = folder_paths.folder_names_and_paths.get("foley", [None])[0]
                if foley_models_dir and len(foley_models_dir) > 0:
                    # Prefer concrete subfolder for this variant if it exists
                    root_path = foley_models_dir[0]
                    expected_dir_name = MODEL_URLS.get(model_variant, {}).get("extracted_dir", "hunyuanvideo-foley-xxl")
                    candidate_path = os.path.join(root_path, expected_dir_name)
                    model_path = candidate_path if os.path.isdir(candidate_path) else root_path
                else:
                    # Fallback to custom node directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(current_dir, "pretrained_models")
            
            if not config_path.strip():
                current_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(current_dir, "configs", "hunyuanvideo-foley-xxl.yaml")
            
            # Check if models are already loaded with the same path
            if (cls._model_dict is not None and 
                cls._cfg is not None and 
                cls._model_path == model_path):
                return True, "Models already loaded"
            
            # Verify paths exist, attempt auto-download if not found
            # If a root directory is given, try to refine to the expected subdir
            if os.path.isdir(model_path):
                expected_dir_name = MODEL_URLS.get(model_variant, {}).get("extracted_dir", "hunyuanvideo-foley-xxl")
                candidate_path = os.path.join(model_path, expected_dir_name)
                if os.path.isdir(candidate_path):
                    model_path = candidate_path

            # Ensure folder exists
            os.makedirs(model_path, exist_ok=True)

            # Determine if any required files are missing
            required_files = cls._required_model_filenames(model_variant)
            missing_files = [f for f in required_files if not os.path.isfile(os.path.join(model_path, f))]

            if not os.path.exists(model_path) or not os.listdir(model_path) or missing_files:
                if auto_download:
                    if missing_files:
                        logger.info(f"Missing model files: {missing_files}")
                    logger.info(f"Attempting to download {model_variant} models automatically into ComfyUI/models/foley...")
                    
                    download_success, download_message, downloaded_path = cls.download_models(model_variant)
                    if download_success and downloaded_path:
                        model_path = downloaded_path
                        logger.info(f"Using downloaded models at: {model_path}")
                    else:
                        return False, f"Model path does not exist and auto-download failed: {download_message}. Please manually place models in ComfyUI/models/foley/ or specify a valid path."
                else:
                    return False, f"Model path does not exist: {model_path}. Please place models in ComfyUI/models/foley/ or specify a valid path. Auto-download is disabled."
            
            if not os.path.exists(config_path):
                return False, f"Config path does not exist: {config_path}"
            
            # Setup device
            cls._device = cls.setup_device("auto", 0)
            
            logger.info(f"Loading models from: {model_path}")
            logger.info(f"Config: {config_path}")
            
            # Load models
            cls._model_dict, cls._cfg = load_model(model_path, config_path, cls._device)
            cls._model_path = model_path
            
            logger.info("Models loaded successfully!")
            return True, "Models loaded successfully!"
            
        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}"
            logger.error(error_msg)
            cls._model_dict = None
            cls._cfg = None
            cls._device = None
            cls._model_path = None
            return False, error_msg

    @classmethod
    def unload_models(cls):
        """Unload models to free memory - called automatically by ComfyUI"""
        if cls._model_dict is not None:
            logger.info("Unloading HunyuanVideo-Foley models to free memory")
            cls._model_dict = None
            cls._cfg = None
            cls._device = None
            cls._model_path = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Models unloaded and memory freed")

    def cleanup(self):
        """Cleanup method called by ComfyUI when node is no longer needed"""
        # This method is called automatically by ComfyUI's resource management
        # We don't need to do anything here as models are managed at class level
        pass

    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def custom_feature_process(self, video_path, prompt, model_dict, cfg, user_negative_prompt=""):
        """
        Custom feature processing function that combines built-in negative prompt with user input
        """
        try:
            # Use the original feature_process function
            visual_feats, text_feats, audio_len_in_s = original_feature_process(video_path, prompt, model_dict, cfg)
            
            # If user provided additional negative prompt, we need to re-process the text features
            if user_negative_prompt and user_negative_prompt.strip():
                # Import the text encoding function
                from hunyuanvideo_foley.utils.feature_utils import encode_text_feat
                
                # Combine built-in negative prompt with user input
                built_in_neg = "noisy, harsh"
                combined_neg_prompt = f"{built_in_neg}, {user_negative_prompt.strip()}"
                
                # Re-encode the combined prompts
                prompts = [combined_neg_prompt, prompt]
                text_feat_res, text_feat_mask = encode_text_feat(prompts, model_dict)
                
                # Update text features
                text_feat = text_feat_res[1:]
                uncond_text_feat = text_feat_res[:1]
                
                # Apply text length limit if needed
                if cfg.model_config.model_kwargs.text_length < text_feat.shape[1]:
                    text_seq_length = cfg.model_config.model_kwargs.text_length
                    text_feat = text_feat[:, :text_seq_length]
                    uncond_text_feat = uncond_text_feat[:, :text_seq_length]
                
                # Update text_feats with new features
                from hunyuanvideo_foley.utils.config_utils import AttributeDict
                text_feats = AttributeDict({
                    'text_feat': text_feat,
                    'uncond_text_feat': uncond_text_feat,
                })
                
                logger.info(f"Applied combined negative prompt: {combined_neg_prompt}")
            
            return visual_feats, text_feats, audio_len_in_s
            
        except Exception as e:
            logger.error(f"Custom feature processing failed: {e}")
            # Fallback to original function
            return original_feature_process(video_path, prompt, model_dict, cfg)

    @torch.inference_mode()
    def generate_audio(self, images: Any, text_prompt: str, guidance_scale: float, 
                        num_inference_steps: int, sample_nums: int, seed: int,
                        negative_prompt: str = "",
                        fps: float = 24.0,
                        model_path: str = "", 
                        config_path: str = "",
                        auto_download: bool = True,
                        model_variant: str = "hunyuanvideo-foley-xxl",
                        output_folder: str = "hunyuan_foley",
                        filename_prefix: str = "foley_"):
        """
        Generate audio for the input image frames with the given text prompt
        """
        try:
            # Initialize progress bar for real-time progress display
            if HAS_PROGRESS_BAR:
                # Total steps: model loading(1) + frame extraction(1) + feature processing(1) + audio generation(1) + output saving(1)
                total_steps = 5
                progress_bar = ProgressBar(total_steps)
                current_step = 0
                
                def update_progress(step_name: str, step_weight: int = 1):
                    nonlocal current_step
                    current_step += step_weight
                    progress_bar.update_absolute(current_step, total_steps)
                    logger.info(f"Progress: {step_name} ({current_step}/{total_steps})")
            else:
                # Fallback progress function when ProgressBar is not available
                def update_progress(step_name: str, step_weight: int = 1):
                    logger.info(f"Progress: {step_name}")
            
            # Set seed for reproducibility
            self.set_seed(seed)
            
            # Load models if needed
            update_progress("Loading models...", 1)
            success, message = self.load_models(model_path, config_path, auto_download, model_variant)
            if not success:
                # Return empty values that won't cause downstream errors
                logger.error(f"Model loading failed: {message}")
                empty_audio = {"waveform": torch.zeros((1, 48000)), "sample_rate": 48000}
                empty_frames = torch.zeros((1, 3, 256, 256))  # Empty frame sequence
                return (empty_frames, empty_audio, f"❌ {message}")
            
            # Validate inputs
            if images is None:
                empty_audio = {"waveform": torch.zeros((1, 48000)), "sample_rate": 48000}
                empty_frames = torch.zeros((1, 3, 256, 256))  # Empty frame sequence
                return (empty_frames, empty_audio, "❌ Please provide image frames input!")
            
            # Clean text prompt
            if text_prompt is None:
                text_prompt = ""
            text_prompt = text_prompt.strip()
            
            # Clean negative prompt
            if negative_prompt is None:
                negative_prompt = ""
            negative_prompt = negative_prompt.strip()
            
            logger.info(f"Processing image frames with prompt: {text_prompt}")
            if negative_prompt:
                logger.info(f"Using negative prompt: {negative_prompt}")
            logger.info(f"Generating {sample_nums} sample(s)")
            
            # Debug log the images input type
            logger.info(f"Images input type: {type(images)}")
            if hasattr(images, '__dict__'):
                logger.debug(f"Images attributes: {images.__dict__}")
            
            # Extract frames from IMAGE input
            update_progress("Extracting image frames...", 1)
            frames_extracted, frames, video_file, extract_msg = self._extract_frames_from_image_input(images)
            
            if not frames_extracted:
                # Try to handle as video file path (fallback compatibility)
                if video_file and os.path.exists(video_file):
                    logger.info(f"Using video file: {video_file}")
                    temp_video_file = video_file
                else:
                    logger.error(f"Could not extract frames from input type: {type(images)}")
                    logger.debug(f"Images input content: {images}")
                    empty_audio = {"waveform": torch.zeros((1, 48000)), "sample_rate": 48000}
                    empty_frames = torch.zeros((1, 3, 256, 256))
                    return (empty_frames, empty_audio, "❌ Could not process images input format")
            else:
                # Convert frames to temporary video file
                logger.info(f"Extracted {len(frames)} frames from IMAGE input")
                ok, temp_video_file, msg = self._write_temp_video(frames, fps)
                if not ok:
                    logger.error(f"Failed to create temporary video: {msg}")
                    empty_audio = {"waveform": torch.zeros((1, 48000)), "sample_rate": 48000}
                    empty_frames = torch.zeros((1, 3, 256, 256))
                    return (empty_frames, empty_audio, f"❌ {msg}")
                
                if not os.path.exists(temp_video_file):
                    empty_audio = {"waveform": torch.zeros((1, 48000)), "sample_rate": 48000}
                    empty_frames = torch.zeros((1, 3, 256, 256))
                    return (empty_frames, empty_audio, f"❌ Video file not found: {temp_video_file}")
            
            # Feature processing
            update_progress("Processing video features...", 1)
            logger.info("Processing video features...")
            visual_feats, text_feats, audio_len_in_s = self.custom_feature_process(
                temp_video_file,
                text_prompt,
                self._model_dict,
                self._cfg,
                user_negative_prompt=negative_prompt
            )
            
            # Generate audio
            update_progress("Generating audio...", 1)
            logger.info("Generating audio...")
            audio, sample_rate = denoise_process(
                visual_feats,
                text_feats,
                audio_len_in_s,
                self._model_dict,
                self._cfg,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                batch_size=sample_nums
            )
            
            # Create output directory structure
            update_progress("Saving output files...", 1)
            output_dir = folder_paths.get_output_directory()
            
            # Create subfolder if specified
            if output_folder and output_folder.strip():
                output_dir = os.path.join(output_dir, output_folder.strip())
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save audio file
            audio_filename = f"{filename_prefix}audio_{timestamp}_{seed}.wav"
            audio_output = os.path.join(output_dir, audio_filename)
            torchaudio.save(audio_output, audio[0], sample_rate)
            
            # Save video with audio
            video_filename = f"{filename_prefix}video_{timestamp}_{seed}.mp4"
            video_output = os.path.join(output_dir, video_filename)
            
            # Merge audio and video
            merge_audio_video(audio_output, temp_video_file, video_output)
            
            # Create audio result dict
            audio_result = {"waveform": audio[0].unsqueeze(0), "sample_rate": sample_rate}
            
            # Return the original frames as IMAGE output (ComfyUI standard)
            # This allows downstream nodes to use the frames with the generated audio
            if frames_extracted and frames:
                # Convert list of frames back to tensor format (B, C, H, W)
                output_frames = torch.stack(frames, dim=0)
            else:
                # If we used a video file, create placeholder frames
                # You could also load the video file and extract frames here
                output_frames = torch.zeros((1, 3, 256, 256))
            
            success_msg = f"✅ Generated audio and saved video to: {video_output}"
            logger.info(success_msg)
            
            # Return frames as IMAGE type for ComfyUI compatibility
            return (output_frames, audio_result, success_msg)
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            logger.error(error_msg)
            # Return a valid empty audio tensor and frames to prevent downstream errors
            empty_audio = {"waveform": torch.zeros((1, 48000)), "sample_rate": 48000}
            empty_frames = torch.zeros((1, 3, 256, 256))
            return (empty_frames, empty_audio, error_msg)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "HunyuanVideoFoley": YCHunyuanVideoFoley,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVideoFoley": "YC HunyuanVideo-Foley",
}
