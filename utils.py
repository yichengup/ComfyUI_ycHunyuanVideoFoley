"""
Utility functions for ComfyUI HunyuanVideo-Foley custom node
"""

import os
import tempfile
import torch
import numpy as np
from typing import Union, Optional, Tuple
from loguru import logger

def tensor_to_video(video_tensor: torch.Tensor, output_path: str, fps: int = 30) -> str:
    """
    Convert a video tensor to a video file
    
    Args:
        video_tensor: Video tensor with shape (frames, channels, height, width)
        output_path: Output video file path
        fps: Frame rate for the output video
        
    Returns:
        Path to the saved video file
    """
    try:
        import cv2
        
        # Convert tensor to numpy and handle different formats
        if isinstance(video_tensor, torch.Tensor):
            video_np = video_tensor.detach().cpu().numpy()
        else:
            video_np = np.array(video_tensor)
        
        # Handle different tensor formats
        if video_np.ndim == 4:  # (frames, channels, height, width)
            if video_np.shape[1] == 3:  # RGB
                video_np = np.transpose(video_np, (0, 2, 3, 1))  # (frames, height, width, channels)
            elif video_np.shape[1] == 1:  # Grayscale
                video_np = np.transpose(video_np, (0, 2, 3, 1))
                video_np = np.repeat(video_np, 3, axis=3)  # Convert to RGB
        elif video_np.ndim == 5:  # (batch, frames, channels, height, width)
            video_np = video_np[0]  # Take first batch
            if video_np.shape[1] == 3:
                video_np = np.transpose(video_np, (0, 2, 3, 1))
        
        # Normalize values to 0-255 range
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)
        
        # Get video dimensions
        frames, height, width, channels = video_np.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for i in range(frames):
            frame = video_np[i]
            if channels == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            out.write(frame)
        
        out.release()
        logger.info(f"Video saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to convert tensor to video: {e}")
        raise


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
        
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {}


def ensure_video_file(video_input: Union[str, torch.Tensor, np.ndarray]) -> str:
    """
    Ensure the video input is converted to a file path
    
    Args:
        video_input: Video input (path, tensor, or array)
        
    Returns:
        Path to video file
    """
    if isinstance(video_input, str):
        # Already a file path
        if os.path.exists(video_input):
            return video_input
        else:
            raise FileNotFoundError(f"Video file not found: {video_input}")
    
    elif isinstance(video_input, (torch.Tensor, np.ndarray)):
        # Convert tensor/array to video file
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "input_video.mp4")
        return tensor_to_video(video_input, output_path)
    
    else:
        raise ValueError(f"Unsupported video input type: {type(video_input)}")


def validate_model_files(model_path: str) -> Tuple[bool, str]:
    """
    Validate that all required model files exist
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_files = [
        "hunyuanvideo_foley.pth",
        "vae_128d_48k.pth", 
        "synchformer_state_dict.pth"
    ]
    
    missing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(model_path, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        return False, f"Missing model files: {', '.join(missing_files)}"
    
    return True, "All required model files found"


def get_optimal_device() -> torch.device:
    """
    Get the optimal device for model execution
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        # Get the device with most free memory
        max_memory = 0
        best_device = 0
        
        for i in range(torch.cuda.device_count()):
            memory_free = torch.cuda.get_device_properties(i).total_memory
            if memory_free > max_memory:
                max_memory = memory_free
                best_device = i
        
        device = torch.device(f"cuda:{best_device}")
        logger.info(f"Using CUDA device: {device} with {max_memory / 1e9:.1f}GB memory")
        return device
        
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
        return device
        
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
        return device


def check_memory_requirements(device: torch.device, required_gb: float = 16.0) -> Tuple[bool, str]:
    """
    Check if the device has enough memory for model execution
    
    Args:
        device: PyTorch device
        required_gb: Required memory in GB
        
    Returns:
        Tuple of (has_enough_memory, message)
    """
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        total_memory = properties.total_memory / 1e9  # Convert to GB
        
        if total_memory < required_gb:
            return False, f"GPU has {total_memory:.1f}GB memory, but {required_gb}GB is recommended"
        else:
            return True, f"GPU has {total_memory:.1f}GB memory (sufficient)"
            
    elif device.type == "mps":
        # MPS doesn't have a direct way to check memory, assume it's sufficient
        return True, "Using MPS device (memory check not available)"
        
    else:
        # CPU - assume it has enough memory
        return True, "Using CPU (no memory limit)"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"