"""
Utility functions for logging, configuration, and file operations.
"""
import os
import logging
import sys
from datetime import datetime
from pathlib import Path

__all__ = ["setup_logging", "get_device", "create_output_dir"]

def setup_logging(log_level="INFO", log_file=None, log_dir="logs"):
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Custom log file path
        log_dir (str): Directory for log files
    
    Returns:
        str: Path to the log file
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Default log file with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"deepfake_detector_{timestamp}.log")
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

def get_device():
    """
    Get the best available device for PyTorch.
    
    Returns:
        str: Device name ('cuda' or 'cpu')
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"Using GPU: {gpu_name} ({gpu_memory} GB)")
    else:
        print("Using CPU (GPU not available)")
    
    return device

def create_output_dir(base_dir="outputs"):
    """
    Create timestamped output directory.
    
    Args:
        base_dir (str): Base directory name
    
    Returns:
        str: Created directory path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def load_config(config_path=None):
    """Load configuration from file or return defaults."""
    default_config = {
        "img_size": (224, 224),
        "batch_size": 32,
        "device": get_device(),
        "audio_sr": 16000,
        "audio_n_mels": 128,
        "audio_duration": 2.5,
        "video_max_frames": 8
    }
    
    # TODO: Add config file loading if needed
    return default_config
