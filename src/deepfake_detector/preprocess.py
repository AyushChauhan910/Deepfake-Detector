"""
Preprocessing utilities for images, videos, and audio.
"""
import cv2
import numpy as np
import librosa
from PIL import Image
import logging
from pathlib import Path

__all__ = ["extract_frames", "audio_to_mel", "validate_input_file"]

logger = logging.getLogger(__name__)

def extract_frames(video_path, num_frames=8):
    """
    Extract frames from video file.
    
    Args:
        video_path (str): Path to video file
        num_frames (int): Number of frames to extract
    
    Returns:
        list: List of PIL Images
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened or contains no frames
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video contains no frames: {video_path}")
    
    # Calculate frame indices
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames could be extracted from: {video_path}")
    
    return frames

def audio_to_mel(audio_path, sr=16000, n_mels=128, duration=2.5):
    """
    Convert audio file to mel-spectrogram.
    
    Args:
        audio_path (str): Path to audio file
        sr (int): Target sample rate
        n_mels (int): Number of mel frequency bins
        duration (float): Duration in seconds
    
    Returns:
        np.ndarray: Log-scaled mel-spectrogram
    
    Raises:
        FileNotFoundError: If audio file doesn't exist
        Exception: If audio processing fails
    """
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Ensure consistent length
        target_length = int(sr * duration)
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        
        # Generate mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512,
            power=2.0
        )
        
        # Convert to log scale
        S_db = librosa.power_to_db(S, ref=np.max)
        
        return S_db
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed for {audio_path}: {str(e)}")
        raise

def validate_input_file(file_path, expected_extensions):
    """
    Validate input file exists and has expected extension.
    
    Args:
        file_path (str): Path to file
        expected_extensions (list): List of valid extensions (e.g., ['.jpg', '.png'])
    
    Returns:
        bool: True if valid
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file has unsupported extension
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() not in expected_extensions:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Expected: {expected_extensions}"
        )
    
    return True
