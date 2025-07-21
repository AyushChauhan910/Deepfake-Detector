"""
Deepfake Detector - A multimodal deepfake detection toolkit.
"""
__version__ = "0.1.0"

from .models import ImageModel, VideoModel, AudioModel
from .inference import infer_image, infer_video, infer_audio, DeepfakeInferencePipeline
from .preprocess import extract_frames, audio_to_mel
from .utils import setup_logging

__all__ = [
    "ImageModel", "VideoModel", "AudioModel",
    "infer_image", "infer_video", "infer_audio", 
    "DeepfakeInferencePipeline",
    "extract_frames", "audio_to_mel",
    "setup_logging"
]
