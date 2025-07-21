"""
Inference pipeline for deepfake detection.
"""
import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import logging

import torch
import torchvision.transforms as T
from PIL import Image

from .models import ImageModel, VideoModel, AudioModel
from .preprocess import extract_frames, audio_to_mel, validate_input_file
from .utils import setup_logging, get_device

__all__ = ["infer_image", "infer_video", "infer_audio", "DeepfakeInferencePipeline", "main"]

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "device": get_device(),
    "img_size": (224, 224),
    "audio_sr": 16000,
    "audio_n_mels": 128,
    "audio_duration": 2.5,
    "video_max_frames": 8
}

# Image transforms
IMAGE_TRANSFORM = T.Compose([
    T.Resize(DEFAULT_CONFIG["img_size"]),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path, model_type, device=None):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        model_type (str): Type of model ('image', 'video', 'audio')
        device (str, optional): Device to load model on
    
    Returns:
        torch.nn.Module: Loaded model in eval mode
    """
    if device is None:
        device = DEFAULT_CONFIG["device"]
    
    # Initialize model
    if model_type == "image":
        model = ImageModel()
    elif model_type == "video":
        model = VideoModel()
    elif model_type == "audio":
        model = AudioModel()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load with non-strict mode to handle minor key mismatches
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    return model

def infer_image(model, image_path, device=None):
    """
    Run inference on image file.
    
    Args:
        model: Trained image model
        image_path (str): Path to image file
        device (str, optional): Device for inference
    
    Returns:
        dict: Inference results
    """
    if device is None:
        device = DEFAULT_CONFIG["device"]
    
    try:
        # Validate input
        validate_input_file(image_path, ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
        
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        x = IMAGE_TRANSFORM(img).unsqueeze(0).to(device)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, 1).cpu().numpy()[0]
        processing_time = time.time() - start_time
        
        return {
            'file_path': image_path,
            'modality': 'image',
            'probabilities': {
                'real': float(probs[0]),
                'fake': float(probs[1])
            },
            'prediction': 'fake' if probs[1] > probs[0] else 'real',
            'confidence': float(max(probs)),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Image inference failed: {str(e)}")
        return {
            'file_path': image_path,
            'modality': 'image',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def infer_video(model, video_path, device=None):
    """
    Run inference on video file.
    
    Args:
        model: Trained video model
        video_path (str): Path to video file
        device (str, optional): Device for inference
    
    Returns:
        dict: Inference results
    """
    if device is None:
        device = DEFAULT_CONFIG["device"]
    
    try:
        # Validate input
        validate_input_file(video_path, ['.mp4', '.avi', '.mov', '.mkv', '.webm'])
        
        # Extract frames
        frames = extract_frames(video_path, DEFAULT_CONFIG["video_max_frames"])
        
        # Preprocess frames
        batch = torch.stack([IMAGE_TRANSFORM(f) for f in frames]).unsqueeze(0).to(device)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, 1).cpu().numpy()[0]
        processing_time = time.time() - start_time
        
        return {
            'file_path': video_path,
            'modality': 'video',
            'frames_processed': len(frames),
            'probabilities': {
                'real': float(probs[0]),
                'fake': float(probs[1])
            },
            'prediction': 'fake' if probs[1] > probs[0] else 'real',
            'confidence': float(max(probs)),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Video inference failed: {str(e)}")
        return {
            'file_path': video_path,
            'modality': 'video',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def infer_audio(model, audio_path, device=None):
    """
    Run inference on audio file.
    
    Args:
        model: Trained audio model
        audio_path (str): Path to audio file
        device (str, optional): Device for inference
    
    Returns:
        dict: Inference results
    """
    if device is None:
        device = DEFAULT_CONFIG["device"]
    
    try:
        # Validate input
        validate_input_file(audio_path, ['.wav', '.mp3', '.flac', '.m4a', '.ogg'])
        
        # Convert to mel-spectrogram
        mel = audio_to_mel(audio_path, 
                          DEFAULT_CONFIG["audio_sr"], 
                          DEFAULT_CONFIG["audio_n_mels"], 
                          DEFAULT_CONFIG["audio_duration"])
        x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, 1).cpu().numpy()[0]
        processing_time = time.time() - start_time
        
        return {
            'file_path': audio_path,
            'modality': 'audio',
            'probabilities': {
                'real': float(probs[0]),
                'fake': float(probs[1])
            },
            'prediction': 'fake' if probs[1] > probs[0] else 'real',
            'confidence': float(max(probs)),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Audio inference failed: {str(e)}")
        return {
            'file_path': audio_path,
            'modality': 'audio',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

class DeepfakeInferencePipeline:
    """Unified inference pipeline for deepfake detection."""
    
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.device = DEFAULT_CONFIG["device"]
        self.loaded_models = {}
        self.available_models = {}
        
        # Discover available models
        self._discover_models()
    
    def _discover_models(self):
        """Discover available model files."""
        if not os.path.exists(self.models_dir):
            logger.error(f"Models directory not found: {self.models_dir}")
            return
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pth')]
        
        for model_file in model_files:
            # Determine model type from filename
            if 'image' in model_file.lower():
                model_type = 'image'
            elif 'video' in model_file.lower():
                model_type = 'video'
            elif 'audio' in model_file.lower():
                model_type = 'audio'
            else:
                continue
            
            self.available_models[model_type] = os.path.join(self.models_dir, model_file)
        
        logger.info(f"Available models: {list(self.available_models.keys())}")
    
    def get_model(self, model_type):
        """Load and cache model."""
        if model_type not in self.available_models:
            raise ValueError(f"No {model_type} model available")
        
        if model_type not in self.loaded_models:
            model_path = self.available_models[model_type]
            self.loaded_models[model_type] = load_model(model_path, model_type, self.device)
            logger.info(f"Loaded {model_type} model")
        
        return self.loaded_models[model_type]
    
    def infer_single(self, file_path, model_type=None):
        """Run inference on single file."""
        # Auto-detect model type if not specified
        if model_type is None:
            model_type = self._detect_modality(file_path)
        
        # Get model
        model = self.get_model(model_type)
        
        # Run inference
        if model_type == 'image':
            return infer_image(model, file_path, self.device)
        elif model_type == 'video':
            return infer_video(model, file_path, self.device)
        elif model_type == 'audio':
            return infer_audio(model, file_path, self.device)
        else:
            raise ValueError(f"Unsupported modality: {model_type}")
    
    def _detect_modality(self, file_path):
        """Auto-detect file modality."""
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return 'image'
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return 'video'
        elif ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            return 'audio'
        else:
            raise ValueError(f"Cannot detect modality for: {ext}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Deepfake Detection Inference Pipeline"
    )
    parser.add_argument("modality", 
                       choices=["image", "video", "audio", "auto"],
                       help="Input modality")
    parser.add_argument("input_path", help="Path to input file")
    parser.add_argument("--models-dir", required=True,
                       help="Directory containing trained models")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Initialize pipeline
        pipeline = DeepfakeInferencePipeline(args.models_dir)
        
        # Run inference
        result = pipeline.infer_single(
            args.input_path,
            None if args.modality == "auto" else args.modality
        )
        
        # Output result
        print(json.dumps(result, indent=2))
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {args.output}")
            
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
