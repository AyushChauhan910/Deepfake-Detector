# Deepfake Detector

A multimodal deepfake detection toolkit supporting image, video, and audio analysis.

## Features

- **Multimodal Detection**: Support for images, videos, and audio files
- **Pre-trained Models**: EfficientNet-based architectures with CNN+LSTM for videos
- **Easy Installation**: Standard Python package with pip install
- **CLI Interface**: Command-line tool for batch processing
- **Comprehensive Logging**: Detailed logging and performance metrics

## Installation

pip install deepfake-detector


Or install from source:

git clone <repository-url>
cd deepfake_detector_project
pip install -e 


## Usage

### Command Line

Single file inference
deepfake-infer image /path/to/image.jpg --models-dir /path/to/models

Auto-detect modality
deepfake-infer auto /path/to/file.mp4 --models-dir /path/to/models

Save results
deepfake-infer video /path/to/video.mp4 --models-dir /path/to/models --output results.json


### Python API

from deepfake_detector import DeepfakeInferencePipeline

Initialize pipeline
pipeline = DeepfakeInferencePipeline("/path/to/models")

Run inference
result = pipeline.infer_single("/path/to/image.jpg")
print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")


### Expected Output Format

{
"file_path": "/path/to/file.jpg",
"modality": "image",
"probabilities": {
"real": 0.234,
"fake": 0.766
},
"prediction": "fake",
"confidence": 0.766,
"processing_time": 0.123,
"timestamp": "2025-07-20T10:30:00"
}


## Model Requirements

Place your trained model files (`.pth` format) in a directory with these naming conventions:

- Image models: Must contain "image" in filename
- Video models: Must contain "video" in filename  
- Audio models: Must contain "audio" in filename

## License

MIT License
