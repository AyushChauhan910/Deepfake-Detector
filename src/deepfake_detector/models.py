"""
Model definitions for deepfake detection.
"""
import torch
import torch.nn as nn
from torchvision import models

__all__ = ["ImageModel", "VideoModel", "AudioModel", "safe_efficientnet"]

def safe_efficientnet(pretrained=False):
    """Load EfficientNet with fallback for offline environments."""
    try:
        return models.efficientnet_b0(pretrained=pretrained)
    except Exception:
        print("Warning: Failed to load pretrained weights, using random initialization")
        return models.efficientnet_b0(pretrained=False)

class ImageModel(nn.Module):
    """Image-based deepfake detection model using EfficientNet backbone."""
    
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = safe_efficientnet(False)
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        self.net = backbone
    
    def forward(self, x):
        return self.net(x)

class VideoModel(nn.Module):
    """Video-based deepfake detection model using CNN+LSTM architecture."""
    
    def __init__(self, num_classes=2):
        super().__init__()
        # CNN backbone for frame feature extraction
        backbone = safe_efficientnet(False)
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, c, h, w = x.shape
        
        # Extract features from each frame
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, num_frames, -1)
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(features)
        
        # Use last hidden state for classification
        return self.classifier(hidden[-1])

class AudioModel(nn.Module):
    """Audio-based deepfake detection model for mel-spectrograms."""
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(128)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        return self.classifier(x)
