import unittest
import torch
from deepfake_detector.models import ImageModel, VideoModel, AudioModel

class TestModels(unittest.TestCase):
    
    def test_image_model_forward(self):
        model = ImageModel()
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (1, 2))
    
    def test_video_model_forward(self):
        model = VideoModel()
        x = torch.randn(1, 5, 3, 224, 224)  # batch=1, frames=5
        output = model(x)
        self.assertEqual(output.shape, (1, 2))
    
    def test_audio_model_forward(self):
        model = AudioModel()
        x = torch.randn(1, 1, 128, 94)  # typical mel-spectrogram shape
        output = model(x)
        self.assertEqual(output.shape, (1, 2))

if __name__ == '__main__':
    unittest.main()
