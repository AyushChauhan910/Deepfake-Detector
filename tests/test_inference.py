# test_inference.py
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch
import numpy as np
from PIL import Image, ImageDraw
import librosa
import pandas as pd

# Import your package's inference module
from deepfake_detector.inference import (
    infer_image,
    infer_video,
    infer_audio,
    load_model,
    DeepfakeInferencePipeline,
    ImageModel,
    VideoModel,
    AudioModel,
    validate_input_file,
)
from deepfake_detector.preprocess import (
    extract_frames,
    audio_to_mel,
)

# Sample constants (match your package config)
IMG_SIZE = (224, 224)
AUDIO_SR = 16000
AUDIO_DUR = 2.5
AUDIO_N_MELS = 128
BATCH_SIZE = 24

class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create minimal test files for all modalities."""
        cls.tempdir = tempfile.TemporaryDirectory()
        
        # --- Image ---
        cls.test_image = Path(cls.tempdir.name) / "test.jpg"
        img = Image.new("RGB", IMG_SIZE, (100, 200, 50))
        draw = ImageDraw.Draw(img)
        draw.ellipse((50, 50, 150, 150), fill="blue")
        img.save(cls.test_image)
        
        # --- Video --- (mock 3 RGB frames)
        cls.test_video = Path(cls.tempdir.name) / "test.mp4"
        mock_frames = [
            np.random.randint(0, 255, (IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8),
            np.random.randint(0, 255, (IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8),
            np.random.randint(0, 255, (IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8),
        ]
        cls.dummy_video_frames = mock_frames
        
        # --- Audio --- (mock a mel-spectrogram input)
        cls.test_audio = Path(cls.tempdir.name) / "test.wav"
        y = np.random.randn(int(AUDIO_SR * AUDIO_DUR))
        librosa.output.write_wav(cls.test_audio, y, AUDIO_SR)
        cls.dummy_mel = np.zeros((AUDIO_N_MELS, int(AUDIO_DUR * AUDIO_SR / 512) + 1))
        
        # --- Models --- (mocked for this test)
        cls.image_model = ImageModel()
        cls.video_model = VideoModel()
        cls.audio_model = AudioModel()
        cls.device = "cpu"
        os.makedirs(Path(cls.tempdir.name) / "models", exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def setUp(self):
        self.models_dir = Path(self.tempdir.name) / "models"
        self.pipeline = DeepfakeInferencePipeline(self.models_dir)

    # --- Utilities ---
    def test_validate_input_file(self):
        # Valid file
        self.assertTrue(validate_input_file(str(self.test_image), [".jpg"]))
        # Invalid path
        with self.assertRaises(FileNotFoundError):
            validate_input_file("nonexistent.jpg", [".jpg"])
        # Invalid extension
        with self.assertRaises(ValueError):
            validate_input_file("file.pdf", [".jpg"])
    
    # --- Preprocessing helpers ---
    def test_extract_frames(self):
        # Mock cv2.VideoCapture since we can't create a real MP4 here
        with patch("cv2.VideoCapture") as mock_vcap:
            mock_vcap.return_value.isOpened.return_value = True
            mock_vcap.return_value.get.return_value = len(self.dummy_video_frames)
            mock_vcap.return_value.read.side_effect = [
                (True, frame) for frame in self.dummy_video_frames
            ]
            frames = extract_frames(self.test_video, num_frames=3)
            self.assertEqual(len(frames), 3)
            self.assertTrue(isinstance(frames[0], Image.Image))
        
        # Error case
        with patch("cv2.VideoCapture") as mock_vcap:
            mock_vcap.return_value.isOpened.return_value = False
            with self.assertRaises(ValueError):
                extract_frames(self.test_video)
    
    def test_audio_to_mel(self):
        # Mock librosa.load
        with patch("librosa.load", return_value=(np.random.randn(10), AUDIO_SR)):
            # Test successful conversion
            mel = audio_to_mel(self.test_audio, AUDIO_SR, AUDIO_N_MELS, AUDIO_DUR)
            self.assertEqual(mel.shape, (AUDIO_N_MELS, int(AUDIO_DUR * AUDIO_SR / 512) + 1))
        
        # Test error fallback
        with patch("librosa.load", side_effect=Exception("Test error")):
            mel = audio_to_mel(self.test_audio, AUDIO_SR, AUDIO_N_MELS, AUDIO_DUR)
            self.assertEqual(
                mel.shape,
                (AUDIO_N_MELS, int(AUDIO_DUR * AUDIO_SR / 512) + 1),
            )
            self.assertTrue(np.all(mel == 0))
    
    # --- Inference with mocked models ---
    def test_infer_image(self):
        with patch("deepfake_detector.models.ImageModel.forward", return_value=torch.tensor([[0.9, 0.1]])):
            result = infer_image(self.image_model, self.test_image, device="cpu")
            self.assertEqual(result["file_path"], str(self.test_image))
            self.assertIn("real", result["probabilities"])
            self.assertIn("fake", result["probabilities"])
            self.assertIn("confidence", result)
            self.assertIn("prediction", result)
    
    def test_infer_video(self):
        with patch("deepfake_detector.inference.extract_frames", return_value=[
            Image.fromarray(self.dummy_video_frames[0]),
            Image.fromarray(self.dummy_video_frames[1]),
            Image.fromarray(self.dummy_video_frames[2]),
        ]), patch("deepfake_detector.models.VideoModel.forward", return_value=torch.tensor([[0.1, 0.9]])):
            result = infer_video(self.video_model, self.test_video, device="cpu")
            self.assertEqual(result["file_path"], str(self.test_video))
            self.assertEqual(result["frames_processed"], 3)
            self.assertIn("real", result["probabilities"])
            self.assertIn("fake", result["probabilities"])
            self.assertIn("confidence", result)
            self.assertIn("prediction", result)
    
    def test_infer_audio(self):
        with patch("deepfake_detector.preprocess.audio_to_mel", return_value=self.dummy_mel
        ), patch("deepfake_detector.models.AudioModel.forward", return_value=torch.tensor([[0.1, 0.9]])):
            result = infer_audio(self.audio_model, self.test_audio, device="cpu")
            self.assertEqual(result["file_path"], str(self.test_audio))
            self.assertIn("real", result["probabilities"])
            self.assertIn("fake", result["probabilities"])
            self.assertIn("confidence", result)
            self.assertIn("prediction", result)
    
    # --- Pipeline ---
    def test_pipeline_load_model(self):
        # Mock torch.load to avoid loading real models
        with patch("torch.load", return_value={"foo": "bar"}):
            model = self.pipeline.get_model("image")
            self.assertIsInstance(model, ImageModel)
        
        # Test error on unknown type
        with self.assertRaises(ValueError):
            self.pipeline.get_model("unknown_type")
    
    def test_pipeline_infer_single(self):
        # Mock the entire inference workflow
        def mock_get_model(model_type):
            if model_type == "image":
                return self.image_model
            elif model_type == "video":
                return self.video_model
            elif model_type == "audio":
                return self.audio_model
            else:
                raise ValueError(f"Unknown type: {model_type}")
        
        self.pipeline.get_model = mock_get_model
        with patch("deepfake_detector.inference.ImageModel.forward", return_value=torch.tensor([[0.9, 0.1]])):
            # Image inference
            result = self.pipeline.infer_single(self.test_image)
            self.assertEqual(result["modality"], "image")
            self.assertEqual(result["file_path"], str(self.test_image))
        
            # Video inference
            with patch("deepfake_detector.inference.extract_frames", return_value=[Image.fromarray(self.dummy_video_frames[0])]):
                with patch("deepfake_detector.inference.VideoModel.forward", return_value=torch.tensor([[0.1, 0.9]])):
                    result = self.pipeline.infer_single(self.test_video)
                    self.assertEqual(result["modality"], "video")
        
            # Audio inference
            with patch("deepfake_detector.inference.audio_to_mel", return_value=self.dummy_mel):
                with patch("deepfake_detector.inference.AudioModel.forward", return_value=torch.tensor([[0.1, 0.9]])):
                    result = self.pipeline.infer_single(self.test_audio)
                    self.assertEqual(result["modality"], "audio")
    
    def test_pipeline_detect_modality(self):
        # Known extensions
        self.assertEqual(self.pipeline._detect_modality("file.jpg"), "image")
        self.assertEqual(self.pipeline._detect_modality("file.png"), "image")
        self.assertEqual(self.pipeline._detect_modality("file.mp4"), "video")
        self.assertEqual(self.pipeline._detect_modality("file.wav"), "audio")
        # Unknown extension
        with self.assertRaises(ValueError):
            self.pipeline._detect_modality("file.xyz")
    
    # --- Load model (error handling, checkpoint formats) ---
    def test_load_model(self):
        # Mock different checkpoint formats
        mock_state_dict = {"conv1.weight": torch.randn(3,3,3,3)}
        with patch("torch.load", return_value={"model_state_dict": mock_state_dict}):
            model = load_model("dummy_path", "image", device="cpu")
            self.assertIsInstance(model, ImageModel)
        
        with patch("torch.load", return_value=mock_state_dict):
            model = load_model("dummy_path", "image", device="cpu")
            self.assertIsInstance(model, ImageModel)
        
        # Test strict=False (ignore minor key mismatches)
        bad_state_dict = {"misspeled.weight": torch.randn(3,3,3,3)}
        with patch("torch.load", return_value=bad_state_dict):
            model = load_model("dummy_path", "image", device="cpu")
            self.assertIsInstance(model, ImageModel)
    
    # --- Custom CLI args (optional) ---
    def test_cli_help(self):
        with patch("sys.argv", ["test_inference.py", "--help"]), patch("sys.exit") as mock_exit:
            import deepfake_detector.inference
            mock_exit.assert_called_with(0)

if __name__ == "__main__":
    unittest.main()
