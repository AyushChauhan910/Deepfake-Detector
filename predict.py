import os
from typing import BinaryIO
from src.deepfake_detector.inference import DeepfakeInferencePipeline

class Predictor:
    def setup(self):
        # Load the pipeline with the models directory
        self.pipeline = DeepfakeInferencePipeline(models_dir="models")

    def predict(self, file: BinaryIO, model_type: str = "auto") -> dict:
        # Save the uploaded file to a temporary location
        input_path = "/tmp/input"
        with open(input_path, "wb") as f:
            f.write(file.read())
        # Run inference
        result = self.pipeline.infer_single(input_path, model_type=model_type)
        return result 