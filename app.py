import gradio as gr
from src.deepfake_detector.inference import DeepfakeInferencePipeline
import os

# Load your models (assume models are in /app/models or ./models)
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
pipeline = DeepfakeInferencePipeline(MODELS_DIR)

def infer(file, modality):
    # Save uploaded file
    file_path = file.name
    with open(file_path, "wb") as f:
        f.write(file.read())
    # Run inference
    result = pipeline.infer_single(file_path, model_type=modality)
    prediction = result.get("prediction", "error")
    confidence = result.get("confidence", 0.0)
    return prediction, confidence

iface = gr.Interface(
    fn=infer,
    inputs=[
        gr.File(label="Upload File"),
        gr.Radio(["image", "video", "audio"], label="Modality")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence")
    ],
    title="Deepfake Detector",
    description="Upload an image, video, or audio file and select the modality to detect deepfakes."
)

if __name__ == "__main__":
    iface.launch() 