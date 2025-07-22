import streamlit as st
import os
from src.deepfake_detector.inference import DeepfakeInferencePipeline

st.title("Deepfake Detector Demo (Image, Video, Audio)")

@st.cache_resource
def get_pipeline():
    return DeepfakeInferencePipeline(models_dir="models")

uploaded_file = st.file_uploader(
    "Upload an image, video, or audio file",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv", "webm", "wav", "mp3", "flac", "m4a", "ogg"]
)

if uploaded_file is not None:
    # Save to temp file
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Detect modality
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
        model_type = "image"
    elif ext in ["mp4", "avi", "mov", "mkv", "webm"]:
        model_type = "video"
    elif ext in ["wav", "mp3", "flac", "m4a", "ogg"]:
        model_type = "audio"
    else:
        st.error("Unsupported file type")
        st.stop()
    # Run inference
    with st.spinner(f"Running {model_type} inference..."):
        try:
            pipeline = get_pipeline()
            result = pipeline.infer_single(temp_path, model_type=model_type)
            st.success(f"Inference complete! Detected as: {result.get('prediction', 'unknown')}")
            st.json(result)
        except Exception as e:
            st.error(f"Error during inference: {e}")
    # Clean up
    os.remove(temp_path) 