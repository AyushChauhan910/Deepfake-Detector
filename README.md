<div align="center">

# 🕵️ Deepfake Detector

### Multimodal AI-Powered Deepfake Detection for Images, Videos & Audio

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Detect AI-generated faces, synthetic voices, and manipulated videos — all in one toolkit*

<br/>

**[🤗 Live Demo on Hugging Face](https://huggingface.co/spaces/ayush0910/Deepfake_Detector)** &nbsp;·&nbsp;
**[🌐 React Frontend on Vercel](https://deepfakedetector-rouge.vercel.app/)**

</div>

---

## 🌐 Overview

Deepfakes pose a growing threat to digital trust — from fabricated political speeches to synthetic media fraud. This project delivers a **production-ready deepfake detection toolkit** capable of analyzing images, video frames, and audio clips using deep learning models trained on real-world datasets.

With a clean **Streamlit interface**, a **REST API backend**, and **one-command Docker deployment**, it's designed for both researchers and developers who need reliable deepfake detection at their fingertips.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ **Image Detection** | EfficientNet-based classifier detects facial manipulations in still images |
| 🎬 **Video Detection** | CNN+LSTM pipeline analyzes temporal frame sequences for video deepfakes |
| 🎙️ **Audio Detection** | Custom CNN on mel-spectrograms identifies AI-synthesized speech and voice cloning |
| 🔀 **Auto Modality Routing** | Automatically detects file type and routes to the correct model |
| 📊 **Confidence Scores** | Returns prediction label + probability breakdown as structured JSON |
| 🌐 **REST API** | FastAPI `/api/infer` endpoint for programmatic access |
| 🐳 **Docker Ready** | Multi-stage build, deployable to Hugging Face Spaces in minutes |
| ⚡ **Streamlit UI** | Drag-and-drop web interface — no coding needed to try it out |

---

## 🎯 Supported File Types

| Modality | Formats |
|---|---|
| 🖼️ Image | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` |
| 🎬 Video | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` |
| 🎙️ Audio | `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg` |

---

## 🗂️ Project Structure

```
Deepfake-Detector/
│
├── 🐍 Core Application
│   ├── app.py                                        ← Streamlit UI entry point
│   ├── predict.py                                    ← CLI prediction script
│   └── inference-pipelines-for-deepfake-detection.ipynb
│
├── 📦 src/
│   └── deepfake_detector/
│       └── inference.py                              ← DeepfakeInferencePipeline class
│
├── 🧠 models/                                        ← Pre-trained model weights (.pth)
│   ├── image_model.pth
│   ├── video_model.pth
│   └── audio_model.pth
│
├── 📓 Notebooks
│   ├── deepfake-detection-raw-data-processing.ipynb  ← Data prep pipeline
│   └── deepfake-detection-model-training-3 (2).ipynb ← Model training
│
├── 🧪 tests/                                         ← Unit & integration tests
├── 📦 dist/                                          ← Build artifacts
│
├── Dockerfile                                        ← Multi-stage Docker build
├── requirements.txt
├── pyproject.toml
└── setup.cfg
```

---

## 🧠 Model Architecture

```
Input File
    │
    ▼
Modality Detection (by file extension)
    │
    ├─── 🖼️  Image ──► EfficientNet-B4
    │                    └─► Binary classifier (Real / Fake)
    │
    ├─── 🎬  Video ──► Frame Sampler → CNN feature extractor
    │                    └─► LSTM temporal aggregator → prediction
    │
    └─── 🎙️  Audio ──► Mel-spectrogram → Custom CNN
                         └─► Softmax output (Real / Fake)
```

All three pipelines return a unified JSON response:

```json
{
  "prediction": "FAKE",
  "confidence": 0.94,
  "probabilities": {
    "real": 0.06,
    "fake": 0.94
  },
  "modality": "image"
}
```

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/AyushChauhan910/Deepfake-Detector.git
cd Deepfake-Detector

docker build -t deepfake-detector .
docker run -p 7860:7860 deepfake-detector
```

Visit **http://localhost:7860** — the Streamlit UI will be live.

---

### Option 2 — Local Python

**1. Clone & install dependencies**

```bash
git clone https://github.com/AyushChauhan910/Deepfake-Detector.git
cd Deepfake-Detector
pip install -r requirements.txt
```

**2. Launch the Streamlit app**

```bash
streamlit run app.py
```

**3. Or run a quick CLI prediction**

```bash
python predict.py --file path/to/your/image.jpg
```

---

## 🌐 REST API

The FastAPI backend exposes a single inference endpoint.

### `POST /api/infer`

**Request:** `multipart/form-data` with a field named `file`

**Example with `curl`:**

```bash
curl -X POST \
  -F "file=@sample_video.mp4" \
  https://huggingface.co/spaces/ayush0910/Deepfake_Detector/api/infer
```

**Response:**

```json
{
  "prediction": "FAKE",
  "confidence": 0.91,
  "probabilities": { "real": 0.09, "fake": 0.91 },
  "modality": "video"
}
```

---

## 📓 Training & Notebooks

The full ML pipeline is documented across three Kaggle notebooks:

| Notebook | Description | Link |
|---|---|---|
| 🔧 Raw Data Processing | Dataset curation, face extraction, augmentation | [Kaggle →](https://www.kaggle.com/code/ayushchauhan0910/deepfake-detection-raw-data-processing) |
| 🏋️ Model Training | EfficientNet, CNN+LSTM, audio CNN training | [Kaggle →](https://www.kaggle.com/code/ayushchauhan0910/deepfake-detection-model-training-3) |
| 🔍 Inference Pipelines | End-to-end inference across all three modalities | [Kaggle →](https://www.kaggle.com/code/ayushchauhan0910/inference-pipelines-for-deepfake-detection) |

---

## ☁️ Deployment

### Hugging Face Spaces (Docker SDK)

1. Fork this repository
2. Create a new Space → select **Docker** as the SDK
3. Connect your GitHub repo
4. The Space will auto-build and deploy

🔗 [View the live Space](https://huggingface.co/spaces/ayush0910/Deepfake_Detector)

### Vercel (Frontend Only)

The React frontend is independently deployable to Vercel for a fast, CDN-hosted UI.

🔗 [View the Vercel frontend](https://deepfakedetector-rouge.vercel.app/)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch 2.0, TorchVision |
| Image Models | EfficientNet-B4 |
| Video Models | CNN + LSTM |
| Audio Processing | Librosa, Mel-spectrogram CNN |
| Computer Vision | OpenCV |
| Web App | Streamlit |
| API Backend | FastAPI + Uvicorn |
| Containerization | Docker |
| Hosting | Hugging Face Spaces, Vercel |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Hugging Face Spaces](https://huggingface.co/spaces) for free GPU-accelerated hosting
- [Kaggle](https://www.kaggle.com/) for compute and dataset access
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [Librosa](https://librosa.org/) for audio feature extraction

---

<div align="center">

*Built to make the internet a more trustworthy place — one file at a time.*

**[⭐ Star this repo](https://github.com/AyushChauhan910/Deepfake-Detector)** &nbsp;·&nbsp; **[🤗 Try the Demo](https://huggingface.co/spaces/ayush0910/Deepfake_Detector)**

</div>
