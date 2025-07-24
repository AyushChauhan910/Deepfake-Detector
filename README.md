# Deepfake Detector

A powerful, multimodal deepfake detection toolkit supporting **image, video, and audio analysis**. This project provides a robust backend (Flask API), a modern React frontend, and is deployable on Hugging Face Spaces and Vercel for easy public access and demonstration.

---

## 🚀 Live Demos

- **Hugging Face Spaces:**  
  [Deepfake Detector on Hugging Face Spaces](https://huggingface.co/spaces/ayush0910/Deepfake_Detector)
- **Vercel Frontend:**  
  [Deepfake Detector React Frontend](https://deepfakedetector-rouge.vercel.app/)

---

## 📚 Related Notebooks & Resources

- [Raw Data Processing (Kaggle)](https://www.kaggle.com/code/ayushchauhan0910/deepfake-detection-raw-data-processing)
- [Model Training (Kaggle)](https://www.kaggle.com/code/ayushchauhan0910/deepfake-detection-model-training-3)
- [Inference Pipelines (Kaggle)](https://www.kaggle.com/code/ayushchauhan0910/inference-pipelines-for-deepfake-detection)

---

## ✨ Features

- **Multimodal Detection:**  
  Detects deepfakes in images, videos, and audio files using state-of-the-art models.
- **Pre-trained Models:**  
  EfficientNet-based architectures for images, CNN+LSTM for videos, and custom CNNs for audio.
- **User-Friendly Web UI:**  
  Upload files and view results instantly via a modern React interface.
- **REST API:**  
  Flask backend exposes `/api/infer` for programmatic access.
- **Easy Deployment:**  
  Ready for Hugging Face Spaces (Docker SDK) and Vercel.

---

## 🖥️ Project Structure

```
deepfake-detector-space/
│
├── backend/
│   ├── app.py                # Flask API and static file server
│   ├── requirements.txt      # Python dependencies
│   ├── src/                  # Deepfake detector core code
│   └── models/               # Pre-trained model weights (.pth)
│
├── frontend/
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       └── App.js            # React UI
│
├── Dockerfile                # Multi-stage build for Spaces
└── README.md
```

---

## 🛠️ Installation & Local Development

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/deepfake-detector-space.git
cd deepfake-detector-space
```

### 2. **Build the React Frontend**
```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. **Run with Docker**
```bash
docker build -t deepfake-detector-space .
docker run -p 7860:7860 deepfake-detector-space
```
Visit [http://localhost:7860](http://localhost:7860) in your browser.

---

## 🌐 Deployment

### **Hugging Face Spaces (Docker SDK)**
- Push your code to a public GitHub repo.
- Create a new Space, select **Docker** as the SDK, and link your repo.
- [View the live Space](https://huggingface.co/spaces/ayush0910/Deepfake_Detector)

### **Vercel (Frontend Only)**
- Deploy the `frontend/` directory to Vercel for a fast, static UI.
- [View the Vercel frontend](https://deepfakedetector-rouge.vercel.app/)

---

## 🧑‍💻 API Usage

### **POST** `/api/infer`

- **Request:**  
  `multipart/form-data` with a file field named `file`
- **Response:**  
  JSON with prediction, confidence, and probabilities.

**Example using `curl`:**
```bash
curl -X POST -F "file=@yourfile.mp4" https://huggingface.co/spaces/ayush0910/Deepfake_Detector/api/infer
```

---

## 📊 Model Training & Inference

- **Raw Data Processing:**  
  [Kaggle Notebook](https://www.kaggle.com/code/ayushchauhan0910/deepfake-detection-raw-data-processing)
- **Model Training:**  
  [Kaggle Notebook](https://www.kaggle.com/code/ayushchauhan0910/deepfake-detection-model-training-3)
- **Inference Pipelines:**  
  [Kaggle Notebook](https://www.kaggle.com/code/ayushchauhan0910/inference-pipelines-for-deepfake-detection)

---

## 📝 Citation

If you use this project, please cite the relevant Kaggle notebooks and this repository.

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Vercel](https://vercel.com/)
- [Kaggle](https://www.kaggle.com/)

---

## 🔗 Useful Links

- **Live Space:** [https://huggingface.co/spaces/ayush0910/Deepfake_Detector](https://huggingface.co/spaces/ayush0910/Deepfake_Detector)
- **Vercel Frontend:** [https://deepfakedetector-rouge.vercel.app/](https://deepfakedetector-rouge.vercel.app/)
- **Kaggle Raw Data Processing:** [https://www.kaggle.com/code/ayushchauhan0910/deepfake-detection-raw-data-processing](https://www.kaggle.com/code/ayushchauhan0910/deepfake-detection-raw-data-processing)
- **Kaggle Model Training:** [https://www.kaggle.com/code/ayushchauhan0910/deepfake-detection-model-training-3](https://www.kaggle.com/code/ayushchauhan0910/deepfake-detection-model-training-3)
- **Kaggle Inference Pipelines:** [https://www.kaggle.com/code/ayushchauhan0910/inference-pipelines-for-deepfake-detection](https://www.kaggle.com/code/ayushchauhan0910/inference-pipelines-for-deepfake-detection)

---
