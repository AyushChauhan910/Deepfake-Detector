import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil

from .inference import DeepfakeInferencePipeline

app = FastAPI(title="Deepfake Detector API")

UPLOAD_DIR = "/tmp/deepfake_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Get models directory from environment or default
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
pipeline = DeepfakeInferencePipeline(MODELS_DIR)

def save_upload_file(upload_file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_DIR, upload_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

@app.post("/infer/image")
def infer_image_endpoint(file: UploadFile = File(...)):
    file_path = save_upload_file(file)
    try:
        result = pipeline.infer_single(file_path, model_type="image")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(file_path)

@app.post("/infer/video")
def infer_video_endpoint(file: UploadFile = File(...)):
    file_path = save_upload_file(file)
    try:
        result = pipeline.infer_single(file_path, model_type="video")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(file_path)

@app.post("/infer/audio")
def infer_audio_endpoint(file: UploadFile = File(...)):
    file_path = save_upload_file(file)
    try:
        result = pipeline.infer_single(file_path, model_type="audio")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(file_path)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deepfake_detector.api:app", host="0.0.0.0", port=8000, reload=True) 