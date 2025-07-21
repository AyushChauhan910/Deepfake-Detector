# Use official Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Create models directory and copy .pth files if present
RUN mkdir -p /app/models
COPY models/*.pth /app/models/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install .[all] && \
    pip install fastapi uvicorn

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "src.deepfake_detector.api:app", "--host", "0.0.0.0", "--port", "8000"]
