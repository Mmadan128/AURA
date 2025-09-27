# Use Python 3.12 slim
FROM python:3.12-slim

# Set environment
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip

# Force compatible versions for Python 3.12 if needed
# e.g., torch and faiss sometimes require pre-release or specific wheels
RUN pip install --pre --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose HF Spaces port
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]
