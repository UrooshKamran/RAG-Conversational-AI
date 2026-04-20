FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_URL=http://host.docker.internal:11434 \
    WEATHER_API_KEY="" \
    EXCHANGE_API_KEY=""

WORKDIR /app

# System dependencies
RUN apt-get update --fix-missing && apt-get install -y --fix-missing \
    build-essential \
    portaudio19-dev \
    libportaudio2 \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Audio packages — must succeed
RUN pip install --no-cache-dir sounddevice pyaudio

# Core app dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    websockets==12.0 \
    requests==2.32.3 \
    psutil==6.0.0 \
    python-multipart==0.0.12 \
    faster-whisper \
    onnxruntime \
    piper-tts \
    scipy \
    numpy

# Assignment 4 dependencies (no torch needed at runtime)
RUN pip install --no-cache-dir \
    chromadb==0.5.3 \
    httpx==0.27.0 \
    pydantic==2.9.2 \
    faiss-cpu==1.8.0

# Copy all application code
COPY . .

RUN mkdir -p voices

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
