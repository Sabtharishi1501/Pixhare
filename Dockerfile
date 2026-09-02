FROM python:3.11-slim

# System dependencies for DeepFace, OpenCV, FAISS, edge-tts
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required folders
# NOTE: these are still created for local scratch use (e.g. temp file
# handling during a request), but nothing written here persists between
# requests on Vercel — see the storage note below.
RUN mkdir -p static/photos static/qrcodes static/matches \
             static/guests uploads/qr_codes uploads/selfies instance

# Vercel container Functions expect the app to listen on $PORT
# (defaults to 80 if unset — Render's fixed 10000 doesn't apply here)
EXPOSE 80

# Shell form so $PORT is expanded at container start
CMD gunicorn \
    --bind 0.0.0.0:${PORT:-80} \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --preload \
    app:app