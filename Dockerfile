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
RUN mkdir -p static/photos static/qrcodes static/matches \
             static/guests uploads/qr_codes uploads/selfies instance

# Expose Render's default port
EXPOSE 10000

# Production server — gunicorn (NOT flask dev server)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:10000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--preload", \
     "app:app"]