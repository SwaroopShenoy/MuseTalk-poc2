# Use CUDA 12.4 base image (compatible with CUDA 12.9)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies including all build requirements for MMlab
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    cmake \
    build-essential \
    ninja-build \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Clone MuseTalk repository
RUN git clone https://github.com/TMElyralab/MuseTalk.git .

# Install PyTorch 2.0.1 with CUDA 11.7 (matching versions)
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Install core dependencies with compatible versions
RUN pip install \
    diffusers==0.21.4 \
    transformers==4.35.2 \
    huggingface_hub==0.17.3 \
    accelerate==0.24.1 \
    opencv-python==4.8.1.78 \
    opencv-contrib-python==4.8.1.78 \
    scipy==1.11.4 \
    scikit-image==0.21.0 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    omegaconf==2.3.0 \
    einops==0.7.0 \
    xformers==0.0.22.post7 \
    pillow==10.0.1 \
    numpy==1.24.4 \
    pydub==0.25.1 \
    face-alignment==1.3.5 \
    resampy==0.4.2

# Install MMlab packages with pre-built wheels for PyTorch 2.0.1/CUDA 11.7
RUN pip install --no-cache-dir \
    mmengine==0.8.5 \
    mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html \
    mmdet==3.1.0 \
    mmpose==1.1.0

# Skip whisper installation - MuseTalk likely includes its own audio processing
# If needed, can install later: pip install --no-deps openai-whisper tiktoken

# Create models directory structure
RUN mkdir -p models/{musetalk,musetalkV15,syncnet,dwpose,face-parse-bisent,sd-vae,whisper}

# Download model weights script - simplified with working URLs
COPY <<EOF /app/download_models.py
import os
import requests
import subprocess

def download_file(url, filepath):
    print(f"Downloading {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filepath}")

# Install huggingface_hub for downloading MuseTalk
subprocess.run(['pip', 'install', 'huggingface_hub[cli]'], check=True)

# Download MuseTalk models using hf CLI
try:
    subprocess.run([
        'hf', 'download', 
        'TMElyralab/MuseTalk', 
        '--local-dir', './models/',
        '--include', 'musetalk/*',
        '--include', 'musetalkV15/*'
    ], check=True)
    print("Downloaded MuseTalk models successfully!")
except:
    print("HF CLI failed, skipping MuseTalk download - will need manual setup")

# Download other models with working URLs
models = {
    "models/dwpose/dw-ll_ucoco_384.pth": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth",
    "models/face-parse-bisent/79999_iter.pth": "https://huggingface.co/vivym/face-parsing-bisenet/resolve/main/79999_iter.pth",
    "models/face-parse-bisent/resnet18-5c106cde.pth": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "models/sd-vae/diffusion_pytorch_model.bin": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin",
    "models/sd-vae/config.json": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json"
}

for filepath, url in models.items():
    if not os.path.exists(filepath):
        try:
            download_file(url, filepath)
        except Exception as e:
            print(f"Failed to download {filepath}: {e}")

print("Model download completed!")
EOF

RUN python download_models.py

# Remove unnecessary files to reduce image size
RUN rm -rf \
    /root/.cache/pip \
    /tmp/* \
    /var/tmp/* \
    .git \
    download_models.py \
    docs \
    assets \
    dataset \
    *.md

# Add the mock mmpose to Python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Create input/output directories
RUN mkdir -p /app/input /app/output

# Default command runs MuseTalk 1.5 inference
CMD ["python", "-m", "scripts.inference", \
     "--inference_config", "configs/inference/test.yaml", \
     "--result_dir", "/app/output", \
     "--unet_model_path", "models/musetalkV15/unet.pth", \
     "--unet_config", "models/musetalkV15/musetalk.json", \
     "--version", "v15", \
     "--ffmpeg_path", "/usr/bin/ffmpeg"]