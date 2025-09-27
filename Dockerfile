# Use CUDA 12.4 base image (compatible with CUDA 12.9 on RTX 3090)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV FFMPEG_PATH=/usr/bin/ffmpeg

# Install system dependencies including all build requirements for MMlab compilation
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

# Install PyTorch 2.0.1 with CUDA 11.7 (matching versions to avoid conflicts)
# Using older PyTorch version because MMlab packages have pre-built wheels for this combination
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Install core dependencies with compatible versions
# Using older versions that work well together and avoid import conflicts
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
    xformers==0.0.20 \
    pillow==10.0.1 \
    numpy==1.24.4 \
    pydub==0.25.1 \
    face-alignment==1.3.5 \
    resampy==0.4.2 \
    gradio==3.50.2 \
    moviepy==1.0.3

# Install MMlab packages with pre-built wheels for PyTorch 2.0.1/CUDA 11.7
# Using exact versions that are compatible with each other to avoid compilation issues
RUN pip install --no-cache-dir \
    mmengine==0.8.5 \
    mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html \
    mmdet==3.1.0 \
    mmpose==1.1.0

# Create models directory structure
RUN mkdir -p models/{musetalk,musetalkV15,syncnet,dwpose,face-parse-bisent,sd-vae,whisper}

# Copy organized Python scripts for setup and inference
COPY download_models.py /app/download_models.py
COPY musetalk_inference.py /app/musetalk_inference.py  
COPY troubleshooting_fixes.py /app/troubleshooting_fixes.py

# Make scripts executable
RUN chmod +x /app/download_models.py /app/musetalk_inference.py /app/troubleshooting_fixes.py

# Download all required models using our organized script
RUN python /app/download_models.py

# Apply all troubleshooting fixes discovered during development
RUN python /app/troubleshooting_fixes.py

# Remove unnecessary files to reduce image size
RUN rm -rf \
    /root/.cache/pip \
    /tmp/* \
    /var/tmp/* \
    .git \
    docs \
    assets \
    dataset \
    *.md

# Create input/output directories  
RUN mkdir -p /app/input /app/output

# Create a convenient inference wrapper script
RUN echo '#!/bin/bash\n\
# MuseTalk Inference Wrapper\n\
# Usage: musetalk --video input.mp4 --audio input.wav [options]\n\
\n\
python /app/musetalk_inference.py "$@"\n\
' > /usr/local/bin/musetalk && chmod +x /usr/local/bin/musetalk

# Create expression presets script for convenience
RUN echo '#!/bin/bash\n\
# MuseTalk Expression Presets\n\
# Usage: musetalk-natural video.mp4 audio.wav\n\
\n\
case "$(basename "$0")" in\n\
    "musetalk-natural")\n\
        python /app/musetalk_inference.py --natural "$@"\n\
        ;;\n\
    "musetalk-subtle")\n\
        python /app/musetalk_inference.py --subtle "$@"\n\
        ;;\n\
    *)\n\
        echo "Usage: musetalk-natural or musetalk-subtle <video> <audio>"\n\
        ;;\n\
esac\n\
' > /usr/local/bin/musetalk-presets && chmod +x /usr/local/bin/musetalk-presets

# Create symlinks for preset commands
RUN ln -s /usr/local/bin/musetalk-presets /usr/local/bin/musetalk-natural
RUN ln -s /usr/local/bin/musetalk-presets /usr/local/bin/musetalk-subtle

# Set up helpful aliases and environment
RUN echo 'alias ll="ls -la"' >> /root/.bashrc && \
    echo 'alias test-musetalk="python /app/test_setup.py"' >> /root/.bashrc && \
    echo 'export PATH="/app:$PATH"' >> /root/.bashrc

# Default command shows help
CMD ["python", "/app/musetalk_inference.py", "--help"]