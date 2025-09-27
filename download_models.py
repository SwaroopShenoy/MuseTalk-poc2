#!/usr/bin/env python3
"""
MuseTalk Model Download Script
Handles downloading all required models with proper error handling and fallbacks.
"""

import os
import requests
import subprocess
import sys

def download_file(url, filepath, description=""):
    """Download a file with progress indication"""
    print(f"Downloading {description}: {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
        print(f"\n  ✓ Downloaded {description}")
        return True
    except Exception as e:
        print(f"\n  ✗ Failed to download {description}: {e}")
        return False

def download_with_hf_cli():
    """Attempt to download MuseTalk models using HuggingFace CLI"""
    try:
        print("Installing huggingface_hub CLI...")
        subprocess.run(['pip', 'install', 'huggingface_hub[cli]'], check=True, capture_output=True)
        
        print("Downloading MuseTalk models via HF CLI...")
        result = subprocess.run([
            'hf', 'download', 
            'TMElyralab/MuseTalk', 
            '--local-dir', './models/',
            '--include', 'musetalk/*',
            '--include', 'musetalkV15/*'
        ], check=True, capture_output=True, text=True)
        
        print("✓ HuggingFace CLI download successful")
        return True
    except Exception as e:
        print(f"✗ HuggingFace CLI download failed: {e}")
        return False

def manual_download_musetalk():
    """Manually download MuseTalk 1.5 files as fallback"""
    print("Attempting manual download of MuseTalk 1.5...")
    
    # Create directory
    os.makedirs("models/musetalkV15", exist_ok=True)
    
    models = {
        "models/musetalkV15/musetalk.json": {
            "url": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json",
            "description": "MuseTalk 1.5 config"
        },
        "models/musetalkV15/unet.pth": {
            "url": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth", 
            "description": "MuseTalk 1.5 model weights (3.2GB)"
        }
    }
    
    success = True
    for filepath, info in models.items():
        if not os.path.exists(filepath):
            if not download_file(info["url"], filepath, info["description"]):
                success = False
    
    return success

def download_other_models():
    """Download supporting models (pose detection, face parsing, etc.)"""
    models = {
        "models/dwpose/dw-ll_ucoco_384.pth": {
            "url": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth",
            "description": "DWPose model"
        },
        "models/face-parse-bisent/79999_iter.pth": {
            "url": "https://huggingface.co/vivym/face-parsing-bisenet/resolve/main/79999_iter.pth",
            "description": "Face parsing model"
        },
        "models/face-parse-bisent/resnet18-5c106cde.pth": {
            "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            "description": "ResNet-18 backbone"
        },
        "models/sd-vae/diffusion_pytorch_model.bin": {
            "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin",
            "description": "Stable Diffusion VAE"
        },
        "models/sd-vae/config.json": {
            "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json",
            "description": "VAE config"
        }
    }
    
    print("Downloading supporting models...")
    success = True
    for filepath, info in models.items():
        if not os.path.exists(filepath):
            if not download_file(info["url"], filepath, info["description"]):
                success = False
    
    return success

def setup_whisper_models():
    """Set up Whisper models in the expected directory structure"""
    print("Setting up Whisper models...")
    
    # Create expected whisper directory
    os.makedirs("models/whisper", exist_ok=True)
    
    # Download whisper-tiny to multiple locations to satisfy different expectations
    whisper_files = {
        "config.json": "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json",
        "preprocessor_config.json": "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json", 
        "pytorch_model.bin": "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
    }
    
    success = True
    for filename, url in whisper_files.items():
        filepath = f"models/whisper/{filename}"
        if not os.path.exists(filepath):
            if not download_file(url, filepath, f"Whisper {filename}"):
                success = False
    
    return success

def main():
    """Main download orchestration"""
    print("=" * 60)
    print("MuseTalk Model Download Script")
    print("=" * 60)
    
    # Try HuggingFace CLI first
    hf_success = download_with_hf_cli()
    
    # Manual fallback for MuseTalk models if HF CLI failed
    if not hf_success:
        print("Falling back to manual download...")
        manual_download_musetalk()
    
    # Always download other models (these rarely work via HF CLI)
    download_other_models()
    
    # Set up whisper models
    setup_whisper_models()
    
    # Verify critical files exist
    critical_files = [
        "models/musetalkV15/unet.pth",
        "models/musetalkV15/musetalk.json", 
        "models/whisper/config.json",
        "models/dwpose/dw-ll_ucoco_384.pth"
    ]
    
    print("\nVerifying critical files...")
    missing_files = []
    for filepath in critical_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024*1024)
            print(f"  ✓ {filepath} ({size:.1f} MB)")
        else:
            print(f"  ✗ {filepath} - MISSING")
            missing_files.append(filepath)
    
    if missing_files:
        print(f"\n⚠️  WARNING: {len(missing_files)} critical files are missing!")
        print("You may need to download them manually.")
        return False
    else:
        print("\n🎉 All models downloaded successfully!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)