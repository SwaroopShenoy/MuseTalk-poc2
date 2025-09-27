#!/usr/bin/env python3
"""
MuseTalk Troubleshooting Fixes
Applies all runtime fixes discovered during development.
"""

import os
import subprocess
import sys
import shutil

def fix_whisper_paths():
    """Fix whisper model paths - addresses multiple path expectations"""
    print("Fixing Whisper model paths...")
    
    # Create all expected whisper directory structures
    paths_to_create = [
        "models/whisper",
        "models/whisper-tiny", 
        "models/openai/whisper-tiny"
    ]
    
    for path in paths_to_create:
        os.makedirs(path, exist_ok=True)
    
    # If we have whisper files in any location, copy to all expected locations
    source_files = []
    for check_path in paths_to_create:
        if os.path.exists(os.path.join(check_path, "config.json")):
            source_files = [
                os.path.join(check_path, "config.json"),
                os.path.join(check_path, "preprocessor_config.json"),
                os.path.join(check_path, "pytorch_model.bin")
            ]
            break
    
    if source_files:
        for target_path in paths_to_create:
            for src_file in source_files:
                if os.path.exists(src_file):
                    filename = os.path.basename(src_file)
                    dst_file = os.path.join(target_path, filename)
                    if not os.path.exists(dst_file):
                        try:
                            shutil.copy2(src_file, dst_file)
                            print(f"  Copied {filename} to {target_path}")
                        except Exception as e:
                            print(f"  Failed to copy {filename}: {e}")

def create_missing_directories():
    """Create any missing directory structures"""
    print("Creating missing directories...")
    
    directories = [
        "models/musetalk",
        "models/musetalkV15", 
        "models/syncnet",
        "models/dwpose",
        "models/face-parse-bisent",
        "models/sd-vae",
        "models/whisper",
        "configs/inference",
        "output",
        "input"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"  Created directory: {directory}")

def install_missing_dependencies():
    """Install dependencies that were discovered missing during runtime"""
    print("Installing missing dependencies...")
    
    # Dependencies we discovered were needed during troubleshooting
    runtime_deps = [
        "gdown",                # For Google Drive downloads
        "ffmpeg-python",        # Python ffmpeg bindings
        "moviepy",              # Video processing
        "beautifulsoup4",       # HTML parsing (gdown dependency)
        "PySocks",              # SOCKS proxy support
        "librosa",              # Audio processing
        "imageio[ffmpeg]"       # Video I/O with ffmpeg support
    ]
    
    for dep in runtime_deps:
        try:
            print(f"  Installing {dep}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], check=True, capture_output=True)
            print(f"  ✓ {dep}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to install {dep}: {e}")

def fix_xformers_warnings():
    """Install compatible xformers version to reduce warnings"""
    print("Attempting to fix xformers compatibility...")
    
    try:
        # Uninstall incompatible xformers
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y", "xformers"
        ], capture_output=True)
        
        # Install compatible version for PyTorch 2.0.1+cu117
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "xformers==0.0.20", "--no-deps"
        ], check=True, capture_output=True)
        
        print("  ✓ Installed compatible xformers")
    except Exception as e:
        print(f"  ⚠ Could not fix xformers (warnings only): {e}")

def check_gpu_setup():
    """Verify GPU setup and CUDA availability"""
    print("Checking GPU setup...")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        else:
            print("  ⚠ CUDA not available - will use CPU")
            
    except ImportError:
        print("  ✗ PyTorch not installed")

def verify_model_files():
    """Verify all required model files are present"""
    print("Verifying model files...")
    
    required_files = {
        "models/musetalkV15/unet.pth": "MuseTalk 1.5 model weights",
        "models/musetalkV15/musetalk.json": "MuseTalk 1.5 config",
        "models/whisper/config.json": "Whisper config",
        "models/whisper/preprocessor_config.json": "Whisper preprocessor config",
        "models/dwpose/dw-ll_ucoco_384.pth": "DWPose model",
        "models/face-parse-bisent/79999_iter.pth": "Face parsing model",
        "models/sd-vae/diffusion_pytorch_model.bin": "VAE model",
        "models/sd-vae/config.json": "VAE config"
    }
    
    missing_files = []
    for filepath, description in required_files.items():
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"  ✓ {description}: {size_mb:.1f}MB")
        else:
            print(f"  ✗ {description}: MISSING")
            missing_files.append(filepath)
    
    if missing_files:
        print(f"\n⚠ {len(missing_files)} required files are missing!")
        print("Run the model download script to fetch them.")
        return False
    else:
        print("All required model files present")
        return True

def create_test_script():
    """Create a quick test script to verify everything works"""
    test_script = '''#!/usr/bin/env python3
"""Quick test to verify MuseTalk setup"""
import sys
sys.path.append("/app")

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
        
        from musetalk.utils.utils import load_all_model
        print("  ✓ MuseTalk utils")
        
        from omegaconf import OmegaConf
        print("  ✓ OmegaConf")
        
        import librosa
        print("  ✓ Librosa")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("Testing model loading...")
    try:
        import torch
        from musetalk.utils.utils import load_all_model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae, unet, pe = load_all_model(
            unet_model_path="./models/musetalkV15/unet.pth",
            vae_type="sd-vae",
            unet_config="./models/musetalkV15/musetalk.json",
            device=device
        )
        print(f"  ✓ Models loaded on {device}")
        return True
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("MuseTalk Setup Test")
    print("=" * 50)
    
    imports_ok = test_imports()
    models_ok = test_model_loading() if imports_ok else False
    
    print("=" * 50)
    if imports_ok and models_ok:
        print("🎉 All tests passed! MuseTalk is ready.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check errors above.")
        sys.exit(1)
'''
    
    with open("/app/test_setup.py", "w") as f:
        f.write(test_script)
    
    os.chmod("/app/test_setup.py", 0o755)
    print("Created test script: /app/test_setup.py")

def main():
    """Run all troubleshooting fixes"""
    print("=" * 60)
    print("MuseTalk Troubleshooting Fixes")
    print("=" * 60)
    
    create_missing_directories()
    install_missing_dependencies() 
    fix_whisper_paths()
    fix_xformers_warnings()
    check_gpu_setup()
    models_ok = verify_model_files()
    create_test_script()
    
    print("=" * 60)
    print("Troubleshooting fixes completed!")
    
    if models_ok:
        print("✓ All model files present")
        print("✓ Run: python /app/test_setup.py to verify setup")
        print("✓ Run: python /app/musetalk_inference.py --help for inference")
    else:
        print("⚠ Model files missing - run model download script first")
    
    print("=" * 60)

if __name__ == "__main__":
    main()