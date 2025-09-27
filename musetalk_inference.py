#!/usr/bin/env python3
"""
MuseTalk Inference Script
Provides command-line interface for MuseTalk lip-sync with configurable parameters.
"""

import os
import argparse
import subprocess
import sys
from omegaconf import OmegaConf

class MuseTalkInference:
    def __init__(self):
        self.default_params = {
            "bbox_shift": 0,        # -3 to -7 for more natural (less mouth opening)
            "batch_size": 4,        # Smaller for stability, larger for speed
            "fps": 25,             # Match training data
            "version": "v15",      # Use MuseTalk 1.5
            "expression_scale": 1.0 # Future: scale expression intensity
        }
    
    def create_config(self, video_path, audio_path, **kwargs):
        """Create OmegaConf config with proper nested structure"""
        
        # Merge default params with user overrides
        params = {**self.default_params, **kwargs}
        
        config = OmegaConf.create({
            "task1": {
                "video_path": video_path,
                "audio_path": audio_path, 
                "bbox_shift": params["bbox_shift"],
                "result_name": "lipsync_output.mp4"
            }
        })
        
        return config, params
    
    def run_inference(self, video_path, audio_path, output_path, **kwargs):
        """Run MuseTalk inference with specified parameters"""
        
        print("=" * 60)
        print("MuseTalk Lip-Sync Inference")
        print("=" * 60)
        
        # Validate inputs
        if not os.path.exists(video_path):
            print(f"ERROR: Video file not found: {video_path}")
            return False
        if not os.path.exists(audio_path):
            print(f"ERROR: Audio file not found: {audio_path}")
            return False
        
        # Create config
        config, params = self.create_config(video_path, audio_path, **kwargs)
        
        print(f"Video: {video_path}")
        print(f"Audio: {audio_path}")
        print(f"Output: {output_path}")
        print(f"Expression control (bbox_shift): {params['bbox_shift']}")
        print(f"Batch size: {params['batch_size']}")
        print("-" * 60)
        
        # Save config
        config_dir = "configs/inference"
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "inference.yaml")
        OmegaConf.save(config, config_path)
        
        # Prepare command
        cmd = [
            "python", "-m", "scripts.inference",
            "--inference_config", config_path,
            "--result_dir", os.path.dirname(output_path),
            "--unet_model_path", "models/musetalkV15/unet.pth",
            "--unet_config", "models/musetalkV15/musetalk.json", 
            "--version", params["version"],
            "--ffmpeg_path", "/usr/bin/ffmpeg",
            "--whisper_dir", "./models/whisper",
            "--batch_size", str(params["batch_size"]),
            "--fps", str(params["fps"])
        ]
        
        try:
            # Clean old outputs
            output_dir = os.path.dirname(output_path)
            for f in os.listdir(output_dir):
                if f.endswith(".mp4") and "temp" not in f:
                    try:
                        os.remove(os.path.join(output_dir, f))
                    except:
                        pass
            
            print("Starting inference...")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app")
            
            # Show output
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            # Find generated files
            output_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith((".mp4", ".avi")) and "temp" not in file:
                        full_path = os.path.join(root, file)
                        output_files.append(full_path)
            
            if output_files:
                # Find the best output file
                target_file = output_files[0]
                for f in output_files:
                    if "lipsync_output" in f or "task1" in f:
                        target_file = f
                        break
                
                # Copy to final location
                import shutil
                shutil.copy(target_file, output_path)
                
                # Show success info
                size_mb = os.path.getsize(output_path) / (1024*1024)
                print("=" * 60)
                print(f"SUCCESS! Generated: {output_path}")
                print(f"File size: {size_mb:.1f} MB")
                print("=" * 60)
                
                return True
            else:
                print("ERROR: No output files generated")
                return False
                
        except Exception as e:
            print(f"ERROR during inference: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="MuseTalk Lip-Sync Inference")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--audio", required=True, help="Input audio path")
    parser.add_argument("--output", default="/app/output/lipsync_result.mp4", help="Output video path")
    
    # Expression control parameters
    parser.add_argument("--bbox_shift", type=int, default=0, 
                       help="Expression control: 0=normal, -3 to -7=more natural (less mouth opening)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (4=stable, 8=faster if enough VRAM)")
    parser.add_argument("--fps", type=int, default=25,
                       help="Output video FPS")
    
    # Preset modes for convenience
    parser.add_argument("--natural", action="store_true",
                       help="Use natural expression settings (bbox_shift=-3)")
    parser.add_argument("--subtle", action="store_true", 
                       help="Use subtle expression settings (bbox_shift=-5)")
    
    args = parser.parse_args()
    
    # Apply presets
    if args.natural:
        args.bbox_shift = -3
        print("Using natural expression preset (bbox_shift=-3)")
    elif args.subtle:
        args.bbox_shift = -5
        print("Using subtle expression preset (bbox_shift=-5)")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run inference
    inference = MuseTalkInference()
    success = inference.run_inference(
        video_path=args.video,
        audio_path=args.audio,
        output_path=args.output,
        bbox_shift=args.bbox_shift,
        batch_size=args.batch_size,
        fps=args.fps
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()