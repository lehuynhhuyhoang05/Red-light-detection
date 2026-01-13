"""
Download trained model from Roboflow
"""

import os
from pathlib import Path
from loguru import logger

def download_model_from_roboflow():
    """
    Download trained model from Roboflow
    
    Manual steps:
    1. Go to: https://app.roboflow.com/huyhoang/red-light-violation-detect/1
    2. Click "Versions" → Select trained version
    3. Click "Deploy" → "Download Model"
    4. Select format: "PyTorch (.pt)"
    5. Download will start automatically
    
    Or use Roboflow API (if you have API key)
    """
    
    try:
        from roboflow import Roboflow
        
        # Initialize Roboflow
        # You need to get API key from: https://app.roboflow.com/settings/api
        api_key = input("Enter your Roboflow API key (or press Enter to skip): ").strip()
        
        if not api_key:
            logger.warning("No API key provided. Please download manually from web.")
            print_manual_instructions()
            return
        
        rf = Roboflow(api_key=api_key)
        
        # Get project
        project = rf.workspace("huyhoang").project("red-light-violation-detect")
        
        # Get the trained version
        version = project.version(1)
        
        # Download model
        logger.info("Downloading model...")
        model_path = version.download("yolov11", location="models/")
        
        logger.info(f"✓ Model downloaded to: {model_path}")
        
        return model_path
        
    except ImportError:
        logger.error("Roboflow library not installed!")
        logger.info("Install: pip install roboflow")
        print_manual_instructions()
        
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        print_manual_instructions()


def print_manual_instructions():
    """Print manual download instructions"""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\n1. Open browser and go to:")
    print("   https://app.roboflow.com/huyhoang/red-light-violation-detect/1")
    print("\n2. Login to your Roboflow account")
    print("\n3. Click on the 'Versions' tab")
    print("\n4. Find the trained version (should be v1)")
    print("\n5. Click 'View Model' or 'Deploy'")
    print("\n6. In the 'Export' section:")
    print("   - Format: PyTorch")
    print("   - Click 'Download'")
    print("\n7. Extract the downloaded ZIP file")
    print("\n8. Move the .pt file to: models/yolov11_red_light.pt")
    print("\n9. You should also get a 'data.yaml' file")
    print("="*60 + "\n")


def verify_model():
    """Verify if model file exists"""
    model_path = Path("models/yolov11_red_light.pt")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Model found: {model_path}")
        logger.info(f"✓ Size: {size_mb:.2f} MB")
        return True
    else:
        logger.warning(f"✗ Model not found at: {model_path}")
        logger.info("Please download the model first.")
        return False


if __name__ == "__main__":
    print("\n=== ROBOFLOW MODEL DOWNLOADER ===\n")
    
    # Check if model already exists
    if verify_model():
        logger.info("Model already downloaded!")
        logger.info("Ready to use.")
    else:
        # Try to download
        download_model_from_roboflow()
