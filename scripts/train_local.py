"""
Train YOLOv11 locally using Ultralytics
Alternative to Roboflow if you want .pt file
"""

from ultralytics import YOLO
from pathlib import Path
from loguru import logger


def train_local():
    """
    Train YOLOv11 locally using your dataset
    This will give you a .pt file you can use offline
    """
    
    logger.info("=== LOCAL TRAINING WITH ULTRALYTICS ===\n")
    
    # Check if dataset exists
    data_yaml = Path("data/red_light_violation_dataset/data.yaml")
    
    if not data_yaml.exists():
        logger.error(f"Dataset not found: {data_yaml}")
        logger.info("Make sure dataset is extracted at: data/red_light_violation_dataset/")
        return
    
    logger.info(f"✓ Dataset found: {data_yaml}")
    
    # Load pre-trained YOLOv11 small
    logger.info("\nLoading YOLOv11-Small pre-trained model...")
    model = YOLO('yolov11s.pt')  # Will auto-download if not exists
    
    logger.info("✓ Model loaded")
    
    # Train
    logger.info("\nStarting training...")
    logger.info("This will take 2-4 hours depending on your GPU")
    logger.info("Press Ctrl+C to stop\n")
    
    results = model.train(
        data=str(data_yaml),
        epochs=100,  # Start with 100, can increase later
        imgsz=640,
        batch=16,  # Adjust based on your GPU memory
        device=0,  # 0 for GPU, 'cpu' for CPU
        project='runs/train',
        name='yolov11_red_light',
        exist_ok=True,
        patience=50,  # Early stopping
        save=True,
        plots=True
    )
    
    logger.info("\n✓ Training complete!")
    logger.info(f"Best model saved at: runs/train/yolov11_red_light/weights/best.pt")
    
    return results


def quick_train_check():
    """
    Quick check if everything is ready for local training
    """
    import torch
    
    print("\n=== PRE-TRAINING CHECKLIST ===\n")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("✗ PyTorch not installed")
        print("  Install: pip install torch torchvision")
        return False
    
    # Check Ultralytics
    try:
        import ultralytics
        print(f"✓ Ultralytics: {ultralytics.__version__}")
    except:
        print("✗ Ultralytics not installed")
        print("  Install: pip install ultralytics")
        return False
    
    # Check dataset
    data_yaml = Path("data/red_light_violation_dataset/data.yaml")
    if data_yaml.exists():
        print(f"✓ Dataset: {data_yaml}")
    else:
        print(f"✗ Dataset not found: {data_yaml}")
        return False
    
    # Check disk space
    import shutil
    stats = shutil.disk_usage(".")
    free_gb = stats.free / (1024**3)
    print(f"✓ Free disk space: {free_gb:.1f} GB")
    
    if free_gb < 10:
        print("⚠ Warning: Low disk space! Training needs ~10GB")
    
    print("\n✓ Ready to train!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("LOCAL TRAINING OPTION")
    print("="*60)
    print("\nPros:")
    print("✓ Get .pt file for offline use")
    print("✓ No API limits")
    print("✓ Full control over training")
    print("\nCons:")
    print("✗ Requires GPU (2-4 hours) or CPU (12-24 hours)")
    print("✗ Need to install PyTorch + Ultralytics")
    print("✗ Use more disk space")
    print("\n" + "="*60)
    
    choice = input("\nDo you want to train locally? (y/n): ").lower()
    
    if choice == 'y':
        if quick_train_check():
            print("\nStarting training in 5 seconds...")
            print("Press Ctrl+C to cancel\n")
            import time
            time.sleep(5)
            train_local()
        else:
            print("\nPlease fix the issues above first.")
    else:
        print("\nUse API method instead (test_model_api.py)")
