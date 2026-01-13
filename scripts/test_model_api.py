"""
Test trained model using Roboflow Inference API
No need to download weights - use model directly from cloud
"""

from roboflow import Roboflow
import cv2
from loguru import logger

def test_model_with_api():
    """
    Test model using Roboflow hosted inference
    FREE for reasonable usage on free tier
    """
    
    # Initialize Roboflow
    print("\n=== ROBOFLOW INFERENCE API ===\n")
    print("This uses your trained model directly from Roboflow cloud")
    print("No download needed - FREE for testing!\n")
    
    api_key = input("Enter your Roboflow API key: ").strip()
    
    if not api_key:
        print("\nTo get your API key:")
        print("1. Go to: https://app.roboflow.com/settings/api")
        print("2. Copy your API key")
        print("3. Paste it here\n")
        return
    
    try:
        # Initialize
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("huyhoang").project("red-light-violation-detect-hecrg")
        model = project.version(1).model
        
        logger.info("✓ Connected to model!")
        
        # Test with image
        test_image = input("\nEnter path to test image (or press Enter for demo): ").strip()
        
        if not test_image:
            test_image = "data/frames/frame_00001.jpg"
        
        logger.info(f"Testing with: {test_image}")
        
        # Run prediction
        result = model.predict(test_image, confidence=40, overlap=30).json()
        
        # Display results
        logger.info(f"\n✓ Detected {len(result['predictions'])} objects:")
        
        for pred in result['predictions']:
            logger.info(f"  - {pred['class']}: {pred['confidence']:.2%}")
        
        # Save visualization
        model.predict(test_image, confidence=40, overlap=30).save("test_result.jpg")
        logger.info("\n✓ Result saved to: test_result.jpg")
        
        return model
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print("\nMake sure:")
        print("1. API key is correct")
        print("2. You have internet connection")
        print("3. Model training is complete")


def batch_predict(model, image_folder: str, output_folder: str):
    """
    Run predictions on multiple images
    """
    import os
    from pathlib import Path
    
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    images = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    
    logger.info(f"\nProcessing {len(images)} images...")
    
    for i, img_path in enumerate(images, 1):
        try:
            result = model.predict(str(img_path), confidence=40, overlap=30)
            output_path = output_folder / f"result_{img_path.name}"
            result.save(str(output_path))
            
            if i % 10 == 0:
                logger.info(f"  Processed {i}/{len(images)}")
                
        except Exception as e:
            logger.error(f"  Failed {img_path.name}: {e}")
    
    logger.info(f"\n✓ Done! Results in: {output_folder}")


if __name__ == "__main__":
    # Test with API
    model = test_model_with_api()
    
    if model:
        print("\n" + "="*60)
        print("MODEL READY TO USE!")
        print("="*60)
        print("\nYou can now:")
        print("1. Test on more images: model.predict('image.jpg')")
        print("2. Run on video frames")
        print("3. Integrate into your system")
        print("\nNo download needed - all inference runs on Roboflow cloud!")
        print("="*60)
