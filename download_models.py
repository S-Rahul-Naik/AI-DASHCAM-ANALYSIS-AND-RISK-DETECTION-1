import os
import sys
import logging
from pathlib import Path

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import model downloader
from src.model_downloader import download_all_models, ensure_model_exists
from src.config import DETECTION_MODEL, LANE_DETECTION_MODEL, FACE_DETECTION_MODEL, PLATE_DETECTION_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("model_downloader")

def main():
    """Main function to download all required models."""
    logger.info("Starting model download process")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Ensure ultralytics directory exists
    os.makedirs("models/ultralytics", exist_ok=True)
    
    # Download all models
    downloaded_models = download_all_models()
    
    if not downloaded_models:
        logger.warning("No models were downloaded. This may indicate an error or that all models already exist.")
    else:
        logger.info(f"Successfully downloaded/verified {len(downloaded_models)} models")
    
    # Check if all required models are available
    all_models_available = (
        os.path.exists(DETECTION_MODEL) and
        os.path.exists(LANE_DETECTION_MODEL) and
        os.path.exists(FACE_DETECTION_MODEL) and
        os.path.exists(PLATE_DETECTION_MODEL)
    )
    
    if all_models_available:
        logger.info("All required models are available")
        print("\nSuccess! All models have been downloaded or verified.")
        print("The following models are ready to use:")
        print(f"  - Object Detection: {DETECTION_MODEL}")
        print(f"  - Lane Detection: {LANE_DETECTION_MODEL}")
        print(f"  - Face Detection: {FACE_DETECTION_MODEL}")
        print(f"  - License Plate Detection: {PLATE_DETECTION_MODEL}")
        print("\nYou can now run the application with: python run.py")
    else:
        logger.error("Some models are missing")
        print("\nWarning: Not all models are available.")
        print("Missing models:")
        if not os.path.exists(DETECTION_MODEL):
            print(f"  - Object Detection Model: {DETECTION_MODEL}")
        if not os.path.exists(LANE_DETECTION_MODEL):
            print(f"  - Lane Detection Model: {LANE_DETECTION_MODEL}")
        if not os.path.exists(FACE_DETECTION_MODEL):
            print(f"  - Face Detection Model: {FACE_DETECTION_MODEL}")
        if not os.path.exists(PLATE_DETECTION_MODEL):
            print(f"  - License Plate Detection Model: {PLATE_DETECTION_MODEL}")
        print("\nPlease check the log for errors or try running the script again.")
    
    return all_models_available

if __name__ == "__main__":
    main()
    main()
