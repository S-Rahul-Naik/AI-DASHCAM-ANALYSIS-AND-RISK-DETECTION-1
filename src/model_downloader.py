"""
Model downloader for AI Dashcam
This module downloads required ML models if they don't exist locally.
"""

import os
import logging
import shutil
from pathlib import Path
import requests
from tqdm import tqdm
from ultralytics import YOLO

from src import config

logger = logging.getLogger(__name__)

# Define model URLs (for custom models not in YOLO)
MODEL_URLS = {
    'yolov8n-face.pt': 'https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt',
    'yolov8n-plate.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'  # Placeholder, would be an actual plate model
}

def download_file(url, destination):
    """
    Download a file from a URL to a destination with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file to
    
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress bar
        with open(destination, 'wb') as file, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
        
        return True
    
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {e}")
        return False

def ensure_model_exists(model_path):
    """
    Ensure a model exists at the specified path, downloading it if necessary.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        True if the model exists or was downloaded successfully, False otherwise
    """
    # If the model already exists, return True
    if os.path.exists(model_path):
        logger.info(f"Model already exists: {model_path}")
        return True
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Get the model filename
    model_filename = os.path.basename(model_path)
    
    # Try to download using ultralytics for standard YOLOv8 models
    standard_yolo_models = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"
    ]
    
    if model_filename in standard_yolo_models:
        try:
            logger.info(f"Downloading standard YOLOv8 model: {model_filename}")
            model = YOLO(model_filename)
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # Save to the specified path
            model.save(model_path)
            logger.info(f"Model downloaded and saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download standard YOLOv8 model: {e}")
    
    # Try to download from custom URLs
    if model_filename in MODEL_URLS:
        url = MODEL_URLS[model_filename]
        logger.info(f"Downloading model from {url} to {model_path}")
        success = download_file(url, model_path)
        
        if success:
            logger.info(f"Model downloaded successfully to {model_path}")
            return True
        else:
            logger.error(f"Failed to download model from {url}")
    
    logger.error(f"No download source available for model: {model_filename}")
    return False

def download_all_models():
    """
    Download all required models.
    
    Returns:
        List of models that were successfully downloaded
    """
    models = [
        config.DETECTION_MODEL,
        config.LANE_DETECTION_MODEL,
        config.FACE_DETECTION_MODEL,
        config.PLATE_DETECTION_MODEL
    ]
    
    successful = []
    
    for model_path in models:
        if ensure_model_exists(model_path):
            successful.append(model_path)
    
    return successful

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Download all models
    downloaded = download_all_models()
    
    if downloaded:
        print(f"Successfully downloaded {len(downloaded)} models:")
        for model in downloaded:
            print(f"  - {model}")
    else:
        print("No models were downloaded.")
