import cv2
import torch
import numpy as np
import logging
import os
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)

class Detection:
    """Represents an object detection with bounding box and metadata."""
    
    def __init__(self, class_id, class_name, confidence, bbox):
        """
        Initialize a detection.
        
        Args:
            class_id: Numeric ID of the detected class
            class_name: String name of the detected class
            confidence: Detection confidence score (0-1)
            bbox: Bounding box as (x1, y1, x2, y2) where (x1,y1) is top-left and (x2,y2) is bottom-right
        """
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        
    @property
    def width(self):
        """Width of the bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self):
        """Height of the bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self):
        """Area of the bounding box."""
        return self.width * self.height
    
    @property
    def center(self):
        """Center point (x, y) of the bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )

class ObjectDetector:
    """
    Detects objects in images using YOLOv8 or another model specified in config.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to the model file, or None to use the path from config
        """
        self.model_path = model_path or config.DETECTION_MODEL
        self.confidence_threshold = config.DETECTION_CONFIDENCE
        self.classes = config.DETECTION_CLASSES
        
        logger.info(f"Initializing object detector with model: {self.model_path}")
        
        # Check if the model file exists
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found: {self.model_path}")
            logger.info("Downloading YOLOv8 model...")
            self.download_model()
        
        # Load the YOLO model
        try:
            self.model = YOLO(self.model_path)
            logger.info("Object detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def detect(self, frame):
        """
        Detect objects in a frame.
        
        Args:
            frame: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            logger.warning("No model available for detection")
            return []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold)[0]
            
            # Process results
            detections = []
            for result in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, confidence, class_id = result
                class_id = int(class_id)
                
                # Get class name
                class_name = results.names[class_id]
                
                # Check if we're interested in this class
                if self.classes and class_name not in self.classes:
                    continue
                
                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2))
                ))
            
            return detections
        
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []
            
    def download_model(self):
        """
        Download the YOLOv8 model if it doesn't exist.
        This can be called to ensure the model is available.
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Extract model name from path
            model_name = os.path.basename(self.model_path)
            
            # If it's a standard YOLOv8 model, we can download it directly
            if model_name in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]:
                logger.info(f"Downloading standard YOLOv8 model: {model_name}")
                # Download using ultralytics
                self.model = YOLO(model_name)
                # Save to the specified path
                self.model.save(self.model_path)
                logger.info(f"Model downloaded successfully to {self.model_path}")
            else:
                logger.error(f"Cannot automatically download custom model: {model_name}")
                logger.info("Please download the model manually and place it in the models directory")
                # Initialize a basic model as fallback
                logger.info("Loading YOLOv8n as fallback model")
                fallback_path = os.path.join(os.path.dirname(self.model_path), "yolov8n.pt")
                self.model = YOLO("yolov8n")
                self.model.save(fallback_path)
                self.model_path = fallback_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            logger.info("Using YOLOv8n as fallback")
            self.model = YOLO("yolov8n")
