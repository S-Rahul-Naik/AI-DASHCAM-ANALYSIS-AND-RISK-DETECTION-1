import cv2
import numpy as np
import logging
import os
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)

class Anonymizer:
    """
    Anonymizes sensitive information in images (faces, license plates).
    """
    
    def __init__(self, face_model_path=None, plate_model_path=None):
        """
        Initialize the anonymizer.
        
        Privacy protection feature has been disabled.
        This is now a placeholder that doesn't load any models.
        
        Args:
            face_model_path: Not used (kept for compatibility)
            plate_model_path: Not used (kept for compatibility)
        """
        logger.info("Initializing anonymizer (privacy protection disabled)")
        
        # No models are initialized since privacy feature is disabled
        self.face_detector = None
        self.plate_detector = None
    
    def anonymize(self, frame):
        """
        This function now simply returns the original frame without anonymization.
        Privacy protection feature has been disabled.
        
        Args:
            frame: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            The original frame without modification
        """
        return frame
    
    def _anonymize_faces(self, frame):
        """
        Anonymize faces in a frame.
        
        Args:
            frame: Image as numpy array
            
        Returns:
            Frame with anonymized faces
        """
        try:
            # Check if we're using YOLOv8 or OpenCV
            if isinstance(self.face_detector, YOLO):
                # Use YOLOv8 face detection
                results = self.face_detector(frame, conf=0.25)
                
                # Process each detection
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Apply strong Gaussian blur
                        face_roi = frame[y1:y2, x1:x2]
                        if face_roi.size > 0:  # Check if ROI is valid
                            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                            frame[y1:y2, x1:x2] = blurred_face
            else:
                # Use OpenCV's Haar Cascade face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Apply blur to each face
                for (x, y, w, h) in faces:
                    # Apply strong Gaussian blur
                    face_roi = frame[y:y+h, x:x+w]
                    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                    frame[y:y+h, x:x+w] = blurred_face
            
            return frame
        
        except Exception as e:
            logger.error(f"Error during face anonymization: {e}")
            return frame
    
    def _anonymize_license_plates(self, frame):
        """
        Anonymize license plates in a frame.
        
        Args:
            frame: Image as numpy array
            
        Returns:
            Frame with anonymized license plates
        """
        try:
            # Use YOLOv8 license plate detection
            if isinstance(self.plate_detector, YOLO):
                results = self.plate_detector(frame, conf=0.25)
                
                # Process each detection
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Apply strong Gaussian blur
                        plate_roi = frame[y1:y2, x1:x2]
                        if plate_roi.size > 0:  # Check if ROI is valid
                            blurred_plate = cv2.GaussianBlur(plate_roi, (99, 99), 30)
                            frame[y1:y2, x1:x2] = blurred_plate
            
            return frame
        
        except Exception as e:
            logger.error(f"Error during license plate anonymization: {e}")
            return frame
