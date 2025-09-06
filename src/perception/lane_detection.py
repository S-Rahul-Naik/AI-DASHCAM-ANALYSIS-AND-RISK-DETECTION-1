import cv2
import numpy as np
import logging
import os
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)

class Lane:
    """Represents a detected lane with position and curvature."""
    
    def __init__(self, points, is_left=False, is_right=False):
        """
        Initialize a lane.
        
        Args:
            points: List of (x, y) points defining the lane line
            is_left: Whether this is a left lane
            is_right: Whether this is a right lane
        """
        self.points = np.array(points)
        self.is_left = is_left
        self.is_right = is_right
        
        # Fit a polynomial to the lane points
        if len(points) >= 2:
            self.coeffs = np.polyfit(self.points[:, 1], self.points[:, 0], 2)
        else:
            self.coeffs = None
    
    def get_points_in_region(self, y_start, y_end, num_points=10):
        """
        Get evenly spaced points along the lane in a specific y-region.
        
        Args:
            y_start: Starting y-coordinate
            y_end: Ending y-coordinate
            num_points: Number of points to generate
            
        Returns:
            List of (x, y) points
        """
        if self.coeffs is None:
            return []
        
        y_points = np.linspace(y_start, y_end, num_points)
        x_points = self.coeffs[0] * y_points**2 + self.coeffs[1] * y_points + self.coeffs[2]
        
        return np.column_stack((x_points, y_points)).astype(np.int32)
    
    def get_curvature(self, y_eval):
        """
        Calculate the curvature of the lane at a specific y-coordinate.
        
        Args:
            y_eval: Y-coordinate to evaluate curvature at
            
        Returns:
            Curvature in pixel space
        """
        if self.coeffs is None:
            return float('inf')
        
        # First and second derivatives
        first_deriv = 2 * self.coeffs[0] * y_eval + self.coeffs[1]
        second_deriv = 2 * self.coeffs[0]
        
        # Curvature formula: R = (1 + (first_deriv)^2)^(3/2) / |second_deriv|
        curvature = ((1 + first_deriv**2)**(3/2)) / abs(second_deriv)
        
        return curvature if curvature != float('inf') else 1e5

class LaneDetector:
    """
    Detects lane lines in images using YOLOv8 segmentation or computer vision techniques.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the lane detector.
        
        Args:
            model_path: Path to the lane detection model, or None to use the path from config
        """
        self.model_path = model_path or config.LANE_DETECTION_MODEL
        self.confidence = config.LANE_CONFIDENCE
        self.use_cv_fallback = getattr(config, 'USE_CV_FALLBACK', True)
        
        logger.info(f"Initializing lane detector with model: {self.model_path}")
        
        # Flag for using classical CV approach if model not available
        self.use_cv_approach = True
        self.model = None
        
        # Check if model file exists
        if os.path.exists(self.model_path):
            try:
                # Load YOLOv8 segmentation model
                self.model = YOLO(self.model_path)
                logger.info("Lane detection model loaded successfully")
                self.use_cv_approach = False
            except Exception as e:
                logger.error(f"Failed to load lane detection model: {e}")
                logger.info("Falling back to classical computer vision approach")
                self.use_cv_approach = True
        else:
            logger.warning(f"Lane detection model not found: {self.model_path}")
            logger.info("Falling back to classical computer vision approach")
            self.use_cv_approach = True
    
    def detect(self, frame):
        """
        Detect lanes in a frame.
        
        Args:
            frame: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of Lane objects
        """
        if not self.use_cv_approach and self.model is not None:
            try:
                return self._detect_with_model(frame)
            except Exception as e:
                logger.error(f"Error using ML model for lane detection: {e}")
                logger.info("Falling back to CV approach")
                return self._detect_cv(frame)
        else:
            return self._detect_cv(frame)
    
    def _detect_with_model(self, frame):
        """
        Detect lanes using YOLOv8 segmentation model.
        
        Args:
            frame: Image as numpy array
            
        Returns:
            List of Lane objects
        """
        # Run inference with YOLOv8 segmentation model
        results = self.model(frame, conf=self.confidence)
        
        # Initialize lanes list
        lanes = []
        
        # Process segmentation results
        if results and len(results) > 0:
            # Get the masks
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                
                # Process each mask
                for i, mask in enumerate(masks):
                    # Convert mask to binary image
                    binary_mask = (mask > 0.5).astype(np.uint8) * 255
                    
                    # Find contours in the mask
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Get the largest contour
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        # Extract points from contour
                        points = largest_contour.reshape(-1, 2)
                        
                        # Determine if it's a left or right lane based on position
                        h, w = frame.shape[:2]
                        avg_x = np.mean(points[:, 0])
                        
                        is_left = avg_x < w / 2
                        is_right = avg_x >= w / 2
                        
                        # Create Lane object
                        lane = Lane(points, is_left=is_left, is_right=is_right)
                        lanes.append(lane)
        
        # If no lanes were detected with the model, fall back to CV approach
        if not lanes:
            logger.warning("No lanes detected with model, falling back to CV approach")
            return self._detect_cv(frame)
        
        return lanes
    
    def _detect_cv(self, frame):
        """
        Detect lanes using classical computer vision techniques.
        
        Args:
            frame: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of Lane objects
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Define region of interest
            height, width = edges.shape
            mask = np.zeros_like(edges)
            
            # Define a polygon for the region of interest (bottom half of the image)
            polygon = np.array([
                [(0, height), (0, height * 0.6), (width, height * 0.6), (width, height)]
            ], np.int32)
            
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Apply Hough transform to detect lines
            lines = cv2.HoughLinesP(
                masked_edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=20, 
                minLineLength=20, 
                maxLineGap=300
            )
            
            # Process detected lines
            left_lines = []
            right_lines = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate slope
                    if x2 - x1 == 0:
                        continue  # Skip vertical lines
                    
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Filter out horizontal lines
                    if abs(slope) < 0.3:
                        continue
                    
                    # Categorize lines as left or right based on slope and position
                    if slope < 0 and x1 < width / 2:
                        left_lines.append(line[0])
                    elif slope > 0 and x1 > width / 2:
                        right_lines.append(line[0])
            
            # Create Lane objects
            lanes = []
            
            # Process left lane lines
            if left_lines:
                left_points = np.vstack(left_lines)
                left_lane = Lane(left_points, is_left=True)
                lanes.append(left_lane)
            
            # Process right lane lines
            if right_lines:
                right_points = np.vstack(right_lines)
                right_lane = Lane(right_points, is_right=True)
                lanes.append(right_lane)
            
            return lanes
        
        except Exception as e:
            logger.error(f"Error during lane detection: {e}")
            return []
