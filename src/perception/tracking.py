import cv2
import numpy as np
import logging
from collections import defaultdict

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.perception.detection import Detection

logger = logging.getLogger(__name__)

class TrackedObject:
    """Represents a tracked object with ID and history."""
    
    def __init__(self, detection, track_id):
        """
        Initialize a tracked object.
        
        Args:
            detection: Detection object
            track_id: Unique tracking ID
        """
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.bbox = detection.bbox
        self.center = detection.center
        self.positions = [self.center]  # Track position history
        self.last_seen = 0  # Frame counter since last detection
        self.velocity = (0, 0)  # Estimated velocity (dx, dy) in pixels per frame
        
    def update(self, detection):
        """
        Update the tracked object with a new detection.
        
        Args:
            detection: New Detection object
        """
        # Update properties
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.bbox = detection.bbox
        
        # Update position history
        new_center = detection.center
        self.positions.append(new_center)
        
        # Keep only the last 10 positions
        if len(self.positions) > 10:
            self.positions = self.positions[-10:]
        
        # Reset last_seen counter
        self.last_seen = 0
        
        # Calculate velocity if we have at least 2 positions
        if len(self.positions) >= 2:
            prev_x, prev_y = self.positions[-2]
            curr_x, curr_y = self.positions[-1]
            self.velocity = (curr_x - prev_x, curr_y - prev_y)
        
        # Update center
        self.center = new_center
    
    def predict_position(self, frames_ahead=1):
        """
        Predict future position based on velocity.
        
        Args:
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted (x, y) position
        """
        x, y = self.center
        vx, vy = self.velocity
        return (x + vx * frames_ahead, y + vy * frames_ahead)

class ObjectTracker:
    """
    Tracks objects across frames using a simple IoU-based tracking algorithm.
    """
    
    def __init__(self, max_disappeared=30, min_iou=0.3):
        """
        Initialize the object tracker.
        
        Args:
            max_disappeared: Maximum number of frames an object can disappear before being forgotten
            min_iou: Minimum IoU (Intersection over Union) for matching detections to tracks
        """
        self.next_track_id = 0
        self.tracks = {}  # Dictionary of track_id -> TrackedObject
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        
        logger.info("Object tracker initialized")
    
    def update(self, detections):
        """
        Update tracking with new detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of TrackedObject objects
        """
        # If no detections, increment disappeared counters
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].last_seen += 1
                
                # Remove tracks that have disappeared for too long
                if self.tracks[track_id].last_seen > self.max_disappeared:
                    del self.tracks[track_id]
            
            return list(self.tracks.values())
        
        # If no existing tracks, create new tracks for all detections
        if len(self.tracks) == 0:
            for detection in detections:
                self.tracks[self.next_track_id] = TrackedObject(detection, self.next_track_id)
                self.next_track_id += 1
        else:
            # Calculate IoU between all detections and tracks
            iou_matrix = np.zeros((len(detections), len(self.tracks)))
            detection_indices = list(range(len(detections)))
            track_indices = list(range(len(self.tracks)))
            
            # Fill the IoU matrix
            for i, detection in enumerate(detections):
                for j, (track_id, track) in enumerate(self.tracks.items()):
                    iou_matrix[i, j] = self._calculate_iou(detection.bbox, track.bbox)
            
            # Find best matches using greedy assignment
            matches = []
            unmatched_detections = []
            unmatched_tracks = []
            
            # Sort IoU matches by value (highest first)
            matched_indices = np.array(np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape)).T
            
            # Keep track of which detections and tracks have been matched
            used_detections = set()
            used_tracks = set()
            
            # Greedily assign detections to tracks based on IoU
            for d, t in matched_indices:
                # If we've already matched this detection or track, skip
                if d in used_detections or t in used_tracks:
                    continue
                
                # If the IoU is high enough, it's a match
                if iou_matrix[d, t] >= self.min_iou:
                    matches.append((d, t))
                    used_detections.add(d)
                    used_tracks.add(t)
            
            # Find unmatched detections and tracks
            unmatched_detections = [d for d in detection_indices if d not in used_detections]
            unmatched_tracks = [t for t in track_indices if t not in used_tracks]
            
            # Update matched tracks
            track_ids = list(self.tracks.keys())
            for d, t in matches:
                self.tracks[track_ids[t]].update(detections[d])
            
            # Mark unmatched tracks as disappeared
            for t in unmatched_tracks:
                self.tracks[track_ids[t]].last_seen += 1
                
                # Remove tracks that have disappeared for too long
                if self.tracks[track_ids[t]].last_seen > self.max_disappeared:
                    del self.tracks[track_ids[t]]
            
            # Create new tracks for unmatched detections
            for d in unmatched_detections:
                self.tracks[self.next_track_id] = TrackedObject(detections[d], self.next_track_id)
                self.next_track_id += 1
        
        # Return the list of tracked objects
        return list(self.tracks.values())
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bounding box as (x1, y1, x2, y2)
            bbox2: Second bounding box as (x1, y1, x2, y2)
            
        Returns:
            IoU score (0-1)
        """
        # Calculate intersection area
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
