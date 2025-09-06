import cv2
import numpy as np
import logging
from datetime import datetime

import config

logger = logging.getLogger(__name__)

class Risk:
    """Represents a risk assessment for a tracked object."""
    
    def __init__(self, tracked_object, risk_type, risk_score, metrics=None):
        """
        Initialize a risk assessment.
        
        Args:
            tracked_object: TrackedObject that poses the risk
            risk_type: Type of risk (e.g., 'collision', 'tailgating', 'lane_departure', 'lane_change')
            risk_score: Numeric risk score (0-1, where 1 is highest risk)
            metrics: Dictionary of additional metrics (e.g., TTC, THW)
        """
        self.tracked_object = tracked_object
        self.risk_type = risk_type
        self.risk_score = risk_score
        self.metrics = metrics or {}
        self.timestamp = datetime.now()
    
    @property
    def is_critical(self):
        """Whether this risk is critical (score >= 0.8)."""
        return self.risk_score >= 0.8
    
    @property
    def is_warning(self):
        """Whether this risk is a warning (0.5 <= score < 0.8)."""
        return 0.5 <= self.risk_score < 0.8

class Alert:
    """Represents an alert generated from a risk assessment."""
    
    def __init__(self, risk, message, explanation=None):
        """
        Initialize an alert.
        
        Args:
            risk: Risk object that triggered the alert
            message: Short alert message
            explanation: Longer explanation of why the alert was triggered
        """
        self.risk = risk
        self.message = message
        self.explanation = explanation
        self.timestamp = datetime.now()
    
    @property
    def is_critical(self):
        """Whether this alert is critical."""
        return self.risk.is_critical
    
    @property
    def is_warning(self):
        """Whether this alert is a warning."""
        return self.risk.is_warning

class RiskAnalyzer:
    """
    Analyzes risks based on object detections, tracking, and lane information.
    """
    
    def __init__(self):
        """Initialize the risk analyzer."""
        logger.info("Initializing risk analyzer")
        
        # Load thresholds from config
        self.ttc_threshold_critical = config.TTC_THRESHOLD_CRITICAL
        self.ttc_threshold_warning = config.TTC_THRESHOLD_WARNING
        self.following_distance_critical = config.FOLLOWING_DISTANCE_CRITICAL
        self.following_distance_warning = config.FOLLOWING_DISTANCE_WARNING
        self.lane_departure_threshold = config.LANE_DEPARTURE_THRESHOLD
        
        # Risk filtering parameters
        self.risk_cooldown = getattr(config, 'RISK_COOLDOWN_SECONDS', 2.0)  # Reduced from 3.0 to 2.0 for faster response
        self.risk_persistence_threshold = getattr(config, 'RISK_PERSISTENCE_THRESHOLD', 2)  # Reduced to 2 for faster response
        self.risk_velocity_threshold = getattr(config, 'RISK_VELOCITY_THRESHOLD', 1.2)  # Balanced for accuracy
        self.risk_score_threshold = getattr(config, 'RISK_SCORE_THRESHOLD', 0.7)  # Minimum risk score to consider
        
        # Lane change detection parameters
        self.lane_change_horizontal_velocity_threshold = getattr(config, 'LANE_CHANGE_HORIZONTAL_VELOCITY_THRESHOLD', 2.5)
        self.lane_change_detection_time = getattr(config, 'LANE_CHANGE_DETECTION_TIME', 0.8)
        self.lane_change_risk_threshold = getattr(config, 'LANE_CHANGE_RISK_THRESHOLD', 0.65)
        self.lane_change_persistence_threshold = getattr(config, 'LANE_CHANGE_PERSISTENCE_THRESHOLD', 2)
        
        # Pothole detection parameters
        self.pothole_detection_confidence = getattr(config, 'POTHOLE_DETECTION_CONFIDENCE', 0.60)
        self.pothole_risk_distance = getattr(config, 'POTHOLE_RISK_DISTANCE', 0.35)
        self.pothole_risk_threshold = getattr(config, 'POTHOLE_RISK_THRESHOLD', 0.6)
        self.pothole_persistence_threshold = getattr(config, 'POTHOLE_PERSISTENCE_THRESHOLD', 2)
        
        # State tracking with history for better accuracy
        self.last_alert_time = {}  # track_id -> {risk_type -> timestamp}
        self.risk_persistence_count = {}  # track_id -> {risk_type -> count}
        self.previous_risks = {}  # track_id -> previous risk score (for smoothing)
        self.frame_count = 0
        
        # Lane change tracking
        self.lane_change_history = {}  # track_id -> list of (frame_count, x_position, vx) tuples
        self.detected_lane_changes = set()  # track_ids with detected lane changes
        
        # Pothole tracking
        self.pothole_history = {}  # pothole_id -> list of (frame_count, position, size) tuples
        self.detected_potholes = set()  # pothole_ids of confirmed potholes
        
        # Lane detector reference (may be set externally)
        self.lane_detector = None
    
    def analyze(self, tracked_objects, lanes, road_objects=None):
        """
        Analyze risks based on tracked objects, lanes, and road hazards.
        
        Args:
            tracked_objects: List of TrackedObject objects
            lanes: List of Lane objects
            road_objects: Dictionary of detected road objects (potholes, debris, etc.)
            
        Returns:
            Tuple of (risks, alerts) where risks is a list of Risk objects and
            alerts is a list of Alert objects
        """
        self.frame_count += 1
        risks = []
        
        # Update track persistence - remove objects that are no longer being tracked
        tracked_ids = {obj.track_id for obj in tracked_objects}
        self._prune_tracking_state(tracked_ids)
        
        # Analyze collision risks
        collision_risks = self._analyze_collision_risks(tracked_objects)
        risks.extend(collision_risks)
        
        # Analyze lane change risks
        lane_change_risks = self._analyze_lane_change_risks(tracked_objects, lanes)
        risks.extend(lane_change_risks)
        
        # Analyze pothole risks if road objects are provided
        if road_objects and 'potholes' in road_objects:
            pothole_risks = self._analyze_pothole_risks(road_objects['potholes'])
            risks.extend(pothole_risks)
        
        # Analyze lane departure risks - currently disabled
        # if lanes:
        #     lane_risks = self._analyze_lane_risks(tracked_objects, lanes)
        #     risks.extend(lane_risks)
        
        # Apply persistence filtering
        filtered_risks = self._filter_risks_by_persistence(risks)
        
        # Generate alerts from filtered risks
        alerts = self._generate_alerts(filtered_risks)
        
        # Return only filtered risks to ensure only risky objects are displayed
        return filtered_risks, alerts
    
    def _analyze_collision_risks(self, tracked_objects):
        """
        Analyze collision risks between tracked objects.
        
        Args:
            tracked_objects: List of TrackedObject objects
            
        Returns:
            List of Risk objects
        """
        risks = []
        
        # Focus on our main camera vehicle (assumed to be our own car or a vehicle in front)
        # In a dashcam scenario, we're primarily concerned with forward collisions
        for obj1 in tracked_objects:
            # Only consider collision risks for cars and motorcycles
            if obj1.class_name not in ['car', 'truck', 'bus', 'motorcycle']:
                continue
            
            # Skip objects with insufficient velocity - reduces false positives
            velocity_magnitude = (obj1.velocity[0]**2 + obj1.velocity[1]**2)**0.5
            if velocity_magnitude < self.risk_velocity_threshold:
                continue
            
            # Check forward collision risk
            for obj2 in tracked_objects:
                if obj1 == obj2:
                    continue
                
                # Only consider potential collisions with vehicles, people, and bicycles
                if obj2.class_name not in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']:
                    continue
                
                # Calculate distance between objects
                distance = self._calculate_distance(obj1, obj2)
                
                # Skip if objects are too far apart - reduced threshold for more focused risk detection
                if distance > 100:  # Further reduced from 120 to 100 pixels for more focused risk detection
                    continue
                
                # Skip objects that are likely in different lanes
                if self._are_in_different_lanes(obj1, obj2):
                    continue
                
                # Calculate time-to-collision (TTC)
                ttc = self._calculate_ttc(obj1, obj2)
                
                # Skip if no collision risk
                if ttc is None or ttc > self.ttc_threshold_warning:
                    continue
                
                # Calculate risk score based on both TTC and distance - more comprehensive approach
                if ttc <= self.ttc_threshold_critical:
                    # Critical zone: Higher risk score that increases rapidly as TTC decreases
                    ttc_factor = 0.7 + (1.0 - (ttc / self.ttc_threshold_critical)) * 0.3
                    
                    # Add distance factor - closer objects are higher risk
                    distance_factor = 1.0 - min(1.0, distance / 70.0)  # Reduced from 80 to 70 pixels for normalized distance
                    
                    # Combined score with emphasis on TTC for imminent collisions
                    risk_score = 0.7 * ttc_factor + 0.3 * distance_factor
                else:
                    # Warning zone: Moderate risk score
                    ttc_normalized = (self.ttc_threshold_warning - ttc) / (self.ttc_threshold_warning - self.ttc_threshold_critical)
                    ttc_factor = 0.5 + ttc_normalized * 0.3  # Increased base score from 0.3 to 0.5
                    
                    # Add distance factor - closer objects are higher risk
                    distance_factor = 1.0 - min(1.0, distance / 100.0)  # Reduced from 120 to 100 pixels
                    
                    # Combined score with balanced weight on TTC and distance
                    risk_score = 0.6 * ttc_factor + 0.4 * distance_factor
                
                risk_score = max(0.0, min(1.0, risk_score))  # Clamp to [0, 1]
                
                # Enhanced metrics for better alerts
                metrics = {
                    'ttc': ttc,
                    'target_class': obj2.class_name,
                    'distance': distance,
                    'relative_velocity': ((obj2.velocity[0] - obj1.velocity[0])**2 + 
                                          (obj2.velocity[1] - obj1.velocity[1])**2)**0.5
                }
                
                risks.append(Risk(
                    tracked_object=obj1,
                    risk_type='collision',
                    risk_score=risk_score,
                    metrics=metrics
                ))
        
        return risks
        
    def _calculate_distance(self, obj1, obj2):
        """Calculate Euclidean distance between two objects."""
        x1, y1 = obj1.center
        x2, y2 = obj2.center
        return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    
    def _are_in_different_lanes(self, obj1, obj2):
        """
        Determine if two objects are likely in different lanes.
        Enhanced with predictive elements and better perspective handling
        for improved real-time accuracy.
        
        Args:
            obj1: First TrackedObject
            obj2: Second TrackedObject
            
        Returns:
            True if objects are likely in different lanes, False otherwise
        """
        # Get positions and velocities
        x1, y1 = obj1.center
        x2, y2 = obj2.center
        vx1, vy1 = getattr(obj1, 'velocity', (0, 0))
        vx2, vy2 = getattr(obj2, 'velocity', (0, 0))
        
        # Get object dimensions
        width1 = obj1.bbox[2] - obj1.bbox[0]
        width2 = obj2.bbox[2] - obj2.bbox[0]
        height1 = obj1.bbox[3] - obj1.bbox[1]
        height2 = obj2.bbox[3] - obj2.bbox[1]
        
        # Look ahead 0.5s for predictive assessment (better for real-time)
        future_x1 = x1 + vx1 * 15  # 15 frames ≈ 0.5s at 30fps
        future_x2 = x2 + vx2 * 15
        
        # Use both current and predicted positions
        x_diff_current = abs(x2 - x1)
        x_diff_future = abs(future_x2 - future_x1)
        
        # Take the minimum difference (more conservative approach)
        x_diff = min(x_diff_current, x_diff_future)
        
        # Calculate vertical difference for perspective consideration
        y_diff = abs(y2 - y1)
        
        # Lane threshold adaptation based on multiple factors:
        
        # 1. Frame position (perspective)
        frame_height = 720  # Standard frame height
        frame_width = 1280  # Standard frame width
        
        # Y-position indicates distance (higher y = closer to camera)
        normalized_y_pos = (y1 + y2) / (2 * frame_height)  # 0 to 1 (bottom)
        
        # 2. Object size factor (smaller objects need smaller thresholds)
        avg_width = (width1 + width2) / 2
        avg_height = (height1 + height2) / 2
        size_factor = min(1.0, avg_width / (frame_width * 0.15))  # Normalize to typical car width
        
        # 3. Class-specific adjustments
        is_person = obj1.class_name == 'person' or obj2.class_name == 'person'
        is_bicycle = obj1.class_name == 'bicycle' or obj2.class_name == 'bicycle'
        is_motorcycle = obj1.class_name == 'motorcycle' or obj2.class_name == 'motorcycle'
        
        # Calculate core lane threshold
        # Base threshold varies based on object vertical position
        if normalized_y_pos > 0.8:  # Very close objects
            base_threshold = 1.05  # Stricter threshold for close objects
        elif normalized_y_pos > 0.6:  # Medium distance
            base_threshold = 1.2
        else:  # Far objects
            base_threshold = 1.5  # More lenient for distant objects
        
        # Adjust for object size
        size_adjustment = 0.7 + size_factor * 0.6  # 0.7 to 1.3
        
        # Class-specific adjustments
        class_factor = 1.0
        if is_person:
            # Persons need special consideration - they're narrow but important
            class_factor = 1.5  # More conservative (treat as same lane more often)
        elif is_bicycle or is_motorcycle:
            # Two-wheelers are narrow but can be in same lane
            class_factor = 1.3
        
        # Velocity-based adjustment - fast horizontal movement indicates lane change
        rel_vx = abs(vx1 - vx2)
        is_changing_lanes = rel_vx > 3.0  # Significant horizontal movement
        velocity_factor = 0.9 if is_changing_lanes else 1.0  # Reduce threshold during lane changes
        
        # Combine all factors
        lane_threshold = base_threshold * size_adjustment * class_factor * velocity_factor
        
        # Special case for narrow objects at similar heights
        if (is_person or is_bicycle) and y_diff < height1 * 0.5:
            # If vertically aligned and one is a narrow object, use stricter threshold
            lane_threshold *= 1.3
        
        # Now check if the x difference exceeds the threshold
        in_different_lanes = x_diff > lane_threshold * avg_width
        
        # If they're far apart vertically, they're more likely in different lanes
        # unless they're directly in line with each other
        vertical_lane_factor = 1.5  # How much vertical separation suggests different lanes
        if y_diff > avg_height * vertical_lane_factor and abs(x1 - x2) > avg_width * 0.5:
            in_different_lanes = True
        
        return in_different_lanes
    
    def _analyze_lane_change_risks(self, tracked_objects, lanes):
        """
        Detect and analyze lane change risks from other vehicles.
        
        Args:
            tracked_objects: List of TrackedObject objects
            lanes: List of Lane objects (may be used for additional context)
            
        Returns:
            List of Risk objects for lane change risks
        """
        risks = []
        frame_height = 720  # Standard frame height
        frame_width = 1280  # Standard frame width
        fps = 30  # Assuming 30 FPS
        
        # Detection window in frames
        detection_window = int(self.lane_change_detection_time * fps)
        
        # Process each tracked object
        for obj in tracked_objects:
            # Only consider vehicles (cars, trucks, buses, motorcycles)
            if obj.class_name not in ['car', 'truck', 'bus', 'motorcycle']:
                continue
                
            track_id = obj.track_id
            x, y = obj.center
            vx, vy = obj.velocity
            width = obj.bbox[2] - obj.bbox[0]
            height = obj.bbox[3] - obj.bbox[1]
            
            # Skip if the object is too high in the frame (far away)
            if y < frame_height * 0.4:
                continue
                
            # Skip if object is too small
            if width < 30 or height < 30:
                continue
            
            # Initialize history for this object if needed
            if track_id not in self.lane_change_history:
                self.lane_change_history[track_id] = []
                
            # Add current position and velocity to history
            self.lane_change_history[track_id].append((self.frame_count, x, vx))
            
            # Keep only recent history within the detection window
            history = [entry for entry in self.lane_change_history[track_id] 
                       if self.frame_count - entry[0] <= detection_window]
            self.lane_change_history[track_id] = history
            
            # Need at least 3 points to detect a lane change
            if len(history) < 3:
                continue
                
            # Calculate horizontal velocity stats
            vx_values = [entry[2] for entry in history]
            avg_vx = sum(vx_values) / len(vx_values)
            max_vx = max(abs(v) for v in vx_values)
            
            # If max horizontal velocity doesn't meet threshold, skip
            if max_vx < self.lane_change_horizontal_velocity_threshold:
                continue
                
            # Calculate trajectory direction changes (lane change has sign change in velocity)
            has_direction_change = False
            positive_vx = False
            negative_vx = False
            
            # Look for consistent direction first, then change
            for vx in vx_values:
                if vx > 1.0:  # Positive threshold
                    positive_vx = True
                elif vx < -1.0:  # Negative threshold
                    negative_vx = True
            
            # If we have both positive and negative velocities, it's a direction change
            has_direction_change = positive_vx and negative_vx
            
            # Calculate x position change
            start_x = history[0][1]
            end_x = history[-1][1]
            x_change = abs(end_x - start_x)
            
            # A lane change typically involves significant horizontal movement
            # relative to the vehicle width
            significant_movement = x_change > width * 0.7
            
            # Calculate if the object is close to ego vehicle
            # For simplicity, consider lower half of frame and center region
            is_close = y > frame_height * 0.5
            is_in_center_region = x > frame_width * 0.3 and x < frame_width * 0.7
            
            # For vehicles closer to the edges, they need to be moving inward to be a concern
            is_moving_inward = (x < frame_width * 0.5 and avg_vx > 1.0) or \
                               (x > frame_width * 0.5 and avg_vx < -1.0)
            
            # Detection criteria for different scenarios
            lane_change_detected = False
            risk_score = 0.0
            
            # Scenario 1: Significant movement with high velocity
            if significant_movement and max_vx > self.lane_change_horizontal_velocity_threshold * 1.2:
                lane_change_detected = True
                risk_score = 0.7
                
            # Scenario 2: Direction change with reasonable movement
            elif has_direction_change and x_change > width * 0.5:
                lane_change_detected = True
                risk_score = 0.65
                
            # Scenario 3: High velocity movement in critical region
            elif max_vx > self.lane_change_horizontal_velocity_threshold and is_close and is_in_center_region:
                lane_change_detected = True
                risk_score = 0.75
                
            # Scenario 4: Edge case moving inward
            elif is_moving_inward and is_close and max_vx > self.lane_change_horizontal_velocity_threshold:
                lane_change_detected = True
                risk_score = 0.7
            
            # If lane change detected, create a risk
            if lane_change_detected:
                # Adjust risk score based on proximity
                if is_close:
                    # Closer objects are higher risk
                    proximity_factor = 1.0 + ((y / frame_height) - 0.5) * 0.6  # 1.0 to 1.3
                    risk_score *= proximity_factor
                
                # Adjust risk score based on velocity magnitude
                velocity_factor = min(1.2, max(1.0, max_vx / self.lane_change_horizontal_velocity_threshold))
                risk_score *= velocity_factor
                
                # Cap risk score
                risk_score = min(0.95, risk_score)
                
                # Add the object to detected lane changes set
                self.detected_lane_changes.add(track_id)
                
                # Create metrics
                metrics = {
                    'direction': 'right' if avg_vx > 0 else 'left',
                    'velocity': max_vx,
                    'x_change': x_change,
                    'lane_change_confidence': min(1.0, (max_vx / self.lane_change_horizontal_velocity_threshold) * 0.8)
                }
                
                # Create risk object
                risks.append(Risk(
                    tracked_object=obj,
                    risk_type='lane_change',
                    risk_score=risk_score,
                    metrics=metrics
                ))
        
        return risks
    
    def _calculate_ttc(self, obj1, obj2):
        """
        Calculate time-to-collision between two objects.
        Enhanced for real-time accuracy with more predictive elements
        and better handling of dashcam-specific scenarios.
        
        Args:
            obj1: First TrackedObject
            obj2: Second TrackedObject
            
        Returns:
            Time-to-collision in seconds, or None if not on collision course
        """
        # Get current positions
        x1, y1 = obj1.center
        x2, y2 = obj2.center
        
        # Get velocities
        vx1, vy1 = obj1.velocity
        vx2, vy2 = obj2.velocity
        
        # Get acceleration if available (for more accurate prediction)
        ax1, ay1 = getattr(obj1, 'acceleration', (0, 0))
        ax2, ay2 = getattr(obj2, 'acceleration', (0, 0))
        
        # Calculate relative velocity
        rel_vx = vx2 - vx1
        rel_vy = vy2 - vy1
        
        # Calculate relative acceleration
        rel_ax = ax2 - ax1
        rel_ay = ay2 - ay1
        
        # Calculate relative distance
        rel_x = x2 - x1
        rel_y = y2 - y1
        
        # If relative velocity is very small, check acceleration
        if abs(rel_vx) < 0.2 and abs(rel_vy) < 0.2:
            # If acceleration is significant, objects might collide in future
            if abs(rel_ax) > 0.5 or abs(rel_ay) > 0.5:
                # Simple prediction with acceleration (more aggressive detection)
                future_rel_vx = rel_vx + rel_ax * 0.5  # predict 0.5s ahead
                future_rel_vy = rel_vy + rel_ay * 0.5
                
                # If future velocity suggests collision, use that
                if abs(future_rel_vx) > 0.5 or abs(future_rel_vy) > 0.5:
                    rel_vx = future_rel_vx
                    rel_vy = future_rel_vy
                else:
                    return None
            else:
                return None
            
        # Calculate distance and velocity magnitudes
        rel_distance = (rel_x**2 + rel_y**2)**0.5
        rel_velocity = (rel_vx**2 + rel_vy**2)**0.5
        
        # Check if objects are moving toward each other
        # Calculate dot product between relative position and relative velocity
        dot_product = rel_x * rel_vx + rel_y * rel_vy
        
        # Negative dot product means objects are approaching each other
        if dot_product >= 0:  # Not approaching
            # Check if acceleration might change this
            future_dot = (rel_x + rel_vx*0.5) * (rel_vx + rel_ax*0.5) + \
                        (rel_y + rel_vy*0.5) * (rel_vy + rel_ay*0.5)
            if future_dot >= 0:  # Still not approaching in near future
                return None
        
        # Calculate angle between objects (in degrees)
        angle = np.arctan2(rel_y, rel_x) * 180 / np.pi
        
        # Define front collision zone - slightly expanded for better coverage
        is_front_collision = abs(angle) < 40  # Increased from 35 to 40 degrees
        
        # For dashcam view, calculate TTC considering perspective
        frame_height = 720  # Standard frame height
        frame_width = 1280   # Standard frame width
        
        # Calculate bounding box overlap percentage for collision likelihood
        # Get bounding boxes
        x1_min, y1_min, x1_max, y1_max = obj1.bbox
        x2_min, y2_min, x2_max, y2_max = obj2.bbox
        
        # Calculate overlap areas
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        # Calculate overlap area and total area
        overlap_area = x_overlap * y_overlap
        total_area = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - overlap_area
        
        # Calculate overlap percentage
        overlap_percentage = 0
        if total_area > 0:
            overlap_percentage = overlap_area / total_area
        
        # Adjust collision prediction based on collision type
        if is_front_collision:
            # For front collisions, use time to reach same y position
            if abs(rel_vy) < 0.2:  # Avoid division by zero
                # Check if acceleration would make this significant
                if abs(rel_ay) > 0.3:
                    # Calculate TTC with acceleration
                    # Solve quadratic: s = ut + 0.5at^2 for t
                    # Where s is distance, u is initial velocity, a is acceleration
                    # Quadratic formula: t = (-u ± √(u² - 2as))/a
                    discriminant = rel_vy**2 - 2*rel_ay*(-rel_y)
                    if discriminant >= 0 and rel_ay != 0:
                        t1 = (-rel_vy + discriminant**0.5) / rel_ay
                        t2 = (-rel_vy - discriminant**0.5) / rel_ay
                        # Take the smallest positive value
                        if t1 > 0 and t2 > 0:
                            ttc = min(t1, t2)
                        elif t1 > 0:
                            ttc = t1
                        elif t2 > 0:
                            ttc = t2
                        else:
                            return None
                    else:
                        return None
                else:
                    return None
            else:
                # Basic TTC calculation
                ttc_y = -rel_y / rel_vy  # Negative because we want approaching time
                
                if ttc_y <= 0:
                    return None
                
                # Adjust with acceleration if significant
                if abs(rel_ay) > 0.3:
                    # Refine TTC estimate with acceleration: s = ut + 0.5at²
                    # Starting with ttc_y as initial estimate
                    # Iteratively improve the estimate
                    for _ in range(2):  # Two iterations usually sufficient
                        distance_with_accel = rel_vy * ttc_y + 0.5 * rel_ay * ttc_y**2
                        correction = (rel_y + distance_with_accel) / (rel_vy + rel_ay * ttc_y)
                        ttc_y = ttc_y - correction
                
                # Scale TTC based on vertical position (perspective adjustment)
                position_factor = 1.0 + (1.0 - (y2 / frame_height)) * 0.4  # 1.0 to 1.4 (less extreme)
                ttc = ttc_y * position_factor
                
                # Adjust based on overlap - higher overlap means more imminent collision
                if overlap_percentage > 0.05:
                    ttc = ttc * (1.0 - overlap_percentage * 0.5)  # Reduce TTC by up to 50% for significant overlap
        else:
            # For side collisions, use closest approach time with acceleration
            
            # Calculate time to closest approach
            if abs(rel_vx)**2 + abs(rel_vy)**2 < 0.01:
                return None
                
            t_closest = -(rel_x * rel_vx + rel_y * rel_vy) / (rel_vx**2 + rel_vy**2)
            
            # Refine with acceleration if significant
            if abs(rel_ax) > 0.3 or abs(rel_ay) > 0.3:
                # Iterative refinement
                for _ in range(2):
                    rel_vx_t = rel_vx + rel_ax * t_closest
                    rel_vy_t = rel_vy + rel_ay * t_closest
                    t_closest = -(rel_x * rel_vx_t + rel_y * rel_vy_t) / (rel_vx_t**2 + rel_vy_t**2 + 0.001)
            
            if t_closest <= 0:  # Already at closest approach or moving away
                return None
                
            # Calculate distance at closest approach
            closest_distance_x = rel_x + rel_vx * t_closest + 0.5 * rel_ax * t_closest**2
            closest_distance_y = rel_y + rel_vy * t_closest + 0.5 * rel_ay * t_closest**2
            closest_distance = (closest_distance_x**2 + closest_distance_y**2)**0.5
            
            # Check if closest approach distance is small enough to be a collision
            # This threshold should be based on object sizes
            obj1_width = obj1.bbox[2] - obj1.bbox[0]
            obj2_width = obj2.bbox[2] - obj2.bbox[0]
            collision_threshold = (obj1_width + obj2_width) / 3  # More strict threshold
            
            if closest_distance > collision_threshold:
                # If there's significant overlap, still consider it a risk
                if overlap_percentage > 0.1:
                    ttc = t_closest
                else:
                    return None
            else:
                ttc = t_closest
        
        # Convert from frames to seconds (assuming 30 FPS)
        ttc_seconds = ttc / 30.0
        
        # Sanity check - TTC should be reasonable for dashcam scenario
        if ttc_seconds > 3.0 or ttc_seconds <= 0:
            return None
            
        return ttc_seconds
    
    def _calculate_lane_position(self, obj, left_lane, right_lane):
        """
        Calculate normalized lane position (-1.0 to 1.0).
        -1.0 means completely in the left lane, 1.0 means completely in the right lane,
        0.0 means centered between lanes.
        
        Args:
            obj: TrackedObject
            left_lane: Left Lane object
            right_lane: Right Lane object
            
        Returns:
            Normalized lane position, or None if cannot be calculated
        """
        # Get object position
        obj_x, obj_y = obj.center
        
        # Get lane positions at the same y-coordinate
        try:
            left_x = left_lane.coeffs[0] * obj_y**2 + left_lane.coeffs[1] * obj_y + left_lane.coeffs[2]
            right_x = right_lane.coeffs[0] * obj_y**2 + right_lane.coeffs[1] * obj_y + right_lane.coeffs[2]
        except:
            return None
        
        # Calculate lane width and center
        lane_width = right_x - left_x
        lane_center = (left_x + right_x) / 2
        
        if lane_width <= 0:
            return None
        
        # Calculate normalized position (-1.0 to 1.0)
        normalized_position = 2.0 * (obj_x - lane_center) / lane_width
        
        return normalized_position
        
    def _estimate_lane_boundaries(self, frame_width):
        """
        Estimate lane boundaries when explicit lane detection is not available.
        This provides approximate lane positions for lane change detection.
        
        Args:
            frame_width: Width of the video frame
            
        Returns:
            Dictionary with estimated lane boundaries
        """
        # Estimate lane positions based on standard road configurations
        # For a typical dashcam view, we can estimate 3-4 lanes
        
        # Calculate lane widths - assume 3-4 lanes visible
        num_lanes = 3
        lane_width = frame_width / (num_lanes + 0.5)  # Allow for shoulders
        
        # Create lane boundaries
        lane_boundaries = {}
        
        # Left lane boundary (approximately 1/6 of frame from left)
        lane_boundaries['left'] = frame_width * 0.16
        
        # Center lane boundaries
        lane_boundaries['center_left'] = frame_width * 0.33
        lane_boundaries['center'] = frame_width * 0.5
        lane_boundaries['center_right'] = frame_width * 0.66
        
        # Right lane boundary (approximately 1/6 of frame from right)
        lane_boundaries['right'] = frame_width * 0.84
        
        return lane_boundaries
    
    def _generate_alerts(self, risks):
        """
        Generate alerts from risks.
        
        Args:
            risks: List of Risk objects
            
        Returns:
            List of Alert objects
        """
        alerts = []
        
        for risk in risks:
            if risk.risk_type == 'collision':
                alerts.append(self._generate_collision_alert(risk))
            elif risk.risk_type == 'lane_departure':
                alerts.append(self._generate_lane_departure_alert(risk))
            elif risk.risk_type == 'lane_change':
                alerts.append(self._generate_lane_change_alert(risk))
            elif risk.risk_type == 'pothole':
                alerts.append(self._generate_pothole_alert(risk))
        
        return alerts
    
    def _generate_collision_alert(self, risk):
        """
        Generate a collision alert with clear, concise messaging for drivers.
        
        Args:
            risk: Risk object
            
        Returns:
            Alert object
        """
        target_class = risk.metrics.get('target_class', 'object')
        ttc = risk.metrics.get('ttc', 0)
        distance = risk.metrics.get('distance', 0)
        
        # Use more direct, actionable language for drivers
        if risk.is_critical:
            # Critical alerts need to be very clear and direct
            if target_class == 'person':
                message = f"BRAKE NOW! Person ahead"
            else:
                message = f"BRAKE NOW! Vehicle ahead"
                
            # More specific explanation with distance context
            if distance > 0:
                # Convert pixel distance to approximate real-world context
                # This is a simplified estimation - in a real system, you would use proper distance calculation
                distance_estimate = "very close" if distance < 50 else "close" if distance < 100 else "approaching"
                explanation = f"Collision imminent! {target_class.capitalize()} {distance_estimate}. Time to impact: {ttc:.1f}s"
            else:
                explanation = f"Collision imminent! {target_class.capitalize()} ahead. Time to impact: {ttc:.1f}s"
        else:
            # Warning alerts should be noticeable but less alarming
            if target_class == 'person':
                message = f"Caution: Person ahead"
            else:
                message = f"Caution: Vehicle ahead"
            
            # Convert pixel distance to approximate real-world context
            distance_estimate = "approaching" if distance < 100 else "ahead"
            explanation = f"{target_class.capitalize()} {distance_estimate}. Prepare to brake. Time to potential impact: {ttc:.1f}s"
        
        return Alert(risk, message, explanation)
    
    def _generate_lane_change_alert(self, risk):
        """
        Generate a lane change alert with contextual information.
        
        Args:
            risk: Risk object for a lane change
            
        Returns:
            Alert object with appropriate message
        """
        direction = risk.metrics.get('direction', 'unknown')
        velocity = risk.metrics.get('velocity', 0)
        confidence = risk.metrics.get('lane_change_confidence', 0.5)
        
        # Get object class name for better context
        vehicle_type = risk.tracked_object.class_name
        vehicle_display = {
            'car': 'Car',
            'truck': 'Truck',
            'bus': 'Bus',
            'motorcycle': 'Motorcycle'
        }.get(vehicle_type, 'Vehicle')
        
        # Get horizontal position for better context
        x, y = risk.tracked_object.center
        frame_width = 1280  # Standard frame width
        position_context = "nearby"
        
        if x < frame_width * 0.4:
            position_context = "on left"
        elif x > frame_width * 0.6:
            position_context = "on right"
        
        # Create contextual message based on risk level
        if risk.is_critical:
            message = f"CAUTION: {vehicle_display} changing lanes to {direction}"
            
            # For critical risks, give specific warnings based on context
            if position_context == "on left" and direction == "right":
                explanation = f"{vehicle_display} {position_context} rapidly moving into your lane. Prepare to brake."
            elif position_context == "on right" and direction == "left":
                explanation = f"{vehicle_display} {position_context} rapidly moving into your lane. Prepare to brake."
            else:
                explanation = f"{vehicle_display} {position_context} making sudden lane change to the {direction}. Be cautious."
        else:
            message = f"{vehicle_display} changing to {direction} lane"
            
            # For warnings, give more general information
            explanation = f"{vehicle_display} {position_context} appears to be changing to the {direction} lane. Monitor their movement."
        
        return Alert(risk, message, explanation)
        
    def _generate_pothole_alert(self, risk):
        """
        Generate a pothole alert with detailed information about the pothole type and risk.
        
        Args:
            risk: Risk object for a pothole
            
        Returns:
            Alert object with appropriate message
        """
        # Extract pothole metrics
        pothole_type = risk.metrics.get('pothole_type', 'unknown')
        normalized_size = risk.metrics.get('normalized_size', 0.0)
        distance_factor = risk.metrics.get('distance_factor', 0.0)
        nearby_potholes = risk.metrics.get('nearby_potholes', 0)
        
        # Extract pothole base type (before the shape modifier)
        base_type = pothole_type.split('_')[0] if '_' in pothole_type else pothole_type
        shape_type = pothole_type.split('_')[1] if '_' in pothole_type else ''
        
        # Determine urgency based on risk score
        if risk.is_critical:
            if base_type in ['large', 'severe']:
                message = f"CAUTION: Large pothole ahead"
            elif shape_type == 'elongated':
                message = f"CAUTION: Road crack ahead"
            else:
                message = f"CAUTION: Pothole ahead"
                
            # Add details about the pothole based on type
            if nearby_potholes > 1:
                explanation = f"Multiple potholes detected. Significant road damage ahead."
            elif base_type == 'severe':
                explanation = f"Deep pothole detected in your path. Consider changing lanes if safe."
            elif base_type == 'large':
                explanation = f"Large pothole ahead. Prepare to slow down or steer around it."
            elif shape_type == 'elongated':
                explanation = f"Long crack in the road surface ahead. Approach with caution."
            else:
                explanation = f"Significant pothole detected ahead. Reduce speed."
        else:
            # Warning level alerts
            if shape_type == 'elongated':
                message = f"Road crack detected"
            elif nearby_potholes > 1:
                message = f"Multiple potholes ahead"
            else:
                message = f"Pothole detected"
                
            # Add context based on type
            if nearby_potholes > 1:
                explanation = f"Area of damaged road surface ahead. Drive with caution."
            elif base_type == 'medium' or base_type == 'large':
                explanation = f"Moderate pothole detected ahead. Be prepared to navigate around it."
            elif shape_type == 'elongated':
                explanation = f"Linear crack in the road surface. Monitor and avoid if possible."
            elif shape_type == 'circular':
                explanation = f"Circular pothole ahead. Be aware."
            else:
                explanation = f"Minor road damage detected. Proceed with caution."
        
        return Alert(risk, message, explanation)
        
    def _generate_lane_departure_alert(self, risk):
        """
        Generate a lane departure alert with clear, actionable instructions.
        
        Args:
            risk: Risk object
            
        Returns:
            Alert object
        """
        direction = risk.metrics.get('direction', 'unknown')
        severity = risk.metrics.get('severity', 'warning')
        
        # Use simple, direct language focused on the action needed
        if severity == 'critical':
            message = f"Steer {self._opposite_direction(direction)}!"
            explanation = f"Lane departure detected. Immediate correction needed to the {self._opposite_direction(direction)}."
        else:
            message = f"Drifting {direction}"
            explanation = f"Vehicle is drifting toward {direction} lane edge. Consider steering {self._opposite_direction(direction)}."
        
        return Alert(risk, message, explanation)
        
    def _opposite_direction(self, direction):
        """Get the opposite of a direction."""
        return "right" if direction == "left" else "left"
    
    def _analyze_pothole_risks(self, potholes):
        """
        Analyze risks from detected potholes.
        Enhanced to recognize all types of potholes with improved classification.
        
        Args:
            potholes: List of detected pothole objects with positions and sizes
            
        Returns:
            List of Risk objects for pothole risks
        """
        risks = []
        frame_height = 720  # Standard frame height
        frame_width = 1280  # Standard frame width
        
        # Define pothole classifications
        pothole_types = {
            'small': {'min_size_percent': 0.001, 'max_size_percent': 0.01, 'risk_multiplier': 0.8},
            'medium': {'min_size_percent': 0.01, 'max_size_percent': 0.025, 'risk_multiplier': 1.0},
            'large': {'min_size_percent': 0.025, 'max_size_percent': 0.05, 'risk_multiplier': 1.2},
            'severe': {'min_size_percent': 0.05, 'max_size_percent': 1.0, 'risk_multiplier': 1.4},
        }
        
        # Process each pothole
        for pothole in potholes:
            # Extract pothole data
            pothole_id = pothole.get('id', f"pothole_{self.frame_count}_{len(risks)}")
            bbox = pothole.get('bbox', [0, 0, 0, 0])
            confidence = pothole.get('confidence', 0.0)
            
            # Accept lower confidence to capture more pothole types (we'll filter later)
            adjusted_confidence_threshold = self.pothole_detection_confidence * 0.8
            if confidence < adjusted_confidence_threshold:
                continue
                
            # Calculate pothole position (center bottom of bounding box)
            x1, y1, x2, y2 = bbox
            position = (int((x1 + x2) / 2), int(y2))
            size = (x2 - x1) * (y2 - y1)  # Area of pothole
            
            # Calculate pothole dimensions
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / max(1, height)  # Avoid division by zero
            
            # Calculate normalized size relative to frame
            frame_area = frame_width * frame_height
            size_percent = size / frame_area
            
            # Skip if pothole is too high in the frame (very far away)
            # Use a more lenient threshold to catch more distant potholes
            if position[1] < frame_height * 0.35:  # Reduced from 0.4 to 0.35
                continue
            
            # Determine pothole type based on size
            pothole_type = 'unknown'
            risk_multiplier = 1.0
            for p_type, props in pothole_types.items():
                if props['min_size_percent'] <= size_percent < props['max_size_percent']:
                    pothole_type = p_type
                    risk_multiplier = props['risk_multiplier']
                    break
            
            # Add shape-based classification
            if aspect_ratio > 2.0:
                pothole_type += "_elongated"  # Likely a crack or long pothole
            elif aspect_ratio < 0.5:
                pothole_type += "_vertical"  # Unusual vertical shape
            elif 0.9 <= aspect_ratio <= 1.1:
                pothole_type += "_circular"  # Circular pothole
                
            # Initialize history for this pothole if needed
            if pothole_id not in self.pothole_history:
                self.pothole_history[pothole_id] = []
                
            # Add current detection to history
            self.pothole_history[pothole_id].append((self.frame_count, position, size, pothole_type))
            
            # Keep only recent history (last 30 frames)
            history = [entry for entry in self.pothole_history[pothole_id] 
                       if self.frame_count - entry[0] <= 30]
            self.pothole_history[pothole_id] = history
            
            # Calculate risk based on position, size, confidence, and pothole type
            
            # 1. Position-based risk (lower in frame = closer = higher risk)
            normalized_y = position[1] / frame_height  # 0 at top, 1 at bottom
            
            # Progressive risk based on distance with finer gradations
            if normalized_y > 0.85:  # Very close
                position_factor = 1.0
            elif normalized_y > 0.75:  # Close
                position_factor = 0.9
            elif normalized_y > 0.65:  # Medium-close
                position_factor = 0.8
            elif normalized_y > 0.55:  # Medium
                position_factor = 0.7
            elif normalized_y > 0.45:  # Medium-far
                position_factor = 0.6
            elif normalized_y > 0.35:  # Far
                position_factor = 0.5
            else:  # Very far
                position_factor = 0.4
                
            # 2. Size-based risk (larger = higher risk)
            # Normalize size relative to frame with more precise scaling
            max_expected_size = frame_width * frame_height * 0.06  # 6% of frame is very large
            normalized_size = min(1.0, size / max_expected_size)
            size_factor = 0.5 + normalized_size * 0.5  # 0.5 to 1.0
            
            # 3. Lane position risk (center of lane = higher risk)
            # Get horizontal position relative to frame
            normalized_x = position[0] / frame_width  # 0 to 1
            
            # Estimate lane boundaries
            lane_boundaries = self._estimate_lane_boundaries(frame_width)
            
            # Find closest lane center
            lane_centers = [lane_boundaries['center_left'], lane_boundaries['center'], lane_boundaries['center_right']]
            normalized_centers = [center / frame_width for center in lane_centers]
            
            # Calculate distance to nearest lane center
            min_dist_to_center = min(abs(normalized_x - center) for center in normalized_centers)
            
            # Higher risk if closer to lane center
            lane_position_factor = 1.0 - min(1.0, min_dist_to_center * 4.0)  # 0.0 to 1.0
            
            # 4. Confidence factor with sensitivity to lower confidence detections
            # Scale confidence so that even lower confidence detections can be considered
            confidence_scaling = 1.2 if confidence > self.pothole_detection_confidence else 1.0
            confidence_factor = min(1.0, confidence * confidence_scaling)
            
            # 5. Persistence factor - increase risk if pothole is detected consistently
            persistence_factor = min(1.0, len(history) / 15 * 0.3 + 0.7)  # 0.7 to 1.0
            
            # 6. Pattern recognition - check for clusters of potholes
            cluster_factor = 1.0
            nearby_potholes = 0
            
            for other_pothole in potholes:
                if other_pothole == pothole:
                    continue
                    
                other_bbox = other_pothole.get('bbox', [0, 0, 0, 0])
                other_x = (other_bbox[0] + other_bbox[2]) / 2
                other_y = (other_bbox[1] + other_bbox[3]) / 2
                
                # Calculate distance to this pothole
                distance = ((other_x - position[0])**2 + (other_y - position[1])**2)**0.5
                
                # If pothole is nearby, count it as part of a cluster
                if distance < frame_width * 0.15:  # Within 15% of frame width
                    nearby_potholes += 1
            
            # Increase risk for clustered potholes (road section with multiple defects)
            if nearby_potholes >= 2:
                cluster_factor = 1.2  # Significant cluster
            elif nearby_potholes == 1:
                cluster_factor = 1.1  # Pair of potholes
            
            # Combine factors with weights, including pothole type-specific risk multiplier
            risk_score = (
                0.35 * position_factor +      # 35% weight to position/distance
                0.20 * size_factor +          # 20% weight to size
                0.25 * lane_position_factor + # 25% weight to lane position
                0.10 * confidence_factor +    # 10% weight to detection confidence
                0.10 * persistence_factor     # 10% weight to detection persistence
            ) * risk_multiplier * cluster_factor
            
            # Skip if risk score is below threshold
            if risk_score < self.pothole_risk_threshold * 0.85:  # Slightly lower threshold to catch more types
                continue
                
            # Create PotholeObject (similar to TrackedObject interface)
            pothole_obj = type('PotholeObject', (), {
                'id': pothole_id,
                'bbox': bbox,
                'center': position,
                'class_name': 'pothole',
                'confidence': confidence,
                'track_id': pothole_id
            })
            
            # Add metrics for alerting
            metrics = {
                'size': size,
                'normalized_size': normalized_size,
                'position': position,
                'distance_factor': position_factor,
                'lane_position_factor': lane_position_factor,
                'detection_confidence': confidence,
                'pothole_type': pothole_type,
                'nearby_potholes': nearby_potholes
            }
            
            # Create risk object
            risks.append(Risk(
                tracked_object=pothole_obj,
                risk_type='pothole',
                risk_score=risk_score,
                metrics=metrics
            ))
            
            # Add to detected potholes set
            self.detected_potholes.add(pothole_id)
            
        return risks
    
    def _prune_tracking_state(self, current_track_ids):
        """
        Remove state data for objects that are no longer being tracked.
        
        Args:
            current_track_ids: Set of track IDs that are currently active
        """
        # Remove from last_alert_time
        for track_id in list(self.last_alert_time.keys()):
            if track_id not in current_track_ids:
                del self.last_alert_time[track_id]
        
        # Remove from risk_persistence_count
        for track_id in list(self.risk_persistence_count.keys()):
            if track_id not in current_track_ids:
                del self.risk_persistence_count[track_id]
                
        # Remove from previous_risks
        for track_id in list(self.previous_risks.keys()):
            if track_id not in current_track_ids:
                del self.previous_risks[track_id]
                
        # Remove from lane_change_history
        for track_id in list(self.lane_change_history.keys()):
            if track_id not in current_track_ids:
                del self.lane_change_history[track_id]
                
        # Remove from detected_lane_changes
        self.detected_lane_changes = {track_id for track_id in self.detected_lane_changes 
                                     if track_id in current_track_ids}
    
    def _filter_risks_by_persistence(self, risks):
        """
        Filter risks based on persistence with enhanced accuracy for real-time performance.
        Uses risk smoothing and predictive filtering to reduce jitter and improve responsiveness.
        
        Args:
            risks: List of Risk objects
            
        Returns:
            Filtered list of Risk objects
        """
        filtered_risks = []
        current_time = datetime.now().timestamp()
        frame_height = 720  # Standard frame height
        frame_width = 1280  # Standard frame width
        
        # Track current risks by ID for smoothing
        current_risk_ids = {}
        
        for risk in risks:
            # Initial threshold check - use the configurable threshold
            if risk.risk_score < self.risk_score_threshold:
                continue
                
            obj = risk.tracked_object
            track_id = obj.track_id
            risk_type = risk.risk_type
            
            # Store for later processing
            current_risk_ids[track_id] = risk
            
            # Apply risk score smoothing for stability
            if track_id in self.previous_risks:
                # Apply temporal smoothing (70% new, 30% old) - reduces jitter
                smoothed_score = 0.7 * risk.risk_score + 0.3 * self.previous_risks[track_id]
                risk.risk_score = smoothed_score
            
            # Apply additional filtering for collision risks
            if risk_type == 'collision':
                # Get object position and dimensions
                x, y = obj.center
                width = obj.bbox[2] - obj.bbox[0]
                height = obj.bbox[3] - obj.bbox[1]
                
                # Skip if object is too high in the frame (likely far away)
                # Adjusted threshold to focus on lower part of frame (closer objects)
                if y < frame_height * 0.45:  # More aggressive filtering (was 0.5)
                    continue
                
                # Skip if object is too far to the sides
                # Objects on the extreme sides are likely not in our path
                side_margin = 0.12  # Reduced from 0.15 to 0.12 - more precise lane focus
                if x < frame_width * side_margin or x > frame_width * (1 - side_margin):
                    continue
                
                # Skip small objects but with more precise size thresholds
                if height < 35:  # Reduced from 40 to 35 for better detection of distant risks
                    # Exception: If it's a person, use a smaller threshold (people are important)
                    if obj.class_name == 'person' and height < 25:
                        continue
                    elif obj.class_name != 'person':
                        continue
                
                # Calculate object area as percentage of frame
                obj_area = width * height
                frame_area = frame_width * frame_height
                area_percent = (obj_area / frame_area) * 100
                
                # Adjust risk based on area - larger objects (closer) have higher risk
                if obj.class_name == 'person':
                    # Persons are high priority - boost small objects
                    if area_percent < 1.0:
                        # Small person - moderate boost if in lower part of frame
                        if y > frame_height * 0.7:
                            risk.risk_score = min(1.0, risk.risk_score * 1.2)
                    else:
                        # Larger person - significant risk
                        risk.risk_score = min(1.0, risk.risk_score * 1.3)
                else:
                    # Vehicles - adjust based on size and position
                    if area_percent > 5.0:
                        # Very large vehicle - very close
                        risk.risk_score = min(1.0, risk.risk_score * 1.2)
                    elif area_percent < 0.5:
                        # Very small vehicle - likely far away
                        risk.risk_score = risk.risk_score * 0.8
                
                # Position-based adjustment with more precision
                vertical_position = y / frame_height  # 0 at top, 1 at bottom
                
                # Progressive position factor - more aggressive for lower objects
                if vertical_position > 0.8:
                    # Very close - high risk
                    position_factor = 1.3
                elif vertical_position > 0.65:
                    # Moderately close - elevated risk
                    position_factor = 1.15
                elif vertical_position > 0.5:
                    # Middle distance - normal risk
                    position_factor = 1.0
                else:
                    # Far away - reduced risk
                    position_factor = 0.8
                
                # Apply position factor with partial weighting
                risk.risk_score = min(1.0, risk.risk_score * position_factor)
                
                # TTC-based boost for imminent collisions (more aggressive for real-time)
                if 'ttc' in risk.metrics:
                    ttc = risk.metrics['ttc']
                    if ttc < self.ttc_threshold_critical * 0.7:  # Very imminent
                        risk.risk_score = min(1.0, risk.risk_score * 1.3)
                    elif ttc < self.ttc_threshold_critical:
                        risk.risk_score = min(1.0, risk.risk_score * 1.2)
                
                # Final check after all adjustments
                if risk.risk_score < self.risk_score_threshold:
                    continue
            
            # Initialize tracking dictionaries for this object if needed
            if track_id not in self.last_alert_time:
                self.last_alert_time[track_id] = {}
            if track_id not in self.risk_persistence_count:
                self.risk_persistence_count[track_id] = {}
            
            # Increment persistence count for this risk
            self.risk_persistence_count[track_id][risk_type] = self.risk_persistence_count[track_id].get(risk_type, 0) + 1
            
            # More responsive persistence threshold for high-risk objects
            effective_threshold = self.risk_persistence_threshold
            if risk.risk_score > 0.85:  # Very high risk
                effective_threshold = max(1, self.risk_persistence_threshold - 1)  # More responsive
            
            # Check if we've reached the persistence threshold
            if self.risk_persistence_count[track_id].get(risk_type, 0) >= effective_threshold:
                # Check if we're past the cooldown period
                last_alert = self.last_alert_time[track_id].get(risk_type, 0)
                cooldown_factor = 1.0  # Normal cooldown
                
                # Reduce cooldown for higher risk objects (more responsive)
                if risk.risk_score > 0.9:
                    cooldown_factor = 0.5  # Half cooldown for critical risks
                elif risk.risk_score > 0.8:
                    cooldown_factor = 0.7  # 70% cooldown for high risks
                
                if current_time - last_alert >= self.risk_cooldown * cooldown_factor:
                    # This risk has persisted and is past the cooldown period
                    filtered_risks.append(risk)
                    self.last_alert_time[track_id][risk_type] = current_time
        
        # Update previous risk scores for smoothing
        self.previous_risks = {track_id: risk.risk_score for track_id, risk in current_risk_ids.items()}
        
        # Reset persistence count for risk types not in this frame
        for track_id, risk_types in self.risk_persistence_count.items():
            current_risk_types = {risk.risk_type for risk in risks if risk.tracked_object.track_id == track_id}
            for risk_type in list(risk_types.keys()):
                if risk_type not in current_risk_types:
                    del self.risk_persistence_count[track_id][risk_type]
        
        return filtered_risks
