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
