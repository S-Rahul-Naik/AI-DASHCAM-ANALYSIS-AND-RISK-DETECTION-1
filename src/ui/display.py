import cv2
import numpy as np
import logging
import time
from datetime import datetime

import config
from src.ui.speech_alert import SpeechAlertSystem

logger = logging.getLogger(__name__)

class DashcamDisplay:
    """
    Handles the visual display of the dashcam feed with overlays.
    """
    
    def __init__(self, window_name="AI Dashcam"):
        """
        Initialize the dashcam display.
        
        Args:
            window_name: Name of the display window
        """
        self.window_name = window_name
        self.display_detection_boxes = config.DISPLAY_DETECTION_BOXES
        self.display_risk_metrics = config.DISPLAY_RISK_METRICS
        self.display_lane_overlay = config.DISPLAY_LANE_OVERLAY
        self.display_alerts = config.DISPLAY_ALERTS
        self.alert_display_time = config.ALERT_DISPLAY_TIME
        
        self.current_alerts = []  # List of (alert, timestamp) tuples
        
        # Initialize speech alert system
        self.speech_alert = SpeechAlertSystem()
        
        # Colors for different object classes (BGR format)
        self.colors = {
            'car': (0, 255, 0),       # Green
            'truck': (0, 255, 0),     # Green
            'bus': (0, 255, 0),       # Green
            'motorcycle': (0, 255, 0), # Green
            'bicycle': (0, 255, 0),   # Green
            'person': (0, 255, 0),    # Green (changed from red)
            'traffic light': (0, 165, 255),  # Orange
            'default': (0, 255, 0)    # Green (default color for all objects)
        }
        
        # Risk colors
        self.risk_color = (0, 0, 255)  # Red for high risk objects
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        logger.info("Display initialized")
    
    def update(self, frame, detections=None, tracking=None, lanes=None, risks=None, alerts=None):
        """
        Update the display with a new frame and overlays.
        
        Args:
            frame: Image frame
            detections: List of Detection objects
            tracking: List of TrackedObject objects
            lanes: List of Lane objects
            risks: List of Risk objects
            alerts: List of Alert objects
        """
        if frame is None:
            return
        
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Draw lane overlays
        if self.display_lane_overlay and lanes:
            display_frame = self._draw_lanes(display_frame, lanes)
        
        # Draw detection boxes
        if self.display_detection_boxes and tracking:
            display_frame = self._draw_tracking(display_frame, tracking, risks)
        
        # Draw risk metrics
        if self.display_risk_metrics and risks:
            display_frame = self._draw_risk_metrics(display_frame, risks)
        
        # Update and draw alerts
        if self.display_alerts:
            if alerts:
                # Sort alerts by priority (critical first)
                sorted_alerts = sorted(alerts, key=lambda a: 0 if a.is_critical else 1)
                
                # Process the most critical alerts immediately
                for alert in sorted_alerts:
                    if alert.is_critical:
                        # Send critical alerts to speech system with high priority
                        self.speech_alert.add_alert(alert)
                    
                    # Add to current alerts for display
                    self.current_alerts.append((alert, time.time()))
                
                # Process non-critical alerts after critical ones
                for alert in sorted_alerts:
                    if not alert.is_critical:
                        # Send warning alerts to speech system with normal priority
                        self.speech_alert.add_alert(alert)
            
            display_frame = self._draw_alerts(display_frame)
        
        # Draw status info
        display_frame = self._draw_status(display_frame)
        
        # Show the frame
        cv2.imshow(self.window_name, display_frame)
    
    def _draw_tracking(self, frame, tracked_objects, risks=None):
        """
        Draw tracking boxes and IDs.
        
        Args:
            frame: Image frame
            tracked_objects: List of TrackedObject objects
            risks: List of Risk objects
            
        Returns:
            Frame with tracking overlays
        """
        # Create a map of tracked object IDs to risks for fast lookup
        risk_map = {}
        if risks:
            for risk in risks:
                risk_map[risk.tracked_object.track_id] = risk
        
        for obj in tracked_objects:
            # Only display objects that are risky (with risk score > 0.5)
            if obj.track_id in risk_map and risk_map[obj.track_id].risk_score > 0.5:
                # This is a risky object, mark it in red
                color = self.risk_color  # Red for risky objects
                
                # Draw bounding box
                x1, y1, x2, y2 = obj.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID, class and risk info
                risk = risk_map[obj.track_id]
                risk_text = f"{risk.risk_score:.2f}"
                text = f"{obj.class_name} #{obj.track_id} Risk: {risk_text}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Non-risky objects will not be drawn
        
        return frame
    
    def _draw_lanes(self, frame, lanes):
        """
        Draw lane overlays.
        
        Args:
            frame: Image frame
            lanes: List of Lane objects
            
        Returns:
            Frame with lane overlays
        """
        # Lanes are detected but not drawn as lines/arrows as requested
        # Still tracking lane positions for risk assessment
        return frame
    
    def _draw_risk_metrics(self, frame, risks):
        """
        Draw risk metrics.
        
        Args:
            frame: Image frame
            risks: List of Risk objects
            
        Returns:
            Frame with risk metrics
        """
        height, width = frame.shape[:2]
        
        # Draw risk metrics in the top-right corner
        y_offset = 30
        
        for risk in risks:
            if risk.risk_score > 0.3:  # Only show significant risks
                # Get color based on risk level
                if risk.is_critical:
                    color = (0, 0, 255)  # Red for critical
                elif risk.is_warning:
                    color = (0, 165, 255)  # Orange for warning
                else:
                    color = (0, 255, 0)  # Green for low risk
                
                # Create risk text
                obj = risk.tracked_object
                text = f"{risk.risk_type.upper()} - {obj.class_name} #{obj.track_id} - Risk: {risk.risk_score:.2f}"
                
                # Add metric details
                if 'ttc' in risk.metrics:
                    text += f" - TTC: {risk.metrics['ttc']:.1f}s"
                
                if 'lane_position' in risk.metrics:
                    lane_pos = risk.metrics['lane_position']
                    direction = "LEFT" if lane_pos < 0 else "RIGHT"
                    text += f" - {direction}: {abs(lane_pos):.2f}"
                
                # Draw text
                cv2.putText(frame, text, (width - 550, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
        
        return frame
    
    def _draw_alerts(self, frame):
        """
        Draw active alerts with enhanced visibility for drivers.
        
        Args:
            frame: Image frame
            
        Returns:
            Frame with alerts
        """
        height, width = frame.shape[:2]
        current_time = time.time()
        
        # Filter out expired alerts
        self.current_alerts = [(alert, timestamp) for alert, timestamp in self.current_alerts
                              if current_time - timestamp < self.alert_display_time]
        
        if not self.current_alerts:
            return frame
        
        # Create more prominent alert box in the center top of the frame
        overlay = frame.copy()
        
        # Draw critical alerts more prominently than warnings
        critical_alerts = [(a, t) for a, t in self.current_alerts if a.is_critical]
        warning_alerts = [(a, t) for a, t in self.current_alerts if a.is_warning and not a.is_critical]
        
        # Handle critical alerts first - they get the most prominent display
        if critical_alerts:
            # Prominent red banner for critical alerts
            banner_height = 80
            cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 220), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Show the most recent critical alert in large text
            alert, _ = critical_alerts[0]
            # Message in large text
            cv2.putText(frame, alert.message, (width//2 - 180, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Handle warning alerts
        if warning_alerts and not critical_alerts:  # Only show if no critical alerts
            # Yellow/orange banner for warnings
            banner_height = 60
            cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 120, 255), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Show the most recent warning alert
            alert, _ = warning_alerts[0]
            cv2.putText(frame, alert.message, (width//2 - 150, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # For additional alerts beyond the first, show in a side panel
        if len(self.current_alerts) > 1:
            # Side panel for additional alerts
            panel_width = 300
            panel_height = min(len(self.current_alerts) * 30 + 20, 120)
            cv2.rectangle(overlay, (10, 100), (panel_width, 100 + panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Show additional alerts in smaller text
            y_offset = 125
            for i, (alert, timestamp) in enumerate(self.current_alerts):
                if i == 0 and (critical_alerts or warning_alerts):
                    continue  # Skip the first alert if already shown in banner
                
                # Get color based on alert level
                if alert.is_critical:
                    color = (0, 0, 255)  # Red for critical
                elif alert.is_warning:
                    color = (0, 165, 255)  # Orange for warning
                else:
                    color = (255, 255, 255)  # White for info
                
                # Show abbreviated message
                short_msg = alert.message
                if len(short_msg) > 25:
                    short_msg = short_msg[:22] + "..."
                    
                cv2.putText(frame, short_msg, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                y_offset += 25
                
                if y_offset > 100 + panel_height - 10:
                    break  # Don't overflow the panel
        
        return frame
    
    def _draw_status(self, frame):
        """
        Draw status information.
        
        Args:
            frame: Image frame
            
        Returns:
            Frame with status information
        """
        height, width = frame.shape[:2]
        
        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw "REC" indicator for recording
        cv2.circle(frame, (width - 20, 20), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (width - 55, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def close(self):
        """Close the display window and shut down speech system."""
        self.speech_alert.shutdown()
        cv2.destroyWindow(self.window_name)
