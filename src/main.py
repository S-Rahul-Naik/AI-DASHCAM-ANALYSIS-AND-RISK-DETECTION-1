import cv2
import time
import logging
import sys
import os
from datetime import datetime

# Import configuration
from . import config

# Import components
from .webcam import WebcamCapture
from .perception.detection import ObjectDetector
from .perception.tracking import ObjectTracker
from .perception.lane_detection import LaneDetector
from .risk_assessment.risk_analyzer import RiskAnalyzer
from .privacy.anonymizer import Anonymizer
from .storage.recorder import VideoRecorder
from .storage.incident_manager import IncidentManager
from .ui.display import DashcamDisplay

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(os.path.join(config.DATA_DIR, 'dashcam.log'))
    ]
)

logger = logging.getLogger(__name__)

class DashcamApp:
    """
    Main application class that orchestrates all components of the AI dashcam.
    """
    
    def __init__(self):
        """Initialize the dashcam application and its components."""
        logger.info("Initializing AI Dashcam")
        
        # Ensure data directory exists
        os.makedirs(config.DATA_DIR, exist_ok=True)
        
        # Initialize components
        self.webcam = WebcamCapture(
            camera_id=config.WEBCAM_ID,
            width=config.VIDEO_WIDTH,
            height=config.VIDEO_HEIGHT,
            fps=config.FPS
        )
        
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()
        self.lane_detector = LaneDetector()
        self.risk_analyzer = RiskAnalyzer()
        self.anonymizer = Anonymizer()
        self.recorder = VideoRecorder()
        self.incident_manager = IncidentManager()
        self.display = DashcamDisplay()
        
        self.frame_count = 0
        self.running = False
        
    def start(self):
        """Start the dashcam application."""
        logger.info("Starting AI Dashcam")
        self.running = True
        
        # Start webcam
        self.webcam.start()
        
        # Start video recorder
        self.recorder.start()
        
        # Start processing loop
        self._process_frames()
        
    def stop(self):
        """Stop the dashcam application and release resources."""
        logger.info("Stopping AI Dashcam")
        self.running = False
        
        # Stop components
        self.webcam.stop()
        self.recorder.stop()
        
        # Close all windows
        cv2.destroyAllWindows()
        
    def _process_frames(self):
        """Main processing loop for webcam frames."""
        last_detection_time = time.time()
        detection_interval = 1.0 / config.DETECTION_FREQUENCY
        
        while self.running:
            # Get frame from webcam
            frame = self.webcam.read()
            if frame is None:
                logger.warning("No frame received from webcam")
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            current_time = time.time()
            
            # Create a copy of the frame for processing (original will be recorded)
            process_frame = frame.copy()
            
            # Run detection at specified frequency
            detections = None
            if current_time - last_detection_time >= detection_interval:
                detections = self.detector.detect(process_frame)
                last_detection_time = current_time
                
                # Update object tracking with new detections
                tracked_objects = self.tracker.update(detections)
                
                # Detect lanes
                lanes = self.lane_detector.detect(process_frame)
                
                # Analyze risk based on detections, tracking and lanes
                risks, alerts = self.risk_analyzer.analyze(tracked_objects, lanes)
                
                # Handle any critical incidents
                if any(alert.is_critical for alert in alerts):
                    # No anonymization needed - privacy protection is disabled
                    self.incident_manager.record_incident(
                        timestamp=datetime.now(),
                        frame=frame,  # Use original frame
                        detections=tracked_objects,
                        lanes=lanes,
                        risks=risks,
                        alerts=alerts
                    )
            
            # No anonymization needed - privacy protection is disabled
            # Record the original frame
            self.recorder.write_frame(frame)
            
            # Update display with processed frame (not anonymized)
            self.display.update(
                frame=process_frame,
                detections=detections,
                tracking=tracked_objects if detections else None,
                lanes=lanes if detections else None,
                risks=risks if detections else None,
                alerts=alerts if detections else None
            )
            
            # Check for exit key (q)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

def main():
    """Main entry point for the application."""
    app = DashcamApp()
    try:
        app.start()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    finally:
        app.stop()

if __name__ == "__main__":
    main()
