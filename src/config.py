# Project configuration
import os

DEBUG = True
LOG_LEVEL = "INFO"

# Path helpers
def model_path(model_name):
    """Return the absolute path to a model file."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", model_name)

# Video settings
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30
WEBCAM_ID = 0  # Default webcam ID, usually 0 for built-in webcam

# Storage settings
DATA_DIR = "data"
MAX_STORAGE_GB = 10  # Maximum storage in GB for circular buffer
INCIDENT_RETENTION_DAYS = 30  # How long to keep incident data before auto-deletion

# Detection settings
DETECTION_MODEL = model_path("yolov8n.pt")  # Lightweight YOLOv8 nano model
DETECTION_CONFIDENCE = 0.4
DETECTION_CLASSES = ["car", "person", "traffic light"]  # Only detect these classes
DETECTION_FREQUENCY = 5  # Run detection every N frames to save resources

# Lane detection settings
LANE_DETECTION_MODEL = model_path("ultralytics/yolov8n-seg.pt")  # YOLOv8 segmentation model for lanes
LANE_CONFIDENCE = 0.3
USE_CV_FALLBACK = True  # Use classical CV for lane detection if model fails

# Risk assessment settings
TTC_THRESHOLD_CRITICAL = 2.0  # Time to collision threshold in seconds for critical alert (increased for earlier warning)
TTC_THRESHOLD_WARNING = 4.0  # Time to collision threshold in seconds for warning (increased for earlier warning)
FOLLOWING_DISTANCE_CRITICAL = 1.2  # Following distance threshold in seconds for critical alert (increased)
FOLLOWING_DISTANCE_WARNING = 2.5  # Following distance threshold in seconds for warning (increased)
LANE_DEPARTURE_THRESHOLD = 0.7  # Lane departure threshold (0-1) for alert (lowered for earlier warning)

# Risk assessment filtering
RISK_COOLDOWN_SECONDS = 3.0  # Minimum time between repeated alerts of the same type
RISK_PERSISTENCE_THRESHOLD = 2  # Number of consecutive frames a risk must be present to trigger alert
RISK_VELOCITY_THRESHOLD = 0.5  # Minimum velocity (pixels/frame) for collision risk consideration

# Privacy settings
ANONYMIZE_FACES = True
ANONYMIZE_PLATES = True
FACE_DETECTION_MODEL = model_path("ultralytics/yolov8n-face.pt")  # YOLOv8 nano model fine-tuned for face detection
PLATE_DETECTION_MODEL = model_path("ultralytics/yolov8n-plate.pt")  # YOLOv8 nano model fine-tuned for license plate detection

# UI settings
DISPLAY_DETECTION_BOXES = True
DISPLAY_RISK_METRICS = True
DISPLAY_LANE_OVERLAY = True
DISPLAY_ALERTS = True
ALERT_DISPLAY_TIME = 3  # Seconds to display alert
