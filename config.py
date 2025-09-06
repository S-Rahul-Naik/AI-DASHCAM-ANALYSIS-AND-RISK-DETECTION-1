# Project configuration
DEBUG = True
LOG_LEVEL = "INFO"

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
DETECTION_MODEL = "models/yolov8n.pt"  # or "models/nanodet.onnx"
DETECTION_CONFIDENCE = 0.4
DETECTION_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "animal"]
DETECTION_FREQUENCY = 5  # Run detection every N frames to save resources

# Lane detection settings
LANE_DETECTION_MODEL = "models/lane_detection.pt"
LANE_CONFIDENCE = 0.3

# Risk assessment settings
TTC_THRESHOLD_CRITICAL = 0.7  # Time to collision threshold in seconds for critical alert - more accurate
TTC_THRESHOLD_WARNING = 1.3  # Time to collision threshold in seconds for warning - more accurate
FOLLOWING_DISTANCE_CRITICAL = 0.5  # Following distance threshold in seconds for critical alert
FOLLOWING_DISTANCE_WARNING = 1.0  # Following distance threshold in seconds for warning
LANE_DEPARTURE_THRESHOLD = 0.75  # Lane departure threshold (0-1) for alert
RISK_PERSISTENCE_THRESHOLD = 2  # Require risk to persist for 2 frames (better for real-time)
RISK_VELOCITY_THRESHOLD = 1.2  # Balanced velocity threshold for real-time detection
RISK_SCORE_THRESHOLD = 0.7  # Minimum risk score threshold for alerts

# Lane change detection settings
LANE_CHANGE_HORIZONTAL_VELOCITY_THRESHOLD = 2.5  # Minimum horizontal velocity to consider a lane change
LANE_CHANGE_DETECTION_TIME = 0.8  # Time in seconds to track potential lane change
LANE_CHANGE_RISK_THRESHOLD = 0.65  # Risk threshold specific to lane changes
LANE_CHANGE_PERSISTENCE_THRESHOLD = 2  # Frames to persist for lane change detection

# Pothole detection settings
POTHOLE_DETECTION_CONFIDENCE = 0.60  # Minimum confidence for pothole detection
POTHOLE_RISK_DISTANCE = 0.35  # Distance threshold (normalized 0-1) for pothole risk calculation
POTHOLE_RISK_THRESHOLD = 0.6  # Minimum risk score for pothole alerts
POTHOLE_PERSISTENCE_THRESHOLD = 2  # Frames to persist for pothole detection
POTHOLE_SMALL_SIZE_MIN = 0.001  # Minimum size of small potholes (% of frame)
POTHOLE_MEDIUM_SIZE_MIN = 0.01  # Minimum size of medium potholes (% of frame)
POTHOLE_LARGE_SIZE_MIN = 0.025  # Minimum size of large potholes (% of frame)
POTHOLE_SEVERE_SIZE_MIN = 0.05  # Minimum size of severe potholes (% of frame)

# Privacy settings
ANONYMIZE_FACES = False
ANONYMIZE_PLATES = False
FACE_DETECTION_MODEL = "models/face_detection.pt"
PLATE_DETECTION_MODEL = "models/plate_detection.pt"

# UI settings
DISPLAY_DETECTION_BOXES = True
DISPLAY_RISK_METRICS = True
DISPLAY_LANE_OVERLAY = True
DISPLAY_ALERTS = True
ALERT_DISPLAY_TIME = 3  # Seconds to display alert

# Speech settings
SPEECH_ENABLED = True
SPEECH_PRECACHE_ALERTS = True  # Pre-cache common alerts for faster response
SPEECH_RATE = 1  # Speech rate (-10 to 10, higher is faster)
SPEECH_USE_FEMALE_VOICE = True  # Use female voice for better clarity on Windows
SPEECH_CRITICAL_PRIORITY = True  # Interrupt current speech for critical alerts
