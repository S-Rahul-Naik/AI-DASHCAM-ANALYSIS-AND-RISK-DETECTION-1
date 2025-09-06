import cv2
import time
import logging
from threading import Thread

logger = logging.getLogger(__name__)

class WebcamCapture:
    """
    Handles webcam capture with buffering to ensure smooth frame retrieval.
    Uses a separate thread for capture to prevent blocking the main application.
    """
    
    def __init__(self, camera_id=0, width=1280, height=720, fps=30):
        """
        Initialize the webcam capture.
        
        Args:
            camera_id: ID of the webcam (usually 0 for built-in)
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.frame = None
        self.grabbed = False
        self.stopped = False
        self.frame_count = 0
        self.start_time = None
        
    def start(self):
        """Start the webcam capture thread."""
        logger.info(f"Starting webcam capture (ID: {self.camera_id}, {self.width}x{self.height} @ {self.fps}fps)")
        
        # Initialize the webcam
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam with ID {self.camera_id}")
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Start the thread
        self.start_time = time.time()
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Wait for the first frame
        while not self.grabbed and not self.stopped:
            time.sleep(0.1)
            
        return self
    
    def _update(self):
        """Background thread function to continuously grab frames."""
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.frame = frame
                self.grabbed = True
                self.frame_count += 1
            else:
                logger.error("Failed to grab frame from webcam")
                self.stop()
                break
    
    def read(self):
        """
        Return the current frame.
        
        Returns:
            Current frame from the webcam or None if not available
        """
        return self.frame if self.grabbed else None
    
    def get_fps(self):
        """
        Calculate the actual FPS being achieved.
        
        Returns:
            Actual frames per second
        """
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return self.frame_count / elapsed_time
        return 0
    
    def stop(self):
        """Stop the webcam capture thread and release resources."""
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        logger.info("Webcam capture stopped")

    def __del__(self):
        """Ensure resources are released when the object is destroyed."""
        self.stop()
