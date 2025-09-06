import cv2
import time
import logging
import os
from datetime import datetime
from threading import Thread, Lock
import queue

import config

logger = logging.getLogger(__name__)

class VideoRecorder:
    """
    Records video to local storage with circular buffer functionality.
    """
    
    def __init__(self, output_dir=None, max_size_gb=None, segment_duration=60):
        """
        Initialize the video recorder.
        
        Args:
            output_dir: Directory to save videos, or None to use the path from config
            max_size_gb: Maximum storage size in GB, or None to use the value from config
            segment_duration: Duration of each video segment in seconds
        """
        self.output_dir = output_dir or os.path.join(config.DATA_DIR, "recordings")
        self.max_size_gb = max_size_gb or config.MAX_STORAGE_GB
        self.segment_duration = segment_duration
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.current_writer = None
        self.current_filepath = None
        self.segment_start_time = None
        self.frame_queue = queue.Queue(maxsize=100)  # Buffer for frames
        self.lock = Lock()
        self.stopped = False
        self.thread = None
        
        logger.info(f"Video recorder initialized with output directory: {self.output_dir}")
        logger.info(f"Maximum storage size: {self.max_size_gb} GB")
        logger.info(f"Segment duration: {self.segment_duration} seconds")
    
    def start(self):
        """Start the video recorder thread."""
        logger.info("Starting video recorder")
        
        self.stopped = False
        self.thread = Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        
        return self
    
    def stop(self):
        """Stop the video recorder and release resources."""
        logger.info("Stopping video recorder")
        
        self.stopped = True
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        with self.lock:
            if self.current_writer is not None:
                self.current_writer.release()
                self.current_writer = None
    
    def write_frame(self, frame):
        """
        Queue a frame for recording.
        
        Args:
            frame: Frame to record
        """
        if not self.stopped and frame is not None:
            try:
                # Use non-blocking put with a timeout to avoid blocking if queue is full
                self.frame_queue.put(frame.copy(), block=True, timeout=0.1)
            except queue.Full:
                logger.warning("Frame queue is full, dropping frame")
    
    def _record_loop(self):
        """Background thread for recording frames."""
        while not self.stopped:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(block=True, timeout=0.1)
                
                with self.lock:
                    # Check if we need to start a new video segment
                    current_time = time.time()
                    
                    if (self.current_writer is None or 
                        self.segment_start_time is None or 
                        current_time - self.segment_start_time >= self.segment_duration):
                        
                        # Close the current writer if it exists
                        if self.current_writer is not None:
                            self.current_writer.release()
                        
                        # Create a new video segment
                        self._create_new_segment(frame.shape[1], frame.shape[0])
                        
                        # Manage storage
                        self._manage_storage()
                    
                    # Write the frame
                    if self.current_writer is not None:
                        self.current_writer.write(frame)
                
                # Mark the task as done
                self.frame_queue.task_done()
            
            except queue.Empty:
                # No frames available, wait a bit
                pass
            
            except Exception as e:
                logger.error(f"Error in recording loop: {e}")
    
    def _create_new_segment(self, width, height):
        """
        Create a new video segment.
        
        Args:
            width: Frame width
            height: Frame height
        """
        # Generate timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        fps = 30  # Frames per second
        
        self.current_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        self.current_filepath = filepath
        self.segment_start_time = time.time()
        
        logger.info(f"Started new video segment: {filename}")
    
    def _manage_storage(self):
        """
        Manage storage by deleting old files if the total size exceeds the limit.
        """
        try:
            # Get all recording files
            files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) 
                     if f.startswith("recording_") and f.endswith(".mp4")]
            
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: os.path.getmtime(x))
            
            # Calculate total size
            total_size_bytes = sum(os.path.getsize(f) for f in files)
            total_size_gb = total_size_bytes / (1024**3)
            
            # Delete oldest files if over the limit
            while total_size_gb > self.max_size_gb and files:
                file_to_delete = files.pop(0)  # Get oldest file
                file_size = os.path.getsize(file_to_delete) / (1024**3)
                
                try:
                    os.remove(file_to_delete)
                    logger.info(f"Deleted old recording to free space: {os.path.basename(file_to_delete)}")
                    
                    total_size_gb -= file_size
                except Exception as e:
                    logger.error(f"Failed to delete file {file_to_delete}: {e}")
        
        except Exception as e:
            logger.error(f"Error in storage management: {e}")
            
    def get_current_filepath(self):
        """
        Get the filepath of the current recording segment.
        
        Returns:
            Current filepath or None
        """
        with self.lock:
            return self.current_filepath
