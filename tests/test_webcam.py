import cv2
import time
import logging
import os
import sys

# Add the parent directory to sys.path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components
from src.webcam import WebcamCapture

def test_webcam():
    """Test webcam capture functionality."""
    print("Testing webcam capture...")
    
    # Initialize webcam
    webcam = WebcamCapture(camera_id=0, width=640, height=480, fps=30)
    
    try:
        # Start webcam
        webcam.start()
        
        # Create window
        cv2.namedWindow("Webcam Test", cv2.WINDOW_NORMAL)
        
        # Capture frames for 10 seconds
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10:
            frame = webcam.read()
            
            if frame is not None:
                # Add timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Webcam Test", frame)
                frame_count += 1
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Calculate actual FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        
        print(f"Captured {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"Actual FPS: {fps:.2f}")
        
    finally:
        # Clean up
        webcam.stop()
        cv2.destroyAllWindows()
    
    print("Webcam test completed")

if __name__ == "__main__":
    test_webcam()
