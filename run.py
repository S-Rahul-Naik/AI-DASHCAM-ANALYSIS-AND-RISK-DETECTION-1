#!/usr/bin/env python3
"""
AI Dashcam for Indian Roads - Main entry point
"""

import os
import sys
import logging
import argparse
from src.model_downloader import download_all_models
from src.main import main

def setup():
    """Set up the environment for the dashcam."""
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/ultralytics", exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join("data", "dashcam.log"))
        ]
    )
    
    # Download required models
    print("Checking for required models...")
    downloaded = download_all_models()
    
    if downloaded:
        print(f"Successfully downloaded/verified {len(downloaded)} models")
    else:
        print("All models are already downloaded and verified")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Dashcam for Indian Roads")
    parser.add_argument("--setup-only", action="store_true", help="Only download models, don't run the application")
    parser.add_argument("--download-models", action="store_true", help="Download all required models")
    args = parser.parse_args()
    
    if args.download_models:
        print("Running model download only...")
        from download_models import main as download_main
        download_main()
        sys.exit(0)
    
    # Setup environment
    setup_success = setup()
    
    if args.setup_only:
        print("Setup completed. Use 'python run.py' to start the application.")
        sys.exit(0)
    
    if setup_success:
        print("Starting AI Dashcam application...")
        # Run the main application
        main()
    else:
        print("Setup failed. Please check the logs and try again.")
        sys.exit(1)
