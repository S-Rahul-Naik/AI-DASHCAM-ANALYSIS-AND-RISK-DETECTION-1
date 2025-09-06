# Quick Start Guide

This guide will help you get started with the AI Dashcam system.

## Prerequisites

- Python 3.8 or higher
- Webcam connected to your computer
- (Optional but recommended) CUDA-capable GPU

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd dashcam
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Dashcam

To start the AI Dashcam:

```
python src/main.py
```

This will:
- Start capturing video from your webcam
- Process the video in real-time to detect objects, lanes, and assess risks
- Display the processed video with overlays showing detections and alerts
- Record anonymized video and incident data to the `data` directory

## Key Features

- **Object Detection**: Detects vehicles, pedestrians, bicycles, etc.
- **Risk Assessment**: Calculates collision risks, lane departures, etc.
- **Privacy**: Anonymizes faces (and license plates in the full version)
- **Incident Recording**: Automatically stores data about detected incidents
- **Explainable Alerts**: Provides detailed explanations for each alert

## Configuration

You can customize the behavior of the system by editing `config.py`. Key settings include:

- `WEBCAM_ID`: ID of the webcam to use (usually 0 for the default webcam)
- `VIDEO_WIDTH` and `VIDEO_HEIGHT`: Resolution of the capture
- `DETECTION_CONFIDENCE`: Confidence threshold for object detection
- Risk thresholds such as `TTC_THRESHOLD_CRITICAL`
- Privacy settings like `ANONYMIZE_FACES`

## Testing Individual Components

To test just the webcam functionality:

```
python tests/test_webcam.py
```

## Keyboard Controls

While the dashcam is running:
- Press `q` to quit

## Data Storage

The system stores data in the following locations:
- `data/recordings/`: Continuous video recordings
- `data/incidents/`: Incident data and frames
- `data/incidents/reports/`: Generated incident reports
