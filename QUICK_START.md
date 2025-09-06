# AI Dashcam Quick Start Guide

This guide will help you get the AI Dashcam up and running quickly.

## Prerequisites

- Python 3.8 or higher
- A webcam connected to your computer
- 2GB+ of free disk space for models and recordings

## Installation

1. Clone the repository or download the source code.

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## First Run

The easiest way to start is with the all-in-one script:

```
python run.py
```

This will:
1. Set up necessary directories
2. Download all required ML models (if not already present)
3. Start the dashcam application

## Manual Setup

If you prefer to set up step by step:

1. Download the required models:
   ```
   python download_models.py
   ```

2. Start the application:
   ```
   python run.py
   ```

## Command Line Options

- **Standard startup**:
  ```
  python run.py
  ```

- **Setup only** (download models without starting the application):
  ```
  python run.py --setup-only
  ```

- **Download models** (with detailed progress information):
  ```
  python run.py --download-models
  ```

## Using the Application

Once started:

1. The application will activate your webcam and display the video feed with AI overlays.

2. Object detection boxes will highlight vehicles, pedestrians, and other objects.

3. Risk assessment information will be displayed, including warnings for close following, potential collisions, and lane departures.

4. All faces and license plates will be automatically anonymized (blurred).

5. Incidents are automatically saved when risk levels exceed thresholds.

6. Press `q` to quit the application.

## Troubleshooting

- **Missing models error**: Run `python download_models.py` to ensure all models are downloaded
- **Webcam not detected**: Ensure your webcam is connected and not being used by another application
- **Performance issues**: If the application runs slowly, consider using a more powerful computer with a GPU
- **Import errors**: Make sure you have installed all dependencies with `pip install -r requirements.txt`

## Next Steps

For more detailed information about the project, architecture, and customization options, see the full [README.md](README.md) and documentation in the `docs/` directory.
