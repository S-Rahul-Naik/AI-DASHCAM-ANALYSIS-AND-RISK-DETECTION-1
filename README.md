# AI Dashcam for Indian Roads

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-MVP-green.svg)

An edge-based AI dashcam system that detects risky maneuvers, generates explainable alerts, anonymizes video, and produces data for insurance-grade incident reports - all while keeping data local and private.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Components](#components)
- [Models](#models)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

The AI Dashcam for Indian Roads is designed to address the unique challenges of driving conditions in India, providing real-time risk assessment, privacy-focused video recording, and incident detection specifically optimized for Indian traffic patterns. The system processes everything locally on the device, ensuring privacy and data security while offering professional-grade detection and analysis.

### Core Differentiators

1. **Edge-First Architecture with Privacy by Design**
   - All processing happens on the device
   - Private by default, with optional cloud connectivity
   - All PII (faces, license plates) is anonymized at the edge

2. **India-Specific AI Optimization**
   - Detection models specialized for Indian vehicles and traffic patterns
   - Support for two/three-wheelers with multiple passengers
   - Recognition of unstructured traffic patterns and local hazards

3. **Explainable AI with Actionable Insights**
   - Real-time visual alerts with clear explanations
   - Driver behavior pattern analysis
   - Actionable recommendations for safer driving

4. **Legal & Insurance Framework Integration**
   - Tamper-evident incident packages for insurance claims
   - Format compatible with insurance company requirements
   - Evidence-grade recording of incidents

## ✨ Key Features

- **Real-time object detection** using YOLOv8/NanoDet optimized for Indian roads
- **Risk assessment** with time-to-collision, following distance, and lane departure metrics
- **Privacy by design** with on-device anonymization of faces and license plates
- **Explainable alerts** with visual overlays and textual explanations
- **Local storage** with tamper-evident incident packs
- **Webcam input** for easy setup and testing
- **Pothole detection** for road hazard identification
- **Lane change risk assessment** for safe driving guidance

## � Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/username/ai-dashcam-indian-roads.git
cd ai-dashcam-indian-roads

# Install dependencies
pip install -r requirements.txt

# Download required models and run the application
python run.py
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/username/ai-dashcam-indian-roads.git
cd ai-dashcam-indian-roads

# Install dependencies
pip install -r requirements.txt

# Download required models
python download_models.py

# Run the application
python run.py
```

## 🚀 Usage

### Command Line Options

- **Standard startup**:
  ```bash
  python run.py
  ```

- **Setup only** (download models without starting the application):
  ```bash
  python run.py --setup-only
  ```

- **Download models** (with detailed progress information):
  ```bash
  python run.py --download-models
  ```

### Using the Application

Once started:

1. The application will activate your webcam and display the video feed with AI overlays.
2. Object detection boxes will highlight vehicles, pedestrians, and other objects.
3. Risk assessment information will be displayed, including warnings for close following, potential collisions, and lane departures.
4. All faces and license plates will be automatically anonymized (blurred).
5. Incidents are automatically saved when risk levels exceed thresholds.
6. Press `q` to quit the application.

## 🏗️ Architecture

The AI Dashcam uses a modular architecture with these main components:

1. **Perception Layer**
   - Object detection and tracking
   - Lane detection
   - Environmental feature extraction

2. **Analysis Layer**
   - Risk assessment
   - Behavior analysis
   - Incident detection

3. **Privacy Layer**
   - Face anonymization
   - License plate blurring
   - Data minimization

4. **Storage Layer**
   - Local video recording
   - Incident packaging
   - Tamper-evident storage

5. **UI Layer**
   - Real-time visual feedback
   - Alert display
   - Explanation system

## 🧩 Components

### Perception
- **ObjectDetector**: Detects vehicles, pedestrians, obstacles using YOLOv8
- **ObjectTracker**: Tracks objects across frames for motion analysis
- **LaneDetector**: Identifies lane markings and road boundaries

### Risk Assessment
- **RiskAnalyzer**: Calculates various risk metrics:
  - Time to collision
  - Following distance
  - Lane departure risk
  - Pothole avoidance

### Privacy
- **Anonymizer**: Blurs faces and license plates in real-time

### Storage
- **VideoRecorder**: Records continuous video in a circular buffer
- **IncidentManager**: Creates tamper-evident incident packages

### UI
- **DashcamDisplay**: Renders the video feed with overlays and alerts

## 🤖 Models

The system uses the following models:

- **Object Detection**: YOLOv8n - A lightweight but accurate model for detecting vehicles, pedestrians, and obstacles
- **Lane Detection**: YOLOv8n-seg - Segmentation model adapted for lane detection
- **Face Detection**: YOLOv8n-face - Fine-tuned for detecting faces for privacy anonymization
- **License Plate Detection**: YOLOv8n-plate - Fine-tuned for detecting license plates for privacy anonymization

## 📂 Project Structure

```
.
├── config.py                   # Global configuration settings
├── download_models.py          # Utility to download ML models
├── QUICK_START.md              # Quick start guide
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── run.py                      # Main entry point
├── data/                       # Stored data
│   ├── incidents/              # Recorded incident packages
│   └── recordings/             # Continuous video recordings
├── docs/                       # Documentation
│   └── ENHANCED_MVP.md         # Detailed MVP strategy
├── models/                     # Pre-trained ML models
│   ├── face_detection.pt
│   ├── lane_detection.pt
│   ├── plate_detection.pt
│   ├── yolov8n.pt
│   └── ultralytics/            # Model dependencies
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py               # Component-specific configuration
│   ├── main.py                 # Main application logic
│   ├── model_downloader.py     # Model download utilities
│   ├── webcam.py               # Webcam interface
│   ├── perception/             # Object detection, tracking, lanes
│   ├── privacy/                # Anonymization
│   ├── risk_assessment/        # Risk scoring and alerts
│   ├── storage/                # Recording and incident management
│   └── ui/                     # User interface
└── tests/                      # Unit and integration tests
    ├── README.md               # Testing documentation
    └── test_webcam.py          # Webcam module tests
```

## ⚙️ Configuration

The application is highly configurable through the `config.py` file:

- **Video settings**: Resolution, FPS, webcam ID
- **Detection settings**: Model paths, confidence thresholds, detection frequency
- **Storage settings**: Storage limits, retention periods
- **Risk assessment**: Thresholds for different risk factors
- **Lane detection**: Detection parameters and thresholds

Key configuration options:

```python
# Video settings
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30
WEBCAM_ID = 0  # Default webcam ID

# Detection settings
DETECTION_MODEL = "models/yolov8n.pt"
DETECTION_CONFIDENCE = 0.4

# Risk assessment settings
TTC_THRESHOLD_CRITICAL = 0.7  # Time to collision threshold in seconds
TTC_THRESHOLD_WARNING = 1.3
```

## 🔨 Development

This project is designed to be hackathon-ready but also extensible for production use. The modular architecture allows for easy replacement of components (e.g., different object detection models, risk assessment algorithms, etc.).

### Adding New Features

1. **New Risk Assessments**: Extend the `RiskAnalyzer` class with new risk metrics
2. **Custom Detectors**: Create new detectors in the `perception` module
3. **UI Enhancements**: Modify the `DashcamDisplay` class for custom visualizations

## 🔍 Troubleshooting

- **Missing models error**: Run `python download_models.py` to ensure all models are downloaded
- **Webcam not detected**: Ensure your webcam is connected and not being used by another application
- **Performance issues**: If the application runs slowly, consider:
  - Reducing the resolution in `config.py`
  - Decreasing the detection frequency
  - Using a computer with a GPU
- **Import errors**: Make sure you have installed all dependencies with `pip install -r requirements.txt`

## 🛣️ Future Roadmap

- **Multi-Camera Support**: Integration with multiple camera sources
- **Advanced Analytics**: More detailed driving behavior analysis
- **Cloud Integration**: Optional cloud backup and synchronization
- **Mobile App**: Companion mobile application for configuration and viewing
- **Hardware Optimization**: Support for dedicated edge devices (Jetson Nano, etc.)
- **Fleet Management**: Features for commercial vehicle fleets

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
