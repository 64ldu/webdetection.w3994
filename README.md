# Face AI - Real-time Face Detection and Expression Recognition

An AI application that uses your camera to detect faces, recognize facial expressions, and optionally read lip movements.

## Features

- **Real-time Face Detection**: Detects faces in the camera feed with confidence scores
- **Expression Recognition**: Identifies facial expressions including:
  - Happy
  - Sad
  - Surprised
  - Sleepy
  - Neutral
  - Winking
- **Optional Lip Reading**: Detects when you're speaking vs. silent
- **Visual Feedback**: Draws facial landmarks and overlays on the video feed

## Installation

1. Install Python 3.7 or higher
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
source venv/bin/activate
python face_ai_v2.py
```

### Controls

- **'q'**: Quit the application
- **'l'**: Toggle lip reading on/off

## Camera Permissions (macOS)

The application needs camera access to work. On macOS:

1. Go to **System Preferences** > **Security & Privacy** > **Privacy**
2. Select **Camera** from the left sidebar
3. Find your terminal application (Terminal, iTerm2, etc.) and check the box
4. Restart the terminal and run the application again

Alternatively, you can test camera access with:

```bash
source venv/bin/activate
python test_camera.py
```

## How it Works

The application uses:
- **OpenCV**: For camera access and image processing
- **Haar Cascades**: For face and feature detection
- **Custom algorithms**: To analyze facial features and classify expressions

### Expression Detection Algorithm

The system analyzes:
- Eye detection and count
- Smile detection
- Face geometry
- Historical expression data for stability

### Lip Reading

The lip reading feature detects:
- Basic mouth movement patterns
- Distinguishes between "Speaking" and "Silent" states

## Requirements

- Python 3.7+
- A working camera/webcam
- Camera permissions granted to your terminal
- Sufficient lighting for accurate face detection

## Troubleshooting

- **Camera not found**: Ensure camera permissions are granted to your terminal
- **Poor detection**: Ensure proper lighting and frontal face position
- **Application won't start**: Check that all dependencies are installed correctly

## Privacy Notice

This application processes video locally on your device. No data is sent to external servers.
