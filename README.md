 🧠 Gesture AI Demo

**Real-time human pose estimation and gesture control using PyTorch, YOLOv8, and OpenCV.**

This project showcases a modular and responsive gesture recognition system powered by YOLOv8-Pose. It captures human poses via webcam, classifies gestures, and enables intuitive control — ideal for AR/VR, robotics, and accessibility applications.

 🚀 Features

- Real-time pose detection using YOLOv8-Pose
- Gesture classification with customizable logic
- Webcam-based input for live demos
- Modular codebase for easy extension
- Lightweight and fast — runs on CPU or GPU

 📦 Installation

```bash
# Clone the repository
git clone https://github.com/RAJADURAI3/gesture-ai-demo.git
cd gesture-ai-demo

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate   # On Windows
# or
source venv/bin/activate       # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

```

🎮 Usage

```bash

python main.py

```

-Ensure your webcam is connected

-Perform gestures in front of the camera

-Watch live pose detection and gesture classification

🧩 File Overview

| File              | Purpose                                      |
|-------------------|----------------------------------------------|
| `main.py`         | Launches webcam and runs pose detection      |
| `gestures.py`     | Defines gesture logic and classification     |
| `yolov8n-pose.pt` | YOLOv8-Pose model weights (nano version)     |
| `requirements.txt`| Python dependencies                          |
| `.gitignore`      | Excludes virtual environment and cache files |

📚 Future Improvements

-Add more gesture classes

-Integrate with external control systems (robotics, UI, AR/VR)

-Support multi-person tracking

-Deploy as a web app or mobile interface

👨‍💻 Author
Rajadurai MSc in Data Science | Computer Vision 
Enthusiast