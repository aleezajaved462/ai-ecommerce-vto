# AI Engine - Virtual Try-On Module

This module serves as the core Artificial Intelligence engine for the AI E-Commerce Virtual Try-On platform. It handles real-time landmark tracking, pose stabilization, and image overlay processing (T-shirts, watches, eyewear) to deliver an interactive virtual try-on experience.

---

## 🌟 Key Features

- **Real-Time Body & Pose Tracking:** Utilizes MediaPipe for high-precision pose and landmark detection.
- **Stabilization & Smoothing:** Implements Exponential Moving Average (EMA) smoothing and deadzone filtering to reduce visual flickering.
- **Dynamic Anchoring & Overlay:** Automatically warps and anchors products (clothing, watches, eyewear) onto target landmarks.
- **Socket.IO Streaming Server:** Real-time bi-directional communication between the frontend client and the Python AI pipeline.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.9+
- **Core Libraries:** OpenCV (`opencv-python`), MediaPipe (`mediapipe`), NumPy
- **Server:** Python Socket.IO, Eventlet / Flask

---

## 🚀 Getting Started

### 1. Environment Setup
Create and activate a Python virtual environment:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
