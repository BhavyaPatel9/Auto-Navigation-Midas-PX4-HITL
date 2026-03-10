# Autonomous Navigation via MiDaS Depth & PX4 HITL
### Perception-Driven Autonomy via MiDaS Depth Estimation & PX4

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20%7C%20PX4-green)](https://px4.io/)
[![AI Model](https://img.shields.io/badge/AI-MiDaS%20Small-orange)](https://github.com/isl-org/MiDaS)
[![Inference](https://img.shields.io/badge/Inference-TensorRT%20%7C%20ONNX-red)](https://developer.nvidia.com/tensorrt)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

This repository contains a production-ready autonomous navigation stack designed for UAVs operating in GPS-denied or obstacle-dense environments. By leveraging monocular depth estimation (MiDaS), the system transforms a single 2D camera feed into a spatial awareness map, enabling real-time obstacle avoidance and reactive trajectory planning.


## 📺 Project Demo
<p align="center">
  <video src="https://github.com/BhavyaPatel9/Auto-Navigation-Midas-PX4-HITL/Media/Midas_Unreal_2.mp4" width="100%" controls autoplay loop muted>
    Your browser does not support the video tag.
  </video>
</p>

> **Note:** The video above showcases real-time obstacle avoidance using the MiDaS depth engine integrated with PX4.
---

## 🌟 Key Features
* **Dual-Mode Validation:** Optimized scripts for both **SITL** (Software-In-The-Loop) using MAVSDK and **HITL** (Hardware-In-The-Loop) using PyMavlink.
* **Edge-Optimized Inference:** Custom TensorRT implementation with manual CUDA memory management for high-frequency (30+ FPS) depth mapping on NVIDIA Jetson platforms.
* **Vector Field Histogram (VFH) Logic:** Implements a robust "Valley Seeking" algorithm to identify safe navigable sectors within a 90° field of view.
* **Dynamic Speed Scaling:** Automatically modulates forward velocity based on "clearance" metrics, ensuring safe braking and turning maneuvers.

---

## 🏗 System Architecture



The project features two distinct pipelines to bridge the gap between simulation and deployment:

### 1. HITL (Hardware-In-The-Loop) | `pramukh_v1_hitl.py`
* **Communication:** PyMavlink via serial (`/dev/ttyUSB0`).
* **Engine:** TensorRT (FP16 Optimized) for NVIDIA Jetson.
* **Control:** Low-level MAVLink `SET_POSITION_TARGET_LOCAL_NED` commands.
* **Optimization:** Direct CUDA `host-to-device` memory copies for zero-lag perception.

### 2. SITL (Software-In-The-Loop) | `pramukh_v1.py`
* **Communication:** MAVSDK (UDP).
* **Engine:** ONNX Runtime (CPU/GPU).
* **Control:** Asynchronous high-level velocity body commands.
* **Validation:** Integrated with AirSim for realistic sensor data and physics.

---

## 🛠 Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Flight Stack** | PX4, MAVLink, MAVSDK, PyMavlink |
| **Perception** | MiDaS Small, OpenCV, TensorRT, ONNX |
| **Simulation** | Microsoft AirSim, Unreal Engine |
| **Hardware** | NVIDIA Jetson, Pixhawk Flight Controller |
| **Languages** | Python, CUDA, Bash |

---

## 📝 Navigation Logic (The "Valley" Algorithm)

The system processes the environment by partitioning the depth map into **91 sectors**:

1. **Histogram Generation:** Extracts a 30th percentile depth value for each sector to filter out sensor noise.
2. **Thresholding:** Sectors exceeding the `SAFE_THRESHOLD` (0.55 for HITL) are flagged as navigable.
3. **Valley Seeking:** The algorithm identifies the widest contiguous "valley" of safe sectors.
4. **Steering:** The drone calculates the optimal heading toward the center of the widest valley, applying a turn penalty to modulate speed during sharp maneuvers.

---

## 🚀 Getting Started

### Prerequisites
* PX4 Autopilot & AirSim configured environment.
* NVIDIA Jetson (for HITL) or standard Linux machine (for SITL).
* `pip install airsim pymavlink mavsdk onnxruntime opencv-python numpy`

### Deployment
**To run SITL (Prototyping):**
```bash
python3 Scripts/pramukh_v1.py
```
**To run HITL (Production on Jetson)::**
```bash
python3 Scripts/pramukh_v1_hitl.py
```


