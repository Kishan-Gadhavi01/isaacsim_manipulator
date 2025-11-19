# 🤖 Isaac Sim Manipulator Workflows

### *Advanced Robotics, Vision-Guided Control & Autonomous Manipulation in Isaac Sim 4.5.0*

<p align="center">
  <img src="https://img.shields.io/badge/Isaac%20Sim-4.5.0-76B900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?logo=opencv" />
  <img src="https://img.shields.io/badge/Franka%20Panda-Robotics-orange" />
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen" />
</p>

---


<div align="center">
  <img src="./media/sorter.gif" width="100%" style="border-radius: 8px;" />
</div>

---

# 📘 Overview

This repository showcases a collection of **advanced robotic manipulation workflows** developed entirely in **NVIDIA Isaac Sim 4.5.0**.

* Physics-accurate simulation
* Inverse kinematics
* RGB-D perception
* State machines
* High-level robotic controllers
* Safety-aware grasp policies

The primary platform used throughout this work is the **Franka Emika Panda** manipulator.


---

# 📁 Project Structure

```text
.
├── PickPlaceController
│   └── arm
│       ├── hello_pick_place.py       # Foundational manipulation: kinematics, solver tuning
│       └── RGB_cube_sorter.py        # Vision-guided autonomous RGB sorter
├── Sensor                            
│   └── hello_cam.py                  # Camera tests, RGB-D processing, OpenCV pipeline
├── media                             
│   └── Sorter.gif 
└── README.md
```

---

# 🚀 Featured Projects

## 🔴 1. Autonomous RGB Cube Sorter

**File:** `PickPlaceController/arm/RGB_cube_sorter.py`

A fully autonomous RGB-based manipulation system integrating perception, control, and high-level planning.

### 🧠 **Core Capabilities**

* **Vision Pipeline (OpenCV + Depth):**

  * RGB-D fusion
  * Pixel → World coordinate transformation
  * HSV color classification
  * ROI masking (“tunnel vision” to avoid background clutter)
* **State Machine Architecture:**
  `SEARCH → PLAN → PICK → PLACE → RESET`
* **Stable Pick & Place Logic:**

  * Smart gripper yaw alignment
  * Descent clamping to avoid table collisions
  * Controlled release height
* **Bin Placement Intelligence:**

  * Randomized drop-off to prevent stacking collisions
  * Color-coded bin separation
* **Physics Optimization:**

  * 64 solver iterations for stable contacts
  * Custom articulation pose filtering

This script represents a **practical Sim-to-Real pipeline**, suitable for downstream deployment.

---

## 🟦 2. Foundational Pick & Place

**File:** `hello_pick_place.py`

This script builds the essential understanding required for more advanced robotics:

* USD stage creation
* Physics scene configuration
* Direct articulation control
* Gripper open/close tuning
* Basic target-based IK using PickPlaceController

A clean introduction to Isaac Sim’s manipulation framework.

---

## 🟩 3. RGB-D Vision Pipeline

**File:** `hello_cam.py`

A sandbox for experimenting with perception and sensor simulation.

### ✔ Includes:

* Synthetic **RealSense-like RGB-D** camera
* Intrinsic + extrinsic matrix math
* Converting USD synthetic data → OpenCV images
* Depth visualization and calibration
* Noise-robust color detection

This module lays the foundation for the sorter’s perception subsystem.

---

# 🛠 Technology Stack

| Category    | Tools                                          |
| ----------- | ---------------------------------------------- |
| Simulation  | **Isaac Sim 4.5.0**                            |
| Language    | **Python 3.10**                                |
| Robotics    | Franka Panda Articulation, PickPlaceController |
| Vision      | OpenCV, NumPy, Depth Mapping                   |
| Control     | Inverse Kinematics, Rigid Body Dynamics, FSM   |
| Development | VS Code, Omniverse Kit                         |

---

# ⚙️ Installation & Usage

### 1. Prerequisites

* Isaac Sim **4.5.0** installed
* RTX-enabled GPU
* Python 3.10
* Ubuntu 22.04.5 LTS
* ROS2 Humble (Future extension)

### 2. Run the Autonomous Sorter

```bash
./python.sh path/to/repo/PickPlaceController/arm/RGB_cube_sorter.py
```

### 3. Run Vision Pipeline Test

```bash
./python.sh path/to/repo/Sensor/hello_cam.py
```

---

# 🔮 Future plan

* ✔ Basic pick & place logic
* ✔ RGB-D perception pipeline
* ✔ Finite State Machine for autonomy
* ✔ Physics-accurate grasping logic
* ⬜ **ROS 2 bridge + Real Franka deployment**
* ⬜ **Domain Randomization (lighting, textures)**
* ⬜ **Reinforcement Learning extension**

---

# 📜 License

This project is available for **education and research**.

