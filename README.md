

# 🤖 Isaac Sim Manipulator Workflows
### *Closed-Loop Visual Servoing, Task-Level Control & Sim2Real Perception*

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

# 📘 Executive Summary

This repository houses advanced **Robotic Manipulation Workflows** engineered within **NVIDIA Isaac Sim 4.5.0**. 

Moving beyond simple playback, this project implements **intelligent, sensory-guided autonomy**. It leverages the **Omniverse Kit SDK** to bridge the gap between **Synthetic Perception** (RGB-D) and **Task-Level Control**. 

The core focus is on **Sim2Real transferability**: developing robust grasp heuristics, occlusion-aware logic, and vision pipelines that mirror physical deployment constraints.

---

# 🛠️ Technical Stack & Keywords

| Domain | Technologies & Concepts |
| :--- | :--- |
| **Simulation Core** | **NVIDIA Isaac Sim 4.5.0**, **USD** (Universal Scene Description), **PhysX 5** (Rigid Body Dynamics), Omniverse Kit. |
| **Motion Control** | **PickPlaceController** (High-Level Task Abstraction), **Inverse Kinematics (IK)**, Cartesian Interpolation, End-Effector Pose Control. |
| **Perception** | **OpenCV**, Pinhole Camera Model, **Depth Deprojection** (2D-to-3D), Intrinsic/Extrinsic Calibration, HSV Filtering, Synthetic Data Generation. |
| **Control Logic** | Finite State Machines (**FSM**), **Visual Servoing** (Closed-Loop), Geometric Grasp Heuristics, Dynamic Collision Avoidance. |
| **Hardware (Sim)** | **Franka Emika Panda** (7-DOF), Parallel Jaw Gripper, RGB-D Sensors. |

---

# 🚀 Featured Modules

## 🔴 1. Intelligent RGB-D Sorting System
**File:** `PickPlaceController/arm/RGB_cube_sorter.py`

A fully autonomous loop demonstrating **Hand-Eye Coordination**. The system does not use ground-truth hacks; it "sees" the world through a simulated camera.

### 🧠 Engineering Highlights:
* **Synthetic Vision Pipeline:**
    * Real-time **Depth Injection**: Converts 2D pixels $(u,v)$ + Depth $(d)$ into 3D World Coordinates $(x,y,z)$ using inverse intrinsic matrix projection.
    * **Occlusion Filtering**: Dynamic ROI masking to isolate the "Spawn Zone" from the "Bin Zone."
* **Geometric Grasp Heuristics (The "Brain"):**
    * Implements a custom **Collision Logic Gate** that analyzes neighbor geometry.
    * Automatically selects between **Standard Grip ($0^\circ$)** or **Rotated Grip ($90^\circ$)** based on lateral vs. longitudinal clearance.
* **Robust Control Architecture:**
    * **State Machine:** `SEARCH` $\to$ `PLAN` $\to$ `PICK` $\to$ `PLACE` $\to$ `RESET`.
    * **Sensor Fusion:** Monitors gripper width and end-effector height during transport to detect **object slippage** and trigger auto-recovery.

## 🟦 2. Kinematics & Physics Foundation
**File:** `hello_pick_place.py`

The foundational implementation of the **PickPlaceController**, utilizing Isaac Sim's high-level abstraction for multi-phase manipulation.
* Setup of **SingleArticulation** wrappers.
* Tuning of PhysX solver iterations (64 steps) for stable contact dynamics.
* Implementation of basic approach/lift heuristics.

## 🟩 3. Synthetic Sensor Sandbox
**File:** `hello_cam.py`

A standalone module for **Simulated Sensor integration**.
* Configuring `omni.isaac.sensor.Camera` prims.
* Visualizing Depth buffers and Point Clouds.
* Validating **Intrinsic/Extrinsic matrices** for accurate computer vision.

---

# 📁 Project Structure

```text
.
├── PickPlaceController
│   └── arm
│       ├── hello_pick_place.py       # RMPFlow & Kinematics Basics
│       └── RGB_cube_sorter.py        # [MAIN] Visual Servoing & Grasp Heuristics
├── Sensor                            
│   └── hello_cam.py                  # Synthetic Data & OpenCV Pipeline
├── media                             
│   └── Sorter.gif 
└── README.md
```

-----

# ⚙️ Installation & Execution

### Prerequisites

  * **OS:** Ubuntu 22.04 LTS
  * **GPU:** NVIDIA RTX Series (RTX 3060 or higher recommended)
  * **Software:** NVIDIA Isaac Sim 4.5.0
  * **Dependencies:** `opencv-python`, `numpy` (bundled with Isaac Sim python)

### Running the Autonomous Sorter

Execute via the Isaac Sim python wrapper to ensure access to Omniverse kit extensions:

```bash
#run
 ./python.sh path/to/repo/PickPlaceController/arm/RGB_cube_sorter.py
```

-----

# 🔮 Roadmap

  * [x] **Visual Servoing:** Closed-loop pick and place via RGB-D.
  * [x] **Grasp Logic:** Geometric heuristic for clutter management.
  * [ ] **Domain Randomization:** Varying lighting/texture for robust ML training.
  * [ ] **ROS2 Bridge:** Publishing camera frames to `/camera/rgb` and subscribing to `/joint_states`.
  * [ ] **Reinforcement Learning:** Porting the task to `OmniIsaacGymEnvs`.

-----

# 📜 License

This project is released for **Educational and Research Use**.

