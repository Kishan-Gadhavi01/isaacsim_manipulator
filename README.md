


# 🤖 Isaac Sim Manipulator Workflows

### *RMPFlow Motion Generation, Closed-Loop Vision & Dynamic Interception*

<p align="center">
  <img src="https://img.shields.io/badge/Isaac%20Sim-4.5.0-76B900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/RMPFlow-Motion%20Policy-purple" />
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?logo=opencv" />
  <img src="https://img.shields.io/badge/Franka%20Panda-Robotics-orange" />
</p>

<div align="center">
  <img src="./media/sorter.gif" width="100%" style="border-radius: 8px;" />
</div>
<div align="center">
  <img src="./media/live.pick.gif" width="100%" style="border-radius: 8px;" />
</div>
---

# 📘 Executive Summary

This repository houses advanced **Robotic Manipulation Workflows** engineered in **NVIDIA Isaac Sim 4.5.0**.

The project implements **intelligent, sensory-guided autonomy**, bridging the gap between **Synthetic Perception** and **Dynamic Control**. It contrasts high-level task planners with fluid, reactive motion generation.

Key achievements include **sim-to-real-capable Computer Vision** pipelines and solving the **Moving Target Interception** problem using **RMPFlow** for smooth, collision-aware trajectory optimization.

---

# 🛠️ Technical Stack & Keywords

| Domain                | Technologies & Concepts                                  |
| --------------------- | -------------------------------------------------------- |
| **Simulation Core**   | NVIDIA Isaac Sim 4.5.0, USD, PhysX 5, Omniverse Kit      |
| **Motion Generation** | RMPFlow, Lula Kinematics, Dynamic Obstacle Avoidance     |
| **Control Logic**     | PickPlaceController, Visual Servoing, Velocity Sync, FSM |
| **Perception**        | OpenCV, RGB-D, Depth Deprojection, ROI Masking           |
| **Hardware (Sim)**    | Franka Emika Panda, Parallel Gripper, RGB-D Sensors      |

---

# 🚀 Featured Modules

## 🟠 1. Dynamic Conveyor Belt Interception

**File:** `rmpflow/arm/conveyor_belt.py`

A complex kinematic challenge requiring interception of a moving object.

### 🧠 Engineering Highlights

* **RMPFlow Integration** for smooth, collision-aware trajectories
* **Velocity Synchronization**: end-effector matches conveyor speed
* **Predictive Tracking**: computes interception point downstream
* **Finite State Machine:**
  `INTERCEPT → SYNC → GRASP → LIFT`

---

## 🔴 2. Intelligent RGB-D Sorting System

**File:** `PickPlaceController/arm/RGB_cube_sorter.py`

A fully autonomous perception-based pick-and-place pipeline.

### 🧠 Engineering Highlights

* **Synthetic Vision Pipeline**

  * Depth projection from 2D → 3D using inverse intrinsics
  * ROI masking & occlusion filtering
* **Geometric Grasp Heuristics**

  * Auto-select grasp angle (0° or 90°)
  * Collision-aware orientation logic

---

## 🟦 3. Decoupled & Relative Motion

**File:** `rmpflow/arm/decoupled_franka.py`

Explores **whole-body control** concepts using relative coordinates.
Useful for robotic arms mounted on mobile bases.

---

# 📁 Project Structure

```text
.
├── PickPlaceController     # High-Level Task Logic
│   └── arm
│       ├── hello_pick_place.py       # Kinematics Basics
│       └── RGB_cube_sorter.py        # [MAIN] Visual Servoing & Grasp Heuristics
├── rmpflow                 # Advanced Motion Generation (Lula/Riemannian)
│   └── arm
│       ├── base.usd                  # Custom USD Stage
│       ├── conveyor belt.py          # [MAIN] Dynamic Moving Target Interception
│       ├── conveyor.usd              # Conveyor Asset
│       ├── decoupled_franka.py       # Relative Frame Control
│       ├── franka_pick.py            # RMPFlow Pick Logic
│       └── hello_rmpflow.py          # RMPFlow Initialization
├── Sensor                  # Perception Sandbox
│   └── hello_cam.py                  # Synthetic Data & OpenCV Pipeline
├── media                   
│   └── Sorter.gif 
└── README.md
```

---

# ⚙️ Installation & Execution

### Prerequisites

* Ubuntu 22.04 LTS
* NVIDIA RTX GPU
* Isaac Sim **4.5.0**
* Python dependencies:

  ```
  opencv-python
  numpy
  scipy
  ```

### Run Dynamic Conveyor Interception

```bash
./python.sh path/to/repo/rmpflow/arm/conveyor_belt.py
```

### Run RGB-D Sorting System

```bash
./python.sh path/to/repo/PickPlaceController/arm/RGB_cube_sorter.py
```

---

# 🔮 Roadmap

* [x] Visual Servoing
* [x] RMPFlow Integration
* [x] Dynamic Moving-Target Interception
* [ ] Domain Randomization
* [ ] Mobile Manipulation

---

# 📜 License

Released for **Educational & Research Use**.

