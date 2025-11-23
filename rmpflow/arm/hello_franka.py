# Copyright (c) 2025, The Big Brain. All rights reserved.

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from scipy.spatial.transform import Rotation as R  # Standard in Isaac Sim

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid, FixedCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat

# Robot Import
from isaacsim.robot.manipulators.examples.franka import Franka

# RMPFlow Imports
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config


def simulate_vision_detection(cube_quat_isaac):
    """
    Simulates a Smart Vision System.
    1. Detects which face is 'Top' (Up-vector snapping).
    2. Detects the best 'Forward' edge to grasp (90-degree Z-snapping).
       This exploits the 4-way symmetry of a square cube to minimize wrist rotation.
    """
    # Convert Isaac Quaternion (w, x, y, z) to Scipy (x, y, z, w)
    r = R.from_quat([cube_quat_isaac[1], cube_quat_isaac[2], cube_quat_isaac[3], cube_quat_isaac[0]])
    matrix = r.as_matrix()

    # The Cube's Local Axes in World Space
    local_x = matrix[:, 0]
    local_y = matrix[:, 1]
    local_z = matrix[:, 2]

    # Defined Vectors
    world_up = np.array([0, 0, 1])
    world_forward = np.array([1, 0, 0]) # The robot's preferred "Forward" direction

    # All 6 possible directions from the cube's perspective
    axes = [
        (local_x, "X"), (-local_x, "-X"),
        (local_y, "Y"), (-local_y, "-Y"),
        (local_z, "Z"), (-local_z, "-Z")
    ]

    # --- STEP 1: FIND THE NEW TOP (Face Normal) ---
    best_normal_vec = None
    best_normal_dot = -1.0
    
    for vec, name in axes:
        dot = np.dot(vec, world_up)
        if dot > best_normal_dot:
            best_normal_dot = dot
            best_normal_vec = vec

    # --- STEP 2: FIND THE BEST GRIPPER ORIENTATION (Tangent/Yaw Snap) ---
    # We want to align the Gripper X (fingers) with the cube edge that is 
    # closest to World Forward. This handles the 90-degree symmetry.
    
    best_tangent_vec = None
    best_tangent_dot = -1.0

    for vec, name in axes:
        # 1. Ensure this vector is orthogonal to the Up vector (it must be a side edge)
        # We use a dot product check. If dot is near 0, it is perpendicular.
        if abs(np.dot(vec, best_normal_vec)) > 0.05:
            continue

        # 2. Check alignment with World Forward to minimize wrist twist
        dot = np.dot(vec, world_forward)
        if dot > best_tangent_dot:
            best_tangent_dot = dot
            best_tangent_vec = vec

    # --- STEP 3: CONSTRUCT TARGET MATRIX ---
    # Gripper Z points INTO the box (opposite of Normal)
    target_z = -best_normal_vec 
    
    # Gripper X aligns with the best chosen edge (Snap 90)
    target_x = best_tangent_vec
    
    # Gripper Y is the cross product
    target_y = np.cross(target_z, target_x)
    
    # Create Rotation Matrix
    target_matrix = np.column_stack((target_x, target_y, target_z))
    
    # Convert back to Quaternion for Isaac Sim
    target_r = R.from_matrix(target_matrix)
    quat_scipy = target_r.as_quat()
    
    return np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])

def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    world.scene.add_default_ground_plane()

    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

    # --- SCENARIO: 90 DEGREE FLIP ---
    # This cube is rotated 90 degrees on Y. Ideally, the robot should NOT twist 90 degrees.
    # It should realize the "Side" is the "New Top" and pick it normally.
    target_cube = world.scene.add(
        VisualCuboid(
            prim_path="/World/TargetCube",
            name="target_cube",
            position=np.array([0.5, 0.0, 0.5]),
            orientation=euler_angles_to_quat(np.array([0, np.pi/2 + 0.2, np.pi/4])), # 90 deg flip + slight wobble + spin
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([1.0, 0.0, 0.0]),
        )
    )

    wall = world.scene.add(
        FixedCuboid(
            prim_path="/World/Wall",
            name="wall",
            position=np.array([0.5, 0.2, 0.3]),
            scale=np.array([0.3, 0.05, 0.2]),
            color=np.array([0.0, 0.0, 1.0]),
        )
    )

    world.reset()
    franka.initialize()
    franka.set_joints_default_state(positions=np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4, 0.04, 0.04]))

    # RMPFlow
    rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
    rmp_config["end_effector_frame_name"] = "right_gripper"
    rmpflow = RmpFlow(**rmp_config)
    rmpflow.add_obstacle(wall)
    
    articulation_rmpflow = ArticulationMotionPolicy(franka, rmpflow, 1.0/60.0)

    print("🚀 Vision Simulator: Detecting 'Top' face automatically...")

    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_playing():
            # 1. Get True Physics Pose
            cube_pos, raw_quat = target_cube.get_world_pose()

            # 2. SIMULATE VISION
            # This function determines which face is up and creates a "Down" target for it.
            vision_quat = simulate_vision_detection(raw_quat)

            rmpflow.update_world()
            
            # 3. Send Target
            rmpflow.set_end_effector_target(
                target_position=cube_pos,
                target_orientation=vision_quat 
            )

            action = articulation_rmpflow.get_next_articulation_action()
            franka.apply_action(action)

    simulation_app.close()

if __name__ == "__main__":
    main()