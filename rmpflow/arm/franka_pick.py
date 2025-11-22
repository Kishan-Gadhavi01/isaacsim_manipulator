# Copyright (c) 2025, The Big Brain. All rights reserved.
# This script executes a vision-driven, collision-free pick-and-place task using Franka and RMPFlow.

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from scipy.spatial.transform import Rotation as R
import os 

from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid, VisualSphere 
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleXFormPrim 
from isaacsim.core.utils.stage import open_stage

from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config

# --- CONFIGURATION ---
USD_PATH = "/home/popoy/robotics/sim/isaacsim_manipulator/rmpflow/arm/base.usd"
GRASP_BIAS = 0.01 
CUBE_SIZE = 0.05
# ---------------------

# --- STATE MACHINE ENUMS ---
class PickStates:
    HOVER = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3

def simulate_vision_detection(cube_quat_isaac):
    """ 
    Simulates a CV system: Calculates optimal gripper orientation and surface normal.
    Assumes standard Isaac Sim (w, x, y, z) quaternion format for input.
    """
    # Convert Isaac Sim (w,x,y,z) to SciPy (x,y,z,w)
    r = R.from_quat([cube_quat_isaac[1], cube_quat_isaac[2], cube_quat_isaac[3], cube_quat_isaac[0]])
    matrix = r.as_matrix()
    local_x, local_y, local_z = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    
    world_up = np.array([0, 0, 1])
    world_forward = np.array([1, 0, 0]) 

    axes = [(local_x, "X"), (-local_x, "-X"), (local_y, "Y"), (-local_y, "-Y"), (local_z, "Z"), (-local_z, "-Z")]

    # Find the face normal most aligned with World Up (for top grasp)
    best_normal_vec, best_normal_dot = None, -1.0
    for vec, _ in axes:
        if np.dot(vec, world_up) > best_normal_dot:
            best_normal_dot, best_normal_vec = np.dot(vec, world_up), vec

    # Find the tangent vector perpendicular to normal and aligned with World Forward
    best_tangent_vec, best_tangent_dot = None, -1.0
    for vec, _ in axes:
        if abs(np.dot(vec, best_normal_vec)) > 0.05: continue # Must be orthogonal to normal
        if np.dot(vec, world_forward) > best_tangent_dot:
            best_tangent_dot, best_tangent_vec = np.dot(vec, world_forward), vec

    # Construct the target orientation matrix
    target_z = -best_normal_vec # Gripper Z-axis points into the object
    target_x = best_tangent_vec
    target_y = np.cross(target_z, target_x)
    
    target_r = R.from_matrix(np.column_stack((target_x, target_y, target_z)))
    
    # Return quaternion in (w, x, y, z) Isaac Sim format
    qs = target_r.as_quat()
    return np.array([qs[3], qs[0], qs[1], qs[2]]), best_normal_vec

def main():
    # --- 2. LOAD THE CUSTOM STAGE ---
    print(f"Loading custom stage: {USD_PATH}...")
    open_stage(USD_PATH) 

    # --- 3. INITIALIZE THE WORLD AND SCENE ---
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    
    # ADD DEFAULT GROUND PLANE SAFELY (in case base.usd doesn't have one)
    GROUND_PRIM_PATH = "/World/defaultGroundPlane"
    ground_plane_obstacle = world.scene.add_default_ground_plane(
        prim_path=GROUND_PRIM_PATH, name="default_ground_plane"
    )

    # --- 4. ROBOT, TARGET, AND OBSTACLE SETUP ---
    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
    
    # CRITICAL: Get the true TCP link object for accurate end-effector observation
    TCP_PRIM_PATH = "/World/Franka/panda_hand/tool_center" 
    tool_center_link = SingleXFormPrim(prim_path=TCP_PRIM_PATH, name="tool_center_link")
    
    high_friction_mat = PhysicsMaterial(
        prim_path="/World/Physics_Materials/HighFriction",
        name="high_friction_material",
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0       
    )

    # Add the dynamic cube to be picked
    target_cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/TargetCube", 
            position=np.array([0.5, 0.0, 0.5]),
            orientation=euler_angles_to_quat(np.array([0, np.pi/2 + 0.2, np.pi/4])), 
            scale=np.array([CUBE_SIZE]*3), 
            color=np.array([1.0, 0.0, 0.0]),
            mass=0.1,
            physics_material=high_friction_mat
        )
    )
    
    # Add an obstacle wall for RMPFlow to avoid
    wall = world.scene.add(FixedCuboid(prim_path="/World/Wall", position=np.array([0.5, 0.2, 0.3]), scale=np.array([0.3, 0.05, 0.2]), color=np.array([0,0,1])))

    # GHOST SPHERE SETUP (Visual debugging for TRUE TCP)
    GHOST_RADIUS = 0.005 
    ghost_target = world.scene.add(
        VisualSphere(
            prim_path="/World/GhostTargetSphere",
            position=np.array([0.0, 0.0, 0.0]), 
            radius=GHOST_RADIUS,
            color=np.array([1.0, 1.0, 0.0]), 
            name="ghost_sphere",
        )
    )
    
    # --- 5. INITIALIZE AND SET ROBOT HOME POSE ---
    world.reset()
    
    # CRITICAL: Initialize the Franka and set its home pose AFTER world.reset()
    franka.initialize()
    franka.set_joints_default_state(positions=np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4, 0.04, 0.04]))

    # --- 6. RMPFLOW SETUP ---
    rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
    rmp_config["end_effector_frame_name"] = "right_gripper" 
    rmpflow = RmpFlow(**rmp_config) 
    
    # ✅ THE CRITICAL FIX: Register the ground plane with RMPFlow for avoidance
    rmpflow.add_obstacle(ground_plane_obstacle)
    
    # Register the wall obstacle
    rmpflow.add_obstacle(wall)
    
    # Ensure the target object is NOT registered as an obstacle
    rmpflow.remove_obstacle(target_cube) 
    
    # Final Motion Policy Wrapper
    articulation_rmpflow = ArticulationMotionPolicy(franka, rmpflow, 1.0/60.0)

    # --- 7. STATE MACHINE VARIABLES ---
    current_state = PickStates.HOVER
    gripper_target = 0.04 
    wait_counter = 0      
    
    print("🚀 Simulation Started. RMPFlow Ground Collision Avoidance is ACTIVE.")

    # --- 8. THE MAIN SIMULATION LOOP ---
    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_playing():
            # PERCEPTION: Get the current pose of the cube
            cube_pos, raw_quat = target_cube.get_world_pose()
            vision_quat, face_normal = simulate_vision_detection(raw_quat)
            
            # OBSERVATION: Get the current pose of the actual end-effector/TCP
            ee_pos, observed_quat = tool_center_link.get_world_pose()
            
            # --- STATE MACHINE LOGIC ---
            if current_state == PickStates.HOVER:
                # Target Position: Hover safely above the cube face
                target_pos = cube_pos + (face_normal * CUBE_SIZE/2)
                gripper_target = 0.04 
                dist_error = np.linalg.norm(ee_pos - target_pos) 
                
                if world.current_time_step_index % 60 == 0:
                     print(f"State: HOVER | TRUE TCP Error: {dist_error:.4f}m")

                if dist_error < 0.01:
                    print(f"✅ Target Reached (Err: {dist_error:.3f}m). Switching to DESCEND.")
                    current_state = PickStates.DESCEND
                
            elif current_state == PickStates.DESCEND:
                # Move closer to the cube face for grasping
                target_pos = cube_pos + (face_normal * GRASP_BIAS)
                gripper_target = 0.04 
                
                dist_to_cube = np.linalg.norm(ee_pos - target_pos)
                
                if dist_to_cube < 0.065:
                    print("⬇️ Descended. Switching to CLOSE.")
                    current_state = PickStates.CLOSE
                    wait_counter = 0

            elif current_state == PickStates.CLOSE:
                # Hold position and close the gripper
                target_pos = cube_pos + (face_normal * GRASP_BIAS)
                gripper_target = 0.00
                
                wait_counter += 1
                if wait_counter > 60: # Wait for 1 second (60 steps)
                    print("✊ Gripper closed. Switching to LIFT.")
                    current_state = PickStates.LIFT

            elif current_state == PickStates.LIFT:
                # Lift the cube straight up
                target_pos = cube_pos + np.array([0, 0, 0.3]) 
                gripper_target = 0.00 
                
                if ee_pos[2] > 0.45:
                    pass
                            

            # RMPFlow Execution
            rmpflow.update_world() 
            rmpflow.set_end_effector_target(target_pos, vision_quat)
            
            # VISUAL DEBUG: Update Ghost Sphere to show where the robot thinks its TCP is
            ghost_target.set_world_pose(position=ee_pos, orientation=observed_quat) 

            # Apply actions
            arm_action = articulation_rmpflow.get_next_articulation_action()
            full_joint_positions = np.append(arm_action.joint_positions, [gripper_target, gripper_target])
            franka.apply_action(ArticulationAction(joint_positions=full_joint_positions))

    simulation_app.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        simulation_app.close()