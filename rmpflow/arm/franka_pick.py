# Copyright (c) 2025, The Big Brain. All rights reserved.
# REAL-TIME TRACKING: Updates Target Pose constantly until Grasp Lock.

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid, VisualSphere 
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleXFormPrim 

from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config

# --- CONFIGURATION ---
CUBE_SIZE = 0.05
GRASP_BIAS = 0.01 
HOVER_DIST = 0.08  

# POSITIONS
BIN_LOCATION = np.array([0.5, -0.4, 0.15]) 
CRUISE_POSE = np.array([0.4, 0.0, 0.5])    

# --- STATE MACHINE ---
class PickStates:
    SEARCH = 0
    HOVER = 1        # TRACKING ACTIVE
    DESCEND = 2      # TRACKING ACTIVE
    CLOSE = 3        # TRACKING LOCKED (Physical interaction)
    RETRACT = 4      
    CRUISE = 5       
    PLACE_HOVER = 6  
    PLACE_DROP = 7   
    RESET = 8        

def simulate_vision_detection(cube_quat_isaac):
    """ 
    Calculates optimal gripper orientation based on current cube rotation.
   
    """
    r = R.from_quat([cube_quat_isaac[1], cube_quat_isaac[2], cube_quat_isaac[3], cube_quat_isaac[0]])
    matrix = r.as_matrix()
    local_x, local_y, local_z = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    
    world_up = np.array([0, 0, 1])
    world_forward = np.array([1, 0, 0]) 

    axes = [(local_x, "X"), (-local_x, "-X"), (local_y, "Y"), (-local_y, "-Y"), (local_z, "Z"), (-local_z, "-Z")]

    best_normal_vec, best_normal_dot = None, -1.0
    for vec, _ in axes:
        if np.dot(vec, world_up) > best_normal_dot:
            best_normal_dot, best_normal_vec = np.dot(vec, world_up), vec

    best_tangent_vec, best_tangent_dot = None, -1.0
    for vec, _ in axes:
        if abs(np.dot(vec, best_normal_vec)) > 0.05: continue 
        if np.dot(vec, world_forward) > best_tangent_dot:
            best_tangent_dot, best_tangent_vec = np.dot(vec, world_forward), vec

    target_z = -best_normal_vec 
    target_x = best_tangent_vec
    target_y = np.cross(target_z, target_x)
    
    target_r = R.from_matrix(np.column_stack((target_x, target_y, target_z)))
    qs = target_r.as_quat()
    return np.array([qs[3], qs[0], qs[1], qs[2]]), best_normal_vec

def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    world.scene.add_default_ground_plane(prim_path="/World/defaultGroundPlane", name="default_ground_plane")

    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
    
    TCP_PRIM_PATH = "/World/Franka/panda_hand/tool_center" 
    tool_center_link = SingleXFormPrim(prim_path=TCP_PRIM_PATH, name="tool_center_link")

    high_friction_mat = PhysicsMaterial(
        prim_path="/World/Physics_Materials/HighFriction",
        name="high_friction_material",
        static_friction=1.5, dynamic_friction=1.5, restitution=0.0       
    )

    target_cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/TargetCube", 
            position=np.array([0.5, 0.0, 0.05]),
            orientation=euler_angles_to_quat(np.array([0, 0, 0])), 
            scale=np.array([CUBE_SIZE]*3), 
            color=np.array([1.0, 0.0, 0.0]),
            mass=0.1,
            physics_material=high_friction_mat
        )
    )
    
    ghost_target = world.scene.add(
        VisualSphere(prim_path="/World/GhostTargetSphere", radius=0.005, color=np.array([1.0, 1.0, 0.0]))
    )

    wall = world.scene.add(FixedCuboid(prim_path="/World/Wall", position=np.array([0.5, 0.2, 0.3]), scale=np.array([0.3, 0.05, 0.2]), color=np.array([0,0,1])))

    world.reset()
    franka.initialize()
    franka.set_joints_default_state(positions=np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4, 0.04, 0.04]))
    
    rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
    rmp_config["end_effector_frame_name"] = "right_gripper" 
    rmpflow = RmpFlow(**rmp_config) 
    rmpflow.add_obstacle(wall)
    rmpflow.add_obstacle(world.scene.get_object("default_ground_plane")) 
    rmpflow.remove_obstacle(target_cube) 
    articulation_rmpflow = ArticulationMotionPolicy(franka, rmpflow, 1.0/60.0)

    current_state = PickStates.SEARCH
    gripper_target = 0.04 
    wait_counter = 0      
    
    # Persistent targets
    target_pos = CRUISE_POSE
    target_quat = np.array([0, 1, 0, 0])
    
    # Variables to store the "Locked" pose when we start grasping
    locked_pos = None
    locked_quat = None

    print("🚀 REAL-TIME VISION ACTIVE. Try moving the cube!")

    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_playing():
            # --- 1. REAL-TIME DATA ---
            # We read the cube's position EVERY FRAME
            cube_pos, raw_quat = target_cube.get_world_pose()
            ee_pos, ee_rot = tool_center_link.get_world_pose()

            # Calculate vision solution EVERY FRAME (unless locked)
            # This ensures if the cube bounces, we know where the new "Top" is immediately.
            vision_quat, face_normal = simulate_vision_detection(raw_quat)
            
            # --- 2. STATE MACHINE ---
            if current_state == PickStates.SEARCH:
                # Transition immediately to active tracking
                current_state = PickStates.HOVER

            elif current_state == PickStates.HOVER:
                # ACTIVE TRACKING: Target follows the cube dynamically
                target_pos = cube_pos + (face_normal * HOVER_DIST)
                target_quat = vision_quat
                
                # If the cube is moving (bouncing), target_pos will jump. 
                # RMPFlow will smooth this out, but it will chase the cube.
                if np.linalg.norm(ee_pos - target_pos) < 0.02:
                    current_state = PickStates.DESCEND

            elif current_state == PickStates.DESCEND:
                # ACTIVE TRACKING: Still chasing the cube as we dive in
                target_pos = cube_pos + (face_normal * GRASP_BIAS)
                target_quat = vision_quat

                if np.linalg.norm(ee_pos - target_pos) < 0.01:
                    wait_counter = 0
                    # --- LOCK TARGET ---
                    # We stop tracking now. The hand is in position. 
                    # If we keep tracking, closing fingers might wiggle the cube and confuse the robot.
                    locked_pos = target_pos
                    locked_quat = target_quat
                    current_state = PickStates.CLOSE

            elif current_state == PickStates.CLOSE:
                # USE LOCKED TARGET (Ignore cube movement caused by gripper)
                target_pos = locked_pos
                target_quat = locked_quat
                gripper_target = 0.00 
                
                wait_counter += 1
                if wait_counter > 50: 
                    current_state = PickStates.RETRACT

            elif current_state == PickStates.RETRACT:
                # Lift Straight Up (World Z)
                target_pos = np.array([locked_pos[0], locked_pos[1], 0.45])
                target_quat = locked_quat # Maintain grasp orientation
                
                if ee_pos[2] > 0.40:
                    current_state = PickStates.CRUISE

            elif current_state == PickStates.CRUISE:
                # "Bridge" Logic: Travel to Neutral & Unwind Wrist
                target_pos = CRUISE_POSE
                target_quat = euler_angles_to_quat(np.array([np.pi, 0, 0]))
                
                if np.linalg.norm(ee_pos - target_pos) < 0.08:
                    current_state = PickStates.PLACE_HOVER

            elif current_state == PickStates.PLACE_HOVER:
                target_pos = BIN_LOCATION + np.array([0, 0, 0.1])
                target_quat = euler_angles_to_quat(np.array([np.pi, 0, 0]))
                
                if np.linalg.norm(ee_pos - target_pos) < 0.05:
                    wait_counter = 0
                    current_state = PickStates.PLACE_DROP

            elif current_state == PickStates.PLACE_DROP:
                target_pos = BIN_LOCATION + np.array([0, 0, 0.1])
                gripper_target = 0.04 # Open
                
                wait_counter += 1
                if wait_counter > 40:
                    current_state = PickStates.RESET

            elif current_state == PickStates.RESET:
                # Respawn Logic
                rand_x = np.random.uniform(0.4, 0.6)
                rand_y = np.random.uniform(-0.2, 0.2)
                rand_rot = euler_angles_to_quat(np.array([0, np.random.uniform(0, 3), np.random.uniform(0, 3)]))
                
                target_cube.set_world_pose(position=np.array([rand_x, rand_y, 0.05]), orientation=rand_rot)
                # KILL ENERGY: Stop it from flying away after teleport
                target_cube.set_linear_velocity(np.array([0,0,0]))
                target_cube.set_angular_velocity(np.array([0,0,0]))
                
                current_state = PickStates.SEARCH

            # --- RMPFLOW UPDATE ---
            rmpflow.update_world() 
            rmpflow.set_end_effector_target(target_pos, target_quat)
            ghost_target.set_world_pose(position=target_pos, orientation=target_quat) 

            arm_action = articulation_rmpflow.get_next_articulation_action()
            full_joint_positions = np.append(arm_action.joint_positions, [gripper_target, gripper_target])
            franka.apply_action(ArticulationAction(joint_positions=full_joint_positions))

    simulation_app.close()

if __name__ == "__main__":
    main()