# Copyright (c) 2025, The Big Brain. All rights reserved.
# FULL RELATIVE FRAME PICK & LIFT

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, VisualSphere
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleXFormPrim 

from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config

# --- CONFIG (RELATIVE TO ROBOT BASE) ---
# These define where things happen relative to the robot's body
LIFT_HEIGHT_REL = 0.50      # Lift 0.5m up from the pick point
HOME_POS_REL    = np.array([0.5, 0.0, 0.5]) # "Home" is 0.5m forward, 0.5m up
SPAWN_BOX_X     = (0.4, 0.6) # Spawn cubes 40-60cm in front of robot
SPAWN_BOX_Y     = (-0.2, 0.2)

# --- HELPER: LOCAL TO WORLD ---
def local_to_world(robot_pos, robot_quat, local_pos):
    """ Transforms a point from Robot Frame to World Frame. """
    # 1. Convert Quaternion to Rotation Matrix
    rot_matrix = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]).as_matrix()
    
    # 2. Rotate the local vector
    rotated_offset = rot_matrix @ local_pos
    
    # 3. Add the Robot's World Position
    world_pos = robot_pos + rotated_offset
    return world_pos

# --- STATE MACHINE ---
class PickStates:
    HOVER = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    RESET = 4

def simulate_vision_detection(cube_quat_isaac):
    """ Keeps gripper aligned with cube orientation. """
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
    # 1. WORLD SETUP
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    world.scene.add_default_ground_plane()

    # 2. ADD ROBOT
    # Position is (0,0,0) initially, but you can drag it later!
    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka", position=np.array([0.0, 0.0, 0.0])))
    
    TCP_PRIM_PATH = "/World/Franka/panda_hand/tool_center" 
    tool_center_link = SingleXFormPrim(prim_path=TCP_PRIM_PATH, name="tool_center_link")

    # 3. ADD OBJECT (High Friction)
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
            scale=np.array([0.05]*3), 
            color=np.array([1.0, 0.0, 0.0]),
            mass=0.1,
            physics_material=high_friction_mat
        )
    )
    
    ghost_target = world.scene.add(VisualSphere(prim_path="/World/GhostTargetSphere", radius=0.005, color=np.array([1, 1, 0])))

    world.reset()
    franka.initialize()
    franka.set_joints_default_state(positions=np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4, 0.04, 0.04]))
    
    # 4. RMPFLOW CONFIG
    rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
    rmp_config["end_effector_frame_name"] = "right_gripper" 
    rmpflow = RmpFlow(**rmp_config) 
    rmpflow.add_obstacle(world.scene.get_object("default_ground_plane")) 
    rmpflow.remove_obstacle(target_cube) 
    
    articulation_rmpflow = ArticulationMotionPolicy(franka, rmpflow, 1.0/60.0)

    # Loop Variables
    current_state = PickStates.RESET # Start with Reset to handle spawning
    gripper_target = 0.04 
    wait_counter = 0      
    locked_pos, locked_quat = None, None

    print("🚀 Sim Started. You can drag the robot around now!")

    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_playing():
            # --- STEP 1: UPDATE ROBOT POSE FOR RMPFLOW ---
            # This is the critical fix. It tells the controller where the robot body is.
            robot_pos, robot_quat = franka.get_world_pose()
            rmpflow.set_robot_base_pose(robot_pos, robot_quat)

            # --- STEP 2: OBSERVATIONS ---
            cube_pos, raw_quat = target_cube.get_world_pose()
            ee_pos, ee_rot = tool_center_link.get_world_pose()
            vision_quat, face_normal = simulate_vision_detection(raw_quat)

            # --- STEP 3: STATE MACHINE ---
            if current_state == PickStates.HOVER:
                target_pos = cube_pos + (face_normal * 0.08)
                target_quat = vision_quat
                gripper_target = 0.04
                
                if np.linalg.norm(ee_pos - target_pos) < 0.02:
                    current_state = PickStates.DESCEND

            elif current_state == PickStates.DESCEND:
                target_pos = cube_pos + (face_normal * 0.01)
                target_quat = vision_quat
                
                if np.linalg.norm(ee_pos - target_pos) < 0.01:
                    wait_counter = 0
                    locked_pos, locked_quat = target_pos, target_quat
                    current_state = PickStates.CLOSE

            elif current_state == PickStates.CLOSE:
                target_pos, target_quat = locked_pos, locked_quat
                gripper_target = 0.00 
                wait_counter += 1
                if wait_counter > 40: 
                    current_state = PickStates.LIFT

            elif current_state == PickStates.LIFT:
                # Lift 0.5m UP (Relative to World Up)
                target_pos = np.array([locked_pos[0], locked_pos[1], locked_pos[2] + LIFT_HEIGHT_REL])
                target_quat = locked_quat 
                gripper_target = 0.00
                
                # Check if we reached height
                if ee_pos[2] >= (locked_pos[2] + LIFT_HEIGHT_REL - 0.05):
                    print("✅ Lift Complete. Resetting...")
                    current_state = PickStates.RESET
                    wait_counter = 0

            elif current_state == PickStates.RESET:
                # 1. Move Arm to "Home" (Relative to Robot Base)
                home_world = local_to_world(robot_pos, robot_quat, HOME_POS_REL)
                
                target_pos = home_world
                target_quat = robot_quat 
                gripper_target = 0.04 
                
                wait_counter += 1
                if wait_counter == 40:
                    # 2. Respawn Cube (Relative to Robot Base)
                    rand_x = np.random.uniform(SPAWN_BOX_X[0], SPAWN_BOX_X[1])
                    rand_y = np.random.uniform(SPAWN_BOX_Y[0], SPAWN_BOX_Y[1])
                    
                    local_spawn = np.array([rand_x, rand_y, 0.05])
                    spawn_world = local_to_world(robot_pos, robot_quat, local_spawn)
                    
                    rand_rot = euler_angles_to_quat(np.array([0, 0, np.random.uniform(0, 3)]))

                    target_cube.set_world_pose(position=spawn_world, orientation=rand_rot)
                    target_cube.set_linear_velocity(np.array([0,0,0]))
                    target_cube.set_angular_velocity(np.array([0,0,0]))
                    
                    current_state = PickStates.HOVER

            # --- STEP 4: EXECUTE ---
            rmpflow.update_world() 
            rmpflow.set_end_effector_target(target_pos, target_quat)
            ghost_target.set_world_pose(position=target_pos, orientation=target_quat) 

            arm_action = articulation_rmpflow.get_next_articulation_action()
            full_joint_positions = np.append(arm_action.joint_positions, [gripper_target, gripper_target])
            franka.apply_action(ArticulationAction(joint_positions=full_joint_positions))

    simulation_app.close()

if __name__ == "__main__":
    main()