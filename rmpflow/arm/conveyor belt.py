
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import time
from scipy.spatial.transform import Rotation as R

from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, VisualSphere, FixedCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import open_stage

from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
USD_PATH = "/home/popoy/robotics/sim/isaacsim_manipulator/rmpflow/arm/conveyor.usd"

# Spawn slightly lower to reduce bounce (1.83)
GLOBAL_SPAWN_CENTER = np.array([-3.0, 3.63, 1.83]) 
ROBOT_BASE_POS = np.array([0.0, 3.3, 1.8])
ROBOT_BASE_ROT = np.array([0.0, 0.0, 90.0])

BELT_DIR = np.array([1.0, 0.0, 0.0]) 
BELT_SPEED = 0.1 

# --- TUNING ---
# 8cm gives the robot 0.8s to settle. 5cm was too rushed.
INTERCEPT_OFFSET = 0.08  
TRANSITION_TOLERANCE = 0.10 
HOVER_HEIGHT    = 0.15
EXTRA_SQUISH    = 0.01  
GRIP_DEPTH      = 0.015 

GRIPPER_OPEN    = 0.08
SUCCESS_HEIGHT  = GLOBAL_SPAWN_CENTER[2] + 0.30

# ---------------------------------------------------------
# 2. CLASSES & HELPERS
# ---------------------------------------------------------

class PickStates:
    INTERCEPT = 0
    SYNC      = 1
    GRASP     = 2
    LIFT      = 3

class SmoothedPose:
    def __init__(self, alpha=0.4):
        self.pos = None
        self.alpha = alpha
    def update(self, pos):
        if self.pos is None:
            self.pos = pos.copy()
        else:
            self.pos = self.alpha * pos + (1.0 - self.alpha) * self.pos
        return self.pos

def isaac_to_scipy_quat(q_isaac):
    q = np.asarray(q_isaac)
    return np.array([q[1], q[2], q[3], q[0]])

def scipy_to_isaac_quat(q_scipy):
    q = np.asarray(q_scipy)
    return np.array([q[3], q[0], q[1], q[2]])

def simulate_vision_detection_safe(cube_quat_isaac):
    q = np.asarray(cube_quat_isaac)
    if q.shape[0] != 4: return np.array([1,0,0,0]), np.array([0,0,1])

    r = R.from_quat(isaac_to_scipy_quat(q))
    matrix = r.as_matrix()
    local_x, local_y, local_z = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    world_up, world_forward = np.array([0, 0, 1]), np.array([1, 0, 0])

    axes = [(local_x, "X"), (-local_x, "-X"), (local_y, "Y"), (-local_y, "-Y"), (local_z, "Z"), (-local_z, "-Z")]

    best_normal_vec, best_normal_dot = None, -10.0
    for vec, _ in axes:
        d = np.dot(vec, world_up)
        if d > best_normal_dot:
            best_normal_dot, best_normal_vec = d, vec

    candidates = []
    for vec, _ in axes:
        if abs(np.dot(vec, best_normal_vec)) > 0.05: continue
        candidates.append((np.dot(vec, world_forward), vec))
    
    if not candidates: best_tangent_vec = np.array([1.0, 0.0, 0.0])
    else: best_tangent_vec = max(candidates, key=lambda x: x[0])[1]

    target_z = -best_normal_vec
    target_x = best_tangent_vec
    target_y = np.cross(target_z, target_x)

    def safe_norm(v): return v / (np.linalg.norm(v) + 1e-8)
    M = np.column_stack((safe_norm(target_x), safe_norm(target_y), safe_norm(target_z)))
    qs = R.from_matrix(M).as_quat()
    return scipy_to_isaac_quat(qs), best_normal_vec

# ---------------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------------

def main():
    print("Loading Stage...")
    open_stage(USD_PATH)
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)

    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka", position=ROBOT_BASE_POS, orientation=euler_angles_to_quat(np.radians(ROBOT_BASE_ROT))))
    TCP_PRIM_PATH = "/World/Franka/panda_hand/tool_center"
    tool_center_link = SingleXFormPrim(prim_path=TCP_PRIM_PATH, name="tool_center_link")

    high_friction_mat = PhysicsMaterial(
        prim_path="/World/Physics_Materials/HighFriction",
        name="high_friction_mat",
        static_friction=3.0, 
        dynamic_friction=3.0,
        restitution=0.0      
    )

    target_cube = world.scene.add(DynamicCuboid(
        prim_path="/World/TargetCube", 
        name="target_cube",
        position=GLOBAL_SPAWN_CENTER, 
        scale=np.array([0.05]*3), 
        color=np.array([1.0, 0.1, 0.1]), 
        mass=0.1, 
        physics_material=high_friction_mat
    ))

    ghost_is = world.scene.add(VisualSphere(prim_path="/World/GhostIs", name="ghost_is", radius=0.005, color=np.array([1, 1, 0])))
    ghost_aim = world.scene.add(VisualSphere(prim_path="/World/GhostAim", name="ghost_aim", radius=0.005, color=np.array([0, 1, 0])))
    table_shield = world.scene.add(FixedCuboid(prim_path="/World/TableShield", name="table_shield", position=np.array([0.0, 2.6, 1.78]), scale=np.array([5.0, 1.5, 0.02]), visible=False))

    world.reset()
    
    # KICKSTART FIRST CYCLE
    target_cube.set_linear_velocity(np.array([0.1, 0.0, 0.0])) 
    
    franka.initialize()
    franka.set_joints_default_state(positions=np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4, 0.04, 0.04]))

    rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
    rmp_config["end_effector_frame_name"] = "right_gripper"
    rmpflow = RmpFlow(**rmp_config)
    rmpflow.add_obstacle(table_shield)
    try: rmpflow.remove_obstacle(target_cube)
    except: pass
    articulation_rmpflow = ArticulationMotionPolicy(franka, rmpflow, 1.0/60.0)

    current_state = PickStates.INTERCEPT 
    gripper_target = GRIPPER_OPEN
    locked_offset = np.zeros(3)
    smoothing = SmoothedPose(alpha=0.35)
    last_time = time.time()
    wait_time = 0.0

    print(f"🚀 STABLE GRIP: No 'Run Away' offset. 8cm Lead.")

    while simulation_app.is_running():
        world.step(render=True)

        if not world.is_playing():
            continue

        now = time.time()
        dt = max(1.0/240.0, now - last_time)
        last_time = now
        wait_time += dt

        cube_pos, raw_quat = target_cube.get_world_pose()
        smoothed_cube_pos = smoothing.update(cube_pos)
        
        robot_pos, robot_quat = franka.get_world_pose()
        rmpflow.set_robot_base_pose(robot_pos, robot_quat)
        ee_pos, ee_rot = tool_center_link.get_world_pose()
        vision_quat, face_normal = simulate_vision_detection_safe(raw_quat)

        target_pos = ee_pos.copy()
        target_quat = robot_quat

        # --- STATE MACHINE ---

        if current_state == PickStates.INTERCEPT:
            ambush_point = smoothed_cube_pos + (BELT_DIR * INTERCEPT_OFFSET)
            target_pos = ambush_point + (face_normal * HOVER_HEIGHT)
            target_quat = vision_quat
            gripper_target = GRIPPER_OPEN

            xy_error = np.linalg.norm(ee_pos[:2] - target_pos[:2])
            if xy_error < TRANSITION_TOLERANCE:
                current_state = PickStates.SYNC
                wait_time = 0.0

        elif current_state == PickStates.SYNC:
            ambush_point = smoothed_cube_pos + (BELT_DIR * INTERCEPT_OFFSET)
            
            # Sink deep into the cube
            target_pos = ambush_point - (face_normal * EXTRA_SQUISH)
            target_quat = vision_quat
            
            dist_to_real_cube = np.linalg.norm(ee_pos - smoothed_cube_pos)
            z_error = abs(ee_pos[2] - (smoothed_cube_pos[2] - EXTRA_SQUISH))

            # Wait for cube to catch up (4cm range) + Vertical Guard Rail (1.5cm range)
            if dist_to_real_cube < 0.04 and z_error < 0.015: 
                locked_offset = ee_pos - smoothed_cube_pos
                print(f"[STATE] GRASP (Z-Err: {z_error:.3f})")
                current_state = PickStates.GRASP
                wait_time = 0.0

        elif current_state == PickStates.GRASP:
            # FIX: MOVED WITH THE CUBE, DO NOT JUMP AHEAD.
            # Maintain the 'locked_offset' which is the relative position at trigger time.
            target_pos = smoothed_cube_pos + locked_offset
            
            # Continue pushing down
            target_pos[2] -= EXTRA_SQUISH 
            
            target_quat = vision_quat
            gripper_target = 0.0 
            
            if wait_time > 0.30: 
                print("[STATE] LIFT")
                current_state = PickStates.LIFT
                wait_time = 0.0

        elif current_state == PickStates.LIFT:
            target_pos = smoothed_cube_pos + np.array([0.0, 0.0, 0.35])
            target_quat = vision_quat
            gripper_target = 0.0

            if ee_pos[2] > SUCCESS_HEIGHT:
                print(f"[STATE] SUCCESS! Respawning.")
                
                target_cube.set_world_pose(position=GLOBAL_SPAWN_CENTER, orientation=euler_angles_to_quat(np.array([0, 0, 0])))
                target_cube.set_linear_velocity(np.array([0.1, 0.0, 0.0])) # Kickstart
                target_cube.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                
                smoothing.pos = None
                current_state = PickStates.INTERCEPT
                wait_time = 0.0

        rmpflow.update_world()
        rmpflow.set_end_effector_target(np.array(target_pos), np.array(target_quat))
        
        ghost_is.set_world_pose(position=smoothed_cube_pos, orientation=target_quat)
        ghost_aim.set_world_pose(position=np.array(target_pos), orientation=target_quat)

        arm_action = articulation_rmpflow.get_next_articulation_action()
        full_joint_positions = np.append(arm_action.joint_positions, [gripper_target, gripper_target])
        franka.apply_action(ArticulationAction(joint_positions=full_joint_positions))

    simulation_app.close()

if __name__ == "__main__":
    main()  