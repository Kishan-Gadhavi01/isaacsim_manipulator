
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualSphere
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import PickPlaceController
from isaacsim.core.utils.rotations import euler_angles_to_quat

def main():
    # 1. Initialize World
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    world.scene.add_default_ground_plane()

    # 2. Add Robot
    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
    
    # 3. Add Target Cube
    cube_name = "fancy_cube"
    cube = world.scene.add(
        DynamicCuboid(
            prim_path=f"/World/{cube_name}",
            name=cube_name,
            position=np.array([0.6, 0.0, 0.02]), 
            scale=np.array([0.04, 0.04, 0.04]),
            color=np.array([1.0, 0.0, 0.0]),
            mass=0.1, 
        )	
    )

    target_marker = world.scene.add(VisualSphere(prim_path="/World/TargetMarker", name="target_marker", radius=0.01, color=np.array([0, 1, 0])))

    # 4. Initialize Controller
    controller = PickPlaceController(
        name="pick_place_controller",
        gripper=franka.gripper,
        robot_articulation=franka
    )

    goal_position = np.array([0.6, -0.3, 0.025])

    print("Initializing Physics...")
    world.reset()
    
    # --- PHYSICS TUNING (CORRECTED) ---
    
    # A. Solver Iterations (Valid Articulation API)
    # This makes the robot "stiffer" and handles collisions better
    franka.set_solver_position_iteration_count(64)
    franka.set_solver_velocity_iteration_count(64)

    franka.gripper._closed_position = np.array([0.015, 0.015]) 
    franka.gripper._opened_position = np.array([0.04, 0.04])   

    # Open gripper to start
    franka.gripper.open()

    print("Physics Tuned. Starting Loop...")

    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_playing():
            current_joint_positions = franka.get_joint_positions()
            cube_position, _ = cube.get_world_pose()

            # Orientation: Fingers Down
            orientation_down = euler_angles_to_quat(np.array([np.pi, 0, np.pi / 2.0]))

            # Offset: Zero
            ee_offset = np.array([0.0, 0.0, 0.0])

            target_marker.set_world_pose(position=cube_position + ee_offset)

            actions = controller.forward(
                picking_position=cube_position,
                placing_position=goal_position,
                current_joint_positions=current_joint_positions,
                end_effector_orientation=orientation_down,
                end_effector_offset=ee_offset
            )
            
            franka.apply_action(actions)

            if controller.is_done():
                print("Cycle Complete. Resetting...")
                world.reset()
                controller.reset()
                
                franka.gripper._closed_position = np.array([0.015, 0.015])
                franka.gripper.open()
                
                cube.set_world_pose(position=np.array([0.6, np.random.uniform(-0.1, 0.1), 0.02]))

    simulation_app.close()

if __name__ == "__main__":
    main()
