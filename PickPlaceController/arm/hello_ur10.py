# Copyright (c) 2025, The Big Brain. All rights reserved.

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat

# Standard Library Imports
from isaacsim.robot.manipulators.examples.universal_robots import UR10
from isaacsim.robot.manipulators.examples.universal_robots.controllers import PickPlaceController

def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    world.scene.add_default_ground_plane()

    # 1. Add Robot
    ur10 = world.scene.add(
        UR10(prim_path="/World/UR10", name="my_ur10", attach_gripper=True)
    )

    # 2. Add Cube
    cube_name = "fancy_cube"
    cube = world.scene.add(
        DynamicCuboid(
            prim_path=f"/World/{cube_name}",
            name=cube_name,
            position=np.array([0.7, 0.0, 0.05]), 
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([1.0, 0.0, 0.0]),
            mass=0.1, 
        )   
    )

    # 3. Controller
    controller = PickPlaceController(
        name="ur10_pick_place",
        gripper=ur10.gripper,
        robot_articulation=ur10
    )

    goal_position = np.array([0.5, 0.5, 0.05]) 

    world.reset()
    ur10.initialize()
    
    # --- TUNING: BOOST GRIPPER STRENGTH ---
    # Make the vacuum stronger so it catches easier
    ur10.gripper.set_force_limit(1.0e4)
    ur10.gripper.set_torque_limit(1.0e4)
    
    home_joints = np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0])
    ur10.set_joints_default_state(positions=home_joints)
    ur10.set_joint_positions(home_joints)

    print("🚀 Starting TIGHT Pick & Place Loop...")

    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_playing():
            current_joint_positions = ur10.get_joint_positions()
            cube_position, _ = cube.get_world_pose()    
            
            # Rotation: Point Down
            orientation_down = euler_angles_to_quat(np.array([0, np.pi / 2.0, 0]))

           
            picking_offset = np.array([0, 0, 0.03]) 

            actions = controller.forward(
                picking_position=cube_position,
                placing_position=goal_position,
                current_joint_positions=current_joint_positions,
                end_effector_orientation=orientation_down,
                end_effector_offset=picking_offset 
            )
            
            ur10.apply_action(actions)

            if controller.is_done():
                print("✅ Success! Resetting...")
                world.reset()
                ur10.set_joint_positions(home_joints)
                controller.reset()
                cube.set_world_pose(position=np.array([0.7, np.random.uniform(-0.2, 0.2), 0.05]))

    simulation_app.close()

if __name__ == "__main__":
    main()