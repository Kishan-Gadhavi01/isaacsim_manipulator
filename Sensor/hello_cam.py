# final_sorter_v20_dual_stream.py
# Feature: Side-by-Side RGB + Depth Heatmap Visualization

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import cv2
import numpy as np
from pxr import Gf, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.api.objects import DynamicCuboid, VisualSphere
from isaacsim.sensors.camera import Camera
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import PickPlaceController

# --- 1. EXACT COORDINATES ---
CAM_POS = Gf.Vec3d(4.0, -0.8, 2.0) 
CAM_ROT = Gf.Vec3d(65.0, -6.0, 80.0)

BIN_RED = np.array([0.3, 0.3, 0.05])
BIN_BLUE = np.array([0.3, -0.3, 0.05])

class VisionSystem:
    def __init__(self, camera):
        self.camera = camera
        # Rename window to indicate dual stream
        cv2.namedWindow("Robot Eyes (RGB | Depth)", cv2.WINDOW_NORMAL)
        
    def process_frame(self):
        frame = self.camera.get_current_frame()
        if "rgba" not in frame or "depth" not in frame:
            # Create a double-wide blank image
            blank = np.zeros((480, 1280, 3), dtype=np.uint8)
            cv2.putText(blank, "WAITING FOR SENSOR STREAMS...", (300, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Robot Eyes (RGB | Depth)", blank)
            cv2.waitKey(1)
            return None, None, None

        # 1. Process RGB
        img_bgr = cv2.cvtColor(frame["rgba"][:, :, :3], cv2.COLOR_RGB2BGR)
        
        # 2. Process Depth
        depth_map = frame["depth"]
        
        # --- VISUALIZATION MAGIC ---
        # Depth is usually floats (meters). We need to squash it to 0-255 colors.
        # Clip values to 6.0m (since camera is ~4m away, anything beyond is 'background')
        depth_display = np.clip(depth_map, 0, 6.0) 
        # Normalize to 0-255
        depth_display = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Apply "JET" Colormap (Blue=Close, Red=Far)
        depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        # 3. Logic (HSV Detection)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
                   cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask_blue = cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))

        target_pos, target_color = None, None
        
        def find_blob(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            best = None; max_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100 and area > max_area: max_area = area; best = cnt
            return best

        cnt_red = find_blob(mask_red)
        cnt_blue = find_blob(mask_blue)
        
        chosen_cnt = cnt_red if cnt_red is not None else cnt_blue
        if chosen_cnt is not None:
            target_color = "red" if chosen_cnt is cnt_red else "blue"
            x, y, w, h = cv2.boundingRect(chosen_cnt)
            
            # Draw Box on RGB
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw Box on Depth (So you can see the match)
            cv2.rectangle(depth_colored, (x, y), (x+w, y+h), (255, 255, 255), 2)

            cx, cy = x + w//2, y + h//2
            d = depth_map[cy, cx]
            
            if d > 0 and d < 6.0: 
                target_pos = self.camera.get_world_point_from_screen_coords(
                    screen_coords=np.array([cx, cy]), depth=d
                )
                # Draw Depth Text on RGB Image
                cv2.putText(img_bgr, f"Dist: {d:.2f}m", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. STITCH THEM TOGETHER
        # Concatenate Horizontally
        final_view = np.hstack((img_bgr, depth_colored))

        cv2.imshow("Robot Eyes (RGB | Depth)", final_view)
        cv2.waitKey(1)
        return target_pos, target_color, img_bgr

def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    world.scene.add_default_ground_plane()
    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

    camera_path = "/World/FixedCam"
    camera = world.scene.add(Camera(
        prim_path=camera_path,
        name="fixed_cam",
        resolution=(640, 480),
        frequency=30,
    ))
    camera.initialize()
    # CRITICAL: Enable Depth Stream
    camera.add_distance_to_image_plane_to_frame() 
    
    # Force USD Attributes
    stage = world.stage
    prim = stage.GetPrimAtPath(camera_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    op_translate = xform.AddTranslateOp()
    op_rotate = xform.AddRotateXYZOp()
    op_translate.Set(CAM_POS)
    op_rotate.Set(CAM_ROT)
    
    vision = VisionSystem(camera)

    # Cubes
    for i in range(4):
        color = np.array([1, 0, 0]) if i % 2 == 0 else np.array([0, 0, 1])
        world.scene.add(DynamicCuboid(
            prim_path=f"/World/Cube_{i}",
            name=f"cube_{i}",
            position=np.array([0.5 + np.random.uniform(-0.1, 0.1), np.random.uniform(-0.2, 0.2), 0.05]),
            scale=np.array([0.04, 0.04, 0.04]),
            color=color, mass=0.1
        ))

    controller = PickPlaceController(name="controller", gripper=franka.gripper, robot_articulation=franka)
    marker = world.scene.add(VisualSphere(prim_path="/World/Marker", name="marker", radius=0.015, color=np.array([1, 1, 0])))

    world.reset()
    
    # Re-apply pose
    op_translate.Set(CAM_POS)
    op_rotate.Set(CAM_ROT)
    
    franka.set_solver_position_iteration_count(64)
    franka.set_solver_velocity_iteration_count(64)
    franka.gripper._closed_position = np.array([0.015, 0.015]) 
    franka.gripper.open()

    print("Warming up...")
    for _ in range(60): world.step(render=True)

    state, current_target, current_color = "SEARCH", None, None
    
    while simulation_app.is_running():
        world.step(render=True)
        if world.is_playing():
            detected_pos, detected_color, _ = vision.process_frame()
            
            if state == "SEARCH":
                if detected_pos is not None:
                    current_target = detected_pos; current_color = detected_color; state = "PICKING"
                    marker.set_world_pose(position=current_target)
            elif state == "PICKING":
                drop_off = BIN_RED if current_color == "red" else BIN_BLUE
                actions = controller.forward(
                    picking_position=current_target, placing_position=drop_off,
                    current_joint_positions=franka.get_joint_positions(),
                    end_effector_orientation=euler_angles_to_quat(np.array([np.pi, 0, np.pi / 2.0])),
                    end_effector_offset=np.array([0, 0, 0])
                )
                franka.apply_action(actions)
                if controller.is_done():
                    controller.reset(); state = "SEARCH"; current_target = None

    cv2.destroyAllWindows()
    simulation_app.close()

if __name__ == "__main__":
    main()
