
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import cv2
import numpy as np
from pxr import Gf, UsdGeom
from isaacsim.core.api import World
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.api.objects import DynamicCuboid, VisualSphere, VisualCuboid
from isaacsim.sensors.camera import Camera
from isaacsim.robot.manipulators.examples.franka import Franka
from omni.isaac.franka.controllers import PickPlaceController

# --- CONFIGURATION ---
CAM_POS = Gf.Vec3d(0.5, 0, 2.5)
CAM_ROT = Gf.Vec3d(0, 0, 90.0)  

# --- BIN LAYOUT ---
BIN_RED_POS   = np.array([0.15, -0.5, 0.05])
BIN_GREEN_POS = np.array([0.35, -0.5, 0.05])
BIN_BLUE_POS  = np.array([0.55, -0.5, 0.05])

# --- SPAWN ZONE (Tighter to stay in ROI) ---
SPAWN_X_RANGE = (0.40, 0.70)
SPAWN_Y_RANGE = (0.15, 0.45)

# --- SAFETY RULES & DIMENSIONS ---
CUBE_HEIGHT = 0.04
GRIP_DESCENT_INTO_CUBE = CUBE_HEIGHT / 2.0
ABSOLUTE_FLOOR_LIMIT = 0.015

# GRIPPER DIMENSIONS (Meters)
GRIPPER_FINGER_WIDTH = 0.03
GRIPPER_OPEN_WIDTH   = 0.10
GRIPPER_BODY_WIDTH   = 0.045  #

SAFETY_MARGIN = 0.005

# --- ROI CONFIG (UPDATED ROI_X) ---
ROI_Y = (30, 450)
ROI_X = (100, 600)

# Debugging
DEBUG_BLOCKERS = True  # Set True to print which neighbor blocks picks (and visualize it)

class BinManager:
    def __init__(self):
        self.counts = {"red": 0, "green": 0, "blue": 0}
        self.offsets = [[0,0], [0,0.06], [0,-0.06], [0.06,0], [0.06,0.06], [0.06,-0.06]]

    def get_place_target(self, color, base_pos):
        count = self.counts[color]
        idx = count % len(self.offsets)
        z_stack = (count // len(self.offsets)) * 0.04
        return np.array([base_pos[0] + self.offsets[idx][0], base_pos[1] + self.offsets[idx][1], base_pos[2] + z_stack + 0.05])

    def confirm_placement(self, color):
        self.counts[color] += 1
        print(f"SUCCESS: Sorted {color}. Total: {self.counts[color]}")

bin_manager = BinManager()

class VisionSystem:
    def __init__(self):
        self.camera = None
        cv2.namedWindow("Sorter Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sorter Dashboard", 600, 1400)

    def set_camera(self, camera):
        self.camera = camera

    def process_frame(self):
        if not self.camera:
            return None
        frame = self.camera.get_current_frame()
        if "rgba" not in frame:
            return None

        img_bgr = cv2.cvtColor(frame["rgba"][:, :, :3], cv2.COLOR_RGB2BGR)
        depth_map = frame["distance_to_image_plane"]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        mask_red_raw = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
                       cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask_green_raw = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([85, 255, 255]))
        mask_blue_raw = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))

        roi_filter = np.zeros_like(mask_red_raw)
        roi_filter[ROI_Y[0]:ROI_Y[1], ROI_X[0]:ROI_X[1]] = 255

        mask_red   = cv2.bitwise_and(mask_red_raw, roi_filter)
        mask_green = cv2.bitwise_and(mask_green_raw, roi_filter)
        mask_blue  = cv2.bitwise_and(mask_blue_raw, roi_filter)

        candidates = []

        def get_blobs(mask, color_name):
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 100 < area < 2500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w//2, y + h//2
                    # guard against out-of-bounds depth read
                    if cy < 0 or cy >= depth_map.shape[0] or cx < 0 or cx >= depth_map.shape[1]:
                        continue
                    d = depth_map[cy, cx]
                    if 0 < d < 4.0:
                        pts_3d = self.camera.get_world_points_from_image_coords(np.array([[cx, cy]]), np.array([d]))
                        pos = pts_3d[0]
                        candidates.append({
                            "pos": pos,
                            "color": color_name,
                            "box": (x, y, w, h),
                            "safe": False,
                            "angle_offset": 0.0,
                            "blocked_by": None
                        })

        get_blobs(mask_red, "red")
        get_blobs(mask_green, "green")
        get_blobs(mask_blue, "blue")

        # --- SMART GRASP LOGIC (AGGRESSIVE LATERAL CLEARANCE) ---
        # Important mapping due to camera rotation:
        # IMAGE horizontal left/right = WORLD Y
        # IMAGE vertical up/down     = WORLD X
        #
        #   lateral  = world Y (image left/right)
        #   longitud = world X (image up/down)
        #
        req_clear_body = GRIPPER_BODY_WIDTH + SAFETY_MARGIN
        req_clear_fingers = GRIPPER_OPEN_WIDTH + SAFETY_MARGIN

        for i, c1 in enumerate(candidates):
            c1_x, c1_y, c1_z = c1["pos"][0], c1["pos"][1], c1["pos"][2]
            standard_blocked = False
            rotated_blocked = False
            blocker_standard = None
            blocker_rotated = None

            for j, c2 in enumerate(candidates):
                if i == j:
                    continue
                c2_x, c2_y, c2_z = c2["pos"][0], c2["pos"][1], c2["pos"][2]

                # remap axes for rotated camera view:
                lateral  = abs(c1_y - c2_y)   # world Y -> image left/right
                longitud = abs(c1_x - c2_x)   # world X -> image up/down
                dz = abs(c1_z - c2_z)

                # Consider neighbor relevant if it's on the same layer (approx same top surface)
                same_layer = dz < (CUBE_HEIGHT * 0.75)

                # STANDARD GRIP (fingers run along world Y axis if angle_offset = 0)
                # Need clearance in lateral for gripper body and longitud for finger sweep
                if same_layer and (lateral < req_clear_body and longitud < req_clear_fingers):
                    standard_blocked = True
                    blocker_standard = c2
                # ROTATED GRIP (fingers run along world X axis, angle_offset = pi/2)
                # Need clearance in longitud for body and lateral for finger sweep
                if same_layer and (lateral < req_clear_fingers and longitud < req_clear_body):
                    rotated_blocked = True
                    blocker_rotated = c2

                if standard_blocked and rotated_blocked:
                    break

            # Choose approach: prefer standard (angle 0) if available
            if not standard_blocked:
                c1["safe"] = True
                c1["angle_offset"] = 0.0
                c1["blocked_by"] = None
            elif not rotated_blocked:
                c1["safe"] = True
                c1["angle_offset"] = np.pi / 2.0
                c1["blocked_by"] = None
            else:
                c1["safe"] = False
                # Choose which blocker to report (prefer nearest blocker)
                if blocker_standard is None:
                    c1["blocked_by"] = blocker_rotated
                elif blocker_rotated is None:
                    c1["blocked_by"] = blocker_standard
                else:
                    # pick closer in horizontal plane
                    dist_s = np.linalg.norm(np.array(c1["pos"]) - np.array(blocker_standard["pos"]))
                    dist_r = np.linalg.norm(np.array(c1["pos"]) - np.array(blocker_rotated["pos"]))
                    c1["blocked_by"] = blocker_standard if dist_s <= dist_r else blocker_rotated

                if DEBUG_BLOCKERS:
                    if c1["blocked_by"] is not None:
                        print(f"Candidate at {c1['pos']} blocked by neighbor at {c1['blocked_by']['pos']} (color={c1['blocked_by']['color']})")

        # Visualization
        vis_rgb = img_bgr.copy()
        cv2.rectangle(vis_rgb, (ROI_X[0], ROI_Y[0]), (ROI_X[1], ROI_Y[1]), (0, 255, 255), 2)
        for c in candidates:
            x, y, w, h = c["box"]
            cx, cy = x + w//2, y + h//2
            box_col = (0, 255, 0) if c["safe"] else (0, 0, 255)
            label = "PICK" if c["safe"] else "CLUTTER"

            cv2.rectangle(vis_rgb, (x, y), (x+w, y+h), box_col, 2)
            cv2.putText(vis_rgb, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 2)

            if c["safe"]:
                if abs(c["angle_offset"]) > 0.1:
                    # X-grip → fingers move left/right → draw horizontal line
                    cv2.line(vis_rgb, (x, cy), (x+w, cy), (255, 0, 255), 2)
                    cv2.putText(vis_rgb, "X-grip", (x, y+h+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)
                else:
                    # Y-grip → fingers move up/down → draw vertical line
                    cv2.line(vis_rgb, (cx, y), (cx, y+h), (0, 255, 0), 2)
                    cv2.putText(vis_rgb, "Y-grip", (x, y+h+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            else:
                # If blocked, visualize the neighbor blocking (if known)
                if c.get("blocked_by") is not None:
                    b = c["blocked_by"]
                    bx, by, bw, bh = b["box"]
                    # draw blocker box in blue to highlight
                    cv2.rectangle(vis_rgb, (bx, by), (bx+bw, by+bh), (255, 128, 0), 2)
                    cv2.putText(vis_rgb, f"BLOCKER:{b['color']}", (bx, by-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,128,0), 1)

        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        vis_mask = np.zeros_like(img_bgr)
        vis_mask[mask_red > 0] = [0, 0, 255]
        vis_mask[mask_green > 0] = [0, 255, 0]
        vis_mask[mask_blue > 0] = [255, 0, 0]

        dashboard = np.vstack([vis_rgb, vis_depth, vis_mask])
        cv2.imshow("Sorter Dashboard", dashboard)
        cv2.waitKey(1)
        return candidates

def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    world.scene.add_default_ground_plane()
    world.scene.add(VisualCuboid(prim_path="/World/FloorOverlay", name="floor_overlay", position=np.array([0.5, 0, 0.002]), scale=np.array([5.0, 5.0, 0.01]), color=np.array([0.4, 0.4, 0.4])))

    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

    world.scene.add(VisualCuboid(prim_path="/World/BinRed", name="bin_red", position=np.array([BIN_RED_POS[0], BIN_RED_POS[1], 0.02]), scale=np.array([0.18, 0.18, 0.005]), color=np.array([0.8, 0, 0])))
    world.scene.add(VisualCuboid(prim_path="/World/BinGreen", name="bin_green", position=np.array([BIN_GREEN_POS[0], BIN_GREEN_POS[1], 0.02]), scale=np.array([0.18, 0.18, 0.005]), color=np.array([0, 0.8, 0])))
    world.scene.add(VisualCuboid(prim_path="/World/BinBlue", name="bin_blue", position=np.array([BIN_BLUE_POS[0], BIN_BLUE_POS[1], 0.02]), scale=np.array([0.18, 0.18, 0.005]), color=np.array([0, 0, 0.8])))

    MIN_SEPARATION = 0.08 
    placed_positions = []
    
    for i in range(12):
        rem = i % 3
        if rem == 0:
            col_name = "red"; col_val = np.array([1, 0, 0])
        elif rem == 1:
            col_name = "green"; col_val = np.array([0, 1, 0])
        else:
            col_name = "blue"; col_val = np.array([0.2, 0.6, 1.0])

        valid_pos = False
        attempts = 0
        while not valid_pos and attempts < 100:
            spawn_x = np.random.uniform(SPAWN_X_RANGE[0], SPAWN_X_RANGE[1])
            spawn_y = np.random.uniform(SPAWN_Y_RANGE[0], SPAWN_Y_RANGE[1])
            new_pos_xy = np.array([spawn_x, spawn_y])
            valid_pos = True
            
            for existing_pos in placed_positions:
                distance = np.linalg.norm(new_pos_xy - existing_pos)
                if distance < MIN_SEPARATION:
                    valid_pos = False
                    break
            attempts += 1

        if valid_pos:
            placed_positions.append(new_pos_xy)
            world.scene.add(DynamicCuboid(prim_path=f"/World/Cube_{i}", name=f"cube_{i}", position=np.array([spawn_x, spawn_y, 0.05]), scale=np.array([0.04, 0.04, 0.04]), color=col_val, mass=0.1))
        else:
            print(f"Warning: Could not find spaced position for cube {i} after 100 attempts.")
    # -----------------------------------------------------------------------------

    controller = PickPlaceController(name="controller", gripper=franka.gripper, robot_articulation=franka)
    marker = world.scene.add(VisualSphere(prim_path="/World/Marker", name="marker", radius=0.015, color=np.array([1, 1, 0])))

    world.reset()

    camera = Camera(prim_path="/World/FixedCam", name="fixed_cam", resolution=(640, 480), frequency=30)
    camera.initialize()
    xform = UsdGeom.Xformable(world.stage.GetPrimAtPath("/World/FixedCam"))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(CAM_POS)
    xform.AddRotateXYZOp().Set(CAM_ROT)
    camera.add_distance_to_image_plane_to_frame()

    vision = VisionSystem()
    vision.set_camera(camera)

    franka.gripper.open()
    for _ in range(60):
        world.step(render=True)

    state = "SEARCH"
    current_target = None
    current_color = None
    current_angle = 0.0
    while simulation_app.is_running():
        world.step(render=True)
        if world.is_playing():

            # --- VISION FREEZE LOGIC ---
            if state == "SEARCH":
                candidates = vision.process_frame()
            else:
                vision.process_frame() 

                candidates = []   # vision frozen during PICKING and RESET

            if state == "SEARCH":
                safe_candidates = [c for c in candidates if c["safe"]]
                if safe_candidates:
                    best_candidate = min(
                        safe_candidates,
                        key=lambda c: np.linalg.norm(np.array(c["pos"][:2]) - np.array([0.5, 0.0]))
                    )
                else:
                    best_candidate = None

                if best_candidate:
                    target_z = best_candidate["pos"][2] - GRIP_DESCENT_INTO_CUBE
                    current_target = np.array([
                        best_candidate["pos"][0],
                        best_candidate["pos"][1],
                        max(target_z, ABSOLUTE_FLOOR_LIMIT)
                    ])
                    current_color = best_candidate["color"]
                    current_angle = best_candidate["angle_offset"]

                    marker.set_world_pose(position=current_target)
                    print(f"Target Locked: {current_color} | Angle: {np.degrees(current_angle):.0f} deg")
                    state = "PICKING"

            elif state == "PICKING":
                # Do NOT read vision again
                # Use stored current_target, current_color, current_angle ONLY
                if current_color == "red":
                    drop_off = bin_manager.get_place_target("red", BIN_RED_POS)
                elif current_color == "green":
                    drop_off = bin_manager.get_place_target("green", BIN_GREEN_POS)
                else:
                    drop_off = bin_manager.get_place_target("blue", BIN_BLUE_POS)

                target_yaw = np.pi / 2.0 - current_angle
                ee_orientation_eulers = np.array([np.pi, -0.45, target_yaw])

                final_orientation = euler_angles_to_quat(ee_orientation_eulers)

                actions = controller.forward(
                    picking_position=current_target,
                    placing_position=drop_off,
                    current_joint_positions=franka.get_joint_positions(),
                    end_effector_orientation=final_orientation
                )
                franka.apply_action(actions)

                ee_pos = franka.end_effector.get_world_pose()[0]
                gripper_width = franka.gripper.get_joint_positions().sum()

                if ee_pos[2] > 0.15 and gripper_width < 0.005:
                    print(f"!!! CRITICAL: Object Dropped Mid-Air! ({current_color}) !!!")
                    controller.reset()
                    franka.gripper.open()
                    state = "RESET"
                    continue

                if ee_pos[2] < 0.06 and gripper_width > 0.03:
                    print("Pick Failed: Could not grip cube. Restarting search...")
                    controller.reset()
                    franka.gripper.open()
                    state = "RESET"
                    continue

                if controller.is_done():
                    if gripper_width > 0.005:
                        bin_manager.confirm_placement(current_color)
                    else:
                        print("Task finished but gripper empty.")
                    controller.reset()
                    state = "RESET"

            elif state == "RESET":
                franka.gripper.open()
                for _ in range(20):
                    world.step(render=True)
                state = "SEARCH"
                current_target = None


    cv2.destroyAllWindows()
    simulation_app.close()

if __name__ == "__main__":
    main()