"""
双目深度视觉与点云测距模块
==========================
基于安装在机械手末端的双目摄像头系统，实现：
1. 双目图像捕获
2. 红色苹果检测
3. 视差计算与深度估计
4. 点云生成
5. 苹果3D位置精确定位与方位计算

Usage:
    from stereo_vision import StereoCameraSystem, StereoVisionProcessor
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

import mujoco

# ============================================================================
# Camera Parameters
# ============================================================================
CAMERA_RES = (240, 320)  # (height, width)
BASELINE = 0.06  # 双目基线距离 6cm
FOVY = 70  # 垂直视场角 (degrees)


def _camera_intrinsics(res_h, res_w, fovy_deg):
    """Compute camera intrinsic matrix from resolution and fovy."""
    fovy_rad = np.deg2rad(fovy_deg)
    fy = res_h / (2 * np.tan(fovy_rad / 2))
    fx = fy * (res_w / res_h)
    cx = res_w / 2
    cy = res_h / 2
    return fx, fy, cx, cy


# ============================================================================
# Stereo Camera System
# ============================================================================
class StereoCameraSystem:
    """Manage left/right camera image capture from the gripper."""

    def __init__(self, model, data, res=CAMERA_RES, frame_dir="tmp/camera_frames"):
        self.model = model
        self.data = data
        self.renderer = mujoco.Renderer(model, *res)
        self.res_h, self.res_w = res
        self.frame_dir = Path(frame_dir)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0

    def capture(self, camera_name, save=False):
        self.renderer.update_scene(self.data, camera=camera_name)
        rgb = self.renderer.render()
        if save:
            path = self.frame_dir / f"{camera_name}_{self.frame_count:04d}.png"
            Image.fromarray(rgb).save(path)
        return rgb

    def capture_stereo(self, save=False):
        left = self.capture("left_camera", save=save)
        right = self.capture("right_camera", save=save)
        if save:
            self.frame_count += 1
        return {"left": left, "right": right}

    def close(self):
        self.renderer.close()


# ============================================================================
# Stereo Vision Processor
# ============================================================================
class StereoVisionProcessor:
    """
    Binocular vision processor:
    - Detect red apple in both camera views
    - Triangulate 3D position from two rays
    - Generate point cloud from disparity map
    - Compute distance and bearing to apple
    """

    def __init__(self, model, data, res=CAMERA_RES, baseline=BASELINE, fovy=FOVY):
        self.model = model
        self.data = data
        self.res_h, self.res_w = res
        self.baseline = baseline
        self.fovy = fovy
        self.fx, self.fy, self.cx, self.cy = _camera_intrinsics(res[0], res[1], fovy)

    # -------------------------------------------------------------------------
    # Red-object detection
    # -------------------------------------------------------------------------
    def detect_red_object(self, rgb, score_thresh=80, grow_ratio=0.7,
                          min_pixels=10, max_pixels=8000):
        """Detect red apple using score-based flood fill."""
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        red_score = r.astype(float) - 0.5 * g.astype(float) - 0.5 * b.astype(float)
        max_idx = np.unravel_index(np.argmax(red_score), red_score.shape)
        cy, cx = max_idx
        max_score = red_score[cy, cx]
        if max_score < score_thresh:
            return None

        h, w = red_score.shape
        for ratio in [grow_ratio, grow_ratio + 0.1, grow_ratio + 0.2, grow_ratio + 0.3]:
            threshold = max(max_score * ratio, score_thresh * 0.8)
            mask = red_score > threshold
            visited = np.zeros_like(mask, dtype=bool)
            queue = [(cx, cy)]
            visited[cy, cx] = True
            component = [(cx, cy)]
            while queue:
                x, y = queue.pop(0)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        component.append((nx, ny))
                        queue.append((nx, ny))
            if min_pixels <= len(component) <= max_pixels:
                cx_m = int(np.mean([p[0] for p in component]))
                cy_m = int(np.mean([p[1] for p in component]))
                return {"center": (cx_m, cy_m), "pixels": len(component), "max_score": max_score}
        return None

    # -------------------------------------------------------------------------
    # Pixel -> world ray
    # -------------------------------------------------------------------------
    def pixel_to_world_ray(self, camera_name, cx, cy):
        """Convert pixel coordinate to world-space ray (origin, direction)."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        fovy = self.model.cam_fovy[cam_id] * np.pi / 180
        x_ndc = (cx - self.res_w / 2) / (self.res_w / 2)
        y_ndc = -(cy - self.res_h / 2) / (self.res_h / 2)
        tan_fovy2 = np.tan(fovy / 2)
        aspect = self.res_w / self.res_h
        dx = x_ndc * tan_fovy2 * aspect
        dy = y_ndc * tan_fovy2
        dz = -1.0
        ray_cam = np.array([dx, dy, dz])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        R = self.data.cam_xmat[cam_id].reshape(3, 3)
        ray_world = R @ ray_cam
        cam_pos = self.data.cam_xpos[cam_id].copy()
        return cam_pos, ray_world

    # -------------------------------------------------------------------------
    # Triangulation (most accurate)
    # -------------------------------------------------------------------------
    @staticmethod
    def triangulate(cam1_pos, ray1, cam2_pos, ray2):
        """
        Find the midpoint of the closest points on two 3D rays.
        Returns (midpoint, ray_distance).
        """
        w0 = cam1_pos - cam2_pos
        a = np.dot(ray1, ray1)
        b = np.dot(ray1, ray2)
        c = np.dot(ray2, ray2)
        d = np.dot(ray1, w0)
        e = np.dot(ray2, w0)
        denom = a * c - b * b
        if abs(denom) < 1e-10:
            t1 = t2 = 0.0
        else:
            t1 = (b * e - c * d) / denom
            t2 = (a * e - b * d) / denom
        p1 = cam1_pos + t1 * ray1
        p2 = cam2_pos + t2 * ray2
        midpoint = (p1 + p2) / 2.0
        ray_dist = np.linalg.norm(p1 - p2)
        return midpoint, ray_dist

    def estimate_apple_position_triangulation(self, stereo_frames):
        """
        Detect apple in both views and triangulate its 3D position.
        Returns (result_dict, left_det, right_det).
        """
        left_rgb = stereo_frames["left"]
        right_rgb = stereo_frames["right"]

        left_det = self.detect_red_object(left_rgb, score_thresh=80)
        right_det = self.detect_red_object(right_rgb, score_thresh=80)

        if left_det is None or right_det is None:
            return None, left_det, right_det

        cam1_pos, ray1 = self.pixel_to_world_ray("left_camera", *left_det["center"])
        cam2_pos, ray2 = self.pixel_to_world_ray("right_camera", *right_det["center"])
        apple_pos, ray_dist = self.triangulate(cam1_pos, ray1, cam2_pos, ray2)

        # Compute horizontal disparity in pixels
        disparity = left_det["center"][0] - right_det["center"][0]

        return {
            "position": apple_pos,
            "ray_distance": ray_dist,
            "disparity": disparity,
            "left_detection": left_det,
            "right_detection": right_det,
        }, left_det, right_det

    # -------------------------------------------------------------------------
    # Disparity & depth map (for point-cloud generation)
    # -------------------------------------------------------------------------
    def compute_disparity_depth(self, stereo_frames, block_size=7, max_disp=48):
        """
        Simple block-matching stereo to compute disparity and depth maps.
        Returns (disparity_map, depth_map).
        """
        left_gray = np.mean(stereo_frames["left"], axis=2).astype(np.uint8)
        right_gray = np.mean(stereo_frames["right"], axis=2).astype(np.uint8)

        h, w = left_gray.shape
        disparity = np.zeros((h, w), dtype=np.float32)
        half = block_size // 2

        for y in range(half, h - half):
            for x in range(half + max_disp, w - half):
                best_disp = 0
                best_sad = float("inf")
                left_block = left_gray[y - half:y + half + 1,
                                       x - half:x + half + 1].astype(float)
                for d in range(max_disp):
                    xr = x - d
                    if xr < half:
                        break
                    right_block = right_gray[y - half:y + half + 1,
                                             xr - half:xr + half + 1].astype(float)
                    sad = np.sum(np.abs(left_block - right_block))
                    if sad < best_sad:
                        best_sad = sad
                        best_disp = d
                # Confidence filter
                if best_sad < block_size * block_size * 50:
                    disparity[y, x] = float(best_disp)

        depth = np.zeros_like(disparity)
        valid = disparity > 0.5
        depth[valid] = (self.fx * self.baseline) / disparity[valid]
        return disparity, depth

    # -------------------------------------------------------------------------
    # Point-cloud generation
    # -------------------------------------------------------------------------
    def generate_point_cloud(self, stereo_frames, mask=None, stride=2):
        """
        Generate a 3D point cloud from stereo disparity.
        Args:
            stereo_frames: {"left": rgb, "right": rgb}
            mask: optional bool mask to filter pixels
            stride: subsample every N pixels for speed
        Returns:
            points (N,3), colors (N,3)
        """
        disparity, depth = self.compute_disparity_depth(stereo_frames)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_camera")
        R = self.data.cam_xmat[cam_id].reshape(3, 3)
        cam_pos = self.data.cam_xpos[cam_id].copy()

        h, w = depth.shape
        left_rgb = stereo_frames["left"]

        if mask is None:
            mask = np.ones((h, w), dtype=bool)

        points = []
        colors = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                if not mask[y, x] or depth[y, x] <= 0.01:
                    continue
                Z = depth[y, x]
                X_cam = (x - self.cx) * Z / self.fx
                Y_cam = (y - self.cy) * Z / self.fy
                Z_cam = -Z  # camera looks along -Z
                p_world = R @ np.array([X_cam, Y_cam, Z_cam]) + cam_pos
                points.append(p_world)
                colors.append(left_rgb[y, x])

        if len(points) == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        return np.array(points), np.array(colors)

    # -------------------------------------------------------------------------
    # Apple localization from point cloud
    # -------------------------------------------------------------------------
    def estimate_apple_from_pointcloud(self, stereo_frames):
        """
        Detect red region in left image, then generate point cloud by
        intersecting each pixel's ray with the apple sphere.
        The apple center is first found via triangulation, then each
        red pixel's ray is intersected with a sphere of radius 0.0375m.

        Returns result dict or None.
        """
        # 1. Get apple center from triangulation (most reliable)
        tri_result, left_det, right_det = self.estimate_apple_position_triangulation(stereo_frames)

        # If triangulation fails, try single-camera ground-plane intersection
        if tri_result is None:
            best_det = None
            best_cam = None
            for cam_name, det in [("left_camera", left_det), ("right_camera", right_det)]:
                if det is not None and det["max_score"] > 100:
                    if best_det is None or det["max_score"] > best_det["max_score"]:
                        best_det = det
                        best_cam = cam_name
            if best_det is None:
                return None
            origin, direction = self.pixel_to_world_ray(best_cam, *best_det["center"])
            if abs(direction[2]) < 1e-6:
                return None
            t = (0.0775 - origin[2]) / direction[2]
            if t <= 0:
                return None
            apple_center = origin + t * direction
        else:
            apple_center = tri_result["position"]
            left_det = tri_result.get("left_detection", left_det)

        apple_radius = 0.0375
        left_rgb = stereo_frames["left"]

        # 2. Build red-region mask
        r, g, b = left_rgb[:, :, 0], left_rgb[:, :, 1], left_rgb[:, :, 2]
        red_score = r.astype(float) - 0.5 * g.astype(float) - 0.5 * b.astype(float)
        if left_det is not None:
            threshold = left_det["max_score"] * 0.55
        else:
            threshold = 80
        mask = red_score > threshold

        # 3. For each pixel in mask, intersect ray with apple sphere
        points = []
        colors = []
        h, w = mask.shape

        for y in range(h):
            for x in range(w):
                if not mask[y, x]:
                    continue
                origin, direction = self.pixel_to_world_ray("left_camera", x, y)
                # Ray-sphere intersection: |origin + t*direction - center|^2 = r^2
                oc = origin - apple_center
                a = np.dot(direction, direction)
                b_ray = 2.0 * np.dot(oc, direction)
                c_ray = np.dot(oc, oc) - apple_radius ** 2
                discriminant = b_ray * b_ray - 4 * a * c_ray
                if discriminant >= 0:
                    t = (-b_ray - np.sqrt(discriminant)) / (2 * a)  # nearer intersection
                    if t > 0:
                        point = origin + t * direction
                        points.append(point)
                        colors.append(left_rgb[y, x])

        if len(points) == 0:
            return None

        points = np.array(points)
        apple_pos = np.mean(points, axis=0)
        dists = np.linalg.norm(points - apple_pos, axis=1)

        return {
            "position": apple_pos,
            "point_count": len(points),
            "std": float(np.std(dists)),
            "detection": left_det,
            "points": points,
            "colors": np.array(colors),
        }

    # -------------------------------------------------------------------------
    # Bearing / azimuth calculation
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_bearing(apple_pos, ee_pos, ee_xmat=None):
        """
        Compute distance and bearing angles from gripper to apple.

        Returns dict with:
            distance: Euclidean distance (m)
            dx, dy, dz: relative offsets in world frame
            azimuth: horizontal angle in radians (-pi..pi), 0 = forward (+X)
            elevation: vertical angle in radians (-pi/2..pi/2), 0 = level
        """
        rel = np.array(apple_pos) - np.array(ee_pos)
        dist = float(np.linalg.norm(rel))
        dx, dy, dz = rel

        # Azimuth: angle in XY plane, 0 = +X, positive = CCW toward +Y
        azimuth = float(np.arctan2(dy, dx))

        # Elevation: angle above/below XY plane
        horiz = np.sqrt(dx * dx + dy * dy)
        elevation = float(np.arctan2(dz, horiz))

        result = {
            "distance": dist,
            "dx": float(dx), "dy": float(dy), "dz": float(dz),
            "azimuth": azimuth,
            "elevation": elevation,
            "azimuth_deg": float(np.degrees(azimuth)),
            "elevation_deg": float(np.degrees(elevation)),
        }

        # If gripper orientation provided, compute bearing in gripper frame
        if ee_xmat is not None:
            R = np.array(ee_xmat).reshape(3, 3)
            rel_local = R.T @ rel
            dx_l, dy_l, dz_l = rel_local
            result["dx_local"] = float(dx_l)
            result["dy_local"] = float(dy_l)
            result["dz_local"] = float(dz_l)
            result["azimuth_local"] = float(np.arctan2(dy_l, dx_l))
            result["elevation_local"] = float(np.arctan2(dz_l, np.sqrt(dx_l * dx_l + dy_l * dy_l)))

        return result

    # -------------------------------------------------------------------------
    # Observation formatting for controllers
    # -------------------------------------------------------------------------
    def format_observation(self, stereo_frames, ee_pos, ee_xmat=None):
        """Format all stereo vision results as a text report."""
        tri_result, left_det, right_det = self.estimate_apple_position_triangulation(stereo_frames)
        pc_result = self.estimate_apple_from_pointcloud(stereo_frames)

        lines = []
        lines.append("=== 双目视觉感知 ===")
        lines.append(f"基线: {self.baseline * 1000:.1f}mm | 焦距 fx={self.fx:.1f}px | 分辨率 {self.res_w}x{self.res_h}")

        if left_det:
            cx, cy = left_det["center"]
            lines.append(f"  左相机: 苹果在像素 ({cx}, {cy}), 面积={left_det['pixels']} px")
        else:
            lines.append("  左相机: 未检测到苹果")

        if right_det:
            cx, cy = right_det["center"]
            lines.append(f"  右相机: 苹果在像素 ({cx}, {cy}), 面积={right_det['pixels']} px")
        else:
            lines.append("  右相机: 未检测到苹果")

        if tri_result:
            pos = tri_result["position"]
            bear = self.compute_bearing(pos, ee_pos, ee_xmat)
            lines.append(f"  [三角测量] 苹果位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
            lines.append(f"             视差: {tri_result['disparity']:.1f} px | 光线误差: {tri_result['ray_distance']:.4f} m")
            lines.append(f"             距离: {bear['distance']:.4f} m")
            lines.append(f"             水平方位角: {bear['azimuth_deg']:.1f} deg | 俯仰角: {bear['elevation_deg']:.1f} deg")
            if "azimuth_local" in bear:
                lines.append(f"             夹爪局部方位角: {np.degrees(bear['azimuth_local']):.1f} deg")

        if pc_result:
            pos = pc_result["position"]
            bear = self.compute_bearing(pos, ee_pos, ee_xmat)
            lines.append(f"  [点云测距] 苹果位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
            lines.append(f"             点云: {pc_result['point_count']} 点 | 标准差: {pc_result['std']:.4f} m")
            lines.append(f"             距离: {bear['distance']:.4f} m")
            lines.append(f"             水平方位角: {bear['azimuth_deg']:.1f} deg | 俯仰角: {bear['elevation_deg']:.1f} deg")

        if tri_result is None and pc_result is None:
            lines.append("  无法从双目视觉估计苹果位置")

        return "\n".join(lines)
