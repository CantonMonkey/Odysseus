"""vlm_frontier.py — 3D waypoint projection to image plane for VLM visual prompting."""
import numpy as np
import cv2

# Habitat 640x480, hfov=90 deg -> fx=fy=320, cx=320, cy=240
FX = FY = 320.0
CX, CY  = 320.0, 240.0
EYE_HEIGHT = 1.0  # camera height above robot base (m)


def project_waypoint(world_pos, robot_pos, R):
    """Project a 3D world-frame position to image pixel coordinates.

    R: 3x3 rotation matrix (local->world) from get_rotation_matrix().
    Returns (u, v) int pixel or None if behind camera / outside frame.

    Habitat convention: local -Z = forward, Y = up.
    Camera at robot_pos + R @ [0, EYE_HEIGHT, 0].
    """
    cam_pos = robot_pos + R @ np.array([0.0, EYE_HEIGHT, 0.0])
    P_rel   = np.array(world_pos, dtype=np.float64) - cam_pos
    P_local = R.T @ P_rel        # world -> local frame

    Zc = -P_local[2]             # -Z=forward -> positive depth
    if Zc <= 0.3:
        return None

    u = FX * P_local[0] / Zc + CX
    v = FY * (-P_local[1]) / Zc + CY  # Y up local = Y down image

    if 15 <= u <= 625 and 15 <= v <= 465:
        return (int(u), int(v))
    return None


def annotate_frame(frame, points_uv, labels):
    """Draw numbered green circles on RGB frame (copy, no mutation).

    points_uv: [(u,v), ...]
    labels:    ["1","2",...] or ["A","B",...]
    """
    out = frame.copy()
    for (u, v), label in zip(points_uv, labels):
        cv2.circle(out, (u, v), 18, (0, 220, 0), 3)
        cv2.putText(out, label, (u - 7, v + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    return out
