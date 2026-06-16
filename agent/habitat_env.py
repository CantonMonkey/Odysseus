"""
habitat_env.py

Thin Habitat-Sim 0.3.1 wrapper for headless EGL rendering.
Loads an HM3D .basis.glb scene and exposes step / get_frame / get_robot_pose
for the control loop to call.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Suppress X11 display lookup so EGL headless init succeeds
os.environ.setdefault("DISPLAY", "")

import habitat_sim
from habitat_sim import SensorType

# Action name constants
ACTION_FORWARD = "move_forward"
ACTION_LEFT    = "turn_left"
ACTION_RIGHT   = "turn_right"
ACTION_STOP    = "stop"

# Sensor / motion parameters
IMG_H        = 480
IMG_W        = 640
EYE_HEIGHT   = 1.0   # camera height on a wheeled robot (m)
FORWARD_STEP = 0.25  # distance per move_forward step (m)
TURN_DEG     = 15.0  # degrees per turn step

DATA_DIR = Path("/data3/liangjy/vln/data/hm3d")


def _make_config(scene_glb: str, gpu_id: int = 0) -> habitat_sim.Configuration:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    sim_cfg.gpu_device_id = gpu_id
    sim_cfg.enable_physics = False
    sim_cfg.allow_sliding = True

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color"
    rgb_spec.sensor_type = SensorType.COLOR
    rgb_spec.resolution = [IMG_H, IMG_W]
    rgb_spec.position = [0.0, EYE_HEIGHT, 0.0]
    rgb_spec.hfov = 90

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = SensorType.DEPTH
    depth_spec.resolution = [IMG_H, IMG_W]
    depth_spec.position = [0.0, EYE_HEIGHT, 0.0]
    depth_spec.hfov = 90

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.18
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    agent_cfg.action_space = {
        ACTION_FORWARD: habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=FORWARD_STEP),
        ),
        ACTION_LEFT: habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=TURN_DEG),
        ),
        ACTION_RIGHT: habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=TURN_DEG),
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


class HabitatEnv:
    """
    Lightweight Habitat-Sim wrapper used by the control loop.

    reset()  – load (or reload) scene, place agent on the navmesh.
    step()   – execute one action, return (rgb_frame, done).
    """

    def __init__(self, gpu_id: int = 0):
        self._gpu_id = gpu_id
        self._sim: Optional[habitat_sim.Simulator] = None
        self._scene_glb: Optional[str] = None
        self.done = False

    def reset(
        self,
        scene_dir: Optional[str] = None,
        start_pos: Optional[list] = None,
    ) -> np.ndarray:
        """
        Load or reload a scene and return the initial RGB frame.

        scene_dir  – path to the HM3D scene directory (e.g. .../00800-TEEsavR23oF)
        start_pos  – [x, y, z] spawn position; None → random navigable point
        """
        if scene_dir is None:
            scene_dir = str(DATA_DIR / "00800-TEEsavR23oF")

        scene_path = Path(scene_dir)
        scene_id = scene_path.name.split("-", 1)[1]
        glb = str(scene_path / f"{scene_id}.basis.glb")

        if self._sim is not None and self._scene_glb == glb:
            # Same scene: reset without rebuilding the simulator
            self._sim.reset()
        else:
            if self._sim is not None:
                self._sim.close()
            cfg = _make_config(glb, self._gpu_id)
            self._sim = habitat_sim.Simulator(cfg)
            self._scene_glb = glb

        agent = self._sim.initialize_agent(0)

        if start_pos is not None:
            state = habitat_sim.AgentState()
            state.position = np.array(start_pos, dtype=np.float32)
            agent.set_state(state)
        else:
            # Retry until we land on the ground floor (Y < 1.5 m).
            # Scene 00800 has a second floor at Y≈3.16 m which the navmesh
            # includes, but all navigable objects are on the ground floor.
            for _ in range(50):
                pos = self._sim.pathfinder.get_random_navigable_point()
                if pos[1] < 1.5:
                    break
            state = habitat_sim.AgentState()
            state.position = pos
            agent.set_state(state)

        self.done = False
        return self._obs()

    def step(self, action: str) -> Tuple[np.ndarray, bool]:
        """Execute *action* (one of ACTION_*) and return (rgb_frame, done)."""
        if action == ACTION_STOP:
            self.done = True
            return self._obs(), True

        self._sim.step(action)
        return self._obs(), False

    def get_frame(self) -> np.ndarray:
        """Return the current RGB frame as (H, W, 3) uint8."""
        return self._obs()

    def get_depth(self) -> np.ndarray:
        """Return the current depth frame as (H, W) float32, metres."""
        obs = self._sim.get_sensor_observations()
        return obs["depth"].astype(np.float32)

    def get_rotation_matrix(self) -> np.ndarray:
        """Return the 3x3 rotation matrix that transforms agent-local to world."""
        try:
            import quaternion as npq
            q = self._sim.get_agent(0).get_state().rotation
            return npq.as_rotation_matrix(q)
        except Exception:
            return np.eye(3)

    def get_robot_pose(self) -> Tuple[np.ndarray, float]:
        """Return (position_xyz, heading_degrees).

        heading=0 → facing -Z (default); increases counter-clockwise with turn_left.
        """
        state = self._sim.get_agent(0).get_state()
        pos = state.position  # np.ndarray [x, y, z]
        q = state.rotation
        heading = _quat_to_heading(q)
        return pos, heading

    def navigable_point_near(self, target_pos: list, radius: float = 1.0) -> Optional[np.ndarray]:
        """Snap *target_pos* to the nearest point on the navmesh."""
        pf = self._sim.pathfinder
        if not pf.is_loaded:
            return None
        p = pf.snap_point(np.array(target_pos, dtype=np.float32))
        if np.isnan(p).any():
            return None
        return p

    def distance_to(self, target_pos: list) -> float:
        """Geodesic (navmesh shortest-path) distance to *target_pos*."""
        pos, _ = self.get_robot_pose()
        path = habitat_sim.ShortestPath()
        path.requested_start = pos
        path.requested_end = np.array(target_pos, dtype=np.float32)
        self._sim.pathfinder.find_path(path)
        return path.geodesic_distance

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def _obs(self) -> np.ndarray:
        obs = self._sim.get_sensor_observations()
        rgba = obs["color"]       # (H, W, 4) uint8
        return rgba[:, :, :3]    # drop alpha channel


def _quat_to_heading(q) -> float:
    """Convert a Habitat quaternion to a heading angle in degrees."""
    try:
        import quaternion as npq
        angles = npq.as_euler_angles(q)
        yaw_rad = float(angles[1])
    except Exception:
        # Fallback: extract yaw from quaternion components directly
        w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
        yaw_rad = np.arctan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
    return float(np.degrees(yaw_rad))


if __name__ == "__main__":
    import imageio

    scene = str(DATA_DIR / "00800-TEEsavR23oF")
    print(f"Loading scene: {scene}")
    env = HabitatEnv(gpu_id=0)

    frame = env.reset(scene)
    print(f"Reset OK. Frame shape: {frame.shape}, dtype: {frame.dtype}")

    pos, heading = env.get_robot_pose()
    print(f"Initial pose: pos={pos.round(3)}, heading={heading:.1f}°")

    imageio.imwrite("/tmp/habitat_step000.png", frame)
    print("Saved: /tmp/habitat_step000.png")

    actions = [ACTION_FORWARD, ACTION_FORWARD, ACTION_LEFT, ACTION_FORWARD,
               ACTION_RIGHT, ACTION_FORWARD, ACTION_FORWARD]
    for i, act in enumerate(actions, 1):
        frame, done = env.step(act)
        pos, heading = env.get_robot_pose()
        out = f"/tmp/habitat_step{i:03d}.png"
        imageio.imwrite(out, frame)
        print(f"Step {i} [{act:12s}] pos={pos.round(3)}, heading={heading:.1f}°")

    env.close()
    print("Done. Check /tmp/habitat_step*.png")
