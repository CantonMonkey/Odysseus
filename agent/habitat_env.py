"""
habitat_env.py

Thin Habitat-Sim 0.3.1 wrapper for headless EGL rendering.
Loads an HM3D .basis.glb scene and exposes step / get_frame / get_robot_pose
for the control loop to call.

Robot types
-----------
"fetch"  (default) — Fetch robot specs: Intel RealSense D435i head camera at
    1.3 m, 69° HFOV, base radius 0.32 m.  Matches the hab_fetch URDF shipped
    with habitat-lab / habitat-sim datasets.
"simple" — legacy floating-camera agent (1.0 m, 90° HFOV, tiny capsule).
    Kept for backward-compatibility with pre-Fetch eval runs.

Optional visual body
--------------------
Set load_fetch_urdf=True (or FETCH_URDF_PATH env var) to render the Fetch
URDF mesh in the scene.  Requires downloading the hab_fetch dataset:
    python -m habitat_sim.utils.datasets_download --uids hab_fetch
Physics is set to KINEMATIC so the body tracks the agent without simulation.
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
FORWARD_STEP = 0.25   # metres per move_forward step
TURN_DEG     = 15.0   # degrees per turn step

DATA_DIR = Path(os.environ.get("VLN_DATA_DIR", "/data3/liangjy/vln/data/hm3d"))

# Per-robot kinematic + sensor specs.
# "fetch" values match the real Fetch robot used in Habitat benchmarks:
#   head_rgb/depth_sensor at 1.3 m, RealSense D435i 69° HFOV (640×480 mode),
#   collision capsule h=1.501 m r=0.32 m.
ROBOT_CONFIGS: dict = {
    "fetch": {
        "eye_height":   1.3,    # head camera height above navmesh floor (m)
        "agent_height": 1.501,  # collision capsule height (m)
        "agent_radius": 0.32,   # base footprint radius (m)
        "hfov":         69,     # RealSense D435i HFOV at 640×480
    },
    "simple": {
        "eye_height":   1.0,
        "agent_height": 0.88,
        "agent_radius": 0.18,
        "hfov":         90,
    },
}

# Default path for Fetch URDF (override with FETCH_URDF_PATH env var).
# Populated by:  python -m habitat_sim.utils.datasets_download --uids hab_fetch
_FETCH_URDF = os.environ.get(
    "FETCH_URDF_PATH",
    str(DATA_DIR.parent / "robots/hab_fetch/robots/hab_fetch.urdf"),
)


# ---------------------------------------------------------------------------
# Simulator configuration
# ---------------------------------------------------------------------------

def _make_config(
    scene_glb: str,
    gpu_id: int = 0,
    robot_type: str = "fetch",
    load_fetch_urdf: bool = False,
) -> habitat_sim.Configuration:
    rc = ROBOT_CONFIGS.get(robot_type, ROBOT_CONFIGS["fetch"])
    eye_h = rc["eye_height"]

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id       = scene_glb
    sim_cfg.gpu_device_id  = gpu_id
    # Physics must be on to load the URDF visual body; otherwise keep it off
    # for speed (no rigid-body solver ticks).
    sim_cfg.enable_physics = load_fetch_urdf
    sim_cfg.allow_sliding  = True

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid        = "color"
    rgb_spec.sensor_type = SensorType.COLOR
    rgb_spec.resolution  = [IMG_H, IMG_W]
    rgb_spec.position    = [0.0, eye_h, 0.0]
    rgb_spec.hfov        = rc["hfov"]

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid        = "depth"
    depth_spec.sensor_type = SensorType.DEPTH
    depth_spec.resolution  = [IMG_H, IMG_W]
    depth_spec.position    = [0.0, eye_h, 0.0]
    depth_spec.hfov        = rc["hfov"]

    overhead_spec = habitat_sim.CameraSensorSpec()
    overhead_spec.uuid        = "overhead"
    overhead_spec.sensor_type = SensorType.COLOR
    overhead_spec.resolution  = [320, 320]
    overhead_spec.position    = [0.0, 6.0, 0.0]  # 6 m above agent origin (4 m was inside ceiling)
    # orientation is Euler XYZ in radians; pitch -90° looks straight down
    overhead_spec.orientation = np.array([-np.pi / 2, 0.0, 0.0])

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = rc["agent_height"]
    agent_cfg.radius = rc["agent_radius"]
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec, overhead_spec]
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


# ---------------------------------------------------------------------------
# Fetch URDF visual body helpers
# ---------------------------------------------------------------------------

def _try_load_fetch_body(sim: habitat_sim.Simulator, urdf_path: str):
    """Load the Fetch URDF as a kinematic visual object.  Returns the AO or None."""
    if not Path(urdf_path).exists():
        print(f"[Fetch] URDF not found at {urdf_path} — visual body skipped", flush=True)
        print("[Fetch] Download with: python -m habitat_sim.utils.datasets_download --uids hab_fetch", flush=True)
        return None
    try:
        ao_mgr = sim.get_articulated_object_manager()
        robot = ao_mgr.add_articulated_object_from_urdf(
            filepath=urdf_path,
            fixed_base=False,
        )
        robot.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        return robot
    except Exception as exc:
        print(f"[Fetch] URDF load failed: {exc} — visual body skipped", flush=True)
        return None


def _sync_fetch_body(robot, agent_state) -> None:
    """Align the Fetch visual body to the current agent state."""
    try:
        import quaternion as npq  # numpy-quaternion
        pos = agent_state.position
        q   = agent_state.rotation   # numpy quaternion (w, x, y, z order)

        # habitat-sim ArticulatedObject takes magnum types; numpy arrays also work
        robot.translation = pos.tolist()

        # Convert numpy-quaternion → magnum.Quaternion
        import magnum as mn
        robot.rotation = mn.Quaternion(
            mn.Vector3(float(q.x), float(q.y), float(q.z)),
            float(q.w),
        )
    except Exception:
        pass  # gracefully skip sync if magnum/quaternion unavailable


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class HabitatEnv:
    """
    Lightweight Habitat-Sim wrapper used by the control loop.

    Parameters
    ----------
    gpu_id       : GPU index for EGL rendering.
    robot_type   : "fetch" (default) or "simple".  See ROBOT_CONFIGS.
    load_fetch_urdf : If True, loads the Fetch URDF for visual body rendering.
                     Requires the hab_fetch dataset and enables physics.

    Public API
    ----------
    reset()  – load (or reload) scene, place agent on the navmesh.
    step()   – execute one action, return (rgb_frame, done).
    """

    def __init__(
        self,
        gpu_id: int = 0,
        robot_type: str = "fetch",
        load_fetch_urdf: bool = False,
    ):
        self._gpu_id          = gpu_id
        self._robot_type      = robot_type
        # Auto-enable URDF rendering when the file is present
        self._load_fetch_urdf = load_fetch_urdf or Path(_FETCH_URDF).exists()
        self._sim: Optional[habitat_sim.Simulator] = None
        self._scene_glb: Optional[str] = None
        self._fetch_body = None   # ArticulatedObject for visual body (or None)
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
            self._sim.reset()
        else:
            if self._sim is not None:
                self._sim.close()
            cfg = _make_config(
                glb, self._gpu_id, self._robot_type, self._load_fetch_urdf
            )
            self._sim = habitat_sim.Simulator(cfg)
            self._scene_glb = glb

            if self._load_fetch_urdf:
                self._fetch_body = _try_load_fetch_body(self._sim, _FETCH_URDF)

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

        if self._fetch_body is not None:
            _sync_fetch_body(self._fetch_body, agent.get_state())

        self.done = False
        return self._obs()

    def step(self, action: str) -> Tuple[np.ndarray, bool]:
        """Execute *action* (one of ACTION_*) and return (rgb_frame, done)."""
        if action == ACTION_STOP:
            self.done = True
            return self._obs(), True

        self._sim.step(action)

        if self._fetch_body is not None:
            _sync_fetch_body(self._fetch_body, self._sim.get_agent(0).get_state())

        return self._obs(), False

    def get_frame(self) -> np.ndarray:
        """Return the current RGB frame as (H, W, 3) uint8."""
        return self._obs()

    def get_depth(self) -> np.ndarray:
        """Return the current depth frame as (H, W) float32, metres."""
        obs = self._sim.get_sensor_observations()
        return obs["depth"].astype(np.float32)

    def get_overhead_frame(self) -> Optional[np.ndarray]:
        """Return a top-down RGB view (320×320×3 uint8) from the overhead sensor."""
        obs = self._sim.get_sensor_observations()
        if "overhead" not in obs:
            return None
        return obs["overhead"][:, :, :3]

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
        pos = state.position
        q   = state.rotation
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
        self._fetch_body = None


    def _obs(self) -> np.ndarray:
        obs = self._sim.get_sensor_observations()
        rgba = obs["color"]      # (H, W, 4) uint8
        return rgba[:, :, :3]   # drop alpha channel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_to_heading(q) -> float:
    """Convert a Habitat quaternion to a heading angle in degrees."""
    try:
        import quaternion as npq
        angles = npq.as_euler_angles(q)
        yaw_rad = float(angles[1])
    except Exception:
        w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
        yaw_rad = np.arctan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
    return float(np.degrees(yaw_rad))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import imageio

    scene = str(DATA_DIR / "00800-TEEsavR23oF")
    print(f"Loading scene: {scene}")
    env = HabitatEnv(gpu_id=0, robot_type="fetch")

    frame = env.reset(scene)
    print(f"Reset OK. Frame shape: {frame.shape}, dtype: {frame.dtype}")
    rc = ROBOT_CONFIGS["fetch"]
    print(f"Fetch config — eye_h={rc['eye_height']}m  hfov={rc['hfov']}°  "
          f"h={rc['agent_height']}m  r={rc['agent_radius']}m")

    pos, heading = env.get_robot_pose()
    print(f"Initial pose: pos={pos.round(3)}, heading={heading:.1f}°")

    imageio.imwrite("/tmp/habitat_fetch_step000.png", frame)
    print("Saved: /tmp/habitat_fetch_step000.png")

    actions = [ACTION_FORWARD, ACTION_FORWARD, ACTION_LEFT, ACTION_FORWARD,
               ACTION_RIGHT, ACTION_FORWARD, ACTION_FORWARD]
    for i, act in enumerate(actions, 1):
        frame, done = env.step(act)
        pos, heading = env.get_robot_pose()
        out = f"/tmp/habitat_fetch_step{i:03d}.png"
        imageio.imwrite(out, frame)
        print(f"Step {i} [{act:12s}] pos={pos.round(3)}, heading={heading:.1f}°")

    env.close()
    print("Done. Check /tmp/habitat_fetch_step*.png")
