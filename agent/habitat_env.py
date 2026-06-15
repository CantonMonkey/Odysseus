"""
habitat_env.py
Habitat-Sim 0.3.1 headless EGL 环境封装。
加载 HM3D .basis.glb 场景，提供 step/observe/get_frame 接口供控制循环调用。
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# 强制无显示（EGL headless）
os.environ.setdefault("DISPLAY", "")

import habitat_sim
from habitat_sim import SensorType

# 动作常量
ACTION_FORWARD = "move_forward"
ACTION_LEFT    = "turn_left"
ACTION_RIGHT   = "turn_right"
ACTION_STOP    = "stop"

# 传感器参数
IMG_H = 480
IMG_W = 640
EYE_HEIGHT = 1.0    # 轮式机器人摄像头高度（m）
FORWARD_STEP = 0.25  # 每步前进距离（m）
TURN_DEG = 15.0      # 每步转向角度（度）

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

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.18
    agent_cfg.sensor_specifications = [rgb_spec]
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
    轻量 Habitat-Sim 封装，供控制循环调用。
    reset() 加载场景并初始化机器人位置。
    step(action) 执行一步并返回 RGB 帧。
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
        加载（或重置）场景，返回初始 RGB 帧。
        scene_dir: 场景目录路径，如 /data3/.../00800-TEEsavR23oF
        start_pos: [x, y, z] 初始位置；None 表示随机放置在 navmesh 上
        """
        if scene_dir is None:
            scene_dir = str(DATA_DIR / "00800-TEEsavR23oF")

        scene_path = Path(scene_dir)
        scene_id = scene_path.name.split("-", 1)[1]
        glb = str(scene_path / f"{scene_id}.basis.glb")

        if self._sim is not None and self._scene_glb == glb:
            # 同场景重置：不重建仿真器
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
            pos = self._sim.pathfinder.get_random_navigable_point()
            state = habitat_sim.AgentState()
            state.position = pos
            agent.set_state(state)

        self.done = False
        return self._obs()

    def step(self, action: str) -> Tuple[np.ndarray, bool]:
        """
        执行动作，返回 (rgb_frame, done)。
        action 为 ACTION_* 常量之一。
        """
        if action == ACTION_STOP:
            self.done = True
            return self._obs(), True

        self._sim.step(action)
        return self._obs(), False

    def get_frame(self) -> np.ndarray:
        """返回当前 RGB 帧 (H, W, 3) uint8。"""
        return self._obs()

    def get_robot_pose(self) -> Tuple[np.ndarray, float]:
        """返回 (position_xyz, heading_degrees)。heading=0 朝 +Z，顺时针为正。"""
        state = self._sim.get_agent(0).get_state()
        pos = state.position  # np.ndarray [x, y, z]
        q = state.rotation    # quaternion
        # 从四元数提取 yaw（绕 Y 轴旋转）
        heading = _quat_to_heading(q)
        return pos, heading

    def navigable_point_near(self, target_pos: list, radius: float = 1.0) -> Optional[np.ndarray]:
        """在目标周围找到最近的可导航点。"""
        pf = self._sim.pathfinder
        if not pf.is_loaded:
            return None
        p = pf.snap_point(np.array(target_pos, dtype=np.float32))
        if np.isnan(p).any():
            return None
        return p

    def distance_to(self, target_pos: list) -> float:
        """机器人到目标的路径距离（走 navmesh 最短路）。"""
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
        return rgba[:, :, :3]     # drop alpha


def _quat_to_heading(q) -> float:
    """将 habitat quaternion 转换为 heading（度）。"""
    # habitat quaternion: (w, x, y, z) via numpy-quaternion or similar
    # yaw = atan2(2*(w*y + x*z), 1 - 2*(y^2 + z^2))
    try:
        import quaternion as npq
        angles = npq.as_euler_angles(q)
        yaw_rad = float(angles[1])
    except Exception:
        # fallback: extract from rotation matrix
        w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
        yaw_rad = np.arctan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
    return float(np.degrees(yaw_rad))


if __name__ == "__main__":
    import sys
    import imageio

    scene = str(DATA_DIR / "00800-TEEsavR23oF")
    print(f"Loading scene: {scene}")
    env = HabitatEnv(gpu_id=0)

    frame = env.reset(scene)
    print(f"Reset OK. Frame shape: {frame.shape}, dtype: {frame.dtype}")

    pos, heading = env.get_robot_pose()
    print(f"Initial pose: pos={pos.round(3)}, heading={heading:.1f}°")

    # 保存初始帧
    out = "/tmp/habitat_step000.png"
    imageio.imwrite(out, frame)
    print(f"Saved: {out}")

    # 执行几步动作
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
