"""
loop.py
内层控制循环（Control Loop）：observe → perceive → plan → execute → validate
外层只需调用 run_task(env, task, scene_dir, on_frame, llm_perceive)。
"""

import numpy as np
import habitat_sim
from typing import Callable, Optional

from agent.semantic_map import query_target, DATA_DIR

MAX_STEPS = 300
SCENE_DIR = str(DATA_DIR / "00800-TEEsavR23oF")
ARRIVE_DIST = 1.2


def _nearest_reachable(env, candidates: list) -> Optional[list]:
    """
    从 semantic_map 候选坐标中选取 pathfinder 可达且 geodesic 最近的目标。
    返回 snap 到 navmesh 后的坐标，或 None。
    """
    pf = env._sim.pathfinder
    robot_pos = env.get_robot_pose()[0].astype(np.float32)
    best_dist = float("inf")
    best_target = None

    for cand in candidates:
        tgt = np.array(cand, dtype=np.float32)
        snapped = pf.snap_point(tgt)
        if np.any(np.isnan(snapped)):
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start = robot_pos
        path.requested_end = snapped
        if pf.find_path(path) and path.geodesic_distance < best_dist:
            best_dist = path.geodesic_distance
            best_target = snapped.tolist()

    return best_target


def _init_nav_state(env, task: str, scene_dir: str) -> dict:
    candidates = query_target(scene_dir, task)
    target_pos = _nearest_reachable(env, candidates) if candidates else None

    return {
        "goal": task,
        "target_pos": target_pos,
        "step_count": 0,
        "current_skill": "follow_path" if target_pos else "search_room",
        "done": False,
        "last_frame": None,
        "last_percept": {},
        "waypoints": [],
        "search_rotated": 0.0,
    }


def run_task(
    env,
    task: str,
    scene_dir: str = SCENE_DIR,
    on_frame: Optional[Callable[[np.ndarray, dict], None]] = None,
    llm_perceive=None,
) -> dict:
    """
    执行单个导航任务，阻塞直到完成或超时。
    env: 已 reset 的 HabitatEnv 实例
    task: 中文目标词，如 "沙发"
    on_frame: 每步回调 (rgb_frame, nav_state)，用于 WebSocket 推流
    llm_perceive: (frame, goal) -> {target_visible, distance} 可选
    """
    from agent.skills import follow_path, search_room, verify_arrival

    frame = env.get_frame()
    nav_state = _init_nav_state(env, task, scene_dir)
    nav_state["last_frame"] = frame

    if on_frame:
        on_frame(frame, nav_state)

    skill_map = {
        "follow_path":    follow_path,
        "search_room":    search_room,
        "verify_arrival": verify_arrival,
    }

    while not nav_state["done"] and nav_state["step_count"] < MAX_STEPS:
        # PERCEIVE（LLM 可选）
        if llm_perceive is not None:
            try:
                percept = llm_perceive(nav_state["last_frame"], task)
                nav_state["last_percept"] = percept
                if percept.get("target_visible") and nav_state["current_skill"] == "search_room":
                    nav_state["current_skill"] = "follow_path"
            except Exception:
                pass

        current = nav_state.get("current_skill", "follow_path")
        if current == "done":
            nav_state["done"] = True
            break

        skill_fn = skill_map.get(current, follow_path)
        nav_state = skill_fn(env, nav_state)

        if on_frame and nav_state.get("last_frame") is not None:
            on_frame(nav_state["last_frame"], nav_state)

    if nav_state["step_count"] >= MAX_STEPS and not nav_state["done"]:
        nav_state["timeout"] = True

    return nav_state


def demo(scene_dir: str = SCENE_DIR, target: str = "沙发"):
    """本地测试：运行控制循环，每步保存截图。"""
    import imageio
    from pathlib import Path
    from agent.habitat_env import HabitatEnv

    out_dir = Path("/tmp/loop_frames")
    out_dir.mkdir(exist_ok=True)

    env = HabitatEnv(gpu_id=0)
    env.reset(scene_dir)

    frame_idx = [0]

    def save_frame(frame: np.ndarray, state: dict):
        path = out_dir / f"frame_{frame_idx[0]:04d}.png"
        imageio.imwrite(str(path), frame)
        skill = state.get("current_skill", "?")
        step = state.get("step_count", 0)
        dist_str = ""
        if state.get("target_pos"):
            pos, _ = env.get_robot_pose()
            from agent.skills import _euclidean
            d = _euclidean(pos, state["target_pos"])
            dist_str = f", dist={d:.2f}m"
        if step % 20 == 0 or step < 5:
            print(f"  step={step:03d} skill={skill}{dist_str}")
        frame_idx[0] += 1

    print(f"Task: 导航到 '{target}'")
    result = run_task(env, target, scene_dir=scene_dir, on_frame=save_frame)

    env.close()
    status = "到达" if result["done"] else ("超时" if result.get("timeout") else "未完成")
    print(f"\n{status}: steps={result['step_count']}, target={result.get('target_pos')}")
    print(f"截图已保存到 {out_dir}/ ({frame_idx[0]} 帧)")
    return result


if __name__ == "__main__":
    demo()
