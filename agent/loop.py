"""
loop.py

Inner control loop (Control Loop):
  observe → perceive → plan → execute → validate

The outer Task Loop only needs to call run_task(env, task, ...) and block
until the task completes or times out.
"""

import numpy as np
import habitat_sim
from typing import Callable, Optional

from agent.semantic_map import query_target, DATA_DIR

MAX_STEPS  = 300
SCENE_DIR  = str(DATA_DIR / "00800-TEEsavR23oF")
ARRIVE_DIST = 1.2


def _nearest_reachable(env, candidates: list) -> Optional[list]:
    """Return the geodesically closest reachable candidate from *candidates*.

    Each candidate is snapped to the navmesh first; unreachable candidates
    (snap returns NaN or pathfinder finds no route) are skipped.
    """
    pf        = env._sim.pathfinder
    robot_pos = env.get_robot_pose()[0].astype(np.float32)
    best_dist = float("inf")
    best_target = None

    for cand in candidates:
        tgt     = np.array(cand, dtype=np.float32)
        snapped = pf.snap_point(tgt)
        if np.any(np.isnan(snapped)):
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start = robot_pos
        path.requested_end   = snapped
        if pf.find_path(path) and path.geodesic_distance < best_dist:
            best_dist   = path.geodesic_distance
            best_target = snapped.tolist()

    return best_target


def _init_nav_state(env, task: str, scene_dir: str) -> dict:
    candidates = query_target(scene_dir, task)
    target_pos = _nearest_reachable(env, candidates) if candidates else None

    return {
        "goal":          task,
        "target_pos":    target_pos,
        "step_count":    0,
        "current_skill": "follow_path" if target_pos else "search_room",
        "done":          False,
        "last_frame":    None,
        "last_percept":  {},
        "waypoints":     [],
        "search_rotated": 0.0,
    }


def run_task(
    env,
    task: str,
    scene_dir: str = SCENE_DIR,
    on_frame: Optional[Callable[[np.ndarray, dict], None]] = None,
    llm_perceive=None,
) -> dict:
    """Execute a single navigation task, blocking until done or timed out.

    env          – a HabitatEnv instance that has already been reset
    task         – Chinese goal keyword, e.g. "沙发"
    scene_dir    – path to the HM3D scene directory
    on_frame     – optional callback(frame, nav_state) called after each step
                   (used for WebSocket streaming)
    llm_perceive – optional function(frame, goal) → {target_visible, distance}
                   that augments rule-based navigation with vision perception

    Returns the final nav_state dict (check nav_state["done"]).
    """
    from agent.skills import follow_path, search_room, verify_arrival

    frame     = env.get_frame()
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
        # PERCEIVE: optional LLM vision pass
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

        skill_fn  = skill_map.get(current, follow_path)
        nav_state = skill_fn(env, nav_state)

        if on_frame and nav_state.get("last_frame") is not None:
            on_frame(nav_state["last_frame"], nav_state)

    if nav_state["step_count"] >= MAX_STEPS and not nav_state["done"]:
        nav_state["timeout"] = True

    return nav_state


def demo(scene_dir: str = SCENE_DIR, target: str = "沙发"):
    """Smoke-test the control loop: save one PNG per step to /tmp/loop_frames/."""
    import imageio
    from pathlib import Path
    from agent.habitat_env import HabitatEnv

    out_dir = Path("/tmp/loop_frames")
    out_dir.mkdir(exist_ok=True)

    env       = HabitatEnv(gpu_id=0)
    frame_idx = [0]

    env.reset(scene_dir)

    def save_frame(frame: np.ndarray, state: dict):
        path = out_dir / f"frame_{frame_idx[0]:04d}.png"
        imageio.imwrite(str(path), frame)
        step  = state.get("step_count", 0)
        skill = state.get("current_skill", "?")
        dist_str = ""
        if state.get("target_pos"):
            pos, _ = env.get_robot_pose()
            from agent.skills import _euclidean
            dist_str = f", dist={_euclidean(pos, state['target_pos']):.2f}m"
        if step % 20 == 0 or step < 5:
            print(f"  step={step:03d} skill={skill}{dist_str}")
        frame_idx[0] += 1

    print(f"Task: navigate to '{target}'")
    result = run_task(env, target, scene_dir=scene_dir, on_frame=save_frame)

    env.close()
    status = "arrived" if result["done"] else ("timeout" if result.get("timeout") else "incomplete")
    print(f"\n{status}: steps={result['step_count']}, target={result.get('target_pos')}")
    print(f"Frames saved to {out_dir}/ ({frame_idx[0]} total)")
    return result


if __name__ == "__main__":
    demo()
