"""
loop.py — Exploration loop with semantic topological map.

Pipeline:
  1. ExploreMap: online 2D occupancy + VLM value grid → frontier detection
  2. TopoMap: semantic topological graph — nodes annotated with room labels
     (built from classify_scene() every CLASSIFY_INTERVAL steps)
  3. Frontier selection biased by topo_map.suggest_goal_direction():
       goto      → navigate to known room node
       go_upstairs → find staircase via navmesh sampling
       explore   → default value-weighted frontier
  4. VLM perception every VLM_CALL_INTERVAL steps:
       target visible + confidence ≥ threshold → depth-estimate 3D pos
       → switch to follow_path → verify_arrival
"""

import numpy as np
from typing import Callable, Optional

from agent.explore_map import ExploreMap, VLM_CALL_INTERVAL
from agent.topo_map   import TopoMap

MAX_STEPS          = 300
ARRIVE_DIST        = 1.2
VLM_CONF_THRESHOLD = 0.55
CLASSIFY_INTERVAL  = 20   # room classification + topo node every N steps

from pathlib import Path
_DATA     = Path("/data3/liangjy/vln/data/hm3d")
SCENE_DIR = str(_DATA / "00800-TEEsavR23oF")


# ── 3-D target localisation ─────────────────────────────────────────────────

def _estimate_target_pos(depth_frame, direction, agent_pos, R, hfov=90.0):
    H, W   = depth_frame.shape
    fx     = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0
    col    = {"left": W // 4, "center": W // 2, "right": 3 * W // 4}.get(direction, W // 2)
    u, v   = col, int(cy)
    patch  = depth_frame[max(0,v-40):min(H,v+40), max(0,u-30):min(W,u+30)]
    valid  = patch[(patch > 0.3) & (patch < 7.0)]
    if len(valid) == 0:
        return None
    d     = float(np.median(valid))
    x_c   = (u - cx) * d / fx
    z_c   = -d
    p     = R @ np.array([x_c, 0.0, z_c]) + agent_pos + np.array([0, 1.0, 0])
    return p.astype(np.float32)


# ── frontier navigation skill ────────────────────────────────────────────────

def _explore_frontier(env, nav_state: dict, explore_map: ExploreMap,
                      topo_map: TopoMap) -> dict:
    from agent.skills import _replan, _get_forward, _turn_to, _euclidean
    from agent.skills import WP_REACH, ALIGN_THRESH
    from agent.habitat_env import ACTION_FORWARD, ACTION_LEFT

    robot_pos, _ = env.get_robot_pose()
    task          = nav_state.get("goal", "")

    # Arrive at current frontier → clear it
    f_pos = nav_state.get("frontier_pos")
    if f_pos is not None and _euclidean(robot_pos, np.array(f_pos)) < ARRIVE_DIST:
        nav_state["frontier_pos"] = None
        nav_state["waypoints"]    = []

    # Need a new frontier target
    if not nav_state.get("frontier_pos") or not nav_state.get("waypoints"):
        target = None

        # 1. Ask topo_map for commonsense direction
        hint = topo_map.suggest_goal_direction(task, robot_pos)
        action_type = hint.get("action", "explore")

        if action_type == "goto":
            target = hint["pos"]            # navigate to known room node

        elif action_type == "go_upstairs":
            stair = topo_map.find_staircase_approach(env, robot_pos)
            if stair is not None:
                target = stair

        # 2. Fallback: best VLM-scored frontier
        if target is None:
            target = explore_map.best_frontier(robot_pos)
            if target is None:
                # All frontiers exhausted: rotate in place
                frame, _ = env.step(ACTION_LEFT)
                nav_state["last_frame"]  = frame
                nav_state["step_count"] += 1
                return nav_state

        nav_state["frontier_pos"] = target.tolist() if hasattr(target, 'tolist') else list(target)
        nav_state["waypoints"]    = _replan(env, robot_pos, target)

    # Follow waypoints
    waypoints = nav_state.get("waypoints", [])
    while waypoints and _euclidean(robot_pos, np.array(waypoints[0])) < WP_REACH:
        waypoints.pop(0)

    if not waypoints:
        action = ACTION_LEFT
    else:
        to_wp          = np.array(waypoints[0]) - robot_pos
        forward        = _get_forward(env)
        angle, action  = _turn_to(forward, to_wp)
        if angle < ALIGN_THRESH:
            action = ACTION_FORWARD

    frame, _ = env.step(action)

    new_pos, _ = env.get_robot_pose()
    while waypoints and _euclidean(new_pos, np.array(waypoints[0])) < WP_REACH:
        waypoints.pop(0)

    nav_state["waypoints"]   = waypoints
    nav_state["last_frame"]  = frame
    nav_state["step_count"] += 1
    return nav_state


# ── main task loop ───────────────────────────────────────────────────────────

def run_task(
    env,
    task: str,
    scene_dir: str = SCENE_DIR,
    on_frame: Optional[Callable] = None,
    llm_perceive=None,
    max_steps: int = MAX_STEPS,
) -> dict:
    """
    Exploration-based ObjectNav.  No pre-computed semantic labels used.

    llm_perceive: function(frame, goal) → {target_visible, direction, distance, confidence}
                  Used for both target detection and scene classification.
    """
    from agent.skills    import follow_path, verify_arrival
    from agent.llm_agent import classify_scene

    explore_map = ExploreMap()
    topo_map    = TopoMap()

    nav_state: dict = {
        "goal":           task,
        "target_pos":     None,
        "step_count":     0,
        "current_skill":  "explore_frontier",
        "done":           False,
        "last_frame":     env.get_frame(),
        "last_percept":   {"target_visible": False, "confidence": 0.0},
        "waypoints":      [],
        "explore_map":    explore_map,
        "topo_map":       topo_map,
        "frontier_pos":   None,
        "vlm_step":       -VLM_CALL_INTERVAL,
        "classify_step":  -CLASSIFY_INTERVAL,
        "last_scene":     {},
    }

    if on_frame:
        on_frame(nav_state["last_frame"], nav_state)

    skill_map = {
        "follow_path":    follow_path,
        "verify_arrival": verify_arrival,
    }

    while not nav_state["done"] and nav_state["step_count"] < max_steps:
        step      = nav_state["step_count"]
        robot_pos, _ = env.get_robot_pose()
        R            = env.get_rotation_matrix()

        # ── VLM target detection (every VLM_CALL_INTERVAL steps) ──────
        if llm_perceive is not None and (step - nav_state["vlm_step"]) >= VLM_CALL_INTERVAL:
            try:
                percept = llm_perceive(nav_state["last_frame"], task)
                nav_state["last_percept"] = percept
                nav_state["vlm_step"]     = step
                confidence = float(percept.get("confidence", 0.0))
                if percept.get("target_visible") and confidence >= VLM_CONF_THRESHOLD:
                    depth = env.get_depth()
                    tgt   = _estimate_target_pos(
                        depth, percept.get("direction", "center"), robot_pos, R)
                    if tgt is not None:
                        nav_state["target_pos"]    = tgt.tolist()
                        nav_state["current_skill"] = "follow_path"
                        nav_state["waypoints"]     = []
            except Exception:
                pass

        # ── Scene classification → topo_map node (every CLASSIFY_INTERVAL) ──
        if (
            llm_perceive is not None
            and nav_state.get("target_pos") is None
            and (step - nav_state["classify_step"]) >= CLASSIFY_INTERVAL
        ):
            try:
                scene = classify_scene(nav_state["last_frame"], task)
                nav_state["last_scene"]      = scene
                nav_state["classify_step"]   = step
                room     = scene.get("room", "其他")
                objects  = scene.get("objects", [])
                topo_map.add_node(robot_pos, room, objects, step)
            except Exception:
                pass

        # ── Update value map ───────────────────────────────────────────
        vlm_score = float(nav_state["last_percept"].get("confidence", 0.0))
        explore_map.update(robot_pos, R, vlm_score)

        # ── Execute skill ──────────────────────────────────────────────
        current = nav_state.get("current_skill", "explore_frontier")
        if current == "done":
            nav_state["done"] = True
            break
        elif current in skill_map:
            nav_state = skill_map[current](env, nav_state)
        else:
            nav_state = _explore_frontier(env, nav_state, explore_map, topo_map)

        if on_frame and nav_state.get("last_frame") is not None:
            on_frame(nav_state["last_frame"], nav_state)

    if nav_state["step_count"] >= max_steps and not nav_state["done"]:
        nav_state["timeout"] = True

    return nav_state
