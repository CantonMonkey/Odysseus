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

MAX_STEPS          = 500
ARRIVE_DIST        = 1.2
VLM_CONF_THRESHOLD = 0.25

from pathlib import Path
_DATA     = Path("/data3/liangjy/vln/data/hm3d")
SCENE_DIR = str(_DATA / "00800-TEEsavR23oF")


# ── 3-D target localisation ─────────────────────────────────────────────────

def _estimate_target_pos(depth_frame, direction, agent_pos, R, hfov=90.0,
                          vlm_dist: float = 99.0):
    """Back-project target direction to a world 3-D waypoint.

    VLM gives reliable direction (left/center/right) but poor distance.
    Depth sensor gives precise measurements but may have foreground clutter.

    Strategy: in the direction column, prefer the BACKGROUND depth cluster
    (85th percentile) so we navigate toward the target rather than the wall
    in front of it.  Fall back to a fixed-distance directional waypoint when
    depth has no valid far readings.
    """
    H, W   = depth_frame.shape
    fx     = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    # Sample a vertical strip in the target direction
    col_center = {"left": W // 4, "center": W // 2, "right": 3 * W // 4}.get(direction, W // 2)
    col_lo = max(0, col_center - 40)
    col_hi = min(W, col_center + 40)
    strip  = depth_frame[H//4 : 3*H//4, col_lo:col_hi]   # middle 50% vertically

    valid = strip[(strip > 0.5) & (strip < 8.0)].flatten()

    if len(valid) < 10:
        # No usable depth — use fixed 3m directional waypoint
        d = 3.0
    else:
        # Use 85th-percentile depth to prefer background over foreground clutter
        d = float(np.percentile(valid, 85))
        # Cap navigation waypoint: don't try to go more than 5m in one shot
        d = min(d, 5.0)

    x_c = (col_center - cx) * d / fx
    z_c = -d
    p   = R @ np.array([x_c, 0.0, z_c]) + agent_pos
    return p.astype(np.float32)


# ── frontier navigation skill ────────────────────────────────────────────────

def _explore_frontier(env, nav_state: dict, explore_map: ExploreMap,
                      topo_map: TopoMap) -> dict:
    from agent.skills import _replan, _get_forward, _turn_to, _euclidean
    from agent.skills import WP_REACH, ALIGN_THRESH
    from agent.habitat_env import ACTION_FORWARD, ACTION_LEFT

    robot_pos, _ = env.get_robot_pose()
    task          = nav_state.get("goal", "")

    # Countdown anchor (post-ESCAPE exploration focus area)
    if nav_state.get("anchor_steps_left", 0) > 0:
        nav_state["anchor_steps_left"] -= 1
        if nav_state["anchor_steps_left"] == 0:
            nav_state["explore_anchor"] = None

    # Arrive at current frontier → clear and blacklist (don't revisit)
    f_pos = nav_state.get("frontier_pos")
    if f_pos is not None and _euclidean(robot_pos, np.array(f_pos)) < ARRIVE_DIST:
        # Blacklist the frontier we just visited so we don't return to it
        f_arr = np.array(f_pos)
        fi, fj = explore_map._w2g(float(f_arr[0]), float(f_arr[2]))
        failed = nav_state.get("failed_frontiers", set())
        failed.add((fi, fj))
        nav_state["failed_frontiers"] = failed
        nav_state["frontier_pos"] = None
        nav_state["waypoints"]    = []

    # Need a new frontier target
    if not nav_state.get("frontier_pos") or not nav_state.get("waypoints"):
        target = None
        failed = nav_state.get("failed_frontiers", set())

        # 1. Ask topo_map for commonsense direction
        hint = topo_map.suggest_goal_direction(task, robot_pos)
        action_type = hint.get("action", "explore")

        if action_type == "goto":
            target = hint["pos"]            # navigate to known room node

        elif action_type == "go_upstairs":
            stair = topo_map.find_staircase_approach(env, robot_pos)
            if stair is not None:
                target = stair

        # 2. Fallback: best VLM-scored frontier (skip recently-failed ones)
        if target is None:
            cells = explore_map.frontiers()
            ri, rj = explore_map._w2g(robot_pos[0], robot_pos[2])
            pf_nav = env._sim.pathfinder
            best_score, best_ij = -1.0, None
            for i, j in cells:
                if (i, j) in failed:
                    continue
                wx, wz = explore_map._g2w(i, j)
                # Skip wall-boundary frontiers: if navmesh snap moves > 1.5m, it's in a wall
                cell_world = np.array([wx, robot_pos[1], wz], dtype=np.float32)
                snapped = pf_nav.snap_point(cell_world)
                if not np.any(np.isnan(snapped)):
                    snap_dist = float(np.linalg.norm(snapped[[0,2]] - cell_world[[0,2]]))
                    if snap_dist > 1.5:
                        failed.add((i, j))
                        continue
                v    = float(explore_map.value[i, j])
                dist = np.sqrt((i - ri)**2 + (j - rj)**2) * explore_map.res
                anchor = nav_state.get("explore_anchor")
                if anchor is not None and nav_state.get("anchor_steps_left", 0) > 0:
                    wx2, wz2 = explore_map._g2w(i, j)
                    da = np.sqrt((wx2 - anchor[0])**2 + (wz2 - anchor[2])**2)
                    s  = v + max(0.0, (8.0 - da) / 8.0) * 0.30
                else:
                    prox = max(0.0, (6.0 - dist) / 6.0) * 0.05
                    s    = v + prox
                if s > best_score:
                    best_score, best_ij = s, (i, j)
            if best_ij is not None:
                wx, wz = explore_map._g2w(*best_ij)
                target = np.array([wx, robot_pos[1], wz], dtype=np.float32)
            if target is None:
                # All frontiers exhausted or all failed: reset failed set and rotate
                nav_state["failed_frontiers"] = set()
                frame, _ = env.step(ACTION_LEFT)
                nav_state["last_frame"]  = frame
                nav_state["step_count"] += 1
                return nav_state

        wps = _replan(env, robot_pos, target)
        if not wps:
            # Frontier unreachable — blacklist it and skip
            t_arr = np.array(target)
            fi, fj = explore_map._w2g(float(t_arr[0]), float(t_arr[2]))
            failed.add((fi, fj))
            nav_state["failed_frontiers"] = failed
            nav_state["frontier_pos"] = None
            nav_state["waypoints"]    = []
            frame, _ = env.step(ACTION_LEFT)
            nav_state["last_frame"]  = frame
            nav_state["step_count"] += 1
            return nav_state

        nav_state["frontier_pos"]      = target.tolist() if hasattr(target, 'tolist') else list(target)
        nav_state["waypoints"]         = wps
        nav_state["failed_frontiers"]  = failed

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
        "last_expl":      0.0,
        "stagnant_steps": 0,
        "vlm_step":       -VLM_CALL_INTERVAL,
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

        # ── VLFM-style VLM call (every VLM_CALL_INTERVAL steps) ────────
        # Returns: target_visible, direction, confidence, room, relevance
        # relevance = scene-level expectation of finding goal here (0-1)
        # This updates the value map every call, not just when target is visible
        if llm_perceive is not None and (step - nav_state["vlm_step"]) >= VLM_CALL_INTERVAL:
            try:
                percept = llm_perceive(nav_state["last_frame"], task)
                nav_state["last_percept"] = percept
                nav_state["vlm_step"]     = step
                confidence = float(percept.get("confidence", 0.0))
                vis        = percept.get("target_visible", False)
                room       = percept.get("room", "其他")
                print(f"  [VLM step={step}] vis={vis} conf={confidence:.2f} room={room} rel={percept.get('relevance',0):.2f}", flush=True)

                # Build topo map from every VLM call (room info now in percept)
                if nav_state.get("target_pos") is None:
                    topo_map.add_node(robot_pos, room, [], step)

                # Direct navigation when target confidently spotted
                if vis and confidence >= VLM_CONF_THRESHOLD:
                    depth = env.get_depth()
                    tgt   = _estimate_target_pos(
                        depth, percept.get("direction", "center"), robot_pos, R)
                    if tgt is not None:
                        nav_state["target_pos"]    = tgt.tolist()
                        nav_state["current_skill"] = "follow_path"
                        nav_state["waypoints"]     = []
            except Exception:
                pass

        # ── Update value map (VLFM-style: use relevance, not confidence) ──
        # relevance is non-zero even when target not visible → map has gradient
        vlm_score = float(nav_state["last_percept"].get("relevance", 0.2))
        explore_map.update(robot_pos, R, vlm_score)

        # ── Stagnation detection: escape if exploration stuck ──────────
        if current_skill_pre := nav_state.get("current_skill", "explore_frontier"):
            if current_skill_pre == "explore_frontier":
                cur_expl = explore_map.explored_fraction()
                if abs(cur_expl - nav_state.get("last_expl", 0.0)) < 0.001:
                    nav_state["stagnant_steps"] = nav_state.get("stagnant_steps", 0) + 1
                else:
                    nav_state["stagnant_steps"] = 0
                    nav_state["last_expl"] = cur_expl
                if nav_state["stagnant_steps"] >= 40:
                    # Escape: pick FARTHEST reachable navmesh point to break out of stuck area
                    pf = env._sim.pathfinder
                    from agent.skills import _replan
                    escape_candidates = []
                    for _ in range(50):
                        rp = pf.get_random_navigable_point()
                        if not any(np.isnan(rp)) and abs(rp[1] - robot_pos[1]) < 1.0:
                            dist = float(np.linalg.norm(np.array(rp) - robot_pos))
                            if dist > 3.0:
                                wps = _replan(env, robot_pos, rp)
                                if wps:
                                    escape_candidates.append((dist, rp.tolist(), wps))
                    if not escape_candidates:
                        # Same-floor escape failed — try cross-floor (find stairs/other floor)
                        for _ in range(50):
                            rp = pf.get_random_navigable_point()
                            if not any(np.isnan(rp)):
                                dist = float(np.linalg.norm(np.array(rp) - robot_pos))
                                if dist > 3.0:
                                    wps = _replan(env, robot_pos, rp)
                                    if wps:
                                        escape_candidates.append((dist, rp.tolist(), wps))
                    if not escape_candidates:
                        # Last resort: any reachable point
                        for _ in range(20):
                            rp = pf.get_random_navigable_point()
                            if not any(np.isnan(rp)):
                                dist = float(np.linalg.norm(np.array(rp) - robot_pos))
                                if dist > 0.5:
                                    wps = _replan(env, robot_pos, rp)
                                    if wps:
                                        escape_candidates.append((dist, rp.tolist(), wps))
                                        break
                    if escape_candidates:
                        # Prefer unexplored grid cells; break ties by distance
                        def _esc_score(c):
                            gi, gj = explore_map._w2g(c[1][0], c[1][2])
                            unexp = 1 if (explore_map._valid(gi, gj) and explore_map.grid[gi, gj] == 0) else 0
                            return unexp * 100.0 + c[0]
                        escape_candidates.sort(key=lambda x: -_esc_score(x))
                        best_dist, rp_list, wps = escape_candidates[0]
                        gi, gj = explore_map._w2g(rp_list[0], rp_list[2])
                        _tag = "UNEXP" if (explore_map._valid(gi, gj) and explore_map.grid[gi, gj] == 0) else "EXP"
                        nav_state["frontier_pos"] = rp_list
                        nav_state["waypoints"] = wps
                        nav_state["failed_frontiers"] = set()
                        nav_state["stagnant_steps"] = 0
                        nav_state["last_expl"] = explore_map.explored_fraction()
                        nav_state["explore_anchor"] = rp_list
                        nav_state["anchor_steps_left"] = 80
                        print(f"  [ESCAPE step={step}] {_tag} dist={best_dist:.1f}m", flush=True)
                    else:
                        # Truly stuck in isolated navmesh island — just rotate
                        nav_state["stagnant_steps"] = 0
                        print(f"  [STUCK step={step}] isolated navmesh island, rotating", flush=True)

        # ── Execute skill ──────────────────────────────────────────────
        current = nav_state.get("current_skill", "explore_frontier")
        if step % 30 == 0:
            expl = explore_map.explored_fraction()
            dist_to_tgt = None
            if nav_state.get("target_pos"):
                tp = np.array(nav_state["target_pos"])
                dist_to_tgt = float(np.linalg.norm(robot_pos - tp))
            print(f"  [step={step:03d}] skill={current} expl={expl:.1%} tgt_pos={nav_state.get('target_pos') is not None} dist_to_tgt={dist_to_tgt}", flush=True)
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
