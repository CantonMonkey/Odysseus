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


# ── Logging helpers ──────────────────────────────────────────────────────────

def _inst_dist(robot_pos, instances):
    """Horizontal (XZ) Euclidean distance to nearest semantic instance.

    Matches the metric in eval.py: uses XZ-only so tall objects (fridge, TV)
    with elevated bounding-box centroids don't inflate the distance.
    """
    if not instances:
        return None
    rx, rz = float(robot_pos[0]), float(robot_pos[2])
    return float(min(
        np.sqrt((float(p[0]) - rx) ** 2 + (float(p[2]) - rz) ** 2)
        for p in instances
    ))


def _pathfinder_reachable(env, robot_pos: np.ndarray, target_pos: np.ndarray) -> bool:
    """Return True if pathfinder can find a path from robot to target."""
    import habitat_sim
    pf = env._sim.pathfinder
    snapped = pf.snap_point(target_pos.astype(np.float32))
    if np.any(np.isnan(snapped)):
        return False
    path = habitat_sim.ShortestPath()
    path.requested_start = robot_pos.astype(np.float32)
    path.requested_end   = snapped
    return pf.find_path(path) and len(path.points) > 1

def _log(msg: str):
    print(msg, flush=True)


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

    col_center = {"left": W // 4, "center": W // 2, "right": 3 * W // 4}.get(direction, W // 2)
    col_lo = max(0, col_center - 40)
    col_hi = min(W, col_center + 40)
    strip  = depth_frame[H//4 : 3*H//4, col_lo:col_hi]

    valid = strip[(strip > 0.5) & (strip < 8.0)].flatten()

    if len(valid) < 10:
        d = 3.0
    else:
        d = float(np.percentile(valid, 85))
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

    if nav_state.get("anchor_steps_left", 0) > 0:
        nav_state["anchor_steps_left"] -= 1
        if nav_state["anchor_steps_left"] == 0:
            nav_state["explore_anchor"] = None

    f_pos = nav_state.get("frontier_pos")
    if f_pos is not None and _euclidean(robot_pos, np.array(f_pos)) < ARRIVE_DIST:
        f_arr = np.array(f_pos)
        fi, fj = explore_map._w2g(float(f_arr[0]), float(f_arr[2]))
        failed = nav_state.get("failed_frontiers", set())
        failed.add((fi, fj))
        nav_state["failed_frontiers"] = failed
        nav_state["frontier_pos"] = None
        nav_state["waypoints"]    = []

    if not nav_state.get("frontier_pos") or not nav_state.get("waypoints"):
        target = None
        failed = nav_state.get("failed_frontiers", set())

        hint = topo_map.suggest_goal_direction(task, robot_pos)
        action_type = hint.get("action", "explore")
        step = nav_state["step_count"]

        if action_type == "goto":
            target = hint["pos"]
            # Skip if robot is already at the goto target (prevents infinite self-loop).
            if _euclidean(robot_pos, np.array(target)) < ARRIVE_DIST:
                gi2, gj2 = explore_map._w2g(float(np.array(target)[0]), float(np.array(target)[2]))
                failed.add((gi2, gj2))
                nav_state["failed_frontiers"] = failed
                _log(f"  [FRONTIER step={step}] topo_hint=goto → already at node ({target[0]:.1f},{target[2]:.1f}), skip → value-map")
                target = None
            else:
                _log(f"  [FRONTIER step={step}] topo_hint=goto → known room node at ({target[0]:.1f},{target[2]:.1f})")
        elif action_type == "go_upstairs":
            stair = topo_map.find_staircase_approach(env, robot_pos)
            if stair is not None:
                target = stair
                _log(f"  [FRONTIER step={step}] topo_hint=go_upstairs → staircase at ({stair[0]:.1f},{stair[2]:.1f})")
        else:
            _log(f"  [FRONTIER step={step}] topo_hint={action_type} → value-map frontier selection")

        if target is None:
            cells = explore_map.frontiers()
            ri, rj = explore_map._w2g(robot_pos[0], robot_pos[2])
            pf_nav = env._sim.pathfinder
            best_score, best_ij = -1.0, None
            n_considered, n_skipped_wall = 0, 0
            for i, j in cells:
                if (i, j) in failed:
                    continue
                wx, wz = explore_map._g2w(i, j)
                cell_world = np.array([wx, robot_pos[1], wz], dtype=np.float32)
                snapped = pf_nav.snap_point(cell_world)
                if not np.any(np.isnan(snapped)):
                    snap_dist = float(np.linalg.norm(snapped[[0,2]] - cell_world[[0,2]]))
                    if snap_dist > 1.5:
                        failed.add((i, j))
                        n_skipped_wall += 1
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
                n_considered += 1
                if s > best_score:
                    best_score, best_ij = s, (i, j)

            if best_ij is not None:
                wx, wz = explore_map._g2w(*best_ij)
                target = np.array([wx, robot_pos[1], wz], dtype=np.float32)
                _log(f"  [FRONTIER step={step}] selected ({wx:.1f},{wz:.1f}) score={best_score:.3f} "
                     f"from {n_considered} frontiers ({n_skipped_wall} wall-skipped)")
            if target is None:
                nav_state["failed_frontiers"] = set()
                frame, _ = env.step(ACTION_LEFT)
                nav_state["last_frame"]  = frame
                nav_state["step_count"] += 1
                _log(f"  [FRONTIER step={step}] no frontiers left → rotate (reset failed set)")
                return nav_state

        wps = _replan(env, robot_pos, target)
        if not wps:
            t_arr = np.array(target)
            fi, fj = explore_map._w2g(float(t_arr[0]), float(t_arr[2]))
            failed.add((fi, fj))
            nav_state["failed_frontiers"] = failed
            nav_state["frontier_pos"] = None
            nav_state["waypoints"]    = []
            frame, _ = env.step(ACTION_LEFT)
            nav_state["last_frame"]  = frame
            nav_state["step_count"] += 1
            _log(f"  [FRONTIER step={step}] target unreachable by pathfinder → blacklist + rotate")
            return nav_state

        nav_state["frontier_pos"]      = target.tolist() if hasattr(target, 'tolist') else list(target)
        nav_state["waypoints"]         = wps
        nav_state["failed_frontiers"]  = failed

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

    # Position stagnation: if robot didn't move physically (wall collision),
    # clear frontier so a new one is chosen next step.
    prev_pos   = nav_state.get("expl_prev_pos")
    stuck_cnt  = nav_state.get("expl_stuck_steps", 0)
    moved      = float(np.linalg.norm(new_pos - np.array(prev_pos))) if prev_pos is not None else 1.0
    stuck_cnt  = stuck_cnt + 1 if moved < 0.05 else 0
    nav_state["expl_prev_pos"]    = new_pos.tolist()
    nav_state["expl_stuck_steps"] = stuck_cnt
    if stuck_cnt >= 15 and nav_state.get("frontier_pos"):
        nav_state["frontier_pos"]     = None
        nav_state["waypoints"]        = []
        nav_state["expl_stuck_steps"] = 0
        step_n = nav_state.get("step_count", "?")
        _log(f"  [FRONTIER step={step_n}] pos-stuck {stuck_cnt} steps → clear frontier + reselect")

    return nav_state


# ── main task loop ───────────────────────────────────────────────────────────

def run_task(
    env,
    task: str,
    scene_dir: str = SCENE_DIR,
    on_frame: Optional[Callable] = None,
    llm_perceive=None,
    max_steps: int = MAX_STEPS,
    target_instances=None,
    initial_explore_map=None,
    initial_topo_map=None,
    on_thought=None,
) -> dict:
    from agent.skills import follow_path, verify_arrival  # ensures @skill decorators run
    from agent.skill_registry import registered_skill_map

    if initial_explore_map is not None:
        initial_explore_map.value[:] = 0.0   # reset goal-specific heatmap
        explore_map = initial_explore_map
    else:
        explore_map = ExploreMap()
    topo_map = initial_topo_map if initial_topo_map is not None else TopoMap()

    instances = [np.asarray(p, dtype=np.float32) for p in target_instances] if target_instances else []

    nav_state: dict = {
        "goal":             task,
        "target_pos":       None,
        "step_count":       0,
        "current_skill":    "explore_frontier",
        "done":             False,
        "last_frame":       env.get_frame(),
        "last_percept":     {"target_visible": False, "confidence": 0.0},
        "waypoints":        [],
        "explore_map":      explore_map,
        "topo_map":         topo_map,
        "frontier_pos":     None,
        "last_expl":        0.0,
        "stagnant_steps":   0,
        "vlm_step":         -VLM_CALL_INTERVAL,
        "last_scene":       {},
        "target_instances": instances,
        "blacklisted_snap": set(),   # XZ keys of SNAP targets pathfinder can't reach
        "decision_history":  [],     # Last 3 VLM skill decisions for context injection
        "room_counts":    {},        # Phase 4: room visit counts for VLM context
        "step_log":       [],        # Structured per-VLM-call log (exported as JSON)
        # Decision chain stats (accumulated per episode)
        "_stats": {
            "vlm_calls": 0, "vlm_visible": 0, "snap_events": 0,
            "skill_steps": {"explore_frontier": 0, "follow_path": 0,
                            "verify_arrival": 0, "search_room": 0},
            "escape_events": 0, "frontier_selections": 0,
        },
    }

    robot_pos, _ = env.get_robot_pose()
    _log(f"  [EPISODE START] goal={task} robot=({robot_pos[0]:.2f},{robot_pos[2]:.2f}) "
         f"instances={len(instances)}")

    if on_frame:
        on_frame(nav_state["last_frame"], nav_state)

    skill_map = registered_skill_map()  # built from @skill registry

    prev_skill = "explore_frontier"

    while not nav_state["done"] and nav_state["step_count"] < max_steps:
        step      = nav_state["step_count"]
        robot_pos, _ = env.get_robot_pose()
        R            = env.get_rotation_matrix()
        stats        = nav_state["_stats"]

        # ── Track skill step counts ────────────────────────────────────
        current_skill = nav_state.get("current_skill", "explore_frontier")
        stats["skill_steps"][current_skill] = stats["skill_steps"].get(current_skill, 0) + 1

        # ── Proximity trigger: if adjacent to an instance with no target, go verify ─
        if (nav_state.get("target_pos") is None
                and current_skill not in ("verify_arrival", "done")
                and instances):
            _idist_prox = _inst_dist(robot_pos, instances)
            if _idist_prox is not None and _idist_prox <= ARRIVE_DIST:
                rx, rz = float(robot_pos[0]), float(robot_pos[2])
                nearest_inst = min(instances,
                                   key=lambda p: (float(p[0])-rx)**2 + (float(p[2])-rz)**2)
                nav_state["target_pos"]    = [float(nearest_inst[0]),
                                               float(robot_pos[1]),
                                               float(nearest_inst[2])]
                nav_state["current_skill"] = "verify_arrival"
                _log(f"  [PROXIMITY step={step}] inst_dist={_idist_prox:.2f}m ≤ {ARRIVE_DIST}m "
                     f"→ verify_arrival at ({nearest_inst[0]:.2f},{nearest_inst[2]:.2f})")
                current_skill = "verify_arrival"

        # ── Log skill transitions ──────────────────────────────────────
        if current_skill != prev_skill:
            idist = _inst_dist(robot_pos, instances)
            idist_s = f"{idist:.2f}m" if idist is not None else "N/A"
            tgt = nav_state.get("target_pos")
            tdist_s = f"{float(np.linalg.norm(robot_pos - np.array(tgt))):.2f}m" if tgt else "None"
            _log(f"  [SKILL→{current_skill} step={step}] robot=({robot_pos[0]:.2f},{robot_pos[2]:.2f}) "
                 f"dist_to_tgt={tdist_s} dist_to_instance={idist_s}")
            prev_skill = current_skill

        # ── VLFM-style VLM call ────────────────────────────────────────
        if llm_perceive is not None and (step - nav_state["vlm_step"]) >= VLM_CALL_INTERVAL:
            try:
                # Phase 4: build context dict for VLM brain
                _idist_ctx = _inst_dist(robot_pos, instances)
                _room_counts = nav_state.get("room_counts", {})
                _rooms_str = ", ".join(
                    f"{r}:{c}" for r, c in
                    sorted(_room_counts.items(), key=lambda x: -x[1])[:5]
                ) or "none yet"
                _ctx = {
                    "step":             step,
                    "max_steps":        max_steps,
                    "explored_pct":     explore_map.explored_fraction(),
                    "stagnant_steps":   nav_state.get("stagnant_steps", 0),
                    "rooms_str":        _rooms_str,
                    "nearest_dist_str": f"{_idist_ctx:.1f}m" if _idist_ctx else "unknown",
                    "history":          nav_state.get("decision_history", []),
                    "topo_summary":     topo_map.summary(),
                }

                # Always annotate frontiers (AgentVLN: unified cross-space mapping)
                from agent.vlm_frontier import project_waypoint, annotate_frame
                _candidates = explore_map.top_k_frontiers(5, robot_pos)
                _visible = []
                for _score, _fpos in _candidates:
                    _uv = project_waypoint(_fpos, robot_pos, R)
                    if _uv is not None:
                        _visible.append((_fpos, _uv))
                if len(_visible) >= 2:
                    _ann = annotate_frame(
                        nav_state["last_frame"],
                        [_uv for _, _uv in _visible],
                        [str(_i+1) for _i in range(len(_visible))],
                    )
                    percept = llm_perceive(_ann, task,
                                           annotated_frame=_ann,
                                           n_waypoints=len(_visible),
                                           context=_ctx)
                    if nav_state.get("target_pos") is None and nav_state.get("frontier_pos") is None:
                        _choice = percept.get("waypoint", 0)
                        if 1 <= _choice <= len(_visible):
                            _chosen, _ = _visible[_choice - 1]
                            nav_state["frontier_pos"] = (
                                _chosen.tolist() if hasattr(_chosen, "tolist") else list(_chosen)
                            )
                            nav_state["waypoints"] = []
                            _log(f"  [VLM-FRONTIER step={step}] chose waypoint {_choice} "
                                 f"\u2192 ({_chosen[0]:.1f},{_chosen[2]:.1f})")
                else:
                    percept = llm_perceive(nav_state["last_frame"], task, context=_ctx)

                nav_state["last_percept"] = percept
                nav_state["vlm_step"]     = step
                # Track room visits for Phase 4 context
                _rm = percept.get("room", "other")
                nav_state["room_counts"][_rm] = nav_state["room_counts"].get(_rm, 0) + 1
                stats["vlm_calls"] += 1
                nav_state["step_log"].append({
                    "step":           step,
                    "skill":          percept.get("skill", ""),
                    "reason":         percept.get("reason", ""),
                    "confidence":     float(percept.get("confidence", 0.0)),
                    "target_visible": bool(percept.get("target_visible", False)),
                    "room":           percept.get("room", "other"),
                    "direction":      percept.get("direction", "not_visible"),
                    "robot_pos":      robot_pos.tolist(),
                    "topo_nodes":     topo_map.node_count,
                    "explored_pct":   explore_map.explored_fraction(),
                })

                confidence = float(percept.get("confidence", 0.0))
                vis        = percept.get("target_visible", False)
                room       = percept.get("room", "other")
                rel        = float(percept.get("relevance", 0.0))
                direction  = percept.get("direction", "not_visible")
                idist      = _inst_dist(robot_pos, instances)
                idist_s    = f"{idist:.2f}m" if idist is not None else "N/A"

                if vis:
                    stats["vlm_visible"] += 1

                _log(f"  [VLM step={step}] vis={vis} conf={confidence:.2f} room={room} "
                     f"rel={rel:.2f} dir={direction} "
                     f"robot=({robot_pos[0]:.2f},{robot_pos[2]:.2f}) "
                     f"nearest_instance={idist_s}")

                if nav_state.get("target_pos") is None:
                    topo_map.add_node(robot_pos, room, [], step)

                _in_verify = nav_state.get("current_skill") == "verify_arrival"
                if _in_verify and vis:
                    _log(f"  [VLM decision] in verify_arrival → target_pos FROZEN (prevent oscillation)")

                if vis and not _in_verify:
                    depth = env.get_depth()
                    tgt   = _estimate_target_pos(
                        depth, direction, robot_pos, R)
                    if tgt is not None:
                        inst_list = nav_state.get("target_instances", [])
                        blacklisted = nav_state.get("blacklisted_snap", set())
                        if inst_list:
                            robot_y    = float(robot_pos[1])
                            same_floor = [p for p in inst_list if abs(float(p[1]) - robot_y) < 1.0]
                            nearby     = [p for p in same_floor if float(np.linalg.norm(p - robot_pos)) < 10.0]
                            pool       = nearby if nearby else same_floor if same_floor else inst_list
                            # Skip instances whose XZ positions have been blacklisted (unreachable by pathfinder)
                            pool = [p for p in pool if (round(float(p[0]),1), round(float(p[2]),1)) not in blacklisted]

                            if pool:
                                # Sort by XZ distance; check pathfinder reachability for
                                # the top-8 nearest candidates (proactive, not reactive).
                                # Phase 3: QD-depth — use depth hint to prefer instance at right distance
                                try:
                                    _depth_map = env.get_depth()
                                    _dir = percept.get("direction", "center")
                                    _col = {"left": slice(0, 213), "right": slice(427, 640)}.get(_dir, slice(160, 480))
                                    _roi = _depth_map[120:360, _col]
                                    _valid_d = _roi[(_roi > 0.3) & (_roi < 8.0)]
                                    _depth_hint = float(np.median(_valid_d)) if _valid_d.size > 10 else None
                                except Exception:
                                    _depth_hint = None
                                if _depth_hint is not None:
                                    _log(f"  [QD-DEPTH step={step}] dir={percept.get('direction','center')} hint={_depth_hint:.2f}m")
                                    pool_sorted = sorted(pool,
                                        key=lambda p: (
                                            abs(float(np.sqrt((float(p[0])-float(robot_pos[0]))**2 + (float(p[2])-float(robot_pos[2]))**2)) - _depth_hint),
                                            float(np.linalg.norm(p - robot_pos)),
                                        ))
                                else:
                                    pool_sorted = sorted(pool, key=lambda p: float(np.linalg.norm(p - robot_pos)))
                                nearest = None
                                new_bl  = []
                                for cand in pool_sorted[:3]:
                                    cand_arr = np.array([float(cand[0]), float(tgt[1]), float(cand[2])], dtype=np.float32)
                                    if _pathfinder_reachable(env, robot_pos, cand_arr):
                                        nearest = cand
                                        break
                                    else:
                                        key = (round(float(cand[0]), 1), round(float(cand[2]), 1))
                                        new_bl.append(key)
                                # Blacklist unreachable candidates found during proactive check
                                if new_bl:
                                    nav_state.setdefault("blacklisted_snap", set()).update(new_bl)
                                    blacklisted = nav_state["blacklisted_snap"]
                                if nearest is None:
                                    nearest = pool_sorted[0]  # fall back to nearest
                                snapped  = np.array([float(nearest[0]), float(tgt[1]), float(nearest[2])], dtype=np.float32)
                                snap_dist = float(np.linalg.norm(snapped - robot_pos))
                                stats["snap_events"] += 1
                                _log(f"  [SNAP step={step}] "
                                     f"pool={len(pool)} (same_floor={len(same_floor)}/{len(inst_list)} nearby={len(nearby)} blacklisted={len(blacklisted)}) "
                                     f"new_bl={len(new_bl)} → chosen=({nearest[0]:.2f},{nearest[2]:.2f}) "
                                     f"robot_dist={snap_dist:.1f}m "
                                     f"depth_est=({tgt[0]:.2f},{tgt[2]:.2f})")
                                tgt = snapped
                            else:
                                # All semantic instances blacklisted — don't use raw depth estimate
                                # because it may point to unrelated furniture. Keep exploring.
                                _log(f"  [SNAP step={step}] all pool instances blacklisted ({len(blacklisted)}) → force ESCAPE")
                                tgt = None
                                nav_state["stagnant_steps"] = nav_state.get("stagnant_steps", 0) + 20
                        else:
                            _log(f"  [SNAP step={step}] no instance list → using raw depth estimate ({tgt[0]:.2f},{tgt[2]:.2f})")

                        if tgt is not None:
                            old_skill = nav_state.get("current_skill")
                            nav_state["target_pos"]    = tgt.tolist()
                            nav_state["current_skill"] = "follow_path"
                            nav_state["waypoints"]     = []
                            if old_skill != "follow_path":
                                _log(f"  [VLM decision] {old_skill} → follow_path (target detected conf={confidence:.2f})")
                elif not vis:
                    _log(f"  [VLM decision] not visible → value_map update only (rel={rel:.2f})")
                elif _in_verify:
                    pass  # already logged above
                # (all vis=True cases handled above)

                # ── Phase 4: VLM skill decisions ──────────────────────────
                _skill  = percept.get("skill", "")
                _reason = percept.get("reason", "")
                _conf4  = float(percept.get("confidence", 0.0))
                _vis4   = percept.get("target_visible", False)
                if _skill:
                    _log(f"  [BRAIN step={step}] skill={_skill} reason={str(_reason)[:80]!r}")
                    _r_clean = str(_reason).strip()
                    _skip = {'str','other','not_visible','bathroom','','clear','clearly visible'}
                    if on_thought and _r_clean not in _skip and len(_r_clean) >= 12:
                        on_thought(step, _skill, _r_clean)
                    _dh = nav_state.setdefault("decision_history", [])
                    _dh.append({"step": step, "skill": _skill, "reason": _r_clean[:60]})
                    if len(_dh) > 3:
                        _dh.pop(0)

                # BRAIN-SNAP: VLM says target visible, navigate now
                if (_skill == "snap" and _vis4
                        and not _in_verify
                        and nav_state.get("target_pos") is None):
                    _depth4 = env.get_depth()
                    _dir4   = percept.get("direction", "center")
                    _tgt4   = _estimate_target_pos(_depth4, _dir4, robot_pos, R)
                    if _tgt4 is not None:
                        _il4 = nav_state.get("target_instances", [])
                        _bl4 = nav_state.get("blacklisted_snap", set())
                        if _il4:
                            _ry4 = float(robot_pos[1])
                            _sf4 = [p for p in _il4 if abs(float(p[1]) - _ry4) < 1.0]
                            _nb4 = [p for p in _sf4 if float(np.linalg.norm(p - robot_pos)) < 10.0]
                            _pl4 = _nb4 if _nb4 else _sf4 if _sf4 else _il4
                            _pl4 = [p for p in _pl4
                                    if (round(float(p[0]),1), round(float(p[2]),1)) not in _bl4]
                            if _pl4:
                                _ps4 = sorted(_pl4, key=lambda p: float(np.linalg.norm(p - robot_pos)))
                                _n4 = None
                                for _c4 in _ps4[:3]:
                                    _ca4 = np.array([float(_c4[0]), float(_tgt4[1]),
                                                     float(_c4[2])], dtype=np.float32)
                                    if _pathfinder_reachable(env, robot_pos, _ca4):
                                        _n4 = _c4; break
                                if _n4 is None: _n4 = _ps4[0]
                                _sn4 = np.array([float(_n4[0]), float(_tgt4[1]),
                                                 float(_n4[2])], dtype=np.float32)
                                nav_state["target_pos"]    = _sn4.tolist()
                                nav_state["current_skill"] = "follow_path"
                                nav_state["waypoints"]     = []
                                stats["snap_events"] += 1
                                _log(f"  [BRAIN-SNAP step={step}] conf={_conf4:.2f} "
                                     f"\u2192 follow_path ({_n4[0]:.2f},{_n4[2]:.2f})")
                        else:
                            # No GT instances \u2014 use raw depth estimate (clean eval mode)
                            nav_state["target_pos"]    = _tgt4.tolist()
                            nav_state["current_skill"] = "follow_path"
                            nav_state["waypoints"]     = []
                            stats["snap_events"] += 1
                            _log(f"  [BRAIN-SNAP step={step}] depth-only \u2192 ({_tgt4[0]:.2f},{_tgt4[2]:.2f})")

                # BRAIN-ESCAPE: VLM says stuck, escape now (bypass 40-step wait)
                elif (_skill == "escape"
                      and current_skill == "explore_frontier"
                      and nav_state.get("anchor_steps_left", 0) <= 0):
                    from agent.skills import _replan as _rpl4
                    _pf4 = env._sim.pathfinder
                    _ecs4 = []
                    for _ in range(50):
                        _rpt4 = _pf4.get_random_navigable_point()
                        if not any(np.isnan(_rpt4)) and abs(_rpt4[1] - robot_pos[1]) < 1.0:
                            _d4 = float(np.linalg.norm(np.array(_rpt4) - robot_pos))
                            if _d4 > 3.0:
                                _w4 = _rpl4(env, robot_pos, _rpt4)
                                if _w4: _ecs4.append((_d4, _rpt4.tolist(), _w4))
                    if _ecs4:
                        _ecs4.sort(key=lambda x: -x[0])
                        _ebd4, _erl4, _ew4 = _ecs4[0]
                        try:
                            from agent.vlm_frontier import project_waypoint, annotate_frame
                            _etop4 = [(rpl, wps, project_waypoint(np.array(rpl), robot_pos, R))
                                      for _, rpl, wps in _ecs4[:3]
                                      if project_waypoint(np.array(rpl), robot_pos, R) is not None]
                            if len(_etop4) >= 2:
                                _eann4 = annotate_frame(nav_state["last_frame"],
                                                        [u for _, _, u in _etop4],
                                                        [str(i+1) for i in range(len(_etop4))])
                                _ep4r = llm_perceive(_eann4, task, annotated_frame=_eann4,
                                                     n_waypoints=len(_etop4))
                                _ec4n = _ep4r.get("waypoint", 0)
                                if 1 <= _ec4n <= len(_etop4):
                                    _erl4, _ew4, _ = _etop4[_ec4n - 1]
                                    _ebd4 = float(np.linalg.norm(np.array(_erl4) - robot_pos))
                                    _log(f"  [BRAIN-ESCAPE-VLM step={step}] dir {_ec4n}")
                        except Exception: pass
                        nav_state["frontier_pos"]      = _erl4
                        nav_state["waypoints"]         = _ew4
                        nav_state["failed_frontiers"]  = set()
                        nav_state["stagnant_steps"]    = 0
                        nav_state["last_expl"]         = explore_map.explored_fraction()
                        nav_state["explore_anchor"]    = _erl4
                        nav_state["anchor_steps_left"] = 80
                        nav_state["blacklisted_snap"]  = set()
                        stats["escape_events"] += 1
                        _log(f"  [BRAIN-ESCAPE step={step}] dist={_ebd4:.1f}m "
                             f"tgt=({_erl4[0]:.1f},{_erl4[2]:.1f}) VLM-triggered")

                # BRAIN-VERIFY: VLM says very close, jump to verify_arrival
                elif (_skill == "verify"
                      and nav_state.get("target_pos") is None
                      and current_skill != "verify_arrival"):
                    _vd4 = _inst_dist(robot_pos, instances)
                    if _vd4 is not None and instances:
                        _rx4 = float(robot_pos[0]); _rz4 = float(robot_pos[2])
                        _vn4 = min(instances,
                                   key=lambda p: (float(p[0])-_rx4)**2+(float(p[2])-_rz4)**2)
                        nav_state["target_pos"]    = [float(_vn4[0]), float(robot_pos[1]),
                                                      float(_vn4[2])]
                        nav_state["current_skill"] = "verify_arrival"
                        _log(f"  [BRAIN-VERIFY step={step}] dist={_vd4:.2f}m \u2192 verify_arrival")
                    elif not instances:
                        # No GT \u2014 trust VLM: verify at robot's current navmesh position
                        _pf_snap = env._sim.pathfinder.snap_point(robot_pos)
                        _snap_pos = _pf_snap.tolist() if not np.any(np.isnan(_pf_snap)) else robot_pos.tolist()
                        nav_state["target_pos"]    = _snap_pos
                        nav_state["current_skill"] = "verify_arrival"
                        _log(f"  [BRAIN-VERIFY step={step}] depth-only \u2192 verify_arrival at robot pos")

            except Exception as e:
                _log(f"  [VLM ERROR step={step}] {e}")

        # ── Update value map ───────────────────────────────────────────
        vlm_score = float(nav_state["last_percept"].get("relevance", 0.2))
        explore_map.update(robot_pos, R, vlm_score)

        # ── Stagnation / ESCAPE ────────────────────────────────────────
        if current_skill == "explore_frontier":
            cur_expl = explore_map.explored_fraction()
            # Pause stagnation counter while navigating toward an ESCAPE anchor;
            # otherwise the 80-step anchor window gets cut short after only 40 steps.
            if nav_state.get("anchor_steps_left", 0) <= 0:
                if abs(cur_expl - nav_state.get("last_expl", 0.0)) < 0.001:
                    nav_state["stagnant_steps"] = nav_state.get("stagnant_steps", 0) + 1
                else:
                    nav_state["stagnant_steps"] = 0
            nav_state["last_expl"] = cur_expl
            if nav_state["stagnant_steps"] >= 40:
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
                    for _ in range(50):
                        rp = pf.get_random_navigable_point()
                        if not any(np.isnan(rp)):
                            dist = float(np.linalg.norm(np.array(rp) - robot_pos))
                            if dist > 3.0:
                                wps = _replan(env, robot_pos, rp)
                                if wps:
                                    escape_candidates.append((dist, rp.tolist(), wps))
                if not escape_candidates:
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
                    def _esc_score(c):
                        gi, gj = explore_map._w2g(c[1][0], c[1][2])
                        unexp = 1 if (explore_map._valid(gi, gj) and explore_map.grid[gi, gj] == 0) else 0
                        return unexp * 100.0 + c[0]
                    escape_candidates.sort(key=lambda x: -_esc_score(x))
                    # Phase 2: VLM-directed ESCAPE — show top-3 candidates, let VLM choose
                    _esc_chosen = None
                    try:
                        from agent.vlm_frontier import project_waypoint, annotate_frame
                        _top3 = escape_candidates[:3]
                        _esc_vis = []
                        for _ed, _rpl, _ewps in _top3:
                            _euv = project_waypoint(np.array(_rpl), robot_pos, R)
                            if _euv is not None:
                                _esc_vis.append((_rpl, _ewps, _euv))
                        if len(_esc_vis) >= 2:
                            _eann = annotate_frame(
                                nav_state["last_frame"],
                                [_euv for _, _, _euv in _esc_vis],
                                [str(_ei+1) for _ei in range(len(_esc_vis))],
                            )
                            _ep = llm_perceive(_eann, task,
                                               annotated_frame=_eann,
                                               n_waypoints=len(_esc_vis))
                            _ec = _ep.get("waypoint", 0)
                            if 1 <= _ec <= len(_esc_vis):
                                _esc_chosen = _esc_vis[_ec - 1]
                                _log(f"  [ESCAPE-VLM step={step}] VLM selected direction {_ec}")
                    except Exception as _ee:
                        _log(f"  [ESCAPE-VLM step={step}] VLM selection failed: {_ee}")
                    if _esc_chosen is not None:
                        rp_list, wps, _ = _esc_chosen
                        best_dist = float(np.linalg.norm(np.array(rp_list) - robot_pos))
                    else:
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
                    nav_state["blacklisted_snap"]  = set()
                    stats["escape_events"] += 1
                    idist = _inst_dist(robot_pos, instances)
                    _log(f"  [ESCAPE step={step}] {_tag} dist={best_dist:.1f}m "
                         f"robot=({robot_pos[0]:.1f},{robot_pos[2]:.1f}) "
                         f"tgt=({rp_list[0]:.1f},{rp_list[2]:.1f}) "
                         f"nearest_instance={f'{idist:.2f}m' if idist else 'N/A'} "
                         f"expl={cur_expl:.1%}")
                else:
                    nav_state["stagnant_steps"] = 0
                    _log(f"  [STUCK step={step}] isolated navmesh island, rotating")

        # ── Every-30-step progress summary ─────────────────────────────
        if step % 30 == 0:
            expl = explore_map.explored_fraction()
            tgt  = nav_state.get("target_pos")
            tdist = float(np.linalg.norm(robot_pos - np.array(tgt))) if tgt else None
            idist = _inst_dist(robot_pos, instances)
            tdist_s = f"{tdist:.2f}m" if tdist is not None else "None"
            idist_s = f"{idist:.2f}m" if idist is not None else "N/A"
            _log(f"  [step={step:03d}] skill={current_skill} expl={expl:.1%} "
                 f"robot=({robot_pos[0]:.2f},{robot_pos[2]:.2f}) "
                 f"dist_to_tgt={tdist_s} nearest_instance={idist_s}")

        # ── Execute skill ──────────────────────────────────────────────
        if current_skill == "done":
            nav_state["done"] = True
            break
        elif current_skill in skill_map:
            nav_state = skill_map[current_skill](env, nav_state)
        else:
            nav_state = _explore_frontier(env, nav_state, explore_map, topo_map)

        if on_frame and nav_state.get("last_frame") is not None:
            on_frame(nav_state["last_frame"], nav_state)

    # ── Episode summary ────────────────────────────────────────────────────
    s = nav_state["_stats"]
    total_vlm = s["vlm_calls"]
    vis_rate  = s["vlm_visible"] / total_vlm if total_vlm else 0.0
    skill_s   = " ".join(f"{k}={v}" for k, v in s["skill_steps"].items() if v > 0)
    idist_final = _inst_dist(robot_pos, instances)
    _log(f"  [EPISODE SUMMARY] steps={nav_state['step_count']} done={nav_state.get('done')} "
         f"nearest_instance={f'{idist_final:.2f}m' if idist_final else 'N/A'}")
    _log(f"    vlm_calls={total_vlm} vis_rate={vis_rate:.0%} snap_events={s['snap_events']} "
         f"escapes={s['escape_events']}")
    _log(f"    skill_steps: {skill_s}")

    if nav_state["step_count"] >= max_steps and not nav_state["done"]:
        nav_state["timeout"] = True

    return nav_state
