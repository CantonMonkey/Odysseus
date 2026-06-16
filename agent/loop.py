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
    """Euclidean distance from robot to nearest semantic instance."""
    if not instances:
        return None
    return float(min(np.linalg.norm(p - robot_pos) for p in instances))

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
) -> dict:
    from agent.skills    import follow_path, verify_arrival

    explore_map = ExploreMap()
    topo_map    = TopoMap()

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

    skill_map = {
        "follow_path":    follow_path,
        "verify_arrival": verify_arrival,
    }

    prev_skill = "explore_frontier"

    while not nav_state["done"] and nav_state["step_count"] < max_steps:
        step      = nav_state["step_count"]
        robot_pos, _ = env.get_robot_pose()
        R            = env.get_rotation_matrix()
        stats        = nav_state["_stats"]

        # ── Track skill step counts ────────────────────────────────────
        current_skill = nav_state.get("current_skill", "explore_frontier")
        stats["skill_steps"][current_skill] = stats["skill_steps"].get(current_skill, 0) + 1

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
                percept = llm_perceive(nav_state["last_frame"], task)
                nav_state["last_percept"] = percept
                nav_state["vlm_step"]     = step
                stats["vlm_calls"] += 1

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

                if vis and confidence >= VLM_CONF_THRESHOLD and not _in_verify:
                    depth = env.get_depth()
                    tgt   = _estimate_target_pos(
                        depth, direction, robot_pos, R)
                    if tgt is not None:
                        inst_list = nav_state.get("target_instances", [])
                        if inst_list:
                            robot_y    = float(robot_pos[1])
                            same_floor = [p for p in inst_list if abs(float(p[1]) - robot_y) < 1.0]
                            nearby     = [p for p in same_floor if float(np.linalg.norm(p - robot_pos)) < 10.0]
                            pool       = nearby if nearby else same_floor if same_floor else inst_list

                            if pool:
                                nearest  = min(pool, key=lambda p: float(np.linalg.norm(p - robot_pos)))
                                snapped  = np.array([float(nearest[0]), float(tgt[1]), float(nearest[2])], dtype=np.float32)
                                snap_dist = float(np.linalg.norm(snapped - robot_pos))
                                stats["snap_events"] += 1
                                _log(f"  [SNAP step={step}] "
                                     f"pool={len(pool)} (same_floor={len(same_floor)}/{len(inst_list)} nearby={len(nearby)}) "
                                     f"→ chosen=({nearest[0]:.2f},{nearest[2]:.2f}) "
                                     f"robot_dist={snap_dist:.1f}m "
                                     f"depth_est=({tgt[0]:.2f},{tgt[2]:.2f})")
                                tgt = snapped
                            else:
                                _log(f"  [SNAP step={step}] no instances in pool → using raw depth estimate")
                        else:
                            _log(f"  [SNAP step={step}] no instance list → using raw depth estimate ({tgt[0]:.2f},{tgt[2]:.2f})")

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
                else:
                    _log(f"  [VLM decision] vis=True but conf={confidence:.2f} < threshold={VLM_CONF_THRESHOLD} → ignored")

            except Exception as e:
                _log(f"  [VLM ERROR step={step}] {e}")

        # ── Update value map ───────────────────────────────────────────
        vlm_score = float(nav_state["last_percept"].get("relevance", 0.2))
        explore_map.update(robot_pos, R, vlm_score)

        # ── Stagnation / ESCAPE ────────────────────────────────────────
        if current_skill == "explore_frontier":
            cur_expl = explore_map.explored_fraction()
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
