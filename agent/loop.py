"""
loop.py — Exploration loop with semantic topological map.

Pipeline:
  1. ExploreMap: online 2D occupancy + VLM value grid → frontier detection
  2. TopoMap: semantic topological graph — nodes annotated with room labels
     (built from classify_scene() every CLASSIFY_INTERVAL steps)
  3. Frontier selection biased by topo_map.suggest_goal_direction():
       goto      → navigate to known room node
       go_upstairs → navigate to visually-detected staircase position
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
CLASSIFY_INTERVAL  = 24   # call classify_scene every N steps for room/floor reasoning

from pathlib import Path
import os as _os
_DATA_DEFAULT = Path("/root/autodl-tmp/data/hm3d")
_DATA     = Path(_os.environ.get("VLN_DATA_DIR", str(_DATA_DEFAULT)))
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

    valid = strip[(strip > 0.5) & (strip < 10.0)].flatten()

    if len(valid) < 10:
        d = 3.0
    else:
        d = float(np.percentile(valid, 85))
        d = min(d, 5.0)

    x_c = (col_center - cx) * d / fx
    z_c = -d
    p   = R @ np.array([x_c, 0.0, z_c]) + agent_pos
    return p.astype(np.float32)


def _estimate_pos_from_bbox(depth_frame, bbox, agent_pos, R, hfov=90.0):
    """Back-project bbox center to world 3D using depth pixels within the box.

    Like VLFM's SAM-mask depth, but uses the VLM bbox instead of a segmentation
    mask.  Much more precise than direction-strip 85th-percentile because we
    only sample pixels that belong to the target object.
    """
    if not bbox or len(bbox) != 4:
        return None
    try:
        H, W = depth_frame.shape
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # InternVL3 sometimes outputs 0-1000 normalized coords instead of pixels.
        # Detect by checking if any coordinate exceeds the larger image dimension.
        if max(x1, x2, y1, y2) > max(W, H):
            x1 = int(x1 * W / 1000)
            y1 = int(y1 * H / 1000)
            x2 = int(x2 * W / 1000)
            y2 = int(y2 * H / 1000)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        # Use only the top 70% of bbox rows (de-emphasise floor at base of object)
        y2_crop = y1 + int((y2 - y1) * 0.70)
        if y2_crop <= y1:
            y2_crop = min(y2, y1 + 5)
        region  = depth_frame[y1:y2_crop, x1:x2]
        # Cap at 6.0m: beyond that is likely background wall; 3.5m was too aggressive for far objects
        valid   = region[(region > 0.3) & (region < 10.0)].flatten()
        if len(valid) < 10:
            return None
        # 5th percentile: biased toward nearest surface (front face of object)
        d  = float(np.percentile(valid, 5))
        d  = max(0.4, min(d, 9.5))
        fx = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
        # Physical plausibility: reject if bbox is too small for furniture at this depth.
        # phys_w/h in metres; 0.45m filters hallucinations where VLM sees a small
        # feature (e.g. 100×100px at 1.3m = 0.41m) and mistakes it for a sofa/chair.
        phys_w = (x2 - x1) * d / fx
        phys_h = (y2 - y1) * d / fx   # use full height, not cropped (crop only for depth sampling)
        if max(phys_w, phys_h) < 0.45:
            return None
        cx = W / 2.0
        u_c = (x1 + x2) / 2.0
        x_c = (u_c - cx) * d / fx
        z_c = -d
        p   = R @ np.array([x_c, 0.0, z_c]) + agent_pos
        return p.astype(np.float32)
    except Exception:
        return None


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

        step = nav_state["step_count"]

        # ── 策略阶段引导 (大脑计划 → 小脑执行) ─────────────────────────────
        _phases      = nav_state.get("search_strategy", [])
        _phase_idx   = nav_state.get("strategy_phase", 0)
        _phase_room  = _phases[_phase_idx] if _phase_idx < len(_phases) else None

        if _phase_room:
            _room_visits = nav_state.get("room_counts", {}).get(_phase_room, 0)
            if _room_visits >= 4:
                nav_state["strategy_phase"] = _phase_idx + 1
                _phase_idx  = nav_state["strategy_phase"]
                _phase_room = _phases[_phase_idx] if _phase_idx < len(_phases) else None
                _on_thought = nav_state.get("_on_thought")
                if _phase_room and _on_thought:
                    _on_thought(step, "plan", f"阶段推进 → 搜索 {_phase_room}")
                _log(f"  [STRATEGY step={step}] phase advanced → {_phase_room}")

        hint = topo_map.suggest_goal_direction(task, robot_pos)
        action_type = hint.get("action", "explore")

        if _phase_room and action_type == "explore":
            _phase_node = topo_map.find_room_node(_phase_room)
            if _phase_node is not None:
                hint = {"action": "goto", "pos": _phase_node.pos}
                action_type = "goto"
                _log(f"  [STRATEGY step={step}] phase={_phase_room} → goto known node ({_phase_node.pos[0]:.1f},{_phase_node.pos[2]:.1f})")

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
            _sighting = nav_state.get("staircase_sighting")
            if _sighting is not None:
                target = np.array(_sighting, dtype=np.float32)
                _log(f"  [FRONTIER step={step}] topo_hint=go_upstairs → staircase sighting at ({target[0]:.1f},{target[2]:.1f})")
            else:
                _log(f"  [FRONTIER step={step}] topo_hint=go_upstairs → no staircase seen yet → value-map")
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
        "room_vlm_history": [],     # last 8 VLM-reported room labels (loop detection)
        "step_log":       [],        # Structured per-VLM-call log (exported as JSON)
        "search_strategy":  [],      # 大脑先验推理: ordered room search phases
        "strategy_phase":   0,       # current phase index
        "strategy_floor":   0,       # 0=ground first, 1=upper first
        "_on_thought":      on_thought,  # callback for _explore_frontier phase transitions
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

    # ── Episode-start VLM strategy planning (大脑先验推理) ──────────────────
    if llm_perceive is not None:
        from agent.llm_agent import plan_strategy as _plan_strategy
        _strategy = _plan_strategy(task)
        nav_state["search_strategy"] = _strategy.get("phase_rooms", [])
        nav_state["strategy_phase"]  = 0
        nav_state["strategy_floor"]  = _strategy.get("floor", 0)
        if on_thought and nav_state["search_strategy"]:
            _sr = nav_state["search_strategy"]
            on_thought(0, "plan", f"搜索策略: {' → '.join(_sr)} | {_strategy.get('reasoning', '')}")

    if on_frame:
        on_frame(nav_state["last_frame"], nav_state)

    skill_map = registered_skill_map()  # built from @skill registry

    # CLIP detector: fast per-step target visibility (replaces VLM bbox/target_visible)
    try:
        from agent.clip_detector import CLIPDetector as _CLIPDet
    except Exception:
        _CLIPDet = None

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

        # ── CLIP per-step target detection ────────────────────────────
        _clip_res = {"visible": False, "score": 0.0, "bbox": None, "direction": "not_visible"}
        if _CLIPDet is not None:
            _cf = nav_state.get("last_frame")
            if _cf is not None:
                try:
                    _clip_res = _CLIPDet.detect(_cf, task)
                except Exception as _ce:
                    _log(f"  [CLIP err step={step}] {_ce}")
            nav_state["last_clip"] = _clip_res
            if _clip_res["visible"]:
                _cstrk = nav_state.get("clip_streak", 0) + 1
                nav_state["clip_streak"] = _cstrk
                _log(f"  [CLIP step={step}] score={_clip_res['score']:.2f} "
                     f"streak={_cstrk} dir={_clip_res['direction']}")
            else:
                if nav_state.get("clip_streak", 0) > 0:
                    _log(f"  [CLIP step={step}] score={_clip_res['score']:.2f} → streak reset")
                nav_state["clip_streak"] = 0

        # ── CLIP-streak trigger ────────────────────────────────────────
        # When CLIP has seen the target for 5+ consecutive frames, force an
        # immediate VLM re-evaluation so the large brain makes the stop decision
        # with CLIP evidence — rather than the small brain stopping alone.
        _cstrk_cur = nav_state.get("clip_streak", 0)
        _clip_sc_cur = _clip_res.get("score", 0.0)
        _cstrk_prev = nav_state.get("clip_streak_prev", 0)
        if (step >= 20
                and not nav_state.get("done", False)
                and nav_state.get("current_skill") not in ("verify_arrival", "done")
                and _cstrk_cur >= 5
                and _clip_sc_cur > 0.42
                and _cstrk_prev < 5):
            _log(f"  [STREAK-TRIGGER step={step}] streak={_cstrk_cur} "
                 f"score={_clip_sc_cur:.2f} → forcing immediate VLM re-eval")
            nav_state["vlm_step"] = step - VLM_CALL_INTERVAL  # force trigger next block
            if on_thought:
                _sd = _clip_res.get("direction", "?")
                on_thought(step, "sensor",
                           f"CLIP传感器: 连续{_cstrk_cur}帧检测到目标 "
                           f"(score={_clip_sc_cur:.2f}, dir={_sd}) → 唤醒VLM确认")
        nav_state["clip_streak_prev"] = _cstrk_cur

        # ── VLFM-style VLM call ────────────────────────────────────────
        # Trigger every VLM_CALL_INTERVAL steps, OR immediately when CLIP
        # streak first reaches threshold (STREAK-TRIGGER above resets vlm_step).
        _clip_event = (_cstrk_cur >= 3 and _cstrk_prev < 3)
        if llm_perceive is not None and (
            (step - nav_state["vlm_step"]) >= VLM_CALL_INTERVAL or _clip_event
        ):
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
                    "search_strategy":  nav_state.get("search_strategy", []),
                    "strategy_phase":   nav_state.get("strategy_phase", 0),
                }
                # CLIP state injected into VLM prompt so large brain knows what
                # the small brain sensor is detecting (dual-loop coupling).
                _clip_state_for_vlm = {
                    "streak":    nav_state.get("clip_streak", 0),
                    "score":     nav_state.get("last_clip", {}).get("score", 0.0),
                    "direction": nav_state.get("last_clip", {}).get("direction", "none"),
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
                                           context=_ctx,
                                           clip_state=_clip_state_for_vlm)
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
                    percept = llm_perceive(nav_state["last_frame"], task, context=_ctx,
                                           clip_state=_clip_state_for_vlm)

                # Emit raw VLM JSON to frontend before any merging so the user
                # can see exactly what the model returned each call.
                _vlm_raw = percept.pop("_raw", None)
                if _vlm_raw and on_thought:
                    on_thought(step, "vlm_raw", _vlm_raw[:280])

                # Merge CLIP detection into percept: CLIP handles vis/bbox, VLM handles room/skill.
                # Skip during verify_arrival scan so the confidence window reflects
                # only VLM's own judgment, not CLIP injection.
                _lc = nav_state.get("last_clip", {})
                _in_verify_scan = nav_state.get("current_skill") == "verify_arrival"
                if _lc.get("visible") and not _in_verify_scan:
                    percept["target_visible"] = True
                    percept["bbox"]           = _lc.get("bbox")
                    percept["confidence"]     = max(float(percept.get("confidence", 0.0)),
                                                    float(_lc.get("score", 0.0)))
                    if not percept.get("direction") or percept.get("direction") == "not_visible":
                        percept["direction"] = _lc.get("direction", "center")
                    _log(f"  [CLIP-MERGE step={step}] vis=True score={_lc['score']:.2f} "
                         f"merged into percept")

                nav_state["last_percept"] = percept
                nav_state["vlm_step"]     = step
                # Track room visits for Phase 4 context
                _rm = percept.get("room", "other")
                nav_state["room_counts"][_rm] = nav_state["room_counts"].get(_rm, 0) + 1
                _rvh = nav_state.setdefault("room_vlm_history", [])
                _rvh.append(_rm)
                if len(_rvh) > 8:
                    _rvh.pop(0)
                stats["vlm_calls"] += 1
                nav_state["step_log"].append({
                    "step":           step,
                    "skill":          percept.get("skill", ""),
                    "reason":         percept.get("reason", ""),
                    "confidence":     float(percept.get("confidence", 0.0)),
                    "target_visible": bool(percept.get("target_visible", False)),
                    "room":           percept.get("room", "other"),
                    "relevance":      float(percept.get("relevance", 0.0)),
                    "direction":      percept.get("direction", "not_visible"),
                    "robot_pos":      robot_pos.tolist(),
                    "topo_nodes":     topo_map.node_count,
                    "explored_pct":   explore_map.explored_fraction(),
                    "vlm_raw":        _vlm_raw,
                })

                confidence  = float(percept.get("confidence", 0.0))
                direction   = percept.get("direction", "not_visible")
                # vis: explicit target_visible (old schema) OR VLM says snap+direction (new schema)
                vis = (percept.get("target_visible", False)
                       or (percept.get("skill") == "snap"
                           and direction not in ("not_visible", "")
                           and confidence >= 0.5))
                room        = percept.get("room", "other")
                rel         = float(percept.get("relevance", 0.0))
                target_room = str(percept.get("target_room", "") or "").strip()

                # Record the robot's position when it visually detects a staircase.
                # Used by go_upstairs to navigate there without navmesh sampling.
                if room == "staircase":
                    nav_state["staircase_sighting"] = robot_pos.tolist()
                    _log(f"  [STAIRCASE step={step}] sighted at ({robot_pos[0]:.1f},{robot_pos[2]:.1f})")

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
                _in_servo  = nav_state.get("current_skill") == "visual_servo"
                _in_follow = nav_state.get("current_skill") == "follow_path"
                if _in_verify and vis:
                    _log(f"  [VLM decision] in verify_arrival → routing skipped")
                if _in_follow and vis:
                    _log(f"  [VLM decision] in follow_path → routing skipped (skill autonomy)")

                if (vis and not _in_verify and not _in_servo and not _in_follow
                        and rel >= 0.40 and room != "other"):
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
                            # No GT instances: try CLIP bbox depth only.
                            # Direction-based 5m guess is disabled — it's less accurate
                            # than the value-map frontier and causes the robot to waste
                            # 100-400 steps navigating to empty space.
                            _bbox_s = (percept.get("bbox")
                                       or nav_state.get("last_clip", {}).get("bbox"))
                            _btgt_s = _estimate_pos_from_bbox(
                                env.get_depth(), _bbox_s, robot_pos, R) if _bbox_s else None
                            if _btgt_s is not None:
                                tgt = _btgt_s
                                nav_state["bbox_target"]        = True
                                nav_state["target_arrive_dist"] = 0.5
                                # Record snap XZ so verify_arrival can blacklist on failure
                                nav_state["clip_snap_target"] = (round(float(tgt[0]), 1), round(float(tgt[2]), 1))
                                _log(f"  [SNAP step={step}] no instances → bbox depth "
                                     f"({tgt[0]:.2f},{tgt[2]:.2f}) d={float(np.linalg.norm(tgt-robot_pos)):.1f}m")
                            elif _bbox_s is not None and tgt is not None:
                                # bbox depth failed (too close / noisy) but CLIP saw something.
                                # Keep the direction-based tgt from _estimate_target_pos — it points
                                # ~3-5m in the VLM/CLIP direction. Robot navigates there, then
                                # STREAK-STOP or VALUE-STOP fires when it gets close.
                                _log(f"  [SNAP step={step}] no instances, bbox depth failed → "
                                     f"direction fallback ({tgt[0]:.2f},{tgt[2]:.2f}) "
                                     f"d={float(np.linalg.norm(tgt-robot_pos)):.1f}m")
                            else:
                                # No CLIP bbox and no bbox from VLM — nothing to navigate toward.
                                tgt = None
                                _log(f"  [SNAP step={step}] no instances, no bbox → frontier-only (rel={rel:.2f})")

                        if tgt is not None:
                            old_skill = nav_state.get("current_skill")
                            nav_state["target_pos"]    = tgt.tolist()
                            nav_state["current_skill"] = "follow_path"
                            nav_state["waypoints"]     = []
                            if old_skill != "follow_path":
                                _log(f"  [VLM decision] {old_skill} → follow_path (target detected conf={confidence:.2f})")
                elif not vis:
                    nav_state["vis_stable_count"] = 0  # reset streak on not-visible
                    # VLM commonsense room guidance: if VLM says target_room and we haven't
                    # visited that room type yet, navigate toward it.
                    _tr_used = False
                    if (target_room and target_room not in ("other", "")
                            and nav_state.get("anchor_steps_left", 0) <= 0
                            and nav_state.get("target_pos") is None
                            and current_skill == "explore_frontier"):
                        _tr_nodes = [n for n in topo_map.nodes if target_room in n.room]
                        if _tr_nodes:
                            from agent.skills import _replan
                            _tr_best = min(_tr_nodes, key=lambda n: float(np.linalg.norm(n.pos - robot_pos)))
                            if float(np.linalg.norm(_tr_best.pos - robot_pos)) > 2.0:
                                _wps_tr = _replan(env, robot_pos, _tr_best.pos)
                                if _wps_tr:
                                    nav_state["frontier_pos"]      = _tr_best.pos.tolist()
                                    nav_state["waypoints"]         = _wps_tr
                                    nav_state["explore_anchor"]    = _tr_best.pos.tolist()
                                    nav_state["anchor_steps_left"] = 80
                                    _tr_used = True
                                    _log(f"  [VLM-ROOM step={step}] target_room={target_room} "
                                         f"→ goto known node ({_tr_best.pos[0]:.1f},{_tr_best.pos[2]:.1f})")
                    if not _tr_used:
                        _log(f"  [VLM decision] not visible → value_map update only (rel={rel:.2f})"
                             + (f" target_room={target_room}" if target_room else ""))
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
                    # Append sensor-based depth distance when CLIP sees the target
                    _dist_hint = ""
                    _cr = nav_state.get("last_clip", {})
                    if _cr.get("visible") and _cr.get("bbox"):
                        try:
                            x1, y1, x2, y2 = _cr["bbox"]
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            _d = float(env.get_depth()[cy, cx])
                            if 0.3 < _d < 10.0:
                                _dist_hint = f" [~{_d:.1f}m]"
                        except Exception:
                            pass
                    # Append search_direction hint when target not visible
                    _sd = percept.get("search_direction", "")
                    if _sd and _sd not in ("none", "") and not percept.get("target_visible", False):
                        _dist_hint += f" → {_sd}"

                    # Use VLM search_direction="upstairs" to break the staircase
                    # sighting catch-22: VLM can see stairs before the robot
                    # physically reaches them, so record current pos as sighting.
                    if (_sd == "upstairs"
                            and nav_state.get("staircase_sighting") is None):
                        nav_state["staircase_sighting"] = robot_pos.tolist()
                        _log(f"  [STAIRCASE step={step}] VLM search_direction=upstairs"
                             f" → sighting at ({robot_pos[0]:.1f},{robot_pos[2]:.1f})")
                        if on_thought:
                            on_thought(step, "plan", "Staircase spotted → recorded position, ready to ascend")
                    if on_thought and _r_clean not in _skip and len(_r_clean) >= 12:
                        on_thought(step, _skill, _r_clean + _dist_hint,
                                   percept.get("room"))
                    _dh = nav_state.setdefault("decision_history", [])
                    _dh.append({"step": step, "skill": _skill, "reason": _r_clean[:60], "room": percept.get("room", "other")})
                    if len(_dh) > 3:
                        _dh.pop(0)

                # BRAIN-SNAP: VLM says target visible, navigate now.
                # Require rel>=0.40 AND room!="other" to filter hallucinations
                # where VLM contradicts itself (vis=True but rel=0.20, room=other).
                _snap_rel  = float(percept.get("relevance", 0.0))
                _snap_room = percept.get("room", "other")
                if (_skill == "snap" and _vis4
                        and not _in_verify
                        and not _in_servo
                        and nav_state.get("target_pos") is None
                        and _snap_rel >= 0.40
                        and _snap_room != "other"):
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
                            # No GT instances: use bbox-based depth (like VLFM SAM-mask depth).
                            _bbox4b  = percept.get("bbox")
                            _btgt4b  = _estimate_pos_from_bbox(env.get_depth(), _bbox4b, robot_pos, R) if _bbox4b else None
                            if _btgt4b is not None:
                                nav_state["target_pos"]         = _btgt4b.tolist()
                                nav_state["current_skill"]      = "follow_path"
                                nav_state["waypoints"]          = []
                                nav_state["bbox_target"]        = True
                                nav_state["target_arrive_dist"] = 0.5
                                stats["snap_events"] += 1
                                _log(f"  [BRAIN-SNAP step={step}] bbox \u2192 follow_path "
                                     f"({_btgt4b[0]:.2f},{_btgt4b[2]:.2f}) "
                                     f"d={float(np.linalg.norm(_btgt4b-robot_pos)):.1f}m")
                            else:
                                _log(f"  [BRAIN-SNAP step={step}] depth-only \u2192 skip (no GT instances)")

                # VISUAL SERVO: fallback when target visible but bbox depth unavailable.
                # Requires 2 consecutive sightings for stability.
                if (vis
                        and not instances
                        and not _in_verify
                        and not _in_servo
                        and nav_state.get("current_skill") not in ("follow_path", "done")):
                    _vsc = nav_state.get("vis_consecutive", 0) + 1
                    nav_state["vis_consecutive"] = _vsc
                    if _vsc >= 2:
                        nav_state["current_skill"] = "visual_servo"
                        nav_state["servo_steps"]   = 0
                        nav_state["servo_lost"]    = 0
                        _log(f"  [SERVO trigger step={step}] vis_consecutive={_vsc} conf={_conf4:.2f} dir={direction} \u2192 visual_servo")
                elif not vis:
                    nav_state["vis_consecutive"] = 0

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
                        # No GT: robot position is not a reliable target.
                        # Skip verify_arrival; let value-map exploration continue.
                        _log(f"  [BRAIN-VERIFY step={step}] depth-only \u2192 skip (no GT instances)")

                # \u2500\u2500 Room-step budget: escape if same WRONG room for 6 VLM calls \u2500\u2500
                _ROOM_VLM_BUDGET = 6
                if (current_skill == "explore_frontier"
                        and nav_state.get("anchor_steps_left", 0) <= 0
                        and nav_state.get("target_pos") is None):
                    _rvh2 = nav_state.get("room_vlm_history", [])
                    if len(_rvh2) >= _ROOM_VLM_BUDGET:
                        _recent = _rvh2[-_ROOM_VLM_BUDGET:]
                        _unique_nonother = set(r for r in _recent if r and r != "other")
                        _all_other = all(r == "other" or not r for r in _recent)
                        _unique = _unique_nonother if _unique_nonother else ({"other"} if _all_other else set())
                        if len(_unique) == 1:
                            _stuck_room = list(_unique)[0]
                            from agent.topo_map import OBJECT_ROOM_MAP as _ORM
                            _goal_expected_room = _ORM.get(task, "")
                            if _stuck_room == _goal_expected_room:
                                # In the right room for this goal \u2014 reset counter, keep searching
                                _log(f"  [ROOM-BUDGET step={step}] in goal room '{_stuck_room}' \u2192 keep searching")
                                nav_state["room_vlm_history"] = []
                            else:
                                # Wrong room \u2014 escape to somewhere new
                                _log(f"  [ROOM-BUDGET step={step}] stuck in '{_stuck_room}' "
                                     f"for {_ROOM_VLM_BUDGET} VLM calls \u2192 semantic escape")
                                from agent.skills import _replan
                                _alts = [n for n in topo_map.nodes
                                         if n.room not in (_stuck_room, "other")]
                                _done_escape = False
                                if _alts:
                                    _ne = min(_alts, key=lambda n: float(np.linalg.norm(n.pos - robot_pos)))
                                    _wps_re = _replan(env, robot_pos, _ne.pos)
                                    if _wps_re:
                                        nav_state["frontier_pos"]      = _ne.pos.tolist()
                                        nav_state["waypoints"]         = _wps_re
                                        nav_state["failed_frontiers"]  = set()
                                        nav_state["stagnant_steps"]    = 0
                                        nav_state["room_vlm_history"]  = []
                                        nav_state["explore_anchor"]    = _ne.pos.tolist()
                                        nav_state["anchor_steps_left"] = 100
                                        nav_state["blacklisted_snap"]  = set()
                                        stats["escape_events"] += 1
                                        _done_escape = True
                                        _log(f"  [ROOM-ESCAPE step={step}] \u2192 topo node "
                                             f"room={_ne.room} ({_ne.pos[0]:.1f},{_ne.pos[2]:.1f})")
                                if not _done_escape:
                                    _pf_re = env._sim.pathfinder
                                    _re_cands = []
                                    for _ in range(40):
                                        _rp_re = _pf_re.get_random_navigable_point()
                                        if not any(np.isnan(_rp_re)) and abs(_rp_re[1] - robot_pos[1]) < 1.0:
                                            _d_re = float(np.linalg.norm(np.array(_rp_re) - robot_pos))
                                            if _d_re > 5.0:
                                                _gi_re, _gj_re = explore_map._w2g(_rp_re[0], _rp_re[2])
                                                if explore_map._valid(_gi_re, _gj_re) and explore_map.grid[_gi_re, _gj_re] == 0:
                                                    _wps_re2 = _replan(env, robot_pos, _rp_re)
                                                    if _wps_re2:
                                                        _re_cands.append((_d_re, _rp_re.tolist(), _wps_re2))
                                    if _re_cands:
                                        _re_cands.sort(key=lambda x: -x[0])
                                        _bd_re, _rpl_re, _wps_re3 = _re_cands[0]
                                        nav_state["frontier_pos"]      = _rpl_re
                                        nav_state["waypoints"]         = _wps_re3
                                        nav_state["failed_frontiers"]  = set()
                                        nav_state["stagnant_steps"]    = 0
                                        nav_state["room_vlm_history"]  = []
                                        nav_state["explore_anchor"]    = _rpl_re
                                        nav_state["anchor_steps_left"] = 100
                                        nav_state["blacklisted_snap"]  = set()
                                        stats["escape_events"] += 1
                                        _log(f"  [ROOM-ESCAPE step={step}] \u2192 unexplored random "
                                             f"({_rpl_re[0]:.1f},{_rpl_re[2]:.1f}) dist={_bd_re:.1f}m")

            except Exception as e:
                _log(f"  [VLM ERROR step={step}] {e}")
                nav_state["vlm_step"] = step  # prevent retry storm on persistent error

        # ── Update value map ───────────────────────────────────────────
        vlm_score = float(nav_state["last_percept"].get("relevance", 0.2))
        _vmap_dir = nav_state["last_percept"].get("direction", "center")
        _vmap_room = nav_state["last_percept"].get("room", "other")
        # Room-semantic correction: VLM gives living_room=0.9 for ALL goals,
        # making the value map goal-agnostic. Override with goal-aware room priors.
        _ROOM_PRIOR = {
            "沙发":  {"living_room": 0.9, "bedroom": 0.3, "hallway": 0.2, "kitchen": 0.05, "bathroom": 0.0},
            "床":    {"bedroom": 0.95, "living_room": 0.15, "hallway": 0.1, "kitchen": 0.05, "bathroom": 0.0},
            "电视":  {"living_room": 0.85, "bedroom": 0.5, "hallway": 0.1, "kitchen": 0.1, "bathroom": 0.0},
            "桌子":  {"living_room": 0.8, "kitchen": 0.7, "bedroom": 0.4, "hallway": 0.1, "bathroom": 0.05},
            "冰箱":  {"kitchen": 0.95, "living_room": 0.1, "hallway": 0.05, "bedroom": 0.0, "bathroom": 0.0},
            "椅子":  {"living_room": 0.8, "kitchen": 0.7, "bedroom": 0.5, "hallway": 0.2, "bathroom": 0.0},
        }
        _prior = _ROOM_PRIOR.get(task, {}).get(_vmap_room, None)
        _raw_score = vlm_score
        if _prior is not None:
            # Blend: 70% room prior + 30% VLM relevance (retains some per-frame signal)
            vlm_score = 0.7 * _prior + 0.3 * vlm_score
        # CLIP override: use live per-step score when object is visible.
        # This is the core of VLFM — dense value-map update from image-text
        # similarity at every step, not sparse 8-step VLM relevance.
        _clip_r_vm   = nav_state.get("last_clip", {})
        _clip_vis_vm = _clip_r_vm.get("visible", False)
        _clip_sc_vm  = float(_clip_r_vm.get("score", 0.0))
        _clip_dir_vm = _clip_r_vm.get("direction", "center")
        # With raw cosim rescaling (Fix B), visible=True already means cosim>0.40
        # (threshold in detect()). Extra 0.30 guard ensures meaningful signal.
        if _clip_vis_vm and _clip_sc_vm > 0.30:
            vlm_score = _clip_sc_vm
            _vmap_dir = _clip_dir_vm
            _log(f"  [VMAP step={step}] CLIP-driven score={vlm_score:.2f} dir={_vmap_dir}")
        else:
            _log(f"  [VMAP step={step}] room={_vmap_room} prior={_prior} raw={_raw_score:.2f} → score={vlm_score:.2f} dir={_vmap_dir}")
        explore_map.update(robot_pos, R, vlm_score, direction=_vmap_dir)

        # ── VLFM proximity stop ────────────────────────────────────────
        # Requires BOTH: CLIP > 0.50 AND explicit VLM visual confirmation.
        # The former clip_streak>=3 independent gate caused false positives:
        # white cabinets / door frames score high for 冰箱/衣柜, causing the
        # robot to stop in the wrong room. STREAK-TRIGGER already wakes the VLM
        # when streak>=5, so VLM confirmation is available when it matters.
        _vlm_vis_vm = nav_state.get("last_percept", {}).get("target_visible", False)
        # Room-type sanity gate: block VALUE-STOP when clearly in the wrong room.
        _WRONG_ROOM_BLOCK = {
            "冰箱": {"bedroom", "bathroom", "staircase"},
            "衣柜": {"kitchen", "bathroom", "staircase"},
            "床":   {"kitchen", "bathroom", "staircase"},
        }
        _cur_room_vm  = nav_state.get("last_percept", {}).get("room", "other")
        _room_blocked = _cur_room_vm in _WRONG_ROOM_BLOCK.get(task, set())
        if (step >= 50
                and not nav_state.get("done", False)
                and nav_state.get("current_skill") not in ("verify_arrival", "done")
                and _clip_sc_vm > 0.50
                and _vlm_vis_vm
                and not _room_blocked):
            _bvp = explore_map.best_value_pos(robot_pos)
            if _bvp is not None:
                _bvd = float(np.sqrt(
                    (robot_pos[0] - _bvp[0])**2 + (robot_pos[2] - _bvp[2])**2))
                if _bvd < 1.5:
                    _log(f"  [VALUE-STOP step={step}] dist_to_best_cell={_bvd:.2f}m CLIP={_clip_sc_vm:.2f} room={_cur_room_vm} → done")
                    nav_state["done"] = True
                    if on_thought:
                        on_thought(step, "verify",
                                   f"Reached target zone (heatmap peak {_bvd:.2f}m, CLIP={_clip_sc_vm:.2f}) → stop")

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

        # ── classify_scene: room/floor reasoning every CLASSIFY_INTERVAL steps ──
        if (llm_perceive is not None
                and step > 0
                and step % CLASSIFY_INTERVAL == 0
                and nav_state.get("current_skill") == "explore_frontier"
                and nav_state.get("target_pos") is None):
            try:
                from agent.llm_agent import classify_scene as _classify
                _cls = _classify(nav_state["last_frame"], task)
                _sug = _cls.get("suggest", "none")
                _rm  = _cls.get("room", "other")
                _fh  = _cls.get("floor_hint", "unknown")
                _objs = _cls.get("objects", [])
                topo_map.add_node(robot_pos, _rm, _objs, step)
                _log(f"  [CLASSIFY step={step}] room={_rm} floor={_fh} suggest={_sug} objects={_objs}")
                if _sug == "go_upstairs" and not topo_map.has_explored_floor(1):
                    _sighting = nav_state.get("staircase_sighting")
                    if _sighting is not None:
                        from agent.skills import _replan
                        _stair = np.array(_sighting, dtype=np.float32)
                        wps = _replan(env, robot_pos, _stair)
                        if wps:
                            nav_state["frontier_pos"]      = _sighting
                            nav_state["waypoints"]         = wps
                            nav_state["anchor_steps_left"] = 60
                            nav_state["explore_anchor"]    = _sighting
                            _log(f"  [GO-UPSTAIRS step={step}] → ({_stair[0]:.1f},{_stair[2]:.1f})")
            except Exception as _ce:
                _log(f"  [CLASSIFY ERROR step={step}] {_ce}")

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

        nav_state["_topo_map"] = topo_map
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
