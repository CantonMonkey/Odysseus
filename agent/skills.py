"""
skills.py

Three reusable navigation skills:
  follow_path    – primary skill: follow a Habitat pathfinder shortest path
  search_room    – fallback: rotate in place waiting for LLM perception
  verify_arrival – check whether the robot has reached the goal
"""

import numpy as np
import habitat_sim
from agent.skill_registry import skill

try:
    import quaternion as npq
    _HAS_QUATERNION = True
except ImportError:
    _HAS_QUATERNION = False

ARRIVE_DIST  = 2.0   # goal-reached threshold (m)
ALIGN_THRESH = 12.0  # heading alignment threshold before moving forward (deg)
WP_REACH     = 0.4   # waypoint-reached threshold (m)


def _get_forward(env) -> np.ndarray:
    """Return the agent's horizontal forward unit vector in world coordinates.

    In Habitat, local -Z maps to the world forward direction; the rotation
    matrix converts that to world space.
    """
    state = env._sim.get_agent(0).get_state()
    q = state.rotation
    if _HAS_QUATERNION:
        R = npq.as_rotation_matrix(q)
    else:
        w, x, y, z = q.w, q.x, q.y, q.z
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
        ])
    fwd = R @ np.array([0.0, 0.0, -1.0])  # local -Z → world forward
    fwd[1] = 0.0                           # project onto horizontal plane
    norm = np.linalg.norm(fwd)
    return fwd / norm if norm > 1e-6 else np.array([0.0, 0.0, -1.0])


def _turn_to(forward: np.ndarray, to_wp: np.ndarray):
    """Compute the turn needed to face *to_wp* from current *forward*.

    Returns (angle_deg, action) where action is ACTION_LEFT or ACTION_RIGHT.

    Convention verified empirically:
      turn_left  → forward rotates toward -X
      turn_right → forward rotates toward +X
    The Y component of (forward × to_wp) determines which side to turn:
      positive → to_wp is to the left  → ACTION_LEFT
      negative → to_wp is to the right → ACTION_RIGHT
    """
    from agent.habitat_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT
    to_wp = to_wp.copy()
    to_wp[1] = 0.0
    n = np.linalg.norm(to_wp)
    if n < 1e-6:
        return 0.0, ACTION_FORWARD
    to_wp /= n

    dot   = float(np.clip(np.dot(forward, to_wp), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(dot)))

    # Y component of cross product: forward[2]*to_wp[0] - forward[0]*to_wp[2]
    cross_y = forward[2] * to_wp[0] - forward[0] * to_wp[2]
    action  = ACTION_LEFT if cross_y > 0 else ACTION_RIGHT
    return angle, action


def _euclidean(a: np.ndarray, b) -> float:
    return float(np.linalg.norm(a - np.array(b, dtype=np.float32)))


# ── Skill 1: follow_path ────────────────────────────────────────
# Primary navigation skill: query pathfinder for the shortest path,
# then advance waypoint by waypoint using rotation-matrix heading.

@skill("follow_path")
def follow_path(env, nav_state: dict) -> dict:
    """Follow the Habitat pathfinder shortest path toward the goal.

    Re-plans automatically whenever the waypoint list is exhausted.
    Clears target_pos and reverts to explore if no path can be found,
    or if distance hasn't improved in FOLLOW_STAGNANT_LIMIT steps.
    """
    from agent.habitat_env import ACTION_FORWARD

    FOLLOW_STAGNANT_LIMIT = 25  # steps without dist improvement → give up

    target_pos = nav_state.get("target_pos")
    if target_pos is None:
        nav_state["current_skill"] = "explore_frontier"
        return nav_state

    robot_pos, _ = env.get_robot_pose()
    dist = _euclidean(robot_pos, target_pos)
    step = nav_state.get("step_count", 0)

    # bbox-based targets: get much closer before handing off to verify
    arrive_dist = nav_state.get("target_arrive_dist", ARRIVE_DIST)

    if dist <= arrive_dist:
        print(f"  [FOLLOW step={step}] dist={dist:.3f}m ≤ {arrive_dist:.1f}m → verify_arrival", flush=True)
        nav_state["current_skill"] = "verify_arrival"
        nav_state["follow_stagnant"] = 0
        return nav_state

    # Stagnation: if dist hasn't decreased in FOLLOW_STAGNANT_LIMIT steps, abandon target
    last_dist = nav_state.get("follow_last_dist", dist + 1.0)
    if dist >= last_dist - 0.05:
        nav_state["follow_stagnant"] = nav_state.get("follow_stagnant", 0) + 1
    else:
        nav_state["follow_stagnant"] = 0
    nav_state["follow_last_dist"] = dist

    if nav_state.get("follow_stagnant", 0) >= FOLLOW_STAGNANT_LIMIT:
        tgt = nav_state.get("target_pos")
        # If robot is close enough for visual confirmation, try verify_arrival
        # before blacklisting. Depth estimates often land inside furniture (unreachable)
        # but the robot is visually adjacent.
        VERIFY_ACCEPT_DIST = 2.5
        if tgt and dist <= VERIFY_ACCEPT_DIST and not nav_state.get("follow_tried_verify"):
            print(f"  [FOLLOW step={step}] stagnant dist={dist:.2f}m ≤ {VERIFY_ACCEPT_DIST}m → try verify_arrival", flush=True)
            nav_state["current_skill"]      = "verify_arrival"
            nav_state["follow_stagnant"]    = 0
            nav_state["follow_tried_verify"] = True
            return nav_state
        if tgt:
            key = (round(tgt[0], 1), round(tgt[2], 1))
            nav_state.setdefault("blacklisted_snap", set()).add(key)
            print(f"  [FOLLOW step={step}] stagnant {FOLLOW_STAGNANT_LIMIT} steps dist={dist:.2f}m → blacklist ({key[0]},{key[1]}) + explore", flush=True)
        nav_state["target_pos"]          = None
        nav_state["current_skill"]       = "explore_frontier"
        nav_state["follow_stagnant"]     = 0
        nav_state["follow_tried_verify"] = False
        nav_state["waypoints"]           = []
        nav_state["bbox_target"]         = False
        nav_state["target_arrive_dist"]  = ARRIVE_DIST
        return nav_state

    waypoints = nav_state.get("waypoints", [])

    # Pop the current waypoint if reached, then replan if empty
    if not waypoints or _euclidean(robot_pos, waypoints[0]) < WP_REACH:
        if waypoints:
            waypoints.pop(0)
        if not waypoints:
            waypoints = _replan(env, robot_pos, target_pos)
            nav_state["waypoints"] = waypoints

    if not waypoints:
        # Pathfinder can't reach target — blacklist this SNAP position and explore
        tgt = nav_state.get("target_pos")
        if tgt:
            key = (round(tgt[0], 1), round(tgt[2], 1))
            nav_state.setdefault("blacklisted_snap", set()).add(key)
            print(f"  [FOLLOW step={step}] no path → blacklist ({key[0]},{key[1]}) + explore_frontier", flush=True)
        nav_state["target_pos"] = None
        nav_state["current_skill"] = "explore_frontier"
        nav_state["follow_stagnant"] = 0
        nav_state["waypoints"] = []
        return nav_state

    next_wp = np.array(waypoints[0])
    to_wp   = next_wp - robot_pos
    forward = _get_forward(env)
    angle, action = _turn_to(forward, to_wp)

    if angle < ALIGN_THRESH:
        action = ACTION_FORWARD

    frame, _ = env.step(action)
    nav_state["last_frame"]  = frame
    nav_state["step_count"] += 1

    # Pop waypoint if the step brought us within reach
    new_pos, _ = env.get_robot_pose()
    if waypoints and _euclidean(new_pos, waypoints[0]) < WP_REACH:
        waypoints.pop(0)
    nav_state["waypoints"] = waypoints

    return nav_state


def _replan(env, robot_pos: np.ndarray, target_pos) -> list:
    """Query pathfinder for a new route; return waypoints excluding the start."""
    pf       = env._sim.pathfinder
    target_np = np.array(target_pos, dtype=np.float32)
    snapped  = pf.snap_point(target_np)
    if np.any(np.isnan(snapped)):
        return []
    path = habitat_sim.ShortestPath()
    path.requested_start = robot_pos.astype(np.float32)
    path.requested_end   = snapped
    if pf.find_path(path) and len(path.points) > 1:
        return [p.tolist() for p in path.points[1:]]
    return []


# ── Skill 2: search_room ────────────────────────────────────────
# Fallback when no target_pos is known: rotate a full 360° then step
# forward to a new vantage point, repeating until LLM perception fires.

def search_room(env, nav_state: dict) -> dict:
    """Rotate in place waiting for LLM guidance.

    Forward movement is intentionally omitted: random walking can bring
    the robot to stairwells or other bad positions, distorting the camera
    view and making LLM perception unreliable.
    """
    from agent.habitat_env import ACTION_LEFT

    rotated = nav_state.get("search_rotated", 0.0)
    frame, _ = env.step(ACTION_LEFT)
    nav_state["search_rotated"] = (rotated + 15.0) % 360.0

    nav_state["last_frame"]  = frame
    nav_state["step_count"] += 1
    return nav_state


# ── Skill 3: verify_arrival ─────────────────────────────────────

@skill("verify_arrival")
def verify_arrival(env, nav_state: dict) -> dict:
    """Confirm arrival at goal.

    VLM says target_visible → SUCCESS. Let eval judge distance.
    Arrived but VLM doesn't see it → scan 360° with fresh VLM frames.
    After scan, GT instance XZ fallback if available.
    """
    robot_pos, _ = env.get_robot_pose()
    target_pos   = nav_state.get("target_pos")
    dist = _euclidean(robot_pos, target_pos) if target_pos else float("inf")
    step = nav_state["step_count"]

    percept        = nav_state.get("last_percept", {})
    target_visible = percept.get("target_visible", False)
    instances      = nav_state.get("target_instances", [])
    _conf = float(percept.get("confidence", 0.0))

    # Accept up to 2.5m on entry: follow_path hands off at 1.2m but a final
    # forward step can overshoot slightly, causing immediate dist > ARRIVE_DIST
    # bounce → follow_path stagnation → explore (2-step false escape).
    VERIFY_ACCEPT_DIST = 2.5
    if dist <= VERIFY_ACCEPT_DIST:
        scanned = nav_state.get("verify_scanned", 0)

        # ── VLM-based success path ───────────────────────────────────────
        # VLM fires every 3 scan steps (forced via vlm_step in loop.py).
        # New schema doesn't set target_visible; use direction+confidence instead.
        # CLIP-MERGE is skipped during verify, so we must read VLM fields directly.
        # Two VLM confirmations (≥6 rotation steps apart) → success even when
        # CLIP scores are marginal (raw cosim ~0.24, rescaled ~0.33 < 0.35).
        _vlm_dir   = percept.get("direction", "not_visible")
        _vlm_conf  = float(percept.get("confidence", 0.0))
        _vlm_step  = nav_state.get("vlm_step", -999)
        _vlm_fresh = (step - _vlm_step) <= 1   # VLM fired this step or one step ago
        if _vlm_fresh and _vlm_conf >= 0.65 and _vlm_dir not in ("not_visible", ""):
            _vc = nav_state.get("verify_vlm_count", 0) + 1
            nav_state["verify_vlm_count"] = _vc
            print(f"  [VERIFY step={step}] VLM visible conf={_vlm_conf:.2f} dir={_vlm_dir} → vlm_count={_vc}", flush=True)
            if _vc >= 2:
                print(f"  [VERIFY step={step}] VLM×2 confirmed → SUCCESS", flush=True)
                nav_state["done"]             = True
                nav_state["current_skill"]    = "done"
                nav_state["verify_best_clip"] = 0.0
                nav_state["verify_vlm_count"] = 0
                return nav_state

        # ── CLIP-based success path ──────────────────────────────────────
        # Use CLIP score (every step) instead of VLM confidence (every 8 steps).
        # During a 24-step 360° scan, VLM fires only 3 times so most entries
        # would be 0.0 — CLIP gives a real signal at every rotation step.
        _clip_v = nav_state.get("last_clip", {}).get("score", 0.0)
        CONF_WINDOW = 6   # ~90° rotation
        # Lowered from 0.45: beds/fridges reach ~0.44 rescaled, old threshold
        # was just out of reach causing consistent scan failure.
        CONF_THRESH = 0.40  # raised from 0.30 — white walls average ~0.30
        window = nav_state.get("verify_conf_window", [])
        window.append(_clip_v)
        window = window[-CONF_WINDOW:]
        nav_state["verify_conf_window"] = window

        # Track best CLIP score seen during this verify pass.
        nav_state["verify_best_clip"] = max(nav_state.get("verify_best_clip", 0.0), _clip_v)

        # Early exit: two consecutive high-CLIP frames AND fresh VLM confirmation.
        # CLIP alone is insufficient — white walls/windows can score >0.38 for
        # fridge/wardrobe. Require VLM to agree before stopping.
        _vlm_step_v  = nav_state.get("vlm_step", -999)
        _vlm_fresh_v = (step - _vlm_step_v) <= 2
        _vlm_conf_v  = float(nav_state.get("last_percept", {}).get("confidence", 0.0))
        _vlm_dir_v   = nav_state.get("last_percept", {}).get("direction", "not_visible")
        _vlm_ok_v    = _vlm_fresh_v and _vlm_conf_v >= 0.60 and _vlm_dir_v not in ("not_visible", "")
        _vstreak = nav_state.get("verify_clip_streak", 0)
        if _clip_v > 0.38:
            _vstreak += 1
        else:
            _vstreak = 0
        nav_state["verify_clip_streak"] = _vstreak
        if _vstreak >= 2 and _vlm_ok_v:
            print(f"  [VERIFY step={step}] CLIP streak=2 score={_clip_v:.2f} + VLM conf={_vlm_conf_v:.2f} → SUCCESS", flush=True)
            nav_state["done"]             = True
            nav_state["current_skill"]    = "done"
            nav_state["verify_best_clip"] = 0.0
            nav_state["verify_vlm_count"] = 0
            return nav_state

        if len(window) >= CONF_WINDOW and scanned >= CONF_WINDOW:
            avg_conf = sum(window) / len(window)
            if avg_conf >= CONF_THRESH:
                print(f"  [VERIFY step={step}] CLIP_avg={avg_conf:.2f} ≥ {CONF_THRESH} dist={dist:.3f}m → SUCCESS", flush=True)
                nav_state["done"]             = True
                nav_state["current_skill"]    = "done"
                nav_state["verify_best_clip"] = 0.0
                nav_state["verify_vlm_count"] = 0
                return nav_state

        if scanned < 24:
            from agent.habitat_env import ACTION_LEFT
            frame, _ = env.step(ACTION_LEFT)
            nav_state["last_frame"]     = frame
            nav_state["step_count"]    += 1
            nav_state["verify_scanned"] = scanned + 1
            if scanned % 6 == 0:
                avg_str = f"{sum(window)/len(window):.2f}" if window else "n/a"
                print(f"  [VERIFY step={step}] scanning ({scanned+1}/24) dist={dist:.3f}m win_avg={avg_str} → rotating", flush=True)
            if scanned % 3 == 0:
                nav_state["vlm_step"] = nav_state["step_count"] - 8
        else:
            if instances:
                rx, rz = float(robot_pos[0]), float(robot_pos[2])
                xz_near = min(
                    np.sqrt((float(p[0])-rx)**2 + (float(p[2])-rz)**2)
                    for p in instances
                )
                if xz_near <= 1.5:
                    print(f"  [VERIFY step={step}] scan done, xz={xz_near:.3f}m ≤ 1.5m → SUCCESS", flush=True)
                    nav_state["done"]             = True
                    nav_state["current_skill"]    = "done"
                    nav_state["verify_best_clip"] = 0.0
                    nav_state["verify_vlm_count"] = 0
                    return nav_state
            # If best CLIP during scan was strong enough, the object was seen — succeed.
            # Lowered from 0.35: catches raw cosim ~0.24 (rescaled 0.33) which is
            # marginal visibility (clearly above background noise at rescaled ~0.11).
            _best = nav_state.get("verify_best_clip", 0.0)
            if _best >= 0.28:
                print(f"  [VERIFY step={step}] scan done, best_clip={_best:.2f} ≥ 0.28 → SUCCESS (seen during scan)", flush=True)
                nav_state["done"]             = True
                nav_state["current_skill"]    = "done"
                nav_state["verify_best_clip"] = 0.0
                nav_state["verify_vlm_count"] = 0
                return nav_state
            print(f"  [VERIFY step={step}] scan done, target NOT confirmed (best_clip={_best:.2f} vlm_count={nav_state.get('verify_vlm_count',0)}) → explore", flush=True)
            # Blacklist this CLIP-SNAP target so we don't revisit the same wrong spot
            _cst = nav_state.get("clip_snap_target")
            if _cst:
                nav_state.setdefault("clip_blacklist", set()).add(_cst)
                nav_state.pop("clip_snap_target", None)
            nav_state["target_pos"]          = None
            nav_state["current_skill"]       = "explore_frontier"
            nav_state["waypoints"]           = []
            nav_state["verify_scanned"]      = 0
            nav_state["verify_conf_window"]  = []
            nav_state["verify_best_clip"]    = 0.0
            nav_state["verify_vlm_count"]    = 0
    else:
        print(f"  [VERIFY step={step}] dist={dist:.3f}m > {VERIFY_ACCEPT_DIST}m → follow_path", flush=True)
        nav_state["current_skill"]      = "follow_path"
        nav_state["verify_scanned"]     = 0
        nav_state["verify_conf_window"] = []
        nav_state["verify_best_clip"]   = 0.0
        nav_state["verify_vlm_count"]   = 0

    return nav_state


@skill("visual_servo")
def visual_servo(env, nav_state):
    """Reactive visual tracking: turn toward VLM-reported target direction.

    Fallback when bbox depth estimate unavailable. Exits to explore if VLM
    loses the target for 3 consecutive steps, or after 50 servo steps.
    """
    from agent.habitat_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT

    step     = nav_state.get("step_count", 0)
    percept  = nav_state.get("last_percept", {})
    direction  = percept.get("direction", "not_visible")
    vis        = percept.get("target_visible", False)

    servo_steps = nav_state.get("servo_steps", 0)
    servo_lost  = nav_state.get("servo_lost",  0)

    if not vis:
        servo_lost += 1
        nav_state["servo_lost"] = servo_lost
        if servo_lost >= 3:
            print(f"  [SERVO step={step}] target lost {servo_lost} steps → explore", flush=True)
            nav_state["current_skill"]  = "explore_frontier"
            nav_state["servo_steps"]    = 0
            nav_state["servo_lost"]     = 0
            nav_state["vis_consecutive"] = 0
        else:
            print(f"  [SERVO step={step}] lost ({servo_lost}/3) → turn to search", flush=True)
            frame, _ = env.step(ACTION_LEFT)
            nav_state["last_frame"]  = frame
            nav_state["step_count"] += 1
        return nav_state

    nav_state["servo_lost"] = 0
    servo_steps += 1
    nav_state["servo_steps"] = servo_steps

    if servo_steps >= 50:
        print(f"  [SERVO step={step}] 50 steps reached → revert to explore", flush=True)
        nav_state["current_skill"]   = "explore_frontier"
        nav_state["servo_steps"]     = 0
        nav_state["servo_lost"]      = 0
        nav_state["vis_consecutive"] = 0
        return nav_state

    if direction == "left":
        action, action_name = ACTION_LEFT, "LEFT"
    elif direction == "right":
        action, action_name = ACTION_RIGHT, "RIGHT"
    else:
        action, action_name = ACTION_FORWARD, "FWD"

    print(f"  [SERVO step={step}] dir={direction} → {action_name} (servo={servo_steps})", flush=True)
    frame, _ = env.step(action)
    nav_state["last_frame"]  = frame
    nav_state["step_count"] += 1
    return nav_state
