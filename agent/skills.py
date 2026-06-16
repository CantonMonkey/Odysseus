"""
skills.py

Three reusable navigation skills:
  follow_path    – primary skill: follow a Habitat pathfinder shortest path
  search_room    – fallback: rotate in place waiting for LLM perception
  verify_arrival – check whether the robot has reached the goal
"""

import numpy as np
import habitat_sim

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

    if dist <= ARRIVE_DIST:
        print(f"  [FOLLOW step={step}] dist={dist:.3f}m ≤ {ARRIVE_DIST}m → verify_arrival", flush=True)
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
        if tgt:
            key = (round(tgt[0], 1), round(tgt[2], 1))
            nav_state.setdefault("blacklisted_snap", set()).add(key)
            print(f"  [FOLLOW step={step}] stagnant {FOLLOW_STAGNANT_LIMIT} steps dist={dist:.2f}m → blacklist ({key[0]},{key[1]}) + explore", flush=True)
        nav_state["target_pos"] = None
        nav_state["current_skill"] = "explore_frontier"
        nav_state["follow_stagnant"] = 0
        nav_state["waypoints"] = []
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

def verify_arrival(env, nav_state: dict) -> dict:
    """Confirm arrival at goal.

    Success: VLM confirms target visible while within ARRIVE_DIST of estimate,
    or robot is within 0.8m of estimate and VLM sees it.
    If arrived but VLM doesn't see the target even after scanning, depth
    estimate was wrong — revert to explore.
    """
    robot_pos, _ = env.get_robot_pose()
    target_pos   = nav_state.get("target_pos")
    dist = _euclidean(robot_pos, target_pos) if target_pos else float("inf")
    step = nav_state["step_count"]

    percept        = nav_state.get("last_percept", {})
    target_visible = percept.get("target_visible", False)
    confidence     = float(percept.get("confidence", 0.0))

    # SNAP centroid may be at object centre height (e.g. fridge ~1m off floor),
    # so 3D dist plateaus at ~1.2-1.5m even when the robot is adjacent.  Raise
    # the success threshold to 1.5m; stagnation detection handles the rest.
    vlm_confirmed = target_visible and confidence >= 0.25 and dist <= 1.5

    if vlm_confirmed:
        print(f"  [VERIFY step={step}] SUCCESS dist_to_tgt={dist:.3f}m vis={target_visible} conf={confidence:.2f} → DONE", flush=True)
        nav_state["done"]          = True
        nav_state["current_skill"] = "done"
    elif dist <= ARRIVE_DIST and target_visible and confidence >= 0.25:
        from agent.habitat_env import ACTION_FORWARD, ACTION_LEFT
        to_target = np.array(target_pos) - robot_pos
        forward = _get_forward(env)
        angle, turn_action = _turn_to(forward, to_target)
        step_action = ACTION_FORWARD if angle < ALIGN_THRESH else turn_action
        action_name = "FORWARD" if step_action == ACTION_FORWARD else "TURN"
        # Stagnation: robot is physically blocked by the object — declare success.
        last_vd = nav_state.get("verify_last_dist", dist + 1.0)
        vstag   = nav_state.get("verify_stagnant", 0)
        vstag   = vstag + 1 if dist >= last_vd - 0.05 else 0
        nav_state["verify_last_dist"] = dist
        nav_state["verify_stagnant"]  = vstag
        if vstag >= 8:
            print(f"  [VERIFY step={step}] STAGNANT {vstag} steps dist={dist:.3f}m (blocked by object) vis=True → SUCCESS", flush=True)
            nav_state["done"]          = True
            nav_state["current_skill"] = "done"
            return nav_state
        print(f"  [VERIFY step={step}] stepping {action_name} toward target dist={dist:.3f}m (need ≤1.5m) vis=True conf={confidence:.2f}", flush=True)
        frame, _ = env.step(step_action)
        nav_state["last_frame"]  = frame
        nav_state["step_count"] += 1
        nav_state["verify_scanned"] = 0
    elif dist <= ARRIVE_DIST and not target_visible:
        scanned = nav_state.get("verify_scanned", 0)
        if scanned < 24:
            from agent.habitat_env import ACTION_LEFT
            frame, _ = env.step(ACTION_LEFT)
            nav_state["last_frame"]     = frame
            nav_state["step_count"]    += 1
            nav_state["verify_scanned"] = scanned + 1
            if scanned % 6 == 0:
                print(f"  [VERIFY step={step}] scanning ({scanned+1}/24) dist={dist:.3f}m vis=False → rotating", flush=True)
            if scanned % 3 == 0:
                nav_state["vlm_step"] = nav_state["step_count"] - 8
        else:
            print(f"  [VERIFY step={step}] 360° scan complete, target NOT found → revert to explore (depth estimate wrong)", flush=True)
            nav_state["target_pos"]     = None
            nav_state["current_skill"]  = "explore_frontier"
            nav_state["waypoints"]      = []
            nav_state["verify_scanned"] = 0
    else:
        print(f"  [VERIFY step={step}] dist={dist:.3f}m > ARRIVE_DIST={ARRIVE_DIST}m → back to follow_path", flush=True)
        nav_state["current_skill"] = "follow_path"
        nav_state["verify_scanned"] = 0

    return nav_state
