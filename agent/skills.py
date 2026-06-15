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

ARRIVE_DIST  = 1.2   # goal-reached threshold (m)
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
    Falls back to search_room if no path can be found.
    """
    from agent.habitat_env import ACTION_FORWARD

    target_pos = nav_state.get("target_pos")
    if target_pos is None:
        nav_state["current_skill"] = "search_room"
        return nav_state

    robot_pos, _ = env.get_robot_pose()
    dist = _euclidean(robot_pos, target_pos)

    if dist <= ARRIVE_DIST:
        nav_state["current_skill"] = "verify_arrival"
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
        # Pathfinder found no route; spin in place until LLM can guide
        nav_state["current_skill"] = "search_room"
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
    """Rotate in place (and occasionally step) waiting for LLM guidance."""
    from agent.habitat_env import ACTION_LEFT, ACTION_FORWARD

    rotated = nav_state.get("search_rotated", 0.0)

    if rotated >= 360.0:
        frame, _ = env.step(ACTION_FORWARD)
        nav_state["search_rotated"] = 0.0
    else:
        frame, _ = env.step(ACTION_LEFT)
        nav_state["search_rotated"] = rotated + 15.0

    nav_state["last_frame"]  = frame
    nav_state["step_count"] += 1
    return nav_state


# ── Skill 3: verify_arrival ─────────────────────────────────────

def verify_arrival(env, nav_state: dict) -> dict:
    """Confirm arrival: Euclidean distance < ARRIVE_DIST, or LLM says so."""
    robot_pos, _ = env.get_robot_pose()
    target_pos   = nav_state.get("target_pos")
    dist = _euclidean(robot_pos, target_pos) if target_pos else float("inf")

    percept      = nav_state.get("last_percept", {})
    target_visible = percept.get("target_visible", False)
    percept_dist   = percept.get("distance", float("inf"))

    arrived = dist <= ARRIVE_DIST or (target_visible and percept_dist <= ARRIVE_DIST)

    if arrived:
        nav_state["done"]          = True
        nav_state["current_skill"] = "done"
    else:
        nav_state["current_skill"] = "follow_path"

    return nav_state
