"""
skills.py
三个导航技能：follow_path / search_room / verify_arrival
核心技能 follow_path 使用 Habitat pathfinder 算最短路径，用旋转矩阵确定转向。
"""

import numpy as np
import habitat_sim

try:
    import quaternion as npq
    _HAS_QUATERNION = True
except ImportError:
    _HAS_QUATERNION = False

ARRIVE_DIST = 1.2    # 到达阈值（m）
ALIGN_THRESH = 12.0  # 朝向对齐阈值（度）
WP_REACH = 0.4       # 路径点到达阈值（m）


def _get_forward(env) -> np.ndarray:
    """从 agent rotation 获取世界坐标系中的前进方向向量（水平）。"""
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
    fwd = R @ np.array([0.0, 0.0, -1.0])   # local -Z → world
    fwd[1] = 0.0
    norm = np.linalg.norm(fwd)
    return fwd / norm if norm > 1e-6 else np.array([0.0, 0.0, -1.0])


def _turn_to(forward: np.ndarray, to_wp: np.ndarray):
    """
    返回 (angle_deg, action)：需要转多少度，以及转哪边。
    forward, to_wp 均为归一化水平向量。
    """
    from agent.habitat_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT
    to_wp = to_wp.copy()
    to_wp[1] = 0.0
    n = np.linalg.norm(to_wp)
    if n < 1e-6:
        return 0.0, ACTION_FORWARD
    to_wp /= n

    dot = float(np.clip(np.dot(forward, to_wp), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(dot)))

    # turn_left rotates toward -X, turn_right toward +X (verified empirically)
    # (forward × to_wp)[y] = forward[2]*to_wp[0] - forward[0]*to_wp[2]
    # positive → to_wp is to the LEFT (turn_left), negative → RIGHT (turn_right)
    cross_y = forward[2] * to_wp[0] - forward[0] * to_wp[2]
    action = ACTION_LEFT if cross_y > 0 else ACTION_RIGHT
    return angle, action


def _euclidean(a: np.ndarray, b) -> float:
    return float(np.linalg.norm(a - np.array(b, dtype=np.float32)))


# ── 技能 1：follow_path ─────────────────────────────────────────
# （主导航技能：pathfinder 最短路 → 跟随路径点）

def follow_path(env, nav_state: dict) -> dict:
    """
    用 pathfinder 找到并跟随最短路径到目标。
    路径点用旋转矩阵确定转向方向，不依赖 heading 角度约定。
    """
    from agent.habitat_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT

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

    # 当前路径点已到达或没有路径点 → 重新规划
    if not waypoints or _euclidean(robot_pos, waypoints[0]) < WP_REACH:
        if waypoints:
            waypoints.pop(0)
        if not waypoints:
            waypoints = _replan(env, robot_pos, target_pos)
            nav_state["waypoints"] = waypoints

    if not waypoints:
        # pathfinder 找不到路径：原地旋转等 LLM 引导
        nav_state["current_skill"] = "search_room"
        return nav_state

    next_wp = np.array(waypoints[0])
    to_wp = next_wp - robot_pos
    forward = _get_forward(env)
    angle, action = _turn_to(forward, to_wp)

    if angle < ALIGN_THRESH:
        action = ACTION_FORWARD

    frame, _ = env.step(action)
    nav_state["last_frame"] = frame
    nav_state["step_count"] += 1

    # 如果前进后到达了当前路径点，弹出
    new_pos, _ = env.get_robot_pose()
    if waypoints and _euclidean(new_pos, waypoints[0]) < WP_REACH:
        waypoints.pop(0)
    nav_state["waypoints"] = waypoints

    return nav_state


def _replan(env, robot_pos: np.ndarray, target_pos) -> list:
    """重新用 pathfinder 规划路径，返回路径点列表（不含起点）。"""
    pf = env._sim.pathfinder
    target_np = np.array(target_pos, dtype=np.float32)
    snapped = pf.snap_point(target_np)
    if np.any(np.isnan(snapped)):
        return []
    path = habitat_sim.ShortestPath()
    path.requested_start = robot_pos.astype(np.float32)
    path.requested_end = snapped
    if pf.find_path(path) and len(path.points) > 1:
        return [p.tolist() for p in path.points[1:]]
    return []


# ── 技能 2：search_room ─────────────────────────────────────────
# （fallback：没有 target_pos 时随机旋转探索，等 LLM 感知结果）

def search_room(env, nav_state: dict) -> dict:
    """无目标坐标时：原地旋转等待 LLM 感知提供方向。"""
    from agent.habitat_env import ACTION_LEFT, ACTION_FORWARD

    rotated = nav_state.get("search_rotated", 0.0)

    # 旋转满一圈后前进换位置
    if rotated >= 360.0:
        frame, _ = env.step(ACTION_FORWARD)
        nav_state["search_rotated"] = 0.0
    else:
        frame, _ = env.step(ACTION_LEFT)
        nav_state["search_rotated"] = rotated + 15.0

    nav_state["last_frame"] = frame
    nav_state["step_count"] += 1
    return nav_state


# ── 技能 3：verify_arrival ──────────────────────────────────────

def verify_arrival(env, nav_state: dict) -> dict:
    """硬判断：欧氏距离 < ARRIVE_DIST → 任务完成。"""
    robot_pos, _ = env.get_robot_pose()
    target_pos = nav_state.get("target_pos")
    dist = _euclidean(robot_pos, target_pos) if target_pos else float("inf")

    percept = nav_state.get("last_percept", {})
    target_visible = percept.get("target_visible", False)
    percept_dist = percept.get("distance", float("inf"))

    arrived = dist <= ARRIVE_DIST or (target_visible and percept_dist <= ARRIVE_DIST)

    if arrived:
        nav_state["done"] = True
        nav_state["current_skill"] = "done"
    else:
        nav_state["current_skill"] = "follow_path"

    return nav_state
