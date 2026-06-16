"""
loop.py — VLFM-style online exploration loop.

The robot no longer uses a pre-computed semantic map.  Instead:
  1. Explore the environment by navigating to frontiers (unexplored boundaries).
  2. Call the VLM every VLM_CALL_INTERVAL steps to score the current view.
  3. Accumulate scores in an online value map (ExploreMap).
  4. When VLM reports target visible with high confidence, estimate its 3D
     position from depth + viewing direction and switch to follow_path.
  5. Verify arrival within ARRIVE_DIST of the estimated target.

This satisfies the "only visual info + robot state, no privileged labels"
requirement from the course specification.
"""

import numpy as np
import habitat_sim
from typing import Callable, Optional

from agent.explore_map import ExploreMap, VLM_CALL_INTERVAL

MAX_STEPS  = 300
ARRIVE_DIST = 1.2
VLM_CONF_THRESHOLD = 0.55   # confidence above this → treat target as found
HINT_INTERVAL      = 40     # ask spatial-reasoning LLM every N steps

# Default scene — overridden by server/main.py at startup
from pathlib import Path
_DATA = Path("/data3/liangjy/vln/data/hm3d")
SCENE_DIR = str(_DATA / "00800-TEEsavR23oF")


# ── 3-D target localisation from VLM direction hint ────────────────────────

def _estimate_target_pos(
    depth_frame: np.ndarray,
    direction: str,
    agent_pos: np.ndarray,
    R: np.ndarray,
    hfov: float = 90.0,
) -> Optional[np.ndarray]:
    """
    Convert (direction, depth) to an approximate world-frame 3D position.

    direction : "left" | "center" | "right" (from VLM)
    Returns [x, y, z] in world frame, or None if depth is invalid.
    """
    H, W   = depth_frame.shape
    fx     = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    col = {"left": W // 4, "center": W // 2, "right": 3 * W // 4}.get(
        direction, W // 2
    )
    u, v = col, int(cy)

    # Median depth inside a patch around the hinted column
    u1, u2 = max(0, u - 30), min(W, u + 30)
    v1, v2 = max(0, v - 40), min(H, v + 40)
    patch = depth_frame[v1:v2, u1:u2]
    valid = patch[(patch > 0.3) & (patch < 7.0)]
    if len(valid) == 0:
        return None

    d = float(np.median(valid))

    # Camera-local 3-D point (camera looks along -Z in local frame)
    x_c = (u - cx) * d / fx
    y_c = 0.0    # keep at eye level — we care about XZ for navigation
    z_c = -d

    cam_pos      = agent_pos.copy()
    cam_pos[1]  += 1.0          # EYE_HEIGHT

    p_world = R @ np.array([x_c, y_c, z_c]) + cam_pos
    return p_world.astype(np.float32)


# ── frontier skill ──────────────────────────────────────────────────────────

def _explore_frontier(env, nav_state: dict, explore_map: ExploreMap) -> dict:
    """
    Navigate toward the highest-value unexplored frontier.
    Falls back to slow rotation when no frontiers remain.
    """
    from agent.skills import _replan, _get_forward, _turn_to, _euclidean
    from agent.habitat_env import ACTION_FORWARD, ACTION_LEFT
    from agent.skills import WP_REACH, ALIGN_THRESH

    robot_pos, _ = env.get_robot_pose()

    # If we've reached the current frontier, clear it so we pick a fresh one
    f_pos = nav_state.get("frontier_pos")
    if f_pos is not None and _euclidean(robot_pos, f_pos) < ARRIVE_DIST:
        nav_state["frontier_pos"] = None

    # Pick a new frontier when needed
    if nav_state.get("frontier_pos") is None or not nav_state.get("waypoints"):
        new_f = explore_map.best_frontier(robot_pos)
        if new_f is not None:
            nav_state["frontier_pos"] = new_f.tolist()
            nav_state["waypoints"]    = _replan(env, robot_pos, new_f)
        else:
            # All frontiers exhausted: rotate to fill value map
            frame, _ = env.step(ACTION_LEFT)
            nav_state["last_frame"]   = frame
            nav_state["step_count"]  += 1
            return nav_state

    # Follow waypoints toward the frontier (same steering logic as follow_path)
    waypoints = nav_state.get("waypoints", [])
    if not waypoints:
        action = ACTION_LEFT
    else:
        # Pop waypoints that are already within reach
        while waypoints and _euclidean(robot_pos, waypoints[0]) < WP_REACH:
            waypoints.pop(0)

        if not waypoints:
            action = ACTION_LEFT
        else:
            next_wp         = np.array(waypoints[0])
            to_wp           = next_wp - robot_pos
            forward         = _get_forward(env)
            angle, action   = _turn_to(forward, to_wp)
            if angle < ALIGN_THRESH:
                action = ACTION_FORWARD

    frame, _ = env.step(action)

    # Pop waypoints now that we've moved
    new_pos, _ = env.get_robot_pose()
    while waypoints and _euclidean(new_pos, waypoints[0]) < WP_REACH:
        waypoints.pop(0)

    nav_state["waypoints"]   = waypoints
    nav_state["last_frame"]  = frame
    nav_state["step_count"] += 1
    return nav_state


# ── spatial reasoning hint ──────────────────────────────────────────────────

def _ask_explore_hint(frame, goal: str, llm_perceive) -> dict:
    """
    Ask the VLM a richer spatial question every HINT_INTERVAL steps.

    Returns a hint dict:
        {"room": str, "stairs_visible": bool, "suggest": str}
    suggest: "go_upstairs" | "keep_exploring" | "turn_around" | "none"

    This lets the robot reason: "I'm in the living room, bed is probably
    upstairs — if I see stairs I should go up."
    """
    import json
    try:
        import io, base64
        import numpy as np
        from PIL import Image
        from agent.llm_agent import _get_client, _extract_text, _MODEL_PERCEIVE

        client = _get_client()
        if client is None:
            return {}

        img = Image.fromarray(frame.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = (
            f"你是家居导航机器人。导航目标：{goal}\n"
            "观察当前第一视角画面，回答以下问题，只返回一行JSON：\n"
            '{"room":"客厅|卧室|厨房|走廊|楼梯间|浴室|其他",'
            '"stairs_visible":bool,'
            '"suggest":"go_upstairs|keep_exploring|turn_around|none",'
            '"reason":"一句话说明建议理由"}\n'
            "suggest规则:\n"
            "- 若目标通常在其他楼层（如床在卧室/二楼）且当前看到楼梯 → go_upstairs\n"
            "- 若当前房间不可能有目标且无楼梯 → turn_around\n"
            "- 其他情况 → keep_exploring"
        )

        resp = client.messages.create(
            model=_MODEL_PERCEIVE,
            max_tokens=128,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        text = _extract_text(resp.content).strip()
        if not text or "{" not in text:
            return {}
        return json.loads(text[text.find("{"):text.rfind("}")+1])
    except Exception:
        return {}


# ── main task loop ──────────────────────────────────────────────────────────

def run_task(
    env,
    task: str,
    scene_dir: str = SCENE_DIR,
    on_frame: Optional[Callable] = None,
    llm_perceive=None,
) -> dict:
    """
    Exploration-based navigation.  No pre-computed semantic labels used.

    env          – HabitatEnv (already reset to the target scene)
    task         – Chinese goal keyword, e.g. "沙发"
    on_frame     – optional callback(frame, nav_state) for WebSocket streaming
    llm_perceive – optional function(frame, goal) → percept dict
                   {"target_visible", "direction", "distance", "confidence"}
    """
    from agent.skills import follow_path, verify_arrival

    explore_map = ExploreMap()

    nav_state: dict = {
        "goal":          task,
        "target_pos":    None,        # set when VLM localises the object
        "step_count":    0,
        "current_skill": "explore_frontier",
        "done":          False,
        "last_frame":    env.get_frame(),
        "last_percept":  {"target_visible": False, "confidence": 0.0},
        "waypoints":     [],
        "explore_map":   explore_map,
        "frontier_pos":  None,
        "vlm_step":      -VLM_CALL_INTERVAL,  # trigger VLM immediately
        "hint_step":     -HINT_INTERVAL,       # trigger spatial hint immediately
        "last_hint":     {},
    }

    if on_frame:
        on_frame(nav_state["last_frame"], nav_state)

    skill_map = {
        "follow_path":    follow_path,
        "verify_arrival": verify_arrival,
    }

    while not nav_state["done"] and nav_state["step_count"] < MAX_STEPS:
        step = nav_state["step_count"]

        # ── robot state ────────────────────────────────────────────────
        robot_pos, _ = env.get_robot_pose()
        R             = env.get_rotation_matrix()

        # ── VLM perception (throttled) ─────────────────────────────────
        if (
            llm_perceive is not None
            and (step - nav_state["vlm_step"]) >= VLM_CALL_INTERVAL
        ):
            try:
                percept               = llm_perceive(nav_state["last_frame"], task)
                nav_state["last_percept"] = percept
                nav_state["vlm_step"]     = step

                confidence = float(percept.get("confidence", 0.0))

                if percept.get("target_visible") and confidence >= VLM_CONF_THRESHOLD:
                    # Localise target in 3-D using depth + direction hint
                    depth     = env.get_depth()
                    direction = percept.get("direction", "center")
                    tgt       = _estimate_target_pos(depth, direction, robot_pos, R)
                    if tgt is not None:
                        nav_state["target_pos"]     = tgt.tolist()
                        nav_state["current_skill"]  = "follow_path"
                        nav_state["waypoints"]       = []
            except Exception:
                pass

        # ── spatial reasoning hint (less frequent) ───────────────────
        if (
            llm_perceive is not None
            and nav_state.get("target_pos") is None
            and (step - nav_state["hint_step"]) >= HINT_INTERVAL
        ):
            hint = _ask_explore_hint(nav_state["last_frame"], task, llm_perceive)
            if hint:
                nav_state["last_hint"] = hint
                nav_state["hint_step"] = step
                suggest = hint.get("suggest", "none")
                room    = hint.get("room", "?")
                reason  = hint.get("reason", "")
                stairs  = hint.get("stairs_visible", False)
                # Log the spatial reasoning
                import sys
                print(f"  [HINT step={step}] room={room} stairs={stairs} "
                      f"suggest={suggest} | {reason}", file=sys.stderr)

        # ── update value map ───────────────────────────────────────────
        vlm_score = float(nav_state["last_percept"].get("confidence", 0.0))
        explore_map.update(robot_pos, R, vlm_score)

        # ── execute skill ──────────────────────────────────────────────
        current = nav_state.get("current_skill", "explore_frontier")

        if current == "done":
            nav_state["done"] = True
            break
        elif current in skill_map:
            nav_state = skill_map[current](env, nav_state)
        else:
            nav_state = _explore_frontier(env, nav_state, explore_map)

        # ── stream frame ───────────────────────────────────────────────
        if on_frame and nav_state.get("last_frame") is not None:
            on_frame(nav_state["last_frame"], nav_state)

    if nav_state["step_count"] >= MAX_STEPS and not nav_state["done"]:
        nav_state["timeout"] = True

    return nav_state


# ── smoke test ──────────────────────────────────────────────────────────────

def demo(scene_dir: str = SCENE_DIR, target: str = "沙发"):
    """Run the exploration loop, saving frames to /tmp/loop_frames/."""
    import imageio
    from pathlib import Path
    from agent.habitat_env import HabitatEnv
    from agent.llm_agent import perceive

    out_dir = Path("/tmp/loop_frames")
    out_dir.mkdir(exist_ok=True)

    env       = HabitatEnv(gpu_id=0)
    idx       = [0]

    env.reset(scene_dir)

    def save_frame(frame, state):
        imageio.imwrite(str(out_dir / f"frame_{idx[0]:04d}.png"), frame)
        s = state.get("step_count", 0)
        if s % 10 == 0:
            skill = state.get("current_skill", "?")
            conf  = state.get("last_percept", {}).get("confidence", 0.0)
            expl  = state["explore_map"].explored_fraction() * 100
            print(f"  step={s:03d} skill={skill} conf={conf:.2f} explored={expl:.1f}%")
        idx[0] += 1

    print(f"Task: '{target}' (exploration mode, VLM enabled)")
    result = run_task(
        env, target, scene_dir=scene_dir,
        on_frame=save_frame,
        llm_perceive=lambda f, g: perceive(f, g),
    )
    env.close()

    status = "arrived" if result["done"] else ("timeout" if result.get("timeout") else "?")
    expl   = result["explore_map"].explored_fraction() * 100
    print(f"\n{status}: steps={result['step_count']}, explored={expl:.1f}%")
    return result


if __name__ == "__main__":
    demo()
