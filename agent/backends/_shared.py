"""
backends/_shared.py — Utilities shared across all VLM backends.

Extracted from agent/llm_agent.py:
  _frame_to_jpeg_b64, _build_perceive_prompt, _parse_percept_json
"""
import base64
import json
import numpy as np


def _frame_to_jpeg_b64(frame: np.ndarray) -> str:
    """Encode RGB numpy array as JPEG base64 string."""
    import cv2
    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    return base64.b64encode(buf.tobytes()).decode()


def _build_perceive_prompt(goal: str, n_waypoints: int = 0, context: dict = None) -> str:
    """Build the text portion of the VLM perception prompt."""
    if n_waypoints >= 2:
        waypoint_rule = (
            f"- waypoint: 0-{n_waypoints}, choose the numbered circle most likely "
            f"to lead toward {goal}; 0 means none suitable\n"
        )
        waypoint_field = ',"waypoint":int'
    else:
        waypoint_rule = ""
        waypoint_field = ""

    topo_line = ""
    history_str = ""
    anti_loop_alert = ""
    if context:
        topo = context.get('topo_summary', '')
        if topo and topo != "0 nodes":
            topo_line = f"Map memory: {topo}\n"
        _h = context.get("history", [])
        if _h:
            history_str = "Recent decisions: " + " → ".join(
                f"step{e['step']}:{e['skill']}({e.get('reason','')[:30]})" for e in _h
            ) + "\n"
            # Detect room loop: if last 3 entries all report the same non-trivial room
            _recent_rooms = [e.get("room", "") for e in _h if e.get("room") and e.get("room") != "other"]
            if len(_recent_rooms) >= 3 and len(set(_recent_rooms)) == 1:
                anti_loop_alert = (
                    f"WARNING: You have been in '{_recent_rooms[0]}' for {len(_recent_rooms)} "
                    f"consecutive VLM calls without finding {goal}. "
                    f"If {goal} is STILL not visible, you MUST choose skill=escape "
                    f"and select a waypoint leading to a DIFFERENT area or room.\n"
                )

    if context:
        ctx_str = (
            f"Navigation state: step {context.get('step', 0)}/{context.get('max_steps', 500)}"
            f" | explored {context.get('explored_pct', 0):.0%}"
            f" | stagnant {context.get('stagnant_steps', 0)} steps\n"
            f"Rooms seen: {context.get('rooms_str', 'none yet')}\n"
            f"Nearest {goal}: {context.get('nearest_dist_str', 'unknown')}\n"
            + topo_line + history_str + anti_loop_alert
        )
        skill_field = ',"skill":"explore|snap|escape|verify","reason":"one sentence why"'
        skill_rules = (
            "Skill (choose one action for THIS step):\n"
            f'- "snap": {goal} is clearly visible, navigate to it NOW\n'
            '- "explore": keep searching, pick numbered waypoint (0=auto)\n'
            '- "escape": I am stuck/looping, need a completely different area\n'
            f'- "verify": I am very close to {goal}, confirm arrival\n'
            f'- reason: ONE short sentence explaining your choice '
            f'(e.g. "{goal} visible on left side", "no {goal} found, continuing search")\n'
        )
    else:
        ctx_str = ""
        skill_field = ""
        skill_rules = ""

    # Goal-specific room relevance hints for commonsense reasoning
    _ROOM_HINTS = {
        "沙发": "living_room=0.9, bedroom=0.15, hallway=0.3, kitchen=0.0",
        "床":   "bedroom=0.95, living_room=0.1, hallway=0.2, kitchen=0.0",
        "电视": "living_room=0.85, bedroom=0.45, hallway=0.1, kitchen=0.05",
        "桌子": "kitchen=0.8, living_room=0.75, bedroom=0.3, hallway=0.1",
        "冰箱": "kitchen=0.95, living_room=0.05, bedroom=0.0, hallway=0.05",
        "椅子": "living_room=0.8, kitchen=0.75, bedroom=0.45, hallway=0.2",
        "厕所": "bathroom=0.95, hallway=0.3, bedroom=0.1, living_room=0.0",
        "水槽": "bathroom=0.8, kitchen=0.85, hallway=0.1, living_room=0.0",
    }
    _room_hint = _ROOM_HINTS.get(goal, "use common sense about which rooms typically contain this object")

    _ex_wp   = ',"waypoint":0'   if waypoint_field else ""
    _ex_sk_y = ',"skill":"snap","reason":"target visible"'    if skill_field else ""
    _ex_sk_n = ',"skill":"explore","reason":"not found"'      if skill_field else ""
    _ex_vis  = (
        '{"direction":"center","confidence":0.85,"room":"living_room","relevance":0.9'
        + _ex_wp + _ex_sk_y + "}"
    )
    _ex_hid  = (
        '{"direction":"not_visible","confidence":0.0,"room":"hallway","relevance":0.3'
        + _ex_wp + _ex_sk_n + "}"
    )
    return (
        f"You are a home navigation robot brain. Navigation goal: {goal}\n"
        + ctx_str
        + "Observe the entire image carefully. Return ONE JSON line, no other text.\n"
        + "REQUIRED JSON:\n"
        + '{"direction":"left|center|right|not_visible","confidence":float,'
        + '"room":"living_room|bedroom|hallway|kitchen|staircase|bathroom|other",'
        + f'"relevance":float{waypoint_field}{skill_field}}}\n'
        + f"EXAMPLE when {goal} visible: {_ex_vis}\n"
        + f"EXAMPLE when {goal} not visible: {_ex_hid}\n"
        + "Rules:\n"
        + f"- direction: left|center|right if {goal} seems present in that area, not_visible if absent\n"
        + "- confidence: 0.0-1.0 how sure you are about direction and room\n"
        + "- room: room type you are currently in\n"
        + f"- relevance: 0.0-1.0, use ROOM COMMONSENSE — where is {goal} typically found?\n"
        + f"  ({_room_hint})\n"
        + f"  If you are in the WRONG room type for {goal}, relevance must be LOW (≤0.2).\n"
        + waypoint_rule + skill_rules
    )


def _parse_percept_json(text: str, goal: str) -> dict:
    """Parse VLM JSON output with fallback to rule-based result."""
    text = (text or "").strip()
    if text:
        print(f"[VLM-RAW] {text[:400]}", flush=True)
    if not text or "{" not in text:
        return _rule_fallback(goal)
    try:
        js, je = text.rfind("{"), text.rfind("}") + 1
        result = json.loads(text[js:je])
        # Only zero-out confidence/direction when target_visible is EXPLICITLY false.
        # If the field is absent (new schema without target_visible), preserve VLM signals.
        if "target_visible" in result and not result["target_visible"]:
            result["confidence"] = 0.0
            result["direction"]  = "not_visible"
        return result
    except Exception:
        return _rule_fallback(goal)


def _rule_fallback(goal: str) -> dict:
    return {
        "target_visible": False, "direction": "not_visible", "distance": 99.0,
        "confidence": 0.0, "room": "other", "relevance": 0.2,
    }
