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
    if context:
        topo = context.get('topo_summary', '')
        if topo and topo != "0 nodes":
            topo_line = f"Map memory: {topo}\n"
        _h = context.get("history", [])
        if _h:
            history_str = "Recent decisions: " + " → ".join(
                f"step{e['step']}:{e['skill']}({e.get('reason','')[:30]})" for e in _h
            ) + "\n"

    if context:
        ctx_str = (
            f"Navigation state: step {context.get('step', 0)}/{context.get('max_steps', 500)}"
            f" | explored {context.get('explored_pct', 0):.0%}"
            f" | stagnant {context.get('stagnant_steps', 0)} steps\n"
            f"Rooms seen: {context.get('rooms_str', 'none yet')}\n"
            f"Nearest {goal}: {context.get('nearest_dist_str', 'unknown')}\n"
            + topo_line + history_str
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

    return (
        f"You are a home navigation robot brain. Navigation goal: {goal}\n"
        + ctx_str
        + "Observe the entire image carefully. Return ONE JSON line, no other text:\n"
        + '{"target_visible":bool,"direction":"left|center|right|not_visible",'
        + '"confidence":float,"room":"living_room|bedroom|hallway|kitchen|staircase|bathroom|other",'
        + f'"relevance":float{waypoint_field}{skill_field}}}\n'
        + "Rules:\n"
        + f"- target_visible=true: {goal} is visible ANYWHERE (background/doorway/corner counts)\n"
        + "- confidence: if visible 0.1-1.0 (partial/far=0.3-0.6, clear=0.8+), else 0.0\n"
        + "- room: room type you are currently in\n"
        + f"- relevance: 0.0-1.0, how likely navigating this direction leads to {goal}\n"
        + "  (living_room for sofa/chair=0.9, hallway=0.4, bedroom for sofa=0.1)\n"
        + "- direction: where the target is (left/center/right), not_visible if absent\n"
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
        if not result.get("target_visible", False):
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
