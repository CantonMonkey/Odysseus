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


def _build_perceive_prompt(goal: str, n_waypoints: int = 0, context: dict = None, clip_state: dict = None) -> str:
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
        _strategy_line = ""
        if context.get("search_strategy"):
            _phases = context["search_strategy"]
            _pidx   = context.get("strategy_phase", 0)
            _pcur   = _phases[_pidx] if _pidx < len(_phases) else "done"
            _strategy_line = (
                f"Search plan: {' → '.join(_phases)} | "
                f"Current phase: {_pcur} (phase {_pidx+1}/{len(_phases)})\n"
            )
        ctx_str = (
            _strategy_line
            + f"Navigation state: step {context.get('step', 0)}/{context.get('max_steps', 500)}"
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
    # Floor-level reasoning: which floor is each object most likely on?
    # Used to guide staircase decisions ("go upstairs" vs "stay down").
    _FLOOR_HINTS = {
        "沙发": "ground floor (living room is almost always on ground floor)",
        "床":   "upper floor preferred (bedrooms often upstairs); if you see stairs and haven't found it downstairs, go up",
        "电视": "ground floor preferred (main living room), but may be in upstairs bedroom",
        "桌子": "ground floor preferred (kitchen/dining), could be anywhere",
        "冰箱": "ground floor ONLY — kitchens are never upstairs. Do NOT go upstairs for 冰箱",
        "椅子": "ground floor preferred (kitchen/dining/living room)",
        "厕所": "ground floor preferred, but some homes have upstairs bathroom",
        "水槽": "ground floor preferred (kitchen/bathroom on ground floor)",
    }
    _room_hint  = _ROOM_HINTS.get(goal, "use common sense about which rooms typically contain this object")
    _floor_hint = _FLOOR_HINTS.get(goal, "use common sense about which floor this object is on")

    _ex_wp   = ',"waypoint":0'   if waypoint_field else ""
    _ex_sk_y = ',"skill":"snap","reason":"target visible"'    if skill_field else ""
    _ex_sk_n = ',"skill":"explore","reason":"not found"'      if skill_field else ""
    _ex_vis  = (
        '{"direction":"center","confidence":0.85,"room":"living_room","relevance":0.9,"search_direction":"none"'
        + _ex_wp + _ex_sk_y + "}"
    )
    _ex_hid  = (
        '{"direction":"not_visible","confidence":0.0,"room":"hallway","relevance":0.3,"search_direction":"left"'
        + _ex_wp + _ex_sk_n + "}"
    )
    _clip_alert = ""
    if clip_state and clip_state.get("streak", 0) >= 2:
        _clip_alert = (
            f"SENSOR ALERT: Low-level CLIP detector has seen '{goal}' for "
            f"{clip_state['streak']} consecutive frames "
            f"(confidence={clip_state['score']:.2f}, direction={clip_state['direction']}). "
            f"This is a strong signal from the sensor. If you also see '{goal}' in the image, "
            f"you MUST output skill=snap immediately.\n"
        )

    return (
        f"You are a home navigation robot brain. Navigation goal: {goal}\n"
        + _clip_alert
        + ctx_str
        + "Observe the entire image carefully. Return ONE JSON line, no other text.\n"
        + "REQUIRED JSON:\n"
        + '{"direction":"left|center|right|not_visible","confidence":float,'
        + '"room":"living_room|bedroom|hallway|kitchen|staircase|bathroom|other",'
        + f'"relevance":float,"search_direction":"left|center|right|upstairs|none"'
        + f'{waypoint_field}{skill_field}}}\n'
        + f"EXAMPLE when {goal} visible: {_ex_vis}\n"
        + f"EXAMPLE when {goal} not visible: {_ex_hid}\n"
        + "Rules:\n"
        + f"- direction: left|center|right if {goal} seems present in that area, not_visible if absent\n"
        + "- confidence: 0.0-1.0 how sure you are about direction and room\n"
        + "- room: room type you are currently in\n"
        + f"- relevance: 0.0-1.0, use ROOM COMMONSENSE — where is {goal} typically found?\n"
        + f"  ({_room_hint})\n"
        + f"  If you are in the WRONG room type for {goal}, relevance must be LOW (≤0.2).\n"
        + f"- search_direction: when {goal} NOT visible, reason step-by-step:\n"
        + f"  1. Which room type does {goal} belong in? ({_room_hint})\n"
        + f"  2. Which floor is it most likely on? ({_floor_hint})\n"
        + f"  3. Based on what you see (doorways, hallways, stairs), pick the direction most likely to lead there.\n"
        + f"  Output: left|center|right (if you see a promising door/hallway), upstairs (if stairs visible AND {goal} is more likely upstairs), none (if unsure)\n"
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
        result["_raw"] = text[js:je]  # pass raw JSON up for frontend display
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


# ── Episode-start strategy planning ──────────────────────────────────────────

_STRATEGY_FLOOR_HINTS = {
    "冰箱": "ground floor only — kitchens are never upstairs",
    "床":   "upper floor preferred — bedrooms are often upstairs",
    "沙发": "ground floor — living room is almost always on ground floor",
    "电视": "ground floor preferred, may be in upstairs bedroom",
    "桌子": "ground floor preferred (kitchen/dining)",
    "椅子": "ground floor preferred (kitchen/dining/living room)",
    "厕所": "ground floor preferred, some homes have upstairs bathroom",
    "水槽": "ground floor (kitchen/bathroom)",
}

_STRATEGY_ROOM_ORDER = {
    "冰箱": ["kitchen", "hallway", "living_room"],
    "床":   ["bedroom", "hallway", "staircase"],
    "沙发": ["living_room", "hallway", "bedroom"],
    "电视": ["living_room", "bedroom", "hallway"],
    "桌子": ["kitchen", "living_room", "hallway"],
    "椅子": ["kitchen", "living_room", "hallway"],
    "厕所": ["bathroom", "hallway", "bedroom"],
    "水槽": ["bathroom", "kitchen", "hallway"],
}


def _build_strategy_prompt(goal: str) -> str:
    """Text-only prompt for episode-start search strategy planning (no image)."""
    _floor = _STRATEGY_FLOOR_HINTS.get(goal, "use common sense about which floor")
    _example_rooms = _STRATEGY_ROOM_ORDER.get(goal, ["hallway", "living_room", "bedroom"])
    _example = json.dumps({"phase_rooms": _example_rooms, "floor": 0,
                           "reasoning": f"{goal} is typically found in {_example_rooms[0]}"})
    return (
        f"You are planning a home navigation search strategy to find '{goal}'.\n"
        f"No visual information yet — use commonsense knowledge only.\n"
        f"Floor guidance: {_floor}\n"
        f"Task: Output a search plan as ONE JSON line, no other text.\n"
        f'{{"phase_rooms":["room1","room2","room3"],"floor":int,"reasoning":"one sentence"}}\n'
        f"Rules:\n"
        f"- phase_rooms: ordered list of 2-4 room types to search, most likely FIRST\n"
        f"- room options: living_room|bedroom|kitchen|hallway|bathroom|staircase\n"
        f"- floor: 0=search ground floor first, 1=search upper floor first\n"
        f"- reasoning: one short sentence explaining the strategy\n"
        f"Example for '{goal}': {_example}"
    )
