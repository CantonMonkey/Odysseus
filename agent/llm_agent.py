"""
llm_agent.py

LLM interface for visual perception + dialogue management.

Backends are now implemented in agent/backends/:
  VLLMBackend      — agent/backends/vllm_http.py
  InternVL3Backend — agent/backends/internvl3.py
  AnthropicBackend — agent/backends/anthropic_api.py
  RuleBasedBackend — agent/backends/rule_based.py

This module keeps all public exports for backward compatibility:
  perceive(), classify_scene(), DialogueAgent
"""

import os
from typing import Optional

from agent.backends.vllm_http     import VLLMBackend
from agent.backends.internvl3     import InternVL3Backend
from agent.backends.anthropic_api import AnthropicBackend, _get_client, _extract_text
from agent.backends.rule_based    import RuleBasedBackend

_VLLM_BASE        = os.environ.get("VLN_VLLM_BASE", "")
_LOCAL_MODEL_PATH = os.environ.get("VLN_LOCAL_MODEL", "")
_MODEL_DIALOGUE   = os.environ.get("VLN_DIALOGUE_MODEL", "claude-haiku-4-5-20251001")

# Singletons to avoid re-instantiating on every call
_vllm_backend  = VLLMBackend()      if _VLLM_BASE        else None
_local_backend = InternVL3Backend() if _LOCAL_MODEL_PATH else None
_api_backend   = AnthropicBackend()
_rule_backend  = RuleBasedBackend()


def perceive(frame, goal: str,
             annotated_frame=None,
             n_waypoints: int = 0,
             context: dict = None,
             clip_state: dict = None) -> dict:
    """Analyse the current RGB frame with a VLM.

    Priority: vLLM server -> InternVL3 local -> Anthropic API -> rule-based.
    """
    if _vllm_backend is not None:
        return _vllm_backend.perceive(frame, goal, annotated_frame, n_waypoints, context, clip_state)
    if _local_backend is not None:
        return _local_backend.perceive(frame, goal, annotated_frame, n_waypoints, context, clip_state)
    client = _get_client()
    if client is not None:
        return _api_backend.perceive(frame, goal, annotated_frame, n_waypoints, context, clip_state)
    return _rule_backend.perceive(frame, goal)


def classify_scene(frame, goal: str) -> dict:
    """Identify current room type and navigation suggestion via VLM."""
    if _vllm_backend is not None:
        # vLLM: use perceive prompt (classify_scene not separately implemented for vLLM)
        return {"room": "other", "objects": [], "floor_hint": "unknown", "suggest": "none"}
    if _local_backend is not None:
        return _local_backend.classify_scene(frame, goal)

    client = _get_client()
    if client is None:
        return {"room": "other", "objects": [], "floor_hint": "unknown", "suggest": "none"}

    import json
    from agent.backends.anthropic_api import _frame_to_b64, _MODEL_PERCEIVE
    img_b64 = _frame_to_b64(frame)
    prompt = (
        f"You are a home navigation robot. Navigation goal: {goal}\n"
        "Observe the image and return ONE JSON line, no other text:\n"
        '{"room":"living_room|bedroom|hallway|kitchen|staircase|bathroom|other",'
        '"objects":["visible furniture/objects, up to 5"],'
        '"floor_hint":"ground|upper|unknown",'
        '"suggest":"go_upstairs|search_room|keep_exploring|none"}\n'
        "suggest rules:\n"
        f"- if {goal} is usually on another floor (e.g. bed upstairs) and stairs visible -> go_upstairs\n"
        f"- if current room may contain {goal} but not fully scanned -> search_room\n"
        "- otherwise -> keep_exploring"
    )
    text = ""
    for attempt in range(2):
        try:
            resp = client.messages.create(
                model=_MODEL_PERCEIVE,
                max_tokens=1024,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": prompt},
                ]}],
            )
            text = _extract_text(resp.content).strip()
            if text and "{" in text:
                break
        except Exception:
            pass
        if attempt < 1:
            import time as _t; _t.sleep(0.3)

    if not text or "{" not in text:
        return {"room": "other", "objects": [], "floor_hint": "unknown", "suggest": "none"}
    try:
        return json.loads(text[text.find("{"):text.rfind("}")+1])
    except Exception:
        return {"room": "other", "objects": [], "floor_hint": "unknown", "suggest": "none"}


def _perceive_rule(frame, goal: str) -> dict:
    """Rule-based fallback -- kept for any direct callers."""
    return _rule_backend.perceive(frame, goal)


# -- DIALOGUE ------------------------------------------------------------------

class DialogueAgent:
    """Manage user dialogue: parse goal instructions and compose replies."""

    def __init__(self):
        self._history = []

    def parse_goal(self, user_input: str) -> Optional[str]:
        """Extract a navigation goal keyword from a user utterance."""
        if _vllm_backend is not None:
            result = _vllm_backend.parse_goal(user_input)
            if result:
                print(f'[PARSE_GOAL] vLLM: {user_input!r} -> {result!r}', flush=True)
                return result

        if _local_backend is not None:
            result = _local_backend.parse_goal(user_input)
            if result:
                print(f'[PARSE_GOAL] InternVL3: {user_input!r} -> {result!r}', flush=True)
                return result

        print(f'[PARSE_GOAL] local unavailable, trying API: {user_input!r}', flush=True)
        result = _api_backend.parse_goal(user_input)
        if result:
            return result

        return _rule_backend.parse_goal(user_input)

    def arrival_message(self) -> str:
        """Generate a short reply after the robot reaches the goal."""
        client = _get_client()
        if client is not None:
            try:
                resp = client.messages.create(
                    model=_MODEL_DIALOGUE,
                    max_tokens=1024,
                    messages=[{"role": "user", "content":
                        "You are a home robot. You just reached the navigation goal. "
                        "Reply in one short sentence asking if the user needs anything else."}],
                )
                return _extract_text(resp.content).strip()
            except Exception:
                pass
        return "Arrived! Anything else I can help with?"

    def reset(self):
        self._history.clear()
