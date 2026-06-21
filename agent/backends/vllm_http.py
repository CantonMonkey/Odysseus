"""
backends/vllm_http.py — VLLMBackend: OpenAI-compatible HTTP inference.
"""
import os
import requests
from agent.backends._shared import (
    _frame_to_jpeg_b64, _build_perceive_prompt, _parse_percept_json,
)


class VLLMBackend:
    """Connect to a running vLLM server (or any OpenAI-compatible endpoint)."""

    def __init__(self, base_url: str = "", model: str = ""):
        _local = os.environ.get("VLN_LOCAL_MODEL", "")
        self.base_url = base_url or os.environ.get("VLN_VLLM_BASE", "")
        self.model    = model or os.environ.get(
            "VLN_VLLM_MODEL",
            os.path.basename(_local) if _local else "InternVL3-8B",
        )

    def perceive(self, frame, goal, annotated_frame=None, n_waypoints=0, context=None) -> dict:
        from agent.backends.rule_based import RuleBasedBackend
        use_frame = annotated_frame if annotated_frame is not None else frame
        b64 = _frame_to_jpeg_b64(use_frame)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": _build_perceive_prompt(goal, n_waypoints, context)},
            ]}],
            "max_tokens": 128,
            "temperature": 0.0,
        }
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=30)
            resp.raise_for_status()
            return _parse_percept_json(resp.json()["choices"][0]["message"]["content"], goal)
        except Exception as e:
            print(f"[vLLM] perceive error: {e}", flush=True)
            return RuleBasedBackend().perceive(frame, goal)

    def parse_goal(self, user_input: str) -> "str | None":
        prompt = (
            "Extract the navigation target object from the user instruction. "
            "Return ONLY the object name (one word or short phrase, Chinese or English). "
            f"User: '{user_input}'"
        )
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16,
                "temperature": 0.0,
            }, timeout=10)
            resp.raise_for_status()
            goal = resp.json()["choices"][0]["message"]["content"].strip()
            goal = goal.split("\n")[0].strip().rstrip("。，,.!")
            return goal or None
        except Exception as e:
            print(f"[vLLM] parse_goal error: {e}", flush=True)
            return None
