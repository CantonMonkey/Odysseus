"""
backends/anthropic_api.py — AnthropicBackend: Anthropic-compatible API inference.
"""
import base64
import io
import os
from agent.backends._shared import _parse_percept_json

_MODEL_PERCEIVE = os.environ.get("VLN_PERCEIVE_MODEL", "claude-sonnet-4-6")
_MODEL_DIALOGUE = os.environ.get("VLN_DIALOGUE_MODEL", "claude-haiku-4-5-20251001")


def _get_client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        import anthropic
        kwargs = {"api_key": key}
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
        return anthropic.Anthropic(**kwargs)
    except Exception:
        return None


def _extract_text(content_blocks) -> str:
    return next((b.text for b in content_blocks if hasattr(b, "text")), "")


def _frame_to_b64(frame) -> str:
    import numpy as np
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class AnthropicBackend:
    """Anthropic (or Anthropic-compatible) API backend."""

    def perceive(self, frame, goal, annotated_frame=None, n_waypoints=0, context=None, clip_state=None) -> dict:
        from agent.backends.rule_based import RuleBasedBackend
        client = _get_client()
        if client is None:
            return RuleBasedBackend().perceive(frame, goal)
        use_frame = annotated_frame if annotated_frame is not None else frame
        img_b64 = _frame_to_b64(use_frame)
        prompt = (
            f"You are a home navigation robot. Navigation goal: {goal}\n"
            "Observe the entire image carefully. Return ONE JSON line, no other text:\n"
            '{"target_visible":bool,"direction":"left|center|right|not_visible",'
            '"confidence":float,"room":"living_room|bedroom|hallway|kitchen|staircase|bathroom|other",'
            '"relevance":float}\n'
            "Rules:\n"
            f"- target_visible=true: {goal} is visible ANYWHERE\n"
            "- confidence: if visible 0.1-1.0, else 0.0\n"
            "- room: room type you are currently in\n"
            f"- relevance: 0.0-1.0, how likely this direction leads to {goal}\n"
            "- direction: where the target is, not_visible if unseen"
        )
        text = ""
        for attempt in range(3):
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
            if attempt < 2:
                import time; time.sleep(0.3)
        return _parse_percept_json(text, goal)

    def parse_goal(self, user_input: str) -> "str | None":
        client = _get_client()
        if client is None:
            return None
        try:
            resp = client.messages.create(
                model=_MODEL_DIALOGUE,
                max_tokens=64,
                messages=[{"role": "user", "content": (
                    f"从以下用户指令中提取导航目标（中文名词，如：沙发、床、桌子等）。"
                    f"只返回目标词，不要其他内容。\n用户指令：{user_input}"
                )}],
            )
            goal = _extract_text(resp.content).strip()
            return goal or None
        except Exception:
            return None
