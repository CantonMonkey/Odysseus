"""
llm_agent.py

LLM interface layer: Claude Vision perception + Claude dialogue management.
When ANTHROPIC_API_KEY is not set, both functions fall back to simple rules
so the rest of the pipeline keeps running without modification.

Environment variables (follow the mimo pattern in ~/.zshrc):
  ANTHROPIC_API_KEY      – API key (required for LLM features)
  ANTHROPIC_BASE_URL     – custom base URL, e.g. https://api.xiaomimimo.com/anthropic
  VLN_PERCEIVE_MODEL     – model used for visual perception (default: claude-sonnet-4-6)
  VLN_DIALOGUE_MODEL     – model used for goal parsing / replies (default: claude-haiku-4-5-20251001)
"""

import os
import base64
import numpy as np
from typing import Optional

# Model names: override via env to route through any Anthropic-compatible provider
_MODEL_PERCEIVE = os.environ.get("VLN_PERCEIVE_MODEL",  "claude-sonnet-4-6")
_MODEL_DIALOGUE = os.environ.get("VLN_DIALOGUE_MODEL",  "claude-haiku-4-5-20251001")


# ── API client (lazy-loaded; missing key does not crash the process) ──────────

def _get_client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        import anthropic
        # ANTHROPIC_BASE_URL is read automatically by the SDK if set in the environment
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


def _extract_text(content_blocks) -> str:
    """Return the first text from a list of content blocks, skipping ThinkingBlocks."""
    return next((b.text for b in content_blocks if hasattr(b, "text")), "")


# ── PERCEIVE ──────────────────────────────────────────────────────────────────

def perceive(frame: np.ndarray, goal: str) -> dict:
    """Analyse the current RGB frame with Claude Vision.

    Returns {"target_visible": bool, "direction": str, "distance": float}.

    Without an API key the rule fallback always returns target_visible=False,
    meaning the control loop relies entirely on the semantic-map coordinates
    for navigation.
    """
    client = _get_client()
    if client is None:
        return _perceive_rule(frame, goal)

    try:
        img_b64 = _frame_to_b64(frame)
        # Prompt is in Chinese to match the robot's conversational language
        response = client.messages.create(
            model=_MODEL_PERCEIVE,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"你是一个家居导航机器人的视觉感知模块。\n"
                            f"当前导航目标：{goal}\n"
                            f"请观察这张机器人第一视角图像，判断：\n"
                            f"1. 目标物体是否在视野内（target_visible: true/false）\n"
                            f"2. 目标大致方向（direction: left/center/right/not_visible）\n"
                            f"3. 估计距离（distance: 米，不可见时填 99）\n"
                            f"只返回 JSON，格式：{{\"target_visible\": bool, "
                            f"\"direction\": str, \"distance\": float}}"
                        ),
                    },
                ],
            }],
        )
        import json
        text  = _extract_text(response.content).strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return _perceive_rule(frame, goal)


def _perceive_rule(frame: np.ndarray, goal: str) -> dict:
    """Rule-based fallback: report target not visible, trust semantic-map nav."""
    return {"target_visible": False, "direction": "not_visible", "distance": 99.0}


def _frame_to_b64(frame: np.ndarray) -> str:
    """Encode an RGB uint8 numpy array as a JPEG base64 string."""
    import io
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ── DIALOGUE ──────────────────────────────────────────────────────────────────

class DialogueAgent:
    """Manage user dialogue: parse Chinese goal instructions and compose replies.

    With an API key: uses Claude Haiku for intent extraction and reply generation.
    Without a key:   falls back to keyword matching for goal parsing.
    """

    # Keyword → canonical goal word (used by the rule-based fallback)
    _KEYWORD_MAP = {
        "沙发": "沙发", "sofa": "沙发", "couch": "沙发",
        "床":   "床",   "bed":  "床",
        "椅子": "椅子", "chair": "椅子",
        "桌子": "桌子", "table": "桌子", "desk": "桌子",
        "厕所": "厕所", "toilet": "厕所", "卫生间": "厕所",
        "冰箱": "冰箱", "refrigerator": "冰箱",
        "镜子": "镜子", "mirror": "镜子",
        "电视": "电视", "tv": "电视",
        "衣柜": "衣柜", "wardrobe": "衣柜", "柜子": "柜子",
        "书架": "书架", "bookshelf": "书架",
        "床头柜": "床头柜", "nightstand": "床头柜",
        "台灯": "台灯", "lamp": "台灯",
        "浴缸": "浴缸", "bathtub": "浴缸",
    }

    def __init__(self):
        self._history = []

    def parse_goal(self, user_input: str) -> Optional[str]:
        """Extract a navigation goal keyword from a Chinese user utterance."""
        client = _get_client()
        if client is not None:
            try:
                resp = client.messages.create(
                    model=_MODEL_DIALOGUE,
                    max_tokens=128,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"从以下用户指令中提取导航目标（中文名词，如：沙发、床、椅子、桌子、厕所等）。"
                            f"只返回目标词，不要其他内容。\n用户指令：{user_input}"
                        ),
                    }],
                )
                goal = _extract_text(resp.content).strip()
                if goal:
                    return goal
            except Exception:
                pass
        return self._rule_parse(user_input)

    def _rule_parse(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for kw, goal in self._KEYWORD_MAP.items():
            if kw in text_lower:
                return goal
        return None

    def arrival_message(self) -> str:
        """Generate a short Chinese reply after the robot reaches the goal."""
        client = _get_client()
        if client is not None:
            try:
                resp = client.messages.create(
                    model=_MODEL_DIALOGUE,
                    max_tokens=128,
                    messages=[{
                        "role": "user",
                        "content": "你是家居机器人，刚刚完成导航到达目标位置，用一句简短的中文询问用户还需要什么帮助。",
                    }],
                )
                return _extract_text(resp.content).strip()
            except Exception:
                pass
        return "我已到达，还需要什么？"

    def reset(self):
        self._history.clear()
