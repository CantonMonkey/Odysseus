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
    """Analyse the current RGB frame with a VLM.

    Returns {target_visible, direction, distance, confidence}.
    confidence is meaningful ONLY when target_visible=True; forced to 0.0 otherwise.
    Retries up to 3 times when the model returns an empty text block.
    """
    client = _get_client()
    if client is None:
        return _perceive_rule(frame, goal)

    import json, time
    img_b64 = _frame_to_b64(frame)
    prompt = (
        "你是家居导航机器人的视觉感知模块。\n"
        f"导航目标：{goal}\n"
        "观察第一视角图像，只返回一行JSON，禁止其他文字:\n"
        '{"target_visible":bool,"direction":"left|center|right|not_visible",'
        '"distance":float,"confidence":float}\n'
        "规则:\n"
        "- target_visible=true: 目标在画面中可见\n"
        "- confidence: 仅target_visible=true时填识别把握(0.0-1.0)，否则必须填0.0\n"
        "- distance: 目标估计距离(米)，不可见填99\n"
        "- direction: 不可见填not_visible"
    )

    text = ""
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=_MODEL_PERCEIVE,
                max_tokens=128,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": prompt},
                ]}],
            )
            text = _extract_text(response.content).strip()
            if text and "{" in text:
                break
        except Exception:
            pass
        if attempt < 2:
            time.sleep(0.3)

    if not text or "{" not in text:
        return _perceive_rule(frame, goal)

    try:
        result = json.loads(text[text.find("{"):text.rfind("}")+1])
        # Enforce: confidence must be 0.0 when target is not visible
        if not result.get("target_visible", False):
            result["confidence"] = 0.0
            result["direction"]  = "not_visible"
        return result
    except Exception:
        return _perceive_rule(frame, goal)



def classify_scene(frame, goal: str) -> dict:
    """
    Identify current room type and visible objects (fires every ~20 steps).

    Returns:
        {"room": str, "objects": [str], "floor_hint": str, "suggest": str}
        suggest: "go_upstairs" | "search_room" | "keep_exploring" | "none"

    Used to annotate topo_map nodes and bias frontier selection.
    Does NOT return target_visible — that is perceive()'s job.
    """
    client = _get_client()
    if client is None:
        return {"room": "其他", "objects": [], "floor_hint": "unknown", "suggest": "none"}

    import json, time
    img_b64 = _frame_to_b64(frame)
    prompt = (
        f"你是家居导航机器人。导航目标：{goal}\n"
        "观察当前第一视角图像，只返回一行JSON，禁止其他文字:\n"
        '{"room":"客厅|卧室|走廊|厨房|楼梯间|浴室|其他",'
        '"objects":["列出画面中可见的家具/物品，最多5个"],'
        '"floor_hint":"ground|upper|unknown",'
        '"suggest":"go_upstairs|search_room|keep_exploring|none"}\n'
        "suggest规则:\n"
        f"- 若{goal}通常在其他楼层（如床在二楼卧室）且画面中有楼梯 → go_upstairs\n"
        f"- 若当前房间可能有{goal}但未完全扫描 → search_room\n"
        "- 其他情况 → keep_exploring"
    )

    text = ""
    for attempt in range(2):
        try:
            resp = client.messages.create(
                model=_MODEL_PERCEIVE,
                max_tokens=128,
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
            time.sleep(0.3)

    if not text or "{" not in text:
        return {"room": "其他", "objects": [], "floor_hint": "unknown", "suggest": "none"}
    try:
        return json.loads(text[text.find("{"):text.rfind("}")+1])
    except Exception:
        return {"room": "其他", "objects": [], "floor_hint": "unknown", "suggest": "none"}


def _perceive_rule(frame: np.ndarray, goal: str) -> dict:
    """Rule-based fallback: report target not visible, trust semantic-map nav."""
    return {"target_visible": False, "direction": "not_visible", "distance": 99.0, "confidence": 0.0}


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
