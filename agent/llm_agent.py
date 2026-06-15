"""
llm_agent.py
LLM 接口层：Claude Vision 感知 + Claude 对话管理。
API key 未到位时自动 fallback 到规则实现，主流程不变。
"""

import os
import base64
import numpy as np
from typing import Optional

# ── API 客户端（懒加载，key 缺失时不崩溃）──────────────────────

def _get_client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


# ── 感知（PERCEIVE）──────────────────────────────────────────────

def perceive(frame: np.ndarray, goal: str) -> dict:
    """
    用 Claude Vision 分析当前 RGB 帧，判断目标是否可见。
    返回 {"target_visible": bool, "direction": str, "distance": float}

    key 未到位时用规则 fallback（始终返回 target_visible=False，
    控制循环会全程依赖 semantic_map 导航）。
    """
    client = _get_client()
    if client is None:
        return _perceive_rule(frame, goal)

    try:
        img_b64 = _frame_to_b64(frame)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
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
        text = response.content[0].text.strip()
        # 提取 JSON 部分
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return _perceive_rule(frame, goal)


def _perceive_rule(frame: np.ndarray, goal: str) -> dict:
    """规则 fallback：无 LLM 时返回 not visible，依赖 semantic_map 坐标导航。"""
    return {"target_visible": False, "direction": "not_visible", "distance": 99.0}


def _frame_to_b64(frame: np.ndarray) -> str:
    """RGB uint8 numpy array → JPEG base64 string。"""
    import io
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ── 对话管理（DIALOGUE）─────────────────────────────────────────

class DialogueAgent:
    """
    管理与用户的对话：
    - 解析中文目标指令 → 目标词
    - 到达后回复"还需要什么"
    - key 未到位时用简单规则解析
    """

    # 已知中文指令关键词 → 目标词（供规则 fallback）
    _KEYWORD_MAP = {
        "沙发": "沙发", "sofa": "沙发", "couch": "沙发",
        "床": "床", "bed": "床",
        "椅子": "椅子", "chair": "椅子",
        "桌子": "桌子", "table": "桌子", "desk": "桌子",
        "厕所": "厕所", "toilet": "厕所", "卫生间": "厕所",
        "冰箱": "冰箱", "refrigerator": "冰箱",
        "镜子": "镜子", "mirror": "镜子",
        "电视": "电视", "tv": "电视",
    }

    def __init__(self):
        self._client = None
        self._history = []

    def parse_goal(self, user_input: str) -> Optional[str]:
        """从用户输入中提取导航目标词。先尝试 LLM，失败用关键词匹配。"""
        client = _get_client()
        if client is not None:
            try:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=64,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"从以下用户指令中提取导航目标（中文名词，如：沙发、床、椅子、桌子、厕所等）。"
                            f"只返回目标词，不要其他内容。\n用户指令：{user_input}"
                        ),
                    }],
                )
                goal = resp.content[0].text.strip()
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
        """到达目标后的询问语。"""
        client = _get_client()
        if client is not None:
            try:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=64,
                    messages=[{
                        "role": "user",
                        "content": "你是家居机器人，刚刚完成导航到达目标位置，用一句简短的中文询问用户还需要什么帮助。",
                    }],
                )
                return resp.content[0].text.strip()
            except Exception:
                pass
        return "我已到达，还需要什么？"

    def reset(self):
        self._history.clear()
