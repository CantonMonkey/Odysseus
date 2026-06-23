"""
backends/rule_based.py — Rule-based fallback backend (no model needed).
"""
from agent.backends._shared import _rule_fallback

_KEYWORD_MAP = {
    "沙发": "沙发", "sofa": "沙发", "couch": "沙发",
    "床":   "床",   "bed":  "床",
    "椅子": "椅子", "chair": "椅子",
    "桌子": "桌子", "table": "桌子", "desk": "桌子",
    "厕所": "厕所", "toilet": "厕所", "卫生间": "厕所",
    "冰箱": "冰箱", "refrigerator": "冰箱",
    "镜子": "镜子", "mirror": "镜子",
    "电视": "电视", "tv": "电视",
    "衣柜": "衣柜", "wardrobe": "衣柜",
    "书架": "书架", "bookshelf": "书架",
    "床头柜": "床头柜", "nightstand": "床头柜",
    "台灯": "台灯", "lamp": "台灯",
    "浴缸": "浴缸", "bathtub": "浴缸",
}


class RuleBasedBackend:
    """Always returns target_not_visible; parse_goal uses keyword matching."""

    def perceive(self, frame, goal, annotated_frame=None, n_waypoints=0, context=None, clip_state=None) -> dict:
        return _rule_fallback(goal)

    def parse_goal(self, user_input: str) -> "str | None":
        text_lower = user_input.lower()
        for kw, goal in _KEYWORD_MAP.items():
            if kw in text_lower:
                return goal
        return None
