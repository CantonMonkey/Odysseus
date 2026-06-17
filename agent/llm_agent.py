"""
llm_agent.py

LLM interface for visual perception + dialogue management.

Two backends (local takes priority):
  LOCAL (InternVL3-8B, on-device):
    VLN_LOCAL_MODEL  – path to model weights dir
  API (Anthropic-compatible):
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, VLN_PERCEIVE_MODEL, VLN_DIALOGUE_MODEL

Falls back to rule-based defaults if neither is configured.
"""

import os
import base64
import numpy as np
from typing import Optional

_LOCAL_MODEL_PATH = os.environ.get("VLN_LOCAL_MODEL", "")
_MODEL_PERCEIVE   = os.environ.get("VLN_PERCEIVE_MODEL",  "claude-sonnet-4-6")
_MODEL_DIALOGUE   = os.environ.get("VLN_DIALOGUE_MODEL",  "claude-haiku-4-5-20251001")

# ── Local InternVL3 backend ───────────────────────────────────────────────────

_local_model     = None
_local_tokenizer = None

def _get_local_model():
    """Lazy-load InternVL3 in float16 onto CUDA (singleton)."""
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer
    if not _LOCAL_MODEL_PATH:
        return None, None
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        print(f"[InternVL3] Loading from {_LOCAL_MODEL_PATH} ...", flush=True)
        _local_tokenizer = AutoTokenizer.from_pretrained(
            _LOCAL_MODEL_PATH, trust_remote_code=True, use_fast=False)
        _local_model = AutoModel.from_pretrained(
            _LOCAL_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        print("[InternVL3] Model ready.", flush=True)
        return _local_model, _local_tokenizer
    except Exception as e:
        print(f"[InternVL3] Load failed: {e}", flush=True)
        return None, None


def _internvl_pixel_values(pil_image, max_num=6):
    """Dynamic tiling pre-processing for InternVL2/3."""
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGE_SIZE = 448
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    def _best_ratio(aspect, target_ratios):
        best, best_diff = (1, 1), float("inf")
        for r in target_ratios:
            diff = abs(aspect - r[0] / r[1])
            if diff < best_diff:
                best_diff, best = diff, r
        return best

    w, h = pil_image.size
    aspect = w / h
    target_ratios = sorted(
        {(i, j)
         for n in range(1, max_num + 1)
         for i in range(1, n + 1)
         for j in range(1, n + 1)
         if 1 <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    tr = _best_ratio(aspect, target_ratios)
    tw, th = IMAGE_SIZE * tr[0], IMAGE_SIZE * tr[1]
    resized = pil_image.resize((tw, th))

    tiles = []
    cols = tr[0]
    for idx in range(tr[0] * tr[1]):
        c = idx % cols
        r = idx // cols
        tile = resized.crop((c * IMAGE_SIZE, r * IMAGE_SIZE,
                              (c + 1) * IMAGE_SIZE, (r + 1) * IMAGE_SIZE))
        tiles.append(transform(tile))
    # thumbnail always appended last
    if len(tiles) != 1:
        tiles.append(transform(pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))))
    return torch.stack(tiles).to(torch.bfloat16).cuda()


def _perceive_local(frame: np.ndarray, goal: str,
                    annotated_frame: np.ndarray = None,
                    n_waypoints: int = 0,
                    context: dict = None) -> dict:
    """InternVL3 local inference for visual perception."""
    import json
    from PIL import Image

    model, tokenizer = _get_local_model()
    if model is None:
        return _perceive_rule(frame, goal)

    use_frame = annotated_frame if annotated_frame is not None else frame
    pil = Image.fromarray(use_frame.astype(np.uint8))
    pixel_values = _internvl_pixel_values(pil, max_num=4)

    if n_waypoints >= 2:
        waypoint_rule = (
            f"- waypoint: 0-{n_waypoints}, choose the numbered circle most likely "
            f"to lead toward {goal}; 0 means none suitable\n"
        )
        waypoint_field = f',"waypoint":int'
    else:
        waypoint_rule = ""
        waypoint_field = ""

    # Phase 4: build context block + skill decision field
    if context:
        ctx_str = (
            f"Navigation state: step {context.get('step',0)}/{context.get('max_steps',500)}"
            f" | explored {context.get('explored_pct',0):.0%}"
            f" | stagnant {context.get('stagnant_steps',0)} steps\n"
            f"Rooms seen: {context.get('rooms_str','none yet')}\n"
            f"Nearest {goal}: {context.get('nearest_dist_str','unknown')}\n"
        )
        skill_field = ',"skill":"explore|snap|escape|verify","reason":"str"'
        skill_rules = (
            "Skill (choose one action for THIS step):\n"
            f"- \"snap\": {goal} is clearly visible, navigate to it NOW\n"
            "- \"explore\": keep searching, pick numbered waypoint (0=auto)\n"
            "- \"escape\": I am stuck/looping, need a completely different area\n"
            f"- \"verify\": I am very close to {goal}, confirm arrival\n"
        )
    else:
        ctx_str = ""
        skill_field = ""
        skill_rules = ""

    # CoT: structured thinking block before JSON decision
    if context:
        _goal_room = {
            "衣柜": "bedroom", "床": "bedroom", "床头柜": "bedroom",
            "冰箱": "kitchen", "桌子": "dining room or kitchen",
            "电视": "living room", "沙发": "living room",
            "浴缸": "bathroom", "镜子": "bathroom or bedroom",
        }.get(goal, "context-dependent room")
        cot_block = (
            "Before deciding, think step by step (one line each):\n"
            "where_am_i: <describe current room and key objects visible>\n"
            f"goal_location: {goal} is typically found in {_goal_room!r}\n"
            "plan: <what to do this step — explore which direction, snap, escape, or verify>\n"
            "===\n"
        )
    else:
        cot_block = ""

    prompt = (
        "<image>\n"
        f"You are a home navigation robot brain. Navigation goal: {goal}\n"
        + ctx_str + cot_block +
        "After your reasoning above, output ONE JSON line, no other text:\n"
        '{"target_visible":bool,"direction":"left|center|right|not_visible",'
        '"confidence":float,"room":"living_room|bedroom|hallway|kitchen|staircase|bathroom|other",'
        f'"relevance":float{waypoint_field}{skill_field}}}\n'
        "Rules:\n"
        f"- target_visible=true: {goal} is visible ANYWHERE (background/doorway/corner counts)\n"
        "- confidence: if visible 0.1-1.0 (partial/far=0.3-0.6, clear=0.8+), else 0.0\n"
        "- room: room type you are currently in\n"
        f"- relevance: 0.0-1.0, how likely navigating this direction leads to {goal}\n"
        "  (living_room for sofa/chair=0.9, hallway=0.4, bedroom for sofa=0.1)\n"
        "- direction: where the target is (left/center/right), not_visible if absent\n"
        + waypoint_rule + skill_rules
    )

    try:
        import torch
        gen_cfg = dict(max_new_tokens=256, do_sample=False)
        text = model.chat(tokenizer, pixel_values, prompt, gen_cfg)
        torch.cuda.empty_cache()
        text = (text or "").strip()
        if text:
            print(f"[VLM-RAW] {text[:400]}", flush=True)
        if not text or "{" not in text:
            return _perceive_rule(frame, goal)
        _js = text.rfind("{"); _je = text.rfind("}") + 1
        result = json.loads(text[_js:_je])
        if not result.get("target_visible", False):
            result["confidence"] = 0.0
            result["direction"]  = "not_visible"
        return result
    except Exception as e:
        import torch
        torch.cuda.empty_cache()
        print(f"[InternVL3] perceive error: {e}", flush=True)
        return _perceive_rule(frame, goal)


# ── API (Anthropic-compatible) backend ───────────────────────────────────────

def _get_client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


def _extract_text(content_blocks) -> str:
    """Return the first text block, skipping ThinkingBlocks."""
    return next((b.text for b in content_blocks if hasattr(b, "text")), "")


def _frame_to_b64(frame: np.ndarray) -> str:
    """Encode an RGB uint8 numpy array as a JPEG base64 string."""
    import io
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ── PERCEIVE (public) ─────────────────────────────────────────────────────────

def perceive(frame: np.ndarray, goal: str,
             annotated_frame: np.ndarray = None,
             n_waypoints: int = 0,
             context: dict = None) -> dict:
    """Analyse the current RGB frame with a VLM.

    Uses InternVL3 locally if VLN_LOCAL_MODEL is set; otherwise Anthropic API;
    otherwise rule-based fallback.
    Returns {target_visible, direction, confidence, room, relevance}.
    confidence forced to 0.0 when target_visible=False.
    """
    if _LOCAL_MODEL_PATH:
        return _perceive_local(frame, goal, annotated_frame=annotated_frame, n_waypoints=n_waypoints, context=context)

    client = _get_client()
    if client is None:
        return _perceive_rule(frame, goal)

    import json, time
    img_b64 = _frame_to_b64(frame)
    prompt = (
        f"You are a home navigation robot. Navigation goal: {goal}\n"
        "Observe the entire image carefully (foreground/background/doorways/distance). Return ONE JSON line, no other text:\n"
        '{"target_visible":bool,"direction":"left|center|right|not_visible",'
        '"confidence":float,"room":"living_room|bedroom|hallway|kitchen|staircase|bathroom|other",'
        '"relevance":float}\n'
        "Rules:\n"
        f"- target_visible=true: goal object {goal} is visible ANYWHERE (background/doorway/corner counts)\n"
        "- confidence: if visible 0.1-1.0 (partial/far=0.3-0.6, clear=0.8+), else 0.0\n"
        "- room: room type you are currently in\n"
        f"- relevance: 0.0-1.0, how likely navigating this direction leads to {goal}\n"
        "  (living_room for sofa/chair→0.9, hallway→0.4, bedroom for sofa→0.1, kitchen for sofa→0.05)\n"
        "- direction: where the target is in the image, not_visible if unseen"
    )
    text = ""
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=_MODEL_PERCEIVE,
                max_tokens=1024,
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
        if not result.get("target_visible", False):
            result["confidence"] = 0.0
            result["direction"]  = "not_visible"
        return result
    except Exception:
        return _perceive_rule(frame, goal)


def classify_scene(frame, goal: str) -> dict:
    """Identify current room type (fires every ~20 steps). API-only."""
    if _LOCAL_MODEL_PATH:
        # Skip heavy API call when running local model; perception is enough
        return {"room": "other", "objects": [], "floor_hint": "unknown", "suggest": "none"}

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
            time.sleep(0.3)

    if not text or "{" not in text:
        return {"room": "其他", "objects": [], "floor_hint": "unknown", "suggest": "none"}
    try:
        return json.loads(text[text.find("{"):text.rfind("}")+1])
    except Exception:
        return {"room": "其他", "objects": [], "floor_hint": "unknown", "suggest": "none"}


def _perceive_rule(frame: np.ndarray, goal: str) -> dict:
    """Rule-based fallback: report target not visible, trust semantic-map nav."""
    return {"target_visible": False, "direction": "not_visible", "distance": 99.0,
            "confidence": 0.0, "room": "other", "relevance": 0.2}


# ── DIALOGUE ──────────────────────────────────────────────────────────────────

class DialogueAgent:
    """Manage user dialogue: parse Chinese goal instructions and compose replies."""

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
                    max_tokens=1024,
                    messages=[{"role": "user", "content": (
                        f"从以下用户指令中提取导航目标（中文名词，如：沙发、床、椅子、桌子、厕所等）。"
                        f"只返回目标词，不要其他内容。\n用户指令：{user_input}"
                    )}],
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
                    max_tokens=1024,
                    messages=[{"role": "user", "content":
                        "你是家居机器人，刚刚完成导航到达目标位置，用一句简短的中文询问用户还需要什么帮助。"}],
                )
                return _extract_text(resp.content).strip()
            except Exception:
                pass
        return "我已到达，还需要什么？"

    def reset(self):
        self._history.clear()
