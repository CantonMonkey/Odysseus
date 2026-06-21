"""
llm_agent.py

LLM interface for visual perception + dialogue management.

Three backends (priority order):
  vLLM (OpenAI-compatible server):
    VLN_VLLM_BASE  – e.g. http://127.0.0.1:8088/v1
    VLN_VLLM_MODEL – model name served by vLLM (default: InternVL3-8B)
  LOCAL (InternVL3-8B, on-device, in-process):
    VLN_LOCAL_MODEL  – path to model weights dir
  API (Anthropic-compatible):
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, VLN_PERCEIVE_MODEL, VLN_DIALOGUE_MODEL

Falls back to rule-based defaults if none is configured.
"""

import os
import base64
import numpy as np
from typing import Optional

_LOCAL_MODEL_PATH = os.environ.get("VLN_LOCAL_MODEL", "")
_VLLM_BASE        = os.environ.get("VLN_VLLM_BASE", "")
_VLLM_MODEL       = os.environ.get(
    "VLN_VLLM_MODEL",
    os.path.basename(_LOCAL_MODEL_PATH) if _LOCAL_MODEL_PATH else "InternVL3-8B",
)
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
    if len(tiles) != 1:
        tiles.append(transform(pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))))
    return torch.stack(tiles).to(torch.bfloat16).cuda()


def _build_perceive_prompt(goal: str, n_waypoints: int = 0, context: dict = None) -> str:
    """Build the text portion of the VLM perception prompt (no <image> prefix)."""
    if n_waypoints >= 2:
        waypoint_rule = (
            f"- waypoint: 0-{n_waypoints}, choose the numbered circle most likely "
            f"to lead toward {goal}; 0 means none suitable\n"
        )
        waypoint_field = f',"waypoint":int'
    else:
        waypoint_rule = ""
        waypoint_field = ""

    if context:
        topo = context.get('topo_summary', '')
        topo_line = f"Map memory: {topo}\n" if topo and topo != "0 nodes" else ""
        history = context.get('history', [])
        hist_line = ""
        if history:
            hist_parts = [f"step{h['step']}:{h['skill']}({h.get('reason','')[:30]})" for h in history]
            hist_line = f"Recent decisions: {' → '.join(hist_parts)}\n"
        ctx_str = (
            f"Navigation state: step {context.get('step',0)}/{context.get('max_steps',500)}"
            f" | explored {context.get('explored_pct',0):.0%}"
            f" | stagnant {context.get('stagnant_steps',0)} steps\n"
            f"Rooms seen: {context.get('rooms_str','none yet')}\n"
            f"Nearest {goal}: {context.get('nearest_dist_str','unknown')}\n"
            + topo_line + hist_line
        )
        skill_field = ',"skill":"explore|snap|escape|verify","reason":"one sentence why"'
        skill_rules = (
            "Skill (choose one action for THIS step):\n"
            f"- \"snap\": {goal} is clearly visible, navigate to it NOW\n"
            "- \"explore\": keep searching, pick numbered waypoint (0=auto)\n"
            "- \"escape\": I am stuck/looping, need a completely different area\n"
            f"- \"verify\": I am very close to {goal}, confirm arrival\n"
            f"- reason: ONE short sentence explaining your choice "
            f"(e.g. \"{goal} visible on left side\", \"no {goal} found, continuing search\")\n"
        )
    else:
        ctx_str = ""
        skill_field = ""
        skill_rules = ""

    return (
        f"You are a home navigation robot brain. Navigation goal: {goal}\n"
        + ctx_str +
        "Observe the entire image carefully. Return ONE JSON line, no other text:\n"
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


# ── vLLM backend (OpenAI-compatible HTTP) ────────────────────────────────────

def _frame_to_jpeg_b64(frame: np.ndarray) -> str:
    """Encode RGB numpy array as JPEG base64 string."""
    import cv2
    _, buf = cv2.imencode(
        '.jpg',
        cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    return base64.b64encode(buf.tobytes()).decode()


def _parse_percept_json(text: str, goal: str) -> dict:
    """Parse VLM output JSON, with fallback to rule-based result."""
    import json
    text = (text or "").strip()
    if text:
        print(f"[VLM-RAW] {text[:400]}", flush=True)
    if not text or "{" not in text:
        return _perceive_rule(None, goal)
    try:
        _js = text.rfind("{"); _je = text.rfind("}") + 1
        result = json.loads(text[_js:_je])
        if not result.get("target_visible", False):
            result["confidence"] = 0.0
            result["direction"]  = "not_visible"
        return result
    except Exception:
        return _perceive_rule(None, goal)


def _perceive_vllm(frame: np.ndarray, goal: str,
                   annotated_frame: np.ndarray = None,
                   n_waypoints: int = 0,
                   context: dict = None) -> dict:
    """vLLM HTTP inference via OpenAI-compatible API."""
    import requests

    use_frame = annotated_frame if annotated_frame is not None else frame
    b64 = _frame_to_jpeg_b64(use_frame)
    prompt_text = _build_perceive_prompt(goal, n_waypoints, context)

    payload = {
        "model": _VLLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text",      "text": prompt_text},
        ]}],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(
            f"{_VLLM_BASE}/chat/completions",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return _parse_percept_json(text, goal)
    except Exception as e:
        print(f"[vLLM] perceive error: {e}", flush=True)
        return _perceive_rule(frame, goal)


def _perceive_local(frame: np.ndarray, goal: str,
                    annotated_frame: np.ndarray = None,
                    n_waypoints: int = 0,
                    context: dict = None) -> dict:
    """InternVL3 local inference for visual perception."""
    from PIL import Image

    model, tokenizer = _get_local_model()
    if model is None:
        return _perceive_rule(frame, goal)

    use_frame = annotated_frame if annotated_frame is not None else frame
    pil = Image.fromarray(use_frame.astype(np.uint8))
    pixel_values = _internvl_pixel_values(pil, max_num=4)

    prompt_text = _build_perceive_prompt(goal, n_waypoints, context)
    prompt = "<image>\n" + prompt_text

    try:
        import torch
        gen_cfg = dict(max_new_tokens=128, do_sample=False)
        text = model.chat(tokenizer, pixel_values, prompt, gen_cfg)
        return _parse_percept_json(text, goal)
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

    Priority: vLLM server → InternVL3 local → Anthropic API → rule-based.
    confidence forced to 0.0 when target_visible=False.
    """
    if _VLLM_BASE:
        return _perceive_vllm(frame, goal,
                              annotated_frame=annotated_frame,
                              n_waypoints=n_waypoints,
                              context=context)

    if _LOCAL_MODEL_PATH:
        return _perceive_local(frame, goal,
                               annotated_frame=annotated_frame,
                               n_waypoints=n_waypoints,
                               context=context)

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
            import time as _t; _t.sleep(0.3)

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
    if _VLLM_BASE or _LOCAL_MODEL_PATH:
        return {"room": "other", "objects": [], "floor_hint": "unknown", "suggest": "none"}

    client = _get_client()
    if client is None:
        return {"room": "其他", "objects": [], "floor_hint": "unknown", "suggest": "none"}

    import json
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
            import time as _t; _t.sleep(0.3)

    if not text or "{" not in text:
        return {"room": "其他", "objects": [], "floor_hint": "unknown", "suggest": "none"}
    try:
        return json.loads(text[text.find("{"):text.rfind("}")+1])
    except Exception:
        return {"room": "其他", "objects": [], "floor_hint": "unknown", "suggest": "none"}


def _perceive_rule(frame, goal: str) -> dict:
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
        """Extract a navigation goal keyword from a user utterance."""
        # Priority 1: vLLM server text-only (~20ms, no GPU memory overhead)
        if _VLLM_BASE:
            import requests
            prompt_text = (
                "Extract the navigation target object from the user instruction. "
                "Return ONLY the object name (one word or short phrase, Chinese or English). "
                f"User: '{user_input}'"
            )
            try:
                resp = requests.post(
                    f"{_VLLM_BASE}/chat/completions",
                    json={
                        "model": _VLLM_MODEL,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "max_tokens": 16,
                        "temperature": 0.0,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                _goal = resp.json()["choices"][0]["message"]["content"].strip()
                _goal = _goal.split('\n')[0].strip().rstrip('。，,.!')
                if _goal:
                    print(f'[PARSE_GOAL] vLLM: {user_input!r} → {_goal!r}', flush=True)
                    return _goal
            except Exception as e:
                print(f'[PARSE_GOAL] vLLM failed: {e}', flush=True)

        # Priority 2: InternVL3 local zero-shot (~50ms)
        model, tokenizer = _get_local_model()
        if model is not None:
            import torch
            prompt = (
                "Extract the navigation target object from the user instruction. "
                "Return ONLY the object name (one word or short phrase, Chinese or English). "
                f"User: '{user_input}'"
            )
            try:
                _r = model.chat(tokenizer, None, prompt,
                                dict(max_new_tokens=16, do_sample=False))
                torch.cuda.empty_cache()
                _goal = (_r or "").strip().split('\n')[0].strip().rstrip('。，,.!')
                if _goal:
                    print(f'[PARSE_GOAL] InternVL3: {user_input!r} → {_goal!r}', flush=True)
                    return _goal
            except Exception:
                pass

        # Priority 3: Anthropic API
        print(f'[PARSE_GOAL] local unavailable, trying API: {user_input!r}', flush=True)
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
