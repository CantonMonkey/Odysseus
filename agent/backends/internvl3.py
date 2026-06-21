"""
backends/internvl3.py — InternVL3Backend: local model inference (class-level singleton).
"""
import os
import numpy as np
from agent.backends._shared import _build_perceive_prompt, _parse_percept_json


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
        c, r = idx % cols, idx // cols
        tile = resized.crop((c * IMAGE_SIZE, r * IMAGE_SIZE,
                              (c + 1) * IMAGE_SIZE, (r + 1) * IMAGE_SIZE))
        tiles.append(transform(tile))
    if len(tiles) != 1:
        tiles.append(transform(pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))))
    return torch.stack(tiles).to(torch.bfloat16).cuda()


class InternVL3Backend:
    """Local InternVL3 inference — class-level singleton (loaded once)."""

    _model     = None
    _tokenizer = None

    @classmethod
    def _load(cls):
        if cls._model is not None:
            return cls._model, cls._tokenizer
        path = os.environ.get("VLN_LOCAL_MODEL", "")
        if not path:
            return None, None
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            print(f"[InternVL3] Loading from {path} ...", flush=True)
            cls._tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True, use_fast=False)
            cls._model = AutoModel.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="cuda",
                trust_remote_code=True,
            ).eval()
            print("[InternVL3] Model ready.", flush=True)
        except Exception as e:
            print(f"[InternVL3] Load failed: {e}", flush=True)
        return cls._model, cls._tokenizer

    def perceive(self, frame, goal, annotated_frame=None, n_waypoints=0, context=None) -> dict:
        from agent.backends.rule_based import RuleBasedBackend
        from PIL import Image
        model, tokenizer = self._load()
        if model is None:
            return RuleBasedBackend().perceive(frame, goal)
        use_frame = annotated_frame if annotated_frame is not None else frame
        pil = Image.fromarray(use_frame.astype(np.uint8))
        pixel_values = _internvl_pixel_values(pil, max_num=4)
        prompt = "<image>\n" + _build_perceive_prompt(goal, n_waypoints, context)
        try:
            import torch
            text = model.chat(tokenizer, pixel_values, prompt,
                              dict(max_new_tokens=128, do_sample=False))
            return _parse_percept_json(text, goal)
        except Exception as e:
            import torch
            torch.cuda.empty_cache()
            print(f"[InternVL3] perceive error: {e}", flush=True)
            return RuleBasedBackend().perceive(frame, goal)

    def classify_scene(self, frame, goal: str) -> dict:
        """Room classification with floor hint and VLM navigation suggestion."""
        import json
        from PIL import Image
        model, tokenizer = self._load()
        if model is None:
            return {"room": "other", "objects": [], "floor_hint": "unknown", "suggest": "none"}
        pil = Image.fromarray(frame.astype(np.uint8))
        pixel_values = _internvl_pixel_values(pil, max_num=4)
        prompt = (
            f"<image>\nYou are a home navigation robot. Goal: find {goal}.\n"
            "Return ONE JSON line, no other text:\n"
            '{"room":"living_room|bedroom|hallway|kitchen|staircase|bathroom|other",'
            '"objects":["up to 3 visible items"],'
            '"floor_hint":"ground|upper|unknown",'
            '"suggest":"go_upstairs|search_room|keep_exploring|none"}\n'
            "Rules for suggest:\n"
            f"- go_upstairs: {goal} is typically in upstairs rooms (bed/wardrobe) "
            "AND you see stairs or are on ground floor\n"
            f"- search_room: current room likely contains {goal}, scan carefully\n"
            "- keep_exploring: move to new area\n"
            "- none: already navigating correctly"
        )
        try:
            text = model.chat(tokenizer, pixel_values, prompt,
                              dict(max_new_tokens=64, do_sample=False))
            text = (text or "").strip()
            if text and "{" in text:
                return json.loads(text[text.find("{"):text.rfind("}")+1])
        except Exception as e:
            print(f"[InternVL3] classify_scene error: {e}", flush=True)
        return {"room": "other", "objects": [], "floor_hint": "unknown", "suggest": "none"}

    def parse_goal(self, user_input: str) -> "str | None":
        model, tokenizer = self._load()
        if model is None:
            return None
        import torch
        prompt = (
            "Extract the navigation target object from the user instruction. "
            "Return ONLY the object name (one word or short phrase, Chinese or English). "
            f"User: '{user_input}'"
        )
        try:
            result = model.chat(tokenizer, None, prompt,
                                dict(max_new_tokens=16, do_sample=False))
            torch.cuda.empty_cache()
            goal = (result or "").strip().split("\n")[0].strip().rstrip("。，,.!")
            return goal or None
        except Exception:
            return None
