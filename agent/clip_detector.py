"""CLIP-based object detector: replaces VLM target_visible + bbox.

Runs on CPU (~20ms/call). Provides reliable visibility signal independent
of VLM JSON parsing failures.
"""
import numpy as np
import torch

# Chinese goal → English CLIP query
_ZH_EN = {
    "沙发": "sofa couch",
    "床":   "bed",
    "椅子": "chair",
    "冰箱": "refrigerator fridge",
    "桌子": "table desk",
    "电视": "television screen",
    "厕所": "toilet",
    "水槽": "sink basin",
}


class CLIPDetector:
    _model     = None
    _processor = None
    _device    = "cpu"

    @classmethod
    def _load(cls):
        if cls._model is None:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            # Prefer GPU if available (runs alongside vLLM in a separate process)
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            # Use ModelScope mirror (fast in China); falls back to HuggingFace
            _model_id = "openai/clip-vit-base-patch32"
            print(f"[CLIP] loading {_model_id} on {cls._device} ...", flush=True)
            import os
            # Try cached first; if missing, download via hf-mirror
            try:
                cls._model     = CLIPModel.from_pretrained(
                    _model_id, local_files_only=True).to(cls._device).eval()
                cls._processor = CLIPProcessor.from_pretrained(
                    _model_id, local_files_only=True)
            except Exception:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                cls._model     = CLIPModel.from_pretrained(_model_id).to(cls._device).eval()
                cls._processor = CLIPProcessor.from_pretrained(_model_id)
            print(f"[CLIP] ready ({cls._device})", flush=True)
        return cls._model, cls._processor, cls._device

    @classmethod
    def detect(cls, frame_rgb: np.ndarray, goal_zh: str,
               threshold: float = 0.55) -> dict:
        """
        Args:
            frame_rgb : H×W×3 uint8
            goal_zh   : Chinese goal string
            threshold : 2-class softmax probability threshold for visibility

        Returns dict:
            visible   : bool
            score     : float  (0-1)
            bbox      : [x1,y1,x2,y2] pixel coords, or None
            direction : "left"|"center"|"right"|"not_visible"
        """
        model, processor, device = cls._load()
        goal_en = _ZH_EN.get(goal_zh, goal_zh)
        H, W    = frame_rgb.shape[:2]

        from PIL import Image
        img = Image.fromarray(frame_rgb.astype(np.uint8))

        # ── Step 1: full-image binary classification ──────────────────────
        texts = [f"a {goal_en} in a room", "an empty room with no furniture"]
        inp   = processor(text=texts, images=img, return_tensors="pt", padding=True)
        inp   = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            logits = model(**inp).logits_per_image[0]
            score  = float(logits.softmax(dim=0)[0])

        if score < threshold:
            return {"visible": False, "score": score,
                    "bbox": None, "direction": "not_visible"}

        # ── Step 2: 4×3 patch grid for bbox localisation ─────────────────
        COLS, ROWS = 4, 3
        pw, ph     = W // COLS, H // ROWS
        patches, coords = [], []
        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = c * pw, r * ph
                x2, y2 = min((c + 1) * pw, W), min((r + 1) * ph, H)
                patches.append(img.crop((x1, y1, x2, y2)))
                coords.append((x1, y1, x2, y2))

        pinp = processor(
            text=[f"a {goal_en}"] * len(patches),
            images=patches, return_tensors="pt", padding=True,
        )
        pinp = {k: v.to(device) for k, v in pinp.items()}
        with torch.no_grad():
            pscores = model(**pinp).logits_per_image[:, 0]

        best         = int(pscores.argmax())
        x1, y1, x2, y2 = coords[best]
        cx           = (x1 + x2) / 2.0
        direction    = ("left"   if cx < W / 3 else
                        "right"  if cx > 2 * W / 3 else
                        "center")

        return {
            "visible":   True,
            "score":     score,
            "bbox":      [x1, y1, x2, y2],
            "direction": direction,
        }
