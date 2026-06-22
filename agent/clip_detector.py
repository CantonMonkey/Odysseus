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

    # Raw cosim rescaling constants calibrated from v5 eval data.
    # Observed ViT-B/32 cosim: ~0.22 background, ~0.25 target visible.
    # Tighter window pushes background below threshold and target above it:
    #   raw 0.22 (bg)      → rescaled 0.11  (< 0.40 visible threshold)
    #   raw 0.25 (visible) → rescaled 0.44  (> 0.40 threshold)
    #   raw 0.26 (clear)   → rescaled 0.56  (> 0.50 VALUE-STOP threshold)
    _COSIM_LO    = 0.21
    _COSIM_RANGE = 0.09

    @classmethod
    def detect(cls, frame_rgb: np.ndarray, goal_zh: str,
               threshold: float = 0.40) -> dict:
        """
        Args:
            frame_rgb : H×W×3 uint8
            goal_zh   : Chinese goal string
            threshold : rescaled-cosim threshold for visibility (0-1 scale)

        Returns dict:
            visible   : bool
            score     : float  (0-1, rescaled cosine similarity)
            bbox      : [x1,y1,x2,y2] pixel coords, or None
            direction : "left"|"center"|"right"|"not_visible"

        Uses VLFM-style raw cosine similarity between L2-normalized CLIP
        image and text embeddings with template "Seems like there is a X ahead."
        This is more discriminative than 2-class softmax, which fires on ANY
        furniture (0.69-0.86 even when target is absent).
        """
        model, processor, device = cls._load()
        goal_en = _ZH_EN.get(goal_zh, goal_zh)
        H, W    = frame_rgb.shape[:2]

        from PIL import Image
        img = Image.fromarray(frame_rgb.astype(np.uint8))

        # ── Step 1: full-image raw cosine similarity (VLFM-style) ─────────
        text  = f"Seems like there is a {goal_en} ahead."
        inp   = processor(text=[text], images=img, return_tensors="pt", padding=True)
        inp   = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out      = model(**inp)
            # image_embeds and text_embeds are L2-normalized in HuggingFace CLIP
            cosim    = float((out.image_embeds * out.text_embeds).sum())
        score = float(np.clip((cosim - cls._COSIM_LO) / cls._COSIM_RANGE, 0.0, 1.0))

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
