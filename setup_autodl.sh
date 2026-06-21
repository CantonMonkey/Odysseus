#!/usr/bin/env bash
# Odysseus VLN — AutoDL environment setup
# Run on autodl4090: bash /root/autodl-tmp/Odysseus/setup_autodl.sh
# Tested: RTX 4090, CUDA 12.8, Python 3.9, autodl CN servers (~15 min total)

set -e
CONDA=/root/miniconda3/bin/conda
MAMBA=/root/miniconda3/bin/mamba
ENV_NAME=habitat
ENV_DIR=/root/miniconda3/envs/$ENV_NAME

echo "=== [0/5] Bootstrap mamba (fast solver) ==="
if [ ! -f "$MAMBA" ]; then
    $CONDA install mamba -n base -c conda-forge -y
fi

echo "=== [1/5] Create conda env (Python 3.9) ==="
if [ -d "$ENV_DIR" ]; then
    echo "  already exists, skipping"
else
    $CONDA create -n $ENV_NAME python=3.9 -y
fi

PIP=$ENV_DIR/bin/pip
PYTHON=$ENV_DIR/bin/python

echo "=== [2/5] Install habitat-sim 0.3.1 (aihabitat + conda-forge via mamba) ==="
if $PYTHON -c "import habitat_sim" 2>/dev/null; then
    echo "  already installed: $($PYTHON -c 'import habitat_sim; print(habitat_sim.__version__)')"
else
    # --override-channels: skip Tsinghua pkgs/main (has 404s for older py39 builds)
    # aihabitat channel: pre-built headless binary (no Bullet physics — not needed for ObjectNav)
    # conda-forge: properly mirrored by Tsinghua, provides all system deps
    $MAMBA install -n $ENV_NAME \
        habitat-sim=0.3.1 headless \
        -c aihabitat -c conda-forge \
        --override-channels \
        -y
fi

echo "=== [3/5] Install PyTorch 2.8.0 (CUDA 12.8 — driver 580.x OK) ==="
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  already installed: $($PYTHON -c 'import torch; print(torch.__version__)')"
else
    # Install torch CUDA BEFORE other packages to avoid pip downloading CPU-only torch
    # as dependency of accelerate, transformers, etc.
    $PIP install \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128
fi

echo "=== [4/5] Install Python dependencies (aliyun PyPI mirror) ==="
$PIP install \
    "numpy==1.23.5" \
    "numpy-quaternion==2023.0.4" \
    "scipy==1.13.1" \
    "opencv-python-headless" \
    "Pillow" \
    "transformers==4.57.6" \
    "tokenizers==0.22.2" \
    "sentencepiece==0.2.1" \
    "huggingface_hub==0.36.2" \
    "accelerate==1.10.1" \
    "einops==0.8.2" \
    "timm==1.0.27" \
    "attrs==25.3.0" \
    "requests==2.32.5" \
    "anthropic==0.109.1" \
    "openai==2.43.0" \
    "pydantic==2.13.4" \
    "pydantic-settings==2.11.0" \
    "flask" \
    "flask-cors" \
    "tqdm" \
    "pyquaternion" \
    "modelscope"

echo "=== [4b/5] Install vLLM 0.11.0 ==="
if $PYTHON -c "import vllm" 2>/dev/null; then
    echo "  already installed"
else
    $PIP install vllm==0.11.0
fi

echo "=== [4c/5] Install habitat-lab ==="
if $PYTHON -c "import habitat" 2>/dev/null; then
    echo "  already installed"
else
    $PIP install habitat-lab==0.3.20231024 --no-deps 2>/dev/null || \
    $PIP install "git+https://github.com/facebookresearch/habitat-lab.git@v0.3.1" --no-deps
fi

echo "=== [5/5] Verify ==="
$PYTHON - <<'PYEOF'
import sys
ok = True
for mod, extra in [
    ("habitat_sim", None),
    ("torch", "torch.cuda.is_available()"),
    ("transformers", None),
    ("numpy", None),
]:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        info = f", CUDA={eval(extra)}" if extra else ""
        print(f"  {mod}: {ver}{info}  OK")
    except Exception as e:
        print(f"  {mod}: FAIL — {e}"); ok = False
sys.exit(0 if ok else 1)
PYEOF

echo ""
echo "=== Setup complete! ==="
echo "Activate:        source /root/miniconda3/bin/activate habitat"
echo "Download model:  $PYTHON /root/autodl-tmp/Odysseus/download_model.py"
echo "Set env vars:    cp /root/autodl-tmp/Odysseus/.env.autodl /root/autodl-tmp/Odysseus/.env"
echo "Run Odysseus:    cd /root/autodl-tmp/Odysseus && $PYTHON run.py"
