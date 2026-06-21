#!/usr/bin/env bash
# Odysseus VLN — AutoDL environment setup
# Run on autodl4090: bash /root/autodl-tmp/Odysseus/setup_autodl.sh
# Fresh install: habitat-sim 0.3.1 + PyTorch 2.8.0+cu128 (~10-15 min)

set -e
CONDA=/root/miniconda3/bin/conda
ENV_NAME=habitat
ENV_DIR=/root/miniconda3/envs/$ENV_NAME

echo "=== [1/5] Create conda env (Python 3.9) ==="
if [ -d "$ENV_DIR" ]; then
    echo "  env already exists at $ENV_DIR, skipping"
else
    $CONDA create -n $ENV_NAME python=3.9 -y
fi

PIP=$ENV_DIR/bin/pip
PYTHON=$ENV_DIR/bin/python

echo "=== [2/5] Install habitat-sim 0.3.1 (pre-built, aihabitat channel) ==="
# Check if already installed
$PYTHON -c "import habitat_sim; print('  habitat-sim already installed:', habitat_sim.__version__)" 2>/dev/null && \
    echo "  skipping" || \
$CONDA install -n $ENV_NAME \
    habitat-sim=0.3.1 withbullet headless \
    -c aihabitat -c conda-forge \
    -y

echo "=== [3/5] Install PyTorch 2.8.0 (CUDA 12.8 — driver 580.x compatible) ==="
$PYTHON -c "import torch; print('  torch already installed:', torch.__version__)" 2>/dev/null && \
    echo "  skipping" || \
$PIP install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== [4/5] Install Python dependencies ==="
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
$PYTHON -c "import vllm; print('  vllm already installed')" 2>/dev/null && \
    echo "  skipping" || \
$PIP install vllm==0.11.0

echo "=== [4c/5] Install habitat-lab ==="
$PYTHON -c "import habitat; print('  habitat-lab already installed')" 2>/dev/null && \
    echo "  skipping" || \
$PIP install habitat-lab==0.3.20231024 --no-deps 2>/dev/null || \
$PIP install "git+https://github.com/facebookresearch/habitat-lab.git@v0.3.1" --no-deps

echo "=== [5/5] Verify ==="
$PYTHON - <<'PYEOF'
import sys
ok = True
try:
    import habitat_sim
    print(f"  habitat-sim: {habitat_sim.__version__}  OK")
except Exception as e:
    print(f"  habitat-sim: FAIL — {e}"); ok = False

try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"  torch: {torch.__version__}, CUDA available: {cuda}  {'OK' if cuda else 'WARN (no GPU?)'}")
except Exception as e:
    print(f"  torch: FAIL — {e}"); ok = False

try:
    import transformers
    print(f"  transformers: {transformers.__version__}  OK")
except Exception as e:
    print(f"  transformers: FAIL — {e}"); ok = False

try:
    import numpy as np
    print(f"  numpy: {np.__version__}  OK")
except Exception as e:
    print(f"  numpy: FAIL — {e}"); ok = False

sys.exit(0 if ok else 1)
PYEOF

echo ""
echo "=== Setup complete ==="
echo "Activate:      source /root/miniconda3/bin/activate habitat"
echo "Download model: $PYTHON /root/autodl-tmp/Odysseus/download_model.py"
echo "Run Odysseus:   cd /root/autodl-tmp/Odysseus && $PYTHON run.py"
