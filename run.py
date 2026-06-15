"""
run.py — Odysseus startup entry point
Usage: python run.py [--host HOST] [--port PORT] [--gpu GPU_ID]

Loads .env from the project root (if present) before starting.
Follow the mimo pattern for API configuration:
  ANTHROPIC_API_KEY=<key>
  ANTHROPIC_BASE_URL=https://api.xiaomimimo.com/anthropic
  VLN_PERCEIVE_MODEL=mimo-v2.5-pro
  VLN_DIALOGUE_MODEL=mimo-v2.5-pro
"""

import os
import argparse
from pathlib import Path


def _load_dotenv(path: Path):
    """Minimal .env loader — no dependency on python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv(Path(__file__).parent / ".env")

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Odysseus VLN Agent Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["HABITAT_GPU_ID"] = str(args.gpu)

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )
