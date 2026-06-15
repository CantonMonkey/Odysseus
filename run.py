"""
run.py — Odysseus 启动入口
用法：python run.py [--host HOST] [--port PORT] [--gpu GPU_ID]
"""

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Odysseus VLN Agent Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    import os
    os.environ["HABITAT_GPU_ID"] = str(args.gpu)

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )
