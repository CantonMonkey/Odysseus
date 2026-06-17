"""
server/main.py

FastAPI backend.

All Habitat GL operations run inside a single dedicated habitat_thread because
EGL contexts are thread-local; accessing _sim from any other thread causes a
"no current context" crash.  The async event loop and habitat_thread
communicate through two queues:
  _cmd_q   – main → habitat_thread (goal string or None to stop)
  _frame_q – habitat_thread → main (frame bytes + state dict)
"""

import asyncio
import base64
import io
import json
import queue
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

WEB_DIR    = Path(__file__).parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="Odysseus VLN")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Shared state ───────────────────────────────────────────────────────────────

_cmd_q: queue.Queue   = queue.Queue()   # commands to the Habitat worker
_frame_q: queue.Queue = queue.Queue()   # frames / status events from the worker
_nav_active           = threading.Event()

_ws_clients: list[WebSocket] = []
_event_loop: Optional[asyncio.AbstractEventLoop] = None


# ── Habitat worker thread ──────────────────────────────────────────────────────

def _habitat_worker():
    """Own the EGL context; process navigation commands; push frames to _frame_q."""
    from agent.habitat_env import HabitatEnv
    from agent.loop import run_task, SCENE_DIR
    from agent.llm_agent import DialogueAgent, perceive as llm_perceive

    env      = HabitatEnv(gpu_id=0)
    dialogue = DialogueAgent()

    FIXED_SPAWN = [-8.24, 0.163, -1.47]  # living room, good demo start
    env.reset(SCENE_DIR, start_pos=FIXED_SPAWN)
    _frame_q.put(("frame", env.get_frame(), {"status": "idle"}))  # initial frame

    while True:
        cmd = _cmd_q.get()
        if cmd is None:  # shutdown signal
            env.close()
            break

        goal = cmd  # str

        try:
            env.reset(SCENE_DIR, start_pos=FIXED_SPAWN)

            def on_frame(frame, nav_state):
                state = {
                    "status": "navigating",
                    "goal":   nav_state.get("goal", goal),
                    "step":   nav_state.get("step_count", 0),
                    "skill":  nav_state.get("current_skill", ""),
                }
                _frame_q.put(("frame", frame, state))

            result = run_task(env, goal, scene_dir=SCENE_DIR, on_frame=on_frame, llm_perceive=llm_perceive)

            if result.get("done"):
                msg   = dialogue.arrival_message()
                final = {"status": "arrived", "goal": goal, "message": msg}
            else:
                final = {"status": "timeout", "goal": goal, "message": "Navigation timed out."}

            _frame_q.put(("status", None, final))
        except Exception as e:
            _frame_q.put(("status", None, {"status": "error", "message": str(e)}))
        finally:
            _nav_active.clear()


# ── Frame broadcaster (runs in the async event loop) ──────────────────────────

async def _frame_broadcaster():
    """Drain _frame_q and forward each item to all connected WebSocket clients."""
    while True:
        try:
            item = _frame_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue

        kind, frame, state = item

        if kind == "frame":
            jpeg    = _frame_to_jpeg(frame)
            payload = {
                "type": "frame",
                "img":  base64.b64encode(jpeg).decode(),
                **state,
            }
        else:
            payload = {"type": "status", **state}

        dead = []
        for ws in list(_ws_clients):
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in _ws_clients:
                _ws_clients.remove(ws)


# ── Lifeycle ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _event_loop
    _event_loop = asyncio.get_event_loop()

    threading.Thread(target=_habitat_worker, daemon=True).start()
    asyncio.create_task(_frame_broadcaster())
    print("[Server] Ready.")


@app.on_event("shutdown")
async def shutdown():
    _cmd_q.put(None)  # ask the habitat worker to exit cleanly


# ── HTTP endpoints ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


class CommandRequest(BaseModel):
    text: str


@app.post("/command")
async def command(req: CommandRequest):
    from agent.llm_agent import DialogueAgent
    text = req.text.strip()
    if not text:
        return {"ok": False, "error": "empty input"}

    if _nav_active.is_set():
        return {"ok": False, "error": "Navigation already in progress."}

    goal = DialogueAgent().parse_goal(text)
    if goal is None:
        return {"ok": False, "error": f"Cannot parse goal from: {text}"}

    _nav_active.set()
    _cmd_q.put(goal)
    return {"ok": True, "goal": goal}


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping"}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ── Utilities ──────────────────────────────────────────────────────────────────

def _frame_to_jpeg(frame: np.ndarray) -> bytes:
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()
