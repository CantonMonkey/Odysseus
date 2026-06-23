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
from PIL import Image
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
    _frame_q.put(("frame", env.get_frame(), {"status": "idle"}))
    overhead = env.get_overhead_frame()
    if overhead is not None:
        _frame_q.put(("overhead", overhead, {"status": "idle"}))

    _explore_map = None  # preserved across tasks (grid + topo, value reset per task)
    _topo_map    = None

    while True:
        cmd = _cmd_q.get()
        if cmd is None:  # shutdown signal
            env.close()
            break

        goal = cmd  # str

        try:
            def on_frame(frame, nav_state):
                step = nav_state.get("step_count", 0)
                state = {
                    "status": "navigating",
                    "goal":   nav_state.get("goal", goal),
                    "step":   step,
                    "skill":  nav_state.get("current_skill", ""),
                }
                _frame_q.put(("frame", frame, state))

                overhead = env.get_overhead_frame()
                if overhead is not None:
                    _pct = float(np.count_nonzero(overhead)) / overhead.size
                    if _pct < 0.01:
                        print(f"[Server] overhead {_pct:.1%} nonzero — camera inside geometry", flush=True)
                    _frame_q.put(("overhead", overhead, state))

                # Read live explore_map from nav_state (was using server-level
                # _explore_map which is None during navigation).
                if step % 10 == 0:
                    _live_map = nav_state.get("explore_map")
                    if _live_map is not None:
                        robot_pose = env.get_robot_pose()
                        map_png = _map_to_png(_live_map, robot_pose)
                        _frame_q.put(("map", map_png, state))

            def on_thought(step, skill, reason):
                _frame_q.put(("thought", None, {
                    "status": "navigating", "goal": goal,
                    "step": step, "skill": skill, "reason": reason,
                }))

            result = run_task(env, goal, scene_dir=SCENE_DIR, on_frame=on_frame,
                              llm_perceive=llm_perceive,
                              initial_explore_map=_explore_map,
                              initial_topo_map=_topo_map,
                              on_thought=on_thought)

            _explore_map = result.get("explore_map")  # preserve spatial knowledge
            _topo_map    = result.get("topo_map")

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
        elif kind == "overhead":
            jpeg    = _frame_to_jpeg(frame)
            payload = {"type": "overhead", "img": base64.b64encode(jpeg).decode(), **state}
        elif kind == "map":
            # frame is already PNG bytes here
            payload = {"type": "map", "img": base64.b64encode(frame).decode(), **state}
        elif kind == "thought":
            payload = {"type": "thought", **state}
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
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _map_to_png(explore_map, robot_pose) -> bytes:
    """Render the ExploreMap as a top-down PNG (400×400).

    Layers (bottom→top):
      - black  = UNKNOWN cells
      - gray   = EXPLORED cells
      - plasma = VLM value heatmap (blue → green → red)
      - cyan   = frontier cells
      - red    = robot position
    """
    grid  = explore_map.grid   # (N, N) uint8
    value = explore_map.value  # (N, N) float32
    N = grid.shape[0]

    rgb = np.zeros((N, N, 3), dtype=np.uint8)

    # Explored area: dark gray
    explored = grid == 1
    rgb[explored] = [55, 55, 55]

    # Heatmap overlay where value is non-trivial (pseudo-plasma: blue→green→red)
    hot = (value > 0.01) & explored
    if hot.any():
        v = np.clip(value[hot], 0.0, 1.0)
        r = np.clip((255 * (v * 2 - 0.5)), 0, 255).astype(np.uint8)
        g = np.clip((255 * np.sin(np.pi * v)), 0, 255).astype(np.uint8)
        b = np.clip((255 * (1.0 - v * 2)), 0, 255).astype(np.uint8)
        rgb[hot, 0] = r
        rgb[hot, 1] = g
        rgb[hot, 2] = b

    # Frontier cells: cyan
    for fi, fj in explore_map.frontiers():
        if 0 <= fi < N and 0 <= fj < N:
            r0, r1 = max(0, fi - 1), min(N, fi + 2)
            c0, c1 = max(0, fj - 1), min(N, fj + 2)
            rgb[r0:r1, c0:c1] = [0, 200, 200]

    # Robot position: red dot
    pos, _heading = robot_pose
    ri, rj = explore_map._w2g(pos[0], pos[2])
    if 0 <= ri < N and 0 <= rj < N:
        r0, r1 = max(0, ri - 3), min(N, ri + 4)
        c0, c1 = max(0, rj - 3), min(N, rj + 4)
        rgb[r0:r1, c0:c1] = [255, 70, 70]

    # Flip vertically so world +Z points up on screen
    img = Image.fromarray(np.flipud(rgb))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
