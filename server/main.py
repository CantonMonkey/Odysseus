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

    _explore_map = None  # preserved across tasks (grid + topo, value reset per task)
    _topo_map    = None

    while True:
        cmd = _cmd_q.get()
        if cmd is None:  # shutdown signal
            env.close()
            break

        if cmd == "__reset__":
            env.reset(SCENE_DIR, start_pos=FIXED_SPAWN)
            _explore_map = None
            _topo_map    = None
            _frame_q.put(("frame", env.get_frame(), {"status": "idle"}))
            _frame_q.put(("status", None, {"status": "reset", "message": "已重置"}))
            continue

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

                _live_topo = nav_state.get("_topo_map")
                needs_map  = (step % 10 == 0)
                needs_topo = (_live_topo is not None and bool(_live_topo.nodes))

                if needs_map or needs_topo:
                    robot_pose = env.get_robot_pose()
                    if needs_map:
                        _live_map = nav_state.get("explore_map")
                        if _live_map is not None:
                            map_png = _map_to_png(_live_map, robot_pose)
                            _frame_q.put(("map", map_png, state))
                    if needs_topo:
                        topo_png = _topo_to_png(_live_topo, robot_pose)
                        _frame_q.put(("topo", topo_png, state))

            def on_thought(step, skill, reason, room=None):
                payload = {
                    "status": "navigating", "goal": goal,
                    "step": step, "skill": skill, "reason": reason,
                }
                if room:
                    payload["room"] = room
                _frame_q.put(("thought", None, payload))

            result = run_task(env, goal, scene_dir=SCENE_DIR, on_frame=on_frame,
                              llm_perceive=llm_perceive,
                              initial_explore_map=_explore_map,
                              initial_topo_map=_topo_map,
                              on_thought=on_thought)

            _explore_map = result.get("explore_map")  # preserve spatial knowledge
            _topo_map    = result.get("topo_map")

            # Push final topo snapshot so the canvas shows the complete map on arrival
            if _topo_map and _topo_map.nodes:
                _final_state = {"status": "done", "goal": goal, "step": -1, "skill": ""}
                _final_pose  = env.get_robot_pose()
                _frame_q.put(("topo", _topo_to_png(_topo_map, _final_pose), _final_state))

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
        elif kind in ("map", "topo"):
            # frame is already PNG bytes
            payload = {"type": kind, "img": base64.b64encode(frame).decode(), **state}
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


@app.post("/reset")
async def reset_maps():
    if _nav_active.is_set():
        return {"ok": False, "error": "导航中，无法重置"}
    _cmd_q.put("__reset__")
    return {"ok": True}


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


def _topo_to_png(topo_map, robot_pose, size: int = 320) -> bytes:
    """Render the TopoMap as a top-down PNG: nodes (colored by room) + edges + robot."""
    from PIL import Image, ImageDraw
    img  = Image.new("RGB", (size, size), (18, 18, 18))
    draw = ImageDraw.Draw(img)

    nodes = topo_map.nodes if topo_map else []
    pos, _ = robot_pose

    ROOM_COLOR = {
        "kitchen":     (255, 140,  30),
        "bedroom":     ( 80, 140, 220),
        "living_room": ( 60, 180,  80),
        "bathroom":    (  0, 190, 190),
        "hallway":     (210, 200,  50),
        "staircase":   (180,  90, 220),
        "other":       (110, 110, 110),
    }

    all_x = [n.pos[0] for n in nodes] + [pos[0]]
    all_z = [n.pos[2] for n in nodes] + [pos[2]]
    margin = 24
    span   = max(max(all_x) - min(all_x), max(all_z) - min(all_z), 6.0)
    cx     = (min(all_x) + max(all_x)) / 2
    cz     = (min(all_z) + max(all_z)) / 2
    scale  = (size - 2 * margin) / span

    def w2i(x, z):
        return (int((x - cx) * scale + size / 2),
                int(-(z - cz) * scale + size / 2))

    if not nodes:
        draw.text((size // 2 - 32, size // 2 - 6), "no nodes yet", fill=(70, 70, 70))
    else:
        for edge in (topo_map.edges if topo_map else []):
            a, b = topo_map.nodes[edge.a], topo_map.nodes[edge.b]
            draw.line([w2i(a.pos[0], a.pos[2]), w2i(b.pos[0], b.pos[2])],
                      fill=(50, 50, 50), width=1)
        for node in nodes:
            px, pz = w2i(node.pos[0], node.pos[2])
            color  = ROOM_COLOR.get(node.room, ROOM_COLOR["other"])
            r = 6
            draw.ellipse([px-r, pz-r, px+r, pz+r], fill=color, outline=(200, 200, 200))
            label = node.room[:3] if node.room else "?"
            draw.text((px + r + 2, pz - 4), label, fill=(170, 170, 170))

    rx, rz = w2i(pos[0], pos[2])
    rr = 5
    draw.ellipse([rx-rr, rz-rr, rx+rr, rz+rr], fill=(255, 60, 60), outline=(255, 200, 200))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
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
