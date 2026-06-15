"""
server/main.py
FastAPI 后端。Habitat 所有 GL 操作都在专用 habitat_thread 中执行（线程本地 EGL）。
主线程与 habitat_thread 通过 Queue 通信，避免跨线程 GL 上下文问题。
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

WEB_DIR = Path(__file__).parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="Odysseus VLN")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Habitat worker（单线程，拥有 GL context）─────────────────────

_cmd_q: queue.Queue = queue.Queue()    # 主线程 → habitat_thread
_frame_q: queue.Queue = queue.Queue()  # habitat_thread → 主线程（帧+状态）
_nav_active = threading.Event()        # 导航进行中标志

_ws_clients: list[WebSocket] = []
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _habitat_worker():
    """专用线程：初始化 Habitat，处理导航命令，向 _frame_q 推送帧。"""
    from agent.habitat_env import HabitatEnv
    from agent.loop import run_task, SCENE_DIR
    from agent.llm_agent import DialogueAgent

    env = HabitatEnv(gpu_id=0)
    env.reset(SCENE_DIR)
    dialogue = DialogueAgent()

    # 推送初始帧
    _frame_q.put(("frame", env.get_frame(), {"status": "idle"}))

    while True:
        cmd = _cmd_q.get()
        if cmd is None:
            env.close()
            break

        goal = cmd  # str

        try:
            env.reset(SCENE_DIR)

            def on_frame(frame, nav_state):
                state = {
                    "status": "navigating",
                    "goal": nav_state.get("goal", goal),
                    "step": nav_state.get("step_count", 0),
                    "skill": nav_state.get("current_skill", ""),
                }
                _frame_q.put(("frame", frame, state))

            result = run_task(env, goal, scene_dir=SCENE_DIR, on_frame=on_frame)

            if result.get("done"):
                msg = dialogue.arrival_message()
                final = {"status": "arrived", "goal": goal, "message": msg}
            else:
                final = {"status": "timeout", "goal": goal, "message": "导航超时，未能到达"}

            _frame_q.put(("status", None, final))
        except Exception as e:
            _frame_q.put(("status", None, {"status": "error", "message": str(e)}))
        finally:
            _nav_active.clear()


# ── 帧广播协程（event loop 侧）──────────────────────────────────

async def _frame_broadcaster():
    """持续从 _frame_q 取帧，广播给所有 WebSocket 客户端。"""
    while True:
        try:
            item = _frame_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue

        kind, frame, state = item

        if kind == "frame":
            jpeg = _frame_to_jpeg(frame)
            payload = {
                "type": "frame",
                "img": base64.b64encode(jpeg).decode(),
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


# ── FastAPI 生命周期 ──────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _event_loop
    _event_loop = asyncio.get_event_loop()

    # 启动 Habitat 专用线程
    t = threading.Thread(target=_habitat_worker, daemon=True)
    t.start()

    # 启动帧广播协程
    asyncio.create_task(_frame_broadcaster())
    print("[Server] Ready.")


@app.on_event("shutdown")
async def shutdown():
    _cmd_q.put(None)


# ── HTTP 接口 ──────────────────────────────────────────────────

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
        return {"ok": False, "error": "正在导航中，请稍候"}

    goal = DialogueAgent().parse_goal(text)
    if goal is None:
        return {"ok": False, "error": f"无法解析目标：{text}"}

    _nav_active.set()
    _cmd_q.put(goal)
    return {"ok": True, "goal": goal}


# ── WebSocket ─────────────────────────────────────────────────

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


# ── 工具函数 ──────────────────────────────────────────────────

def _frame_to_jpeg(frame: np.ndarray) -> bytes:
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()
