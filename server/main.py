"""
server/main.py
FastAPI 后端：
  POST /command  接收中文导航指令，启动异步导航任务
  GET  /ws       WebSocket 推流 RGB 帧 + 状态 JSON
  GET  /         服务前端 index.html
"""

import asyncio
import base64
import io
import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.habitat_env import HabitatEnv
from agent.llm_agent import DialogueAgent
from agent.loop import run_task, SCENE_DIR

app = FastAPI(title="Odysseus VLN")

WEB_DIR = Path(__file__).parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── 全局共享状态 ───────────────────────────────────────────────

_env: Optional[HabitatEnv] = None
_dialogue = DialogueAgent()
_nav_lock = threading.Lock()

# 最新帧和状态（用于 WebSocket 广播）
_latest_jpeg: Optional[bytes] = None
_latest_state: dict = {"status": "idle"}
_ws_clients: list[WebSocket] = []


# ── 启动时初始化 Habitat ────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _env
    _env = HabitatEnv(gpu_id=0)
    _env.reset(SCENE_DIR)
    _update_frame(_env.get_frame())
    print("[Server] Habitat env ready.")


@app.on_event("shutdown")
async def shutdown():
    if _env:
        _env.close()


# ── HTTP 接口 ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = WEB_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")


class CommandRequest(BaseModel):
    text: str


@app.post("/command")
async def command(req: CommandRequest):
    user_text = req.text.strip()
    if not user_text:
        return {"ok": False, "error": "empty input"}

    goal = _dialogue.parse_goal(user_text)
    if goal is None:
        return {"ok": False, "error": f"无法解析目标：{user_text}"}

    # 在后台线程执行导航（避免阻塞 FastAPI event loop）
    if not _nav_lock.acquire(blocking=False):
        return {"ok": False, "error": "正在导航中，请稍候"}

    threading.Thread(target=_run_nav, args=(goal,), daemon=True).start()
    return {"ok": True, "goal": goal}


# ── WebSocket ─────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        # 立刻发送当前帧
        if _latest_jpeg:
            await _send_frame(websocket, _latest_jpeg, _latest_state)
        while True:
            # 保持连接，等待 ping 或断开
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ── 内部：导航线程 ─────────────────────────────────────────────

def _run_nav(goal: str):
    try:
        _env.reset(SCENE_DIR)
        _update_state({"status": "navigating", "goal": goal, "step": 0})

        def on_frame(frame: np.ndarray, nav_state: dict):
            jpeg = _frame_to_jpeg(frame)
            state = {
                "status": "navigating",
                "goal": nav_state.get("goal", goal),
                "step": nav_state.get("step_count", 0),
                "skill": nav_state.get("current_skill", ""),
                "done": nav_state.get("done", False),
            }
            _update_frame(frame, state)
            asyncio.run_coroutine_threadsafe(
                _broadcast(jpeg, state),
                _get_loop(),
            )

        result = run_task(_env, goal, scene_dir=SCENE_DIR, on_frame=on_frame)

        if result.get("done"):
            arrival_msg = _dialogue.arrival_message()
            final_state = {"status": "arrived", "goal": goal, "message": arrival_msg}
        else:
            final_state = {"status": "timeout", "goal": goal, "message": "导航超时，未能到达"}

        _update_state(final_state)
        asyncio.run_coroutine_threadsafe(
            _broadcast_json(final_state),
            _get_loop(),
        )
    finally:
        _nav_lock.release()


async def _broadcast(jpeg: bytes, state: dict):
    dead = []
    for ws in list(_ws_clients):
        try:
            await _send_frame(ws, jpeg, state)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


async def _broadcast_json(data: dict):
    dead = []
    for ws in list(_ws_clients):
        try:
            await ws.send_text(json.dumps({"type": "status", **data}))
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


async def _send_frame(ws: WebSocket, jpeg: bytes, state: dict):
    payload = {
        "type": "frame",
        "img": base64.b64encode(jpeg).decode(),
        **state,
    }
    await ws.send_text(json.dumps(payload))


def _update_frame(frame: np.ndarray, state: Optional[dict] = None):
    global _latest_jpeg, _latest_state
    _latest_jpeg = _frame_to_jpeg(frame)
    if state:
        _latest_state = state


def _update_state(state: dict):
    global _latest_state
    _latest_state = state


def _frame_to_jpeg(frame: np.ndarray) -> bytes:
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _event_loop
    if _event_loop is None:
        _event_loop = asyncio.get_event_loop()
    return _event_loop


@app.on_event("startup")
async def capture_loop():
    global _event_loop
    _event_loop = asyncio.get_event_loop()
