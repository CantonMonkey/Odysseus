# Odysseus — Embodied Home Navigation Agent

A wheeled-robot VLN (Vision-and-Language Navigation) system built on
Habitat-Sim + HM3D scenes.  
The user types a Chinese instruction such as *"请到沙发旁边"* (go to the sofa),
the robot navigates autonomously, and replies *"还需要什么？"* (anything else?) on arrival.

> 中文文档：[README.zh-CN.md](README.zh-CN.md)

---

## Features

- **Chinese natural-language commands** — "请到沙发旁边", "去床边", etc.
- **Real-time video stream** — WebSocket push of 640 × 480 first-person RGB frames to the browser
- **Dual-loop architecture** — outer Task Loop (receive → navigate → confirm → wait) wrapping an inner Control Loop (observe → perceive → plan → execute → validate)
- **LLM-enhanced perception** — Claude Vision identifies the target object each step; falls back to semantic-map coordinates when no API key is present
- **Extensible** — `run_task(env, goal)` is the single entry point; adding a cron scheduler later requires no changes to the core logic

---

## Architecture

```
User instruction
       │
       ▼
DialogueAgent.parse_goal()     ← Claude Haiku / keyword fallback
       │
       ▼
run_task(env, goal)            ← inner control loop
  ├── SemanticMap.query_target()   → 3D candidates (HM3D vertex-color parsing)
  ├── pathfinder.find_path()       → Habitat navmesh shortest path
  ├── follow_path()                → waypoint tracking with rotation-matrix steering
  └── verify_arrival()             → dist < 1.2 m → task complete
       │
       ▼
DialogueAgent.arrival_message()   ← "还需要什么？"
```

---

## Installation

```bash
# Prerequisites: Habitat-Sim 0.3.1 headless EGL, HM3D minival dataset
conda activate /path/to/habitat-env
pip install fastapi uvicorn pillow imageio anthropic
```

---

## Quick start

```bash
# 1. Place HM3D minival data under DATA_DIR (see agent/semantic_map.py)
# 2. Optionally set your Anthropic key for LLM perception
export ANTHROPIC_API_KEY=sk-...   # optional; rules-only if omitted

# 3. Start the server
python run.py --host 0.0.0.0 --port 8000 --gpu 0
```

Open `http://<server>:8000` in a browser and type a Chinese navigation command.

---

## File structure

```
Odysseus/
├── agent/
│   ├── semantic_map.py   # HM3D semantic map (vertex-color parsing via trimesh)
│   ├── habitat_env.py    # Habitat-Sim wrapper (headless EGL + pathfinder)
│   ├── skills.py         # follow_path / search_room / verify_arrival
│   ├── loop.py           # inner control loop
│   └── llm_agent.py      # Claude Vision perception + dialogue (with fallback)
├── server/
│   └── main.py           # FastAPI + WebSocket streaming
├── web/
│   └── index.html        # video stream + chat UI
├── run.py                # startup entry point
├── README.md             # this file (English)
└── README.zh-CN.md       # 中文文档
```

---

## Dataset

Uses the **HM3D minival** split (10 scenes, free academic licence).  
Semantic labels are extracted by parsing vertex colors in `*.semantic.glb`
against the `*.semantic.txt` color→category table — no Habitat semantic-sensor
JSON files required.  Verified: 119 non-background categories, furniture objects
(sofa, bed, chair, table, …) all have valid 3D coordinates.

---

## Future extension — home robot scheduler

```python
# Current (course demo): user types in real time
run_task(env, goal="沙发")

# Future: cron-scheduled home tasks, same interface
scheduler.add("0 8 * * *", goal="检查客厅")
scheduler.add("0 18 * * *", goal="到床边")
```

Only the outer Task Queue layer changes; `run_task` stays untouched.
