# Odysseus — 具身家居导航 Agent

基于 Habitat-Sim + HM3D 场景的轮式机器人 VLN（Vision-and-Language Navigation）系统。  
用户输入中文指令（如"请到沙发旁边"），机器人自主导航到目标，到达后询问"还需要什么"。

> English documentation: [README.md](README.md)

---

## 功能特性

- **中文自然语言指令** — 支持"请到沙发旁边"、"去床边"等日常表达
- **实时视频流** — WebSocket 推流 640×480 第一视角 RGB 画面至浏览器
- **双层循环架构** — 外层 Task Loop（接收→导航→确认→等待）嵌套内层 Control Loop（观察→感知→规划→执行→验证）
- **LLM 增强感知** — Claude Vision 逐步识别目标物体；无 API key 时自动 fallback 到语义地图规则导航
- **可扩展** — `run_task(env, goal)` 是统一入口，后续加 cron 调度无需修改核心逻辑

---

## 架构

```
用户输入指令
       │
       ▼
DialogueAgent.parse_goal()     ← Claude Haiku / 关键词 fallback
       │
       ▼
run_task(env, goal)            ← 内层控制循环
  ├── SemanticMap.query_target()   → 目标 3D 坐标（HM3D 顶点颜色解析）
  ├── pathfinder.find_path()       → Habitat navmesh 最短路径
  ├── follow_path()                → 逐点跟随，旋转矩阵确定转向
  └── verify_arrival()             → dist < 1.2m → 任务完成
       │
       ▼
DialogueAgent.arrival_message()   ← "还需要什么？"
```

---

## 安装

```bash
# 前置：Habitat-Sim 0.3.1 headless EGL，HM3D minival 数据集
conda activate /path/to/habitat-env
pip install fastapi uvicorn pillow imageio anthropic
```

---

## 快速启动

```bash
# 1. 将 HM3D minival 数据放至 DATA_DIR（见 agent/semantic_map.py）
# 2. 可选：设置 Anthropic API key 以启用 LLM 感知
export ANTHROPIC_API_KEY=sk-...   # 不设置则使用规则 fallback

# 3. 启动服务
python run.py --host 0.0.0.0 --port 8000 --gpu 0
```

浏览器打开 `http://<server>:8000`，在聊天框输入中文导航指令。

---

## 文件结构

```
Odysseus/
├── agent/
│   ├── semantic_map.py   # HM3D 语义地图（trimesh 顶点颜色解析）
│   ├── habitat_env.py    # Habitat-Sim 封装（headless EGL + pathfinder）
│   ├── skills.py         # follow_path / search_room / verify_arrival
│   ├── loop.py           # 内层控制循环
│   └── llm_agent.py      # Claude Vision 感知 + 对话管理（含 fallback）
├── server/
│   └── main.py           # FastAPI + WebSocket 推流
├── web/
│   └── index.html        # 视频流 + 聊天界面
├── run.py                # 启动入口
├── README.md             # 英文文档
└── README.zh-CN.md       # 本文件（简体中文）
```

---

## 数据集

使用 **HM3D minival**（10 个场景，免费学术授权）。  
语义标注通过解析 `*.semantic.glb` 顶点颜色与 `*.semantic.txt` 的 color→category 映射获取，
无需 Habitat 内置语义传感器的 JSON 配置文件。  
已验证：119 个非背景类别，家具目标（沙发、床、椅子、桌子等）均有有效 3D 坐标。

---

## 未来拓展 — 家居机器人调度

```python
# 当前（课程 Demo）：用户实时输入
run_task(env, goal="沙发")

# 未来：定时家务调度，接口不变
scheduler.add("0 8 * * *", goal="检查客厅")
scheduler.add("0 18 * * *", goal="到床边")
```

只需在外层 Task Queue 加入调度层，`run_task` 本身零改动。
