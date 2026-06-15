# Odysseus — 具身家居导航 Agent

轮式机器人 VLN（Vision-and-Language Navigation）系统，运行于 Habitat-Sim + HM3D 场景。
用户输入中文指令（如"请到沙发旁边"），机器人自主导航到目标，到达后询问"还需要什么"。

## 功能特性

- **中文自然语言指令**：支持"请到沙发旁边"、"去床边"等日常表达
- **实时视频流**：浏览器内 WebSocket 推流，640×480 第一视角实时画面
- **双层循环架构**：外层 Task Loop（用户触发 → 验证完成 → 等待下一指令）+ 内层 Control Loop（observe → perceive → plan → execute → validate）
- **LLM 增强感知**：接入 Claude Vision API 做每步目标识别，key 缺失时自动 fallback 到语义地图规则导航
- **可扩展**：`agent.execute(task)` 接口统一，后续加 cron 调度支持家居机器人场景

## 架构

```
用户输入指令
    ↓
DialogueAgent.parse_goal()   ← Claude Haiku 解析 / 关键词 fallback
    ↓
run_task(env, goal)          ← 内层控制循环
  ├── SemanticMap.query_target()   → 目标 3D 坐标（HM3D vertex color 解析）
  ├── pathfinder.find_path()       → Habitat navmesh 最短路径
  ├── follow_path()                → 逐点跟随，旋转矩阵确定转向方向
  └── verify_arrival()             → dist < 1.2m → 任务完成
    ↓
DialogueAgent.arrival_message()   ← "还需要什么？"
```

## 安装

```bash
# 依赖：Habitat-Sim 0.3.1 headless EGL，HM3D minival 数据集
conda activate /path/to/habitat-env

pip install fastapi uvicorn pillow imageio anthropic
```

## 快速启动

```bash
# 1. 下载 HM3D minival 数据集到 /path/to/hm3d/
# 2. 设置数据路径（默认 /data3/liangjy/vln/data/hm3d/00800-TEEsavR23oF）
# 3. 启动服务
python run.py --host 0.0.0.0 --port 8000 --gpu 0
```

浏览器打开 `http://<server>:8000`，在聊天框输入"请到沙发旁边"开始导航。

## LLM 配置（可选）

```bash
export ANTHROPIC_API_KEY=sk-...
```

key 未设置时系统自动使用规则 fallback，全部功能可用，只是感知质量略低。

## 文件结构

```
Odysseus/
├── agent/
│   ├── semantic_map.py  # HM3D 语义地图（trimesh vertex color 解析）
│   ├── habitat_env.py   # Habitat-Sim 封装（headless EGL + pathfinder）
│   ├── skills.py        # follow_path / search_room / verify_arrival
│   ├── loop.py          # 内层控制循环
│   └── llm_agent.py     # Claude Vision 感知 + 对话管理（含 fallback）
├── server/
│   └── main.py          # FastAPI + WebSocket 推流
├── web/
│   └── index.html       # 视频流 + 聊天界面
├── run.py               # 启动入口
└── README.md
```

## 数据集

使用 HM3D（Habitat-Matterport 3D Dataset）minival 10 个场景，免费学术授权。
语义标注通过解析 `*.semantic.glb` 顶点颜色 + `*.semantic.txt` color→category 映射获取，
无需 Habitat 内置语义 API，实测 119 个物体类别，覆盖 sofa/bed/chair/table 等家居目标。

## 未来拓展

```python
# 当前：用户实时输入
agent.execute(task="navigate_to sofa")

# 未来：家居调度（只需在外层加 cron）
scheduler.add("0 8 * * *", task="检查客厅")
scheduler.add("0 18 * * *", task="到床边")
```
