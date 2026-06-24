# Odysseus — VLN Agent (Habitat-Sim + InternVL3)

A zero-shot Vision-and-Language Navigation agent for indoor scenes.  
The user types a Chinese object goal such as *"沙发"* and the robot explores the HM3D scene autonomously using a VLM brain + CLIP detector + online map.

---

## Current Results

**Single scene · HM3D 00800-TEEsavR23oF · InternVL3-8B brain (vLLM) · 3 episodes per goal**

### Single-Goal Navigation (300 steps/ep)

| Goal | SR | SPL | SoftSPL | Avg Steps |
|------|----|-----|---------|-----------|
| 沙发 | **100%** | 0.109 | 0.059 | 300 |
| 冰箱 | 66.7% | 0.048 | 0.013 | 262 |
| 衣柜 | 66.7% | 0.095 | 0.058 | 300 |
| 床   | 33.3% | 0.013 | 0.004 | 300 |
| **Overall** | **66.7%** | **0.066** | **0.033** | 291 |

### Multi-Stage Chain Navigation (沙发 → 冰箱 → 床, shared map, 300 steps/goal)

| Goal | SR | SPL | SoftSPL | Avg Steps |
|------|----|-----|---------|-----------|
| 沙发 | **100%** | 0.165 | 0.080 | 300 |
| 冰箱 | 66.7% | 0.184 | 0.052 | 277 |
| 床   | **100%** | 0.249 | 0.103 | 300 |
| **Overall** | **88.9%** | **0.199** | **0.078** | 292 |

Chain mode outperforms single-goal by +22% SR and 3× SPL. Shared topological map across goals allows the agent to reuse room knowledge, particularly helping 床 (0→100%) which benefits from accumulated spatial context.

---

## Architecture

A dual-cadence design: a slow VLM brain (every 8 steps) sets semantic strategy while a fast CLIP sensor (every step) provides continuous target visibility. Both feed an online value map that biases frontier selection toward the goal.

```
User goal (Chinese)
       │
       ├─[Episode start] VLM text-only → search strategy (room order, floor hint)
       │
       └─ Navigation loop
              │
              ├─ Every step
              │     CLIP ViT-B/32 ──► target score + direction
              │                           │
              │                    ExploreMap value update (EMA per column)
              │
              ├─ Every 8 steps  (or CLIP streak ≥ 5 triggers early)
              │     VLM InternVL3-8B  [RGB + context]
              │       └─ room · relevance · skill · direction · reason
              │                │              │
              │           TopoMap         ExploreMap
              │          add_node()    update_value()
              │
              └─ Skill dispatcher
                    ├── explore_frontier  value-map frontier + topo room hints
                    │                     Habitat A* pathfinder → waypoints
                    ├── snap              depth estimate → 3D target → follow_path
                    ├── follow_path       waypoint tracking toward locked target
                    ├── verify_arrival    360° CLIP scan, sliding-window confirm
                    └── escape            stagnation recovery → teleport
```

**Key design choices**

- **VLM as brain, CLIP as fast sensor** — VLM sets 8-step strategy; CLIP catches the target every step without VLM latency
- **Value map = spatial memory** — VLM relevance scores accumulated (EMA) over direction columns; frontiers ranked by expected target proximity
- **Topological map** — room-level graph routes macro navigation (go to known kitchen) while frontier exploration handles local search
- **No privileged simulator info** — agent uses only RGB-D + odometry; no ground-truth positions or semantic labels at runtime

---

## Progress

### Done
- [x] Chinese NL goal parsing (keyword + VLM fallback)
- [x] InternVL3-8B VLM brain via vLLM HTTP (port 8088)
- [x] CLIP ViT-B/32 per-step target detector
- [x] Online ExploreMap: occupancy + VLM-scored value map (EMA, VLFM-style directional weighting)
- [x] Topological map for room-level navigation history
- [x] Frontier-based exploration with VLM scoring + room priors
- [x] SNAP skill: direction-column depth estimate → 3D target position
- [x] ESCAPE: stagnation detection + random teleport to unexplored area
- [x] Sliding-window verify_arrival (N=6 frames, avg conf ≥ 0.65)
- [x] CLIP-MERGE disabled during verify_arrival scan (prevents false SUCCESS)
- [x] Web server with WebSocket RGB streaming (port 6006)
- [x] Eval harness: 3 goals × 3 episodes, SR / SPL / SoftSPL / PathLen

### TODO
- [ ] **CLIP → value map**: accumulate per-step CLIP scores into the value map for continuous VLFM-style guidance
- [ ] **Scene coverage**: improve frontier scoring to escape local loops (current coverage ~5–8%)
- [ ] **Multi-scene eval**: generalise beyond the single training scene; test on full HM3D minival
- [ ] **VLM-direct decisions**: let the VLM skill output drive navigation without intermediate confidence gates (AgentVLN style)
- [ ] **Pluggable backend**: swap VLM (vLLM / local / API) and add custom skills without touching core loop

---

## Quickstart

```bash
# On GPU server (requires InternVL3 + Habitat-Sim)
conda activate habitat

# 1. Start vLLM server (InternVL3-8B)
bash start_vllm.sh

# 2. Run evaluation (single goal)
python eval.py --goals 沙发 --episodes 3 --max-steps 300

# 2b. Run full suite (single + cross-floor + multi-stage chain)
python eval_full.py --log-dir /tmp/eval_full

# 3. Start web server (port 6006)
python -m uvicorn server.main:app --host 0.0.0.0 --port 6006
```

SSH tunnel for local browser access:
```bash
ssh -CNg -L 6006:127.0.0.1:6006 <user>@<server> -p <port>
# then open http://localhost:6006
```

---

## File Structure

```
Odysseus/
├── agent/
│   ├── loop.py            # main navigation loop (VLM + CLIP + map + skills)
│   ├── skills.py          # follow_path / verify_arrival / snap / visual_servo
│   ├── skill_registry.py  # @skill decorator + dispatch registry
│   ├── explore_map.py     # 2D occupancy grid + VLM value map
│   ├── topo_map.py        # topological room graph
│   ├── clip_detector.py   # CLIP ViT-B/32 per-step target detector
│   ├── habitat_env.py     # Habitat-Sim wrapper (headless EGL, RGBD)
│   ├── llm_agent.py       # VLM perceive() routing (vLLM / local / API / rule)
│   └── backends/          # pluggable VLM backends
│       ├── vllm_http.py   #   OpenAI-compatible HTTP (vLLM)
│       ├── internvl3.py   #   local InternVL3 inference
│       ├── anthropic_api.py
│       └── rule_based.py  #   fallback heuristic
├── server/main.py         # FastAPI + WebSocket frame/topo/map streaming
├── web/index.html         # browser UI (ego view · topo map · chat)
├── eval.py                # single-goal evaluation harness
├── eval_full.py           # full suite (single + cross-floor + chain)
├── run.py                 # server entry point (loads .env, starts uvicorn)
└── start_vllm.sh          # launch vLLM server (InternVL3-8B, port 8088)
```

---

## Dataset

**HM3D minival** — 10 scenes, free academic licence.
