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

```
User goal (Chinese)
      │
      ▼
VLM Brain (InternVL3-8B via vLLM)    ← perceive every 8 steps
  └─ returns: room / relevance / skill / direction / confidence

CLIP Detector (ViT-B/32, every step) ← fast target visibility signal

Online ExploreMap (2D occupancy + value map)
  └─ frontier selection biased by VLM relevance scores

TopoMap (room-level graph)
  └─ route to known room node; trigger go_upstairs when needed

Skills
  ├── explore_frontier  ← go to highest-value unexplored frontier
  ├── follow_path       ← Habitat pathfinder → waypoint tracking
  ├── snap              ← CLIP/depth-based 3D target localization
  └── verify_arrival    ← sliding-window confidence scan (N=6, thresh=0.65)
```

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
│   ├── loop.py           # main control loop (VLM + CLIP + map + skills)
│   ├── skills.py         # follow_path / verify_arrival / snap / visual_servo
│   ├── skill_registry.py # @skill decorator + registry
│   ├── explore_map.py    # online 2D occupancy + VLM value map
│   ├── topo_map.py       # topological room graph
│   ├── clip_detector.py  # CLIP ViT-B/32 target detector
│   ├── habitat_env.py    # Habitat-Sim wrapper (headless EGL, RGBD)
│   └── llm_agent.py      # VLM perceive() routing (vLLM / local / API / rule)
├── server/main.py        # FastAPI + WebSocket streaming
├── web/index.html        # browser UI
├── eval.py               # evaluation harness
└── run.py                # startup entry point
```

---

## Dataset

**HM3D minival** — 10 scenes, free academic licence.
