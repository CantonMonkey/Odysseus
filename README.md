# Odysseus — VLN Agent (Habitat-Sim + InternVL3)

A zero-shot Vision-and-Language Navigation agent for indoor scenes.  
The user types a Chinese object goal such as *"沙发"* and the robot explores the HM3D scene autonomously using a VLM brain + CLIP detector + online map.

---

## Current Status

**eval v25 · single scene (HM3D 00800-TEEsavR23oF) · 9 episodes**

| Goal | SR | Closest dist | Steps |
|------|----|-------------|-------|
| 沙发 | 0% | 5.24 m | 500 |
| 椅子 | 0% | 4.91 m | 500 |
| 床   | 0% | 5.13 m | 500 |

Zero false-early-terminations (fixed in v21–v25). Robot consistently reaches within ~5 m but fails the 3 m threshold. Root cause: limited scene coverage (expl ≈ 5–8%) — the robot loops in a small area instead of spreading out.

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
- [ ] **CLIP → value map accumulation**: CLIP runs every step but scores are not written to the value map; only VLM (every 8 steps) updates it. Accumulating CLIP scores at each step would give VLFM-style continuous guidance and pull frontiers toward observed target locations.
- [ ] **Scene coverage**: current expl ≈ 5–8%. Frontier scoring needs to balance exploration breadth vs. target-seeking to escape local loops.
- [ ] **Multi-scene eval**: currently locked to scene 00800. Generalisation across HM3D minival not yet tested.
- [ ] **VLM direct decision**: remove intermediate confidence/visibility gates; let the VLM skill output drive navigation directly (AgentVLN paradigm).
- [ ] **Pluggable framework**: swap VLM backend (vLLM / local / API) and add custom skills without touching core loop (hloc-style one-liner).

---

## Quickstart

```bash
# On autodl4090 (GPU required for InternVL3 + Habitat-Sim)
conda activate habitat

# 1. Start vLLM server (InternVL3-8B)
bash start_vllm.sh

# 2. Run evaluation
python eval.py --goal 沙发 --n_episodes 3

# 3. Start web server (accessible via SSH tunnel on port 6006)
python run.py
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
