"""
eval.py — ObjectNav evaluation harness for the Odysseus VLN agent.

Standard ObjectNav metrics (HM3D / MP3D benchmarks):
  SR      : fraction of episodes where final_pos is within success_dist of
            the nearest object instance
  SPL     : SR × L* / max(L*, p)   where L* = geodesic shortest path from
            start to nearest instance, p = actual path length travelled
  SoftSPL : same formula but replaces the binary success indicator with a
            soft distance reward  max(0, 1 - dist_to_nearest / success_dist)

Ground-truth object positions are obtained from Habitat's sim.semantic_scene
(correct world coordinates) and geodesic distances via habitat_sim.ShortestPath.
These are ONLY used for metric computation; the agent navigates purely from
RGB-D + VLM output.
"""

import os
import sys
import numpy as np
import habitat_sim
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from collections import defaultdict

# Load .env before anything else so VLN_DATA_DIR etc. are available
def _load_dotenv(path: Path):
    if not path.exists():
        return
    with open(path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            _v = _v.strip().strip('"').strip("'")
            if _k and _k not in os.environ:
                os.environ[_k] = _v
_load_dotenv(Path(__file__).parent / ".env")

# Project root — prefer current directory, fall back to legacy hardcoded path
_PROJECT = Path(os.environ.get("VLN_PROJECT_DIR", str(Path(__file__).parent.resolve())))
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from agent.habitat_env import HabitatEnv
from agent.loop import run_task
from agent.semantic_map import CHINESE_TO_CATEGORY, IGNORE_CATEGORIES

# ---------------------------------------------------------------------------
# Instance discovery (evaluation only — uses Habitat's semantic scene)
# ---------------------------------------------------------------------------

def _glb_to_habitat(p: np.ndarray) -> np.ndarray:
    """Convert GLB Z-up local coords to Habitat Y-up world coords.

    HM3D semantic.glb files are authored in Z-up convention; Habitat uses Y-up.
    Ground-floor objects snap fine without the transform (small Y offset stays
    within the 2m snap threshold), but upper-floor objects (beds, Y_glb≈6.7)
    land far outside the navmesh unless transformed first.
    """
    return np.array([p[0], p[2], -p[1]], dtype=np.float32)


def _load_all_instances(scene_dir: str, goals: List[str],
                        snap_threshold: float = 2.0) -> dict:
    """Load GT instance positions for all goals using navmesh snapping.

    Parses 3D vertex positions from <scene>.semantic.glb via trimesh
    (cached in semantic_cache.json), applies the GLB→Habitat coordinate
    transform, then snaps each position to the navmesh.  Positions that
    don't snap within snap_threshold metres are unreachable and discarded.
    """
    from agent.semantic_map import query_target

    scene_dir_p = Path(scene_dir)
    scene_id    = scene_dir_p.name.split("-", 1)[1]
    scene_glb   = str(scene_dir_p / f"{scene_id}.basis.glb")

    # Temporary sim just to access the navmesh pathfinder.
    _sim_cfg              = habitat_sim.SimulatorConfiguration()
    _sim_cfg.scene_id     = scene_glb
    _a_cfg                = habitat_sim.agent.AgentConfiguration()
    _a_cfg.sensor_specifications = []
    _tmp_sim = habitat_sim.Simulator(habitat_sim.Configuration(_sim_cfg, [_a_cfg]))
    pf       = _tmp_sim.pathfinder

    try:
        result: dict = {}
        for goal in goals:
            raw = [np.asarray(p, dtype=np.float32)
                   for p in query_target(scene_dir, goal)]
            snapped = []
            for p in raw:
                p_hab = _glb_to_habitat(p)
                sp = pf.snap_point(p_hab)
                if not np.isnan(sp).any():
                    d = float(np.linalg.norm(sp - p_hab))
                    if d < snap_threshold:
                        snapped.append(sp)
            result[goal] = snapped
            print(f"  [semantic_map] '{goal}': {len(snapped)}/{len(raw)} "
                  f"navmesh-reachable instance(s) (snap_thr={snap_threshold}m)")
    finally:
        _tmp_sim.close()

    return result


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUCCESS_DIST  = 3.0           # metres
_DATA_DEFAULT = Path("/root/autodl-tmp/data/hm3d") if Path("/root/autodl-tmp/data/hm3d").exists() else Path("/data3/liangjy/vln/data/hm3d")
DATA_DIR      = Path(os.environ.get("VLN_DATA_DIR", str(_DATA_DEFAULT)))
DEFAULT_SCENE = str(DATA_DIR / "00800-TEEsavR23oF")


# ---------------------------------------------------------------------------
# Geodesic distance helper
# ---------------------------------------------------------------------------

def _geodesic(pathfinder, start: np.ndarray, end: np.ndarray) -> float:
    """Return geodesic distance from *start* to *end* via the navmesh.

    Returns float('inf') if no path exists.
    """
    snapped = pathfinder.snap_point(np.asarray(end, dtype=np.float32))
    if np.any(np.isnan(snapped)):
        return float("inf")
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end   = snapped
    if pathfinder.find_path(path):
        return float(path.geodesic_distance)
    return float("inf")


def _nearest_geodesic(
    pathfinder,
    start: np.ndarray,
    instances: List[np.ndarray],
) -> float:
    """Return the geodesic distance to the *nearest reachable* instance."""
    best = float("inf")
    for pos in instances:
        d = _geodesic(pathfinder, start, pos)
        if d < best:
            best = d
    return best


def _nearest_euclidean(robot_pos: np.ndarray, instances: List[np.ndarray]) -> float:
    """Return the horizontal (XZ) Euclidean distance to the nearest instance.

    Uses XZ-only distance so that tall objects (fridge, TV) whose bounding-box
    centroid is elevated above the floor don't inflate the metric.
    """
    if not instances:
        return float("inf")
    rx, rz = float(robot_pos[0]), float(robot_pos[2])
    return float(min(
        np.sqrt((float(p[0]) - rx) ** 2 + (float(p[2]) - rz) ** 2)
        for p in instances
    ))


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    env: HabitatEnv,
    scene_dir: str,
    goal: str,
    instances: List[np.ndarray],
    max_steps: int,
    use_vlm: bool,
    episode_idx: int,
    spawn_pos: Optional[np.ndarray] = None,
    log_dir: Optional[str] = None,
    initial_explore_map=None,
    initial_topo_map=None,
) -> Optional[Dict[str, Any]]:
    """Run one episode and return metrics, or None if episode is unreachable.

    The agent spawns at a random ground-floor position (env.reset handles
    the retry logic to land on Y < 1.5 m).

    Returned dict keys
    ------------------
    success          : bool
    spl              : float
    soft_spl         : float
    path_length      : float    actual Euclidean path length (metres)
    geodesic_dist    : float    L* — geodesic shortest path from start
    dist_to_nearest  : float    final Euclidean distance to nearest instance
    steps            : int
    goal             : str
    episode          : int
    """
    # 1. Spawn at a random ground-floor position (or fixed if spawn_pos given)
    frame = env.reset(scene_dir, start_pos=spawn_pos)
    start_pos, _ = env.get_robot_pose()

    # 2. Compute L* = geodesic distance from start to nearest instance
    pf = env._sim.pathfinder
    L_star = _nearest_geodesic(pf, start_pos, instances)

    if L_star == float("inf"):
        # Target not reachable from start — run anyway, counts as hard failure.
        # VLM may still find the target via semantic reasoning (e.g. go upstairs).
        print(f"    [warn] goal={goal} ep={episode_idx}: L*=inf (target unreachable from spawn) — running")
        L_star = 99.0  # SPL denominator: failure gives SPL=0, success is rare but counted

    # 3. Set up llm_perceive
    if use_vlm:
        try:
            from agent.llm_agent import perceive
            llm_perceive = lambda frame, g, **kw: perceive(frame, g, **kw)
        except ImportError:
            print("    [warn] llm_agent not importable — running without VLM")
            llm_perceive = None
    else:
        llm_perceive = None

    # 4. Track path by recording positions before each step.
    #    We monkey-patch env.step to intercept positions.
    path_positions: List[np.ndarray] = [start_pos.copy()]
    _original_step = env.step

    def _tracked_step(action):
        result = _original_step(action)
        pos, _ = env.get_robot_pose()
        path_positions.append(pos.copy())
        return result

    env.step = _tracked_step  # type: ignore[assignment]

    try:
        # 5. Run the navigation loop.
        # target_instances is intentionally NOT passed — the agent navigates
        # purely from RGB-D + VLM perception, with no privileged GT coordinates.
        # GT instance positions are only used below for metric computation.
        result = run_task(
            env,
            goal,
            scene_dir=scene_dir,
            on_frame=None,
            llm_perceive=llm_perceive,
            max_steps=max_steps,
            target_instances=[],
            initial_explore_map=initial_explore_map,
            initial_topo_map=initial_topo_map,
        )
    finally:
        env.step = _original_step  # restore

    # 6. Compute actual path length p
    p = 0.0
    for i in range(1, len(path_positions)):
        p += float(np.linalg.norm(path_positions[i] - path_positions[i - 1]))

    # 7. Final position and distance to nearest instance
    final_pos, _ = env.get_robot_pose()
    dist_to_nearest = _nearest_euclidean(final_pos, instances)

    # 8. Binary success
    success = dist_to_nearest < SUCCESS_DIST

    # 9. SPL
    denom = max(L_star, p)
    spl   = (float(success) * L_star / denom) if denom > 0 else 0.0

    # 10. SoftSPL — soft distance reward in [0, 1]
    soft_success = max(0.0, 1.0 - dist_to_nearest / SUCCESS_DIST)
    soft_spl     = (soft_success * L_star / denom) if denom > 0 else 0.0

    steps = result.get("step_count", len(path_positions) - 1)

    print(
        f"    ep={episode_idx} success={success} "
        f"dist={dist_to_nearest:.2f}m L*={L_star:.1f}m spl={spl:.3f} steps={steps} "
        f"spawn=({start_pos[0]:.1f},{start_pos[2]:.1f})"
    )

    ep_result = {
        "success":         success,
        "spl":             spl,
        "soft_spl":        soft_spl,
        "path_length":     p,
        "geodesic_dist":   L_star,
        "dist_to_nearest": dist_to_nearest,
        "steps":           steps,
        "goal":            goal,
        "episode":         episode_idx,
        "spawn_pos":       start_pos.tolist(),
    }

    # Write structured per-step VLM log as JSON sidecar
    step_log = result.get("step_log", [])
    if step_log and log_dir:
        _sidecar = Path(log_dir) / f"steplog_{goal}_ep{episode_idx:02d}.json"
        try:
            _sidecar.write_text(json.dumps({
                "episode": ep_result,
                "steps":   step_log,
            }, ensure_ascii=False, indent=2))
            print(f"    [log] {_sidecar}")
        except Exception as _e:
            print(f"    [log] sidecar write failed: {_e}")

    return ep_result


# ---------------------------------------------------------------------------
# Multi-goal chained episode
# ---------------------------------------------------------------------------

def _run_chain_episode(
    env: HabitatEnv,
    scene_dir: str,
    goals: List[str],
    all_instances: Dict[str, List[np.ndarray]],
    max_steps_total: int,
    use_vlm: bool,
    episode_idx: int,
    spawn_pos: Optional[np.ndarray] = None,
    log_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run all goals sequentially in one episode with shared explore/topo maps.

    Implements continuous multi-goal navigation:  after reaching goal N the
    agent continues from its current position towards goal N+1, retaining the
    same occupancy grid and topological memory.  Steps budget is shared across
    all goals.
    """
    from agent.explore_map import ExploreMap
    from agent.topo_map   import TopoMap
    import math

    frame = env.reset(scene_dir, start_pos=spawn_pos)
    start_pos_chain, _ = env.get_robot_pose()

    if use_vlm:
        try:
            from agent.llm_agent import perceive
            llm_perceive = lambda fr, g, **kw: perceive(fr, g, **kw)
        except ImportError:
            llm_perceive = None
    else:
        llm_perceive = None

    shared_explore = ExploreMap()
    shared_topo    = TopoMap()
    results        = []
    steps_used     = 0
    max_per_goal   = math.ceil(max_steps_total / len(goals))

    path_all: List[np.ndarray] = [start_pos_chain.copy()]
    _original_step = env.step

    def _tracked_step(action):
        r = _original_step(action)
        pos, _ = env.get_robot_pose()
        path_all.append(pos.copy())
        return r

    env.step = _tracked_step  # type: ignore[assignment]

    try:
        for gi, goal in enumerate(goals):
            instances = all_instances.get(goal, [])
            if not instances:
                print(f"    [chain] goal={goal}: no instances, skip")
                continue
            cur_pos, _ = env.get_robot_pose()
            L_star = _nearest_geodesic(env._sim.pathfinder, cur_pos, instances)
            if L_star == float("inf"):
                print(f"    [chain] goal={goal}: unreachable, skip")
                continue

            steps_left = max_steps_total - steps_used
            budget     = min(max_per_goal, steps_left)
            path_before = len(path_all)

            result = run_task(
                env, goal, scene_dir=scene_dir, on_frame=None,
                llm_perceive=llm_perceive, max_steps=budget,
                target_instances=[],   # no GT coords in navigation
                initial_explore_map=shared_explore,
                initial_topo_map=shared_topo,
            )
            steps_used += result.get("step_count", 0)

            path_seg = path_all[path_before:]
            p_seg = sum(
                float(np.linalg.norm(path_seg[i] - path_seg[i-1]))
                for i in range(1, len(path_seg))
            ) if len(path_seg) > 1 else 0.0

            final_pos, _ = env.get_robot_pose()
            dist = _nearest_euclidean(final_pos, instances)
            success = dist < SUCCESS_DIST
            denom   = max(L_star, p_seg)
            spl     = (float(success) * L_star / denom) if denom > 0 else 0.0

            soft_success = max(0.0, (SUCCESS_DIST - dist) / SUCCESS_DIST)
            soft_spl = (soft_success * L_star / denom) if denom > 0 else 0.0
            ep_r = {
                "success": success, "spl": spl, "soft_spl": soft_spl,
                "dist_to_nearest": dist, "path_length": p_seg,
                "geodesic_dist": L_star, "steps": result.get("step_count", 0),
                "goal": goal, "episode": episode_idx, "chain_idx": gi,
                "spawn_pos": start_pos_chain.tolist(),
            }
            print(f"    [chain gi={gi}] goal={goal} success={success} "
                  f"dist={dist:.2f}m steps={ep_r['steps']}")
            results.append(ep_r)

            step_log = result.get("step_log", [])
            if step_log and log_dir:
                _s = Path(log_dir) / f"chain_steplog_{goal}_ep{episode_idx:02d}.json"
                try:
                    _s.write_text(json.dumps({"episode": ep_r, "steps": step_log},
                                             ensure_ascii=False, indent=2))
                except Exception:
                    pass
    finally:
        env.step = _original_step

    return results


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    scene_dir: str = DEFAULT_SCENE,
    goals: Optional[List[str]] = None,
    n_episodes_per_goal: int = 3,
    max_steps: int = 200,
    use_vlm: bool = True,
    spawn_positions: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None,
    chain_goals: bool = False,
) -> Dict[str, Any]:
    """Run ObjectNav evaluation and return aggregated metrics.

    Parameters
    ----------
    scene_dir            : path to an HM3D scene directory
    goals                : list of Chinese goal keywords, e.g. ["沙发", "床"]
    n_episodes_per_goal  : number of random-spawn episodes per goal
    max_steps            : maximum navigation steps per episode
    use_vlm              : whether to enable VLM perception (agent.llm_agent)

    Returns
    -------
    dict with keys:
      "per_goal"  : {goal: {"sr", "spl", "soft_spl", "avg_steps",
                            "avg_path_length", "n_episodes", "n_skipped"}}
      "overall"   : {"sr", "spl", "soft_spl", "avg_steps", "avg_path_length",
                     "n_episodes", "n_skipped"}
      "episodes"  : list of per-episode result dicts
    """
    if goals is None:
        goals = ["沙发", "椅子", "床"]

    # Query GT instances BEFORE creating HabitatEnv (temp sim, then closed)
    print(f"[instance discovery] loading semantic scene ...")
    all_gt_instances = _load_all_instances(scene_dir, goals)

    env = HabitatEnv(gpu_id=0)
    all_episodes: List[Dict[str, Any]] = []
    per_goal_episodes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for goal in goals:
        print(f"\n[goal: {goal}]")

        instances = all_gt_instances.get(goal, [])
        if not instances:
            print(f"  No instances found for '{goal}' — skipping goal")
            continue

        print(f"  {len(instances)} instance(s) found for '{goal}'")

        skipped = 0
        for ep_idx in range(n_episodes_per_goal):
            try:
                _spawn = (
                    np.asarray(spawn_positions[goal][ep_idx], dtype=np.float32)
                    if spawn_positions and goal in spawn_positions
                        and ep_idx < len(spawn_positions[goal])
                    else None
                )
                ep_result = _run_episode(
                    env=env,
                    scene_dir=scene_dir,
                    goal=goal,
                    instances=instances,
                    max_steps=max_steps,
                    use_vlm=use_vlm,
                    episode_idx=ep_idx,
                    spawn_pos=_spawn,
                    log_dir=log_dir,
                )
            except Exception as _e:
                print(f"    [error] ep={ep_idx}: {_e}")
                ep_result = None
            if ep_result is None:
                skipped += 1
                continue
            all_episodes.append(ep_result)
            per_goal_episodes[goal].append(ep_result)

    env.close()

    if chain_goals and goals:
        # Run additional chain episodes (shared map across all goals per episode)
        all_instances_map = all_gt_instances  # already loaded above
        env2 = HabitatEnv(gpu_id=0)
        print(f"\n[chain eval: {goals}]")
        for ep_idx in range(n_episodes_per_goal):
            try:
                chain_eps = _run_chain_episode(
                    env=env2,
                    scene_dir=scene_dir,
                    goals=goals,
                    all_instances=all_instances_map,
                    max_steps_total=max_steps * len(goals),
                    use_vlm=use_vlm,
                    episode_idx=ep_idx,
                    log_dir=log_dir,
                )
            except Exception as _e:
                print(f"    [chain error] ep={ep_idx}: {_e}")
                chain_eps = []
            for ep_r in chain_eps:
                g = ep_r["goal"]
                all_episodes.append(ep_r)
                per_goal_episodes[g].append(ep_r)
        env2.close()

    # Aggregate per-goal metrics
    per_goal: Dict[str, Any] = {}
    for goal, eps in per_goal_episodes.items():
        n = len(eps)
        skipped = n_episodes_per_goal - n
        per_goal[goal] = {
            "sr":              float(np.mean([e["success"] for e in eps])) if eps else 0.0,
            "spl":             float(np.mean([e["spl"]     for e in eps])) if eps else 0.0,
            "soft_spl":        float(np.mean([e["soft_spl"] for e in eps])) if eps else 0.0,
            "avg_steps":       float(np.mean([e["steps"]   for e in eps])) if eps else 0.0,
            "avg_path_length": float(np.mean([e["path_length"] for e in eps])) if eps else 0.0,
            "n_episodes":      n,
            "n_skipped":       skipped,
        }

    # Overall aggregation across all goals
    if all_episodes:
        overall = {
            "sr":              float(np.mean([e["success"]     for e in all_episodes])),
            "spl":             float(np.mean([e["spl"]         for e in all_episodes])),
            "soft_spl":        float(np.mean([e["soft_spl"]    for e in all_episodes])),
            "avg_steps":       float(np.mean([e["steps"]       for e in all_episodes])),
            "avg_path_length": float(np.mean([e["path_length"] for e in all_episodes])),
            "n_episodes":      len(all_episodes),
            "n_skipped":       n_episodes_per_goal * len(goals) - len(all_episodes),
        }
    else:
        overall = {
            "sr": 0.0, "spl": 0.0, "soft_spl": 0.0,
            "avg_steps": 0.0, "avg_path_length": 0.0,
            "n_episodes": 0, "n_skipped": 0,
        }

    # Collect spawn positions (for saving to file)
    collected_spawns: Dict[str, Any] = {}
    for ep in all_episodes:
        g = ep["goal"]
        if g not in collected_spawns:
            collected_spawns[g] = []
        collected_spawns[g].append(ep["spawn_pos"])

    return {
        "per_goal":      per_goal,
        "overall":       overall,
        "episodes":      all_episodes,
        "spawn_positions": collected_spawns,
    }


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------

def print_results(results: Dict[str, Any]) -> None:
    """Print a formatted table of evaluation results."""
    print("\n" + "=" * 68)
    print(f"{'ObjectNav Evaluation Results':^68}")
    print("=" * 68)

    hdr = f"{'Goal':<10} {'SR':>6} {'SPL':>7} {'SoftSPL':>9} {'Steps':>7} {'PathLen':>8} {'N':>4}"
    print(hdr)
    print("-" * 68)

    per_goal = results.get("per_goal", {})
    for goal, m in sorted(per_goal.items()):
        n_str = f"{m['n_episodes']}"
        if m["n_skipped"]:
            n_str += f"({m['n_skipped']}↷)"
        print(
            f"{goal:<10} "
            f"{m['sr']:>6.1%} "
            f"{m['spl']:>7.3f} "
            f"{m['soft_spl']:>9.3f} "
            f"{m['avg_steps']:>7.1f} "
            f"{m['avg_path_length']:>8.2f} "
            f"{n_str:>4}"
        )

    print("-" * 68)
    ov = results.get("overall", {})
    n_str = f"{ov.get('n_episodes', 0)}"
    if ov.get("n_skipped", 0):
        n_str += f"({ov['n_skipped']}↷)"
    print(
        f"{'OVERALL':<10} "
        f"{ov.get('sr', 0):>6.1%} "
        f"{ov.get('spl', 0):>7.3f} "
        f"{ov.get('soft_spl', 0):>9.3f} "
        f"{ov.get('avg_steps', 0):>7.1f} "
        f"{ov.get('avg_path_length', 0):>8.2f} "
        f"{n_str:>4}"
    )
    print("=" * 68)
    print("(↷ = skipped episodes with no reachable instance)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ObjectNav evaluation for Odysseus")
    parser.add_argument("--scene", default=DEFAULT_SCENE,
                        help="Path to HM3D scene directory")
    parser.add_argument("--goals", nargs="+", default=["沙发", "椅子", "床"],
                        help="Chinese goal keywords")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per goal")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum navigation steps per episode")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Disable VLM perception (blind agent)")
    parser.add_argument("--spawn-file", default=None,
                        help="JSON file of fixed spawn positions. Load if exists, save if not.")
    parser.add_argument("--log-dir", default="/tmp",
                        help="Directory for per-step JSON sidelogs (default: /tmp)")
    parser.add_argument("--chain-goals", action="store_true",
                        help="Run all goals sequentially in one episode (shared topo/explore maps)")
    args = parser.parse_args()

    print(f"Scene     : {args.scene}")
    print(f"Goals     : {args.goals}")
    print(f"Eps/goal  : {args.episodes}  max_steps: {args.max_steps}  vlm: {not args.no_vlm}")
    print(f"Chain mode: {args.chain_goals}  log_dir: {args.log_dir}")

    spawn_positions = None
    if args.spawn_file:
        from pathlib import Path as _Path
        _sp = _Path(args.spawn_file)
        if _sp.exists():
            spawn_positions = json.loads(_sp.read_text())
            print(f"Loaded spawn positions from {args.spawn_file}")
        else:
            print(f"No spawn file found at {args.spawn_file} — will generate and save")

    results = run_evaluation(
        scene_dir=args.scene,
        goals=args.goals,
        n_episodes_per_goal=args.episodes,
        max_steps=args.max_steps,
        use_vlm=not args.no_vlm,
        spawn_positions=spawn_positions,
        log_dir=args.log_dir,
        chain_goals=args.chain_goals,
    )

    if args.spawn_file and spawn_positions is None:
        from pathlib import Path as _Path
        _Path(args.spawn_file).write_text(json.dumps(results["spawn_positions"], indent=2))
        print(f"Saved spawn positions to {args.spawn_file}")
    print_results(results)
