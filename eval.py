"""
eval.py — ObjectNav evaluation harness for the Odysseus VLN agent.

Standard ObjectNav metrics (HM3D / MP3D benchmarks):
  SR      : fraction of episodes where final_pos is within success_dist of
            the nearest object instance
  SPL     : SR × L* / max(L*, p)   where L* = geodesic shortest path from
            start to nearest instance, p = actual path length travelled
  SoftSPL : same formula but replaces the binary success indicator with a
            soft distance reward  max(0, 1 - dist_to_nearest / success_dist)

Ground-truth object positions are obtained via agent.semantic_map.query_target
and geodesic distances via habitat_sim.ShortestPath.  These are ONLY used for
metric computation; the agent itself navigates purely from RGB-D + VLM output.
"""

import sys
import numpy as np
import habitat_sim
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from collections import defaultdict

# Project root on the server
_PROJECT = Path("/data3/liangjy/vln/Odysseus")
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from agent.habitat_env import HabitatEnv
from agent.loop import run_task
from agent.semantic_map import query_target  # EVALUATION ONLY — not used by agent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUCCESS_DIST = 1.0           # metres
DATA_DIR     = Path("/data3/liangjy/vln/data/hm3d")
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
    """Return the Euclidean distance to the nearest instance."""
    if not instances:
        return float("inf")
    robot = np.asarray(robot_pos, dtype=np.float32)
    return float(min(np.linalg.norm(np.asarray(p) - robot) for p in instances))


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
        # No reachable instance — skip (not counted as failure)
        print(
            f"    [skip] goal={goal} ep={episode_idx}: "
            "no reachable instance on navmesh"
        )
        return None

    # 3. Set up llm_perceive
    if use_vlm:
        try:
            from agent.llm_agent import perceive
            llm_perceive = lambda frame, g: perceive(frame, g)
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
        # 5. Run the navigation loop
        result = run_task(
            env,
            goal,
            scene_dir=scene_dir,
            on_frame=None,
            llm_perceive=llm_perceive,
            max_steps=max_steps,
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
    return {
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

    env = HabitatEnv(gpu_id=0)
    all_episodes: List[Dict[str, Any]] = []
    per_goal_episodes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for goal in goals:
        print(f"\n[goal: {goal}]")

        # Retrieve ground-truth positions ONCE per goal (evaluation only)
        instances = [np.asarray(p, dtype=np.float32) for p in query_target(scene_dir, goal)]
        if not instances:
            print(f"  No semantic-map instances found for '{goal}' — skipping goal")
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
    args = parser.parse_args()

    print(f"Scene   : {args.scene}")
    print(f"Goals   : {args.goals}")
    print(f"Eps/goal: {args.episodes}  max_steps: {args.max_steps}  vlm: {not args.no_vlm}")

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
    )

    if args.spawn_file and spawn_positions is None:
        from pathlib import Path as _Path
        _Path(args.spawn_file).write_text(json.dumps(results["spawn_positions"], indent=2))
        print(f"Saved spawn positions to {args.spawn_file}")
    print_results(results)
