"""
eval_suite.py — Overnight evaluation suite: easy / mid / hard × single / chain.

Runs six configurations sequentially, saves results to /tmp/suite_results.json
and a human-readable summary to /tmp/suite_summary.txt.

Difficulty definition
---------------------
Easy   : single goal, ground-floor objects only (沙发, 椅子)
Mid    : single goal, all objects incl. upper-floor bed (沙发, 椅子, 床)
Hard   : chain navigation — all three goals in one episode, shared map

Usage
-----
python eval_suite.py [--episodes N]   # default 5 episodes per goal
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────

N_EPISODES  = 5        # per goal (overridden by --episodes)
MAX_STEPS   = 500
SCENE_DIR   = None     # resolved at runtime from VLN_DATA_DIR env var

SUITES = [
    # (label,       goals,                chain,  description)
    ("easy_single", ["沙发", "椅子"],     False, "Single-goal, ground-floor objects"),
    ("mid_single",  ["沙发", "椅子", "床"], False, "Single-goal, all objects (incl. upper floor)"),
    ("hard_chain",  ["沙发", "椅子", "床"], True,  "Multi-stage chain: 沙发→椅子→床, shared map"),
]

SUCCESS_DIST = 3.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_table(label: str, per_goal: dict, overall: dict):
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"{'Goal':<10} {'SR':>7} {'SPL':>7} {'SoftSPL':>9} {'Steps':>7}  N")
    print("-" * 68)
    for g, m in per_goal.items():
        skip = f"({m['n_skipped']}↷)" if m["n_skipped"] else ""
        print(f"{g:<10} {m['sr']*100:>6.1f}% {m['spl']:>7.3f} {m['soft_spl']:>9.3f}"
              f" {m['avg_steps']:>7.1f}  {m['n_episodes']}{skip}")
    print("-" * 68)
    print(f"{'OVERALL':<10} {overall['sr']*100:>6.1f}% {overall['spl']:>7.3f}"
          f" {overall['soft_spl']:>9.3f} {overall['avg_steps']:>7.1f}  {overall['n_episodes']}")
    print(sep)


def _aggregate(episodes: list, n_expected: int, goal_list: list) -> tuple:
    """Return (per_goal dict, overall dict) from a flat episode list."""
    from collections import defaultdict
    by_goal = defaultdict(list)
    for e in episodes:
        by_goal[e["goal"]].append(e)

    per_goal = {}
    for g in goal_list:
        eps = by_goal[g]
        n   = len(eps)
        per_goal[g] = {
            "sr":       float(np.mean([e["success"]      for e in eps])) if eps else 0.0,
            "spl":      float(np.mean([e["spl"]          for e in eps])) if eps else 0.0,
            "soft_spl": float(np.mean([e.get("soft_spl", e["spl"]) for e in eps])) if eps else 0.0,
            "avg_steps":float(np.mean([e["steps"]        for e in eps])) if eps else 0.0,
            "avg_path": float(np.mean([e["path_length"]  for e in eps])) if eps else 0.0,
            "n_episodes": n,
            "n_skipped":  max(0, n_expected - n),
        }

    all_eps = list(episodes)
    overall = {
        "sr":        float(np.mean([e["success"]      for e in all_eps])) if all_eps else 0.0,
        "spl":       float(np.mean([e["spl"]          for e in all_eps])) if all_eps else 0.0,
        "soft_spl":  float(np.mean([e.get("soft_spl", e["spl"]) for e in all_eps])) if all_eps else 0.0,
        "avg_steps": float(np.mean([e["steps"]        for e in all_eps])) if all_eps else 0.0,
        "n_episodes":len(all_eps),
    }
    return per_goal, overall


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    args = parser.parse_args()
    n_eps = args.episodes

    # Resolve scene dir
    from pathlib import Path as P
    _default = P("/root/autodl-tmp/data/hm3d") if P("/root/autodl-tmp/data/hm3d").exists() \
               else P("/data3/liangjy/vln/data/hm3d")
    scene_dir = str(P(os.environ.get("VLN_DATA_DIR", str(_default))) / "00800-TEEsavR23oF")

    from eval import run_evaluation, _load_all_instances, _run_chain_episode, \
                     DEFAULT_SCENE, SUCCESS_DIST as SD
    from agent.habitat_env import HabitatEnv

    all_results = {}
    summary_lines = []
    t0_total = time.time()

    for label, goals, chain, description in SUITES:
        print(f"\n{'#'*68}")
        print(f"# SUITE: {label}  —  {description}")
        print(f"# goals={goals}  chain={chain}  episodes={n_eps}")
        print(f"{'#'*68}\n")
        t0 = time.time()

        try:
            results = run_evaluation(
                scene_dir=scene_dir,
                goals=goals,
                n_episodes_per_goal=n_eps,
                max_steps=MAX_STEPS,
                use_vlm=True,
                chain_goals=chain,
            )

            # run_evaluation returns its own structure; extract episode list
            all_eps = results.get("episodes", [])
            per_goal, overall = _aggregate(all_eps, n_eps, goals)

            elapsed = time.time() - t0
            _print_table(f"{label} — {description}", per_goal, overall)
            print(f"  Elapsed: {elapsed/60:.1f} min")

            suite_result = {
                "label": label,
                "description": description,
                "goals": goals,
                "chain": chain,
                "n_episodes": n_eps,
                "per_goal": per_goal,
                "overall": overall,
                "elapsed_sec": elapsed,
            }
            all_results[label] = suite_result

            summary_lines.append(
                f"{label:<15} SR={overall['sr']*100:.1f}%  SPL={overall['spl']:.3f}"
                f"  SoftSPL={overall['soft_spl']:.3f}  N={overall['n_episodes']}"
                f"  ({elapsed/60:.0f}min)"
            )

        except Exception as e:
            print(f"[SUITE ERROR] {label}: {e}")
            import traceback; traceback.print_exc()
            all_results[label] = {"error": str(e)}
            summary_lines.append(f"{label:<15} ERROR: {e}")

    # ── Final summary ──────────────────────────────────────────────────────────
    total_min = (time.time() - t0_total) / 60
    print(f"\n{'='*68}")
    print(f"  EVAL SUITE COMPLETE  ({total_min:.0f} min total)")
    print(f"{'='*68}")
    for line in summary_lines:
        print(f"  {line}")
    print(f"{'='*68}\n")

    # Save results
    out_json = Path("/tmp/suite_results.json")
    out_txt  = Path("/tmp/suite_summary.txt")
    out_json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    out_txt.write_text("\n".join(summary_lines) + f"\n\nTotal: {total_min:.0f} min\n")
    print(f"Results saved to {out_json} and {out_txt}")


if __name__ == "__main__":
    main()
