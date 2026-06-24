"""
eval_full.py — Full evaluation suite: single-goal, cross-floor, multi-stage chain.

Runs three categories sequentially:
  1. single_goal  : 冰箱, 沙发, 床, 衣柜  (3 eps × 300 steps each)
  2. cross_floor  : 床, 衣柜              (3 eps × 400 steps, separate per-goal)
  3. multi_stage  : 沙发 → 冰箱 → 床     (3 chain eps × 300 steps/goal)

Logs:
  - SR / SPL / SoftSPL summary table per category
  - Per-step sidecar JSON (vlm_raw CoT) under --log-dir/<run_name>/
  - CoT sample printed to console after each run
  - /tmp/eval_full_summary.json  aggregated across all runs

Usage:
  python eval_full.py
  python eval_full.py --log-dir /root/eval_results
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_DATA_DEFAULT = (
    Path("/root/autodl-tmp/data/hm3d")
    if Path("/root/autodl-tmp/data/hm3d").exists()
    else Path("/data3/liangjy/vln/data/hm3d")
)
DATA_DIR      = Path(os.environ.get("VLN_DATA_DIR", str(_DATA_DEFAULT)))
DEFAULT_SCENE = str(DATA_DIR / "00800-TEEsavR23oF")

RUNS = [
    dict(name="single_goal",  goals=["冰箱", "沙发", "床", "衣柜"], episodes=3, max_steps=300, chain=False),
    dict(name="cross_floor",  goals=["床", "衣柜"],                  episodes=3, max_steps=400, chain=False),
    dict(name="multi_stage",  goals=["沙发", "冰箱", "床"],          episodes=3, max_steps=300, chain=True),
]


def _print_table(name: str, results: dict) -> None:
    per_goal = results.get("per_goal", {})
    overall  = results.get("overall", {})
    bar = "=" * 66
    print(f"\n{bar}")
    print(f"  {name}")
    print(bar)
    print(f"  {'Goal':<14} {'SR':>6} {'SPL':>7} {'SoftSPL':>9} {'Steps':>7}  N")
    print(f"  {'-'*60}")
    for goal, m in per_goal.items():
        skip = f"(-{m['n_skipped']}↷)" if m.get("n_skipped") else ""
        print(f"  {goal:<14} {m['sr']*100:>5.1f}%  {m['spl']:>6.3f}  "
              f"{m['soft_spl']:>8.3f}  {m['avg_steps']:>6.1f}  {m['n_episodes']}{skip}")
    print(f"  {'-'*60}")
    n_skip = sum(m.get("n_skipped", 0) for m in per_goal.values())
    skip_s = f"(-{n_skip}↷)" if n_skip else ""
    print(f"  {'OVERALL':<14} {overall.get('sr',0)*100:>5.1f}%  {overall.get('spl',0):>6.3f}  "
          f"{overall.get('soft_spl',0):>8.3f}  {overall.get('avg_steps',0):>6.1f}  "
          f"{overall.get('n_episodes',0)}{skip_s}")
    print(bar)


def _print_cot_sample(log_dir: Path, max_eps: int = 2) -> None:
    files = sorted(log_dir.glob("steplog_*.json"))
    if not files:
        print("  (no CoT sidecar logs found)")
        return
    shown = 0
    for f in files:
        if shown >= max_eps:
            break
        try:
            d     = json.loads(f.read_text())
            ep    = d.get("episode", {})
            steps = [s for s in d.get("steps", []) if s.get("vlm_raw")]
            if not steps:
                continue
            print(f"\n  [{f.name}]  success={ep.get('success')}  steps={ep.get('steps')}  "
                  f"CoT hits={len(steps)}")
            for s in steps[:6]:
                raw    = json.loads(s["vlm_raw"])
                reason = raw.get("reason", "")
                print(f"    step={s['step']:3d}  room={s.get('room',''):12s}  "
                      f"conf={raw.get('confidence',0):.2f}  rel={raw.get('relevance',0):.2f}"
                      f"  reason={reason!r}")
            shown += 1
        except Exception:
            continue


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",   default=DEFAULT_SCENE)
    parser.add_argument("--log-dir", default="/tmp/eval_full")
    args = parser.parse_args()

    log_root = Path(args.log_dir)
    log_root.mkdir(parents=True, exist_ok=True)

    from eval import run_evaluation

    all_results = {}
    suite_start = time.time()

    print(f"\nScene    : {args.scene}")
    print(f"Log root : {log_root}")
    print(f"Runs     : {[r['name'] for r in RUNS]}\n")

    for cfg in RUNS:
        run_dir = log_root / cfg["name"]
        run_dir.mkdir(exist_ok=True)

        print(f"\n{'#'*66}")
        print(f"# {cfg['name'].upper()}  goals={cfg['goals']}  "
              f"eps={cfg['episodes']}  max_steps={cfg['max_steps']}  chain={cfg['chain']}")
        print(f"{'#'*66}")

        t0 = time.time()
        try:
            results = run_evaluation(
                scene_dir           = args.scene,
                goals               = cfg["goals"],
                n_episodes_per_goal = cfg["episodes"],
                max_steps           = cfg["max_steps"],
                use_vlm             = True,
                log_dir             = str(run_dir),
                chain_goals         = cfg["chain"],
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results = {"per_goal": {}, "overall": {}, "error": str(exc)}

        elapsed = time.time() - t0
        results["elapsed_min"] = elapsed / 60
        all_results[cfg["name"]] = results

        _print_table(cfg["name"], results)
        print(f"  Elapsed: {elapsed/60:.1f} min")

        print("\n  -- CoT sample --")
        _print_cot_sample(run_dir)

    # Final cross-run summary
    total_min = (time.time() - suite_start) / 60
    bar = "=" * 66
    print(f"\n{bar}")
    print(f"  SUITE COMPLETE  ({total_min:.1f} min total)")
    print(bar)
    print(f"  {'Run':<18} {'SR':>6} {'SPL':>7} {'SoftSPL':>9}")
    print(f"  {'-'*44}")
    for run_name, res in all_results.items():
        ov = res.get("overall", {})
        err = " ERROR" if res.get("error") else ""
        print(f"  {run_name:<18} {ov.get('sr',0)*100:>5.1f}%  {ov.get('spl',0):>6.3f}  "
              f"{ov.get('soft_spl',0):>8.3f}{err}")
    print(bar)

    out = log_root / "summary.json"
    out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nFull logs : {log_root}/")
    print(f"Summary   : {out}")


if __name__ == "__main__":
    main()
