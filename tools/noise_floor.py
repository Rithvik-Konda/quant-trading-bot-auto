"""
noise_floor.py — Measure how much of the OOS CAGR is the model and how much is the seed.

What it does:
  1. Saves the current ranker .joblib files as a backup.
  2. For each seed in SEEDS:
       a. Patches ml_model.py to use that seed.
       b. Retrains the ranker (--save-ensemble --rolling --lightgbm).
       c. Runs the v2 backtester on the OOS window.
       d. Parses CAGR / Sharpe / MaxDD from the log.
  3. Restores the original ranker .joblib files.
  4. Prints mean / std / min / max across all seeds.

Why this matters:
  Your backtester is fully deterministic given trained models. The only
  randomness in the whole stack is the LightGBM seed. If retraining with
  5 seeds gives you a CAGR std of 3%, then ANY "improvement" smaller than
  ~6% (2 std) is statistically indistinguishable from noise.

  This is the number you need before believing any future result.

Usage:
  cd ~/ai_trading_bot_v2
  python3.11 tools/noise_floor.py --seeds 5

Runtime:
  Depends on retrain time. Each seed = 1 retrain + 1 backtest.
  If retrain takes 20 min and backtest takes 30 min → ~5 hours for 5 seeds.
  Run it overnight.
"""

from __future__ import annotations
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
ML_MODEL = REPO / "ml_model.py"
BACKTESTER = REPO / "v2" / "backtester_v2.py"
JOBLIB_GLOB = ["cross_sectional_ranker_3d.joblib",
               "cross_sectional_ranker_5d.joblib",
               "cross_sectional_ranker_7d.joblib"]

SEED_LINE_PATTERN = re.compile(r"random_state=(\d+)\+horizon")
RESULT_PATTERNS = {
    "cagr":   re.compile(r"CAGR[^0-9\-]*([\-\d\.]+)\s*%"),
    "sharpe": re.compile(r"Sharpe[^0-9\-]*([\-\d\.]+)"),
    "maxdd":  re.compile(r"Max\s*DD[^0-9\-]*([\-\d\.]+)\s*%"),
}


def backup_joblibs(suffix: str) -> list[Path]:
    backups = []
    for name in JOBLIB_GLOB:
        src = REPO / name
        if src.exists():
            dst = REPO / f"{name}.{suffix}.bak"
            shutil.copy2(src, dst)
            backups.append(dst)
    return backups


def restore_joblibs(suffix: str) -> None:
    for name in JOBLIB_GLOB:
        bak = REPO / f"{name}.{suffix}.bak"
        if bak.exists():
            shutil.copy2(bak, REPO / name)


def patch_seed(new_base: int) -> int:
    """Patch ml_model.py random_state=N+horizon. Returns the previous base."""
    text = ML_MODEL.read_text()
    matches = SEED_LINE_PATTERN.findall(text)
    if not matches:
        raise RuntimeError("Could not find random_state=N+horizon in ml_model.py")
    prev = int(matches[0])
    new_text = SEED_LINE_PATTERN.sub(f"random_state={new_base}+horizon", text)
    ML_MODEL.write_text(new_text)
    return prev


def clear_feat_cache() -> None:
    cache = REPO / "cache_prices" / "feat_cache"
    if cache.exists():
        for f in cache.glob("*.pkl"):
            f.unlink()


def run_cmd(cmd: list[str], log_path: Path) -> int:
    print(f"  $ {' '.join(cmd)}", flush=True)
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def parse_results(log_path: Path) -> dict[str, float] | None:
    text = log_path.read_text()
    out = {}
    for k, pat in RESULT_PATTERNS.items():
        # take last match (final results table)
        matches = pat.findall(text)
        if not matches:
            return None
        try:
            out[k] = float(matches[-1])
        except ValueError:
            return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--first-seed", type=int, default=42)
    ap.add_argument("--days", type=int, default=3650)
    ap.add_argument("--skip-retrain", action="store_true",
                    help="DEV: only run backtests, assume models already trained")
    args = ap.parse_args()

    os.chdir(REPO)
    log_dir = REPO / "tools" / "noise_floor_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== NOISE FLOOR TEST ({args.seeds} seeds) ===")
    print(f"Repo: {REPO}")
    print(f"Logs: {log_dir}")
    print()

    # Backup current models so we can restore at the end.
    print("[backup] saving current ranker .joblib files...")
    backup_joblibs("noise_floor_orig")

    # Stash original seed value so we can restore the source file too.
    original_text = ML_MODEL.read_text()

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    results: list[dict] = []
    t0 = time.time()

    try:
        for i, seed in enumerate(seeds, start=1):
            print(f"\n--- seed {i}/{len(seeds)} (base={seed}) ---")
            patch_seed(seed)
            clear_feat_cache()

            if not args.skip_retrain:
                retrain_log = log_dir / f"retrain_seed{seed}.log"
                rc = run_cmd(
                    [sys.executable, "ml_model.py",
                     "--save-ensemble", "--rolling", "--lightgbm"],
                    retrain_log,
                )
                if rc != 0:
                    print(f"  [FAIL] retrain rc={rc} — see {retrain_log}")
                    continue

            backtest_log = log_dir / f"backtest_seed{seed}.log"
            rc = run_cmd(
                [sys.executable, "v2/backtester_v2.py",
                 "--days", str(args.days), "--oos"],
                backtest_log,
            )
            if rc != 0:
                print(f"  [FAIL] backtest rc={rc} — see {backtest_log}")
                continue

            parsed = parse_results(backtest_log)
            if parsed is None:
                print(f"  [FAIL] could not parse results from {backtest_log}")
                continue
            parsed["seed"] = seed
            results.append(parsed)
            print(f"  CAGR={parsed['cagr']:+.2f}%  Sharpe={parsed['sharpe']:.2f}  "
                  f"MaxDD={parsed['maxdd']:.2f}%")

    finally:
        print("\n[restore] restoring original ml_model.py and .joblib files...")
        ML_MODEL.write_text(original_text)
        restore_joblibs("noise_floor_orig")

    # Report
    elapsed = (time.time() - t0) / 60
    print("\n" + "=" * 60)
    print(f"NOISE FLOOR RESULTS  ({elapsed:.1f} min total)")
    print("=" * 60)
    if not results:
        print("No successful runs.")
        return

    print(f"{'seed':>6}  {'CAGR':>8}  {'Sharpe':>8}  {'MaxDD':>8}")
    for r in results:
        print(f"{r['seed']:>6}  {r['cagr']:>+7.2f}%  {r['sharpe']:>8.2f}  {r['maxdd']:>+7.2f}%")
    print()
    if len(results) >= 2:
        for k in ("cagr", "sharpe", "maxdd"):
            vals = [r[k] for r in results]
            print(f"  {k:6}  mean={mean(vals):+.2f}  std={stdev(vals):.2f}  "
                  f"min={min(vals):+.2f}  max={max(vals):+.2f}  "
                  f"range={max(vals)-min(vals):.2f}")
        print()
        cagr_std = stdev([r["cagr"] for r in results])
        print(f"==> Noise floor (1σ CAGR): ±{cagr_std:.2f}%")
        print(f"==> Min detectable improvement (2σ): {2*cagr_std:.2f}% CAGR")
        print(f"==> Anything smaller than this is indistinguishable from luck.")


if __name__ == "__main__":
    main()
