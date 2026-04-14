"""
noise_floor_reparse.py — Reparse noise_floor_logs/ after the fact.

The original noise_floor.py had a regex that matched 'Max DD' but the
backtester's first results block uses 'Max Drawdown'. The OOS sub-block
uses 'Max DD'. Either is correct, but the parser only saw one.

This script re-parses every backtest_seed*.log in tools/noise_floor_logs/,
specifically extracting the OUT-OF-SAMPLE block (which is what we care
about for the noise floor measurement, not the full-period results), and
prints a clean σ table.

Run it AFTER all seeds have finished, OR run it now with whatever logs
exist to see partial results.

Usage:
  python3.11 tools/noise_floor_reparse.py
"""

from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "tools" / "noise_floor_logs"


def parse_oos_block(text: str) -> dict | None:
    """
    Parse the -- OUT-OF-SAMPLE 2022-2025 -- block.
    Format:
      -- OUT-OF-SAMPLE 2022-2025 --
      CAGR     :   16.78%
      Sharpe   :     1.91
      Max DD   :   -5.29%
      Win Rate :   61.43%
      Trades   :      503
    """
    m = re.search(
        r"--\s*OUT-OF-SAMPLE.*?--\s*\n"
        r"\s*CAGR\s*:\s*([\-\d\.]+)\s*%\s*\n"
        r"\s*Sharpe\s*:\s*([\-\d\.]+)\s*\n"
        r"\s*Max\s*DD\s*:\s*([\-\d\.]+)\s*%\s*\n"
        r"\s*Win\s*Rate\s*:\s*([\-\d\.]+)\s*%\s*\n"
        r"\s*Trades\s*:\s*(\d+)",
        text,
        re.MULTILINE,
    )
    if not m:
        return None
    return {
        "cagr": float(m.group(1)),
        "sharpe": float(m.group(2)),
        "maxdd": float(m.group(3)),
        "win_rate": float(m.group(4)),
        "trades": int(m.group(5)),
    }


def parse_year_block(text: str) -> dict | None:
    """Parse the year-by-year table inside the OOS block."""
    # Find the OOS block region
    oos_idx = text.find("-- OUT-OF-SAMPLE")
    if oos_idx == -1:
        return None
    region = text[oos_idx:]
    rows = re.findall(
        r"^\s*(\d{4})\s+([\-\d\.]+)%\s+([\-\d\.]+)\s+([\-\d\.]+)%\s+(\d+)",
        region,
        re.MULTILINE,
    )
    return {int(r[0]): {"return": float(r[1]), "sharpe": float(r[2]), "maxdd": float(r[3]), "trades": int(r[4])} for r in rows}


def main():
    if not LOG_DIR.exists():
        print(f"FATAL: {LOG_DIR} not found.")
        sys.exit(1)

    files = sorted(LOG_DIR.glob("backtest_seed*.log"))
    if not files:
        print("No backtest logs found.")
        return

    print("═" * 78)
    print("  NOISE FLOOR REPARSE")
    print("═" * 78)
    print(f"\n  found {len(files)} backtest logs in {LOG_DIR}\n")

    results = []
    for f in files:
        # Extract seed from filename
        m = re.search(r"backtest_seed(\d+)\.log", f.name)
        if not m:
            continue
        seed = int(m.group(1))
        text = f.read_text()
        parsed = parse_oos_block(text)
        years = parse_year_block(text)
        if parsed is None:
            print(f"  ✗ seed {seed}: could not parse OOS block")
            continue
        parsed["seed"] = seed
        parsed["years"] = years
        results.append(parsed)
        print(f"  ✓ seed {seed}: CAGR={parsed['cagr']:>+6.2f}%  "
              f"Sharpe={parsed['sharpe']:>5.2f}  "
              f"MaxDD={parsed['maxdd']:>+6.2f}%  "
              f"WR={parsed['win_rate']:>5.2f}%  "
              f"trades={parsed['trades']}")

    if len(results) < 2:
        print(f"\n  Only {len(results)} parsed result — need ≥2 for σ.")
        return

    # σ table
    print("\n" + "─" * 78)
    print("  POPULATION STATISTICS  (σ across seeds)")
    print("─" * 78)
    for metric in ("cagr", "sharpe", "maxdd", "win_rate"):
        vals = [r[metric] for r in results]
        m = mean(vals)
        s = stdev(vals)
        lo = min(vals)
        hi = max(vals)
        rng = hi - lo
        print(f"  {metric:<10}  mean={m:>+8.3f}  σ={s:>6.3f}  min={lo:>+8.3f}  max={hi:>+8.3f}  range={rng:>6.3f}")

    cagr_sigma = stdev([r["cagr"] for r in results])
    sharpe_sigma = stdev([r["sharpe"] for r in results])
    print(f"\n  ==> CAGR noise floor (1σ):   ±{cagr_sigma:.3f}%")
    print(f"  ==> CAGR min detectable (2σ): {2*cagr_sigma:.3f}%")
    print(f"  ==> Sharpe noise floor (1σ): ±{sharpe_sigma:.3f}")
    print(f"  ==> Sharpe min detectable (2σ): {2*sharpe_sigma:.3f}")

    # Per-year table
    print("\n" + "─" * 78)
    print("  YEAR-BY-YEAR ACROSS SEEDS")
    print("─" * 78)
    all_years = set()
    for r in results:
        if r.get("years"):
            all_years.update(r["years"].keys())
    all_years = sorted(all_years)
    if all_years:
        # Header
        print(f"\n  {'year':<6}", end="")
        for r in results:
            print(f"  seed{r['seed']:>3}", end="")
        print(f"  {'mean':>8}  {'σ':>6}")
        # Rows
        for y in all_years:
            print(f"  {y:<6}", end="")
            yvals = []
            for r in results:
                v = r.get("years", {}).get(y, {}).get("return")
                if v is None:
                    print(f"  {'—':>7}", end="")
                else:
                    print(f"  {v:>+6.1f}%", end="")
                    yvals.append(v)
            if len(yvals) >= 2:
                ym = mean(yvals)
                ys = stdev(yvals)
                print(f"  {ym:>+7.2f}%  {ys:>5.2f}")
            else:
                print()

    print("\n" + "═" * 78)
    print("  HOW TO READ THIS")
    print("═" * 78)
    print("""
  CAGR σ tells you the noise floor of your backtest under seed variation.
  Anything you build from now on must beat 2σ to be 'real':

    if change improves CAGR by < 1σ → noise, ignore
    if change improves CAGR by 1-2σ → suggestive, run more seeds
    if change improves CAGR by > 2σ → real, commit

  Same logic applies to Sharpe.

  Year-by-year table: any year where σ across seeds is large is a year
  where the system's performance is highly seed-dependent (= structurally
  fragile). Years with small σ are stable and the result is real.
""")


if __name__ == "__main__":
    main()
