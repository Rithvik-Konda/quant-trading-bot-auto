"""
concentration_analysis.py — Does portfolio concentration predict stop damage?

The stop forensics showed 67% of stops fire while portfolio has ≥6 positions.
But that's not informative on its own — maybe the portfolio is usually full.
This script asks the real question: are stop-days DIFFERENT from non-stop-days
in terms of concentration metrics?

Reads:
  trades_v2.csv                         — to find stop dates
  cache_backtester/position_lifecycle.csv — daily concentration metrics

For every trading day in the OOS window, computes:
  portfolio_size       — N open positions
  portfolio_corr       — average pairwise correlation of open positions
  vix_level            — at that day
  spy_5d_prior         — SPY 5d return heading into that day

Then splits days into:
  - days where ≥1 stop fired in the NEXT 5 trading days ("high stress")
  - days where 0 stops fired in the next 5 days                 ("low stress")

And compares the distributions of each metric between those two groups.

If high-stress days are distinguishable (higher concentration, higher corr,
rising VIX, falling SPY), we have an a priori case for a correlation throttle.
If they're indistinguishable, the "67% fully loaded" observation is base rate
and correlation throttling wouldn't help.

NO BACKFITTING: we are not choosing any threshold. We are just testing whether
there is a difference between the two populations. If there is, the next step
(in a future session) would be to train a model on 2015-2021 data only to
find a throttle function, then test it on 2022-2026 OOS.

Usage:
  python3.11 tools/concentration_analysis.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRADES = REPO / "trades_v2.csv"
LIFECYCLE = REPO / "cache_backtester" / "position_lifecycle.csv"
SPY_PATHS = [REPO / "cache_prices" / "SPY_regime.csv",
             REPO / "cache_prices" / "SPY_3650d.csv"]
VIX_PATHS = [REPO / "cache_prices" / "VIX_regime.csv",
             REPO / "cache_prices" / "VIXY_regime.csv"]


def _load_macro(paths):
    for p in paths:
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0)
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                df = df.loc[~df.index.isna()]
                if getattr(df.index, "tz", None):
                    df.index = df.index.tz_convert("UTC").tz_localize(None)
                df.columns = [str(c).lower() for c in df.columns]
                return df
            except Exception:
                continue
    return None


def _fmt_stat(name, a, b):
    """Format a comparison row: name | mean_a | mean_b | diff | cohens_d"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return f"  {name:<30} (no data)"
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(ddof=1) if len(a) > 1 else 0, b.std(ddof=1) if len(b) > 1 else 0
    pooled = np.sqrt(((len(a)-1)*sa**2 + (len(b)-1)*sb**2) / max(len(a)+len(b)-2, 1)) if len(a)+len(b) > 2 else 0
    d = (ma - mb) / pooled if pooled > 0 else 0
    return f"  {name:<30}  stress={ma:>8.3f}  calm={mb:>8.3f}  Δ={ma-mb:>+7.3f}  d={d:>+5.2f}"


def main():
    os.chdir(REPO)
    print("═" * 78)
    print("  CONCENTRATION vs STOP STRESS — a priori comparison")
    print("═" * 78)

    if not TRADES.exists() or not LIFECYCLE.exists():
        print(f"FATAL: missing {TRADES} or {LIFECYCLE}")
        sys.exit(1)

    trades = pd.read_csv(TRADES)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")

    stops = trades[trades["reason"] == "stop"].copy()
    stop_dates = set(stops["exit_date"].dropna().dt.normalize())
    print(f"\n  {len(stops)} stops on {len(stop_dates)} unique dates")

    lifecycle = pd.read_csv(LIFECYCLE)
    lifecycle["date"] = pd.to_datetime(lifecycle["date"], errors="coerce")

    # Daily aggregates: one row per date
    daily = lifecycle.groupby("date").agg(
        portfolio_size=("portfolio_size", "max"),
        portfolio_corr=("portfolio_corr", "max"),
        vix_level=("vix_now", "max"),
        regime=("regime", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
    ).reset_index()

    spy = _load_macro(SPY_PATHS)
    if spy is not None and "close" in spy.columns:
        spy_series = spy["close"]
        daily["spy_5d_prior"] = daily["date"].apply(
            lambda d: float(spy_series.loc[:d].pct_change(5).iloc[-1]) if d in spy_series.index or any(spy_series.index <= d) else np.nan
        )
    else:
        daily["spy_5d_prior"] = np.nan

    # Label each day: will a stop fire in the next 5 trading days?
    # Use a forward rolling window on sorted dates
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["stop_in_next_5d"] = False
    date_array = daily["date"].values
    for i in range(len(daily)):
        d = daily.loc[i, "date"]
        window_end = d + pd.Timedelta(days=7)  # 5 trading days ≈ 7 calendar days
        future_dates = daily.loc[(daily["date"] > d) & (daily["date"] <= window_end), "date"]
        if any(pd.Timestamp(fd).normalize() in stop_dates for fd in future_dates):
            daily.loc[i, "stop_in_next_5d"] = True

    n_stress = daily["stop_in_next_5d"].sum()
    n_calm = (~daily["stop_in_next_5d"]).sum()
    print(f"\n  Days labeled:")
    print(f"    stress (stop in next 5 days): {n_stress:>4}  ({n_stress/len(daily):.0%})")
    print(f"    calm   (no stop in next 5):  {n_calm:>4}  ({n_calm/len(daily):.0%})")

    if n_stress == 0 or n_calm == 0:
        print("  Cannot compare — one group is empty.")
        return

    stress = daily[daily["stop_in_next_5d"]]
    calm = daily[~daily["stop_in_next_5d"]]

    print("\n" + "─" * 78)
    print("  POPULATION COMPARISON — stress days vs calm days")
    print("─" * 78)
    print("  (Cohen's d: |d|>0.2 small, >0.5 medium, >0.8 large)")
    print()
    print(_fmt_stat("portfolio_size",      stress["portfolio_size"],      calm["portfolio_size"]))
    print(_fmt_stat("portfolio_corr",      stress["portfolio_corr"],      calm["portfolio_corr"]))
    print(_fmt_stat("vix_level",           stress["vix_level"],           calm["vix_level"]))
    print(_fmt_stat("spy_5d_prior",        stress["spy_5d_prior"],        calm["spy_5d_prior"]))

    # Regime distribution
    print("\n  Regime mix:")
    stress_rg = stress["regime"].value_counts(normalize=True) * 100
    calm_rg = calm["regime"].value_counts(normalize=True) * 100
    for r in set(list(stress_rg.index) + list(calm_rg.index)):
        print(f"    {r:<16} stress={stress_rg.get(r,0):>5.1f}%   calm={calm_rg.get(r,0):>5.1f}%")

    # Time-in-market split — how much of "stress" is just TRENDING_BULL base rate?
    print("\n" + "─" * 78)
    print("  SUBGROUP BY REGIME  (controls for 'bull markets stop more because they trade more')")
    print("─" * 78)
    for regime in ["TRENDING_BULL", "CHOPPY", "BEAR"]:
        s = stress[stress["regime"] == regime]
        c = calm[calm["regime"] == regime]
        if len(s) < 5 or len(c) < 5:
            print(f"\n  {regime}: insufficient sample ({len(s)} stress / {len(c)} calm)")
            continue
        print(f"\n  {regime}  ({len(s)} stress / {len(c)} calm days)")
        print(_fmt_stat("portfolio_size", s["portfolio_size"], c["portfolio_size"]))
        print(_fmt_stat("portfolio_corr", s["portfolio_corr"], c["portfolio_corr"]))
        print(_fmt_stat("vix_level",      s["vix_level"],      c["vix_level"]))

    # Also print: what fraction of stress days had portfolio_corr > 0.5?
    print("\n" + "─" * 78)
    print("  SANITY: is high-correlation the distinguishing feature?")
    print("─" * 78)
    for threshold in [0.3, 0.5, 0.7]:
        p_stress = (stress["portfolio_corr"] > threshold).mean()
        p_calm = (calm["portfolio_corr"] > threshold).mean()
        lift = p_stress / p_calm if p_calm > 0 else float("inf")
        print(f"  corr > {threshold:.1f}:  stress={p_stress:.0%}  calm={p_calm:.0%}  lift={lift:.2f}x")

    # VIX threshold distinctions
    for threshold in [20, 25, 30]:
        p_stress = (stress["vix_level"] > threshold).mean()
        p_calm = (calm["vix_level"] > threshold).mean()
        lift = p_stress / p_calm if p_calm > 0 else float("inf")
        print(f"  vix > {threshold}:    stress={p_stress:.0%}  calm={p_calm:.0%}  lift={lift:.2f}x")

    print("\n" + "═" * 78)
    print("  INTERPRETATION")
    print("═" * 78)
    print("""
  If Cohen's d for portfolio_corr is > 0.3 in TRENDING_BULL:
    → Correlation throttle is a real opportunity. Next step:
      train a throttle function on 2015-2021 ONLY, test on 2022-2026 OOS.

  If Cohen's d is near zero:
    → Concentration is NOT distinguishing feature. Leak comes from
      something that looks the same on normal days and pre-stress days.
      (Which means 'add more signal', not 'size down'.)

  If VIX-level lift is > 1.5x in TRENDING_BULL:
    → Vol regime transitions are visible and actionable.
""")


if __name__ == "__main__":
    main()
