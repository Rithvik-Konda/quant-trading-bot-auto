"""
vix_term_structure.py — Does VIX term structure predict SPY returns?

This is a test of a KNOWN risk premium, not a fitted signal. The mechanism:
  - VIX9D     = 9-day expected vol
  - VIX       = 30-day expected vol  (what everyone calls "VIX")
  - VIX3M     = 93-day expected vol
  - VIX6M     = 6-month expected vol

TERM STRUCTURE STATES
  Normal (contango): VIX9D < VIX < VIX3M
    Market is calm, traders pay more for longer-dated vol.
    This is the "carry" state — short vol / long equity earns premium.

  Stressed (backwardation): VIX9D > VIX or VIX > VIX3M
    Short-term fear exceeds long-term fear.
    Historically associated with strong mean-reversion — fear is overpriced.
    Classic "buy when others are fearful" regime.

HYPOTHESIS (standard academic result, not a novel finding)
  When VIX term structure is backwardated, next 5-20 days of SPY returns
  are elevated. When heavily contangoed, returns are muted or slightly
  negative (carry unwind).

This script tests whether that hypothesis holds IN YOUR DATA so we know
if it's a usable overlay for your system.

NO TUNING
  Thresholds are the natural ones: backwardation means ratio > 1, contango
  means ratio < 1. We do NOT pick any threshold to maximize a metric. We
  look at the monotonic relationship between the ratio and forward returns.

Usage:
  python3.11 tools/vix_term_structure.py

Dependencies:
  yfinance (you already have it) for VIX9D, VIX3M
  Your existing SPY and VIX caches
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "cache_prices"


def _load_cached(name_options: list[str]) -> pd.DataFrame | None:
    for name in name_options:
        p = CACHE_DIR / name
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


def _download_yf(ticker: str, period: str = "10y") -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
        if df is None or len(df) == 0:
            return None
        if getattr(df.index, "tz", None):
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.columns = [str(c).lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"    [ERROR] yfinance download {ticker}: {e}")
        return None


def main():
    os.chdir(REPO)
    print("═" * 78)
    print("  VIX TERM STRUCTURE — does backwardation predict SPY returns?")
    print("═" * 78)

    # Load SPY from existing cache
    print("\n  [1/4] loading SPY from cache...")
    spy = _load_cached(["SPY_regime.csv", "SPY_3650d.csv"])
    if spy is None or "close" not in spy.columns:
        print("  ERROR: SPY not in cache. Aborting.")
        sys.exit(1)
    print(f"    SPY: {len(spy)} rows, {spy.index.min().date()} → {spy.index.max().date()}")

    # Load VIX from existing cache
    print("\n  [2/4] loading VIX from cache...")
    vix = _load_cached(["VIX_regime.csv", "VIXY_regime.csv", "VIX_3650d.csv"])
    if vix is None or "close" not in vix.columns:
        print("  ERROR: VIX not in cache. Aborting.")
        sys.exit(1)
    print(f"    VIX: {len(vix)} rows, {vix.index.min().date()} → {vix.index.max().date()}")

    # VIX9D and VIX3M are not normally cached — download them
    print("\n  [3/4] downloading VIX9D and VIX3M from yfinance...")
    vix9d_cache = CACHE_DIR / "VIX9D_regime.csv"
    vix3m_cache = CACHE_DIR / "VIX3M_regime.csv"

    if vix9d_cache.exists():
        print(f"    [cache] VIX9D")
        vix9d = pd.read_csv(vix9d_cache, index_col=0)
        vix9d.index = pd.to_datetime(vix9d.index, utc=True, errors="coerce")
        vix9d = vix9d.loc[~vix9d.index.isna()]
        if getattr(vix9d.index, "tz", None):
            vix9d.index = vix9d.index.tz_convert("UTC").tz_localize(None)
        vix9d.columns = [c.lower() for c in vix9d.columns]
    else:
        vix9d = _download_yf("^VIX9D")
        if vix9d is not None:
            vix9d.to_csv(vix9d_cache)

    if vix3m_cache.exists():
        print(f"    [cache] VIX3M")
        vix3m = pd.read_csv(vix3m_cache, index_col=0)
        vix3m.index = pd.to_datetime(vix3m.index, utc=True, errors="coerce")
        vix3m = vix3m.loc[~vix3m.index.isna()]
        if getattr(vix3m.index, "tz", None):
            vix3m.index = vix3m.index.tz_convert("UTC").tz_localize(None)
        vix3m.columns = [c.lower() for c in vix3m.columns]
    else:
        vix3m = _download_yf("^VIX3M")
        if vix3m is not None:
            vix3m.to_csv(vix3m_cache)

    if vix9d is None or vix3m is None:
        print("  ERROR: could not load VIX9D or VIX3M.")
        print("  yfinance may have rate-limited. Try again in a few minutes.")
        sys.exit(1)
    print(f"    VIX9D: {len(vix9d)} rows, {vix9d.index.min().date()} → {vix9d.index.max().date()}")
    print(f"    VIX3M: {len(vix3m)} rows, {vix3m.index.min().date()} → {vix3m.index.max().date()}")

    # Build aligned dataframe
    print("\n  [4/4] aligning and computing forward returns...")

    # Normalize all indices to date-only (strips any tz and time-of-day)
    def _norm(df):
        idx = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
        return df.set_axis(idx.normalize(), axis=0)

    spy_n   = _norm(spy)
    vix_n   = _norm(vix)
    vix9d_n = _norm(vix9d)
    vix3m_n = _norm(vix3m)

    df = pd.DataFrame({
        "spy":    spy_n["close"],
        "vix":    vix_n["close"],
        "vix9d":  vix9d_n["close"],
        "vix3m":  vix3m_n["close"],
    }).dropna()
    print(f"    aligned: {len(df)} days, {df.index.min().date() if len(df) else 'none'} → {df.index.max().date() if len(df) else 'none'}")

    if len(df) < 250:
        print("  WARN: very few aligned days. Results may be noisy.")

    # Compute term structure ratios
    df["ratio_9d_30d"]  = df["vix9d"] / df["vix"]    # >1 = short-term backwardation
    df["ratio_30d_3m"]  = df["vix"] / df["vix3m"]    # >1 = medium backwardation
    df["vix_level"]     = df["vix"]

    # Forward SPY returns
    for horizon in (5, 10, 20):
        df[f"spy_fwd_{horizon}d"] = df["spy"].pct_change(horizon).shift(-horizon)

    df = df.dropna()
    print(f"    after dropna: {len(df)} days")

    # ── Test 1: Correlation ──
    print("\n" + "─" * 78)
    print("  TEST 1 — linear correlation of ratio to forward return")
    print("─" * 78)
    print(f"  {'signal':<22} {'corr_5d':>10} {'corr_10d':>10} {'corr_20d':>10}")
    for sig in ("ratio_9d_30d", "ratio_30d_3m"):
        c5  = df[sig].corr(df["spy_fwd_5d"])
        c10 = df[sig].corr(df["spy_fwd_10d"])
        c20 = df[sig].corr(df["spy_fwd_20d"])
        print(f"  {sig:<22} {c5:>+10.3f} {c10:>+10.3f} {c20:>+10.3f}")
    print("\n  Positive correlation = higher ratio (more backwardation) → higher forward returns")
    print("  This is the predicted mean-reversion effect.")

    # ── Test 2: Quintile sort ──
    print("\n" + "─" * 78)
    print("  TEST 2 — quintile sort on ratio_9d_30d → avg forward return")
    print("─" * 78)
    df["q5_short"] = pd.qcut(df["ratio_9d_30d"], 5, labels=["Q1 (most contango)", "Q2", "Q3", "Q4", "Q5 (most backwardation)"])
    print(f"\n  {'quintile':<28} {'mean ratio':>12} {'avg 5d fwd':>12} {'avg 10d fwd':>12} {'avg 20d fwd':>12} {'n':>6}")
    for q, grp in df.groupby("q5_short", observed=True):
        print(f"  {str(q):<28} {grp['ratio_9d_30d'].mean():>12.3f} "
              f"{grp['spy_fwd_5d'].mean()*100:>+11.2f}% "
              f"{grp['spy_fwd_10d'].mean()*100:>+11.2f}% "
              f"{grp['spy_fwd_20d'].mean()*100:>+11.2f}% "
              f"{len(grp):>6}")

    # Spread: Q5 - Q1
    q5 = df[df["q5_short"] == "Q5 (most backwardation)"]
    q1 = df[df["q5_short"] == "Q1 (most contango)"]
    if len(q5) > 0 and len(q1) > 0:
        for horizon in (5, 10, 20):
            spread = q5[f"spy_fwd_{horizon}d"].mean() - q1[f"spy_fwd_{horizon}d"].mean()
            # t-test on the spread
            var_spread = (q5[f"spy_fwd_{horizon}d"].var() / len(q5)) + (q1[f"spy_fwd_{horizon}d"].var() / len(q1))
            se_spread = np.sqrt(var_spread) if var_spread > 0 else 1e-9
            t = spread / se_spread
            print(f"  Q5 - Q1 spread @ {horizon}d: {spread*100:+.2f}%   t={t:+.2f}")

    # ── Test 3: Medium-term ratio (vix / vix3m) ──
    print("\n" + "─" * 78)
    print("  TEST 3 — quintile sort on ratio_30d_3m → avg forward return")
    print("─" * 78)
    df["q5_med"] = pd.qcut(df["ratio_30d_3m"], 5, labels=["Q1 (most contango)", "Q2", "Q3", "Q4", "Q5 (most backwardation)"])
    print(f"\n  {'quintile':<28} {'mean ratio':>12} {'avg 5d fwd':>12} {'avg 10d fwd':>12} {'avg 20d fwd':>12} {'n':>6}")
    for q, grp in df.groupby("q5_med", observed=True):
        print(f"  {str(q):<28} {grp['ratio_30d_3m'].mean():>12.3f} "
              f"{grp['spy_fwd_5d'].mean()*100:>+11.2f}% "
              f"{grp['spy_fwd_10d'].mean()*100:>+11.2f}% "
              f"{grp['spy_fwd_20d'].mean()*100:>+11.2f}% "
              f"{len(grp):>6}")

    q5 = df[df["q5_med"] == "Q5 (most backwardation)"]
    q1 = df[df["q5_med"] == "Q1 (most contango)"]
    if len(q5) > 0 and len(q1) > 0:
        for horizon in (5, 10, 20):
            spread = q5[f"spy_fwd_{horizon}d"].mean() - q1[f"spy_fwd_{horizon}d"].mean()
            var_spread = (q5[f"spy_fwd_{horizon}d"].var() / len(q5)) + (q1[f"spy_fwd_{horizon}d"].var() / len(q1))
            se_spread = np.sqrt(var_spread) if var_spread > 0 else 1e-9
            t = spread / se_spread
            print(f"  Q5 - Q1 spread @ {horizon}d: {spread*100:+.2f}%   t={t:+.2f}")

    # ── Test 4: Binary regime split ──
    print("\n" + "─" * 78)
    print("  TEST 4 — binary regime split at ratio = 1.0 (contango vs backwardation)")
    print("─" * 78)
    print(f"  Split by ratio_9d_30d > 1.0")
    bwd = df[df["ratio_9d_30d"] > 1.0]
    con = df[df["ratio_9d_30d"] <= 1.0]
    print(f"    backwardation days: {len(bwd)} ({len(bwd)/len(df):.0%})")
    print(f"    contango days:      {len(con)} ({len(con)/len(df):.0%})")
    for horizon in (5, 10, 20):
        bwd_mean = bwd[f"spy_fwd_{horizon}d"].mean() * 100
        con_mean = con[f"spy_fwd_{horizon}d"].mean() * 100
        diff = bwd_mean - con_mean
        print(f"    {horizon}d fwd:  backwardation={bwd_mean:+.2f}%   contango={con_mean:+.2f}%   diff={diff:+.2f}%")

    # ── Test 5: Current state ──
    print("\n" + "─" * 78)
    print("  TEST 5 — current term structure state")
    print("─" * 78)
    latest = df.iloc[-1]
    print(f"  as of {df.index[-1].date()}:")
    print(f"    VIX9D  = {latest['vix9d']:.2f}")
    print(f"    VIX    = {latest['vix']:.2f}")
    print(f"    VIX3M  = {latest['vix3m']:.2f}")
    print(f"    9d/30d = {latest['ratio_9d_30d']:.3f}   {'(backwardation)' if latest['ratio_9d_30d'] > 1 else '(contango)'}")
    print(f"    30d/3m = {latest['ratio_30d_3m']:.3f}   {'(backwardation)' if latest['ratio_30d_3m'] > 1 else '(contango)'}")

    print("\n" + "═" * 78)
    print("  INTERPRETATION")
    print("═" * 78)
    print("""
  How to read this:

  If Q5-Q1 spread at 10d or 20d is POSITIVE and t-stat > 2:
    → Term structure predicts forward returns in your data. This is a
      real risk-premium signal you can use as a position-sizing overlay:
        backwardation → scale up gross exposure
        contango      → scale down gross exposure
      Magnitude of the effect = order-of-magnitude your Sharpe can improve.

  If the spread is near zero:
    → Term structure does not predict SPY in your sample. Probably because
      your sample is short or post-2020 which has been weird. Still worth
      knowing — means one hypothesis dies.

  If Test 4 shows meaningful backwardation-vs-contango difference:
    → Simple binary signal. Easiest possible implementation: whenever
      VIX9D > VIX, scale gross exposure up 20%. No retraining required.
""")


if __name__ == "__main__":
    main()
