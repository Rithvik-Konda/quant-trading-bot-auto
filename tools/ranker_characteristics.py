"""
ranker_characteristics.py — What is the ranker actually trading?

The factor regression (factor_decomp.py) showed your system has:
  - +18% annualized alpha (t-stat 15, highly significant)
  - ZERO momentum factor loading
  - ZERO value, quality, size, investment, market loading
  - R² = 1.6% — 98% of variance is orthogonal to Fama-French+Mom

That's genuinely surprising. It means the ranker is NOT what Session 10
diagnosed it as. This script asks the complementary question by looking
at WHAT THE SYSTEM ACTUALLY HELD, not how its returns correlated.

METHOD (holdings-based attribution, Barra-style)

  For each momentum entry trade in trades_v2.csv, compute the entry-time
  characteristics of the held name:

    price_vs_sma20            — short-term trend
    price_vs_sma50
    price_vs_sma200
    ret_1d, ret_5d, ret_20d   — recent returns
    ret_60d, ret_120d, ret_252d — longer-term returns
    vol_20d, vol_60d          — realized vol
    dollar_vol_20d            — liquidity
    high_52w_proximity        — % below 52w high
    close                     — price level (stocks vs. penny stocks)

  Then compute the same characteristics for a RANDOM SAMPLE of universe
  names on the same dates, as a control. Difference = what your ranker
  systematically prefers.

HYPOTHESIS (pre-committed, Rick's session #14, 2026-04-13)

  The factor decomp killed the momentum-machine theory. My guess is the
  ranker is picking up one of:

    H1: Short-term reversal — dips within broader uptrends
        (held names have NEGATIVE recent returns, POSITIVE long-term)
    H2: Earnings-drift residual — post-surprise continuation
        (no specific signature, but held names cluster near earnings dates)
    H3: Liquidity & quality tilt — biggest most-liquid names
        (held names have HIGHER dollar volume, HIGHER price level)
    H4: Something I haven't guessed

NO BACKFITTING
  We are describing population differences. No thresholds are chosen.
  We are not optimizing anything. We compute Cohen's d between held-names
  and random universe names, report it.

Usage:
  python3.11 tools/ranker_characteristics.py
"""

from __future__ import annotations
import os
import random
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRADES = REPO / "trades_v2.csv"
PRICES_DIR = REPO / "cache_prices"

random.seed(42)
np.random.seed(42)


def _load_price(sym: str) -> pd.DataFrame | None:
    p = PRICES_DIR / f"{sym}_3650d.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.loc[~df.index.isna()]
        if getattr(df.index, "tz", None):
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index = df.index.normalize()
        df.columns = [str(c).lower() for c in df.columns]
        return df
    except Exception:
        return None


def compute_characteristics(sym: str, as_of: pd.Timestamp, cache: dict) -> dict | None:
    """
    Compute entry-time characteristics for a symbol at a given date.
    Uses cached price data only.
    """
    if sym not in cache:
        cache[sym] = _load_price(sym)
    df = cache[sym]
    if df is None or "close" not in df.columns:
        return None
    df = df.loc[df.index <= as_of]
    if len(df) < 260:
        return None

    c = df["close"]
    last = float(c.iloc[-1])
    if last <= 0:
        return None

    out: dict = {}
    out["close"] = last

    # SMA ratios
    if len(c) >= 20:  out["vs_sma20"]  = last / float(c.tail(20).mean())  - 1
    if len(c) >= 50:  out["vs_sma50"]  = last / float(c.tail(50).mean())  - 1
    if len(c) >= 200: out["vs_sma200"] = last / float(c.tail(200).mean()) - 1

    # Trailing returns
    if len(c) >= 2:   out["ret_1d"]   = last / float(c.iloc[-2])   - 1
    if len(c) >= 6:   out["ret_5d"]   = last / float(c.iloc[-6])   - 1
    if len(c) >= 21:  out["ret_20d"]  = last / float(c.iloc[-21])  - 1
    if len(c) >= 61:  out["ret_60d"]  = last / float(c.iloc[-61])  - 1
    if len(c) >= 121: out["ret_120d"] = last / float(c.iloc[-121]) - 1
    if len(c) >= 253: out["ret_252d"] = last / float(c.iloc[-253]) - 1

    # FF-style momentum factor: 12-month-minus-1-month return
    if len(c) >= 253:
        ret_252 = last / float(c.iloc[-253]) - 1
        ret_21 = last / float(c.iloc[-21]) - 1 if len(c) >= 21 else 0
        out["mom_12_1"] = (1 + ret_252) / (1 + ret_21) - 1

    # Realized vol
    rets = c.pct_change().dropna()
    if len(rets) >= 20:
        out["vol_20d"] = float(rets.tail(20).std()) * np.sqrt(252)
    if len(rets) >= 60:
        out["vol_60d"] = float(rets.tail(60).std()) * np.sqrt(252)

    # 52-week high proximity
    if len(c) >= 252:
        hi_52 = float(c.tail(252).max())
        if hi_52 > 0:
            out["dist_from_52w_hi"] = last / hi_52 - 1

    # Dollar volume (liquidity proxy)
    if "volume" in df.columns:
        v = df["volume"].dropna()
        if len(v) >= 20:
            avg_vol_20d = float(v.tail(20).mean())
            out["dollar_vol_20d"] = avg_vol_20d * last

    # Price level buckets (penny vs large)
    out["log_close"] = float(np.log(max(last, 0.01)))

    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0, float(a.mean()) if len(a) else np.nan, float(b.mean()) if len(b) else np.nan
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((len(a)-1)*sa**2 + (len(b)-1)*sb**2) / (len(a)+len(b)-2))
    d = (ma - mb) / pooled if pooled > 0 else 0.0
    return float(d), float(ma), float(mb)


def main():
    os.chdir(REPO)
    print("═" * 78)
    print("  RANKER HOLDINGS CHARACTERISTICS")
    print("  Pre-committed hypothesis: the ranker is trading short-term reversal")
    print("  within broader uptrends (negative 1m ret + positive 12m ret)")
    print("═" * 78)

    trades = pd.read_csv(TRADES)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades = trades.dropna(subset=["entry_date", "symbol"])

    # Filter to momentum engine only (meanrev has obviously different dynamics)
    if "engine" in trades.columns:
        mom = trades[trades["engine"] == "momentum"].copy()
    else:
        mom = trades.copy()
    print(f"\n  {len(mom)} momentum trades")

    # Universe = every symbol that appears anywhere in trades (proxy for your WATCHLIST)
    # Plus any price CSV in cache_prices
    universe = set()
    for p in PRICES_DIR.glob("*_3650d.csv"):
        # Parse "SYM_3650d.csv"
        name = p.name
        if name.endswith("_3650d.csv") and "regime" not in name:
            sym = name[:-len("_3650d.csv")]
            universe.add(sym)
    print(f"  universe: {len(universe)} symbols with price caches")

    # Build the two populations:
    #   HELD    — characteristics of the actual held name on entry_date
    #   CONTROL — characteristics of a RANDOM universe symbol on the same entry_date
    print(f"\n  computing characteristics for held names...")
    cache: dict = {}
    held_rows: list = []
    for _, r in mom.iterrows():
        sym = str(r["symbol"]).upper()
        dt = pd.Timestamp(r["entry_date"]).normalize()
        c = compute_characteristics(sym, dt, cache)
        if c is None:
            continue
        c["date"] = dt
        held_rows.append(c)
    held = pd.DataFrame(held_rows)
    print(f"    {len(held)} held-name observations")

    print(f"\n  computing characteristics for random-universe controls on same dates...")
    control_rows: list = []
    unique_dates = sorted(mom["entry_date"].dt.normalize().unique())
    universe_list = sorted(universe)
    # For each entry date, sample N=5 random universe names as controls
    for dt in unique_dates:
        sample = random.sample(universe_list, min(5, len(universe_list)))
        for sym in sample:
            c = compute_characteristics(sym, dt, cache)
            if c is None:
                continue
            c["date"] = dt
            control_rows.append(c)
    control = pd.DataFrame(control_rows)
    print(f"    {len(control)} control observations")

    # Compute Cohen's d for each characteristic
    features = [
        "vs_sma20", "vs_sma50", "vs_sma200",
        "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d", "ret_252d",
        "mom_12_1",
        "vol_20d", "vol_60d",
        "dist_from_52w_hi",
        "dollar_vol_20d",
        "log_close",
    ]

    rows = []
    for f in features:
        if f not in held.columns or f not in control.columns:
            continue
        h = held[f].to_numpy(dtype=float)
        c = control[f].to_numpy(dtype=float)
        d, mh, mc = cohens_d(h, c)
        rows.append({"feature": f, "held_mean": mh, "control_mean": mc, "diff": mh - mc, "cohens_d": d, "abs_d": abs(d)})
    out = pd.DataFrame(rows).sort_values("abs_d", ascending=False)

    print("\n" + "─" * 78)
    print("  FEATURE TILT — held names vs. random universe (same dates)")
    print("─" * 78)
    print(f"\n  {'feature':<22} {'held':>12} {'universe':>12} {'diff':>12} {'d':>8}")
    print("  " + "─" * 68)
    for _, r in out.iterrows():
        if not np.isfinite(r["cohens_d"]):
            continue
        flag = ""
        if abs(r["cohens_d"]) >= 0.5:
            flag = " ★★★"
        elif abs(r["cohens_d"]) >= 0.3:
            flag = " ★★"
        elif abs(r["cohens_d"]) >= 0.2:
            flag = " ★"
        # Format differently for big vs small numbers
        if "vol" in r["feature"] or "dollar" in r["feature"]:
            print(f"  {r['feature']:<22} {r['held_mean']:>12.4f} {r['control_mean']:>12.4f} {r['diff']:>+12.4f} {r['cohens_d']:>+8.2f}{flag}")
        else:
            print(f"  {r['feature']:<22} {r['held_mean']:>+11.4f} {r['control_mean']:>+11.4f} {r['diff']:>+12.4f} {r['cohens_d']:>+8.2f}{flag}")

    # Hypothesis scoring
    print("\n" + "─" * 78)
    print("  HYPOTHESIS SCORECARD  (pre-committed before seeing the data)")
    print("─" * 78)

    ret_20d_d = out[out["feature"] == "ret_20d"]["cohens_d"].iloc[0] if "ret_20d" in out["feature"].values else 0
    ret_252d_d = out[out["feature"] == "ret_252d"]["cohens_d"].iloc[0] if "ret_252d" in out["feature"].values else 0
    mom_12_1_d = out[out["feature"] == "mom_12_1"]["cohens_d"].iloc[0] if "mom_12_1" in out["feature"].values else 0
    vol_20d_d = out[out["feature"] == "vol_20d"]["cohens_d"].iloc[0] if "vol_20d" in out["feature"].values else 0
    dol_d = out[out["feature"] == "dollar_vol_20d"]["cohens_d"].iloc[0] if "dollar_vol_20d" in out["feature"].values else 0
    dist_hi_d = out[out["feature"] == "dist_from_52w_hi"]["cohens_d"].iloc[0] if "dist_from_52w_hi" in out["feature"].values else 0
    vs_sma200_d = out[out["feature"] == "vs_sma200"]["cohens_d"].iloc[0] if "vs_sma200" in out["feature"].values else 0

    print()
    print(f"  H1: short-term reversal (need: ret_20d NEGATIVE d, ret_252d POSITIVE d, mom_12_1 POSITIVE d)")
    print(f"      ret_20d d   = {ret_20d_d:+.2f}  {'✓' if ret_20d_d < -0.1 else '✗'}")
    print(f"      ret_252d d  = {ret_252d_d:+.2f}  {'✓' if ret_252d_d > +0.1 else '✗'}")
    print(f"      mom_12_1 d  = {mom_12_1_d:+.2f}  {'✓' if mom_12_1_d > +0.1 else '✗'}")

    print()
    print(f"  H3: liquidity/quality tilt (need: dollar_vol d BIG POSITIVE)")
    print(f"      dollar_vol d = {dol_d:+.2f}  {'✓' if dol_d > +0.3 else '✗'}")

    print()
    print(f"  BONUS: 'classic momentum chasing' test")
    print(f"  (need: vs_sma200 STRONGLY POSITIVE, dist_from_52w_hi NEAR ZERO, vol_20d NEUTRAL)")
    print(f"      vs_sma200 d       = {vs_sma200_d:+.2f}  {'✓' if vs_sma200_d > +0.3 else '✗'}")
    print(f"      dist_from_52w_hi  = {dist_hi_d:+.2f}  {'✓' if dist_hi_d > +0.3 else '✗'}")
    print(f"      vol_20d d         = {vol_20d_d:+.2f}")

    print("\n" + "═" * 78)
    print("  INTERPRETATION")
    print("═" * 78)
    print("""
  Three things to read off the output:

  1. The top 2-3 features by |d| tell you what the ranker systematically
     prefers. These are the 'characteristic tilts' of your system.

  2. Sign of the tilt matters — a positive d means held names have HIGHER
     value of that feature than random universe controls on the same date.

  3. Check the hypothesis scorecard above. If none of the named hypotheses
     match cleanly, the ranker is picking up something else — probably
     a combination of features from the 387-feature set that doesn't map
     onto any one characteristic I named. Which would be consistent with
     the factor regression result (18% alpha not explained by standard factors).

  IMPORTANT: this analysis isolates 'what's different about held names at
  entry vs. random names.' It does NOT tell you WHY those names were
  profitable. The ranker could be picking names with these characteristics
  for reasons we can't see directly in OHLCV (the 387 features include
  SI velocity, earnings streak, options IV term, etc).
""")


if __name__ == "__main__":
    main()
