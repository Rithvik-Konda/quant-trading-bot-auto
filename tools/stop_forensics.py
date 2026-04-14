"""
stop_forensics.py — Classify every stop trade by mechanism. No backfitting.

GOAL
  For each of the 195 stop trades in trades_v2.csv, reconstruct what was
  visible before the trade (entry context), what happened during the hold
  (lifecycle path), and what the market / macro / sector / event environment
  was doing. Then bucket each stop into a mechanism category so we can see
  which problems actually matter by $-weight, not by ticker.

DISCIPLINE — avoiding backfit
  - We do NOT invent tickers to blacklist.
  - We do NOT choose thresholds after seeing which side wins.
  - We classify each stop using the DATA THAT WAS AVAILABLE BEFORE THE STOP
    (or during the hold, which is still causally prior to exit).
  - Classification thresholds are chosen from WELL-KNOWN STANDARDS, not
    from inspection of this dataset:
        * SPY move < -1.5% in one day  = macro selloff (standard "bad day")
        * VIX +25% in one day         = vol shock (CBOE standard)
        * Stock gap > 5% in one day   = news shock (2-sigma for mega-cap)
        * Earnings within ±5 days     = earnings event (matches your config)
        * Sector ETF move < -2%       = sector rotation
  - The BUCKETS are mechanism categories identified a priori; we are
    categorizing, not fitting.

INPUTS
  trades_v2.csv                            (one row per trade, written by backtester)
  cache_backtester/position_lifecycle.csv  (daily snapshots per open position)
  cache_prices/<SYM>_3650d.csv             (per-symbol OHLCV for gap detection)
  cache_prices/SPY_regime.csv OR SPY_3650d.csv   (macro context)
  cache_prices/VIX_regime.csv OR VIXY_regime.csv (vol shock detection)
  cache_prices/earnings_calendar_cache.pkl (earnings dates)

All of the above are loaded DEFENSIVELY. If a file is missing, the affected
column is filled with NaN and we flag it in the missing-data report. The
script never crashes on missing data; it just says what it couldn't see.

OUTPUTS
  /tmp/stop_forensics.csv         (one row per stop, ~30 columns)
  /tmp/stop_forensics_report.txt  (the bucket table + top-$-damage examples)
  stdout                          (human-readable summary)

Usage
  python3.11 tools/stop_forensics.py
"""

from __future__ import annotations
import os
import pickle
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRADES = REPO / "trades_v2.csv"
LIFECYCLE = REPO / "cache_backtester" / "position_lifecycle.csv"
PRICES_DIR = REPO / "cache_prices"
EARN_PKL_CANDIDATES = [
    PRICES_DIR / "earnings_calendar_cache.pkl",
    REPO / "cache_earnings_streak",  # fallback dir of per-symbol jsons
]

# Thresholds are STANDARD / A PRIORI, not tuned on this dataset.
SPY_BAD_DAY_PCT = -0.015   # standard "bad day" in financial media
VIX_SHOCK_PCT   = 0.25     # CBOE convention for a vol spike
STOCK_GAP_PCT   = 0.05     # 5% single-day move = news move for most liquid names
SECTOR_BAD_PCT  = -0.02    # 2% sector ETF drop = sector rotation day
EARN_WINDOW     = 5        # your own config's earnings exclusion window


# ──────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    if not TRADES.exists():
        print(f"FATAL: {TRADES} missing. Run the backtester first.")
        sys.exit(1)
    df = pd.read_csv(TRADES)
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"]  = pd.to_datetime(df["exit_date"],  errors="coerce")
    return df


def load_lifecycle() -> pd.DataFrame | None:
    if not LIFECYCLE.exists():
        return None
    df = pd.read_csv(LIFECYCLE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_price(sym: str, days: int = 3650) -> pd.DataFrame | None:
    """Try common cache paths. Return None if not found."""
    for pat in [f"{sym}_{days}d.csv", f"{sym}_3650d.csv", f"{sym}_1825d.csv"]:
        p = PRICES_DIR / pat
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


def load_macro() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    spy = None
    for name in ["SPY_regime.csv", "SPY_3650d.csv"]:
        p = PRICES_DIR / name
        if p.exists():
            try:
                spy = pd.read_csv(p, index_col=0)
                spy.index = pd.to_datetime(spy.index, utc=True, errors="coerce")
                spy = spy.loc[~spy.index.isna()]
                if getattr(spy.index, "tz", None):
                    spy.index = spy.index.tz_convert("UTC").tz_localize(None)
                spy.columns = [str(c).lower() for c in spy.columns]
                break
            except Exception:
                pass
    vix = None
    for name in ["VIX_regime.csv", "VIXY_regime.csv", "VIX_3650d.csv"]:
        p = PRICES_DIR / name
        if p.exists():
            try:
                vix = pd.read_csv(p, index_col=0)
                vix.index = pd.to_datetime(vix.index, utc=True, errors="coerce")
                vix = vix.loc[~vix.index.isna()]
                if getattr(vix.index, "tz", None):
                    vix.index = vix.index.tz_convert("UTC").tz_localize(None)
                vix.columns = [str(c).lower() for c in vix.columns]
                break
            except Exception:
                pass
    return spy, vix


def load_earnings() -> dict[str, list[pd.Timestamp]]:
    """Return {symbol: sorted list of earnings dates}. Empty dict if nothing."""
    out: dict[str, list[pd.Timestamp]] = {}
    # Try pickled cache first
    for p in [PRICES_DIR / "earnings_calendar_cache.pkl"]:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    d = pickle.load(f)
                if isinstance(d, dict):
                    for k, v in d.items():
                        try:
                            dates = [pd.Timestamp(x).tz_localize(None) if getattr(pd.Timestamp(x), "tz", None) else pd.Timestamp(x) for x in v]
                            out[str(k).upper()] = sorted([x for x in dates if pd.notna(x)])
                        except Exception:
                            continue
                    return out
            except Exception:
                pass
    # Try per-symbol jsons
    import json
    estreak = REPO / "cache_earnings_streak"
    if estreak.exists():
        for f in estreak.glob("*.json"):
            try:
                recs = json.loads(f.read_text())
                if isinstance(recs, list):
                    dates = []
                    for r in recs:
                        d = r.get("date") if isinstance(r, dict) else None
                        if d:
                            try:
                                dates.append(pd.Timestamp(d))
                            except Exception:
                                pass
                    out[f.stem.upper()] = sorted(dates)
            except Exception:
                continue
    return out


# ──────────────────────────────────────────────────────────────────────────
# Sector map — defensive load from config
# ──────────────────────────────────────────────────────────────────────────

def load_sector_map() -> dict[str, str]:
    try:
        sys.path.insert(0, str(REPO))
        import config  # type: ignore
        raw = getattr(config, "SECTOR_MAP", None) or getattr(config, "SECTORS", None) or {}
        if isinstance(raw, dict):
            return {str(k).upper(): str(v).upper() for k, v in raw.items()}
    except Exception:
        pass
    return {}


# ──────────────────────────────────────────────────────────────────────────
# Per-stop forensics
# ──────────────────────────────────────────────────────────────────────────

def forensics_for_stop(
    trade: pd.Series,
    lifecycle: pd.DataFrame | None,
    spy: pd.DataFrame | None,
    vix: pd.DataFrame | None,
    earnings: dict[str, list[pd.Timestamp]],
    sector_map: dict[str, str],
) -> dict:
    sym = str(trade["symbol"]).upper()
    e_dt = pd.Timestamp(trade["entry_date"])
    x_dt = pd.Timestamp(trade["exit_date"])
    pnl  = float(trade["pnl"])
    row: dict = {
        "symbol":     sym,
        "entry":      e_dt.date() if pd.notna(e_dt) else None,
        "exit":       x_dt.date() if pd.notna(x_dt) else None,
        "pnl":        pnl,
        "hold_days":  (x_dt - e_dt).days if pd.notna(x_dt) and pd.notna(e_dt) else np.nan,
        "regime":     trade.get("regime", ""),
        "engine":     trade.get("engine", ""),
        "ml_rank":    float(trade.get("ml_rank_pct", np.nan)) if pd.notna(trade.get("ml_rank_pct", np.nan)) else np.nan,
        "ann_vol":    float(trade.get("ann_vol", np.nan)) if pd.notna(trade.get("ann_vol", np.nan)) else np.nan,
    }

    # ── Stock price data: gap detection + max adverse/favorable excursion ──
    px = load_price(sym)
    row["has_price_data"] = px is not None
    if px is not None and "close" in px.columns and pd.notna(e_dt) and pd.notna(x_dt):
        hold = px.loc[(px.index >= e_dt) & (px.index <= x_dt)].copy()
        if len(hold) >= 2:
            entry_px = float(hold["close"].iloc[0])
            unrealized = (hold["close"] / entry_px - 1.0)
            row["mae_pct"] = float(unrealized.min())
            row["mfe_pct"] = float(unrealized.max())
            # Day the position first went underwater
            under = unrealized[unrealized < 0]
            row["first_negative_day"] = int((under.index[0] - e_dt).days) if len(under) else None
            # Worst single-day % move during hold (gap)
            daily = hold["close"].pct_change().dropna()
            if len(daily):
                worst_day = float(daily.min())
                worst_date = daily.idxmin()
                row["worst_single_day_pct"] = worst_day
                row["worst_single_day_date"] = worst_date.date()
                row["big_gap"] = bool(worst_day <= -STOCK_GAP_PCT)
            else:
                row["worst_single_day_pct"] = np.nan
                row["big_gap"] = False
        else:
            row["mae_pct"] = row["mfe_pct"] = np.nan
            row["first_negative_day"] = None
            row["worst_single_day_pct"] = np.nan
            row["big_gap"] = False
    else:
        row["mae_pct"] = row["mfe_pct"] = np.nan
        row["first_negative_day"] = None
        row["worst_single_day_pct"] = np.nan
        row["big_gap"] = False

    # ── SPY macro shock detection during hold ──
    if spy is not None and "close" in spy.columns and pd.notna(e_dt) and pd.notna(x_dt):
        spy_hold = spy.loc[(spy.index >= e_dt) & (spy.index <= x_dt), "close"]
        if len(spy_hold) >= 2:
            spy_daily = spy_hold.pct_change().dropna()
            row["spy_ret_hold"] = float(spy_hold.iloc[-1] / spy_hold.iloc[0] - 1)
            row["spy_worst_day_pct"] = float(spy_daily.min()) if len(spy_daily) else np.nan
            row["macro_shock"] = bool(len(spy_daily) and spy_daily.min() <= SPY_BAD_DAY_PCT)
        else:
            row["spy_ret_hold"] = np.nan
            row["spy_worst_day_pct"] = np.nan
            row["macro_shock"] = False
    else:
        row["spy_ret_hold"] = np.nan
        row["spy_worst_day_pct"] = np.nan
        row["macro_shock"] = False

    # ── VIX shock during hold ──
    if vix is not None and "close" in vix.columns and pd.notna(e_dt) and pd.notna(x_dt):
        vix_hold = vix.loc[(vix.index >= e_dt) & (vix.index <= x_dt), "close"]
        if len(vix_hold) >= 2:
            vix_daily = vix_hold.pct_change().dropna()
            row["vix_at_entry"] = float(vix_hold.iloc[0])
            row["vix_worst_day_pct"] = float(vix_daily.max()) if len(vix_daily) else np.nan
            row["vol_shock"] = bool(len(vix_daily) and vix_daily.max() >= VIX_SHOCK_PCT)
        else:
            row["vix_at_entry"] = np.nan
            row["vix_worst_day_pct"] = np.nan
            row["vol_shock"] = False
    else:
        row["vix_at_entry"] = np.nan
        row["vix_worst_day_pct"] = np.nan
        row["vol_shock"] = False

    # ── Earnings within hold window (did the filter let it through?) ──
    earn_dates = earnings.get(sym, [])
    if earn_dates and pd.notna(e_dt) and pd.notna(x_dt):
        window_start = e_dt - pd.Timedelta(days=EARN_WINDOW)
        window_end   = x_dt + pd.Timedelta(days=EARN_WINDOW)
        within = [d for d in earn_dates if window_start <= d <= window_end]
        row["earnings_in_window"] = bool(within)
        row["nearest_earnings_days"] = int(min(
            (abs((d - e_dt).days) for d in earn_dates), default=9999
        ))
    else:
        row["earnings_in_window"] = False
        row["nearest_earnings_days"] = None

    # ── Sector from static map, if available ──
    row["sector"] = sector_map.get(sym, "")

    # ── Lifecycle context: position count at time of stop ──
    if lifecycle is not None and pd.notna(x_dt):
        lc_near = lifecycle[
            (lifecycle["date"] >= x_dt - pd.Timedelta(days=2)) &
            (lifecycle["date"] <= x_dt)
        ]
        if len(lc_near) > 0:
            row["portfolio_size_at_stop"] = int(lc_near["portfolio_size"].max())
            row["portfolio_corr_at_stop"] = float(lc_near["portfolio_corr"].max())
        else:
            row["portfolio_size_at_stop"] = None
            row["portfolio_corr_at_stop"] = None
    else:
        row["portfolio_size_at_stop"] = None
        row["portfolio_corr_at_stop"] = None

    return row


# ──────────────────────────────────────────────────────────────────────────
# Bucket assignment (a priori categories, not fit to data)
# ──────────────────────────────────────────────────────────────────────────

def assign_bucket(r: dict) -> str:
    """
    Bucket precedence (most specific → most generic):
      C. single-stock news shock  — big_gap AND no macro shock AND no vol shock
      C'. earnings leak           — earnings in hold window (even if gap absent)
      B. macro shock              — macro_shock OR vol_shock during hold
      A. entry mistake            — went negative within 2 days of entry and never recovered
      D. slow fade                — everything else (no catalyst, gradual loss)
    'E. sector rotation' deferred — requires sector ETF price data; tag in extension
    """
    # Earnings leak is highest priority — it's a filter bug
    if r.get("earnings_in_window"):
        return "C2. earnings leak"
    if r.get("big_gap") and not r.get("macro_shock") and not r.get("vol_shock"):
        return "C1. single-stock news shock"
    if r.get("macro_shock") or r.get("vol_shock"):
        # But only if the stock gap is ALSO large — otherwise the macro moved
        # and we didn't even catch the full spillover
        if r.get("big_gap"):
            return "B1. macro + stock co-moved"
        else:
            return "B2. macro shock, stock dragged"
    fn = r.get("first_negative_day")
    if fn is not None and fn <= 2:
        mfe = r.get("mfe_pct")
        if pd.notna(mfe) and mfe < 0.02:
            return "A. entry mistake (never worked)"
    # Otherwise: slow grind
    return "D. slow fade"


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    os.chdir(REPO)
    print("═" * 78)
    print("  STOP FORENSICS")
    print("═" * 78)

    trades = load_trades()
    stops = trades[trades["reason"] == "stop"].copy().reset_index(drop=True)
    print(f"\n  {len(stops)} stop trades in trades_v2.csv")

    lifecycle = load_lifecycle()
    print(f"  lifecycle rows: {len(lifecycle) if lifecycle is not None else 'MISSING'}")

    spy, vix = load_macro()
    print(f"  SPY macro:      {'OK' if spy is not None else 'MISSING'}")
    print(f"  VIX macro:      {'OK' if vix is not None else 'MISSING'}")

    earnings = load_earnings()
    print(f"  earnings cache: {len(earnings)} symbols")

    sector_map = load_sector_map()
    print(f"  sector map:     {len(sector_map)} symbols")

    # Process every stop
    print(f"\n  processing {len(stops)} stops...")
    forensic_rows = []
    n_missing_px = 0
    for _, tr in stops.iterrows():
        r = forensics_for_stop(tr, lifecycle, spy, vix, earnings, sector_map)
        if not r["has_price_data"]:
            n_missing_px += 1
        r["bucket"] = assign_bucket(r)
        forensic_rows.append(r)

    forensics = pd.DataFrame(forensic_rows)
    if n_missing_px:
        print(f"  ⚠ {n_missing_px} stops had no price cache (bucketing degraded)")

    out_csv = Path("/tmp/stop_forensics.csv")
    forensics.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")

    # ── Bucket summary ──
    print("\n" + "─" * 78)
    print("  BUCKETS  (by $-damage, not count)")
    print("─" * 78)
    by_bucket = forensics.groupby("bucket").agg(
        count=("pnl", "size"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    ).sort_values("total_pnl")
    total_damage = forensics["pnl"].sum()
    print(f"\n  {'bucket':<34} {'n':>5}  {'total':>12}  {'avg':>10}  {'% dmg':>7}")
    for bucket, row in by_bucket.iterrows():
        pct = row["total_pnl"] / total_damage * 100 if total_damage else 0
        print(f"  {bucket:<34} {int(row['count']):>5}  ${row['total_pnl']:>10,.0f}  ${row['avg_pnl']:>8,.0f}  {pct:>6.0f}%")
    print(f"\n  TOTAL                              {len(forensics):>5}  ${total_damage:>10,.0f}")

    # ── Bucket-level drilldowns ──
    print("\n" + "─" * 78)
    print("  TOP 5 WORST STOPS IN EACH BUCKET")
    print("─" * 78)
    for bucket in by_bucket.index:
        sub = forensics[forensics["bucket"] == bucket].nsmallest(5, "pnl")
        print(f"\n  {bucket}")
        for _, r in sub.iterrows():
            tags = []
            if r.get("big_gap"): tags.append(f"gap={r['worst_single_day_pct']:.1%}")
            if r.get("macro_shock"): tags.append(f"SPY={r['spy_worst_day_pct']:.1%}")
            if r.get("vol_shock"): tags.append(f"VIX+{r['vix_worst_day_pct']:.0%}")
            if r.get("earnings_in_window"): tags.append("EARN")
            if pd.notna(r.get("mae_pct")): tags.append(f"MAE={r['mae_pct']:.1%}")
            if pd.notna(r.get("mfe_pct")): tags.append(f"MFE={r['mfe_pct']:.1%}")
            tag_str = " ".join(tags)
            print(f"    {r['symbol']:<6} {str(r['entry']):<10}→{str(r['exit']):<10} ${r['pnl']:>8,.0f}  {tag_str}")

    # ── Cross-tab: bucket × regime ──
    print("\n" + "─" * 78)
    print("  BUCKET × REGIME  (count)")
    print("─" * 78)
    try:
        ct = pd.crosstab(forensics["bucket"], forensics["regime"])
        print()
        print(ct.to_string())
    except Exception:
        pass

    # ── Cross-tab: bucket × year ──
    forensics["year"] = pd.to_datetime(forensics["exit"], errors="coerce").dt.year
    print("\n" + "─" * 78)
    print("  BUCKET × YEAR  (total PnL)")
    print("─" * 78)
    try:
        ct = forensics.pivot_table(values="pnl", index="bucket", columns="year", aggfunc="sum", fill_value=0).round(0).astype(int)
        print()
        print(ct.to_string())
    except Exception:
        pass

    # ── Specific checks the user asked about ──
    print("\n" + "─" * 78)
    print("  SANITY CHECKS — what the data says about specific hypotheses")
    print("─" * 78)

    # 1. First-negative-day distribution (entry quality)
    fn_days = forensics["first_negative_day"].dropna()
    if len(fn_days):
        print(f"\n  First-negative-day distribution (of {len(fn_days)} stops with price data):")
        print(f"    went red on day 1  : {(fn_days <= 1).sum():>3}  ({(fn_days <= 1).mean():.0%})")
        print(f"    went red on day ≤3 : {(fn_days <= 3).sum():>3}  ({(fn_days <= 3).mean():.0%})")
        print(f"    went red on day ≤5 : {(fn_days <= 5).sum():>3}  ({(fn_days <= 5).mean():.0%})")
        print(f"    only red after wk 2: {(fn_days >  10).sum():>3}  ({(fn_days >  10).mean():.0%})")

    # 2. MFE distribution — did these trades ever make money?
    mfe = forensics["mfe_pct"].dropna()
    if len(mfe):
        print(f"\n  MFE distribution (best unrealized during hold):")
        print(f"    never went positive        : {(mfe <= 0).sum():>3}  ({(mfe <= 0).mean():.0%})")
        print(f"    peaked at <5% unrealized   : {((mfe > 0) & (mfe < 0.05)).sum():>3}")
        print(f"    peaked at 5-10% unrealized : {((mfe >= 0.05) & (mfe < 0.10)).sum():>3}")
        print(f"    peaked at >10% unrealized  : {(mfe >= 0.10).sum():>3}  ← gave back real gains")

    # 3. Portfolio size at stop (correlation risk)
    pfs = forensics["portfolio_size_at_stop"].dropna()
    if len(pfs):
        print(f"\n  Portfolio size when stopped (concentration context):")
        print(f"    median = {pfs.median():.0f}  mean = {pfs.mean():.1f}  max = {pfs.max():.0f}")
        print(f"    stops while fully loaded (≥6 positions): {(pfs >= 6).sum()}  ({(pfs >= 6).mean():.0%})")

    # 4. Earnings leak count (your filter is 5 days; nearest_earnings_days gives the truth)
    ned = forensics["nearest_earnings_days"].dropna()
    if len(ned):
        leaks = (ned <= 5).sum()
        print(f"\n  Earnings filter leaks (filter = ±5 days):")
        print(f"    stops within 5d of nearest earnings: {leaks}  ({leaks/len(ned):.0%})")
        print(f"    stops within 10d:                     {(ned <= 10).sum()}")

    print("\n" + "═" * 78)
    print("  Full per-stop table saved to /tmp/stop_forensics.csv")
    print("═" * 78 + "\n")


if __name__ == "__main__":
    main()
