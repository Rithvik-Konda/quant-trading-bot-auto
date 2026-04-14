"""
autopsy.py — Post-hoc diagnostics on the committed baseline.

Reads:
  trades_v2.csv                       (every trade, one row)
  cache_backtester/position_lifecycle.csv   (daily snapshots per open position)

Runs 4 analyses:
  1. Stop loss autopsy — where is the -$162k leak coming from?
  2. 2025 failure autopsy — 121 trades, +0.1% vs SPY +17.7%. Why?
  3. Meanrev reality check — 357 trades, $65 avg. Is it even deploying capital?
  4. Regime transition timing — how late is the regime classifier reacting?

This is read-only. It touches no models, no caches, no configs. Safe to run
while noise_floor is training in the background.

Usage:
  cd ~/ai_trading_bot_v2
  python3.11 tools/autopsy.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRADES_CSV = REPO / "trades_v2.csv"
LIFECYCLE_CSV = REPO / "cache_backtester" / "position_lifecycle.csv"


def _hr(title: str) -> None:
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


def load_trades() -> pd.DataFrame:
    if not TRADES_CSV.exists():
        print(f"ERROR: {TRADES_CSV} not found. Run v2/backtester_v2.py first.")
        sys.exit(1)
    df = pd.read_csv(TRADES_CSV)
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["year"] = df["exit_date"].dt.year
    df["hold_days"] = (df["exit_date"] - df["entry_date"]).dt.days
    return df


def load_lifecycle() -> pd.DataFrame | None:
    if not LIFECYCLE_CSV.exists():
        return None
    df = pd.read_csv(LIFECYCLE_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    return df


# ── 1. Stop autopsy ──────────────────────────────────────────────────────────

def stop_autopsy(df: pd.DataFrame) -> None:
    _hr("1. STOP AUTOPSY — the $162k leak")
    stops = df[df["reason"] == "stop"].copy()
    if len(stops) == 0:
        print("  No stop trades found.")
        return

    total = stops["pnl"].sum()
    print(f"\n  195 stops | ${total:,.0f} total | {(stops['pnl']>0).mean():.0%} WR | avg ${stops['pnl'].mean():,.0f}")

    # Catastrophes vs paper cuts
    quintiles = stops["pnl"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(0)
    print(f"\n  PnL percentiles:")
    print(f"    p10 = ${quintiles[0.1]:>8,.0f}  (worst losses)")
    print(f"    p25 = ${quintiles[0.25]:>8,.0f}")
    print(f"    p50 = ${quintiles[0.5]:>8,.0f}")
    print(f"    p75 = ${quintiles[0.75]:>8,.0f}")
    print(f"    p90 = ${quintiles[0.9]:>8,.0f}")

    # How concentrated is the damage?
    sorted_pnl = stops["pnl"].sort_values()
    n = len(sorted_pnl)
    worst_10pct = sorted_pnl.head(max(1, n // 10)).sum()
    worst_25pct = sorted_pnl.head(max(1, n // 4)).sum()
    print(f"\n  Damage concentration:")
    print(f"    Worst 10% of stops ({n//10} trades) : ${worst_10pct:>10,.0f}  ({worst_10pct/total:.0%} of total)")
    print(f"    Worst 25% of stops ({n//4} trades) : ${worst_25pct:>10,.0f}  ({worst_25pct/total:.0%} of total)")

    # By regime
    if "regime" in stops.columns:
        print(f"\n  By regime entered:")
        for regime, grp in stops.groupby("regime"):
            wr = (grp["pnl"] > 0).mean()
            print(f"    {regime:<16} {len(grp):>4} stops  WR={wr:.0%}  avg=${grp['pnl'].mean():>8,.0f}  total=${grp['pnl'].sum():>10,.0f}")

    # By year
    print(f"\n  By year:")
    for year, grp in stops.groupby("year"):
        if pd.isna(year):
            continue
        print(f"    {int(year)}  {len(grp):>4} stops  avg=${grp['pnl'].mean():>8,.0f}  total=${grp['pnl'].sum():>10,.0f}")

    # Top 10 worst tickers (repeat offenders?)
    by_sym = stops.groupby("symbol")["pnl"].agg(["count", "sum", "mean"]).sort_values("sum").head(10)
    print(f"\n  Top 10 worst symbols (biggest PnL damage):")
    print(f"    {'symbol':<8} {'n':>4}  {'total':>12}  {'avg':>10}")
    for sym, row in by_sym.iterrows():
        print(f"    {sym:<8} {int(row['count']):>4}  ${row['sum']:>10,.0f}  ${row['mean']:>8,.0f}")

    # Repeat-offender count
    repeats = stops.groupby("symbol").size()
    multi = repeats[repeats >= 2]
    print(f"\n  Symbols stopped out ≥2 times: {len(multi)}  (total repeat stops: {multi.sum()})")
    print(f"  Symbols stopped out ≥3 times: {(repeats >= 3).sum()}")
    if (repeats >= 3).any():
        top_repeats = repeats[repeats >= 3].sort_values(ascending=False).head(10)
        print(f"    Top repeat offenders: " + ", ".join(f"{s}({n})" for s, n in top_repeats.items()))

    # Hold days distribution
    if "hold_days" in stops.columns:
        print(f"\n  Hold days at stop:")
        print(f"    median = {stops['hold_days'].median():.0f} days")
        print(f"    mean   = {stops['hold_days'].mean():.1f} days")
        stopped_within_3 = (stops["hold_days"] <= 3).sum()
        print(f"    stopped ≤3 days after entry: {stopped_within_3} ({stopped_within_3/len(stops):.0%})")


# ── 2. 2025 autopsy ──────────────────────────────────────────────────────────

def year_2025_autopsy(df: pd.DataFrame, lifecycle: pd.DataFrame | None) -> None:
    _hr("2. 2025 AUTOPSY — the $17.6 pt gap vs SPY")
    t25 = df[df["year"] == 2025].copy()
    if len(t25) == 0:
        print("  No 2025 trades found.")
        return

    total_pnl = t25["pnl"].sum()
    print(f"\n  {len(t25)} trades | total PnL ${total_pnl:,.0f} | WR {(t25['pnl']>0).mean():.0%} | avg ${t25['pnl'].mean():,.0f}")

    # By exit reason
    print(f"\n  Exit breakdown:")
    for reason, grp in t25.groupby("reason"):
        wr = (grp["pnl"] > 0).mean()
        print(f"    {reason:<20} {len(grp):>4}  WR={wr:.0%}  avg=${grp['pnl'].mean():>8,.0f}  total=${grp['pnl'].sum():>10,.0f}")

    # By engine
    if "engine" in t25.columns:
        print(f"\n  By engine:")
        for eng, grp in t25.groupby("engine"):
            wr = (grp["pnl"] > 0).mean()
            print(f"    {eng:<12} {len(grp):>4}  WR={wr:.0%}  avg=${grp['pnl'].mean():>8,.0f}  total=${grp['pnl'].sum():>10,.0f}")

    # By regime entered
    if "regime" in t25.columns:
        print(f"\n  By regime entered:")
        for regime, grp in t25.groupby("regime"):
            wr = (grp["pnl"] > 0).mean()
            print(f"    {regime:<16} {len(grp):>4}  WR={wr:.0%}  avg=${grp['pnl'].mean():>8,.0f}  total=${grp['pnl'].sum():>10,.0f}")

    # Best and worst trades of 2025
    print(f"\n  Top 5 winners:")
    for _, row in t25.nlargest(5, "pnl").iterrows():
        print(f"    {row['symbol']:<6} {row['entry_date'].date()} → {row['exit_date'].date()}  ${row['pnl']:>8,.0f}  ({row.get('reason','?')})")
    print(f"\n  Top 5 losers:")
    for _, row in t25.nsmallest(5, "pnl").iterrows():
        print(f"    {row['symbol']:<6} {row['entry_date'].date()} → {row['exit_date'].date()}  ${row['pnl']:>8,.0f}  ({row.get('reason','?')})")

    # Was the system actually in the market?
    if lifecycle is not None:
        lc25 = lifecycle[lifecycle["year"] == 2025]
        if len(lc25) > 0:
            days_with_positions = lc25["date"].nunique()
            unique_days_2025 = 252  # approx trading days
            avg_positions = lc25.groupby("date")["symbol"].nunique().mean()
            print(f"\n  Capital deployment:")
            print(f"    days with any open position: {days_with_positions}/~{unique_days_2025}")
            print(f"    avg open positions per day:  {avg_positions:.1f}")
            print(f"    regime mix of those days:")
            regime_mix = lc25.drop_duplicates("date")["regime"].value_counts()
            for r, n in regime_mix.items():
                print(f"      {r:<16} {n} days")


# ── 3. Meanrev reality check ─────────────────────────────────────────────────

def meanrev_reality_check(df: pd.DataFrame, lifecycle: pd.DataFrame | None) -> None:
    _hr("3. MEANREV REALITY CHECK — 357 trades, $65 avg")
    if "engine" not in df.columns:
        print("  No 'engine' column — cannot split by engine.")
        return
    mr = df[df["engine"] == "meanrev"].copy()
    if len(mr) == 0:
        print("  No meanrev trades found.")
        return

    total = mr["pnl"].sum()
    print(f"\n  {len(mr)} trades | total ${total:,.0f} | WR {(mr['pnl']>0).mean():.0%} | avg ${mr['pnl'].mean():,.0f}")

    # Distribution
    winners = mr[mr["pnl"] > 0]
    losers = mr[mr["pnl"] <= 0]
    print(f"\n  Winners : {len(winners)} trades  avg ${winners['pnl'].mean():>8,.0f}  total ${winners['pnl'].sum():>10,.0f}")
    print(f"  Losers  : {len(losers)} trades  avg ${losers['pnl'].mean():>8,.0f}  total ${losers['pnl'].sum():>10,.0f}")

    # By year
    print(f"\n  By year:")
    for year, grp in mr.groupby("year"):
        if pd.isna(year):
            continue
        print(f"    {int(year)}  {len(grp):>4} trades  avg=${grp['pnl'].mean():>8,.0f}  total=${grp['pnl'].sum():>9,.0f}")

    # By regime entered
    if "regime" in mr.columns:
        print(f"\n  By regime entered:")
        for regime, grp in mr.groupby("regime"):
            wr = (grp["pnl"] > 0).mean()
            print(f"    {regime:<16} {len(grp):>4}  WR={wr:.0%}  avg=${grp['pnl'].mean():>7,.0f}")

    # Hold distribution
    if "hold_days" in mr.columns:
        print(f"\n  Hold days: median={mr['hold_days'].median():.0f}  mean={mr['hold_days'].mean():.1f}  max={mr['hold_days'].max():.0f}")

    # Firing rate — how often does meanrev actually deploy during CHOPPY?
    if lifecycle is not None:
        choppy_days = lifecycle[lifecycle["regime"] == "CHOPPY"]["date"].nunique()
        mr_days_deployed = lifecycle[
            (lifecycle["regime"] == "CHOPPY") & (lifecycle["engine"] == "meanrev")
        ]["date"].nunique() if "engine" in lifecycle.columns else 0
        if choppy_days > 0:
            print(f"\n  Deployment during CHOPPY:")
            print(f"    CHOPPY days total                    : {choppy_days}")
            print(f"    CHOPPY days with meanrev position    : {mr_days_deployed}  ({mr_days_deployed/choppy_days:.0%})")


# ── 4. Regime transition timing ──────────────────────────────────────────────

def regime_timing(lifecycle: pd.DataFrame | None) -> None:
    _hr("4. REGIME TRANSITION TIMING")
    if lifecycle is None:
        print("  No position_lifecycle.csv. Skipping.")
        return
    if "regime" not in lifecycle.columns:
        print("  No regime column in lifecycle data.")
        return

    # Build per-day regime series from lifecycle
    daily = lifecycle.drop_duplicates("date")[["date", "regime"]].sort_values("date").reset_index(drop=True)
    print(f"\n  {len(daily)} days of regime data")

    # Count transitions
    daily["prev"] = daily["regime"].shift(1)
    trans = daily[(daily["regime"] != daily["prev"]) & daily["prev"].notna()]
    print(f"  {len(trans)} regime transitions over the full backtest")

    # Transition table
    trans_counts = trans.groupby(["prev", "regime"]).size().reset_index(name="count")
    print(f"\n  Transition matrix (count):")
    for _, row in trans_counts.iterrows():
        print(f"    {row['prev']:<16} → {row['regime']:<16}  {row['count']:>4}")

    # Duration of each regime spell
    daily["block"] = (daily["regime"] != daily["prev"]).cumsum()
    spells = daily.groupby(["block", "regime"])["date"].agg(["min", "max", "count"]).reset_index()
    spells.columns = ["block", "regime", "start", "end", "days"]
    print(f"\n  Average spell length by regime:")
    for regime, grp in spells.groupby("regime"):
        print(f"    {regime:<16} avg={grp['days'].mean():>5.1f} days  median={grp['days'].median():>4.0f}  max={grp['days'].max():>4.0f}  n_spells={len(grp)}")

    # When does it leave BEAR? (transition to CHOPPY or TRENDING_BULL)
    bear_exits = trans[trans["prev"] == "BEAR"]
    if len(bear_exits) > 0:
        print(f"\n  BEAR exits:")
        for _, row in bear_exits.iterrows():
            print(f"    {row['date'].date()}  BEAR → {row['regime']}")


def main():
    os.chdir(REPO)
    print("═" * 72)
    print("  POST-HOC AUTOPSY  —  baseline 16.91% OOS")
    print("═" * 72)
    df = load_trades()
    lifecycle = load_lifecycle()
    print(f"\n  trades: {len(df)}   lifecycle rows: {len(lifecycle) if lifecycle is not None else 0}")

    stop_autopsy(df)
    year_2025_autopsy(df, lifecycle)
    meanrev_reality_check(df, lifecycle)
    regime_timing(lifecycle)

    print("\n" + "═" * 72)
    print("  DONE. Paste this whole output back and we'll read it together.")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
