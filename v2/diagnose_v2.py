"""
diagnose_v2.py — V2 Diagnostic Script
======================================
Run this from ~/ai_trading_bot_v2/v2/

  cd ~/ai_trading_bot_v2/v2
  /opt/homebrew/bin/python3.11 diagnose_v2.py

Answers three questions before we touch any code:
  Q1: 2018 drawdown anatomy — was it a few big positions or death by 1000 cuts?
       Would a 10% circuit breaker have actually saved it?
  Q2: 2023 regime day-count — how many days was it calling CHOPPY vs TRENDING_BULL?
       Is the choppy threshold fix masking a regime error?
  Q3: Short stop vs max_hold breakdown by year — confirm the diagnosis
       and check if widening stops helps or just moves the problem.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# ── Run the backtest and capture trades + equity ──────────────────────────────
print("Running v2 backtest (this will take a few minutes)...")
from backtester_v2 import run_backtest_v2
equity, trades, stats = run_backtest_v2(days=3650)

# Save trades to CSV for future sessions
trade_df = pd.DataFrame([t.__dict__ for t in trades])
trade_df.to_csv("trades_v2.csv", index=False)
print(f"\nSaved {len(trade_df)} trades to trades_v2.csv")

# Save equity curve too
equity.to_csv("equity_v2.csv")
print(f"Saved equity curve to equity_v2.csv")

# ── Helper: reconstruct daily portfolio value with drawdown ───────────────────
def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak

# ── Q1: 2018 Drawdown Anatomy ─────────────────────────────────────────────────
print("\n" + "="*60)
print("Q1: 2018 DRAWDOWN ANATOMY")
print("="*60)

dd = compute_drawdown_series(equity)
eq_2018 = equity["2018-01-01":"2018-12-31"]
dd_2018 = dd["2018-01-01":"2018-12-31"]

if len(eq_2018) > 0:
    peak_date = eq_2018.idxmax()
    trough_date = dd_2018.idxmin()
    max_dd_2018 = dd_2018.min()
    print(f"  2018 peak:   {peak_date.date()} (${eq_2018.max():,.0f})")
    print(f"  2018 trough: {trough_date.date()} (${eq_2018.min():,.0f})")
    print(f"  Max DD 2018: {max_dd_2018:.1%}")

    # Trades that were open during the drawdown period
    t2018 = trade_df[
        (trade_df["exit_date"] >= "2018-01-01") &
        (trade_df["exit_date"] <= "2018-12-31")
    ].copy()
    t2018["exit_date"] = pd.to_datetime(t2018["exit_date"])

    if len(t2018) > 0:
        # Trades during the drawdown period (peak to trough)
        dd_trades = t2018[
            (t2018["exit_date"] >= peak_date) &
            (t2018["exit_date"] <= trough_date)
        ]
        print(f"\n  Trades during drawdown ({peak_date.date()} → {trough_date.date()}):")
        print(f"    Total trades: {len(dd_trades)}")
        print(f"    Losing trades: {(dd_trades['pnl'] < 0).sum()}")
        print(f"    Total PnL loss: ${dd_trades['pnl'].sum():,.0f}")
        print(f"    Avg loss per trade: ${dd_trades[dd_trades['pnl']<0]['pnl'].mean():,.0f}")

        # How concentrated was the damage?
        worst = dd_trades.nsmallest(5, "pnl")[["symbol","exit_date","pnl","reason"]]
        print(f"\n  Top 5 worst trades in drawdown:")
        print(worst.to_string(index=False))

        # Simulate circuit breaker: halt entries when DD > 10%, resume at 7%
        print(f"\n  CIRCUIT BREAKER SIMULATION (halt at -10%, resume at -7%):")
        halted = False
        saved_pnl = 0
        prevented_trades = 0
        running_equity = equity.copy()

        # Walk through the drawdown period day by day
        dd_period = dd["2018-10-01":"2018-12-31"]  # Q4 is where damage happened
        for date, dd_val in dd_period.items():
            if not halted and dd_val <= -0.10:
                halted = True
                halt_date = date
                print(f"    Circuit breaker TRIPS at {date.date()} (DD={dd_val:.1%})")
            elif halted and dd_val > -0.07:
                halted = False
                print(f"    Circuit breaker RESETS at {date.date()} (DD={dd_val:.1%})")

        # Trades that would have been prevented
        if 'halt_date' in dir():
            prevented = dd_trades[
                (dd_trades["exit_date"] >= halt_date) &
                (dd_trades["pnl"] < 0) &
                (dd_trades["reason"].isin(["stop", "max_hold"]))
            ]
            print(f"    Trades that would be prevented: {len(prevented)}")
            print(f"    PnL saved: ${prevented['pnl'].sum():,.0f}")
        else:
            print(f"    DD never hit -10% in Q4 2018 — circuit breaker wouldn't have helped!")
            print(f"    Check: was the damage spread across the whole year?")
            monthly_pnl = t2018.groupby(t2018["exit_date"].dt.month)["pnl"].sum()
            print(f"    Monthly PnL 2018: {monthly_pnl.to_dict()}")

# ── Q2: 2023 Regime Day Count ─────────────────────────────────────────────────
print("\n" + "="*60)
print("Q2: 2023 REGIME CLASSIFICATION — WAS IT MISCLASSIFYING?")
print("="*60)

from regime_classifier import load_macro_data, build_regime_series

macro_cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache_prices")
spy_df, hyg_df, vix_df = load_macro_data(cache_dir=macro_cache)

dates_2023 = spy_df["2023-01-01":"2023-12-31"].index
regime_2023 = build_regime_series(spy_df, hyg_df, vix_df, dates_2023)

counts = regime_2023.value_counts()
total  = len(regime_2023)
print(f"\n  2023 regime distribution ({total} trading days):")
for r, n in counts.items():
    print(f"    {r:<16}: {n:3d} days ({n/total:.0%})")

# Monthly breakdown
monthly = regime_2023.resample("ME").agg(lambda x: x.value_counts().index[0])
print(f"\n  Month-by-month dominant regime:")
for month, regime in monthly.items():
    print(f"    {month.strftime('%Y-%m')}: {regime}")

# Trades in CHOPPY days in 2023
if "regime" in trade_df.columns:
    t2023 = trade_df[
        (trade_df["exit_date"] >= "2023-01-01") &
        (trade_df["exit_date"] <= "2023-12-31")
    ]
    choppy_2023 = t2023[t2023["regime"] == "CHOPPY"]
    bull_2023   = t2023[t2023["regime"] == "TRENDING_BULL"]
    print(f"\n  2023 trades entered in CHOPPY regime:       {len(choppy_2023)}")
    print(f"  2023 trades entered in TRENDING_BULL regime: {len(bull_2023)}")
    if len(choppy_2023) > 0:
        wr = (choppy_2023["pnl"] > 0).mean()
        print(f"  CHOPPY 2023 win rate: {wr:.0%}  avg PnL: ${choppy_2023['pnl'].mean():.0f}")
    if len(bull_2023) > 0:
        wr = (bull_2023["pnl"] > 0).mean()
        print(f"  BULL   2023 win rate: {wr:.0%}  avg PnL: ${bull_2023['pnl'].mean():.0f}")

    # Key question: what ML rank trades were BLOCKED by 0.93 threshold in CHOPPY days?
    # We can't see blocked trades, but we can see what rank the 2023 CHOPPY entries had
    if len(choppy_2023) > 0:
        print(f"\n  ML rank distribution of 2023 CHOPPY entries:")
        print(f"    Min:    {choppy_2023['ml_rank_pct'].min():.2f}")
        print(f"    Median: {choppy_2023['ml_rank_pct'].median():.2f}")
        print(f"    Mean:   {choppy_2023['ml_rank_pct'].mean():.2f}")
        print(f"  (All >= 0.93 since that's the current threshold)")
        print(f"  If regime was correct, lowering to 0.88 adds ranks 0.88-0.93.")
        print(f"  Check 2023 BULL entries in range 0.88-0.93 as a proxy for quality:")
        band = bull_2023[(bull_2023["ml_rank_pct"] >= 0.88) & (bull_2023["ml_rank_pct"] < 0.93)]
        if len(band) > 0:
            wr = (band["pnl"] > 0).mean()
            print(f"    Rank 0.88-0.93 in BULL 2023: {len(band)} trades, WR={wr:.0%}, avg=${band['pnl'].mean():.0f}")
        else:
            print(f"    No BULL trades in 0.88-0.93 band in 2023")

# ── Q3: Short Stop vs Max_Hold by Year ───────────────────────────────────────
print("\n" + "="*60)
print("Q3: SHORT EXIT BREAKDOWN BY YEAR")
print("="*60)

shorts = trade_df[trade_df["side"] == "short"].copy()
shorts["year"] = pd.to_datetime(shorts["exit_date"]).dt.year

if len(shorts) > 0:
    print(f"\n  Overall short exit breakdown:")
    for reason, grp in shorts.groupby("reason"):
        wr  = (grp["pnl"] > 0).mean()
        tot = grp["pnl"].sum()
        print(f"    {reason:<22} {len(grp):4d} trades  WR={wr:.0%}  avg=${grp['pnl'].mean():,.0f}  total=${tot:,.0f}")

    print(f"\n  Short stops by year (the problem):")
    stops = shorts[shorts["reason"] == "short_stop"]
    if len(stops) > 0:
        for yr, grp in stops.groupby("year"):
            print(f"    {yr}: {len(grp):3d} stops  total=${grp['pnl'].sum():,.0f}  avg=${grp['pnl'].mean():,.0f}")
    else:
        print("    No short stops found")

    print(f"\n  Short max_hold by year (what works):")
    holds = shorts[shorts["reason"] == "short_max_hold"]
    if len(holds) > 0:
        for yr, grp in holds.groupby("year"):
            wr = (grp["pnl"] > 0).mean()
            print(f"    {yr}: {len(grp):3d} max_hold  WR={wr:.0%}  total=${grp['pnl'].sum():,.0f}  avg=${grp['pnl'].mean():,.0f}")

    # Simulate: what if we had NO price stops on shorts (time exits only)?
    print(f"\n  SIMULATION: remove short price stops entirely (time exits only)")
    stops_loss = stops[stops["pnl"] < 0]["pnl"].sum()
    stops_win  = stops[stops["pnl"] > 0]["pnl"].sum()
    print(f"    Current stop losses: ${stops_loss:,.0f}")
    print(f"    Current stop wins:   ${stops_win:,.0f}")
    print(f"    Net cost of price stops: ${stops[' pnl'].sum() if 'pnl' in stops else stops_loss + stops_win:,.0f}")

    # How many stops were within first N days (entered too early)
    if "entry_date" in shorts.columns:
        stops_with_hold = stops.copy()
        stops_with_hold["hold_days"] = (
            pd.to_datetime(stops_with_hold["exit_date"]) -
            pd.to_datetime(stops_with_hold["entry_date"])
        ).dt.days
        print(f"\n  Short stops by hold duration:")
        bins = [0, 3, 7, 14, 30, 999]
        labels = ["0-3d", "4-7d", "8-14d", "15-30d", "30d+"]
        stops_with_hold["hold_bucket"] = pd.cut(stops_with_hold["hold_days"], bins=bins, labels=labels)
        for bucket, grp in stops_with_hold.groupby("hold_bucket", observed=True):
            print(f"    {bucket}: {len(grp):3d} stops  avg=${grp['pnl'].mean():,.0f}  total=${grp['pnl'].sum():,.0f}")
        print(f"  (Stops in first 3 days = entered into bounce, not breakdown)")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("trades_v2.csv and equity_v2.csv saved in ~/ai_trading_bot_v2/v2/")
print("="*60)