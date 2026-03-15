"""
pead_backtest.py — Pure PEAD Strategy Validation
=================================================
No ML, no regime filter, no momentum.
Just: buy within 3 days of earnings beat > threshold, hold 60 days.

This validates whether PEAD has standalone alpha before we integrate
it into the main system. If this doesn't work on its own, blending
it into the momentum system won't fix it.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import config
from pead_signal import build_earnings_store
from backtester_clean import fetch_history, apply_fill_cost, calc_stats
from strategy_core import normalize_ohlcv

# ── PEAD parameters to test ───────────────────────────────────────────────────
MIN_SURPRISE_PCT   = 10.0   # minimum earnings beat to trigger entry
MAX_ENTRY_DAYS     = 3      # must enter within this many days of announcement
HOLD_DAYS          = 60     # hold for 60 days (full drift window)
STOP_PCT           = 0.10   # wider stop — PEAD is fundamental, not noise
TAKE_PROFIT_PCT    = 0.30   # take profit at 30%
MAX_POSITIONS      = 10     # more positions — PEAD diversifies well
RISK_PER_TRADE     = 0.020  # smaller risk per trade
MAX_DOLLARS        = 20_000 # max per position
INITIAL_CAPITAL    = 100_000
SLIPPAGE_BPS       = 10     # slightly higher slippage for earnings plays

@dataclass
class PEADTrade:
    symbol:       str
    entry_date:   str
    exit_date:    str
    entry_price:  float
    exit_price:   float
    qty:          int
    pnl:          float
    reason:       str
    surprise_pct: float
    days_to_entry: int  # how many days after announcement we entered


def run_pead_backtest(
    days: int = 3650,
    min_surprise: float = MIN_SURPRISE_PCT,
    max_entry_days: int = MAX_ENTRY_DAYS,
    hold_days: int = HOLD_DAYS,
    verbose: bool = True,
) -> Tuple[pd.Series, List[PEADTrade], dict]:

    print(f"\n{'='*60}")
    print(f"  PURE PEAD BACKTEST")
    print(f"  Min surprise: {min_surprise}%")
    print(f"  Max entry days after announcement: {max_entry_days}")
    print(f"  Hold days: {hold_days}")
    print(f"  Universe: {len(config.WATCHLIST)} stocks")
    print(f"{'='*60}\n")

    # Load price data
    symbols = list(config.WATCHLIST)
    hist: Dict[str, pd.DataFrame] = {}
    print("Loading price data...")
    for s in symbols:
        try:
            df = fetch_history(s, days)
            if df is not None and len(df) > 200:
                hist[s] = df
        except Exception:
            pass
    print(f"Loaded {len(hist)} symbols\n")

    # Load earnings data
    print("Loading earnings data...")
    earnings_store = build_earnings_store(list(hist.keys()), verbose=True)
    print()

    # Build trade date index
    spy = fetch_history(config.BENCHMARK_SYMBOL, days)
    all_dates = spy.index.sort_values()

    # Main backtest loop
    cash = float(INITIAL_CAPITAL)
    positions: Dict[str, dict] = {}  # symbol -> {entry_price, qty, entry_date, surprise, days_to_entry, entry_ann_date}
    trades: List[PEADTrade] = []
    equity: List[Tuple] = []

    print(f"Running backtest over {len(all_dates)} trading days...\n")

    for i, date in enumerate(all_dates):
        if i % 200 == 0 and verbose:
            pct = i / len(all_dates) * 100
            print(f"  {pct:.0f}%  date={date.date()}  positions={len(positions)}  trades={len(trades)}  cash=${cash:,.0f}",
                  end="\r", flush=True)

        # ── Exit positions ────────────────────────────────────────────────
        for sym in list(positions.keys()):
            pos = positions[sym]
            if sym not in hist or date not in hist[sym].index:
                continue

            df_now = hist[sym].loc[:date]
            close = float(df_now["close"].iloc[-1])
            low   = float(df_now["low"].iloc[-1])

            entry_date_ts = pd.Timestamp(pos["entry_date"])
            hold_d = (date - entry_date_ts).days

            exit_reason = exit_px = None

            # Stop loss
            stop_px = pos["entry_price"] * (1 - STOP_PCT)
            if low <= stop_px:
                exit_reason = "stop"
                exit_px = stop_px

            # Take profit
            elif close >= pos["entry_price"] * (1 + TAKE_PROFIT_PCT):
                exit_reason = "take_profit"
                exit_px = close

            # Time exit — PEAD drift window exhausted
            elif hold_d >= hold_days:
                exit_reason = "time_exit"
                exit_px = close

            if exit_reason:
                fill, comm = apply_fill_cost(exit_px, pos["qty"], "sell")
                pnl = (fill - pos["entry_price"]) * pos["qty"] - comm
                cash += fill * pos["qty"] - comm
                trades.append(PEADTrade(
                    symbol=sym,
                    entry_date=pos["entry_date"],
                    exit_date=str(date.date()),
                    entry_price=pos["entry_price"],
                    exit_price=fill,
                    qty=pos["qty"],
                    pnl=pnl,
                    reason=exit_reason,
                    surprise_pct=pos["surprise"],
                    days_to_entry=pos["days_to_entry"],
                ))
                del positions[sym]

        # ── Entry: scan for fresh earnings beats ──────────────────────────
        if len(positions) < MAX_POSITIONS:
            for sym in list(hist.keys()):
                if sym in positions:
                    continue
                if sym not in earnings_store:
                    continue
                if date not in hist[sym].index:
                    continue

                earn_df = earnings_store[sym]
                # Find earnings announced within max_entry_days of today
                recent = earn_df[
                    (earn_df["date"] <= date) &
                    (earn_df["date"] >= date - pd.Timedelta(days=max_entry_days))
                ]

                if len(recent) == 0:
                    continue

                # Take the most recent announcement
                latest = recent.iloc[-1]
                surprise = float(latest["surprise"])
                days_since = (date - latest["date"]).days

                # Must beat by minimum threshold
                if surprise < min_surprise:
                    continue

                # Must not have already entered on this earnings announcement
                # (check if we recently exited this symbol after this announcement)
                already_traded = any(
                    t.symbol == sym and
                    pd.Timestamp(t.entry_date) >= latest["date"]
                    for t in trades
                )
                if already_traded:
                    continue

                # Get entry price — next day open
                future = hist[sym].loc[hist[sym].index > date]
                if len(future) == 0:
                    continue
                entry_px = float(future.iloc[0]["open"])

                # Check stock hasn't already run away (>15% since announcement)
                ann_date = latest["date"]
                df_since_ann = hist[sym].loc[ann_date:date]
                if len(df_since_ann) > 1:
                    price_at_ann = float(df_since_ann.iloc[0]["close"])
                    move_since = (entry_px - price_at_ann) / price_at_ann
                    if move_since > 0.15:  # already moved 15%+ — missed it
                        continue

                # Size position
                risk_budget = INITIAL_CAPITAL * RISK_PER_TRADE
                stop_per_share = entry_px * STOP_PCT
                qty_risk = int(risk_budget / stop_per_share) if stop_per_share > 0 else 0
                qty = min(qty_risk, int(min(MAX_DOLLARS, cash * 0.3) / entry_px) if entry_px > 0 else 0)
                if qty <= 0:
                    continue

                cost = entry_px * qty * 1.001  # include slippage
                if cost > cash:
                    continue

                cash -= cost
                positions[sym] = {
                    "entry_price":    entry_px,
                    "qty":            qty,
                    "entry_date":     str(future.iloc[0].name.date()),
                    "surprise":       surprise,
                    "days_to_entry":  days_since,
                    "ann_date":       str(ann_date.date()),
                }

                if len(positions) >= MAX_POSITIONS:
                    break

        # Portfolio value
        port_val = cash + sum(
            float(hist[sym].loc[:date]["close"].iloc[-1]) * pos["qty"]
            if sym in hist and date in hist[sym].index
            else pos["entry_price"] * pos["qty"]
            for sym, pos in positions.items()
        )
        equity.append((date, port_val))

    print("\n")

    equity_curve = pd.Series(
        data=[v for _, v in equity],
        index=pd.to_datetime([d for d, _ in equity]),
    )

    # ── Results ───────────────────────────────────────────────────────────
    trade_df = pd.DataFrame([t.__dict__ for t in trades])

    print(f"{'='*60}")
    print(f"  PEAD BACKTEST RESULTS")
    print(f"{'='*60}")

    if len(trades) == 0:
        print("  No trades generated.")
        return equity_curve, trades, {}

    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = len(equity_curve) / 252
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1
    rets = equity_curve.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    dd = (equity_curve / equity_curve.cummax() - 1).min()
    wr = (trade_df.pnl > 0).mean()

    print(f"  Total Return : {total_ret:.2%}")
    print(f"  CAGR         : {cagr:.2%}")
    print(f"  Sharpe       : {sharpe:.2f}")
    print(f"  Max Drawdown : {dd:.2%}")
    print(f"  Trades       : {len(trades)}")
    print(f"  Win Rate     : {wr:.2%}")
    print(f"  Avg PnL      : ${trade_df.pnl.mean():.0f}")
    print(f"  Total PnL    : ${trade_df.pnl.sum():.0f}")

    print(f"\n  Exit breakdown:")
    for reason, g in trade_df.groupby("reason"):
        print(f"    {reason:<20}: {len(g):4d}  WR={(g.pnl>0).mean():.1%}  avg=${g.pnl.mean():.0f}")

    print(f"\n  By surprise bucket:")
    for bucket, g in trade_df.groupby(pd.cut(trade_df.surprise_pct, bins=[10,15,20,30,50,100])):
        if len(g) > 0:
            print(f"    surprise {str(bucket):<20}: {len(g):4d} trades  WR={(g.pnl>0).mean():.1%}  avg=${g.pnl.mean():.0f}")

    print(f"\n  By days to entry:")
    for bucket, g in trade_df.groupby(pd.cut(trade_df.days_to_entry, bins=[-1,0,1,2,3])):
        if len(g) > 0:
            print(f"    entry day {str(bucket):<15}: {len(g):4d} trades  WR={(g.pnl>0).mean():.1%}  avg=${g.pnl.mean():.0f}")

    print(f"\n  By year:")
    trade_df["year"] = pd.to_datetime(trade_df["entry_date"]).dt.year
    for year, g in trade_df.groupby("year"):
        eq_yr = equity_curve[equity_curve.index.year == year]
        yr_ret = eq_yr.iloc[-1] / eq_yr.iloc[0] - 1 if len(eq_yr) > 1 else 0
        print(f"    {year}: {len(g):4d} trades  WR={(g.pnl>0).mean():.1%}  avg=${g.pnl.mean():.0f}  yr_ret={yr_ret:.1%}")

    print(f"\n  Top 10 symbols:")
    sym_pnl = trade_df.groupby("symbol").pnl.sum().sort_values(ascending=False)
    for sym, pnl in sym_pnl.head(10).items():
        n = (trade_df.symbol == sym).sum()
        print(f"    {sym:<8}: ${pnl:.0f}  ({n} trades)")

    print(f"{'='*60}\n")

    # OOS split
    oos_cutoff = pd.Timestamp("2022-01-01")
    eq_oos = equity_curve[equity_curve.index >= oos_cutoff]
    if len(eq_oos) > 1:
        eq_oos_norm = eq_oos / eq_oos.iloc[0] * INITIAL_CAPITAL
        oos_ret = eq_oos_norm.iloc[-1] / eq_oos_norm.iloc[0] - 1
        oos_years = len(eq_oos) / 252
        oos_cagr = (eq_oos_norm.iloc[-1] / eq_oos_norm.iloc[0]) ** (1/oos_years) - 1
        oos_rets = eq_oos_norm.pct_change().dropna()
        oos_sharpe = (oos_rets.mean() / oos_rets.std()) * np.sqrt(252) if oos_rets.std() > 0 else 0
        oos_dd = (eq_oos_norm / eq_oos_norm.cummax() - 1).min()
        oos_trades = trade_df[pd.to_datetime(trade_df.entry_date) >= oos_cutoff]

        print(f"  OOS (2022-2025):")
        print(f"    CAGR    : {oos_cagr:.2%}")
        print(f"    Sharpe  : {oos_sharpe:.2f}")
        print(f"    Max DD  : {oos_dd:.2%}")
        print(f"    Trades  : {len(oos_trades)}")
        print(f"    Win Rate: {(oos_trades.pnl > 0).mean():.1%}" if len(oos_trades) > 0 else "    Win Rate: N/A")

        print(f"\n  OOS by year:")
        for year, g in oos_trades.groupby("year"):
            eq_yr = eq_oos_norm[eq_oos_norm.index.year == year]
            yr_ret = eq_yr.iloc[-1] / eq_yr.iloc[0] - 1 if len(eq_yr) > 1 else 0
            print(f"    {year}: {len(g):4d} trades  WR={(g.pnl>0).mean():.1%}  avg=${g.pnl.mean():.0f}  yr_ret={yr_ret:.1%}")

    return equity_curve, trades, {}


if __name__ == "__main__":
    equity, trades, stats = run_pead_backtest(
        days=3650,
        min_surprise=10.0,
        max_entry_days=3,
        hold_days=60,
    )


def run_pead_variations():
    """Test key parameter variations to find optimal PEAD config."""

    configs = [
        {"name": "Baseline (stops on)",        "min_surprise": 10, "max_entry_days": 3,  "hold_days": 60, "stop_pct": 0.10},
        {"name": "No stops",                    "min_surprise": 10, "max_entry_days": 3,  "hold_days": 60, "stop_pct": 9.99},
        {"name": "No stops + day 1-3 only",     "min_surprise": 10, "max_entry_days": 3,  "hold_days": 60, "stop_pct": 9.99},
        {"name": "No stops + higher threshold", "min_surprise": 15, "max_entry_days": 3,  "hold_days": 60, "stop_pct": 9.99},
        {"name": "No stops + longer hold",      "min_surprise": 10, "max_entry_days": 3,  "hold_days": 90, "stop_pct": 9.99},
        {"name": "No stops + tight surprise",   "min_surprise": 20, "max_entry_days": 3,  "hold_days": 60, "stop_pct": 9.99},
    ]

    import os, sys
    results = []

    for cfg in configs:
        # Temporarily override module constants
        import pead_backtest as pb
        pb.STOP_PCT          = cfg["stop_pct"]
        pb.MIN_SURPRISE_PCT  = cfg["min_surprise"]
        pb.MAX_ENTRY_DAYS    = cfg["max_entry_days"]
        pb.HOLD_DAYS         = cfg["hold_days"]

        equity, trades, _ = pb.run_pead_backtest(
            days=3650,
            min_surprise=cfg["min_surprise"],
            max_entry_days=cfg["max_entry_days"],
            hold_days=cfg["hold_days"],
            verbose=False,
        )

        if len(trades) == 0:
            continue

        import pandas as pd
        import numpy as np
        trade_df = pd.DataFrame([t.__dict__ for t in trades])

        total_ret = equity.iloc[-1] / equity.iloc[0] - 1
        years = len(equity) / 252
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
        rets = equity.pct_change().dropna()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        dd = (equity / equity.cummax() - 1).min()
        wr = (trade_df.pnl > 0).mean()

        oos_cutoff = pd.Timestamp("2022-01-01")
        eq_oos = equity[equity.index >= oos_cutoff]
        eq_oos_norm = eq_oos / eq_oos.iloc[0] * 100_000
        oos_years = len(eq_oos) / 252
        oos_cagr = (eq_oos_norm.iloc[-1] / eq_oos_norm.iloc[0]) ** (1/oos_years) - 1
        oos_rets = eq_oos_norm.pct_change().dropna()
        oos_sharpe = (oos_rets.mean() / oos_rets.std()) * np.sqrt(252) if oos_rets.std() > 0 else 0
        oos_trades = trade_df[pd.to_datetime(trade_df.entry_date) >= oos_cutoff]
        oos_wr = (oos_trades.pnl > 0).mean() if len(oos_trades) > 0 else 0

        results.append({
            "name":       cfg["name"],
            "cagr":       cagr,
            "sharpe":     sharpe,
            "max_dd":     dd,
            "wr":         wr,
            "trades":     len(trades),
            "oos_cagr":   oos_cagr,
            "oos_sharpe": oos_sharpe,
            "oos_wr":     oos_wr,
            "oos_trades": len(oos_trades),
        })

        print(f"  {cfg['name']:<40} CAGR={cagr:.1%}  Sharpe={sharpe:.2f}  DD={dd:.1%}  OOS_CAGR={oos_cagr:.1%}  OOS_Sharpe={oos_sharpe:.2f}  Trades={len(trades)}")

    print("\n\nSummary table:")
    print(f"{'Config':<40} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'WR':>6} {'OOS_CAGR':>9} {'OOS_SH':>8} {'Trades':>7}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<40} {r['cagr']:>7.1%} {r['sharpe']:>7.2f} {r['max_dd']:>7.1%} {r['wr']:>6.1%} {r['oos_cagr']:>9.1%} {r['oos_sharpe']:>8.2f} {r['trades']:>7}")


if __name__ == "__main__":
    run_pead_variations()
