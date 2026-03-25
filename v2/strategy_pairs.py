"""
strategy_pairs.py — Statistical Arbitrage (Pairs Trading) Engine
================================================================
Trades mean-reversion of cointegrated pairs alongside the main
momentum engine. Generates alpha that is NEGATIVELY correlated
with momentum — when momentum crashes, pairs profits.

Research basis:
  - Yale / Zhu 2024: pairs trading Sharpe ~1.35 OOS standalone
  - Negatively correlated with momentum → natural hedge
  - Expected contribution: +4-6% CAGR

Confirmed cointegrated pairs (tested March 25 2026):
  V/MA    p=0.0006  hedge=0.554  half_life=32.8d  lookback=60d
  CSX/UNP p=0.0083  hedge=0.153  half_life=46.3d  lookback=90d
  LMT/NOC p=0.034   hedge=0.743  half_life=72.3d  lookback=120d

Mechanism:
  1. Compute rolling hedge ratio (OLS, rolling window)
  2. Compute spread = leg_a - hedge * leg_b
  3. Normalize spread to z-score using rolling mean/std
  4. Enter when z > ENTRY_ZSCORE (spread too wide: short A, long B)
     or z < -ENTRY_ZSCORE (spread too narrow: long A, short B)
  5. Exit when z reverts to EXIT_ZSCORE
  6. Stop if z exceeds STOP_ZSCORE (spread keeps diverging)

Integration:
  Import in backtester_v2.py and call update_pairs() each day.
  Pair trades are appended to the same trades list with
  reason='pairs_exit', reason='pairs_stop', etc.
  side field uses 'pairs_long' or 'pairs_short'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from backtester_clean import Trade, apply_fill_cost


# ── Pair definitions ──────────────────────────────────────────────────────────

@dataclass
class PairConfig:
    leg_a:       str    # symbol we go long when spread is low
    leg_b:       str    # symbol we go short when spread is low
    lookback:    int    # rolling window for hedge ratio + z-score
    entry_z:     float = 2.5   # enter when |z| exceeds this
    exit_z:      float = 0.5   # exit when |z| falls below this
    stop_z:      float = 3.5   # stop if |z| exceeds this (spread keeps diverging)
    max_hold:    int   = 45    # max hold days regardless of z-score
    capital_pct: float = 0.06  # % of capital per pair leg (each side)


# Confirmed cointegrated pairs — do not add pairs without running
# cointegration test (p < 0.05) and half-life check first
PAIRS: List[PairConfig] = [
    PairConfig("V",   "MA",  lookback=60,  capital_pct=0.06),
    PairConfig("CSX", "UNP", lookback=90,  capital_pct=0.06),
    PairConfig("LMT", "NOC", lookback=120, capital_pct=0.04),  # weaker, smaller size
]


# ── Position tracking ─────────────────────────────────────────────────────────

@dataclass
class PairPosition:
    """Tracks one active pairs trade (long one leg, short the other)."""
    pair_id:       str          # e.g. "V_MA"
    long_symbol:   str
    short_symbol:  str
    long_qty:      int
    short_qty:     int
    long_entry:    float
    short_entry:   float
    entry_date:    str
    entry_z:       float        # z-score at entry
    direction:     str          # "long_a" or "long_b"

    def age_days(self, current_date) -> int:
        from datetime import datetime
        try:
            entry = datetime.strptime(self.entry_date, "%Y-%m-%d")
            if hasattr(current_date, "to_pydatetime"):
                current_date = current_date.to_pydatetime()
            return max(0, (current_date - entry).days)
        except Exception:
            return 0


# ── Spread computation ────────────────────────────────────────────────────────

def compute_spread_zscore(
    prices_a: pd.Series,
    prices_b: pd.Series,
    lookback: int,
) -> Tuple[float, float]:
    """
    Compute rolling hedge ratio and z-score for the spread.

    Returns:
        (z_score, hedge_ratio) as of the last available date.
        Returns (nan, nan) if insufficient data.
    """
    common = prices_a.index.intersection(prices_b.index)
    if len(common) < lookback + 10:
        return float("nan"), float("nan")

    pa = prices_a[common].iloc[-lookback:]
    pb = prices_b[common].iloc[-lookback:]

    # OLS hedge ratio: regress pa on pb over lookback window
    from numpy.linalg import lstsq
    X = np.column_stack([pb.values, np.ones(len(pb))])
    coeffs, _, _, _ = lstsq(X, pa.values, rcond=None)
    hedge = coeffs[0]

    spread = pa - hedge * pb

    mean   = spread.mean()
    std    = spread.std()
    if std < 1e-8:
        return float("nan"), float("nan")

    z_score = (spread.iloc[-1] - mean) / std
    return float(z_score), float(hedge)


def get_zscore_series(
    prices_a: pd.Series,
    prices_b: pd.Series,
    lookback: int,
) -> pd.Series:
    """
    Compute full rolling z-score series. Used for analysis / debugging.
    """
    common = prices_a.index.intersection(prices_b.index)
    pa = prices_a[common]
    pb = prices_b[common]

    hedge  = pa.rolling(lookback).mean() / pb.rolling(lookback).mean()
    spread = pa - hedge * pb
    mean   = spread.rolling(lookback).mean()
    std    = spread.rolling(lookback).std()
    return ((spread - mean) / std).dropna()


# ── Main engine ───────────────────────────────────────────────────────────────

class PairsEngine:
    """
    Manages all pair positions. Call update() once per trading day
    from the main backtest loop.
    """

    def __init__(self, pairs: List[PairConfig] = PAIRS):
        self.pairs      = {f"{p.leg_a}_{p.leg_b}": p for p in pairs}
        self.positions: Dict[str, PairPosition] = {}
        self.cooldowns: Dict[str, pd.Timestamp] = {}  # pair_id -> last exit date
        self.COOLDOWN_DAYS = 5

    def required_symbols(self) -> List[str]:
        """Return all symbols needed by the pairs engine."""
        syms = set()
        for p in self.pairs.values():
            syms.add(p.leg_a)
            syms.add(p.leg_b)
        return list(syms)

    def update(
        self,
        date:           pd.Timestamp,
        next_date:      pd.Timestamp,
        prices_by_sym:  Dict[str, pd.DataFrame],
        cash:           float,
        capital:        float,
        trades:         List[Trade],
    ) -> float:
        """
        Main daily update. Handles exits first, then entries.

        Args:
            date:          current trading date
            next_date:     next trading date (entry fills at open)
            prices_by_sym: full price history keyed by symbol
            cash:          current cash balance
            capital:       initial capital (for sizing)
            trades:        shared trades list — appended to directly

        Returns:
            updated cash balance
        """
        cash = self._handle_exits(date, prices_by_sym, cash, capital, trades)
        cash = self._handle_entries(date, next_date, prices_by_sym, cash, capital, trades)
        return cash

    def portfolio_value(
        self,
        close_prices: Dict[str, float],
    ) -> float:
        """Mark-to-market value of all open pair positions."""
        val = 0.0
        for pos in self.positions.values():
            long_px  = close_prices.get(pos.long_symbol,  pos.long_entry)
            short_px = close_prices.get(pos.short_symbol, pos.short_entry)
            val += (long_px  - pos.long_entry)  * pos.long_qty
            val += (pos.short_entry - short_px) * pos.short_qty
        return val

    # ── Exits ─────────────────────────────────────────────────────────────

    def _handle_exits(
        self,
        date:          pd.Timestamp,
        prices_by_sym: Dict[str, pd.DataFrame],
        cash:          float,
        capital:       float,
        trades:        List[Trade],
    ) -> float:

        for pair_id in list(self.positions.keys()):
            pos    = self.positions[pair_id]
            cfg    = self.pairs[pair_id]

            df_a   = prices_by_sym.get(pos.long_symbol)
            df_b   = prices_by_sym.get(pos.short_symbol)
            if df_a is None or df_b is None:
                continue

            pa = df_a["close"].loc[:date]
            pb = df_b["close"].loc[:date]
            if len(pa) < cfg.lookback or len(pb) < cfg.lookback:
                continue

            z, hedge = compute_spread_zscore(pa, pb, cfg.lookback)
            if np.isnan(z):
                continue

            close_a = float(pa.iloc[-1])
            close_b = float(pb.iloc[-1])
            age     = pos.age_days(date)

            exit_reason = None

            # Mean reversion achieved
            if abs(z) <= cfg.exit_z:
                exit_reason = "pairs_exit"
            # Stop — spread kept diverging in wrong direction
            elif abs(z) >= cfg.stop_z:
                exit_reason = "pairs_stop"
            # Max hold
            elif age >= cfg.max_hold:
                exit_reason = "pairs_maxhold"

            if exit_reason:
                cash = self._close_position(
                    pos=pos, pair_id=pair_id, date=date,
                    close_a=close_a, close_b=close_b,
                    exit_reason=exit_reason, cash=cash, trades=trades,
                )
                self.cooldowns[pair_id] = date

        return cash

    def _close_position(
        self,
        pos:          PairPosition,
        pair_id:      str,
        date:         pd.Timestamp,
        close_a:      float,
        close_b:      float,
        exit_reason:  str,
        cash:         float,
        trades:       List[Trade],
    ) -> float:
        # Close long leg
        fill_long, comm_long   = apply_fill_cost(close_a, pos.long_qty,  "sell")
        pnl_long = (fill_long - pos.long_entry) * pos.long_qty - comm_long
        cash += fill_long * pos.long_qty - comm_long

        # Close short leg (buy back)
        fill_short, comm_short = apply_fill_cost(close_b, pos.short_qty, "buy")
        pnl_short = (pos.short_entry - fill_short) * pos.short_qty - comm_short
        cash += pnl_short
        cash += pos.short_qty * pos.short_entry * 0.50  # return margin held at entry

        total_pnl = pnl_long + pnl_short

        # Record as single trade entry for clean reporting
        trades.append(Trade(
            symbol=f"{pos.long_symbol}/{pos.short_symbol}",
            entry_date=pos.entry_date,
            exit_date=str(date.date()),
            entry_price=pos.long_entry,
            exit_price=fill_long,
            qty=pos.long_qty,
            pnl=total_pnl,
            reason=exit_reason,
            ml_rank_pct=0.0,
            rule_score=0.0,
            combined_score=abs(pos.entry_z),  # use entry z-score as "conviction"
            side="pairs",
        ))

        del self.positions[pair_id]
        return cash

    # ── Entries ───────────────────────────────────────────────────────────

    def _handle_entries(
        self,
        date:          pd.Timestamp,
        next_date:     pd.Timestamp,
        prices_by_sym: Dict[str, pd.DataFrame],
        cash:          float,
        capital:       float,
        trades:        List[Trade],
    ) -> float:

        for pair_id, cfg in self.pairs.items():
            # Already in this pair
            if pair_id in self.positions:
                continue

            # Cooldown after recent exit
            last_exit = self.cooldowns.get(pair_id)
            if last_exit and (date - last_exit).days < self.COOLDOWN_DAYS:
                continue

            df_a = prices_by_sym.get(cfg.leg_a)
            df_b = prices_by_sym.get(cfg.leg_b)
            if df_a is None or df_b is None:
                continue

            pa = df_a["close"].loc[:date]
            pb = df_b["close"].loc[:date]
            if len(pa) < cfg.lookback + 10 or len(pb) < cfg.lookback + 10:
                continue

            z, hedge = compute_spread_zscore(pa, pb, cfg.lookback)
            if np.isnan(z) or abs(z) < cfg.entry_z:
                continue

            # Get next-day open prices for fills
            open_a = self._get_next_open(df_a, date)
            open_b = self._get_next_open(df_b, date)
            if open_a is None or open_b is None:
                continue

            # Determine direction
            # z > entry_z: spread is wide (A expensive vs B) → short A, long B
            # z < -entry_z: spread is narrow (A cheap vs B) → long A, short B
            if z > cfg.entry_z:
                long_sym,  long_px  = cfg.leg_b, open_b
                short_sym, short_px = cfg.leg_a, open_a
                direction = "long_b"
            else:
                long_sym,  long_px  = cfg.leg_a, open_a
                short_sym, short_px = cfg.leg_b, open_b
                direction = "long_a"

            # Size: capital_pct of portfolio per leg
            leg_dollars = capital * cfg.capital_pct
            long_qty    = int(leg_dollars / long_px)  if long_px  > 0 else 0
            short_qty   = int(leg_dollars / short_px) if short_px > 0 else 0

            if long_qty <= 0 or short_qty <= 0:
                continue

            # Cost check: long leg costs cash, short leg requires margin (50%)
            long_cost    = long_qty * long_px
            short_margin = short_qty * short_px * 0.50
            total_cost   = long_cost + short_margin

            if total_cost > cash * 0.95:  # leave 5% buffer
                continue

            # Fill
            fill_long,  comm_long  = apply_fill_cost(long_px,  long_qty,  "buy")
            fill_short, comm_short = apply_fill_cost(short_px, short_qty, "sell")

            cash -= fill_long  * long_qty  + comm_long
            cash -= short_margin  # margin hold

            self.positions[pair_id] = PairPosition(
                pair_id=pair_id,
                long_symbol=long_sym,
                short_symbol=short_sym,
                long_qty=long_qty,
                short_qty=short_qty,
                long_entry=fill_long,
                short_entry=fill_short,
                entry_date=str(next_date.date()),
                entry_z=z,
                direction=direction,
            )

        return cash

    @staticmethod
    def _get_next_open(
        df: pd.DataFrame,
        date: pd.Timestamp,
    ) -> Optional[float]:
        """Get the next trading day's open price after date."""
        future = df.loc[df.index > date]
        if len(future) == 0:
            return None
        return float(future.iloc[0]["open"])


# ── Standalone backtest ───────────────────────────────────────────────────────

def backtest_pairs_standalone(
    days: int = 3650,
) -> Tuple[pd.Series, List[Trade], dict]:
    """
    Run pairs engine in isolation to measure its standalone contribution
    before integrating into the main backtester.

    Usage:
        from strategy_pairs import backtest_pairs_standalone
        equity, trades, stats = backtest_pairs_standalone()
    """
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from backtester_clean import fetch_history, calc_stats
    import config

    print("="*60)
    print("  PAIRS ENGINE STANDALONE BACKTEST")
    print("="*60)

    engine   = PairsEngine()
    req_syms = engine.required_symbols()
    print(f"  Pairs: {[f'{p.leg_a}/{p.leg_b}' for p in PAIRS]}")
    print(f"  Required symbols: {req_syms}")
    print()

    # Load price data
    hist: Dict[str, pd.DataFrame] = {}
    for s in req_syms:
        try:
            hist[s] = fetch_history(s, days)
            print(f"  Loaded {s}: {len(hist[s])} bars")
        except Exception as e:
            print(f"  ERROR loading {s}: {e}")

    if len(hist) < 2:
        raise RuntimeError("Insufficient price data for pairs backtest")

    # Build trade date index from intersection of all pair symbols
    all_dates = None
    for df in hist.values():
        if all_dates is None:
            all_dates = df.index
        else:
            all_dates = all_dates.intersection(df.index)
    all_dates = all_dates.sort_values()

    capital = float(config.INITIAL_CAPITAL)
    cash    = capital
    trades: List[Trade] = []
    equity: List[Tuple] = []

    print(f"\n  Running {len(all_dates)} trading days...\n")

    for i, date in enumerate(all_dates[120:], start=120):  # warmup period
        next_idx = i + 1
        if next_idx >= len(all_dates):
            break
        next_date = all_dates[next_idx]

        # Close prices for portfolio valuation
        close_prices = {s: float(df["close"].loc[:date].iloc[-1])
                        for s, df in hist.items()
                        if date in df.index and len(df.loc[:date]) > 0}

        cash = engine.update(
            date=date, next_date=next_date,
            prices_by_sym=hist,
            cash=cash, capital=capital,
            trades=trades,
        )

        pair_val = engine.portfolio_value(close_prices)
        equity.append((date, cash + pair_val))

        if i % 250 == 0:
            print(f"  Step {i}/{len(all_dates)}  positions={len(engine.positions)}"
                  f"  trades={len(trades)}  equity=${cash+pair_val:,.0f}")

    equity_curve = pd.Series(
        data=[v for _, v in equity],
        index=pd.to_datetime([d for d, _ in equity]),
        name="pairs_equity",
    )

    stats = calc_stats(equity_curve, trades)

    print("\n" + "="*60)
    print("  PAIRS STANDALONE RESULTS")
    print("="*60)
    print(f"  CAGR         : {stats['cagr']:>10.2%}")
    print(f"  Sharpe       : {stats['sharpe']:>10.2f}")
    print(f"  Max Drawdown : {stats['max_drawdown']:>10.2%}")
    print(f"  Trades       : {stats['trades']:>10}")
    print(f"  Win Rate     : {stats['win_rate']:>10.2%}")
    print()

    if trades:
        df_t = pd.DataFrame([t.__dict__ for t in trades])
        for sym, grp in df_t.groupby("symbol"):
            wr = (grp["pnl"] > 0).mean()
            print(f"  {sym:<12} {len(grp):4d} trades  WR={wr:.0%}  avg=${grp['pnl'].mean():.0f}")

    print("="*60)
    return equity_curve, trades, stats


# ── Integration guide (printed when run directly) ─────────────────────────────

INTEGRATION_NOTES = """
HOW TO INTEGRATE INTO backtester_v2.py
=======================================

1. Add import at top of backtester_v2.py:
   from strategy_pairs import PairsEngine

2. After the main loop setup (after earnings_dates loading), add:
   pairs_engine = PairsEngine()

3. Inside the main loop, after the long exits section, add:
   cash = pairs_engine.update(
       date=date, next_date=next_date,
       prices_by_sym=prices_by_symbol,
       cash=cash, capital=config.INITIAL_CAPITAL,
       trades=trades,
   )

4. Update _portfolio_value call to include pairs:
   pair_val = pairs_engine.portfolio_value(close_prices)
   equity.append((date, port_val + pair_val))
   (or pass pairs_engine to _portfolio_value)

5. First run standalone to confirm Sharpe > 1.0:
   python3.11 strategy_pairs.py --standalone

IMPORTANT: Run standalone first. Only integrate if standalone
Sharpe > 1.0. Do not integrate a broken pairs engine into the
main system.
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", action="store_true",
                        help="Run standalone pairs backtest")
    parser.add_argument("--zscore",     action="store_true",
                        help="Print current z-scores for all pairs")
    args = parser.parse_args()

    if args.standalone:
        backtest_pairs_standalone()
    elif args.zscore:
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from backtester_clean import fetch_history
        print("\nCurrent z-scores:")
        print("-" * 40)
        for cfg in PAIRS:
            try:
                da = fetch_history(cfg.leg_a, 365)["close"]
                db = fetch_history(cfg.leg_b, 365)["close"]
                z, hedge = compute_spread_zscore(da, db, cfg.lookback)
                signal = ""
                if abs(z) >= cfg.entry_z:
                    signal = f"  ← ENTRY SIGNAL (z={z:.2f})"
                print(f"  {cfg.leg_a}/{cfg.leg_b}: z={z:.3f}  hedge={hedge:.3f}{signal}")
            except Exception as e:
                print(f"  {cfg.leg_a}/{cfg.leg_b}: error — {e}")
        print()
    else:
        print(INTEGRATION_NOTES)
