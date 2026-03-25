"""
strategy_pairs.py — Statistical Arbitrage (Pairs Trading) Engine
================================================================
Architecture rules (DO NOT VIOLATE):
  1. z-score ALWAYS computed as z = f(leg_a_prices, leg_b_prices)
     z > 0 means leg_a expensive relative to leg_b
     z < 0 means leg_a cheap relative to leg_b

  2. Entry:
     z > +entry_z → SHORT leg_a, LONG leg_b  (direction="long_b")
     z < -entry_z → LONG leg_a, SHORT leg_b  (direction="long_a")

  3. Exit:
     direction="long_b" → exit when z < +exit_z, stop when z > +stop_z
     direction="long_a" → exit when z > -exit_z, stop when z < -stop_z

  4. PnL fills ALWAYS use pos.long_symbol / pos.short_symbol prices
  5. Exit z ALWAYS computed from cfg.leg_a / cfg.leg_b prices
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from backtester_clean import Trade, apply_fill_cost


@dataclass
class PairConfig:
    leg_a:       str
    leg_b:       str
    lookback:    int
    entry_z:     float = 2.5
    exit_z:      float = 0.5
    stop_z:      float = 4.0
    max_hold:    int   = 50
    capital_pct: float = 0.04
    direction:   str   = "both"  # "both", "long_a", "long_b"


PAIRS: List[PairConfig] = [
    # XLF
    PairConfig("AXP",  "ALL",  lookback=60, max_hold=40),
    PairConfig("PGR",  "APO",  lookback=60, max_hold=50,  direction="long_a"),
    PairConfig("ICE",  "WTW",  lookback=60, max_hold=45),
    PairConfig("V",    "MA",   lookback=60, max_hold=47),
    PairConfig("AXP",  "TRV",  lookback=60, max_hold=39),
    PairConfig("SCHW", "RF",   lookback=90, max_hold=63),
    # XLK
    PairConfig("MRVL", "MPWR", lookback=90, max_hold=74),
    PairConfig("ORCL", "PSTG", lookback=60, max_hold=53),
    PairConfig("ADI",  "MPWR", lookback=90, max_hold=57),
    # XLI
    PairConfig("PH",   "PWR",  lookback=60, max_hold=50),
    PairConfig("CSX",  "EXPD", lookback=60, max_hold=42),
    PairConfig("HON",  "NSC",  lookback=60, max_hold=37),
    PairConfig("PRIM", "STRL", lookback=30, max_hold=24, capital_pct=0.03),
    # XLV
    PairConfig("TMO",  "HOLX", lookback=60, max_hold=50),
    PairConfig("ISRG", "SYK",  lookback=60, max_hold=51),
]


@dataclass
class PairPosition:
    pair_id:      str
    long_symbol:  str
    short_symbol: str
    long_qty:     int
    short_qty:    int
    long_entry:   float
    short_entry:  float
    entry_date:   str
    entry_z:      float
    direction:    str    # "long_a" or "long_b"
    short_margin: float  # cash held as margin

    def age_days(self, current_date) -> int:
        from datetime import datetime
        try:
            entry = datetime.strptime(self.entry_date, "%Y-%m-%d")
            if hasattr(current_date, "to_pydatetime"):
                current_date = current_date.to_pydatetime()
            return max(0, (current_date - entry).days)
        except Exception:
            return 0


def compute_zscore(
    prices_a: pd.Series,
    prices_b: pd.Series,
    lookback: int,
) -> Tuple[float, float]:
    common = prices_a.index.intersection(prices_b.index)
    if len(common) < lookback + 10:
        return float("nan"), float("nan")
    pa = prices_a[common].iloc[-lookback:]
    pb = prices_b[common].iloc[-lookback:]
    X = np.column_stack([pb.values, np.ones(len(pb))])
    coeffs, _, _, _ = np.linalg.lstsq(X, pa.values, rcond=None)
    hedge = float(coeffs[0])
    spread = pa - hedge * pb
    std = spread.std()
    if std < 1e-8:
        return float("nan"), float("nan")
    return float((spread.iloc[-1] - spread.mean()) / std), hedge


def get_zscore_series(prices_a: pd.Series, prices_b: pd.Series, lookback: int) -> pd.Series:
    common = prices_a.index.intersection(prices_b.index)
    pa = prices_a[common]
    pb = prices_b[common]
    hedge  = pa.rolling(lookback).mean() / pb.rolling(lookback).mean()
    spread = pa - hedge * pb
    return ((spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()).dropna()


class PairsEngine:
    MAX_CONCURRENT = 6

    def __init__(self, pairs: List[PairConfig] = PAIRS):
        self.pairs      = {f"{p.leg_a}_{p.leg_b}": p for p in pairs}
        self.positions: Dict[str, PairPosition] = {}
        self.cooldowns: Dict[str, pd.Timestamp] = {}
        self.COOLDOWN_DAYS = 5

    def required_symbols(self) -> List[str]:
        syms: set = set()
        for p in self.pairs.values():
            syms.add(p.leg_a); syms.add(p.leg_b)
        return list(syms)

    def update(self, date, next_date, prices_by_sym, cash, capital, trades):
        cash = self._handle_exits(date, prices_by_sym, cash, trades)
        cash = self._handle_entries(date, next_date, prices_by_sym, cash, capital, trades)
        return cash

    def portfolio_value(self, prices_by_sym: Dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
        val = 0.0
        for pos in self.positions.values():
            try:
                lp = float(prices_by_sym[pos.long_symbol]["close"].loc[:date].iloc[-1])
                sp = float(prices_by_sym[pos.short_symbol]["close"].loc[:date].iloc[-1])
                val += (lp - pos.long_entry) * pos.long_qty
                val += (pos.short_entry - sp) * pos.short_qty
            except Exception:
                pass
        return val

    def _handle_exits(self, date, prices_by_sym, cash, trades):
        for pair_id in list(self.positions.keys()):
            pos = self.positions[pair_id]
            cfg = self.pairs[pair_id]

            # z-score from canonical pair direction (leg_a, leg_b)
            try:
                pa = prices_by_sym[cfg.leg_a]["close"].loc[:date]
                pb = prices_by_sym[cfg.leg_b]["close"].loc[:date]
            except KeyError:
                continue
            if len(pa) < cfg.lookback or len(pb) < cfg.lookback:
                continue

            z, _ = compute_zscore(pa, pb, cfg.lookback)
            if np.isnan(z):
                continue

            age = pos.age_days(date)
            exit_reason = None

            if pos.direction == "long_b":
                # Entered because z was HIGH — profit when z falls
                if z <= cfg.exit_z:
                    exit_reason = "pairs_exit"
                elif z >= cfg.stop_z:
                    exit_reason = "pairs_stop"
            else:  # long_a
                # Entered because z was LOW — profit when z rises
                if z >= -cfg.exit_z:
                    exit_reason = "pairs_exit"
                elif z <= -cfg.stop_z:
                    exit_reason = "pairs_stop"

            if age >= cfg.max_hold:
                exit_reason = "pairs_maxhold"

            if exit_reason:
                cash = self._close(pos, pair_id, date, prices_by_sym, exit_reason, cash, trades)
                self.cooldowns[pair_id] = date
        return cash

    def _close(self, pos, pair_id, date, prices_by_sym, exit_reason, cash, trades):
        try:
            cl = float(prices_by_sym[pos.long_symbol]["close"].loc[:date].iloc[-1])
            cs = float(prices_by_sym[pos.short_symbol]["close"].loc[:date].iloc[-1])
        except Exception:
            return cash

        fill_l, comm_l = apply_fill_cost(cl, pos.long_qty, "sell")
        pnl_l = (fill_l - pos.long_entry) * pos.long_qty - comm_l
        cash += fill_l * pos.long_qty - comm_l

        fill_s, comm_s = apply_fill_cost(cs, pos.short_qty, "buy")
        pnl_s = (pos.short_entry - fill_s) * pos.short_qty - comm_s
        cash += pnl_s
        cash += pos.short_margin  # return margin

        trades.append(Trade(
            symbol=f"{pos.long_symbol}/{pos.short_symbol}",
            entry_date=pos.entry_date, exit_date=str(date.date()),
            entry_price=pos.long_entry, exit_price=fill_l,
            qty=pos.long_qty, pnl=pnl_l + pnl_s,
            reason=exit_reason, ml_rank_pct=0.0, rule_score=0.0,
            combined_score=abs(pos.entry_z), side="pairs",
        ))
        del self.positions[pair_id]
        return cash

    def _handle_entries(self, date, next_date, prices_by_sym, cash, capital, trades):
        for pair_id, cfg in self.pairs.items():
            if pair_id in self.positions:
                continue
            if len(self.positions) >= self.MAX_CONCURRENT:
                break

            last_exit = self.cooldowns.get(pair_id)
            if last_exit and (date - last_exit).days < self.COOLDOWN_DAYS:
                continue

            try:
                pa = prices_by_sym[cfg.leg_a]["close"].loc[:date]
                pb = prices_by_sym[cfg.leg_b]["close"].loc[:date]
            except KeyError:
                continue
            if len(pa) < cfg.lookback + 10 or len(pb) < cfg.lookback + 10:
                continue

            z, _ = compute_zscore(pa, pb, cfg.lookback)
            if np.isnan(z) or abs(z) < cfg.entry_z:
                continue

            # Direction filter
            if cfg.direction == "long_a" and z > 0:
                continue
            if cfg.direction == "long_b" and z < 0:
                continue

            # z > 0: leg_a dear  → short leg_a, long leg_b
            # z < 0: leg_a cheap → long leg_a, short leg_b
            if z > 0:
                long_sym, short_sym, direction = cfg.leg_b, cfg.leg_a, "long_b"
            else:
                long_sym, short_sym, direction = cfg.leg_a, cfg.leg_b, "long_a"

            try:
                long_px  = self._next_open(prices_by_sym[long_sym],  date)
                short_px = self._next_open(prices_by_sym[short_sym], date)
            except KeyError:
                continue
            if long_px is None or short_px is None:
                continue

            leg_dollars  = capital * cfg.capital_pct
            long_qty     = int(leg_dollars / long_px)  if long_px  > 0 else 0
            short_qty    = int(leg_dollars / short_px) if short_px > 0 else 0
            if long_qty <= 0 or short_qty <= 0:
                continue

            short_margin = short_qty * short_px * 0.50
            total_cost   = long_qty * long_px + short_margin
            if total_cost > cash * 0.95:
                continue

            fill_l, comm_l = apply_fill_cost(long_px,  long_qty,  "buy")
            fill_s, comm_s = apply_fill_cost(short_px, short_qty, "sell")
            cash -= fill_l * long_qty + comm_l
            cash -= short_margin

            self.positions[pair_id] = PairPosition(
                pair_id=pair_id, long_symbol=long_sym, short_symbol=short_sym,
                long_qty=long_qty, short_qty=short_qty,
                long_entry=fill_l, short_entry=fill_s,
                entry_date=str(next_date.date()), entry_z=z,
                direction=direction, short_margin=short_margin,
            )
        return cash

    @staticmethod
    def _next_open(df: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
        future = df.loc[df.index > date]
        return float(future.iloc[0]["open"]) if len(future) > 0 else None


def backtest_pairs_standalone(days: int = 3650):
    from backtester_clean import fetch_history, calc_stats
    import config

    print("=" * 60)
    print("  PAIRS ENGINE STANDALONE BACKTEST")
    print("=" * 60)

    engine   = PairsEngine()
    req_syms = engine.required_symbols()
    print(f"  Pairs: {[f'{p.leg_a}/{p.leg_b}' for p in PAIRS]}")
    print()

    hist: Dict[str, pd.DataFrame] = {}
    for s in req_syms:
        try:
            hist[s] = fetch_history(s, days)
            print(f"  Loaded {s}: {len(hist[s])} bars")
        except Exception as e:
            print(f"  ERROR {s}: {e}")

    all_dates = None
    for df in hist.values():
        all_dates = df.index if all_dates is None else all_dates.intersection(df.index)
    all_dates = all_dates.sort_values()

    capital = float(config.INITIAL_CAPITAL)
    cash    = capital
    trades: List[Trade] = []
    equity  = []

    print(f"\n  Running {len(all_dates)} trading days...\n")

    for i in range(120, len(all_dates) - 1):
        date      = all_dates[i]
        next_date = all_dates[i + 1]
        cash = engine.update(date=date, next_date=next_date, prices_by_sym=hist,
                             cash=cash, capital=capital, trades=trades)
        pair_val = engine.portfolio_value(hist, date)
        equity.append((date, cash + pair_val))
        if i % 250 == 0:
            print(f"  Step {i}/{len(all_dates)}  open={len(engine.positions)}"
                  f"  trades={len(trades)}  equity=${cash+pair_val:,.0f}")

    equity_curve = pd.Series(
        data=[v for _, v in equity],
        index=pd.to_datetime([d for d, _ in equity]),
        name="pairs_equity",
    )
    stats = calc_stats(equity_curve, trades)

    print("\n" + "=" * 60)
    print("  PAIRS STANDALONE RESULTS")
    print("=" * 60)
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
            print(f"  {sym:<16} {len(grp):3d} trades  WR={wr:.0%}  avg=${grp['pnl'].mean():.0f}")
    if stats.get("annual"):
        print()
        for yr in sorted(stats["annual"]):
            a = stats["annual"][yr]
            flag = "✓" if a["cagr"] > 0 else "✗"
            print(f"  {yr}  {a['cagr']:>7.1%}  Sharpe {a['sharpe']:>5.2f}  {flag}")
    print("=" * 60)
    return equity_curve, trades, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--zscore",     action="store_true")
    args = parser.parse_args()
    if args.standalone:
        backtest_pairs_standalone()
    elif args.zscore:
        from backtester_clean import fetch_history
        print("\nCurrent z-scores:")
        for cfg in PAIRS:
            try:
                da = fetch_history(cfg.leg_a, 365)["close"]
                db = fetch_history(cfg.leg_b, 365)["close"]
                z, hedge = compute_zscore(da, db, cfg.lookback)
                sig = f"  ← ENTRY (z={z:.2f})" if abs(z) >= cfg.entry_z else ""
                print(f"  {cfg.leg_a}/{cfg.leg_b:<12} z={z:>6.3f}{sig}")
            except Exception as e:
                print(f"  {cfg.leg_a}/{cfg.leg_b}: {e}")
    else:
        print("Usage: python strategy_pairs.py --standalone | --zscore")
