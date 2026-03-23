"""
strategy_bear.py — Bear Market Strategy
========================================
Used when regime = BEAR.

Philosophy: In bear markets, the primary job is capital preservation
and opportunistic shorting. The market is in a downtrend driven by
fundamental forces (rates, credit, earnings deterioration). Fighting
this with longs is expensive. The edge is on the short side.

Key differences from other regimes:
- Very few longs (1-2 max, only defensive sectors)
- Active short book (4-6 shorts)
- Short entry: ML rank < 0.20 + distribution confirmed + below 200MA
- Longs only in: utilities, consumer staples, healthcare, energy
- Position scalar: 40% of normal on longs, 60% on shorts
- Very tight stops on longs (2% min)
- Wider profit targets on shorts (25% take profit)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from entry_filter import is_in_accumulation, accumulation_score
from regime_classifier import BEAR


# ── Strategy parameters ───────────────────────────────────────────────────────
MAX_POSITIONS_LONG   = 2       # minimal longs
MAX_POSITIONS_SHORT  = 6       # active short book
ML_RANK_MIN_LONG     = 0.95    # only top 5% for longs
ML_RANK_MAX_SHORT    = 0.15    # bottom 15% for shorts
MAX_HOLD_DAYS_LONG   = 8       # short hold for longs
MAX_HOLD_DAYS_SHORT  = 20      # longer hold for shorts
TAKE_PROFIT_LONG     = 0.15    # quick take profit on longs
TAKE_PROFIT_SHORT    = 0.25    # bigger target on shorts
STOP_MIN_LONG        = 0.020   # tight stop on longs
STOP_MAX_LONG        = 0.06    # very tight max
STOP_SHORT           = 0.06    # stop out shorts at 6% adverse
RISK_PER_TRADE_LONG  = 0.015   # 1.5% risk on longs
RISK_PER_TRADE_SHORT = 0.025   # 2.5% risk on shorts
POSITION_SCALAR_LONG = 0.40    # 40% sizing on longs
POSITION_SCALAR_SHORT= 0.60    # 60% sizing on shorts

# Defensive sectors allowed for longs in bear market
DEFENSIVE_SECTORS: Set[str] = {"XLU", "XLP", "XLV", "XLE"}


@dataclass
class StrategyParams:
    max_positions_long:    int   = MAX_POSITIONS_LONG
    max_positions_short:   int   = MAX_POSITIONS_SHORT
    ml_rank_min_long:      float = ML_RANK_MIN_LONG
    ml_rank_max_short:     float = ML_RANK_MAX_SHORT
    max_hold_days_long:    int   = MAX_HOLD_DAYS_LONG
    max_hold_days_short:   int   = MAX_HOLD_DAYS_SHORT
    take_profit_long:      float = TAKE_PROFIT_LONG
    take_profit_short:     float = TAKE_PROFIT_SHORT
    stop_min_long:         float = STOP_MIN_LONG
    stop_max_long:         float = STOP_MAX_LONG
    stop_short:            float = STOP_SHORT
    risk_per_trade_long:   float = RISK_PER_TRADE_LONG
    risk_per_trade_short:  float = RISK_PER_TRADE_SHORT
    position_scalar_long:  float = POSITION_SCALAR_LONG
    position_scalar_short: float = POSITION_SCALAR_SHORT


def get_params() -> StrategyParams:
    return StrategyParams()


def should_enter_long(
    symbol: str,
    ml_rank_pct: float,
    combined_score: float,
    df: pd.DataFrame,
    sector: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Very strict long entry in bear market.
    Only defensive sectors in strong accumulation.
    """
    params = get_params()

    # Only defensive sectors
    if sector not in DEFENSIVE_SECTORS:
        return False, f"sector {sector} not defensive — no longs in bear"

    if ml_rank_pct < params.ml_rank_min_long:
        return False, f"ML rank {ml_rank_pct:.2f} below bear long threshold"

    # Must be in strong accumulation
    if not is_in_accumulation(df, BEAR):
        return False, "distribution detected — no longs in bear"

    acc = accumulation_score(df)
    if acc < 0.65:
        return False, f"accumulation score {acc:.2f} too weak for bear long"

    # Must be above 200MA
    if len(df) >= 200:
        close = df["close"]
        ma200 = close.rolling(200).mean().iloc[-1]
        if close.iloc[-1] < ma200:
            return False, "below 200MA — no longs in bear"

    return True, ""


def should_enter_short(
    symbol: str,
    ml_rank_pct: float,
    df: pd.DataFrame,
    sector: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Short entry in bear market.
    Requires: low ML rank + distribution + below 200MA + non-defensive sector.
    """
    params = get_params()

    # Don't short defensive sectors
    if sector in DEFENSIVE_SECTORS:
        return False, f"sector {sector} is defensive — skip short"

    if ml_rank_pct > params.ml_rank_max_short:
        return False, f"ML rank {ml_rank_pct:.2f} too high for short"

    # Must show distribution
    if is_in_accumulation(df, BEAR):
        return False, "stock in accumulation — skip short"

    # Must be below 200MA (genuinely broken)
    if len(df) >= 200:
        close = df["close"]
        ma200 = close.rolling(200).mean().iloc[-1]
        if close.iloc[-1] > ma200 * 1.02:
            return False, "above 200MA — not broken enough to short"

    return True, ""


def score_short_candidates(
    snapshots: dict,
    prices: Dict[str, pd.DataFrame],
    sector_map: Dict[str, str],
    as_of_date: pd.Timestamp,
) -> List[dict]:
    """Score and rank short candidates for bear strategy."""
    params = get_params()
    scored = []

    for sym, snap in snapshots.items():
        if snap.ml_rank_pct > params.ml_rank_max_short:
            continue

        df = prices.get(sym)
        if df is None:
            continue
        df_asof = df.loc[:as_of_date]
        sector = sector_map.get(sym)

        ok, reason = should_enter_short(sym, snap.ml_rank_pct, df_asof, sector)
        if not ok:
            continue

        # Lower score = better short candidate
        short_score = 1.0 - snap.ml_rank_pct
        acc = accumulation_score(df_asof)
        dist_score = 1.0 - acc  # high distribution = good short

        scored.append({
            "symbol":      sym,
            "ml_rank_pct": snap.ml_rank_pct,
            "short_score": short_score * dist_score,
            "stop_pct":    params.stop_short,
        })

    scored.sort(key=lambda x: x["short_score"], reverse=True)
    return scored[:params.max_positions_short * 2]


if __name__ == "__main__":
    print("Bear Market Strategy Parameters:")
    p = get_params()
    print(f"  Max longs:        {p.max_positions_long}")
    print(f"  Max shorts:       {p.max_positions_short}")
    print(f"  Long ML min:      {p.ml_rank_min_long:.0%}")
    print(f"  Short ML max:     {p.ml_rank_max_short:.0%}")
    print(f"  Long hold days:   {p.max_hold_days_long}")
    print(f"  Short hold days:  {p.max_hold_days_short}")
    print(f"  Long scalar:      {p.position_scalar_long:.0%}")
    print(f"  Short scalar:     {p.position_scalar_short:.0%}")
    print(f"  Defensive sectors for longs: {DEFENSIVE_SECTORS}")
    print("\nStrategy ready.")
