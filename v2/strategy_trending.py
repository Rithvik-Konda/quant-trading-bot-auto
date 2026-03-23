"""
strategy_trending.py — Trending Bull Market Strategy
=====================================================
Used when regime = TRENDING_BULL.

Philosophy: In trending bull markets, momentum is the dominant force.
Stocks that have been going up tend to keep going up. The job is to
identify the strongest momentum stocks, enter them, hold for 12-20 days,
and let the trend do the work.

Key differences from v1:
- A/D filter is LOOSE (only blocks extreme distribution)
- Max positions increased to 8 (more diversification)
- Hold period extended to 20 days (momentum needs time)
- ML threshold at 0.80 (cast wider net in strong markets)
- Full position sizing (3.5% risk per trade)
- No shorts (bull market — don't fight the tape)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from entry_filter import is_in_accumulation, accumulation_score
from regime_classifier import TRENDING_BULL


# ── Strategy parameters ───────────────────────────────────────────────────────
MAX_POSITIONS        = 8       # more positions in bull — diversify the winners
ML_RANK_MIN          = 0.80    # top 20% of ML signals
COMBINED_SCORE_MIN   = 0.18    # minimum combined score
MAX_HOLD_DAYS        = 20      # extended hold — let momentum play out
TAKE_PROFIT_PCT      = 0.40    # 40% take profit
STOP_MIN_PCT         = 0.035   # 3.5% minimum stop
STOP_MAX_PCT         = 0.12    # 12% maximum stop
STOP_VOL_MULTIPLIER  = 2.5     # ATR × 2.5 for stop width
RISK_PER_TRADE       = 0.035   # 3.5% of capital at risk per trade
MAX_POSITION_WEIGHT  = 0.35    # max 35% of capital in one position
MAX_TOTAL_EXPOSURE   = 1.60    # 160% gross exposure max
MAX_POSITIONS_SHORT  = 0       # no shorts in trending bull
POSITION_SCALAR      = 1.0     # full sizing


@dataclass
class StrategyParams:
    max_positions:       int   = MAX_POSITIONS
    ml_rank_min:         float = ML_RANK_MIN
    combined_score_min:  float = COMBINED_SCORE_MIN
    max_hold_days:       int   = MAX_HOLD_DAYS
    take_profit_pct:     float = TAKE_PROFIT_PCT
    stop_min_pct:        float = STOP_MIN_PCT
    stop_max_pct:        float = STOP_MAX_PCT
    stop_vol_multiplier: float = STOP_VOL_MULTIPLIER
    risk_per_trade:      float = RISK_PER_TRADE
    max_position_weight: float = MAX_POSITION_WEIGHT
    max_total_exposure:  float = MAX_TOTAL_EXPOSURE
    max_positions_short: int   = MAX_POSITIONS_SHORT
    position_scalar:     float = POSITION_SCALAR


def get_params() -> StrategyParams:
    return StrategyParams()


def should_enter(
    symbol: str,
    ml_rank_pct: float,
    combined_score: float,
    df: pd.DataFrame,
    regime: str,
) -> Tuple[bool, str]:
    """
    Entry decision for trending bull strategy.
    Returns (should_enter, reason_if_blocked).
    """
    params = get_params()

    if ml_rank_pct < params.ml_rank_min:
        return False, f"ML rank {ml_rank_pct:.2f} below threshold {params.ml_rank_min}"

    if combined_score < params.combined_score_min:
        return False, f"combined score {combined_score:.3f} below threshold"

    # Loose A/D filter in bull market
    if not is_in_accumulation(df, TRENDING_BULL):
        return False, "extreme distribution detected"

    return True, ""


def score_candidates(
    snapshots: dict,
    prices: Dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
) -> List[dict]:
    """
    Score and rank candidates for the trending bull strategy.
    Returns list of dicts sorted by combined score descending.
    """
    params = get_params()
    scored = []

    for sym, snap in snapshots.items():
        if snap.ml_rank_pct < params.ml_rank_min:
            continue
        if snap.combined_score < params.combined_score_min:
            continue

        df = prices.get(sym)
        if df is None:
            continue
        df_asof = df.loc[:as_of_date]

        # Boost score by accumulation strength
        acc_score = accumulation_score(df_asof)
        boosted_score = snap.combined_score * (0.8 + 0.2 * acc_score)

        scored.append({
            "symbol":         sym,
            "ml_rank_pct":    snap.ml_rank_pct,
            "combined_score": boosted_score,
            "raw_score":      snap.combined_score,
            "acc_score":      acc_score,
            "stop_pct":       snap.stop_pct,
            "atr_pct":        snap.atr_pct,
        })

    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    return scored[:params.max_positions * 2]  # return 2x candidates for position mgr


def position_size(
    entry_price: float,
    stop_pct: float,
    capital: float,
    current_exposure: float,
    cash: float,
    conviction_mult: float = 1.0,
) -> int:
    """
    Calculate position size for trending bull strategy.
    Uses risk-based sizing: risk_per_trade / stop_pct = shares.
    """
    params = get_params()

    scalar = params.position_scalar * conviction_mult
    risk_budget = capital * params.risk_per_trade * scalar

    risk_per_share = entry_price * stop_pct
    if risk_per_share <= 0:
        return 0

    qty_risk = int(risk_budget / risk_per_share)

    max_dollars = min(
        capital * params.max_position_weight * scalar,
        cash,
        max(0, capital * params.max_total_exposure - current_exposure),
    )
    qty_dollars = int(max_dollars / entry_price) if entry_price > 0 else 0

    return max(0, min(qty_risk, qty_dollars))


if __name__ == "__main__":
    print("Trending Bull Strategy Parameters:")
    p = get_params()
    print(f"  Max positions:    {p.max_positions}")
    print(f"  ML rank min:      {p.ml_rank_min:.0%}")
    print(f"  Max hold days:    {p.max_hold_days}")
    print(f"  Risk per trade:   {p.risk_per_trade:.1%}")
    print(f"  Stop min/max:     {p.stop_min_pct:.1%} / {p.stop_max_pct:.1%}")
    print(f"  Take profit:      {p.take_profit_pct:.0%}")
    print(f"  Position scalar:  {p.position_scalar:.0%}")
    print(f"  Shorts allowed:   {p.max_positions_short}")
    print("\nStrategy ready.")
