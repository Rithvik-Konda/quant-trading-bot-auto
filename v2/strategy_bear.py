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
- Short entry: ML rank < 0.15 + distribution confirmed 3+ consecutive days
             + below 20MA for 3+ consecutive days + below 200MA
- Longs only in: utilities, consumer staples, healthcare, energy
- Position scalar: 40% of normal on longs, 60% on shorts
- Very tight stops on longs (2% min)
- Wider profit targets on shorts (25% take profit)

SHORT EXIT PHILOSOPHY (v2.1):
  Diagnostic showed 143 price stops at 0% win rate = -$216k destroyed.
  95 max_hold exits at 85% win rate = +$182k. The direction is right,
  timing is wrong. Bear market bounces of 5-10% are normal and temporary.
  A 6% price stop fires on every bounce before the thesis plays out.

  Solution: Remove price stops on shorts. Use time-based exits only.
  Keep a 15% DISASTER stop as backstop ONLY for catastrophic squeezes
  (short squeeze, unexpected takeover bid, etc). This is not a normal
  stop — it should almost never fire.

  Economic logic: if a stock is genuinely breaking down (confirmed by
  3+ days below 20MA + 3+ days distribution), a 5-8% bounce is noise,
  not a signal to exit. The thesis plays out over 10-20 days.

LONG ENTRY TIGHTENING (v2.1):
  521 long stops at 15% win rate = entering stocks not yet ready.
  Added 20MA confirmation: stock must be above rising 20MA for longs.
  This is structural — a stock below its 20MA is not in an uptrend
  regardless of what the ML model says.
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
MAX_HOLD_DAYS_SHORT  = 20      # longer hold for shorts (primary exit mechanism)
TAKE_PROFIT_LONG     = 0.15    # quick take profit on longs
TAKE_PROFIT_SHORT    = 0.25    # bigger target on shorts
STOP_MIN_LONG        = 0.020   # tight stop on longs
STOP_MAX_LONG        = 0.06    # very tight max on longs

# SHORT STOP: disaster backstop only — NOT a normal stop
# Should fire only on catastrophic events (squeeze, takeover bid)
# Normal bear bounces of 5-15% should NOT trigger this
STOP_SHORT_DISASTER  = 0.25    # 25% disaster backstop — only for catastrophic squeezes

RISK_PER_TRADE_LONG  = 0.015   # 1.5% risk on longs
RISK_PER_TRADE_SHORT = 0.025   # 2.5% risk on shorts
POSITION_SCALAR_LONG = 0.40    # 40% sizing on longs
POSITION_SCALAR_SHORT= 0.60    # 60% sizing on shorts

# Short entry confirmation thresholds
# Require confirmed breakdown, not just a bad day
SHORT_ENTRY_MA20_DAYS        = 3   # must be below 20MA for this many consecutive days
SHORT_ENTRY_DISTRIBUTION_DAYS = 3  # must show distribution for this many consecutive days

# Defensive sectors allowed for longs in bear market
DEFENSIVE_SECTORS: Set[str] = {"XLU", "XLP", "XLV", "XLE"}


@dataclass
class StrategyParams:
    max_positions_long:         int   = MAX_POSITIONS_LONG
    max_positions_short:        int   = MAX_POSITIONS_SHORT
    ml_rank_min_long:           float = ML_RANK_MIN_LONG
    ml_rank_max_short:          float = ML_RANK_MAX_SHORT
    max_hold_days_long:         int   = MAX_HOLD_DAYS_LONG
    max_hold_days_short:        int   = MAX_HOLD_DAYS_SHORT
    take_profit_long:           float = TAKE_PROFIT_LONG
    take_profit_short:          float = TAKE_PROFIT_SHORT
    stop_min_long:              float = STOP_MIN_LONG
    stop_max_long:              float = STOP_MAX_LONG
    stop_short:                 float = STOP_SHORT_DISASTER  # disaster only
    risk_per_trade_long:        float = RISK_PER_TRADE_LONG
    risk_per_trade_short:       float = RISK_PER_TRADE_SHORT
    position_scalar_long:       float = POSITION_SCALAR_LONG
    position_scalar_short:      float = POSITION_SCALAR_SHORT
    short_entry_ma20_days:      int   = SHORT_ENTRY_MA20_DAYS
    short_entry_dist_days:      int   = SHORT_ENTRY_DISTRIBUTION_DAYS


def get_params() -> StrategyParams:
    return StrategyParams()


def _days_below_ma20(df: pd.DataFrame) -> int:
    """
    Count consecutive days the stock has closed below its 20-day MA.
    Returns 0 if above MA today.
    Used to confirm genuine breakdown vs single-day dip.
    """
    if len(df) < 22:
        return 0
    close = df["close"]
    ma20  = close.rolling(20).mean()

    # Walk back from today counting consecutive below-MA days
    count = 0
    for i in range(len(df) - 1, max(len(df) - 30, 19), -1):
        if pd.isna(ma20.iloc[i]):
            break
        if close.iloc[i] < ma20.iloc[i]:
            count += 1
        else:
            break
    return count


def _days_in_distribution(df: pd.DataFrame, lookback: int = 10) -> int:
    """
    Count consecutive days showing distribution (down-volume > up-volume).
    Uses a rolling 3-day window to determine each day's character.
    Returns number of consecutive distribution days ending today.

    This is stricter than the entry_filter's binary check — we need
    sustained distribution, not just a single distribution reading.
    """
    if len(df) < lookback + 2:
        return 0

    close  = df["close"]
    volume = df["volume"]
    price_change = close.diff()

    # For each day, is it a distribution day?
    # Distribution day: price down on above-average volume
    avg_vol = volume.rolling(20).mean()
    dist_days = (price_change < 0) & (volume > avg_vol)

    # Count consecutive distribution days from end
    count = 0
    for i in range(len(dist_days) - 1, max(len(dist_days) - 15, 0), -1):
        if dist_days.iloc[i]:
            count += 1
        else:
            break
    return count


def should_enter_long(
    symbol: str,
    ml_rank_pct: float,
    combined_score: float,
    df: pd.DataFrame,
    sector: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Very strict long entry in bear market.
    Only defensive sectors in strong accumulation AND above 20MA.

    v2.1: Added 20MA check — stock must be above rising 20MA.
    A stock below its 20MA in a bear market is not a long candidate
    regardless of ML rank or sector.
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

    # v2.1: Must be above 20MA (trending up short-term)
    if len(df) >= 22:
        close = df["close"]
        ma20  = close.rolling(20).mean().iloc[-1]
        if close.iloc[-1] < ma20:
            return False, "below 20MA — not in uptrend for bear long"

    return True, ""


def should_enter_short(
    symbol: str,
    ml_rank_pct: float,
    df: pd.DataFrame,
    sector: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Short entry in bear market. v2.1: Much tighter confirmation required.

    Requires ALL of:
    1. Low ML rank (bottom 15%)
    2. Non-defensive sector
    3. Below 200MA (structurally broken)
    4. Below 20MA for 3+ consecutive days (confirmed short-term downtrend)
    5. Distribution for 3+ consecutive days (institutional selling confirmed)

    Rationale: The old system entered shorts too early, getting squeezed
    on bear bounces. By requiring 3+ days of confirmed breakdown on BOTH
    price and volume, we enter AFTER the bounce, not before it.

    The cost: we miss the first 3 days of a short. That's acceptable —
    we're trading the middle of the move, not trying to pick the exact top.
    """
    params = get_params()

    # Don't short defensive sectors
    if sector in DEFENSIVE_SECTORS:
        return False, f"sector {sector} is defensive — skip short"

    if ml_rank_pct > params.ml_rank_max_short:
        return False, f"ML rank {ml_rank_pct:.2f} too high for short"

    # Must show distribution (entry_filter check)
    if is_in_accumulation(df, BEAR):
        return False, "stock in accumulation — skip short"

    # Must be below 200MA (genuinely broken structurally)
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
    """
    Score and rank short candidates for bear strategy.
    Returns stop_pct as the disaster backstop (0.15), NOT a normal stop.
    """
    params = get_params()
    scored = []

    for sym, snap in snapshots.items():
        if snap.ml_rank_pct > params.ml_rank_max_short:
            continue

        df = prices.get(sym)
        if df is None:
            continue
        df_asof = df.loc[:as_of_date]
        sector  = sector_map.get(sym)

        ok, reason = should_enter_short(sym, snap.ml_rank_pct, df_asof, sector)
        if not ok:
            continue

        # Lower ML rank = better short candidate
        short_score = 1.0 - snap.ml_rank_pct
        acc         = accumulation_score(df_asof)
        dist_score  = 1.0 - acc  # high distribution = good short

        # Boost score by how many days below MA (more confirmed = higher priority)
        days_below  = _days_below_ma20(df_asof)
        confirmation_boost = min(days_below / 10.0, 0.5)  # cap at 0.5 boost

        scored.append({
            "symbol":      sym,
            "ml_rank_pct": snap.ml_rank_pct,
            "short_score": short_score * dist_score * (1 + confirmation_boost),
            "stop_pct":    params.stop_short,  # 0.15 disaster backstop
            "days_below_ma20": days_below,
        })

    scored.sort(key=lambda x: x["short_score"], reverse=True)
    return scored[:params.max_positions_short * 2]


if __name__ == "__main__":
    print("Bear Market Strategy Parameters (v2.1):")
    p = get_params()
    print(f"  Max longs:             {p.max_positions_long}")
    print(f"  Max shorts:            {p.max_positions_short}")
    print(f"  Long ML min:           {p.ml_rank_min_long:.0%}")
    print(f"  Short ML max:          {p.ml_rank_max_short:.0%}")
    print(f"  Long hold days:        {p.max_hold_days_long}")
    print(f"  Short hold days:       {p.max_hold_days_short}  (primary exit)")
    print(f"  Long scalar:           {p.position_scalar_long:.0%}")
    print(f"  Short scalar:          {p.position_scalar_short:.0%}")
    print(f"  Short disaster stop:   {p.stop_short:.0%}  (backstop only, not normal stop)")
    print(f"  Short MA20 confirm:    {p.short_entry_ma20_days} consecutive days below")
    print(f"  Short dist confirm:    {p.short_entry_dist_days} consecutive distribution days")
    print(f"  Defensive sectors:     {DEFENSIVE_SECTORS}")
    print()
    print("  EXIT LOGIC:")
    print("  Shorts: take_profit (25%) > max_hold (20d) > disaster_stop (15%)")
    print("  Price stops removed — time-based exits only for shorts")
    print("\nStrategy ready.")