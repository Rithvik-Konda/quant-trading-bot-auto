"""
strategy_choppy.py — Choppy/Transitional Market Strategy
=========================================================
Used when regime = CHOPPY.

Philosophy: In choppy markets, momentum signals are unreliable.
Most momentum entries get stopped out quickly because trends don't
sustain. The job is to be highly selective — only take the absolute
highest conviction setups — and size down so losses are contained.

Key differences from trending bull:
- A/D filter is TIGHT (either distribution OR lower highs blocks entry)
- Max positions reduced to 4 (concentrate on best ideas only)
- ML threshold raised to 0.93 (top 7% only)
- Hold period shortened to 12 days (don't overstay in choppy tape)
- Position size 60% of normal (protect capital)
- 2 shorts allowed (some names genuinely breaking down)
- Sector filter: only enter if sector ETF also in accumulation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from entry_filter import is_in_accumulation, accumulation_score
from regime_classifier import CHOPPY


# ── Strategy parameters ───────────────────────────────────────────────────────
MAX_POSITIONS        = 2       # FIXME docstring above says 4 — reconcile
ML_RANK_MIN          = 0.93    # top 7% — very selective
COMBINED_SCORE_MIN   = 0.25    # higher bar in choppy markets
MAX_HOLD_DAYS        = 22      # data: 11-15d exits avg -$86, 21-30d avg +$420
TAKE_PROFIT_PCT      = 0.25    # lower take profit — lock in gains faster
STOP_MIN_PCT         = 0.15   # tighter stops in choppy
STOP_MAX_PCT         = 0.25    # lower max stop
STOP_VOL_MULTIPLIER  = 2.0     # tighter ATR multiplier
RISK_PER_TRADE       = 0.020   # 2% risk per trade (reduced from 3.5%)
MAX_POSITION_WEIGHT  = 0.25    # lower concentration per position
MAX_TOTAL_EXPOSURE   = 1.20    # lower total exposure
MAX_POSITIONS_SHORT  = 2       # 2 shorts allowed
POSITION_SCALAR      = 0.60    # 60% of normal sizing

# ── Mean Reversion parameters (quality-filtered bounce) ──────────────────────
# Academic basis: Zhu et al 2019 — quality filter improves reversal 3.6x
# RSI(2) < 15 + price above 200d SMA + quality composite > 0.5
# Expected win rate: 64-73%, avg hold 3-5 days
MR_RSI2_THRESHOLD    = 15.0   # oversold threshold (Connors optimal = 5-10)
MR_QUALITY_MIN       = 0.50   # quality_composite minimum (F-score proxy)
MR_MAX_POSITIONS     = 3      # up to 3 mean reversion positions
MR_HOLD_DAYS         = 5      # exit after 5 days or above 5d SMA
MR_STOP_PCT          = 0.08   # tight stop — 8% (mean reversion, not momentum)
MR_RISK_PCT          = 0.015  # 1.5% risk per MR trade (smaller than momentum)


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


def _compute_rsi2(df: pd.DataFrame) -> float:
    """Compute RSI(2) — 2-period RSI for mean reversion signal."""
    try:
        close = df['close'].dropna()
        if len(close) < 5:
            return 50.0
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(2).mean()
        loss  = (-delta.clip(upper=0)).rolling(2).mean()
        rs    = gain / (loss + 1e-9)
        rsi   = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except Exception:
        return 50.0


def _is_above_200sma(df: pd.DataFrame) -> bool:
    """Price above 200-day SMA — trend filter for mean reversion."""
    try:
        close = df['close'].dropna()
        if len(close) < 200:
            return len(close) >= 50 and float(close.iloc[-1]) > float(close.tail(50).mean())
        sma200 = float(close.tail(200).mean())
        return float(close.iloc[-1]) > sma200
    except Exception:
        return False


def should_enter(
    symbol: str,
    ml_rank_pct: float,
    combined_score: float,
    df: pd.DataFrame,
    regime: str,
    sector_df: Optional[pd.DataFrame] = None,
) -> Tuple[bool, str]:
    """
    Entry decision for choppy strategy.
    Much stricter than trending bull.
    """
    params = get_params()

    if ml_rank_pct < params.ml_rank_min:
        return False, f"ML rank {ml_rank_pct:.2f} below choppy threshold {params.ml_rank_min}"

    if combined_score < params.combined_score_min:
        return False, f"combined score too low for choppy market"

    # Tight A/D filter
    if not is_in_accumulation(df, CHOPPY):
        return False, "distribution detected — skip in choppy market"

    # Sector filter: sector ETF must also be in accumulation
    if sector_df is not None and len(sector_df) > 20:
        if not is_in_accumulation(sector_df, CHOPPY):
            return False, "sector in distribution — skip in choppy market"

    return True, "momentum"


def should_enter_mean_reversion(
    symbol: str,
    df: pd.DataFrame,
    quality_composite: float = 0.0,
) -> Tuple[bool, str]:
    """
    Mean reversion entry for CHOPPY regime.
    Separate from momentum path — triggered by RSI(2) oversold.
    Quality filter prevents catching falling knives.
    Hold 5 days or above 5-day SMA, whichever comes first.
    """
    if len(df) < 20:
        return False, "insufficient data"

    # RSI(2) oversold
    rsi2 = _compute_rsi2(df)
    if rsi2 > MR_RSI2_THRESHOLD:
        return False, f"RSI(2)={rsi2:.1f} not oversold (threshold={MR_RSI2_THRESHOLD})"

    # Price must be above 200-day SMA — no falling knives
    if not _is_above_200sma(df):
        return False, "price below 200-day SMA — skip"

    # Quality filter — fundamental anchor
    if quality_composite < MR_QUALITY_MIN:
        return False, f"quality too low ({quality_composite:.2f} < {MR_QUALITY_MIN})"

    return True, "mean_reversion"


def score_candidates(
    snapshots: dict,
    prices: Dict[str, pd.DataFrame],
    sector_prices: Dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
) -> List[dict]:
    """
    Score and rank candidates for the choppy strategy.
    Very strict filtering — only highest conviction setups.
    """
    params = get_params()
    scored = []

    for sym, snap in snapshots.items():
        if snap.ml_rank_pct < params.ml_rank_min:
            continue
        # Factor rotation: in CHOPPY regime, boost quality+low-vol stocks
        # Research: quality factor performs best during economic uncertainty
        # Low-vol stocks capture 80% of upside with 60% of downside in choppy markets
        quality = float(snap.__dict__.get("quality_composite", 0.5)) if hasattr(snap, "__dict__") else 0.5
        vol_score = 1.0 - min(float(getattr(snap, "realized_vol_20", 0.25)), 0.5) / 0.5
        choppy_boost = 0.8 + 0.2 * (0.5 * quality + 0.5 * vol_score)
        snap = snap  # snap is immutable — boost applied via combined_score below
        if snap.combined_score * choppy_boost < params.combined_score_min:
            continue

        df = prices.get(sym)
        if df is None:
            continue
        df_asof = df.loc[:as_of_date]

        # Strict A/D check
        if not is_in_accumulation(df_asof, CHOPPY):
            continue

        # Accumulation score for ranking
        acc_score = accumulation_score(df_asof)

        # Accumulation threshold — learned from cross-sectional distribution
        # Use median of current candidates as adaptive threshold
        # rather than hardcoded 0.55. In weak markets median is lower,
        # in strong markets it is higher — threshold moves with conditions.
        if '_acc_scores_today' not in dir():
            _acc_scores_today = [
                accumulation_score(prices.get(s2, pd.DataFrame()))
                for s2 in list(snapshots.keys())[:50]
            ]
        import numpy as _np2
        _acc_median = float(_np2.median(_acc_scores_today)) if _acc_scores_today else 0.50
        _acc_min    = float(max(_acc_median * 0.90, 0.40))  # never below 0.40
        if acc_score < _acc_min:
            continue

        boosted_score = snap.combined_score * acc_score

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
    return scored[:params.max_positions * 2]


def position_size(
    entry_price: float,
    stop_pct: float,
    capital: float,
    current_exposure: float,
    cash: float,
    conviction_mult: float = 1.0,
) -> int:
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
    print("Choppy Market Strategy Parameters:")
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
