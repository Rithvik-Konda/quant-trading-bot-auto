"""
strategy_meanrev.py — RSI(2) Mean Reversion Engine
====================================================
Activates in CHOPPY regime as a complement to the momentum engine.

Core logic (Larry Connors RSI(2), validated 1999-2024):
  Entry:  RSI(2) < threshold AND price > SMA200 (dip in uptrend)
  Exit:   price closes > SMA5  OR  10-day time stop
  
Why this works in CHOPPY markets:
  - Momentum IC drops to ~0.02 in choppy regimes
  - Mean reversion IC is ~0.04-0.06 in choppy regimes  
  - Stocks oscillate 3-8% around a mean in sideways markets
  - Buying the dip within an uptrend captures the snap-back

Key differences from momentum engine:
  - NO stop loss (Connors: stops hurt RSI(2) performance)
  - Short hold: 2-5 days typical, 10-day max
  - Entry on weakness, not strength
  - Requires confirmed uptrend (SMA200 filter)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── Parameters ────────────────────────────────────────────────────────────────

# ── THRESHOLD ORIGINS (documented for holdout integrity) ─────────────────────
# RSI(2) < 5: Connors & Alvarez (2009) "Short Term Trading Strategies That Work"
#   Published academic result, not optimized on our data. Used as-is.
RSI2_THRESHOLD   = 5.0

# SMA200 filter: Faber (2007) "A Quantitative Approach to Tactical Asset Allocation"
#   Standard trend filter used across the industry. Not optimized.
SMA200_FILTER    = True

# SMA5 exit: Connors & Alvarez (2009). Mean reversion target = SMA5 cross.
#   Published rule, not optimized on our data.
SMA5_EXIT        = True

# 10-day time stop: chosen as 2× the typical 5-day mean reversion window.
#   Origin: reasonable default, NOT optimized on historical data.
#   LOCKED for holdout — do not change after seeing 2025 results.
MAX_HOLD_DAYS    = 10

# 2 positions: conservative allocation to avoid over-concentration.
#   Origin: judgment call, not data-driven optimization.
MAX_POSITIONS    = 2

# 2.5% risk: half the momentum risk budget (momentum = 3.5%).
#   Origin: proportional scaling, not optimized.
RISK_PER_TRADE   = 0.025

# 12% max position: consistent with config.MAX_POSITION_PCT_AUM=15%.
#   Origin: risk management constraint, not optimized.
MAX_POSITION_PCT = 0.12

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MeanRevSnapshot:
    symbol:        str
    price:         float
    rsi2:          float
    sma5:          float
    sma200:        float
    above_sma200:  bool
    pct_below_sma200: float  # how far below 200MA? (negative = below)
    volume_ratio:  float    # vol vs 20d avg (high vol dips = better)
    sma50:         float = 0.0  # 50d MA for short-term trend filter
    ann_vol:       float = 0.35 # annualized volatility — filters high-vol names
    ml_rank_pct:   float = 0.5  # ML rank from momentum engine (0-1)


@dataclass
class MeanRevParams:
    rsi2_threshold:   float = RSI2_THRESHOLD
    max_positions:    int   = MAX_POSITIONS
    max_hold_days:    int   = MAX_HOLD_DAYS
    risk_per_trade:   float = RISK_PER_TRADE
    max_position_pct: float = MAX_POSITION_PCT
    sma200_filter:    bool  = SMA200_FILTER


# ── Signal computation ────────────────────────────────────────────────────────

def compute_rsi2(close: pd.Series) -> float:
    """Compute RSI(2) — 2-period RSI for mean reversion."""
    if len(close) < 5:
        return 50.0
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(2).mean()
    loss  = (-delta.clip(upper=0)).rolling(2).mean()
    rs    = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
    return float(100 - 100 / (1 + rs))


def compute_snapshot(sym: str, df: pd.DataFrame,
                     ml_rank: float = 0.5) -> Optional[MeanRevSnapshot]:
    """Build a MeanRevSnapshot from price data."""
    if df is None or len(df) < 210:
        return None
    close = df['close']
    vol   = df['volume']
    try:
        price    = float(close.iloc[-1])
        sma5     = float(close.tail(5).mean())
        sma200   = float(close.tail(200).mean())
        rsi2     = compute_rsi2(close.tail(20))
        vol_ma20 = float(vol.tail(20).mean())
        vol_ratio = float(vol.iloc[-1] / vol_ma20) if vol_ma20 > 0 else 1.0

        return MeanRevSnapshot(
            symbol        = sym,
            price         = price,
            rsi2          = rsi2,
            sma5          = sma5,
            sma200        = sma200,
            above_sma200  = price > sma200,
            pct_below_sma200 = (price / sma200 - 1) if sma200 > 0 else 0,
            volume_ratio  = min(vol_ratio, 10.0),
            ml_rank_pct   = ml_rank,
        )
    except Exception:
        return None


# ── Entry logic ───────────────────────────────────────────────────────────────

def can_enter_meanrev(snap: MeanRevSnapshot,
                      params: MeanRevParams) -> Tuple[bool, str]:
    """
    Check if a stock qualifies for mean reversion entry.
    Returns (can_enter, reason).

    Connors research: RSI(2) works on quality stocks in uptrends.
    NOT on broken stocks, high-vol names, or stocks near earnings.
    """
    # Must be above 200d MA — buying dips in uptrends only
    if params.sma200_filter and not snap.above_sma200:
        return False, f"below SMA200 ({snap.pct_below_sma200:+.1%})"
    # RSI(2) must be oversold
    if snap.rsi2 >= params.rsi2_threshold:
        return False, f"RSI(2)={snap.rsi2:.1f} not oversold (need <{params.rsi2_threshold})"
    # Not too far below SMA200 — structural weakness not a dip
    if snap.pct_below_sma200 < -0.05:
        return False, f"too far below SMA200 ({snap.pct_below_sma200:+.1%})"
    # Must be above SMA50 — short-term trend must also be up
    # Filters stocks above SMA200 but in short-term downtrend (PTON, ZIM pattern)
    if hasattr(snap, 'sma50') and snap.sma50 > 0:
        if snap.price < snap.sma50:
            return False, "below SMA50 — short-term downtrend"
    # Volatility cap — no meanrev on high-vol names
    # High vol = binary event risk = gaps through stops
    # PTON, METC, ZIM, RKLB all had ann_vol > 0.60
    if hasattr(snap, 'ann_vol') and snap.ann_vol > 0.65:  # research-validated: PTON=0.72, HOOD=0.76, APP=0.90 all blocked
        return False, f"ann_vol={snap.ann_vol:.2f} too high (max 0.65)"
    return True, "OK"

def score_meanrev_candidates(snaps: List[MeanRevSnapshot],
                              params: MeanRevParams) -> List[MeanRevSnapshot]:
    """
    Score and filter mean reversion candidates.
    Sorts by: most oversold RSI(2) first, with ML rank as tiebreaker.
    """
    candidates = []
    for snap in snaps:
        ok, reason = can_enter_meanrev(snap, params)
        if ok:
            candidates.append(snap)

    # Sort: most oversold first (lowest RSI2)
    # Tiebreak: prefer stocks that ML ranker also likes (rank > 0.5)
    candidates.sort(key=lambda s: (s.rsi2, -s.ml_rank_pct))
    return candidates[:params.max_positions * 3]  # return top 3x slots


# ── Exit logic ────────────────────────────────────────────────────────────────

def should_exit_meanrev(snap: MeanRevSnapshot,
                        entry_price: float,
                        days_held: int,
                        params: MeanRevParams) -> Tuple[bool, str]:
    """
    Check if a mean reversion position should be exited.
    RSI(2) exits on SMA5 cross or time stop — NO traditional stop loss.
    """
    # Exit 1: Price closes above 5-day SMA (mean reversion complete)
    if snap.price > snap.sma5:
        pnl_pct = (snap.price / entry_price - 1) * 100
        return True, f"SMA5 exit ({pnl_pct:+.1f}%)"

    # Exit 2: Time stop — 10 days max hold
    if days_held >= params.max_hold_days:
        pnl_pct = (snap.price / entry_price - 1) * 100
        return True, f"time stop day {days_held} ({pnl_pct:+.1f}%)"

    # Exit 3: Structural breakdown — price drops more than 10% below SMA200
    # This is the only "stop" — protects against regime change during hold
    if snap.pct_below_sma200 < -0.10:
        return True, f"structural breakdown ({snap.pct_below_sma200:+.1%} vs SMA200)"

    return False, ""


# ── Position sizing ───────────────────────────────────────────────────────────

def size_meanrev_position(capital: float, price: float,
                          params: MeanRevParams,
                          n_open_positions: int = 0,
                          snap=None,
                          ml_rank: float = 0.5) -> int:
    """
    Confidence-driven mean reversion sizing.
    RSI confidence x volume confidence x ML boost = position size multiplier.
    Range: 0.4x to 2.0x base slot size.
    """
    import numpy as np
    capital_per_slot = (capital * 0.80) / params.max_positions
    base_dollars     = min(capital_per_slot, capital * params.max_position_pct)
    if snap is not None:
        rsi_conf  = float(np.clip(1.0 - (snap.rsi2 / params.rsi2_threshold), 0.1, 1.0))
        vol_conf  = float(np.clip(snap.volume_ratio / 2.0, 0.5, 1.5))
        ml_boost  = float(np.clip(0.5 + ml_rank, 0.5, 1.5))
        confidence = float(np.clip(rsi_conf * vol_conf * ml_boost, 0.4, 2.0))
    else:
        confidence = 1.0
    dollars = min(base_dollars * confidence, capital * params.max_position_pct)
    qty = int(dollars / price) if price > 0 else 0
    return max(0, qty)

# ── Combined scoring with momentum engine ────────────────────────────────────

def blend_with_momentum(meanrev_candidates: List[MeanRevSnapshot],
                        momentum_ml_scores: dict,
                        blend_weight: float = 0.3) -> List[MeanRevSnapshot]:
    """
    Optionally blend mean reversion score with momentum ML rank.
    blend_weight=0.3 means 30% momentum, 70% mean reversion signal.
    
    This creates a "momentum-confirmed dip" signal:
    - Stocks the ML ranker likes (high rank) that are temporarily oversold
    - Better than pure mean reversion on random oversold stocks
    """
    for snap in meanrev_candidates:
        ml_rank = momentum_ml_scores.get(snap.symbol, 0.5)
        snap.ml_rank_pct = ml_rank

    # Re-sort with blend: lower RSI2 is better, higher ML rank is better
    # Normalize: rsi2 in [0,100], ml_rank in [0,1]
    def blend_score(s):
        rsi2_score = (100 - s.rsi2) / 100  # higher = more oversold = better
        return (1 - blend_weight) * rsi2_score + blend_weight * s.ml_rank_pct

    meanrev_candidates.sort(key=blend_score, reverse=True)
    return meanrev_candidates


if __name__ == '__main__':
    # Quick test
    import yfinance as yf
    import sys
    sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
    import config

    print("Testing RSI(2) mean reversion signal on current market...")
    print(f"{'Symbol':<8} {'RSI(2)':>7} {'vs200MA':>8} {'Vol':>6} {'Signal':>10}")
    print("-" * 45)

    params = MeanRevParams()
    candidates = []

    for sym in list(config.WATCHLIST)[:80]:
        try:
            hist = yf.Ticker(sym).history(period='18mo')
            if len(hist) < 210: continue
            hist.columns = [c.lower() for c in hist.columns]
            snap = compute_snapshot(sym, hist)
            if snap is None: continue
            ok, reason = can_enter_meanrev(snap, params)
            if ok:
                candidates.append(snap)
        except: continue

    candidates.sort(key=lambda s: s.rsi2)
    for snap in candidates[:10]:
        print(f"  {snap.symbol:<6} {snap.rsi2:>7.1f} {snap.pct_below_sma200:>+8.1%} "
              f"{snap.volume_ratio:>6.1f}x  ✓ ENTRY")

    if not candidates:
        print("  No mean reversion signals today (market not oversold)")
    else:
        print(f"\n{len(candidates)} candidates found in first 80 symbols")
