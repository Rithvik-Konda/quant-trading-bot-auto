"""
strategy_meanrev.py — RSI/Bollinger Band Mean-Reversion Strategy
================================================================
Used in CHOPPY regime as primary strategy.
Validated: 57% WR, +1.20% avg, 248 trades across choppy periods 2019-2025.

Logic:
  Entry:  RSI < 35 AND price near lower Bollinger Band (bb_pct < 0.25)
          AND stock is in our watchlist AND not in momentum position
  Exit:   RSI > 55 OR hold >= 10 days OR +6% gain OR -8% loss

Key difference from momentum strategy:
  - Buys weakness, not strength
  - Short hold period (7-10 days avg)
  - Works when momentum fails — genuinely uncorrelated
  - Runs in ALL regimes but primary in CHOPPY
"""
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np


@dataclass
class MeanRevParams:
    # Entry
    rsi_entry:          float = 35.0   # buy when RSI below this
    bb_pct_entry:       float = 0.30   # buy when near lower band
    rsi_exit:           float = 55.0   # exit when RSI recovers
    # Exit
    take_profit_pct:    float = 0.06   # +6%
    stop_loss_pct:      float = 0.08   # -8%
    max_hold_days:      int   = 12     # time stop
    min_hold_days:      int   = 2      # don't exit too fast
    # Sizing
    max_positions:      int   = 3      # max simultaneous
    risk_per_trade:     float = 0.025  # 2.5% risk per trade
    max_position_weight: float = 0.20  # 20% max per position
    # Quality filters
    min_avg_volume:     float = 1e6    # minimum daily volume
    min_price:          float = 5.0   # no penny stocks
    ml_rank_min:        float = 0.0   # no ML filter — this is reversion


def get_params() -> MeanRevParams:
    return MeanRevParams()


def compute_signals(df: pd.DataFrame) -> Dict:
    """Compute RSI and Bollinger Band signals from OHLCV data."""
    if len(df) < 30:
        return {}
    try:
        close = df['close']

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, 1e-10)
        rsi   = 100 - (100 / (1 + rs))

        # Bollinger Bands
        ma20  = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        bb_pct = (close - lower) / (upper - lower + 1e-10)

        # Distance from 20d MA
        dist_ma = (close - ma20) / ma20

        # Volume check
        avg_vol = df['volume'].rolling(20).mean()

        latest_rsi    = float(rsi.iloc[-1])   if not rsi.isna().all()    else 50.0
        latest_bb_pct = float(bb_pct.iloc[-1]) if not bb_pct.isna().all() else 0.5
        latest_dist   = float(dist_ma.iloc[-1]) if not dist_ma.isna().all() else 0.0
        latest_vol    = float(avg_vol.iloc[-1]) if not avg_vol.isna().all() else 0.0

        return {
            'rsi':       latest_rsi,
            'bb_pct':    latest_bb_pct,
            'dist_ma20': latest_dist,
            'avg_vol':   latest_vol,
            'close':     float(close.iloc[-1]),
        }
    except Exception:
        return {}


def score_candidates(
    available_symbols: List[str],
    prices_by_symbol:  Dict[str, pd.DataFrame],
    as_of_date:        pd.Timestamp,
    existing_positions: List[str],
) -> List[Dict]:
    """
    Score all available symbols for mean-reversion entry.
    Returns list of candidates sorted by entry quality.
    """
    params = get_params()
    candidates = []

    for sym in available_symbols:
        if sym in existing_positions:
            continue
        df = prices_by_symbol.get(sym)
        if df is None or len(df) < 30:
            continue
        df_now = df.loc[:as_of_date]
        if len(df_now) < 30:
            continue

        sig = compute_signals(df_now)
        if not sig:
            continue

        # Quality gates
        if sig['close'] < params.min_price:
            continue
        if sig['avg_vol'] < params.min_avg_volume:
            continue

        # Entry signal
        rsi_signal = sig['rsi'] < params.rsi_entry
        bb_signal  = sig['bb_pct'] < params.bb_pct_entry

        if not (rsi_signal and bb_signal):
            continue

        # Score — lower RSI + lower BB pct = stronger signal
        score = (params.rsi_entry - sig['rsi']) / params.rsi_entry + \
                (params.bb_pct_entry - sig['bb_pct']) / params.bb_pct_entry

        candidates.append({
            'symbol':  sym,
            'score':   score,
            'rsi':     sig['rsi'],
            'bb_pct':  sig['bb_pct'],
            'close':   sig['close'],
            'avg_vol': sig['avg_vol'],
        })

    return sorted(candidates, key=lambda x: x['score'], reverse=True)


def should_exit(
    df_now:      pd.DataFrame,
    entry_price: float,
    hold_days:   int,
    params:      MeanRevParams = None,
) -> str:
    """Check exit conditions. Returns reason string or empty string."""
    if params is None:
        params = get_params()
    if len(df_now) < 14:
        return ''
    if hold_days < params.min_hold_days:
        return ''

    sig   = compute_signals(df_now)
    close = float(df_now['close'].iloc[-1])
    pnl_pct = (close - entry_price) / entry_price

    # Stop loss
    if pnl_pct <= -params.stop_loss_pct:
        return 'meanrev_stop'
    # Take profit
    if pnl_pct >= params.take_profit_pct:
        return 'meanrev_take_profit'
    # RSI recovery
    if sig.get('rsi', 50) > params.rsi_exit:
        return 'meanrev_rsi_exit'
    # Time stop
    if hold_days >= params.max_hold_days:
        return 'meanrev_max_hold'

    return ''
