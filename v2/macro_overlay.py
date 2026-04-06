"""
macro_overlay.py — Macro Stress Detection Layer
================================================
Identifies when macro risk dominates fundamentals using 5 cross-asset signals.
Returns a position sizing scalar that reduces exposure during macro stress.

Signals: VIX spike, credit stress, correlation shock, VIX term inversion,
cross-asset flight to safety.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))

import numpy as np
import pandas as pd


def _load_cached(sym: str) -> Optional[pd.DataFrame]:
    """Load price data from cache."""
    cache_dir = os.path.expanduser("~/ai_trading_bot_v2/cache_prices")
    for suffix in ["_3650d.csv", "_9999d.csv", "_etf.csv", "_regime.csv"]:
        path = os.path.join(cache_dir, f"{sym}{suffix}")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            if "close" in df.columns:
                return df
    return None


def get_macro_signals(date: pd.Timestamp, price_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, bool]:
    """
    Compute 5 macro stress signals for a given date.

    Args:
        date: evaluation date
        price_data: dict of {symbol: DataFrame} with 'close' column.
                    If None, loads from cache.

    Returns:
        Dict of {signal_name: bool}
    """
    if price_data is None:
        price_data = {}
        for sym in ["SPY", "HYG", "^VIX", "TLT"]:
            df = _load_cached(sym)
            if df is not None:
                price_data[sym] = df

    vix_df  = price_data.get("^VIX", price_data.get("VIX"))
    spy_df  = price_data.get("SPY")
    hyg_df  = price_data.get("HYG")
    tlt_df  = price_data.get("TLT")

    signals = {
        "vix_spike": False,
        "credit_stress": False,
        "correlation_shock": False,
        "vix_term_inversion": False,
        "cross_asset_flight": False,
    }

    # 1. VIX spike: VIX > 25 AND VIX 5d change > +30%
    if vix_df is not None and "close" in vix_df.columns:
        vix = vix_df["close"].loc[:date]
        if len(vix) >= 6:
            vix_now = float(vix.iloc[-1])
            vix_5d_ago = float(vix.iloc[-6])
            vix_5d_chg = (vix_now - vix_5d_ago) / vix_5d_ago if vix_5d_ago > 0 else 0
            signals["vix_spike"] = vix_now > 25 and vix_5d_chg > 0.30

    # 2. Credit stress: HYG 10d return < -2%
    if hyg_df is not None and "close" in hyg_df.columns:
        hyg = hyg_df["close"].loc[:date]
        if len(hyg) >= 11:
            hyg_ret_10d = float(hyg.iloc[-1] / hyg.iloc[-11] - 1)
            signals["credit_stress"] = hyg_ret_10d < -0.02

    # 3. Correlation shock: rolling 10d corr between SPY returns and
    #    median stock return > 0.85 (everything moving together)
    if spy_df is not None and "close" in spy_df.columns:
        spy = spy_df["close"].loc[:date]
        if len(spy) >= 12:
            spy_rets = spy.pct_change().tail(10)
            # Use SPY vs its own lagged returns as proxy for market-wide correlation
            # (full cross-sectional correlation requires all stock data)
            spy_autocorr = float(spy_rets.autocorr(lag=1)) if len(spy_rets.dropna()) >= 5 else 0
            # High autocorrelation = trending/panic market = correlation shock proxy
            signals["correlation_shock"] = abs(spy_autocorr) > 0.85

    # 4. VIX term inversion: VIX > VIX 20d SMA by more than 20%
    if vix_df is not None and "close" in vix_df.columns:
        vix = vix_df["close"].loc[:date]
        if len(vix) >= 20:
            vix_now = float(vix.iloc[-1])
            vix_sma20 = float(vix.tail(20).mean())
            if vix_sma20 > 0:
                signals["vix_term_inversion"] = (vix_now / vix_sma20 - 1) > 0.20

    # 5. Cross-asset flight: TLT 5d return > +2% AND SPY 5d return < -2%
    if tlt_df is not None and spy_df is not None:
        tlt = tlt_df["close"].loc[:date] if "close" in tlt_df.columns else None
        spy = spy_df["close"].loc[:date] if "close" in spy_df.columns else None
        if tlt is not None and spy is not None and len(tlt) >= 6 and len(spy) >= 6:
            tlt_ret_5d = float(tlt.iloc[-1] / tlt.iloc[-6] - 1)
            spy_ret_5d = float(spy.iloc[-1] / spy.iloc[-6] - 1)
            signals["cross_asset_flight"] = tlt_ret_5d > 0.02 and spy_ret_5d < -0.02

    return signals


def get_macro_risk_score(date: pd.Timestamp, price_data: Optional[Dict[str, pd.DataFrame]] = None) -> float:
    """
    Returns macro risk score 0.0-1.0.
    Sum of active signals / 5.
    """
    signals = get_macro_signals(date, price_data)
    active = sum(1 for v in signals.values() if v)
    return active / 5.0


def get_macro_regime(date: pd.Timestamp, price_data: Optional[Dict[str, pd.DataFrame]] = None) -> str:
    """
    Returns macro regime string.
    - score >= 0.6: MACRO_STRESS
    - score >= 0.3: MACRO_CAUTION
    - else: NORMAL
    """
    score = get_macro_risk_score(date, price_data)
    if score >= 0.6:
        return "MACRO_STRESS"
    elif score >= 0.3:
        return "MACRO_CAUTION"
    return "NORMAL"


# Position scalar lookup: (macro_regime, market_regime) -> float
_POSITION_SCALARS = {
    ("MACRO_STRESS", "BEAR"):          0.0,
    ("MACRO_STRESS", "CHOPPY"):        0.25,
    ("MACRO_STRESS", "TRENDING_BULL"): 0.5,
    ("MACRO_CAUTION", "BEAR"):         0.25,
    ("MACRO_CAUTION", "CHOPPY"):       0.5,
    ("MACRO_CAUTION", "TRENDING_BULL"): 0.75,
}


def get_position_scalar(macro_regime: str, current_regime: str) -> float:
    """
    Returns position sizing scalar 0.0-1.0.
    Reduces exposure during macro stress periods.
    """
    return _POSITION_SCALARS.get((macro_regime, current_regime), 1.0)


if __name__ == "__main__":
    date = pd.Timestamp.now().normalize()
    signals = get_macro_signals(date)
    score = get_macro_risk_score(date)
    regime = get_macro_regime(date)

    print(f"Macro Overlay — {date.date()}")
    print(f"  Risk score: {score:.1f}")
    print(f"  Regime:     {regime}")
    print(f"  Signals:")
    for name, active in signals.items():
        print(f"    {name:25s} {'ACTIVE' if active else '-'}")
