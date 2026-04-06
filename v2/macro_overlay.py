"""
macro_overlay.py — Macro Stress Detection Layer
================================================
8 cross-asset signals from FRED, VIX term structure, and ETF returns.
Uses real data from cache_macro/ (built by macro_data.py).
Returns a position sizing scalar that reduces exposure during macro stress.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))
sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2/v2"))

import numpy as np
import pandas as pd

# ── Tunable thresholds — change here, not in signal logic ────────────────────
MACRO_CONFIG = {
    "vix_spike_level":     22,      # VIX must be above this
    "vix_spike_5d_chg":    0.15,    # VIX 5d % change must exceed this
    "hy_spread_5d_bps":    0.25,    # HY OAS 5d change in percentage points
    "ig_spread_5d_bps":    0.10,    # IG OAS 5d change in percentage points
    "put_call_level":      1.0,     # put/call ratio above this = stress
    "hyg_10d_return":     -0.01,    # HYG cumulative 10d return below this
    "tlt_5d_return":       0.02,    # TLT 5d return above this (flight to safety)
    "spy_5d_return":      -0.02,    # SPY 5d return below this (equity selloff)
    "spy_10d_return":     -0.03,    # SPY 10d return below this (extended selloff)
    "stress_threshold":    0.375,   # risk score >= this → MACRO_STRESS
    "caution_threshold":   0.125,   # risk score >= this → MACRO_CAUTION
}

# ── Module-level data cache (loaded once, reloaded if stale) ─────────────────

_CACHE_DIR = os.path.expanduser("~/ai_trading_bot_v2/cache_macro")
_fred_data: Optional[pd.DataFrame] = None
_vix_term: Optional[pd.DataFrame] = None
_cross_asset: Optional[pd.DataFrame] = None
_pcr_data: Optional[pd.Series] = None
_cache_loaded_at: float = 0


def _load_macro_cache() -> None:
    """Load all macro data from cache_macro/. Reload if > 24h old."""
    global _fred_data, _vix_term, _cross_asset, _pcr_data, _cache_loaded_at
    import time

    now = time.time()
    if _cache_loaded_at > 0 and (now - _cache_loaded_at) < 86400:
        return  # still fresh

    def _read(name):
        path = os.path.join(_CACHE_DIR, f"{name}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if hasattr(df.index, "tz") and df.index.tz:
                    df.index = df.index.tz_localize(None)
                df.index = df.index.normalize()
                return df
            except Exception:
                return None
        return None

    _fred_data = _read("fred_combined")
    _vix_term = _read("vix_term")
    _cross_asset = _read("cross_asset")

    pcr_df = _read("put_call")
    _pcr_data = pcr_df.iloc[:, 0] if pcr_df is not None and len(pcr_df.columns) > 0 else None

    # Fallback: load from cache_prices/ if cache_macro/ is empty
    if _vix_term is None or len(_vix_term) == 0:
        _vix_term = _load_vix_fallback()
    if _cross_asset is None or len(_cross_asset) == 0:
        _cross_asset = _load_cross_asset_fallback()

    _cache_loaded_at = now


def _load_vix_fallback() -> Optional[pd.DataFrame]:
    """Load VIX from cache_prices/ as fallback."""
    cache_dir = os.path.expanduser("~/ai_trading_bot_v2/cache_prices")
    for name in ["^VIX_etf.csv", "^VIX_9999d.csv"]:
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.columns = [c.lower() for c in df.columns]
                if hasattr(df.index, "tz") and df.index.tz:
                    df.index = df.index.tz_localize(None)
                df.index = df.index.normalize()
                if "close" in df.columns:
                    return pd.DataFrame({"vix_spot": df["close"]})
            except Exception:
                pass
    return None


def _load_cross_asset_fallback() -> Optional[pd.DataFrame]:
    """Build cross-asset returns from cache_prices/ as fallback."""
    cache_dir = os.path.expanduser("~/ai_trading_bot_v2/cache_prices")
    frames = {}
    for sym in ["SPY", "TLT", "HYG"]:
        for suffix in ["_3650d.csv", "_9999d.csv", "_etf.csv"]:
            path = os.path.join(cache_dir, f"{sym}{suffix}")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, index_col=0, parse_dates=True)
                    df.columns = [c.lower() for c in df.columns]
                    if hasattr(df.index, "tz") and df.index.tz:
                        df.index = df.index.tz_localize(None)
                    df.index = df.index.normalize()
                    if "close" in df.columns:
                        frames[sym] = df["close"].pct_change()
                except Exception:
                    pass
                break
    return pd.DataFrame(frames) if frames else None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION — 8 real-data signals
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_val(series, date, offset=0):
    """Get value from series at date-offset, returns NaN on failure."""
    if series is None or len(series) == 0:
        return np.nan
    s = series.loc[:date]
    idx = -(1 + offset)
    if len(s) >= abs(idx):
        return float(s.iloc[idx])
    return np.nan


def get_macro_signals(date: pd.Timestamp, price_data: Optional[Dict] = None) -> Dict[str, bool]:
    """
    Compute 8 macro stress signals using real FRED + VIX term + cross-asset data.

    Returns:
        Dict of {signal_name: bool}
    """
    _load_macro_cache()

    signals = {
        "vix_spike": False,
        "vix_term_inversion": False,
        "hy_spread_spike": False,
        "ig_spread_spike": False,
        "put_call_spike": False,
        "credit_stress": False,
        "cross_asset_flight": False,
        "yield_curve_stress": False,
    }

    # 1. VIX spike: VIX spot > level AND 5d change > threshold
    if _vix_term is not None and "vix_spot" in _vix_term.columns:
        vix_now = _safe_val(_vix_term["vix_spot"], date)
        vix_5d = _safe_val(_vix_term["vix_spot"], date, 5)
        if not np.isnan(vix_now) and not np.isnan(vix_5d) and vix_5d > 0:
            signals["vix_spike"] = (vix_now > MACRO_CONFIG["vix_spike_level"] and
                                    (vix_now / vix_5d - 1) > MACRO_CONFIG["vix_spike_5d_chg"])

    # 2. VIX term inversion: vix_spot > vix_3m (fear spike, not chronic)
    if _vix_term is not None and "vix_spot" in _vix_term.columns and "vix_3m" in _vix_term.columns:
        vix_spot = _safe_val(_vix_term["vix_spot"], date)
        vix_3m = _safe_val(_vix_term["vix_3m"], date)
        if not np.isnan(vix_spot) and not np.isnan(vix_3m) and vix_3m > 0:
            signals["vix_term_inversion"] = vix_spot > vix_3m

    # 3. HY spread spike: 5d change > threshold
    if _fred_data is not None and "hy_spread" in _fred_data.columns:
        hy_now = _safe_val(_fred_data["hy_spread"], date)
        hy_5d = _safe_val(_fred_data["hy_spread"], date, 5)
        if not np.isnan(hy_now) and not np.isnan(hy_5d):
            signals["hy_spread_spike"] = (hy_now - hy_5d) > MACRO_CONFIG["hy_spread_5d_bps"]

    # 4. IG spread spike: 5d change > threshold
    if _fred_data is not None and "ig_spread" in _fred_data.columns:
        ig_now = _safe_val(_fred_data["ig_spread"], date)
        ig_5d = _safe_val(_fred_data["ig_spread"], date, 5)
        if not np.isnan(ig_now) and not np.isnan(ig_5d):
            signals["ig_spread_spike"] = (ig_now - ig_5d) > MACRO_CONFIG["ig_spread_5d_bps"]

    # 5. Put/call spike: ratio > threshold
    if _pcr_data is not None:
        pcr = _safe_val(_pcr_data, date)
        if not np.isnan(pcr):
            signals["put_call_spike"] = pcr > MACRO_CONFIG["put_call_level"]

    # 6. Credit stress: HYG 10d return < threshold
    if _cross_asset is not None and "HYG" in _cross_asset.columns:
        hyg_rets = _cross_asset["HYG"].loc[:date]
        if len(hyg_rets) >= 10:
            hyg_10d = float(hyg_rets.tail(10).sum())
            signals["credit_stress"] = hyg_10d < MACRO_CONFIG["hyg_10d_return"]

    # 7. Cross-asset flight: (TLT up AND SPY down) OR SPY extended selloff
    if _cross_asset is not None:
        has_tlt = "TLT" in _cross_asset.columns
        has_spy = "SPY" in _cross_asset.columns
        if has_spy:
            spy_rets = _cross_asset["SPY"].loc[:date]
            tlt_rets = _cross_asset["TLT"].loc[:date] if has_tlt else None

            flight = False
            if tlt_rets is not None and len(tlt_rets) >= 5 and len(spy_rets) >= 5:
                tlt_5d = float(tlt_rets.tail(5).sum())
                spy_5d = float(spy_rets.tail(5).sum())
                flight = tlt_5d > MACRO_CONFIG["tlt_5d_return"] and spy_5d < MACRO_CONFIG["spy_5d_return"]
            if not flight and len(spy_rets) >= 10:
                spy_10d = float(spy_rets.tail(10).sum())
                flight = spy_10d < MACRO_CONFIG["spy_10d_return"]
            signals["cross_asset_flight"] = flight

    # 8. Yield curve stress: 2Y-10Y inversion > 50bps AND widening
    if _fred_data is not None and "yield_2y" in _fred_data.columns and "yield_10y" in _fred_data.columns:
        y2_now = _safe_val(_fred_data["yield_2y"], date)
        y10_now = _safe_val(_fred_data["yield_10y"], date)
        y2_5d = _safe_val(_fred_data["yield_2y"], date, 5)
        y10_5d = _safe_val(_fred_data["yield_10y"], date, 5)
        if not any(np.isnan(x) for x in [y2_now, y10_now, y2_5d, y10_5d]):
            inversion_now = y2_now - y10_now
            inversion_5d = y2_5d - y10_5d
            signals["yield_curve_stress"] = inversion_now > 0.50 and inversion_now > inversion_5d  # unchanged — 50bps is structural, not tunable

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
#  RISK SCORE, REGIME, POSITION SCALAR
# ═══════════════════════════════════════════════════════════════════════════════

N_SIGNALS = 8


def get_macro_risk_score(date: pd.Timestamp, price_data: Optional[Dict] = None) -> float:
    """
    Returns macro risk score 0.0-1.0.
    Sum of active signals / 8.
    """
    signals = get_macro_signals(date, price_data)
    active = sum(1 for v in signals.values() if v)
    return active / N_SIGNALS


def get_macro_regime(date: pd.Timestamp, price_data: Optional[Dict] = None) -> str:
    """
    Returns macro regime string.
    Thresholds from MACRO_CONFIG:
    - score >= stress_threshold: MACRO_STRESS
    - score >= caution_threshold: MACRO_CAUTION
    - else: NORMAL
    """
    score = get_macro_risk_score(date, price_data)
    if score >= MACRO_CONFIG["stress_threshold"]:
        return "MACRO_STRESS"
    elif score >= MACRO_CONFIG["caution_threshold"]:
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


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    date = pd.Timestamp.now().normalize()
    signals = get_macro_signals(date)
    score = get_macro_risk_score(date)
    regime = get_macro_regime(date)

    print(f"Macro Overlay — {date.date()}")
    print(f"  Risk score: {score:.2f} ({int(score * N_SIGNALS)}/{N_SIGNALS} signals)")
    print(f"  Regime:     {regime}")
    print(f"  Signals:")
    for name, active in signals.items():
        print(f"    {name:25s} {'ACTIVE' if active else '-'}")
    print(f"\n  Position scalars:")
    for mkt in ["TRENDING_BULL", "CHOPPY", "BEAR"]:
        scalar = get_position_scalar(regime, mkt)
        print(f"    {regime} + {mkt}: {scalar:.2f}")
