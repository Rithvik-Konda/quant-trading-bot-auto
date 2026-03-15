"""
pead_signal.py — Post-Earnings Announcement Drift Signal
=========================================================
Fetches earnings surprise data from yfinance and computes a
point-in-time PEAD score for each stock on each date.

Academic basis: Stocks that beat earnings estimates by >5% drift
upward for 60-90 days post-announcement. This is one of the most
robust anomalies in finance, documented since the 1960s.

Key design decisions:
- Uses yfinance earnings_dates for actual vs estimate (split-adjusted)
- Caches to disk — no re-fetch on every backtest run
- Point-in-time safe — only uses surprises BEFORE current date
- Score decays linearly from day 0 to day 90 post-announcement
- Clips surprise at ±50% to handle outliers (JPM 2008 etc)
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

CACHE_DIR  = "cache_earnings"
os.makedirs(CACHE_DIR, exist_ok=True)

# PEAD parameters
PEAD_DRIFT_DAYS       = 90    # drift window — score decays to zero after this
PEAD_MIN_SURPRISE_PCT = 5.0   # minimum surprise to generate a signal
PEAD_MAX_SURPRISE_PCT = 50.0  # clip outliers (JPM 2008 = -2318%)
PEAD_STRONG_BEAT      = 10.0  # above this = strong signal
PEAD_FETCH_LIMIT      = 100   # quarters to fetch per stock


def _cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_earnings.json")


def _fetch_earnings(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch earnings surprise data from yfinance with caching."""
    path = _cache_path(symbol)

    # Use cache if fresh (less than 7 days old)
    if os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400
        if age_days < 7:
            try:
                with open(path) as f:
                    records = json.load(f)
                if records:
                    df = pd.DataFrame(records)
                    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
                    return df.sort_values("date")
            except Exception:
                pass

    # Fetch from yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        raw = ticker.get_earnings_dates(limit=PEAD_FETCH_LIMIT)
        if raw is None or len(raw) == 0:
            return None

        records = []
        for dt, row in raw.iterrows():
            surprise = row.get("Surprise(%)")
            estimate = row.get("EPS Estimate")
            actual   = row.get("Reported EPS")

            # Skip future earnings (no actual yet)
            if pd.isna(actual) or pd.isna(surprise):
                continue

            # Normalize datetime — strip timezone for consistency
            if hasattr(dt, "tz_localize"):
                dt_clean = dt.tz_localize(None) if dt.tzinfo is None else dt.tz_convert(None)
            else:
                dt_clean = pd.Timestamp(dt).tz_localize(None) if pd.Timestamp(dt).tzinfo is None else pd.Timestamp(dt).tz_convert(None)

            records.append({
                "date":     dt_clean.isoformat(),
                "surprise": float(np.clip(surprise, -PEAD_MAX_SURPRISE_PCT, PEAD_MAX_SURPRISE_PCT)),
                "estimate": float(estimate) if not pd.isna(estimate) else None,
                "actual":   float(actual),
            })

        if not records:
            return None

        # Cache to disk
        with open(path, "w") as f:
            json.dump(records, f)

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")

    except Exception as e:
        print(f"  [pead] {symbol}: fetch error {e}")
        return None


def build_earnings_store(
    symbols: List[str],
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch and cache earnings data for all symbols.
    Returns dict: symbol → DataFrame with columns [date, surprise, estimate, actual]
    """
    store: Dict[str, pd.DataFrame] = {}
    total = len(symbols)
    for i, sym in enumerate(symbols):
        if verbose:
            print(f"  [pead] {i+1}/{total} {sym}          ", end="\r", flush=True)
        df = _fetch_earnings(sym)
        if df is not None and len(df) > 0:
            store[sym] = df
    if verbose:
        print(f"  [pead] loaded {len(store)}/{total} symbols with earnings data    ")
    return store


def pead_score(
    symbol: str,
    current_date: pd.Timestamp,
    earnings_store: Dict[str, pd.DataFrame],
) -> float:
    """
    Compute PEAD score for a symbol on a given date.

    Returns a score between 0 and 1:
    - 0.0: no recent earnings beat, or earnings miss
    - 0.5: modest beat (5-10%) early in drift window
    - 1.0: strong beat (>20%) within first 10 days

    Critically point-in-time safe: only uses earnings announced
    BEFORE current_date. Never looks ahead.
    """
    df = earnings_store.get(symbol)
    if df is None or len(df) == 0:
        return 0.0

    # Only earnings announced BEFORE today (point-in-time safe)
    past = df[df["date"] < current_date].copy()
    if len(past) == 0:
        return 0.0

    # Most recent earnings announcement
    latest = past.iloc[-1]
    surprise_pct = float(latest["surprise"])
    days_since   = (current_date - latest["date"]).days

    # Outside drift window — no signal
    if days_since > PEAD_DRIFT_DAYS or days_since < 0:
        return 0.0

    # Below minimum surprise threshold — no signal
    if surprise_pct < PEAD_MIN_SURPRISE_PCT:
        return 0.0

    # Decay factor: 1.0 at day 0, 0.0 at day 90
    decay = 1.0 - (days_since / PEAD_DRIFT_DAYS)

    # Surprise magnitude score: 5% = 0.5, 20%+ = 1.0
    surprise_score = min(1.0, (surprise_pct - PEAD_MIN_SURPRISE_PCT) / 15.0 + 0.5)

    # Combined score
    raw_score = surprise_score * decay

    return float(np.clip(raw_score, 0.0, 1.0))


def pead_score_batch(
    symbols: List[str],
    current_date: pd.Timestamp,
    earnings_store: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Compute PEAD scores for all symbols on a given date."""
    return {s: pead_score(s, current_date, earnings_store) for s in symbols}


def build_pead_feature_matrix(
    symbols: List[str],
    all_dates: pd.DatetimeIndex,
    earnings_store: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Pre-compute PEAD scores for all symbols × all dates.
    Returns DataFrame with symbols as columns, dates as index.
    Used for fast lookup during backtesting.
    """
    print("[pead] building PEAD feature matrix...", flush=True)
    data = {}
    total = len(symbols)
    for i, sym in enumerate(symbols):
        if (i+1) % 50 == 0:
            print(f"  [pead] {i+1}/{total}          ", end="\r", flush=True)
        scores = []
        for date in all_dates:
            scores.append(pead_score(sym, date, earnings_store))
        data[sym] = scores
    matrix = pd.DataFrame(data, index=all_dates)
    print(f"  [pead] matrix built: {matrix.shape}          ")
    return matrix


if __name__ == "__main__":
    # Test the signal on a few stocks
    import config
    symbols_test = ["NVDA", "MSFT", "AAPL", "JPM", "AMZN"]

    print("Fetching earnings data...")
    store = build_earnings_store(symbols_test, verbose=True)

    print("\nPEAD scores on key dates:")
    test_dates = [
        pd.Timestamp("2023-08-25"),  # 2 days after NVDA massive beat
        pd.Timestamp("2023-09-15"),  # 23 days after
        pd.Timestamp("2023-11-01"),  # 70 days after
        pd.Timestamp("2023-12-01"),  # 100 days after — should be 0
        pd.Timestamp("2024-02-25"),  # 4 days after NVDA next beat
    ]

    for date in test_dates:
        print(f"\n{date.date()}:")
        for sym in symbols_test:
            score = pead_score(sym, date, store)
            df = store.get(sym)
            if df is not None:
                past = df[df["date"] < date]
                if len(past) > 0:
                    latest = past.iloc[-1]
                    days = (date - latest["date"]).days
                    if score > 0:
                        print(f"  {sym:6s}: score={score:.3f}  surprise={latest['surprise']:.1f}%  days_since={days}")
