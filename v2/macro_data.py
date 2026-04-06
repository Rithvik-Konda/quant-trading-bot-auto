"""
macro_data.py — Macro Data Pipeline (FRED + CBOE + yfinance)
=============================================================
Pulls credit spreads, yields, VIX term structure, put/call ratios,
and cross-asset returns. Caches to cache_macro/ with 24h TTL.

Usage:
    python v2/macro_data.py --build                    # full cache build
    python v2/macro_data.py --build --start 2020-01-01 # from specific date
"""
from __future__ import annotations

import os
import sys
import time
import requests
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))

import pandas as pd
import numpy as np

FRED_API_KEY = "bde40ee4b3e062d99fc64e98ceb6c74f"
CACHE_DIR    = Path(os.path.expanduser("~/ai_trading_bot_v2/cache_macro"))
CACHE_DIR.mkdir(exist_ok=True)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "BAMLH0A0HYM2": "hy_spread",
    "BAMLC0A0CM":    "ig_spread",
    "DGS10":         "yield_10y",
    "DGS2":          "yield_2y",
    "VIXCLS":        "vix_fred",
}

CROSS_ASSET_SYMBOLS = ["SPY", "TLT", "GLD", "UUP", "HYG", "LQD"]


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.csv"


def _cache_fresh(name: str, max_age_hours: float = 24.0) -> bool:
    path = _cache_path(name)
    if not path.exists():
        return False
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600
    return age < max_age_hours


def _load_cache(name: str) -> Optional[pd.DataFrame]:
    path = _cache_path(name)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception:
        return None


def _save_cache(df: pd.DataFrame, name: str) -> None:
    df.to_csv(_cache_path(name))


# ═══════════════════════════════════════════════════════════════════════════════
#  1. FRED SERIES
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_fred_series(series_id: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch a single FRED time series.

    Args:
        series_id: FRED series identifier e.g. 'DGS10'
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'

    Returns:
        pd.Series indexed by date with float values.
    """
    cache_name = f"fred_{series_id}"
    if _cache_fresh(cache_name):
        cached = _load_cache(cache_name)
        if cached is not None and len(cached) > 0:
            return cached.iloc[:, 0]

    params = {
        "series_id":         series_id,
        "observation_start": start_date,
        "observation_end":   end_date,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
    }

    try:
        resp = requests.get(FRED_BASE, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"  FRED {series_id}: HTTP {resp.status_code}")
            return pd.Series(dtype=float)

        data = resp.json()
        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float)

        dates = []
        values = []
        for o in obs:
            val = o.get("value", ".")
            if val == "." or val is None:
                continue
            try:
                dates.append(pd.Timestamp(o["date"]))
                values.append(float(val))
            except (ValueError, KeyError):
                continue

        series = pd.Series(values, index=dates, name=series_id)
        series = series[~series.index.duplicated(keep="last")]
        series = series.sort_index()

        _save_cache(series.to_frame(), cache_name)
        time.sleep(0.2)  # FRED rate limit: ~120 req/min
        return series

    except Exception as e:
        print(f"  FRED {series_id} failed: {e}")
        return pd.Series(dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. ALL FRED SERIES
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_all_fred(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch all tracked FRED series and combine into a DataFrame.
    Forward-fills weekends/holidays.

    Returns:
        DataFrame with columns: hy_spread, ig_spread, yield_10y, yield_2y, vix_fred
    """
    cache_name = "fred_combined"
    if _cache_fresh(cache_name):
        cached = _load_cache(cache_name)
        if cached is not None and len(cached) > 0:
            return cached

    frames = {}
    for series_id, col_name in FRED_SERIES.items():
        print(f"  Fetching FRED {series_id} ({col_name})...")
        s = fetch_fred_series(series_id, start_date, end_date)
        if len(s) > 0:
            frames[col_name] = s

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df = df.sort_index()
    df = df.ffill()  # forward-fill weekends/holidays

    _save_cache(df, cache_name)
    print(f"  FRED combined: {len(df)} rows, {list(df.columns)}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  3. VIX TERM STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_vix_term_structure(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch VIX spot and 3-month VIX from yfinance.

    Returns:
        DataFrame with columns: vix_spot, vix_3m
    """
    cache_name = "vix_term"
    if _cache_fresh(cache_name):
        cached = _load_cache(cache_name)
        if cached is not None and len(cached) > 0:
            return cached

    try:
        import yfinance as yf

        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        vix3m = yf.download("^VIX3M", start=start_date, end=end_date, progress=False)

        df = pd.DataFrame()
        if len(vix) > 0:
            close_col = "Close"
            if isinstance(vix.columns, pd.MultiIndex):
                close_col = ("Close", "^VIX")
            df["vix_spot"] = vix[close_col]
        if len(vix3m) > 0:
            close_col = "Close"
            if isinstance(vix3m.columns, pd.MultiIndex):
                close_col = ("Close", "^VIX3M")
            df["vix_3m"] = vix3m[close_col]

        if hasattr(df.index, "tz") and df.index.tz:
            df.index = df.index.tz_localize(None)
        df = df.ffill()

        _save_cache(df, cache_name)
        print(f"  VIX term structure: {len(df)} rows")
        return df

    except Exception as e:
        print(f"  VIX term structure failed: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  4. CBOE PUT/CALL RATIO
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_cboe_put_call(start_date: str, end_date: str) -> Optional[pd.Series]:
    """
    Fetch CBOE equity put/call ratio.
    Falls back to VIX-based approximation if CBOE scraping fails.

    Returns:
        pd.Series of daily put/call ratio, or None.
    """
    cache_name = "put_call"
    if _cache_fresh(cache_name):
        cached = _load_cache(cache_name)
        if cached is not None and len(cached) > 0:
            return cached.iloc[:, 0]

    # Try CBOE direct download
    try:
        url = "https://www.cboe.com/us/options/market_statistics/daily/"
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and "text/csv" in resp.headers.get("Content-Type", ""):
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), parse_dates=["TRADE_DATE"])
            if "P_C_RATIO" in df.columns:
                series = df.set_index("TRADE_DATE")["P_C_RATIO"]
                _save_cache(series.to_frame(), cache_name)
                print(f"  CBOE put/call: {len(series)} rows")
                return series
    except Exception:
        pass

    # Fallback: approximate from cached PCR history
    pcr_path = os.path.expanduser("~/ai_trading_bot_v2/cache_pcr/pcr_history.csv")
    if os.path.exists(pcr_path):
        try:
            df = pd.read_csv(pcr_path, index_col=0, parse_dates=True)
            if "pcr" in df.columns:
                series = df["pcr"]
                _save_cache(series.to_frame(), cache_name)
                print(f"  Put/call (from pcr_history cache): {len(series)} rows")
                return series
        except Exception:
            pass

    # Fallback: VIX-based approximation
    # PCR ≈ 0.5 + 0.02 × (VIX - 15), clipped to [0.3, 2.0]
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        if len(vix) > 0:
            close_col = "Close"
            if isinstance(vix.columns, pd.MultiIndex):
                close_col = ("Close", "^VIX")
            vix_close = vix[close_col]
            pcr_approx = (0.5 + 0.02 * (vix_close - 15)).clip(0.3, 2.0)
            pcr_approx.name = "pcr_approx"
            _save_cache(pcr_approx.to_frame(), cache_name)
            print(f"  Put/call (VIX approximation): {len(pcr_approx)} rows")
            return pcr_approx
    except Exception:
        pass

    print("  Put/call ratio: all sources failed")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  5. CROSS-ASSET RETURNS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_cross_asset(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily returns for cross-asset ETFs: SPY, TLT, GLD, UUP, HYG, LQD.

    Returns:
        DataFrame of daily returns indexed by date.
    """
    cache_name = "cross_asset"
    if _cache_fresh(cache_name):
        cached = _load_cache(cache_name)
        if cached is not None and len(cached) > 0:
            return cached

    try:
        import yfinance as yf

        prices = yf.download(CROSS_ASSET_SYMBOLS, start=start_date, end=end_date, progress=False)
        if len(prices) == 0:
            return pd.DataFrame()

        if isinstance(prices.columns, pd.MultiIndex):
            close = prices["Close"]
        else:
            close = prices[["Close"]].rename(columns={"Close": CROSS_ASSET_SYMBOLS[0]})

        returns = close.pct_change().dropna()
        if hasattr(returns.index, "tz") and returns.index.tz:
            returns.index = returns.index.tz_localize(None)

        _save_cache(returns, cache_name)
        print(f"  Cross-asset returns: {len(returns)} rows × {len(returns.columns)} assets")
        return returns

    except Exception as e:
        print(f"  Cross-asset fetch failed: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  6. BUILD FULL CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def build_macro_cache(start_date: str = "2015-01-01", end_date: Optional[str] = None) -> None:
    """
    Fetch all macro data sources and save to cache_macro/.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"  MACRO DATA PIPELINE — Building Cache")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Cache:  {CACHE_DIR}")
    print("=" * 60)

    print("\n[1/4] FRED series (credit spreads, yields, VIX)...")
    fred = fetch_all_fred(start_date, end_date)

    print("\n[2/4] VIX term structure (spot vs 3M)...")
    vix_term = fetch_vix_term_structure(start_date, end_date)

    print("\n[3/4] Put/call ratio...")
    pcr = fetch_cboe_put_call(start_date, end_date)

    print("\n[4/4] Cross-asset returns (SPY, TLT, GLD, UUP, HYG, LQD)...")
    cross = fetch_cross_asset(start_date, end_date)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  FRED:        {len(fred)} rows, {list(fred.columns) if len(fred) else 'empty'}")
    print(f"  VIX term:    {len(vix_term)} rows")
    print(f"  Put/call:    {len(pcr) if pcr is not None else 0} rows")
    print(f"  Cross-asset: {len(cross)} rows × {len(cross.columns) if len(cross) else 0} assets")
    print(f"\n  Cache dir:   {CACHE_DIR}")
    print(f"  Files:")
    for f in sorted(CACHE_DIR.glob("*.csv")):
        size = f.stat().st_size
        print(f"    {f.name:30s} {size // 1024:>6d} KB")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Macro Data Pipeline")
    parser.add_argument("--build", action="store_true", help="Build full macro cache")
    parser.add_argument("--start", default="2015-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    args = parser.parse_args()

    if args.build:
        build_macro_cache(start_date=args.start, end_date=args.end)
    else:
        # Show cache status
        print(f"Macro cache: {CACHE_DIR}")
        if CACHE_DIR.exists():
            for f in sorted(CACHE_DIR.glob("*.csv")):
                age = (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime))
                print(f"  {f.name:30s} {f.stat().st_size // 1024:>6d} KB  "
                      f"age={age.total_seconds() / 3600:.1f}h")
        else:
            print("  No cache. Run with --build")
