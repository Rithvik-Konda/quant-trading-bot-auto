"""
options_flow.py — ThetaData Options Flow Aggregator
====================================================
Queries ThetaData local terminal for put/call ratios and unusual activity.
Requires ThetaData terminal running at http://127.0.0.1:25503.

Usage:
    python v2/options_flow.py --date 2024-06-15 --symbols NVDA AAPL META
    python v2/options_flow.py --unusual --date 2024-06-15
"""
from __future__ import annotations

import os
import sys
import json
import time
import requests
from datetime import datetime
from typing import List, Optional
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))

import pandas as pd
import numpy as np

THETADATA_BASE = "http://127.0.0.1:25503"
CACHE_DIR      = Path(os.path.expanduser("~/ai_trading_bot_v2/cache_macro"))
CACHE_DIR.mkdir(exist_ok=True)

RATE_LIMIT_SEC = 3  # ThetaData rate limit


# ═══════════════════════════════════════════════════════════════════════════════
#  1. SYMBOL PUT/CALL RATIO
# ═══════════════════════════════════════════════════════════════════════════════

def get_symbol_put_call_ratio(symbol: str, date_str: str) -> float:
    """
    Get put/call volume ratio for a single symbol on a given date.

    Args:
        symbol: stock ticker e.g. 'NVDA'
        date_str: 'YYYYMMDD' format

    Returns:
        put_volume / call_volume, or 1.0 on failure.
    """
    url = f"{THETADATA_BASE}/v3/option/quote"
    params = {
        "root": symbol,
        "date": date_str,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        time.sleep(RATE_LIMIT_SEC)

        if resp.status_code != 200:
            return 1.0

        data = resp.json()
        rows = data.get("response", [])
        if not rows:
            return 1.0

        put_vol = 0
        call_vol = 0

        for row in rows:
            if not isinstance(row, dict):
                continue
            right = row.get("right", row.get("contract", {}).get("right", ""))
            volume = row.get("volume", row.get("ms_of_day2", 0))

            # Try to extract volume from various field names
            if volume == 0:
                volume = row.get("count", 0)
            if volume == 0:
                volume = row.get("size", 0)

            try:
                volume = int(volume)
            except (ValueError, TypeError):
                volume = 0

            if isinstance(right, str):
                right = right.upper()
            if right in ("P", "PUT"):
                put_vol += volume
            elif right in ("C", "CALL"):
                call_vol += volume

        if call_vol == 0:
            return 1.0

        return put_vol / call_vol

    except requests.exceptions.ConnectionError:
        return 1.0
    except requests.exceptions.Timeout:
        return 1.0
    except Exception:
        return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  2. MARKET PUT/CALL RATIO
# ═══════════════════════════════════════════════════════════════════════════════

def get_market_put_call_ratio(
    symbols: List[str],
    date_str: str,
    max_symbols: int = 50,
) -> float:
    """
    Compute market-wide put/call ratio from a sample of symbols.

    Args:
        symbols: list of tickers
        date_str: 'YYYYMMDD' format
        max_symbols: max symbols to query (for rate limiting)

    Returns:
        Volume-weighted average put/call ratio across sampled symbols.
    """
    cache_file = CACHE_DIR / f"options_flow_{date_str}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if "market_pc_ratio" in cached:
                return cached["market_pc_ratio"]
        except Exception:
            pass

    sample = symbols[:max_symbols]
    ratios = []
    symbol_ratios = {}

    for sym in sample:
        ratio = get_symbol_put_call_ratio(sym, date_str)
        ratios.append(ratio)
        symbol_ratios[sym] = ratio

    if not ratios:
        return 1.0

    # Simple average (dollar-volume weighting would require price data)
    market_ratio = float(np.mean(ratios))

    # Cache result
    result = {
        "date": date_str,
        "market_pc_ratio": round(market_ratio, 4),
        "n_symbols": len(ratios),
        "symbol_ratios": {k: round(v, 4) for k, v in symbol_ratios.items()},
    }
    try:
        cache_file.write_text(json.dumps(result, indent=2))
    except Exception:
        pass

    return market_ratio


# ═══════════════════════════════════════════════════════════════════════════════
#  3. UNUSUAL PUT ACTIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def get_unusual_put_activity(
    symbols: List[str],
    date_str: str,
    threshold: float = 3.0,
) -> List[str]:
    """
    Find symbols with unusual put volume (today > threshold × 20d average).

    Args:
        symbols: list of tickers to scan
        date_str: 'YYYYMMDD' format for today's date
        threshold: multiple of average to trigger alert (default 3.0)

    Returns:
        List of symbols with unusual put activity.
    """
    cache_file = CACHE_DIR / f"unusual_puts_{date_str}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            return cached.get("unusual_symbols", [])
        except Exception:
            pass

    # Parse date for lookback
    try:
        today = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return []

    unusual = []
    details = {}

    for sym in symbols:
        # Get today's put/call data
        today_ratio = get_symbol_put_call_ratio(sym, date_str)

        # Get historical ratios (sample 5 dates from last 20 trading days)
        hist_ratios = []
        for offset in [5, 10, 15, 20, 25]:
            hist_date = today - pd.Timedelta(days=offset)
            # Skip weekends
            while hist_date.weekday() >= 5:
                hist_date -= pd.Timedelta(days=1)
            hist_str = hist_date.strftime("%Y%m%d")
            hr = get_symbol_put_call_ratio(sym, hist_str)
            if hr != 1.0:  # 1.0 is the default/failure value
                hist_ratios.append(hr)

        if not hist_ratios:
            continue

        avg_ratio = np.mean(hist_ratios)
        if avg_ratio > 0 and today_ratio > threshold * avg_ratio:
            unusual.append(sym)
            details[sym] = {
                "today_ratio": round(today_ratio, 4),
                "avg_ratio": round(avg_ratio, 4),
                "multiple": round(today_ratio / avg_ratio, 2),
            }

    # Cache result
    result = {
        "date": date_str,
        "threshold": threshold,
        "unusual_symbols": unusual,
        "details": details,
    }
    try:
        cache_file.write_text(json.dumps(result, indent=2))
    except Exception:
        pass

    return unusual


# ═══════════════════════════════════════════════════════════════════════════════
#  4. LOAD FLOW HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

def load_options_flow_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load cached daily options flow files into a DataFrame.

    Args:
        start_date: 'YYYY-MM-DD' or 'YYYYMMDD'
        end_date: 'YYYY-MM-DD' or 'YYYYMMDD'

    Returns:
        DataFrame with date index and market_pc_ratio column.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    rows = []
    for f in sorted(CACHE_DIR.glob("options_flow_*.json")):
        try:
            date_part = f.stem.replace("options_flow_", "")
            file_date = pd.Timestamp(datetime.strptime(date_part, "%Y%m%d"))
            if file_date < start or file_date > end:
                continue

            data = json.loads(f.read_text())
            rows.append({
                "date": file_date,
                "market_pc_ratio": data.get("market_pc_ratio", np.nan),
                "n_symbols": data.get("n_symbols", 0),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["date", "market_pc_ratio", "n_symbols"])

    df = pd.DataFrame(rows)
    df = df.set_index("date").sort_index()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Options Flow Aggregator")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"),
                        help="Date in YYYYMMDD format")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to scan")
    parser.add_argument("--unusual", action="store_true",
                        help="Scan for unusual put activity")
    parser.add_argument("--history", nargs=2, metavar=("START", "END"),
                        help="Load flow history between dates")
    args = parser.parse_args()

    if args.symbols is None:
        try:
            import config
            symbols = list(config.WATCHLIST)[:50]
        except ImportError:
            symbols = ["NVDA", "AAPL", "META", "AMZN", "GOOGL",
                        "MSFT", "TSLA", "JPM", "XOM", "UNH"]

    else:
        symbols = args.symbols

    if args.history:
        df = load_options_flow_history(args.history[0], args.history[1])
        print(f"Options flow history: {len(df)} days")
        print(df.to_string())

    elif args.unusual:
        print(f"Scanning {len(symbols)} symbols for unusual put activity on {args.date}...")
        unusual = get_unusual_put_activity(symbols, args.date)
        if unusual:
            print(f"Unusual put activity: {unusual}")
        else:
            print("No unusual put activity detected.")

    else:
        print(f"Computing market P/C ratio for {len(symbols)} symbols on {args.date}...")
        ratio = get_market_put_call_ratio(symbols, args.date)
        print(f"Market put/call ratio: {ratio:.4f}")
