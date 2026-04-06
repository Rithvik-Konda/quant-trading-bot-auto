"""
iv_term_structure.py — Individual Stock IV Term Structure Signal
================================================================
Academic basis: Pan & Poteshman (2006), Cremers & Weinbaum (2010)
When near-term IV > far-term IV (backwardation) for individual stock:
  = temporary fear = mean reversion opportunity
When near-term IV << far-term IV (steep contango):
  = market expects future event = momentum signal (earnings, catalyst)

IC estimate: 0.03-0.05 on 20-day forward returns
Computed daily, cached with 1-day TTL
"""
import os, json, time
import numpy as np
import pandas as pd
from typing import Dict, Optional

CACHE_DIR  = "cache_options_iv"
CACHE_DAYS = 1  # refresh daily
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_iv_term_structure(symbol: str) -> Dict[str, float]:
    """
    Compute IV term structure for a single symbol.
    Returns near_iv, far_iv, term_structure_ratio, iv_backwardation
    """
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        price_hist = t.history(period='2d')
        if len(price_hist) == 0:
            return {}
        price = float(price_hist['Close'].iloc[-1])

        dates = t.options
        if not dates or len(dates) < 3:
            return {}

        iv_by_expiry = []
        for exp in dates[:8]:
            try:
                chain = t.option_chain(exp)
                calls = chain.calls.copy()
                calls = calls[calls['impliedVolatility'] > 0.01]
                if len(calls) == 0:
                    continue
                calls['dist'] = abs(calls['strike'] - price)
                atm = calls.nsmallest(3, 'dist')
                avg_iv = float(atm['impliedVolatility'].mean())
                if avg_iv > 0.01:
                    iv_by_expiry.append(avg_iv)
            except Exception:
                continue

        if len(iv_by_expiry) < 2:
            return {}

        near_iv = iv_by_expiry[0]
        far_iv  = iv_by_expiry[-1]
        ratio   = near_iv / max(far_iv, 0.001)

        return {
            'near_iv':            near_iv,
            'far_iv':             far_iv,
            'iv_term_structure':  ratio,          # > 1 = backwardation (fear)
            'iv_backwardation':   float(ratio > 1.15),  # binary signal
            'iv_contango_steep':  float(ratio < 0.70),  # steep contango = catalyst expected
        }
    except Exception:
        return {}


def build_iv_store(symbols: list) -> Dict[str, Dict]:
    """Build IV term structure for all symbols. Cache with 1-day TTL."""
    cache_path = os.path.join(CACHE_DIR, "iv_term_structure.json")

    # Check cache
    if os.path.exists(cache_path):
        age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age_days < CACHE_DAYS:
            try:
                return json.load(open(cache_path))
            except Exception:
                pass

    store = {}
    total = len(symbols)
    for i, sym in enumerate(symbols):
        if i % 50 == 0:
            print(f"  [iv] {i}/{total} symbols...", flush=True)
        result = _get_iv_term_structure(sym)
        if result:
            store[sym] = result
        time.sleep(0.1)  # rate limit

    try:
        json.dump(store, open(cache_path, 'w'))
        print(f"  [iv] Saved {len(store)} symbols to cache")
    except Exception:
        pass

    return store


def get_iv_signals(symbol: str, store: Optional[Dict] = None) -> Dict[str, float]:
    """Get IV term structure signals for a single symbol."""
    if store and symbol in store:
        return store[symbol]
    return _get_iv_term_structure(symbol)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    symbols = list(config.WATCHLIST)[:20]  # test on 20
    print(f"Testing IV term structure on {len(symbols)} symbols...")
    store = build_iv_store(symbols)
    print(f"\nResults ({len(store)} symbols with data):")
    for sym, v in list(store.items())[:10]:
        print(f"  {sym}: near={v['near_iv']:.1%} far={v['far_iv']:.1%} "
              f"ratio={v['iv_term_structure']:.2f} "
              f"backwardation={bool(v['iv_backwardation'])}")
