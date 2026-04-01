"""
analyst_recommendations.py — Analyst Recommendation Momentum
=============================================================
Uses yfinance monthly recommendation counts (strongBuy/buy/hold/sell/strongSell)
Computes consensus score and 3-month momentum.

LIVE ONLY — not for backtesting (no historical time series).
Snapshot reflects current analyst sentiment.

IC estimate: 0.02-0.04 on forward returns (Jegadeesh et al 2004)
Momentum of revisions > level of revisions
"""
import os, json, time
import numpy as np
from typing import Dict, Optional

CACHE_DIR  = "cache_revisions"
CACHE_DAYS = 7
os.makedirs(CACHE_DIR, exist_ok=True)


def _score_row(row) -> float:
    total = row['strongBuy'] + row['buy'] + row['hold'] + row['sell'] + row['strongSell']
    if total == 0:
        return 0.0
    return float(row['strongBuy']*2 + row['buy'] - row['sell'] - row['strongSell']*2) / total


def get_analyst_signals(symbol: str) -> Dict[str, float]:
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_analyst.json")
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age < CACHE_DAYS:
            try:
                return json.load(open(cache_path))
            except Exception:
                pass
    try:
        import yfinance as yf
        t   = yf.Ticker(symbol)
        rec = t.recommendations
        if rec is None or len(rec) < 2:
            return {}

        scores = rec.apply(_score_row, axis=1).values
        current  = float(scores[0])
        prior    = float(scores[-1])
        momentum = current - prior

        row0 = rec.iloc[0]
        total0 = float(row0['strongBuy'] + row0['buy'] + row0['hold'] + row0['sell'] + row0['strongSell'])
        consensus = float(row0['strongBuy'] + row0['buy']) / max(total0, 1)

        result = {
            'analyst_score':     current,
            'analyst_momentum':  momentum,
            'analyst_consensus': consensus,
        }
        json.dump(result, open(cache_path, 'w'))
        return result
    except Exception:
        return {}


def build_analyst_store(symbols: list) -> Dict[str, Dict]:
    store = {}
    for i, sym in enumerate(symbols):
        if i % 50 == 0:
            print(f"  [analyst] {i}/{len(symbols)}...", flush=True)
        r = get_analyst_signals(sym)
        if r:
            store[sym] = r
        time.sleep(0.05)
    return store


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    syms = list(config.WATCHLIST)[:20]
    print(f"Testing analyst signals on {len(syms)} symbols...")
    store = build_analyst_store(syms)
    for sym, v in list(store.items())[:10]:
        print(f"  {sym}: score={v['analyst_score']:.2f}  momentum={v['analyst_momentum']:.2f}  consensus={v['analyst_consensus']:.0%}")
