"""
analyst_revision.py — Analyst Estimate Revision Momentum
=========================================================
Academic basis: Mill Street Research 2023 — revisions persist 5 months.
Direction of revisions today predicts direction next quarter.
Free from yfinance earnings_history. Independent of price momentum.
"""
import os, json, time
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

CACHE_DIR = "cache_analyst"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_DAYS = 7

def _fetch(symbol: str) -> Optional[pd.DataFrame]:
    path = os.path.join(CACHE_DIR, f"{symbol}.json")
    if os.path.exists(path):
        if (time.time() - os.path.getmtime(path)) / 86400 < CACHE_DAYS:
            try:
                with open(path) as f:
                    records = json.load(f)
                if records:
                    df = pd.DataFrame(records)
                    df["date"] = pd.to_datetime(df["date"])
                    return df
            except Exception:
                pass
    try:
        import yfinance as yf
        raw = yf.Ticker(symbol).earnings_history
        if raw is None or len(raw) == 0:
            return None
        records = []
        for dt, row in raw.iterrows():
            est = row.get("epsEstimate")
            act = row.get("epsActual")
            if pd.isna(est) or pd.isna(act):
                continue
            records.append({
                "date": pd.Timestamp(dt).isoformat(),
                "estimate": float(est),
                "actual":   float(act),
                "surprise": float(act - est),
            })
        if not records:
            return None
        with open(path, "w") as f:
            json.dump(records, f)
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")
    except Exception:
        return None


def revision_momentum_score(
    symbol: str,
    current_date: pd.Timestamp,
    store: Dict[str, pd.DataFrame],
) -> float:
    """
    Returns -1.0 to +1.0 revision momentum score.
    Positive = analysts consistently beating estimates (upward revision pressure)
    Negative = analysts consistently missing (downward revision pressure)
    0 = neutral / no data
    """
    df = store.get(symbol)
    if df is None or len(df) == 0:
        return 0.0
    past = df[df["date"] < current_date].sort_values("date")
    if len(past) < 2:
        return 0.0
    # Last 4 quarters of surprise direction
    recent = past.tail(4)
    beats = (recent["surprise"] > 0).sum()
    total = len(recent)
    # Normalize: 4/4 beats = +1.0, 0/4 = -1.0
    score = (beats / total) * 2 - 1.0
    # Weight by recency — most recent quarter counts double
    last = recent.iloc[-1]
    recency_boost = 0.15 if last["surprise"] > 0 else -0.15
    return float(np.clip(score + recency_boost, -1.0, 1.0))


def build_revision_store(
    symbols: List[str],
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    store = {}
    for i, sym in enumerate(symbols):
        if verbose and (i+1) % 50 == 0:
            print(f"  [revision] {i+1}/{len(symbols)}", end="\r", flush=True)
        df = _fetch(sym)
        if df is not None and len(df) > 0:
            store[sym] = df
    if verbose:
        print(f"  [revision] loaded {len(store)}/{len(symbols)}    ")
    return store


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
    test = ["AXON","WING","NVDA","MSFT","INTU","AVGO","LLY","CAVA","APP","DUOL"]
    print("Building revision store...")
    store = build_revision_store(test, verbose=False)
    today = pd.Timestamp.now().normalize()
    print(f"\nRevision momentum scores as of {today.date()}:")
    for sym in test:
        score = revision_momentum_score(sym, today, store)
        df = store.get(sym)
        if df is not None:
            past = df[df["date"] < today].tail(4)
            surps = "".join(["↑" if r["surprise"]>0 else "↓" for _,r in past.iterrows()])
        else:
            surps = "no data"
        print(f"  {sym:<8} score={score:+.2f}  last 4 qtrs: {surps}")
