"""
earnings_streak.py — Earnings Beat Streak Signal
=================================================
Academic basis: He & Narayanamoorthy (2020), t-stat > 3.0
Consecutive quarters beating consensus EPS predicts forward returns.
Improves on PEAD by ~45% when combined with momentum.
Free from yfinance. Works on mid-caps where analyst coverage is thinner.
"""
import os, json, time
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

CACHE_DIR = "cache_earnings_streak"
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
            actual = row.get("epsActual")
            est    = row.get("epsEstimate")
            surp   = row.get("surprisePercent")
            if pd.isna(actual) or pd.isna(est):
                continue
            beat = bool(float(actual) > float(est)) if not pd.isna(est) else None
            records.append({
                "date":    pd.Timestamp(dt).isoformat(),
                "actual":  float(actual),
                "est":     float(est) if not pd.isna(est) else None,
                "surp":    float(surp) if not pd.isna(surp) else 0.0,
                "beat":    beat,
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


def beat_streak_score(
    symbol: str,
    current_date: pd.Timestamp,
    store: Dict[str, pd.DataFrame],
) -> float:
    """
    Returns 0.0-1.0 score based on consecutive earnings beats.
    0.0 = no data or recent miss
    0.3 = 1 consecutive beat
    0.6 = 2 consecutive beats
    0.85 = 3 consecutive beats
    1.0 = 4+ consecutive beats
    """
    df = store.get(symbol)
    if df is None or len(df) == 0:
        return 0.0

    past = df[df["date"] < current_date].sort_values("date", ascending=False)
    if len(past) == 0:
        return 0.0

    # Count consecutive beats from most recent
    streak = 0
    for _, row in past.iterrows():
        if row.get("beat") is True:
            streak += 1
        else:
            break

    if streak == 0:
        return 0.0
    elif streak == 1:
        return 0.30
    elif streak == 2:
        return 0.60
    elif streak == 3:
        return 0.85
    else:
        return 1.0


def build_streak_store(
    symbols: List[str],
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    store = {}
    for i, sym in enumerate(symbols):
        if verbose and (i+1) % 50 == 0:
            print(f"  [streak] {i+1}/{len(symbols)}", end="\r", flush=True)
        df = _fetch(sym)
        if df is not None and len(df) > 0:
            store[sym] = df
    if verbose:
        print(f"  [streak] loaded {len(store)}/{len(symbols)} symbols    ")
    return store


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
    import config
    test = ["AXON","WING","CAVA","NVDA","MSFT","INTU","AVGO","LLY","APP","DUOL"]
    print("Building earnings streak store...")
    store = build_streak_store(test, verbose=False)
    today = pd.Timestamp.now().normalize()
    print(f"\nEarnings beat streak scores as of {today.date()}:")
    for sym in test:
        score = beat_streak_score(sym, today, store)
        df = store.get(sym)
        if df is not None:
            past = df[df["date"] < today].sort_values("date", ascending=False)
            recent = past.head(4)
            beats = "".join(["✓" if r.get("beat") else "✗" for _, r in recent.iterrows()])
        else:
            beats = "no data"
        print(f"  {sym:<8} score={score:.2f}  recent quarters: {beats}")
