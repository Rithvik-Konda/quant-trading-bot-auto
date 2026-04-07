"""
signal_propagation.py — Causal Feature Extraction from Entity Graph + Event Stream
====================================================================================
Combines SEC event sentiment with corporate relationship graph to compute
propagation scores: how events at suppliers/customers/competitors affect a stock.

Usage:
    python v2/signal_propagation.py --test --tickers NVDA AAPL META
"""
from __future__ import annotations

import os
import sys
import math
import pickle
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))
sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2/v2"))

import pandas as pd
import numpy as np

try:
    from entity_graph import _load_graph, get_propagation_score
except ImportError:
    _load_graph = None
    get_propagation_score = None

try:
    from event_stream import load_8k_events, get_composite_sentiment
except ImportError:
    load_8k_events = None
    get_composite_sentiment = None

CACHE_DIR = Path(os.path.expanduser("~/ai_trading_bot_v2/cache_events"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. BUILD EVENT SCORES
# ═══════════════════════════════════════════════════════════════════════════════

def build_event_scores(
    event_stream_df: pd.DataFrame,
    date: str,
    lookback_days: int = 10,
) -> Dict[str, float]:
    """
    Compute weighted sentiment score per entity from the event stream.

    Args:
        event_stream_df: DataFrame with columns date, ticker, sentiment, magnitude
        date: evaluation date ('YYYY-MM-DD')
        lookback_days: number of days to look back

    Returns:
        Dict of {ticker: weighted_sentiment}
    """
    if event_stream_df is None or len(event_stream_df) == 0:
        return {}

    qd = pd.Timestamp(date)
    window_start = qd - pd.Timedelta(days=lookback_days)

    mask = (
        (event_stream_df["date"] >= window_start) &
        (event_stream_df["date"] <= qd)
    )
    window = event_stream_df[mask]

    if len(window) == 0:
        return {}

    scores: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    for _, row in window.iterrows():
        ticker = str(row.get("ticker", ""))
        if not ticker:
            continue

        days_ago = max(0, (qd - pd.Timestamp(row["date"])).days)
        decay = math.exp(-days_ago / 5.0)
        magnitude = float(row.get("magnitude", 0.5))
        sentiment = float(row.get("sentiment", 0.0))
        w = magnitude * decay

        scores[ticker] = scores.get(ticker, 0.0) + sentiment * w
        weights[ticker] = weights.get(ticker, 0.0) + w

    # Normalize to weighted average
    result = {}
    for ticker in scores:
        if weights[ticker] > 0:
            result[ticker] = max(-1.0, min(1.0, scores[ticker] / weights[ticker]))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  2. PROPAGATION SCORE FOR SINGLE TICKER
# ═══════════════════════════════════════════════════════════════════════════════

def get_propagation_score_for_ticker(
    ticker: str,
    date: str,
    event_stream_df: pd.DataFrame,
    graph=None,
    lookback_days: int = 10,
) -> float:
    """
    Compute propagation score: how events at related companies affect this ticker.

    Returns float -1.0 to 1.0.
    """
    if get_propagation_score is None:
        return 0.0

    if graph is None:
        if _load_graph is not None:
            graph = _load_graph()
        if graph is None:
            return 0.0

    event_scores = build_event_scores(event_stream_df, date, lookback_days)
    if not event_scores:
        return 0.0

    return get_propagation_score(ticker, graph, event_scores)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. PROPAGATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def compute_propagation_matrix(
    tickers: List[str],
    dates: List[str],
    event_stream_df: pd.DataFrame,
    graph=None,
) -> pd.DataFrame:
    """
    Compute propagation score for every (ticker, date) pair.

    Returns DataFrame with index=dates, columns=tickers.
    Cached to cache_events/propagation_cache.pkl (1-day TTL).
    """
    cache_path = CACHE_DIR / "propagation_cache.pkl"
    if cache_path.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
        if age_hours < 24:
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

    if graph is None:
        if _load_graph is not None:
            graph = _load_graph()

    matrix = pd.DataFrame(0.0, index=dates, columns=tickers)

    for i, date in enumerate(dates):
        if (i + 1) % 100 == 0:
            print(f"  Propagation: {i + 1}/{len(dates)} dates processed...")

        event_scores = build_event_scores(event_stream_df, date)

        if not event_scores or graph is None:
            continue

        for ticker in tickers:
            try:
                score = get_propagation_score(ticker, graph, event_scores)
                matrix.loc[date, ticker] = score
            except Exception:
                continue

    # Cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(matrix, f)
    except Exception:
        pass

    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
#  4. CAUSAL FEATURES FOR ML
# ═══════════════════════════════════════════════════════════════════════════════

def get_causal_features(
    ticker: str,
    date: str,
    event_stream_df: pd.DataFrame,
    graph=None,
    lookback_days: int = 10,
) -> Dict[str, float]:
    """
    Returns a dict of causal features for ML consumption.

    Features:
        causal_prop_score:        propagation score from graph neighbors
        causal_own_sentiment:     ticker's own composite sentiment
        causal_insider_magnitude: max insider transaction magnitude in window
        causal_insider_direction: +1 buy, -1 sell, 0 none
        causal_event_count:       number of events in window
        causal_has_earnings:      1 if earnings 8-K in window
        causal_has_guidance:      1 if guidance 8-K in window
    """
    features = {
        "causal_prop_score": 0.0,
        "causal_own_sentiment": 0.0,
        "causal_insider_magnitude": 0.0,
        "causal_insider_direction": 0.0,
        "causal_event_count": 0,
        "causal_has_earnings": 0,
        "causal_has_guidance": 0,
    }

    if event_stream_df is None or len(event_stream_df) == 0:
        return features

    # Propagation score
    features["causal_prop_score"] = get_propagation_score_for_ticker(
        ticker, date, event_stream_df, graph, lookback_days
    )

    # Own sentiment
    if get_composite_sentiment is not None:
        features["causal_own_sentiment"] = get_composite_sentiment(
            ticker, date, lookback_days, event_stream_df
        )

    # Filter events for this ticker in the lookback window
    qd = pd.Timestamp(date)
    window_start = qd - pd.Timedelta(days=lookback_days)

    mask = (
        (event_stream_df["ticker"].str.upper() == ticker.upper()) &
        (event_stream_df["date"] >= window_start) &
        (event_stream_df["date"] <= qd)
    )
    ticker_events = event_stream_df[mask].sort_values("date", ascending=False)

    features["causal_event_count"] = len(ticker_events)

    # Insider features
    insider_events = ticker_events[
        ticker_events["event_type"].isin(["insider_buy", "insider_sell"])
    ]
    if len(insider_events) > 0:
        features["causal_insider_magnitude"] = float(insider_events["magnitude"].max())
        most_recent = insider_events.iloc[0]
        if most_recent["event_type"] == "insider_buy":
            features["causal_insider_direction"] = 1.0
        elif most_recent["event_type"] == "insider_sell":
            features["causal_insider_direction"] = -1.0

    # Earnings and guidance flags
    if len(ticker_events) > 0:
        event_types = set(ticker_events["event_type"].tolist())
        features["causal_has_earnings"] = 1 if "8k_earnings" in event_types else 0
        features["causal_has_guidance"] = 1 if "8k_guidance" in event_types else 0

    return features


# ═══════════════════════════════════════════════════════════════════════════════
#  5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Signal Propagation")
    parser.add_argument("--test", action="store_true", help="Test causal features")
    parser.add_argument("--tickers", nargs="+", default=["NVDA", "AAPL", "META"])
    args = parser.parse_args()

    if args.test:
        print("Loading entity graph...")
        graph = _load_graph() if _load_graph else None
        if graph is None:
            print("  No graph found (run: python v2/entity_graph.py --build)")
        else:
            print(f"  Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        print("Loading 8-K events (last 30 days)...")
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

        events_list = load_8k_events(start_date, end_date) if load_8k_events else []
        if events_list:
            event_df = pd.DataFrame(events_list)
            event_df["date"] = pd.to_datetime(event_df["date"], errors="coerce")
            print(f"  Events: {len(event_df)}")
        else:
            event_df = pd.DataFrame(columns=["date", "ticker", "event_type",
                                             "sentiment", "magnitude", "raw_text", "source"])
            print("  No 8-K events found (run: python v2/sec_8k_pipeline.py --backfill)")

        print(f"\nCausal features for {args.tickers}:")
        for ticker in args.tickers:
            features = get_causal_features(ticker, end_date, event_df, graph)
            print(f"\n  {ticker}:")
            for k, v in features.items():
                if isinstance(v, float):
                    print(f"    {k:30s} {v:+.4f}")
                else:
                    print(f"    {k:30s} {v}")
