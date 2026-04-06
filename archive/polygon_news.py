"""
polygon_news.py — Real-time news sentiment filter
==================================================
Uses Polygon.io news API with AI-generated sentiment insights.
Scans watchlist daily before market open.

Live trading use:
  - Score < -0.3 = negative sentiment = skip entry today
  - Score > +0.3 = positive = proceed normally
  - Cached for 6 hours to avoid rate limits

Not used in backtesting — historical news not available on Starter plan.
"""
import os, json, time, requests
from datetime import datetime, timedelta
from typing import Dict, List

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")
CACHE_DIR   = "/Users/rick/ai_trading_bot_v2/cache_news"
CACHE_FILE  = os.path.join(CACHE_DIR, "sentiment.json")
CACHE_TTL   = 6 * 3600  # 6 hours

os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_sentiment(symbol: str, days_back: int = 2) -> Dict:
    """Fetch news sentiment for a symbol from Polygon."""
    since = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            "https://api.polygon.io/v2/reference/news",
            params={
                "ticker": symbol,
                "limit": 20,
                "apiKey": POLYGON_KEY,
                "published_utc.gte": since,
                "order": "desc",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return {"score": 0.0, "total": 0, "error": resp.status_code}

        news = resp.json().get("results", [])
        pos = neg = neu = 0
        headlines = []

        for article in news:
            insights = article.get("insights", [])
            for ins in insights:
                if ins.get("ticker") == symbol:
                    s = ins.get("sentiment", "neutral")
                    if s == "positive":
                        pos += 1
                    elif s == "negative":
                        neg += 1
                    else:
                        neu += 1
            headlines.append(article.get("title", "")[:80])

        total = pos + neg + neu
        score = (pos - neg) / total if total > 0 else 0.0

        return {
            "symbol":    symbol,
            "score":     round(score, 3),
            "positive":  pos,
            "negative":  neg,
            "neutral":   neu,
            "total":     total,
            "signal":    "BULLISH" if score > 0.3 else "BEARISH" if score < -0.3 else "NEUTRAL",
            "headlines": headlines[:3],
            "updated":   datetime.now().isoformat(),
        }
    except Exception as e:
        return {"symbol": symbol, "score": 0.0, "total": 0, "error": str(e)}


def scan_watchlist(symbols: List[str], force: bool = False) -> Dict[str, Dict]:
    """Scan all symbols, using cache if fresh enough."""
    # Load cache
    cache = {}
    if os.path.exists(CACHE_FILE) and not force:
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            # Check if cache is fresh
            if cache:
                first = next(iter(cache.values()))
                updated = datetime.fromisoformat(first.get("updated", "2000-01-01"))
                if (datetime.now() - updated).seconds < CACHE_TTL:
                    return cache
        except Exception:
            cache = {}

    print(f"Scanning news sentiment for {len(symbols)} symbols...")
    results = {}
    for i, sym in enumerate(symbols):
        result = fetch_sentiment(sym)
        results[sym] = result
        if result["total"] > 0:
            print(f"  {sym:<6}: score={result['score']:+.2f}  "
                  f"pos={result['positive']}  neg={result['negative']}  "
                  f"signal={result['signal']}")
        time.sleep(0.15)  # rate limit — Starter plan

        if (i + 1) % 50 == 0:
            # Save intermediate results
            with open(CACHE_FILE, "w") as f:
                json.dump(results, f)

    with open(CACHE_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved sentiment for {len(results)} symbols")
    return results


def get_sentiment(symbol: str) -> float:
    """Get cached sentiment score for a symbol. Returns 0.0 if not available."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            return float(cache.get(symbol, {}).get("score", 0.0))
    except Exception:
        pass
    return 0.0


def is_entry_ok(symbol: str, threshold: float = -0.3) -> bool:
    """Return True if sentiment is not bearish — safe to enter."""
    score = get_sentiment(symbol)
    return score >= threshold


def get_bearish_symbols() -> List[str]:
    """Return list of symbols with negative news sentiment — avoid today."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            return [sym for sym, data in cache.items()
                    if data.get("score", 0) < -0.3]
    except Exception:
        pass
    return []


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/rick/ai_trading_bot_v2")
    import config

    results = scan_watchlist(config.WATCHLIST, force=True)

    bearish = [s for s, d in results.items() if d.get("score", 0) < -0.3]
    bullish = [s for s, d in results.items() if d.get("score", 0) > 0.3]

    print(f"\nBEARISH (skip entry today): {bearish}")
    print(f"BULLISH (green light):      {bullish[:10]}")

    # Show any with negative news
    print(f"\nStocks with negative headlines:")
    for sym, d in results.items():
        if d.get("negative", 0) > 0:
            print(f"  {sym}: score={d['score']:+.2f}  neg={d['negative']}")
            for h in d.get("headlines", [])[:2]:
                print(f"    - {h}")
