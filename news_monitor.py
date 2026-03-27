"""
news_monitor.py — Real-time news monitor for open positions
===========================================================
Runs continuously during market hours.
Monitors Polygon news API every 60 seconds for all open positions.
Alerts immediately on negative sentiment.

Usage:
  python news_monitor.py --positions NVDA,PLTR,APP
  python news_monitor.py --auto  (reads from Alpaca positions)

Alerts:
  - Negative headline detected → print alert + sound
  - 3+ negative headlines → suggest exit
"""
import os, sys, json, time, requests
from datetime import datetime, timedelta
import argparse

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")
CACHE_DIR   = "/Users/rick/ai_trading_bot_v2/cache_news"
os.makedirs(CACHE_DIR, exist_ok=True)

# Track headlines we've already seen
SEEN_HEADLINES = set()


def fetch_recent_news(symbol: str, minutes_back: int = 65) -> list:
    """Fetch news published in the last N minutes."""
    since = (datetime.utcnow() - timedelta(minutes=minutes_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        resp = requests.get(
            "https://api.polygon.io/v2/reference/news",
            params={
                "ticker":             symbol,
                "limit":              10,
                "apiKey":             POLYGON_KEY,
                "published_utc.gte":  since,
                "order":              "desc",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("results", [])
    except Exception:
        return []


def score_article(article: dict, symbol: str) -> dict:
    """Extract sentiment for a specific symbol from article insights."""
    insights = article.get("insights", [])
    sentiment = "neutral"
    for ins in insights:
        if ins.get("ticker") == symbol:
            sentiment = ins.get("sentiment", "neutral")
            break
    return {
        "title":     article.get("title", ""),
        "published": article.get("published_utc", "")[:19],
        "sentiment": sentiment,
        "url":       article.get("article_url", ""),
    }


def check_positions(symbols: list, alert_log: list) -> dict:
    """Check all positions for new negative news."""
    alerts = {}
    for sym in symbols:
        news = fetch_recent_news(sym)
        neg_articles = []
        for article in news:
            uid = article.get("id", article.get("title", ""))[:50]
            if uid in SEEN_HEADLINES:
                continue
            SEEN_HEADLINES.add(uid)
            scored = score_article(article, sym)
            if scored["sentiment"] == "negative":
                neg_articles.append(scored)
                alert_log.append({
                    "time":      datetime.now().strftime("%H:%M:%S"),
                    "symbol":    sym,
                    "headline":  scored["title"][:80],
                    "sentiment": "NEGATIVE",
                })
        if neg_articles:
            alerts[sym] = neg_articles
        time.sleep(0.15)  # rate limit
    return alerts


def print_alert(sym: str, articles: list):
    """Print a prominent alert."""
    print(f"\n{'!'*60}")
    print(f"  ⚠️  NEGATIVE NEWS: {sym}")
    print(f"{'!'*60}")
    for a in articles:
        print(f"  [{a['published']}] {a['title'][:75]}")
    print(f"  → Consider reducing or exiting {sym} position")
    # macOS notification sound
    os.system("afplay /System/Library/Sounds/Ping.aiff 2>/dev/null &")


def run_monitor(symbols: list, interval: int = 60):
    """Main monitoring loop."""
    print(f"\n{'='*60}")
    print(f"NEWS MONITOR — Real-time sentiment tracking")
    print(f"Monitoring: {', '.join(symbols)}")
    print(f"Refresh: every {interval}s")
    print(f"{'='*60}\n")

    alert_log = []
    scan_count = 0

    while True:
        now = datetime.now()

        # Only run during market hours (9:30am - 5:00pm ET)
        if now.hour < 9 or now.hour >= 17 or now.weekday() >= 5:
            print(f"[{now.strftime('%H:%M')}] Market closed — sleeping 5min")
            time.sleep(300)
            continue

        scan_count += 1
        print(f"[{now.strftime('%H:%M:%S')}] Scan #{scan_count} — checking {len(symbols)} positions...", end="\r")

        alerts = check_positions(symbols, alert_log)

        if alerts:
            for sym, articles in alerts.items():
                print_alert(sym, articles)
                # Count total negatives for this symbol today
                total_neg = sum(1 for a in alert_log if a["symbol"] == sym)
                if total_neg >= 3:
                    print(f"\n  🚨 {sym}: {total_neg} negative articles today — STRONG EXIT SIGNAL")

            # Save alert log
            with open(os.path.join(CACHE_DIR, "alert_log.json"), "w") as f:
                json.dump(alert_log, f, indent=2)
        else:
            # Show clean status
            neg_today = len([a for a in alert_log
                            if a["time"] >= "09:30"])
            print(f"[{now.strftime('%H:%M:%S')}] Scan #{scan_count} — "
                  f"{len(symbols)} positions clean | "
                  f"{neg_today} alerts today        ", end="\r")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=str, help="Comma-separated symbols")
    parser.add_argument("--auto",      action="store_true", help="Read from Alpaca")
    parser.add_argument("--interval",  type=int, default=60)
    args = parser.parse_args()

    if args.auto:
        # Will read from Alpaca when live — for now use test symbols
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(
                os.environ.get("ALPACA_KEY", ""),
                os.environ.get("ALPACA_SECRET", ""),
                base_url="https://paper-api.alpaca.markets"
            )
            positions = [p.symbol for p in api.list_positions()]
            print(f"Loaded {len(positions)} positions from Alpaca")
        except Exception as e:
            print(f"Alpaca not available: {e}")
            positions = ["NVDA", "PLTR", "APP", "SNOW", "MSFT"]
    elif args.positions:
        positions = [s.strip().upper() for s in args.positions.split(",")]
    else:
        # Default — monitor worst historical offenders
        positions = ["NVDA", "PLTR", "APP", "SNOW", "MSFT",
                     "SHOP", "INTU", "VRT", "FTNT", "AVAV"]

    run_monitor(positions, interval=args.interval)
