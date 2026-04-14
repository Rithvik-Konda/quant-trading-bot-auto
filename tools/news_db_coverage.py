"""
news_db_coverage.py — How much news history do we actually have?

Queries cache_news/news_events.db (from news_intelligence.py) to answer:
  1. Date range of news events (earliest / latest)
  2. Events per month
  3. Breakdown by source
  4. Breakdown by event_type
  5. Ticker coverage — how many unique tickers have at least N events?
  6. Overlap with OOS window (2022-2026)
  7. Sample of most recent events

This tells us whether wiring news features into the ranker is a
"add a column and retrain" project or a "backfill 5 years of history first"
project. The answer has very different costs.

Read-only. Safe alongside noise_floor.

Usage:
  python3.11 tools/news_db_coverage.py
"""

from __future__ import annotations
import os
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "cache_news" / "news_events.db"


def main():
    if not DB_PATH.exists():
        print(f"FATAL: {DB_PATH} not found.")
        sys.exit(1)

    print("═" * 72)
    print(f"  NEWS DB COVERAGE  —  {DB_PATH}")
    print("═" * 72)
    print(f"  file size: {DB_PATH.stat().st_size / 1024:.0f} KB")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # 1. Schema
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"\n  Tables: {tables}")

    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = cur.fetchall()
        print(f"\n  {t}:")
        for col in cols:
            print(f"    {col[1]:<20} {col[2]}")

    # Assume 'news_events' is the main one
    main_table = "news_events" if "news_events" in tables else (tables[0] if tables else None)
    if main_table is None:
        print("\n  No tables found. Bailing.")
        return

    # 2. Total row count
    cur.execute(f"SELECT COUNT(*) FROM {main_table}")
    total = cur.fetchone()[0]
    print(f"\n  Total rows: {total:,}")

    if total == 0:
        print("  DB is empty.")
        return

    # Find a timestamp column
    cur.execute(f"PRAGMA table_info({main_table})")
    col_names = [r[1] for r in cur.fetchall()]
    ts_col = None
    for candidate in ("timestamp", "time", "date", "created_at"):
        if candidate in col_names:
            ts_col = candidate
            break

    if ts_col:
        cur.execute(f"SELECT MIN({ts_col}), MAX({ts_col}) FROM {main_table}")
        mn, mx = cur.fetchone()
        print(f"\n  Date range: {mn}  →  {mx}")

        # Events per month
        cur.execute(f"""
            SELECT substr({ts_col}, 1, 7) AS month, COUNT(*)
            FROM {main_table}
            GROUP BY month
            ORDER BY month
        """)
        rows = cur.fetchall()
        if rows:
            print(f"\n  Events per month ({len(rows)} months):")
            for month, n in rows[-24:]:  # last 24 months
                bar = "█" * min(int(n / 50), 40)
                print(f"    {month}  {n:>6,}  {bar}")

    # 3. By source
    if "source" in col_names:
        cur.execute(f"SELECT source, COUNT(*) FROM {main_table} GROUP BY source ORDER BY 2 DESC")
        print(f"\n  By source:")
        for src, n in cur.fetchall():
            print(f"    {str(src):<24} {n:>6,}")

    # 4. By event_type
    for et_col in ("event_type", "event", "category"):
        if et_col in col_names:
            cur.execute(f"SELECT {et_col}, COUNT(*) FROM {main_table} GROUP BY {et_col} ORDER BY 2 DESC")
            print(f"\n  By {et_col}:")
            for et, n in cur.fetchall():
                print(f"    {str(et):<24} {n:>6,}")
            break

    # 5. Ticker coverage (scan affected_json)
    if "affected_json" in col_names:
        cur.execute(f"SELECT affected_json FROM {main_table} WHERE affected_json IS NOT NULL AND affected_json != ''")
        import json
        from collections import Counter
        ticker_counts: Counter = Counter()
        for (aj,) in cur.fetchall():
            try:
                d = json.loads(aj) if aj else {}
                if isinstance(d, dict):
                    for t in d.keys():
                        ticker_counts[str(t).upper()] += 1
            except Exception:
                continue
        print(f"\n  Ticker coverage:")
        print(f"    unique tickers touched:   {len(ticker_counts):,}")
        print(f"    tickers with ≥1 event:    {(sum(1 for v in ticker_counts.values() if v >= 1)):,}")
        print(f"    tickers with ≥5 events:   {(sum(1 for v in ticker_counts.values() if v >= 5)):,}")
        print(f"    tickers with ≥20 events:  {(sum(1 for v in ticker_counts.values() if v >= 20)):,}")
        top = ticker_counts.most_common(15)
        if top:
            print(f"    top 15 by event count:")
            for t, n in top:
                print(f"      {t:<8} {n}")

    # 6. OOS window overlap
    if ts_col:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {main_table} WHERE {ts_col} >= '2022-01-01'")
            n_oos = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {main_table} WHERE {ts_col} >= '2025-01-01'")
            n_2025 = cur.fetchone()[0]
            print(f"\n  OOS overlap:")
            print(f"    events in 2022+: {n_oos:,}")
            print(f"    events in 2025+: {n_2025:,}")
            if n_oos == 0:
                print(f"    ⚠ DB has NO coverage of the OOS backtest window.")
                print(f"    → News features cannot be used retroactively without backfill.")
            elif n_oos < 1000:
                print(f"    ⚠ Sparse OOS coverage ({n_oos} events for 4 years).")
                print(f"    → News features usable but will have lots of zeros.")
        except Exception as e:
            print(f"  (date filter failed: {e})")

    # 7. Most recent 10 events
    if ts_col:
        cur.execute(f"SELECT * FROM {main_table} ORDER BY {ts_col} DESC LIMIT 5")
        rows = cur.fetchall()
        if rows:
            print(f"\n  5 most recent rows (truncated):")
            for r in rows:
                s = " | ".join(str(c)[:40] for c in r[:6])
                print(f"    {s}")

    conn.close()

    print("\n" + "═" * 72)
    print("  INTERPRETATION")
    print("═" * 72)
    print("""
  If OOS overlap is thick (10k+ events in 2022+):
    → Wire features into ranker, retrain on a branch, measure vs σ.
      This is a 1-2 day project.

  If OOS overlap is thin or zero:
    → Backfill from SEC EDGAR (8-K only has full history, free).
      Reuters/AP RSS feeds cannot be backfilled — that data is lost.
      This is a 1-2 week project BUT 8-K sentiment alone is a real signal.

  Either way: do not wire news into ranker tonight. Retraining needs to
  happen on a branch, measured against the noise floor, compared to
  BASELINE_2026_04. Do it right, not fast.
""")


if __name__ == "__main__":
    main()
