"""
event_stream.py — Unified Event Feed: 8-K Filings + Form 4 Insider Transactions
=================================================================================
Normalizes SEC events into a single stream for signal generation.

Usage:
    python v2/event_stream.py --fetch --tickers NVDA AAPL TSLA
    python v2/event_stream.py --show NVDA
    python v2/event_stream.py --stream --tickers NVDA AAPL --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations

import os
import sys
import re
import json
import time
import math
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))
sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2/v2"))

import pandas as pd
import numpy as np

CACHE_DIR = Path(os.path.expanduser("~/ai_trading_bot_v2/cache_events"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEC_HEADERS = {"User-Agent": "research@tradingbot.com"}
SEC_SLEEP = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 — 8-K LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_8k_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Dict]:
    """
    Load 8-K events from cached history CSV and normalize to event dicts.
    """
    path = os.path.expanduser("~/ai_trading_bot_v2/cache_8k/8k_history.csv")
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    if len(df) == 0:
        return []

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    events = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", ""))
        date_str = row["date"].strftime("%Y-%m-%d")
        items = str(row.get("item_types", ""))

        # Map item_types to event_type
        if "2.02" in items:
            event_type = "8k_earnings"
        elif "7.01" in items or "8.01" in items:
            event_type = "8k_guidance"
        elif items and items != "nan":
            event_type = "8k_material"
        else:
            event_type = "8k_other"

        # Sentiment: prefer finbert if meaningful
        finbert = float(row.get("finbert_score", 0) or 0)
        rule = float(row.get("sentiment_score", 0) or 0)
        sentiment = finbert if abs(finbert) > 0.1 else rule

        # Magnitude
        novelty = float(row.get("novelty_score", 0) or 0)
        magnitude = min(1.0, abs(sentiment) + novelty * 0.3)

        events.append({
            "date": date_str,
            "ticker": ticker,
            "event_type": event_type,
            "sentiment": round(sentiment, 4),
            "magnitude": round(magnitude, 4),
            "raw_text": f"{ticker} {event_type} {date_str}",
            "source": "edgar_8k",
        })

    return events


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 — FORM 4 INSIDER TRANSACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_form4_xml(xml_text: str) -> List[Dict]:
    """
    Parse Form 4 XML to extract insider transactions.
    Returns list of transaction dicts.
    """
    transactions = []

    try:
        # Handle namespace issues — strip namespaces for simpler parsing
        clean = re.sub(r'\sxmlns[^"]*"[^"]*"', '', xml_text)
        root = ET.fromstring(clean)
    except ET.ParseError:
        return []

    # Extract owner info
    owner_name = ""
    is_director = False
    is_officer = False

    owner_id = root.find(".//reportingOwnerId")
    if owner_id is not None:
        name_el = owner_id.find("rptOwnerName")
        if name_el is not None and name_el.text:
            owner_name = name_el.text.strip()

    owner_rel = root.find(".//reportingOwnerRelationship")
    if owner_rel is not None:
        dir_el = owner_rel.find("isDirector")
        if dir_el is not None and dir_el.text:
            is_director = dir_el.text.strip() in ("1", "true", "True")
        off_el = owner_rel.find("isOfficer")
        if off_el is not None and off_el.text:
            is_officer = off_el.text.strip() in ("1", "true", "True")

    # Extract non-derivative transactions
    for txn in root.findall(".//nonDerivativeTransaction"):
        try:
            # Transaction date
            date_el = txn.find(".//transactionDate/value")
            txn_date = date_el.text.strip() if date_el is not None and date_el.text else ""

            # Transaction code (P=purchase, S=sale, F=tax withholding, etc.)
            code_el = txn.find(".//transactionCoding/transactionCode")
            txn_code = code_el.text.strip() if code_el is not None and code_el.text else ""

            # Skip tax withholding
            if txn_code == "F":
                continue

            # Shares
            shares_el = txn.find(".//transactionAmounts/transactionShares/value")
            shares = float(shares_el.text.strip()) if shares_el is not None and shares_el.text else 0

            # Price
            price_el = txn.find(".//transactionAmounts/transactionPricePerShare/value")
            price = float(price_el.text.strip()) if price_el is not None and price_el.text else 0

            # Acquired or Disposed
            ad_el = txn.find(".//transactionAmounts/transactionAcquiredDisposedCode/value")
            ad_code = ad_el.text.strip() if ad_el is not None and ad_el.text else ""

            transactions.append({
                "rptOwnerName": owner_name,
                "isDirector": is_director,
                "isOfficer": is_officer,
                "transactionDate": txn_date,
                "transactionCode": txn_code,
                "transactionShares": shares,
                "transactionPricePerShare": price,
                "transactionAcquiredDisposedCode": ad_code,
            })
        except Exception:
            continue

    return transactions


def fetch_insider_events(
    ticker: str,
    cik: str,
    start_date: str,
    end_date: str,
) -> List[Dict]:
    """
    Fetch Form 4 insider transactions from EDGAR for a single ticker.
    Caches to cache_events/{ticker}_form4.json (7-day TTL).
    """
    cache_file = CACHE_DIR / f"{ticker}_form4.json"
    if cache_file.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days < 7:
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass

    cik_padded = cik.zfill(10)
    cik_clean = cik.lstrip("0") or "0"

    # Get filing list
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        time.sleep(SEC_SLEEP)

        if resp.status_code != 200:
            return []

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
    except Exception:
        return []

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    events = []

    for i, form in enumerate(forms):
        if form != "4":
            continue

        filing_date = filing_dates[i] if i < len(filing_dates) else ""
        if not filing_date:
            continue

        fd = pd.Timestamp(filing_date)
        if fd < start_ts or fd > end_ts:
            continue

        acc_dashed = accessions[i]
        acc_nodash = acc_dashed.replace("-", "")

        # Fetch index page to find the XML file
        try:
            index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_nodash}/{acc_dashed}-index.htm"
            idx_resp = requests.get(index_url, headers=SEC_HEADERS, timeout=10)
            time.sleep(SEC_SLEEP)

            if idx_resp.status_code != 200:
                continue

            # Find XML file link — exclude xslF345X rendered version
            xml_links = re.findall(r'href="([^"]+\.xml)"', idx_resp.text, re.IGNORECASE)
            xml_url = None
            for xl in xml_links:
                if "xslF345X" not in xl and "xsl" not in xl.lower():
                    filename = xl.split("/")[-1]
                    if xl.startswith("http"):
                        xml_url = xl
                    elif xl.startswith("/"):
                        xml_url = f"https://www.sec.gov{xl}"
                    else:
                        xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_nodash}/{filename}"
                    break

            if xml_url is None:
                continue

            xml_resp = requests.get(xml_url, headers=SEC_HEADERS, timeout=10)
            time.sleep(SEC_SLEEP)

            if xml_resp.status_code != 200:
                continue

            txns = _parse_form4_xml(xml_resp.text)

            for txn in txns:
                ad_code = txn["transactionAcquiredDisposedCode"]
                is_insider = txn["isDirector"] or txn["isOfficer"]
                shares = txn["transactionShares"]
                price = txn["transactionPricePerShare"]
                dollar_value = shares * price

                if ad_code == "A":
                    event_type = "insider_buy"
                    sentiment = 0.7 if is_insider else 0.3
                elif ad_code == "D":
                    event_type = "insider_sell"
                    sentiment = -0.5 if is_insider else -0.2
                else:
                    continue

                magnitude = min(1.0, dollar_value / 5_000_000)

                events.append({
                    "date": txn["transactionDate"] or filing_date,
                    "ticker": ticker,
                    "event_type": event_type,
                    "sentiment": round(sentiment, 4),
                    "magnitude": round(magnitude, 4),
                    "raw_text": f"{txn['rptOwnerName']} {event_type} {int(shares)} shares at ${price:.2f}",
                    "source": "edgar_form4",
                })

        except Exception:
            continue

    # Cache results
    try:
        cache_file.write_text(json.dumps(events, indent=2, default=str))
    except Exception:
        pass

    return events


def load_insider_events(
    tickers: List[str],
    cik_map: Dict[str, str],
    start_date: str,
    end_date: str,
) -> List[Dict]:
    """
    Fetch insider events for multiple tickers.
    cik_map: {ticker: cik_string}
    """
    all_events = []
    for ticker in tickers:
        cik = cik_map.get(ticker)
        if cik is None:
            continue
        events = fetch_insider_events(ticker, cik, start_date, end_date)
        all_events.extend(events)

    all_events.sort(key=lambda e: e.get("date", ""))
    return all_events


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 — COMBINED STREAM
# ═══════════════════════════════════════════════════════════════════════════════

def get_event_stream(
    tickers: List[str],
    start_date: str,
    end_date: str,
    include_insider: bool = True,
) -> pd.DataFrame:
    """
    Build combined event stream from 8-K filings and Form 4 insider transactions.

    Returns DataFrame with columns:
        date, ticker, event_type, sentiment, magnitude, raw_text, source
    """
    # 8-K events
    all_8k = load_8k_events(start_date, end_date)
    ticker_set = set(t.upper() for t in tickers)
    filtered_8k = [e for e in all_8k if e["ticker"].upper() in ticker_set]

    # Insider events
    insider_events = []
    if include_insider:
        try:
            from sec_8k_pipeline import get_cik
            cik_map = {}
            for t in tickers:
                cik = get_cik(t)
                if cik:
                    cik_map[t] = cik
            insider_events = load_insider_events(tickers, cik_map, start_date, end_date)
        except ImportError:
            pass

    combined = filtered_8k + insider_events
    if not combined:
        return pd.DataFrame(columns=["date", "ticker", "event_type", "sentiment",
                                     "magnitude", "raw_text", "source"])

    df = pd.DataFrame(combined)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_ticker_events(
    ticker: str,
    date: str,
    lookback_days: int,
    event_stream_df: pd.DataFrame,
) -> List[Dict]:
    """
    Returns all events for ticker within lookback_days before date.
    Sorted by date descending (most recent first).
    """
    if event_stream_df is None or len(event_stream_df) == 0:
        return []

    qd = pd.Timestamp(date)
    window_start = qd - pd.Timedelta(days=lookback_days)

    mask = (
        (event_stream_df["ticker"].str.upper() == ticker.upper()) &
        (event_stream_df["date"] >= window_start) &
        (event_stream_df["date"] <= qd)
    )
    result = event_stream_df[mask].sort_values("date", ascending=False)
    return result.to_dict("records")


def get_composite_sentiment(
    ticker: str,
    date: str,
    lookback_days: int,
    event_stream_df: pd.DataFrame,
) -> float:
    """
    Weighted average sentiment for ticker over lookback window.
    Weight = magnitude * exp(-days_ago / 5.0)  (half-life ~3.5 days)

    Returns float -1.0 to 1.0, or 0.0 if no events.
    """
    events = get_ticker_events(ticker, date, lookback_days, event_stream_df)
    if not events:
        return 0.0

    qd = pd.Timestamp(date)
    total_weight = 0.0
    weighted_sentiment = 0.0

    for e in events:
        days_ago = max(0, (qd - pd.Timestamp(e["date"])).days)
        decay = math.exp(-days_ago / 5.0)
        weight = e.get("magnitude", 0.5) * decay

        weighted_sentiment += e.get("sentiment", 0.0) * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return max(-1.0, min(1.0, weighted_sentiment / total_weight))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Event Stream")
    parser.add_argument("--fetch", action="store_true", help="Fetch insider events")
    parser.add_argument("--show", type=str, help="Show recent events for ticker")
    parser.add_argument("--stream", action="store_true", help="Print full event stream")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    if args.tickers is None:
        try:
            import config
            args.tickers = list(config.WATCHLIST)[:20]
        except ImportError:
            args.tickers = ["NVDA", "AAPL", "META", "AMZN", "GOOGL"]

    if args.fetch:
        print(f"Fetching insider events for {len(args.tickers)} tickers...")
        try:
            from sec_8k_pipeline import get_cik
            cik_map = {}
            for t in args.tickers:
                cik = get_cik(t)
                if cik:
                    cik_map[t] = cik
                    print(f"  {t}: CIK={cik}")
            events = load_insider_events(args.tickers, cik_map, args.start, args.end)
            print(f"\nFetched {len(events)} insider events")
            for e in events[:10]:
                print(f"  {e['date']}  {e['ticker']:6s}  {e['event_type']:15s}  "
                      f"sent={e['sentiment']:+.2f}  mag={e['magnitude']:.2f}  {e['raw_text'][:60]}")
        except ImportError:
            print("ERROR: cannot import get_cik from sec_8k_pipeline")

    elif args.show:
        stream = get_event_stream(args.tickers, args.start, args.end)
        events = get_ticker_events(args.show, args.end, lookback_days=90, event_stream_df=stream)
        comp = get_composite_sentiment(args.show, args.end, lookback_days=30, event_stream_df=stream)
        print(f"\nEvents for {args.show} (last 90 days): {len(events)}")
        print(f"Composite sentiment (30d): {comp:+.3f}")
        for e in events[:20]:
            print(f"  {e['date']}  {e['event_type']:15s}  sent={e['sentiment']:+.2f}  "
                  f"mag={e['magnitude']:.2f}  {e['raw_text'][:60]}")

    elif args.stream:
        stream = get_event_stream(args.tickers, args.start, args.end)
        print(f"\nEvent stream: {len(stream)} events, "
              f"{stream['ticker'].nunique()} tickers")
        print(f"Date range: {stream['date'].min()} to {stream['date'].max()}")
        print(f"\nBy type:")
        for et, grp in stream.groupby("event_type"):
            print(f"  {et:20s}  {len(grp):5d}  avg_sent={grp['sentiment'].mean():+.3f}")
        print(f"\nRecent events:")
        for _, e in stream.tail(20).iterrows():
            print(f"  {str(e['date'])[:10]}  {e['ticker']:6s}  {e['event_type']:15s}  "
                  f"sent={e['sentiment']:+.2f}  {e['raw_text'][:50]}")

    else:
        # Default: show summary
        stream = get_event_stream(args.tickers, args.start, args.end, include_insider=False)
        print(f"Event stream (8-K only): {len(stream)} events")
        if len(stream) > 0:
            print(f"  Tickers: {stream['ticker'].nunique()}")
            print(f"  Date range: {stream['date'].min()} to {stream['date'].max()}")
