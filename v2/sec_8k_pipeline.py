"""
sec_8k_pipeline.py — SEC EDGAR 8-K Ingestion and Rule-Based Scoring
====================================================================
Standalone module. No ML models. Pure item-type classification.

Usage:
  python v2/sec_8k_pipeline.py --backfill --start 2024-01-01 --end 2024-12-31
  python v2/sec_8k_pipeline.py --signal NVDA 2024-06-15
"""
from __future__ import annotations

import os
import sys
import re
import csv
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))

import pandas as pd

CACHE_DIR = Path(os.path.expanduser("~/ai_trading_bot_v2/cache_8k"))
CACHE_DIR.mkdir(exist_ok=True)

SEC_HEADERS = {
    "User-Agent": "TradingBot research@example.com",
    "Accept-Encoding": "gzip, deflate",
}
SEC_RATE_LIMIT = 0.1  # 10 requests/sec allowed

_cik_cache: Dict[str, str] = {}


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 — EDGAR INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

# ── 3. CIK lookup (defined first since get_8k_filings depends on it) ─────────

def get_cik(ticker: str) -> Optional[str]:
    """
    Look up CIK number for a ticker from SEC EDGAR.
    Results cached in-memory.

    Returns CIK string (zero-padded to 10 digits) or None.
    """
    ticker = ticker.upper()
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        "action": "getcompany",
        "company": ticker,
        "CIK": ticker,
        "type": "8-K",
        "dateb": "",
        "owner": "include",
        "count": "1",
        "search_text": "",
        "output": "atom",
    }

    try:
        resp = requests.get(url, params=params, headers=SEC_HEADERS, timeout=10)
        time.sleep(SEC_RATE_LIMIT)

        if resp.status_code != 200:
            return None

        # Parse CIK from Atom XML response
        # Look for <CIK> tag or cik= in URLs
        text = resp.text
        cik_match = re.search(r"<CIK>(\d+)</CIK>", text, re.IGNORECASE)
        if cik_match:
            cik = cik_match.group(1).zfill(10)
            _cik_cache[ticker] = cik
            return cik

        # Fallback: look for CIK in accession URLs
        cik_match = re.search(r"/cgi-bin/browse-edgar\?action=getcompany&amp;CIK=(\d+)", text)
        if cik_match:
            cik = cik_match.group(1).zfill(10)
            _cik_cache[ticker] = cik
            return cik

        # Fallback: try the company tickers JSON
        try:
            tickers_resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=SEC_HEADERS, timeout=10,
            )
            time.sleep(SEC_RATE_LIMIT)
            if tickers_resp.status_code == 200:
                tickers_data = tickers_resp.json()
                for entry in tickers_data.values():
                    if entry.get("ticker", "").upper() == ticker:
                        cik = str(entry["cik_str"]).zfill(10)
                        _cik_cache[ticker] = cik
                        return cik
        except Exception:
            pass

        return None

    except Exception:
        return None


# ── 1. 8-K filing search ─────────────────────────────────────────────────────

def get_8k_filings(
    ticker: str,
    start_date: str,
    end_date: str,
) -> List[Dict]:
    """
    Search EDGAR full-text search for 8-K filings.

    Args:
        ticker: stock ticker e.g. 'NVDA'
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'

    Returns:
        List of dicts: {ticker, date, accession_number, url, item_types}
    """
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": f'"{ticker}"',
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "forms": "8-K",
    }

    try:
        resp = requests.get(url, params=params, headers=SEC_HEADERS, timeout=15)
        time.sleep(SEC_RATE_LIMIT)

        if resp.status_code != 200:
            # Fallback to EDGAR full-text search API
            return _get_8k_filings_fulltext(ticker, start_date, end_date)

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return _get_8k_filings_fulltext(ticker, start_date, end_date)

        filings = []
        for hit in hits:
            source = hit.get("_source", {})
            filing = {
                "ticker": ticker,
                "date": source.get("file_date", ""),
                "accession_number": source.get("accession_no", "").replace("-", ""),
                "url": f"https://www.sec.gov/Archives/edgar/data/{source.get('entity_id', '')}/{source.get('accession_no', '').replace('-', '')}/",
                "item_types": source.get("items", ""),
            }
            if filing["date"]:
                filings.append(filing)

        return filings

    except Exception:
        return _get_8k_filings_fulltext(ticker, start_date, end_date)


def _get_8k_filings_fulltext(
    ticker: str,
    start_date: str,
    end_date: str,
) -> List[Dict]:
    """Fallback: use EDGAR full-text search API."""
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": f'"{ticker}"',
        "forms": "8-K",
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
    }

    try:
        # Try the newer EDGAR search endpoint
        alt_url = "https://efts.sec.gov/LATEST/search-index"
        resp = requests.get(alt_url, params=params, headers=SEC_HEADERS, timeout=15)
        time.sleep(SEC_RATE_LIMIT)

        if resp.status_code != 200:
            return []

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        filings = []
        for hit in hits:
            source = hit.get("_source", {})
            filings.append({
                "ticker": ticker,
                "date": source.get("file_date", ""),
                "accession_number": source.get("accession_no", "").replace("-", ""),
                "url": "",
                "item_types": source.get("items", ""),
            })

        return [f for f in filings if f["date"]]

    except Exception:
        return []


# ── 2. Fetch 8-K text ────────────────────────────────────────────────────────

def fetch_8k_text(accession_number: str, cik: Optional[str] = None) -> Optional[str]:
    """
    Fetch the actual 8-K document text from EDGAR.

    Args:
        accession_number: e.g. '0001193125-24-012345' or '000119312524012345'
        cik: CIK number (optional, speeds up URL construction)

    Returns:
        Raw text truncated to first 5000 characters, or None on failure.
    """
    # Normalize accession number
    acc = accession_number.replace("-", "")
    if len(acc) < 18:
        return None

    # Format with dashes: XXXXXXXXXX-YY-ZZZZZZ
    acc_dashed = f"{acc[:10]}-{acc[10:12]}-{acc[12:]}"

    # Try to find the filing index page
    if cik:
        cik_clean = cik.lstrip("0") or "0"
    else:
        cik_clean = acc[:10].lstrip("0") or "0"

    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc}/"

    try:
        resp = requests.get(index_url, headers=SEC_HEADERS, timeout=10)
        time.sleep(SEC_RATE_LIMIT)

        if resp.status_code != 200:
            return None

        # Find the primary 8-K document (usually .htm or .txt)
        doc_match = re.search(
            r'href="([^"]*(?:8-k|8k)[^"]*\.(?:htm|txt))"',
            resp.text, re.IGNORECASE,
        )
        if not doc_match:
            # Try any .htm file
            doc_match = re.search(r'href="([^"]*\.htm)"', resp.text, re.IGNORECASE)

        if not doc_match:
            return None

        doc_path = doc_match.group(1)
        if not doc_path.startswith("http"):
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc}/{doc_path}"
        else:
            doc_url = doc_path

        doc_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=10)
        time.sleep(SEC_RATE_LIMIT)

        if doc_resp.status_code != 200:
            return None

        # Strip HTML tags for cleaner text
        text = re.sub(r"<[^>]+>", " ", doc_resp.text)
        text = re.sub(r"\s+", " ", text).strip()

        return text[:5000]

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 — ITEM CLASSIFICATION (rule-based)
# ═══════════════════════════════════════════════════════════════════════════════

ITEM_NUMBERS = [
    "1.01", "1.02", "1.03",
    "2.01", "2.02", "2.04", "2.05", "2.06",
    "5.01", "5.02", "5.03",
    "7.01",
    "8.01",
    "9.01",
]

# ── 4. Classify items ────────────────────────────────────────────────────────

def classify_8k_items(filing_text: str) -> Dict[str, bool]:
    """
    Detect which 8-K item types are present in the filing text.
    Simple regex search for "Item X.XX" patterns.

    Returns dict: {item_number: True/False} for each tracked item.
    """
    if not filing_text:
        return {}

    result = {}
    for item in ITEM_NUMBERS:
        # Match "Item 1.01" with flexible spacing and formatting
        pattern = rf"Item\s+{re.escape(item)}"
        result[item] = bool(re.search(pattern, filing_text, re.IGNORECASE))

    return result


# ── 5. Score sentiment ────────────────────────────────────────────────────────

# Sentiment weights per item type
# Positive: new agreement, earnings, reg FD disclosure
# Negative: termination, bankruptcy, missed payment, exit costs, impairment, exec departure
ITEM_SENTIMENT = {
    "1.01": +0.2,   # Entry into material definitive agreement
    "1.02": -0.3,   # Termination of material definitive agreement
    "1.03": -1.0,   # Bankruptcy or receivership
    "2.01": +0.0,   # Completion of acquisition/disposition (neutral — direction unknown)
    "2.02": +0.1,   # Results of operations (earnings — usually slightly positive if filed off-cycle)
    "2.04": -0.5,   # Triggering events that accelerate obligation (missed payment)
    "2.05": -0.2,   # Exit/disposal costs
    "2.06": -0.3,   # Material impairment
    "5.01": +0.0,   # Changes in control (neutral)
    "5.02": -0.4,   # Departure of directors/officers
    "5.03": +0.0,   # Amendments to articles (neutral)
    "7.01": +0.1,   # Regulation FD disclosure (usually positive guidance)
    "8.01": +0.0,   # Other events (neutral)
    "9.01": +0.0,   # Financial statements and exhibits (neutral)
}


def score_8k_sentiment(item_types: Dict[str, bool]) -> float:
    """
    Rule-based sentiment score from detected 8-K items.
    No ML model — pure item-type weights.

    Args:
        item_types: dict from classify_8k_items()

    Returns:
        Float -1.0 to +1.0
    """
    if not item_types:
        return 0.0

    total = 0.0
    for item, present in item_types.items():
        if present and item in ITEM_SENTIMENT:
            total += ITEM_SENTIMENT[item]

    return max(-1.0, min(1.0, total))


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 — HISTORICAL BACKFILL
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT = CACHE_DIR / "8k_history.csv"

# ── 6. Build history ─────────────────────────────────────────────────────────

def build_8k_history(
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_path: str = str(DEFAULT_OUTPUT),
) -> pd.DataFrame:
    """
    Fetch all 8-Ks for given tickers and date range, score them,
    and save to CSV.

    Args:
        tickers: list of stock tickers
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        output_path: CSV output path

    Returns:
        DataFrame of all filings with sentiment scores.
    """
    # Load existing data for resume support
    existing_tickers = set()
    if os.path.exists(output_path):
        try:
            existing = pd.read_csv(output_path)
            existing_tickers = set(existing["ticker"].unique())
        except Exception:
            pass

    try:
        from tqdm import tqdm
        iterator = tqdm(tickers, desc="Fetching 8-Ks")
    except ImportError:
        iterator = tickers
        print(f"Processing {len(tickers)} tickers...")

    rows = []
    for ticker in iterator:
        if ticker in existing_tickers:
            continue

        filings = get_8k_filings(ticker, start_date, end_date)

        for filing in filings:
            # Classify items from the filing metadata
            item_str = filing.get("item_types", "")
            if item_str:
                # Parse pre-extracted items
                items = {item: (item in item_str) for item in ITEM_NUMBERS}
            else:
                items = {}

            sentiment = score_8k_sentiment(items)

            rows.append({
                "ticker": ticker,
                "date": filing["date"],
                "accession_number": filing["accession_number"],
                "item_types": ",".join(k for k, v in items.items() if v),
                "sentiment_score": round(sentiment, 3),
            })

    # Combine with existing data
    new_df = pd.DataFrame(rows)
    if os.path.exists(output_path) and len(existing_tickers) > 0:
        try:
            old_df = pd.read_csv(output_path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    if len(combined) > 0:
        combined = combined.drop_duplicates(subset=["ticker", "accession_number"])
        combined = combined.sort_values(["ticker", "date"])
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"Saved {len(combined)} filings to {output_path}")

    return combined


# ── 7. Load history ──────────────────────────────────────────────────────────

def load_8k_history(path: str = str(DEFAULT_OUTPUT)) -> pd.DataFrame:
    """
    Load the CSV built by build_8k_history.

    Returns DataFrame indexed by (ticker, date).
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ticker", "date", "accession_number",
                                     "item_types", "sentiment_score"])

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4 — SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── 8. Get signal ────────────────────────────────────────────────────────────

def get_8k_signal(
    ticker: str,
    query_date: str,
    history_df: pd.DataFrame,
    lookback_days: int = 5,
) -> float:
    """
    Returns the max absolute sentiment score from any 8-K filed
    in the lookback window.

    Args:
        ticker: stock ticker
        query_date: 'YYYY-MM-DD' or datetime
        history_df: DataFrame from load_8k_history()
        lookback_days: number of days to look back (default 5)

    Returns:
        Float -1.0 to +1.0. Returns 0.0 if no filings in window.
    """
    if history_df is None or len(history_df) == 0:
        return 0.0

    if isinstance(query_date, str):
        qd = pd.Timestamp(query_date)
    else:
        qd = pd.Timestamp(query_date)

    window_start = qd - pd.Timedelta(days=lookback_days)

    mask = (
        (history_df["ticker"] == ticker) &
        (history_df["date"] >= window_start) &
        (history_df["date"] <= qd)
    )
    window = history_df[mask]

    if len(window) == 0:
        return 0.0

    scores = window["sentiment_score"].dropna()
    if len(scores) == 0:
        return 0.0

    # Return the score with the largest absolute value
    max_idx = scores.abs().idxmax()
    return float(scores.loc[max_idx])


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SEC 8-K Pipeline")
    parser.add_argument("--backfill", action="store_true", help="Run historical backfill")
    parser.add_argument("--signal", nargs=2, metavar=("TICKER", "DATE"),
                        help="Get 8-K signal for ticker on date")
    parser.add_argument("--start", default="2024-01-01", help="Backfill start date")
    parser.add_argument("--end", default="2024-12-31", help="Backfill end date")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to backfill")
    args = parser.parse_args()

    if args.backfill:
        if args.tickers:
            tickers = args.tickers
        else:
            try:
                import config
                tickers = list(config.WATCHLIST)[:50]
            except ImportError:
                tickers = ["NVDA", "AAPL", "META", "AMZN", "GOOGL"]

        build_8k_history(tickers, args.start, args.end)

    elif args.signal:
        ticker, date_str = args.signal
        history = load_8k_history()
        score = get_8k_signal(ticker, date_str, history)
        print(f"8-K signal for {ticker} on {date_str}: {score:+.3f}")

    else:
        # Default: show summary of cached history
        history = load_8k_history()
        if len(history) == 0:
            print("No 8-K history cached. Run with --backfill first.")
        else:
            print(f"8-K History: {len(history)} filings, "
                  f"{history['ticker'].nunique()} tickers")
            print(f"Date range: {history['date'].min()} to {history['date'].max()}")
            neg = history[history["sentiment_score"] < -0.1]
            pos = history[history["sentiment_score"] > 0.1]
            print(f"Negative filings: {len(neg)}")
            print(f"Positive filings: {len(pos)}")
            if len(neg) > 0:
                print(f"\nMost negative recent:")
                for _, r in neg.nlargest(5, "date").iterrows():
                    print(f"  {r['ticker']:6s} {str(r['date'])[:10]}  "
                          f"score={r['sentiment_score']:+.2f}  items={r['item_types']}")
