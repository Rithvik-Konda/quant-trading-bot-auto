"""
entity_graph.py — EDGAR 10-K Relationship Extractor and Supply Chain Graph
===========================================================================
Extracts supplier/customer/competitor/geographic/regulatory relationships
from SEC 10-K filings. Builds a directed graph for event propagation scoring.

Usage:
    python v2/entity_graph.py --build                # build full graph
    python v2/entity_graph.py --build --tickers NVDA AAPL
    python v2/entity_graph.py --query NVDA           # show relationships
"""
from __future__ import annotations

import os
import sys
import re
import csv
import html
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))
sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2/v2"))

import pandas as pd

CACHE_DIR = Path(os.path.expanduser("~/ai_trading_bot_v2/cache_graph"))
CACHE_DIR.mkdir(exist_ok=True)

SEC_HEADERS = {"User-Agent": "research@tradingbot.com"}
SEC_SLEEP = 0.5  # 2 requests/sec


# ── Relationship patterns ────────────────────────────────────────────────────

SUPPLIER_PATTERNS = [
    r"we purchase from", r"we source from", r"our supplier",
    r"supplied by", r"we rely on", r"sole supplier",
    r"primary supplier", r"key supplier", r"we obtain from",
    r"manufactured by", r"produced by",
]

CUSTOMER_PATTERNS = [
    r"our customers include", r"sold to", r"our largest customer",
    r"our key customers", r"customer concentration", r"revenue from",
    r"sales to", r"we sell to", r"our clients include",
]

COMPETITOR_PATTERNS = [
    r"compete with", r"competition from", r"our competitors include",
    r"competitive with", r"we compete against",
]

GEOGRAPHIC_PATTERNS = [
    r"operations in", r"manufacturing in", r"facilities in",
    r"revenue from .{0,30} region", r"exposure to",
    r"located in",
]

REGULATORY_PATTERNS = [
    r"subject to", r"regulated by", r"approval from",
    r"oversight by", r"compliance with",
    r"\bFDA\b", r"\bFCC\b", r"\bEPA\b", r"\bCFIUS\b",
]

HIGH_STRENGTH_WORDS = {"sole", "primary", "largest", "only", "exclusive", "critical"}
MED_STRENGTH_WORDS = {"key", "significant", "major", "important", "principal", "substantial"}


# ═══════════════════════════════════════════════════════════════════════════════
#  1. FETCH 10-K TEXT
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_business_risk_sections(raw_text: str) -> str:
    """
    Extract Item 1 (Business) + Item 1A (Risk Factors) from 10-K text.
    Falls back to characters 20000-120000 if section markers not found
    or if the extracted section is under 5000 chars (TOC false match).
    Caps at 100000 characters.
    """
    # Find ALL Item 1 matches — the first is often a TOC entry
    # The real section heading is the one followed by substantial text
    item1_pattern = re.compile(r"(?:^|\s)ITEM\s+1[.\s:]+\s*(?:BUSINESS|Business)", re.IGNORECASE)
    item2_pattern = re.compile(r"(?:^|\s)ITEM\s+2[.\s:]", re.IGNORECASE)

    best_text = ""

    for item1_match in item1_pattern.finditer(raw_text):
        start = item1_match.start()
        # Find the next Item 2 after this Item 1
        item2_match = item2_pattern.search(raw_text, pos=start + 200)
        if item2_match:
            end = item2_match.start()
        else:
            end = start + 100000
        candidate = raw_text[start:end]

        # The real section is the longest match (TOC entries are short)
        if len(candidate) > len(best_text):
            best_text = candidate

    # If no "BUSINESS" label, try broader Item 1 pattern
    if len(best_text) < 5000:
        item1_broad = re.compile(r"(?:^|\s)ITEM\s+1[.\s]", re.IGNORECASE)
        for item1_match in item1_broad.finditer(raw_text):
            start = item1_match.start()
            item2_match = item2_pattern.search(raw_text, pos=start + 200)
            end = item2_match.start() if item2_match else start + 100000
            candidate = raw_text[start:end]
            if len(candidate) > len(best_text):
                best_text = candidate

    # If extracted section is under 5000 chars, it's a TOC match — fall back
    if len(best_text) < 5000:
        best_text = raw_text[20000:120000]

    return best_text[:100000]


def get_10k_text(ticker: str, cik: Optional[str] = None) -> Optional[str]:
    """
    Fetch latest 10-K filing text from EDGAR.
    Cached to cache_graph/{ticker}_10k.txt — skips if cache < 90 days old.

    Args:
        ticker: stock ticker
        cik: CIK number (fetched via get_cik if None)

    Returns:
        Business + Risk Factors sections (up to 100K chars), or None on failure.
    """
    cache_file = CACHE_DIR / f"{ticker}_10k.txt"
    if cache_file.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days < 90:
            return cache_file.read_text()

    if cik is None:
        try:
            from sec_8k_pipeline import get_cik
            cik = get_cik(ticker)
        except ImportError:
            return None
    if cik is None:
        return None

    cik_padded = cik.zfill(10)

    # Step 1: get filing list from submissions endpoint
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        time.sleep(SEC_SLEEP)

        if resp.status_code != 200:
            return None

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        # Find most recent 10-K, or 20-F for foreign private issuers (e.g. TSMC)
        tenk_idx = None
        for i, form in enumerate(forms):
            if form == "10-K":
                tenk_idx = i
                break
        if tenk_idx is None:
            for i, form in enumerate(forms):
                if form == "20-F":
                    tenk_idx = i
                    break

        if tenk_idx is None:
            return None

        acc_dashed = accessions[tenk_idx]           # e.g. "0001045810-26-000021"
        acc_nodash = acc_dashed.replace("-", "")     # e.g. "000104581026000021"
        primary_doc = primary_docs[tenk_idx] if tenk_idx < len(primary_docs) else None
        cik_clean = cik.lstrip("0") or "0"

    except Exception:
        return None

    # Step 2: fetch index page, find the largest .htm document (the actual 10-K)
    try:
        base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_nodash}"
        index_url = f"{base_url}/{acc_dashed}-index.htm"
        idx_resp = requests.get(index_url, headers=SEC_HEADERS, timeout=15)
        time.sleep(SEC_SLEEP)

        # If -index.htm fails, try the directory listing
        if idx_resp.status_code != 200:
            index_url = f"{base_url}/"
            idx_resp = requests.get(index_url, headers=SEC_HEADERS, timeout=15)
            time.sleep(SEC_SLEEP)

        if idx_resp.status_code != 200:
            return None

        # Find all .htm/.txt document links, including XBRL ix?doc= wrapper links
        # EDGAR wraps the primary 10-K in an inline viewer: /ix?doc=/Archives/...
        raw_links = re.findall(
            r'href="(?:/ix\?doc=)?([^"]+\.(?:htm|txt))"',
            idx_resp.text, re.IGNORECASE,
        )
        # Also extract paths from ix?doc= style links separately
        ix_links = re.findall(
            r'href="/ix\?doc=(/Archives/[^"]+\.(?:htm|txt))"',
            idx_resp.text, re.IGNORECASE,
        )
        doc_links = list(dict.fromkeys(raw_links + ix_links))  # dedupe, preserve order

        # Filter out index pages, R-files, and exhibits
        def _is_exhibit(href: str) -> bool:
            filename = href.split("/")[-1]
            return bool(re.search(r'(?:^|[-_x])ex\d', filename, re.IGNORECASE))

        doc_links = [
            d for d in doc_links
            if not d.endswith("-index.htm")
            and "/R" not in d
            and not _is_exhibit(d)
        ]

        if not doc_links:
            # Fallback to primaryDocument from submissions API
            if primary_doc:
                doc_links = [primary_doc]
            else:
                return None

        # Score documents by size — the real 10-K is the largest .htm file
        best_doc_url = None
        best_size = 0

        for doc_href in doc_links:
            if doc_href.startswith("http"):
                full_url = doc_href
            elif doc_href.startswith("/Archives/"):
                full_url = f"https://www.sec.gov{doc_href}"
            else:
                full_url = f"{base_url}/{doc_href}"
            try:
                head_resp = requests.head(full_url, headers=SEC_HEADERS, timeout=10)
                time.sleep(SEC_SLEEP)
                size = int(head_resp.headers.get("Content-Length", 0))
                if size > best_size:
                    best_size = size
                    best_doc_url = full_url
            except Exception:
                continue

        # If HEAD requests didn't work, just take the first non-exhibit .htm
        if best_doc_url is None and doc_links:
            doc_href = doc_links[0]
            if doc_href.startswith("http"):
                best_doc_url = doc_href
            elif doc_href.startswith("/Archives/"):
                best_doc_url = f"https://www.sec.gov{doc_href}"
            else:
                best_doc_url = f"{base_url}/{doc_href}"

        if best_doc_url is None:
            return None

        doc_resp = requests.get(best_doc_url, headers=SEC_HEADERS, timeout=30)
        time.sleep(SEC_SLEEP)

        if doc_resp.status_code != 200:
            return None

        # Strip HTML
        raw = re.sub(r"<[^>]+>", " ", doc_resp.text)
        raw = re.sub(r"\s+", " ", raw).strip()

        # Extract Business (Item 1) + Risk Factors (Item 1A) sections
        text = _extract_business_risk_sections(raw)

        # Cache
        cache_file.write_text(text)
        return text

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  2. EXTRACT RELATIONSHIPS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_sentence(text: str, match_start: int, max_len: int = 200) -> str:
    """Extract the sentence surrounding a regex match."""
    # Find sentence boundaries
    start = max(0, match_start - 80)
    end = min(len(text), match_start + max_len)

    # Walk back to sentence start
    for i in range(match_start, start, -1):
        if text[i] in ".!?\n":
            start = i + 1
            break

    # Walk forward to sentence end
    for i in range(match_start, end):
        if text[i] in ".!?\n":
            end = i + 1
            break

    return text[start:end].strip()


def _assign_strength(sentence: str) -> float:
    """Assign relationship strength based on language intensity."""
    lower = sentence.lower()
    for word in HIGH_STRENGTH_WORDS:
        if word in lower:
            return 1.0
    for word in MED_STRENGTH_WORDS:
        if word in lower:
            return 0.7
    return 0.4


def extract_relationships(ticker: str, text: str) -> List[Dict]:
    """
    Extract supplier/customer/competitor/geographic/regulatory relationships
    from 10-K text using regex pattern matching.

    Returns list of relationship dicts.
    """
    if not text:
        return []

    # Clean residual HTML entities and normalize whitespace
    text = html.unescape(text)
    text = re.sub(r'&#\d+;|&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    relationships = []

    pattern_groups = [
        ("supplier",    SUPPLIER_PATTERNS),
        ("customer",    CUSTOMER_PATTERNS),
        ("competitor",  COMPETITOR_PATTERNS),
        ("geographic",  GEOGRAPHIC_PATTERNS),
        ("regulatory",  REGULATORY_PATTERNS),
    ]

    for rel_type, patterns in pattern_groups:
        for pattern in patterns:
            try:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    sentence = _extract_sentence(text, match.start())
                    strength = _assign_strength(sentence)

                    # Try to extract the target entity from the sentence
                    # Look for capitalized words near the match as entity candidates
                    target = _extract_entity(sentence, pattern)

                    relationships.append({
                        "source": ticker,
                        "target": target,
                        "relationship": rel_type,
                        "strength": strength,
                        "raw_text": sentence[:200],
                        "filing_date": "",  # filled by caller if known
                    })
            except Exception:
                continue

    return relationships


def _extract_entity(sentence: str, pattern: str) -> str:
    """Extract company/entity name from sentence, in priority order."""
    # Priority 1: quoted strings
    quoted = re.findall(r'"([^"]{3,60})"', sentence)
    if quoted:
        return quoted[0][:60]

    # Priority 2: ALL-CAPS ticker-like tokens (2-5 uppercase letters)
    SKIP_CAPS = {'AND', 'OR', 'FOR', 'THE', 'NOT', 'INC', 'LLC', 'ANY',
                 'ALL', 'OUR', 'WE', 'IF', 'US', 'CEO', 'CFO', 'SEC',
                 'FDA', 'FCC', 'EPA', 'AI', 'IP', 'IT', 'R&D'}
    caps_tokens = re.findall(r'\b([A-Z]{2,5})\b', sentence)
    for token in caps_tokens:
        if token not in SKIP_CAPS:
            return token

    # Priority 3: multi-word Title Case sequences (2+ words)
    SKIP_STARTS = {'We', 'Our', 'If', 'For', 'The', 'This', 'These',
                   'Furthermore', 'Additionally', 'However', 'Such',
                   'Any', 'Each', 'All', 'While', 'When', 'As'}
    multi_word = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b', sentence)
    candidates = [m for m in multi_word if m.split()[0] not in SKIP_STARTS]
    if candidates:
        return max(candidates, key=len)[:60]

    # Priority 4: fallback — text after the keyword
    m = re.search(pattern, sentence, re.IGNORECASE)
    if m:
        after = sentence[m.end():m.end() + 50].strip()
        return " ".join(after.split()[:4]) if after else "unknown"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  3. BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(tickers: List[str]):
    """
    Build entity relationship graph from 10-K filings.
    Saves to cache_graph/entity_graph.json and cache_graph/relationships.csv.

    Args:
        tickers: list of stock tickers to process
    """
    try:
        import networkx as nx
    except ImportError:
        print("ERROR: networkx not installed. Run: pip install networkx")
        return None

    try:
        from sec_8k_pipeline import get_cik
    except ImportError:
        print("ERROR: cannot import get_cik from sec_8k_pipeline")
        return None

    G = nx.DiGraph()
    all_relationships = []

    print(f"Building entity graph for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        # Resume support: skip if cached text exists and is fresh
        cache_file = CACHE_DIR / f"{ticker}_10k.txt"

        print(f"  [{i + 1}/{len(tickers)}] {ticker}...", end=" ", flush=True)

        cik = get_cik(ticker)
        if cik is None:
            # Fallback: search EDGAR for this ticker's 10-K/20-F filings
            try:
                _search_url = "https://efts.sec.gov/LATEST/search-index"
                _search_params = {
                    "q": f'"{ticker}"',
                    "dateRange": "custom",
                    "startdt": "2020-01-01",
                    "forms": "10-K,20-F",
                }
                _search_resp = requests.get(_search_url, params=_search_params,
                                            headers=SEC_HEADERS, timeout=15)
                time.sleep(SEC_SLEEP)
                if _search_resp.status_code == 200:
                    _hits = _search_resp.json().get("hits", {}).get("hits", [])
                    for _hit in _hits:
                        _src = _hit.get("_source", {})
                        _display = _src.get("display_names", [])
                        if any(f"({ticker.upper()})" in dn for dn in _display):
                            _ciks = _src.get("ciks", [])
                            if _ciks:
                                cik = _ciks[0].lstrip("0") or _ciks[0]
                                break
            except Exception:
                pass
        if cik is None:
            print("no CIK")
            continue

        text = get_10k_text(ticker, cik)
        if text is None:
            print("no 10-K")
            continue

        rels = extract_relationships(ticker, text)
        print(f"{len(rels)} relationships")

        # Add to graph
        G.add_node(ticker, type="company", cik=cik)
        for rel in rels:
            target = rel["target"]
            G.add_node(target, type="entity")
            G.add_edge(
                rel["source"], target,
                relationship=rel["relationship"],
                strength=rel["strength"],
                raw_text=rel["raw_text"][:100],
            )
            all_relationships.append(rel)

    # Save graph as node-link JSON
    graph_data = nx.node_link_data(G)
    graph_file = CACHE_DIR / "entity_graph.json"
    graph_file.write_text(json.dumps(graph_data, indent=2, default=str))
    print(f"\nGraph saved: {len(G.nodes)} nodes, {len(G.edges)} edges → {graph_file}")

    # Save relationships CSV
    if all_relationships:
        csv_file = CACHE_DIR / "relationships.csv"
        df = pd.DataFrame(all_relationships)
        df.to_csv(csv_file, index=False)
        print(f"Relationships saved: {len(df)} rows → {csv_file}")

    return G


# ═══════════════════════════════════════════════════════════════════════════════
#  4. QUERY NEIGHBORS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_graph():
    """Load graph from cache."""
    try:
        import networkx as nx
    except ImportError:
        return None

    graph_file = CACHE_DIR / "entity_graph.json"
    if not graph_file.exists():
        return None
    try:
        data = json.loads(graph_file.read_text())
        return nx.node_link_graph(data)
    except Exception:
        return None


def get_neighbors(
    ticker: str,
    graph=None,
    relationship_type: Optional[str] = None,
    direction: str = "both",
) -> List[Dict]:
    """
    Get connected entities for a ticker.

    Args:
        ticker: stock ticker
        graph: networkx DiGraph (loaded from cache if None)
        relationship_type: filter by type (supplier/customer/competitor/etc.)
        direction: "upstream" (suppliers), "downstream" (customers), "both"

    Returns:
        List of {ticker, relationship, strength, raw_text}
    """
    if graph is None:
        graph = _load_graph()
    if graph is None or ticker not in graph:
        return []

    results = []

    # Outgoing edges: ticker → target (downstream: customers, competitors)
    if direction in ("both", "downstream"):
        for _, target, data in graph.out_edges(ticker, data=True):
            rel = data.get("relationship", "")
            if relationship_type and rel != relationship_type:
                continue
            results.append({
                "ticker": target,
                "relationship": rel,
                "strength": data.get("strength", 0),
                "raw_text": data.get("raw_text", ""),
                "direction": "downstream",
            })

    # Incoming edges: source → ticker (upstream: suppliers)
    if direction in ("both", "upstream"):
        for source, _, data in graph.in_edges(ticker, data=True):
            rel = data.get("relationship", "")
            if relationship_type and rel != relationship_type:
                continue
            results.append({
                "ticker": source,
                "relationship": rel,
                "strength": data.get("strength", 0),
                "raw_text": data.get("raw_text", ""),
                "direction": "upstream",
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  5. EVENT PROPAGATION SCORING
# ═══════════════════════════════════════════════════════════════════════════════

PROPAGATION_WEIGHTS = {
    "supplier":   0.6,   # supplier event affects us moderately
    "customer":   0.8,   # customer event affects us strongly
    "competitor": -0.4,  # competitor event has opposite effect
    "geographic": 0.3,   # geographic co-location: shared risk
    "regulatory": 0.5,   # regulatory event affects similarly regulated peers
}


def get_propagation_score(
    ticker: str,
    graph=None,
    event_scores: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute how events at related companies propagate to this ticker.

    Args:
        ticker: stock to evaluate
        graph: networkx DiGraph
        event_scores: dict of {company_name: sentiment_score} for recent events

    Returns:
        Weighted sum of propagation effects, clipped to [-1, 1].
    """
    if event_scores is None or not event_scores:
        return 0.0

    if graph is None:
        graph = _load_graph()
    if graph is None or ticker not in graph:
        return 0.0

    total = 0.0

    # Check all neighbors
    neighbors = get_neighbors(ticker, graph)
    for nbr in neighbors:
        nbr_name = nbr["ticker"]
        rel_type = nbr["relationship"]
        strength = nbr["strength"]

        # Check if this neighbor has a recent event score
        event_score = event_scores.get(nbr_name, 0.0)
        if event_score == 0.0:
            # Try case-insensitive match
            for name, score in event_scores.items():
                if name.upper() == nbr_name.upper():
                    event_score = score
                    break

        if event_score == 0.0:
            continue

        # Apply propagation weight
        prop_weight = PROPAGATION_WEIGHTS.get(rel_type, 0.3)
        contribution = event_score * strength * prop_weight
        total += contribution

    return max(-1.0, min(1.0, total))


# ═══════════════════════════════════════════════════════════════════════════════
#  6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entity Relationship Graph Builder")
    parser.add_argument("--build", action="store_true", help="Build graph from 10-K filings")
    parser.add_argument("--query", type=str, help="Query relationships for a ticker")
    parser.add_argument("--tickers", nargs="+", default=None, help="Specific tickers to process")
    parser.add_argument("--type", type=str, default=None,
                        help="Filter by relationship type: supplier/customer/competitor/geographic/regulatory")
    args = parser.parse_args()

    if args.build:
        if args.tickers:
            tickers = args.tickers
        else:
            try:
                import config
                tickers = list(config.WATCHLIST)
            except ImportError:
                tickers = ["NVDA", "AAPL", "META", "AMZN", "GOOGL",
                            "MSFT", "TSLA", "JPM", "XOM", "UNH"]
        build_graph(tickers)

    elif args.query:
        graph = _load_graph()
        if graph is None:
            print("No graph built. Run with --build first.")
        else:
            rels = get_neighbors(args.query, graph, relationship_type=args.type)
            print(f"\nRelationships for {args.query}: {len(rels)}")
            for r in rels:
                print(f"  {r['direction']:>10s}  {r['relationship']:12s}  "
                      f"strength={r['strength']:.1f}  {r['ticker']}")
                if r["raw_text"]:
                    print(f"{'':14s}  \"{r['raw_text'][:80]}\"")

    else:
        # Show graph summary
        graph = _load_graph()
        if graph is None:
            print("No graph built. Run with --build first.")
        else:
            print(f"Entity graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
            rel_counts: Dict[str, int] = {}
            for _, _, data in graph.edges(data=True):
                rel = data.get("relationship", "unknown")
                rel_counts[rel] = rel_counts.get(rel, 0) + 1
            print("  By type:")
            for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
                print(f"    {rel:15s} {count:5d}")
