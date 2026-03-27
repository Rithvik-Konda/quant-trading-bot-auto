"""
institutional_flow.py — Institutional ownership changes from SEC EDGAR 13F

Parses Vanguard + BlackRock + State Street 13F filings quarterly.
Computes share count changes between quarters → net buying signal.
Maps company names to ticker symbols via CUSIP → ticker lookup.

Research: Nofsinger & Sias (1999) JF — institutional buying predicts
+12 month returns with t-stat > 4.
"""
import os, sys, json, time, re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

CACHE_DIR = '/Users/rick/ai_trading_bot_v2/cache_13f'
os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'quant-trading-bot admin@example.com'}
NS = 'http://www.sec.gov/edgar/document/thirteenf/informationtable'

# Top institutions — diverse mix of dedicated + index
INSTITUTIONS = {
    'vanguard':     '0000102909',
    'blackrock':    '0001364742',
    'statestreet':  '0000093751',
}


def get_13f_urls(cik: str, n: int = 2) -> list:
    """Get last n 13F-HR filing XML URLs for an institution."""
    url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return []

    data    = resp.json()
    filings = data['filings']['recent']
    forms   = filings['form']
    dates   = filings['filingDate']
    accnums = filings['accessionNumber']

    results = []
    for i, f in enumerate(forms):
        if f == '13F-HR' and len(results) < n:
            acc      = accnums[i]
            date     = dates[i]
            acc_nd   = acc.replace('-', '')
            cik_int  = int(cik)

            # Fetch index to find XML filename
            idx_url  = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nd}/{acc}-index.htm"
            try:
                idx_resp = requests.get(idx_url, headers=HEADERS, timeout=10)
                xml_links = re.findall(
                    r'/Archives/edgar/data/[^"]*\.xml',
                    idx_resp.text
                )
                # Pick the infotable XML (largest, not primary_doc)
                info_links = [l for l in xml_links if 'primary_doc' not in l and 'xsl' not in l]
                if info_links:
                    results.append((date, f"https://www.sec.gov{info_links[0]}"))
                time.sleep(0.3)
            except Exception:
                continue

    return results


def parse_13f_xml(url: str) -> dict:
    """
    Parse 13F XML into {company_name: shares} dict.
    Sums across share classes for same issuer.
    """
    cache_key = url.split('/')[-1].replace('.xml', '')
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        if resp.status_code != 200:
            return {}

        root     = ET.fromstring(resp.content)
        holdings = {}

        for info in root.findall(f'{{{NS}}}infoTable'):
            try:
                name   = info.find(f'{{{NS}}}nameOfIssuer').text.strip().upper()
                shares = int(info.find(f'.//{{{NS}}}sshPrnamt').text.replace(',', ''))
                # Sum across share classes
                holdings[name] = holdings.get(name, 0) + shares
            except Exception:
                continue

        with open(cache_path, 'w') as f:
            json.dump(holdings, f)

        print(f"  Parsed {len(holdings)} positions from {url.split('/')[-1]}")
        return holdings

    except Exception as e:
        print(f"  Error parsing {url}: {e}")
        return {}


def build_name_to_ticker_map(watchlist: list) -> dict:
    """
    Build mapping from company name patterns to tickers.
    Uses yfinance longName for each ticker.
    """
    cache_path = os.path.join(CACHE_DIR, 'name_to_ticker.json')
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime < 30 * 86400:
            with open(cache_path) as f:
                return json.load(f)

    import yfinance as yf
    name_map = {}
    print(f"Building name→ticker map for {len(watchlist)} symbols...")

    for sym in watchlist:
        try:
            info = yf.Ticker(sym).info
            long_name  = (info.get('longName') or '').upper()
            short_name = (info.get('shortName') or '').upper()
            # Store multiple name variants
            for name in [long_name, short_name, sym.upper()]:
                if name:
                    # Normalize: remove common suffixes
                    clean = re.sub(r'\b(INC|CORP|LTD|LLC|CO|PLC|NV|SA|AG|SE|THE|CLASS A|COM|HOLDINGS?)\b', '', name)
                    clean = re.sub(r'\s+', ' ', clean).strip()
                    if clean:
                        name_map[clean] = sym
            time.sleep(0.05)
        except Exception:
            continue

    with open(cache_path, 'w') as f:
        json.dump(name_map, f)

    print(f"Built {len(name_map)} name→ticker mappings")
    return name_map


def match_name_to_ticker(inst_name: str, name_map: dict) -> str:
    """Fuzzy match institution holding name to ticker symbol."""
    clean = re.sub(r'\b(INC|CORP|LTD|LLC|CO|PLC|NV|SA|AG|SE|THE|CLASS A|COM|HOLDINGS?|CL A|CL B)\b', '', inst_name)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Exact match first
    if clean in name_map:
        return name_map[clean]

    # Partial match — find best overlap
    best_match = None
    best_score = 0
    for map_name, ticker in name_map.items():
        if len(map_name) < 4:
            continue
        # Check if map_name is substring of inst_name or vice versa
        if map_name in clean or clean in map_name:
            score = len(min(map_name, clean, key=len))
            if score > best_score:
                best_score = score
                best_match = ticker

    return best_match if best_score >= 4 else None


def compute_institutional_changes(watchlist: list) -> pd.DataFrame:
    """
    Compute quarter-over-quarter institutional ownership changes
    for each stock in watchlist. Aggregates across top 3 institutions.
    """
    cache_path = os.path.join(CACHE_DIR, 'inst_changes.json')
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime < 7 * 86400:  # refresh weekly
            return pd.DataFrame(json.load(cache_path))

    name_map = build_name_to_ticker_map(watchlist)

    # Aggregate changes across all institutions
    ticker_changes = {sym: {'q1': 0, 'q2': 0, 'institutions': 0} for sym in watchlist}

    for inst_name, cik in INSTITUTIONS.items():
        print(f"\nFetching {inst_name} 13F filings...")
        urls = get_13f_urls(cik, n=2)

        if len(urls) < 2:
            print(f"  Only {len(urls)} filings found for {inst_name}")
            continue

        date1, url1 = urls[0]  # most recent quarter
        date2, url2 = urls[1]  # prior quarter

        print(f"  Q_recent={date1}, Q_prior={date2}")
        h1 = parse_13f_xml(url1)
        h2 = parse_13f_xml(url2)

        # Match holdings to tickers
        matched = 0
        for inst_name_holding, shares_q1 in h1.items():
            ticker = match_name_to_ticker(inst_name_holding, name_map)
            if ticker and ticker in ticker_changes:
                shares_q2 = h2.get(inst_name_holding, 0)
                ticker_changes[ticker]['q1'] += shares_q1
                ticker_changes[ticker]['q2'] += shares_q2
                ticker_changes[ticker]['institutions'] += 1
                matched += 1

        print(f"  Matched {matched} holdings to watchlist tickers")
        time.sleep(1.0)  # be nice to SEC servers

    # Compute features
    rows = []
    for sym, data in ticker_changes.items():
        q1 = data['q1']
        q2 = data['q2']
        n_inst = data['institutions']

        if q2 > 0:
            pct_change = (q1 - q2) / q2
        elif q1 > 0:
            pct_change = 1.0  # new position
        else:
            pct_change = 0.0

        rows.append({
            'symbol':           sym,
            'inst_shares_q1':   q1,
            'inst_shares_q2':   q2,
            'inst_pct_change':  float(np.clip(pct_change, -1, 2)),
            'inst_n_holders':   n_inst,
            'inst_buying':      float(np.clip(max(pct_change, 0), 0, 1)),
            'inst_selling':     float(np.clip(-min(pct_change, 0), 0, 1)),
            'inst_new_position': float(q2 == 0 and q1 > 0),
        })

    with open(cache_path, 'w') as f:
        json.dump(rows, f)

    return pd.DataFrame(rows)


def compute_13f_features(symbol: str, df_inst: pd.DataFrame = None) -> dict:
    """Return 13F features for a single symbol."""
    defaults = {
        'inst_pct_change':   0.0,
        'inst_buying':       0.0,
        'inst_selling':      0.0,
        'inst_n_holders':    0.0,
        'inst_new_position': 0.0,
    }

    if df_inst is None or len(df_inst) == 0:
        return defaults

    row = df_inst[df_inst['symbol'] == symbol]
    if len(row) == 0:
        return defaults

    r = row.iloc[0]
    return {
        'inst_pct_change':   float(r.get('inst_pct_change', 0)),
        'inst_buying':       float(r.get('inst_buying', 0)),
        'inst_selling':      float(r.get('inst_selling', 0)),
        'inst_n_holders':    float(r.get('inst_n_holders', 0)),
        'inst_new_position': float(r.get('inst_new_position', 0)),
    }


if __name__ == "__main__":
    import config

    print("Building institutional ownership change database...")
    df = compute_institutional_changes(config.WATCHLIST)

    print(f"\n{'='*65}")
    print(f"TOP INSTITUTIONAL BUYERS (QoQ increase)")
    print(f"{'='*65}")
    df_buy = df[df['inst_buying'] > 0].sort_values('inst_buying', ascending=False)
    print(f"{'Symbol':<8} {'Q2_shares':>12} {'Q1_shares':>12} {'Change':>8} {'N_inst':>7}")
    print("-"*50)
    for _, r in df_buy.head(15).iterrows():
        print(f"  {r['symbol']:<6} {r['inst_shares_q2']:>12,.0f} "
              f"{r['inst_shares_q1']:>12,.0f} "
              f"{r['inst_pct_change']:>+8.1%} "
              f"{r['inst_n_holders']:>7.0f}")

    print(f"\n{'='*65}")
    print(f"TOP INSTITUTIONAL SELLERS (QoQ decrease)")
    print(f"{'='*65}")
    df_sell = df[df['inst_selling'] > 0].sort_values('inst_selling', ascending=False)
    for _, r in df_sell.head(10).iterrows():
        print(f"  {r['symbol']:<6} {r['inst_shares_q2']:>12,.0f} "
              f"{r['inst_shares_q1']:>12,.0f} "
              f"{r['inst_pct_change']:>+8.1%}")

    df.to_csv('/Users/rick/ai_trading_bot_v2/inst_flow_today.csv', index=False)
    print(f"\nSaved: inst_flow_today.csv")
