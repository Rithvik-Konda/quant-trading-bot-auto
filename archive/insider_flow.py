"""
insider_flow.py — Insider buying clusters from SEC EDGAR Form 4

Research basis:
- Seyhun (1998): insider buying predicts +18% 12-month returns
- Jeng, Metrick & Zeckhauser (2003 RFS): insider purchases earn 6% abnormal return
- Most powerful in BEAR regime — insiders buying their own stock during market panic
  is the strongest possible signal of intrinsic value

Data source: SEC EDGAR Form 4, filed within 2 business days. Free.
Refresh: daily (Form 4s filed continuously)

Features:
  insider_buy_count    — number of insiders buying in last 90 days
  insider_buy_value    — total $ value of insider purchases
  insider_cluster      — 3+ insiders buying = cluster signal (strongest)
  insider_ceo_buying   — CEO specifically buying (strongest individual signal)
  insider_net_buying   — insider buys minus sells (net signal)
"""
import os, sys, json, re, time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

CACHE_DIR = '/Users/rick/ai_trading_bot_v2/cache_insider'
os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'quant-trading-bot admin@example.com'}
CIK_MAP_PATH = os.path.join(CACHE_DIR, 'cik_map.json')


def get_cik(symbol: str) -> str:
    """Get SEC CIK for a ticker symbol."""
    # Load cached map
    cik_map = {}
    if os.path.exists(CIK_MAP_PATH):
        with open(CIK_MAP_PATH) as f:
            cik_map = json.load(f)

    if symbol in cik_map:
        return cik_map[symbol]

    try:
        url  = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            tickers = resp.json()
            for entry in tickers.values():
                if entry.get('ticker', '').upper() == symbol.upper():
                    cik = str(entry['cik_str']).zfill(10)
                    cik_map[symbol] = cik
                    with open(CIK_MAP_PATH, 'w') as f:
                        json.dump(cik_map, f)
                    return cik
    except Exception:
        pass
    return ''


def fetch_form4_transactions(cik: str, symbol: str, lookback_days: int = 90) -> list:
    """
    Fetch Form 4 insider transactions for a company.
    Returns list of {date, name, title, type, shares, value, is_buy}
    """
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_form4.json")

    # Use cache if < 1 day old
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime < 86400:
            with open(cache_path) as f:
                return json.load(f)

    transactions = []
    try:
        url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []

        data    = resp.json()
        filings = data['filings']['recent']
        forms   = filings['form']
        dates   = filings['filingDate']
        accnums = filings['accessionNumber']

        cutoff = datetime.now() - timedelta(days=lookback_days)
        form4s = [
            (dates[i], accnums[i])
            for i, f in enumerate(forms)
            if f == '4' and datetime.strptime(dates[i], '%Y-%m-%d') >= cutoff
        ]

        for date_str, acc in form4s[:20]:  # limit to 20 most recent
            try:
                acc_nd   = acc.replace('-', '')
                cik_int  = int(cik)

                # Find raw XML via index page — skip xsl rendered version
                idx_url  = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nd}/{acc}-index.htm"
                idx_resp = requests.get(idx_url, headers=HEADERS, timeout=10)
                xml_url  = None
                if idx_resp.status_code == 200:
                    import re as _re
                    xml_links = _re.findall(r'href="(/Archives/edgar/data/[^"]*\.xml)"', idx_resp.text)
                    # Skip xsl rendered versions — use raw XML (no xsl in path)
                    raw_links = [l for l in xml_links if 'xsl' not in l.lower()]
                    if raw_links:
                        xml_url = f"https://www.sec.gov{raw_links[0]}"
                if not xml_url:
                    continue
                resp2    = requests.get(xml_url, headers=HEADERS, timeout=10)

                if resp2.status_code != 200:
                    continue

                root = ET.fromstring(resp2.content)

                # Get reporter info
                reporter  = root.find('.//reportingOwner')
                name      = ''
                title     = ''
                is_director = False
                is_officer  = False

                if reporter is not None:
                    name_el = reporter.find('.//rptOwnerName')
                    name    = name_el.text.strip() if name_el is not None else ''

                    rel     = reporter.find('.//reportingOwnerRelationship')
                    if rel is not None:
                        dir_el  = rel.find('isDirector')
                        off_el  = rel.find('isOfficer')
                        title_el= rel.find('officerTitle')
                        is_director = dir_el is not None and dir_el.text == '1'
                        is_officer  = off_el is not None and off_el.text == '1'
                        title = title_el.text.strip() if title_el is not None else ''

                # Get transactions
                for tx in root.findall('.//nonDerivativeTransaction'):
                    try:
                        # Correct path: transactionCode is under transactionCoding
                        code_el   = tx.find('.//transactionCoding/transactionCode')
                        # Shares and price nested under value tags
                        shares_el = tx.find('.//transactionAmounts/transactionShares/value')
                        price_el  = tx.find('.//transactionAmounts/transactionPricePerShare/value')
                        disp_el   = tx.find('.//transactionAmounts/transactionAcquiredDisposedCode/value')

                        if code_el is None or shares_el is None:
                            continue

                        code   = (code_el.text or '').strip()
                        shares = float((shares_el.text or '0').replace(',', ''))
                        price  = float((price_el.text or '0').replace(',', '')) if price_el is not None else 0.0
                        disp   = (disp_el.text or '').strip() if disp_el is not None else ''

                        # P = open market purchase (real buying signal)
                        # S = sale, F = tax withholding (not a sell signal)
                        # A = award/grant (not a buy signal)
                        # D = disposed
                        is_buy  = code == 'P'
                        is_sell = code == 'S' and disp == 'D'

                        if not (is_buy or is_sell):
                            continue

                        is_ceo = any(t in title.upper() for t in ['CEO','CHIEF EXECUTIVE','PRESIDENT'])

                        transactions.append({
                            'date':        date_str,
                            'name':        name,
                            'title':       title,
                            'code':        code,
                            'shares':      shares,
                            'price':       price,
                            'value':       shares * price,
                            'is_buy':      is_buy,
                            'is_sell':     is_sell,
                            'is_director': is_director,
                            'is_officer':  is_officer,
                            'is_ceo':      is_ceo,
                        })
                    except Exception:
                        continue

                time.sleep(0.1)

            except Exception:
                continue

    except Exception:
        pass

    with open(cache_path, 'w') as f:
        json.dump(transactions, f)

    return transactions


def compute_insider_features(symbol: str, lookback_days: int = 90) -> dict:
    """Compute insider flow features for ML ranker."""
    defaults = {
        'insider_buy_count':  0.0,
        'insider_buy_value':  0.0,
        'insider_cluster':    0.0,
        'insider_ceo_buying': 0.0,
        'insider_net_buying': 0.0,
    }

    cik = get_cik(symbol)
    if not cik:
        return defaults

    try:
        txns = fetch_form4_transactions(cik, symbol, lookback_days)
        if not txns:
            return defaults

        buys  = [t for t in txns if t['is_buy']]
        sells = [t for t in txns if t['is_sell']]

        buy_count  = len(buys)
        buy_value  = sum(t['value'] for t in buys)
        sell_value = sum(t['value'] for t in sells)

        # Unique buyers
        unique_buyers = len(set(t['name'] for t in buys))

        # Cluster: 3+ unique insiders buying
        cluster = float(unique_buyers >= 3)

        # CEO buying specifically
        ceo_buying = float(any(t['is_ceo'] for t in buys))

        # Net buying (buys - sells, normalized)
        net_value = buy_value - sell_value
        total     = buy_value + sell_value
        net_ratio = float(net_value / total) if total > 0 else 0.0

        return {
            'insider_buy_count':  float(np.clip(buy_count, 0, 20)),
            'insider_buy_value':  float(np.clip(buy_value / 1e6, 0, 10)),  # in $M
            'insider_cluster':    cluster,
            'insider_ceo_buying': ceo_buying,
            'insider_net_buying': float(np.clip(net_ratio, -1, 1)),
        }

    except Exception:
        return defaults


def scan_watchlist_insiders(watchlist: list, bear_only: bool = False) -> pd.DataFrame:
    """
    Scan full watchlist for insider buying clusters.
    bear_only: focus on BEAR regime where signal is strongest.
    """
    print(f"Scanning insider activity for {len(watchlist)} symbols...")
    rows = []

    for i, sym in enumerate(watchlist):
        try:
            feats = compute_insider_features(sym)
            feats['symbol'] = sym
            rows.append(feats)
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(watchlist)}", end='\r', flush=True)
            time.sleep(0.15)  # respect SEC rate limits
        except Exception:
            continue

    print()
    df = pd.DataFrame(rows)

    # Sort by cluster signal first, then buy value
    df = df.sort_values(['insider_cluster', 'insider_buy_value'], ascending=False)

    print(f"\n{'='*60}")
    print(f"TOP INSIDER BUYING CLUSTERS")
    print(f"{'='*60}")
    print(f"{'Symbol':<8} {'BuyCount':>9} {'BuyValue$M':>11} {'Cluster':>8} {'CEO':>5} {'NetBuy':>8}")
    print("-"*55)

    clusters = df[df['insider_cluster'] > 0]
    for _, r in clusters.head(15).iterrows():
        print(f"  {r['symbol']:<6} {r['insider_buy_count']:>9.0f} "
              f"{r['insider_buy_value']:>11.2f} "
              f"{'YES' if r['insider_cluster'] else 'no':>8} "
              f"{'YES' if r['insider_ceo_buying'] else 'no':>5} "
              f"{r['insider_net_buying']:>+8.1%}")

    print(f"\nTotal clusters found: {clusters['symbol'].count()}")
    print(f"CEO buying:           {(df['insider_ceo_buying'] > 0).sum()}")

    # Save for ML features
    records = df.to_dict('records')
    with open(os.path.join(CACHE_DIR, 'insider_signals.json'), 'w') as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved: insider_signals.json")

    return df


if __name__ == "__main__":
    import config

    # Test single symbol first
    print("Testing Form 4 fetch for NVDA...")
    feats = compute_insider_features('NVDA')
    print(f"NVDA insider features:")
    for k, v in feats.items():
        print(f"  {k}: {v}")

    print()

    # Scan first 50 symbols
    df = scan_watchlist_insiders(config.WATCHLIST[:50])

    # Show bear regime emphasis
    print(f"\n=== BEAR REGIME EMPHASIS ===")
    print("In BEAR regime, insider clusters are the strongest signal.")
    print("Insiders buying during market panic = intrinsic value floor.")
    print("Seyhun (1998): +18% 12-month alpha from insider clusters in bear markets.")
