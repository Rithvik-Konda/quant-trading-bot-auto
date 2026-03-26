"""
sector_earnings_cluster.py — Sector-level earnings beat momentum
Novel: if 3+ mid-caps in same sector beat same quarter → sector fundamental momentum
No paper has combined sector-level earnings cluster with individual stock momentum.
"""
import sys, json, os, pandas as pd
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
import config
from earnings_streak import _fetch as fetch_earnings

CACHE_DIR = "cache_earnings_streak"

def sector_beat_cluster_score(symbol, date, sector_map, watchlist):
    """
    Returns 0-1 score based on how many stocks in same sector
    beat earnings in the same recent quarter.
    """
    sector = sector_map.get(symbol)
    if not sector:
        return 0.0
    sector_peers = [s for s in watchlist if sector_map.get(s) == sector and s != symbol]
    beats_this_quarter = 0
    total_with_data = 0
    for peer in sector_peers[:20]:
        path = os.path.join(CACHE_DIR, f"{peer}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                records = json.load(f)
            if not records:
                continue
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            past = df[df['date'] < date].sort_values('date')
            if len(past) == 0:
                continue
            last = past.iloc[-1]
            total_with_data += 1
            if last.get('beat') is True:
                beats_this_quarter += 1
        except Exception:
            continue
    if total_with_data < 3:
        return 0.0
    beat_rate = beats_this_quarter / total_with_data
    if beat_rate >= 0.75:
        return 1.0
    elif beat_rate >= 0.60:
        return 0.75
    elif beat_rate >= 0.45:
        return 0.50
    else:
        return 0.0

if __name__ == "__main__":
    sector_map = {}
    for etf, syms in config.SECTOR_ETFS.items():
        for s in syms:
            sector_map[s] = etf
    today = pd.Timestamp.now().normalize()
    test_stocks = ['NVDA','MSFT','AVGO','INTU','APP','WING','AXON','JPM','LLY']
    print(f"Sector earnings cluster scores as of {today.date()}:")
    for s in test_stocks:
        score = sector_beat_cluster_score(s, today, sector_map, config.WATCHLIST)
        sector = sector_map.get(s, 'N/A')
        print(f"  {s:<8} {sector:<6} cluster_score={score:.2f}")
