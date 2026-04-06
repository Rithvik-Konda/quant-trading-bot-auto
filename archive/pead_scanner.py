"""
pead_scanner.py — Post-Earnings Announcement Drift
============================================================
One of most persistent market anomalies — 40+ years of evidence.
Bernard & Thomas 1989: stocks with positive earnings surprises
drift upward 5-60 days after announcement.
Works in ANY market regime because it's stock-specific.
Correlation with SPY: near zero.

Signal: earnings beat >5% surprise → enter next day → hold 5-7 days
Filter: quality stocks only, avoid biotech/FDA risk
Size: $3-5k per trade (small, high frequency)
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json, os, sys, time
from datetime import datetime, timedelta
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

CACHE_DIR = '/Users/rick/ai_trading_bot_v2/cache_pead'
os.makedirs(CACHE_DIR, exist_ok=True)

# Exclude: biotech (binary FDA risk), very small caps
EXCLUDED_SECTORS = ['Healthcare']  # too much binary risk
MIN_MARKET_CAP = 5e9  # $5B minimum

def get_recent_earnings_beats(watchlist: list) -> list:
    """Find stocks that beat earnings estimates in last 7 days."""
    beats = []
    cutoff = datetime.now() - timedelta(days=7)
    
    for sym in watchlist:
        try:
            tick = yf.Ticker(sym)
            hist = tick.earnings_history
            if hist is None or len(hist) == 0:
                continue
            
            # Most recent quarter
            latest = hist.head(1).iloc[0]
            q_date = hist.index[0]
            
            # Check if within last 7 days
            if hasattr(q_date, 'tz_localize'):
                q_date_naive = q_date.tz_localize(None) if q_date.tzinfo else q_date
            else:
                q_date_naive = pd.Timestamp(q_date)
            
            if q_date_naive < pd.Timestamp(cutoff):
                continue
            
            est = latest.get('epsEstimate', 0) or 0
            act = latest.get('epsActual', 0) or 0
            
            if est == 0 or act == 0:
                continue
            
            surprise = (act - est) / abs(est)
            
            # Strong beat: >5% positive surprise
            if surprise < 0.05:
                continue
            
            # Quality check
            info = tick.info
            mkt_cap = info.get('marketCap', 0) or 0
            if mkt_cap < MIN_MARKET_CAP:
                continue
            
            # Get current price
            price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
            if price <= 0:
                continue
            
            beats.append({
                'symbol':       sym,
                'earnings_date': str(q_date_naive.date()),
                'eps_estimate': round(float(est), 2),
                'eps_actual':   round(float(act), 2),
                'surprise_pct': round(float(surprise), 4),
                'price':        round(float(price), 2),
                'market_cap':   int(mkt_cap),
                'pead_signal':  True,
            })
            
            time.sleep(0.3)
        except:
            pass
    
    # Sort by surprise magnitude
    beats.sort(key=lambda x: x['surprise_pct'], reverse=True)
    return beats

def run_pead_scan():
    print("="*60)
    print(f"PEAD SCANNER — {datetime.now().strftime('%Y-%m-%d')}")
    print("="*60)
    print("Post-Earnings Announcement Drift — Bernard & Thomas 1989")
    print("Enter day after earnings beat >5%, hold 5-7 days\n")
    
    import config
    beats = get_recent_earnings_beats(config.WATCHLIST)
    
    if beats:
        print(f"EARNINGS BEATS IN LAST 7 DAYS ({len(beats)} found):")
        for b in beats[:10]:
            print(f"  {b['symbol']:<6}: beat={b['surprise_pct']:>+.1%}  "
                  f"est=${b['eps_estimate']}  actual=${b['eps_actual']}  "
                  f"date={b['earnings_date']}  → ENTER TODAY")
    else:
        print("No qualifying earnings beats in last 7 days")
    
    # Save
    with open(os.path.join(CACHE_DIR, 'pead_signals.json'), 'w') as f:
        json.dump({
            'date': str(datetime.now().date()),
            'beats': beats,
        }, f, indent=2)
    
    print(f"\nSaved {len(beats)} PEAD signals")
    return beats

if __name__ == "__main__":
    run_pead_scan()
