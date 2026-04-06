"""
new_listing_scanner.py — Finds IPOs/spinoffs showing early relative strength
============================================================
Principle: stocks with strong RS vs sector in first 90 days of trading
often continue outperforming. No ML needed — pure relative strength.
Research: Ritter 1991, Gompers/Lerner 2003 — IPO momentum persists 3-12 months
Not fitted: RS rank is computed fresh daily vs current universe
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json, os, sys, time
from datetime import datetime, timedelta
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

# New listings to monitor — add any IPO/spinoff after 90 days of trading
# Update this list monthly
NEW_LISTINGS = [
    'SNDK',   # SanDisk spinoff Feb 2024 — memory storage
    'KVYO',   # Klaviyo IPO Sep 2023 — marketing automation
    'ARM',    # ARM Holdings IPO Sep 2023 — chip architecture
    'BIRK',   # Birkenstock IPO Oct 2023 — consumer brand
    'CART',   # Instacart IPO Sep 2023 — grocery delivery
    'CAVA',   # Cava IPO Jun 2023 — restaurant
    'RDDT',   # Reddit IPO Mar 2024 — social media
    'ALAB',   # Astera Labs IPO Mar 2024 — AI connectivity
    'RKLB',   # Rocket Lab — space
    'ASTS',   # AST SpaceMobile — satellite
]

# Sector ETFs for RS comparison
SECTOR_ETFS = {
    'tech':       'XLK',
    'consumer':   'XLY',
    'industrial': 'XLI',
    'energy':     'XLE',
    'healthcare': 'XLV',
    'financial':  'XLF',
}

def compute_rs_score(ticker: str, days: int = 90) -> dict:
    """Compute relative strength vs market and sector."""
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=f'{days+30}d')
        if len(hist) < 60:
            return {'symbol': ticker, 'rs_score': 0, 'tradeable': False,
                    'reason': f'insufficient history ({len(hist)} days)'}

        # Price returns
        ret_20d  = float(hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1)
        ret_60d  = float(hist['Close'].iloc[-1] / hist['Close'].iloc[-60] - 1)
        ret_90d  = float(hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)

        # Volume trend — institutional accumulation proxy
        vol_20d_avg = float(hist['Volume'].tail(20).mean())
        vol_60d_avg = float(hist['Volume'].tail(60).mean())
        vol_trend   = vol_20d_avg / vol_60d_avg if vol_60d_avg > 0 else 1.0

        # SPY benchmark
        spy = yf.Ticker('SPY').history(period=f'{days+30}d')
        spy_ret_60d = float(spy['Close'].iloc[-1] / spy['Close'].iloc[-60] - 1)

        # RS vs market
        rs_vs_spy = ret_60d - spy_ret_60d

        # Composite RS score
        rs_score = (
            ret_20d  * 0.30 +
            ret_60d  * 0.40 +
            ret_90d  * 0.20 +
            (vol_trend - 1) * 0.10
        )

        # Tradeable: >90 days history, positive RS vs SPY, volume expanding
        tradeable = (
            len(hist) >= 90 and
            rs_vs_spy > 0.05 and
            vol_trend > 1.0 and
            ret_20d > 0
        )

        return {
            'symbol':     ticker,
            'rs_score':   round(rs_score, 4),
            'ret_20d':    round(ret_20d, 4),
            'ret_60d':    round(ret_60d, 4),
            'ret_90d':    round(ret_90d, 4),
            'rs_vs_spy':  round(rs_vs_spy, 4),
            'vol_trend':  round(vol_trend, 3),
            'tradeable':  tradeable,
            'days_listed': len(hist),
            'price':      round(float(hist['Close'].iloc[-1]), 2),
        }
    except Exception as e:
        return {'symbol': ticker, 'rs_score': 0, 'tradeable': False, 'reason': str(e)}

def scan_new_listings():
    print("="*60)
    print(f"NEW LISTING MOMENTUM SCANNER — {datetime.now().strftime('%Y-%m-%d')}")
    print("="*60)
    print("Principle: IPO/spinoff relative strength in first 90-180 days")
    print()

    results = []
    for ticker in NEW_LISTINGS:
        rs = compute_rs_score(ticker)
        results.append(rs)
        status = "✓ TRADEABLE" if rs.get('tradeable') else "✗ skip"
        print(f"  {ticker:<8}: rs={rs.get('rs_score',0):>+6.3f}  "
              f"20d={rs.get('ret_20d',0):>+5.1%}  "
              f"60d={rs.get('ret_60d',0):>+5.1%}  "
              f"vol_trend={rs.get('vol_trend',1):.2f}  {status}")
        time.sleep(0.5)

    # Rank by RS score
    tradeable = [r for r in results if r.get('tradeable')]
    tradeable.sort(key=lambda x: x['rs_score'], reverse=True)

    print(f"\n── TRADEABLE NEW LISTINGS ──")
    if tradeable:
        for r in tradeable:
            print(f"  {r['symbol']}: rs_score={r['rs_score']:>+.3f}  "
                  f"price=${r['price']}  days_listed={r['days_listed']}")
    else:
        print("  None meet criteria today")

    # Save
    os.makedirs('/Users/rick/ai_trading_bot_v2/cache_newlistings', exist_ok=True)
    with open('/Users/rick/ai_trading_bot_v2/cache_newlistings/rs_scores.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to cache_newlistings/rs_scores.json")
    return results

if __name__ == "__main__":
    scan_new_listings()
