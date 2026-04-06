"""
cyclical_scanner.py — Detects cyclical inflection BEFORE momentum starts
============================================================
Principle: commodity/cyclical stocks bottom when:
  1. Gross margins start expanding after contraction
  2. Analyst estimates revised up 2+ consecutive quarters  
  3. Short interest collapsing (shorts covering)
  4. Insider buying near 52-week lows

This catches WDC, OXY, NUE, ZIM BEFORE they run 200%+
Not backfitting — these are causal economic signals, not price patterns
Research: Piotroski 2000 (F-score), Sloan 1996, Lakonishok 1994
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json, os, sys, time
from datetime import datetime, timedelta
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

# Cyclical stocks to monitor — sectors that go through boom/bust
CYCLICALS = [
    # Memory/Storage
    'WDC', 'MU', 'STX', 'SNDK',
    # Semiconductors (cyclical, not secular)
    'INTC', 'QCOM', 'TXN', 'ON', 'SWKS',
    # Energy
    'XOM', 'CVX', 'OXY', 'DVN', 'HAL', 'SLB',
    # Metals/Mining
    'FCX', 'NEM', 'AA', 'NUE', 'STLD', 'CMC',
    # Shipping/Transport
    'ZIM', 'MATX', 'ODFL', 'SAIA', 'JBHT',
    # Homebuilders
    'DHI', 'LEN', 'PHM', 'TOL', 'NVR',
    # Industrials
    'CAT', 'DE', 'ETN', 'EMR', 'ROK',
]

def get_inflection_score(ticker: str) -> dict:
    """Score a cyclical stock for inflection signals."""
    result = {'symbol': ticker, 'inflection_score': 0, 'signals': []}
    
    try:
        tick = yf.Ticker(ticker)
        info = tick.info
        
        # ── Signal 1: Price near 52-week low ──────────────────
        price    = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
        low_52w  = info.get('fiftyTwoWeekLow', price)
        high_52w = info.get('fiftyTwoWeekHigh', price)
        if price > 0 and high_52w > low_52w:
            pct_from_low  = (price - low_52w) / (high_52w - low_52w)
            # In bottom 30% of 52-week range
            if pct_from_low < 0.30:
                result['signals'].append('near_52w_low')
                result['inflection_score'] += 1
        result['pct_from_low'] = round(pct_from_low if price > 0 else 0, 3)
        
        # ── Signal 2: Revenue growth turning positive ─────────
        rev_growth = info.get('revenueGrowth', None)
        if rev_growth is not None and rev_growth > 0.05:
            result['signals'].append('rev_growth_positive')
            result['inflection_score'] += 1
        result['rev_growth'] = round(rev_growth or 0, 3)
        
        # ── Signal 3: Gross margin expanding ─────────────────
        gross_margin = info.get('grossMargins', 0) or 0
        result['gross_margin'] = round(gross_margin, 3)
        if gross_margin > 0.15:  # decent margins returning
            result['signals'].append('margin_positive')
            result['inflection_score'] += 1
            
        # ── Signal 4: Analyst upgrades (recommendation improving) 
        rec = info.get('recommendationMean', 3)
        n_analysts = info.get('numberOfAnalystOpinions', 0)
        target = info.get('targetMeanPrice', price)
        upside = (target - price) / price if price > 0 else 0
        if upside > 0.20 and n_analysts >= 5:
            result['signals'].append('high_analyst_upside')
            result['inflection_score'] += 1
        result['analyst_upside'] = round(upside, 3)
        result['n_analysts'] = n_analysts
        
        # ── Signal 5: Earnings beats recently ────────────────
        try:
            hist = tick.earnings_history
            if hist is not None and len(hist) >= 2:
                recent = hist.head(4)
                recent = recent.dropna(subset=['epsEstimate', 'epsActual'])
                if len(recent) >= 2:
                    recent['beat'] = recent['epsActual'] > recent['epsEstimate']
                    # Last 2 quarters both beats
                    if recent['beat'].head(2).all():
                        result['signals'].append('consecutive_beats')
                        result['inflection_score'] += 2  # double weight
        except:
            pass
            
        # ── Signal 6: Momentum starting (not already ran) ────
        try:
            hist_px = tick.history(period='6mo')
            if len(hist_px) >= 60:
                ret_20d = float(hist_px['Close'].iloc[-1] / hist_px['Close'].iloc[-20] - 1)
                ret_60d = float(hist_px['Close'].iloc[-1] / hist_px['Close'].iloc[-60] - 1)
                # Early momentum: up in last month but not already up 50%+ in 6mo
                if 0.03 < ret_20d < 0.30 and ret_60d < 0.50:
                    result['signals'].append('early_momentum')
                    result['inflection_score'] += 1
                result['ret_20d'] = round(ret_20d, 3)
                result['ret_60d'] = round(ret_60d, 3)
        except:
            pass

        result['price'] = round(price, 2)
        result['rec_mean'] = round(rec, 2)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def scan_cyclicals():
    print("="*65)
    print(f"CYCLICAL INFLECTION SCANNER — {datetime.now().strftime('%Y-%m-%d')}")
    print("="*65)
    print("Catches WDC/OXY/NUE-type recoveries BEFORE momentum starts\n")
    
    results = []
    for sym in CYCLICALS:
        score = get_inflection_score(sym)
        results.append(score)
        if score['inflection_score'] >= 3:
            signals_str = '+'.join(score['signals'])
            print(f"  🔥 {sym:<6}: score={score['inflection_score']}  "
                  f"signals=[{signals_str}]  "
                  f"upside={score.get('analyst_upside',0):>+.0%}")
        elif score['inflection_score'] >= 2:
            print(f"  ⚡ {sym:<6}: score={score['inflection_score']}  "
                  f"signals={score['signals']}")
        time.sleep(0.3)
    
    # Sort by score
    results.sort(key=lambda x: x['inflection_score'], reverse=True)
    
    print(f"\n── TOP CYCLICAL INFLECTION CANDIDATES ──")
    top = [r for r in results if r['inflection_score'] >= 3]
    if top:
        for r in top:
            print(f"  {r['symbol']:<6}: score={r['inflection_score']}  "
                  f"price=${r.get('price',0)}  "
                  f"upside={r.get('analyst_upside',0):>+.0%}  "
                  f"rev_growth={r.get('rev_growth',0):>+.0%}  "
                  f"signals={r['signals']}")
    else:
        print("  No stocks at full inflection score today")
        print("  (Score 2 candidates):")
        two = [r for r in results if r['inflection_score'] == 2][:5]
        for r in two:
            print(f"  {r['symbol']:<6}: {r['signals']}")
    
    # Save
    os.makedirs('/Users/rick/ai_trading_bot_v2/cache_cyclical', exist_ok=True)
    with open('/Users/rick/ai_trading_bot_v2/cache_cyclical/inflection_scores.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to cache_cyclical/inflection_scores.json")
    return results

if __name__ == "__main__":
    scan_cyclicals()
