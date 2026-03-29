"""
institutional_flow.py — Dark pool volume + options flow signals
============================================================
Research:
  FibAlgo 2026: Dark pool prints >0.5% ADV = institutional positioning
  InsiderFinance: prints below VWAP = bullish (bought before price rose)
  Luxalgo 2025: stocks with unusual options activity 5x more likely to move
  
Two signals combined:
  1. Dark pool ratio: off-exchange volume / total volume
     High ratio + rising price = institutional accumulation
     High ratio + flat/falling price = distribution (AVOID)
     
  2. Options call volume ratio vs 20-day average
     Call/vol ratio >3x = smart money positioning for upside
     Combined with dark pool bullish = very high conviction

Data source: Polygon.io (already have API key)
  - /v2/aggs/ for total volume 
  - Off-exchange volume from trade conditions
  
Causal logic (not fitted):
  Institutions MUST use dark pools for large orders — market impact cost
  Their accumulation leaves volume footprints
  Options flow shows their directional conviction
  Both signals together = institutional alignment with our momentum signal
"""
import os, sys, json, time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

POLYGON_KEY = os.environ.get('POLYGON_API_KEY', '')
CACHE_DIR   = '/Users/rick/ai_trading_bot_v2/cache_flow'
os.makedirs(CACHE_DIR, exist_ok=True)


def get_options_call_volume(ticker: str, lookback_days: int = 5) -> dict:
    """
    Get recent call option volume vs historical average.
    Uses Polygon options data.
    Returns call_vol_ratio (current / 20d avg)
    """
    import requests
    
    try:
        # Get recent options activity
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
        params = {'apiKey': POLYGON_KEY, 'limit': 50}
        r = requests.get(url, params=params, timeout=10)
        
        if r.status_code != 200:
            return {'call_vol_ratio': 1.0, 'error': r.status_code}
        
        data = r.json().get('results', [])
        
        # Sum call volume for near-term contracts (0-30 DTE)
        call_vol = 0
        put_vol  = 0
        
        today = datetime.now().date()
        for opt in data:
            details = opt.get('details', {})
            exp = details.get('expiration_date', '')
            if not exp:
                continue
            
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            dte = (exp_date - today).days
            
            if 0 < dte <= 45:  # Near-term options
                day_vol = opt.get('day', {}).get('volume', 0) or 0
                if details.get('contract_type') == 'call':
                    call_vol += day_vol
                elif details.get('contract_type') == 'put':
                    put_vol  += day_vol
        
        total_options = call_vol + put_vol
        pcr = put_vol / call_vol if call_vol > 0 else 1.0  # put/call ratio
        
        # Load historical baseline from cache
        cache_path = os.path.join(CACHE_DIR, f'{ticker}_options_hist.json')
        if os.path.exists(cache_path):
            hist = json.load(open(cache_path))
            avg_call_vol = np.mean(hist.get('call_vols', [call_vol]) or [call_vol])
        else:
            avg_call_vol = call_vol  # no baseline yet
            hist = {'call_vols': []}
        
        # Update history
        hist['call_vols'].append(call_vol)
        hist['call_vols'] = hist['call_vols'][-21:]  # keep 21 days
        with open(cache_path, 'w') as f:
            json.dump(hist, f)
        
        call_vol_ratio = call_vol / avg_call_vol if avg_call_vol > 0 else 1.0
        
        return {
            'call_vol':       call_vol,
            'put_vol':        put_vol,
            'pcr':            round(pcr, 3),
            'call_vol_ratio': round(call_vol_ratio, 2),
            'unusual_calls':  call_vol_ratio > 3.0,
        }
    except Exception as e:
        return {'call_vol_ratio': 1.0, 'error': str(e)}


def get_dark_pool_signal(ticker: str, date_str: str = None) -> dict:
    """
    Estimate dark pool activity from Polygon trade data.
    Off-exchange trades = trades with conditions including dark pool codes.
    
    Dark pool condition codes (FINRA ATS):
      38 = Form T (after hours)
      41 = Block trade  
      53 = Intermarket sweep
    
    Returns dark_pool_ratio and bullish/bearish signal.
    """
    import requests
    
    if not date_str:
        date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        url = f"https://api.polygon.io/v3/trades/{ticker}"
        params = {
            'timestamp.gte': f"{date_str}T09:30:00Z",
            'timestamp.lte': f"{date_str}T16:00:00Z",
            'limit': 50000,
            'apiKey': POLYGON_KEY,
        }
        r = requests.get(url, params=params, timeout=15)
        
        if r.status_code != 200:
            return {'dark_pool_ratio': 0.0, 'error': r.status_code}
        
        trades = r.json().get('results', [])
        
        if not trades:
            return {'dark_pool_ratio': 0.0, 'reason': 'no trades'}
        
        total_vol  = 0
        dark_vol   = 0
        dark_above_vwap = 0
        dark_below_vwap = 0
        
        # Compute VWAP
        prices = [t.get('price', 0) for t in trades]
        vols   = [t.get('size', 0) for t in trades]
        vwap   = (sum(p*v for p,v in zip(prices, vols)) / 
                  sum(vols)) if sum(vols) > 0 else 0
        
        dark_pool_conditions = {38, 41, 53, 29, 32, 40}  # off-exchange codes
        
        for trade in trades:
            size       = trade.get('size', 0) or 0
            price      = trade.get('price', 0) or 0
            conditions = set(trade.get('conditions', []) or [])
            
            total_vol += size
            
            is_dark = bool(conditions & dark_pool_conditions)
            if not is_dark and len(conditions) == 0:
                is_dark = False  # exchange trade
            
            if is_dark:
                dark_vol += size
                if price < vwap:
                    dark_below_vwap += size  # bought below VWAP = bullish
                else:
                    dark_above_vwap += size  # sold above VWAP = bearish
        
        dark_ratio = dark_vol / total_vol if total_vol > 0 else 0
        
        # Bullish: more dark volume BELOW vwap (institutions buying cheaply)
        dark_bullish_ratio = dark_below_vwap / dark_vol if dark_vol > 0 else 0.5
        
        return {
            'dark_pool_ratio':   round(dark_ratio, 3),
            'dark_bullish_ratio': round(dark_bullish_ratio, 3),
            'vwap':              round(vwap, 2),
            'dark_vol':          dark_vol,
            'total_vol':         total_vol,
            'bullish':           dark_bullish_ratio > 0.6 and dark_ratio > 0.20,
            'bearish':           dark_bullish_ratio < 0.4 and dark_ratio > 0.20,
        }
    except Exception as e:
        return {'dark_pool_ratio': 0.0, 'error': str(e)}


def get_composite_flow_score(ticker: str) -> dict:
    """
    Combine dark pool + options flow into single institutional conviction score.
    Score 0-100: 
      >70 = strong institutional buy signal
      30-70 = neutral
      <30 = institutional distribution (avoid or short)
    """
    # Options signal
    opts = get_options_call_volume(ticker)
    call_ratio = opts.get('call_vol_ratio', 1.0)
    unusual    = opts.get('unusual_calls', False)
    
    # Dark pool signal
    dark = get_dark_pool_signal(ticker)
    dark_ratio   = dark.get('dark_pool_ratio', 0)
    dark_bullish = dark.get('dark_bullish_ratio', 0.5)
    
    # Composite score
    score = 50  # neutral baseline
    
    # Options component (+/- 25 points)
    if call_ratio > 5.0:   score += 25
    elif call_ratio > 3.0: score += 15
    elif call_ratio > 1.5: score += 5
    elif call_ratio < 0.5: score -= 10
    
    # Dark pool component (+/- 25 points)
    if dark_ratio > 0.30 and dark_bullish > 0.65:  score += 25
    elif dark_ratio > 0.20 and dark_bullish > 0.55: score += 10
    elif dark_ratio > 0.20 and dark_bullish < 0.40: score -= 15
    
    # PCR (put/call ratio) component
    pcr = opts.get('pcr', 1.0)
    if pcr < 0.5:    score += 10  # low put/call = bullish
    elif pcr > 2.0:  score -= 10  # high put/call = bearish hedge
    
    score = int(np.clip(score, 0, 100))
    
    signal = 'STRONG_BUY' if score >= 70 else \
             'BUY'        if score >= 60 else \
             'NEUTRAL'    if score >= 40 else \
             'AVOID'      if score >= 30 else 'SHORT_SIGNAL'
    
    return {
        'symbol':          ticker,
        'flow_score':      score,
        'signal':          signal,
        'call_vol_ratio':  round(call_ratio, 2),
        'unusual_calls':   unusual,
        'dark_pool_ratio': round(dark_ratio, 3),
        'dark_bullish':    round(dark_bullish, 3),
        'pcr':             opts.get('pcr', 1.0),
    }


if __name__ == "__main__":
    TEST = ['NVDA', 'APP', 'PLTR', 'MU', 'QCOM', 'ARM']
    print("INSTITUTIONAL FLOW SCANNER")
    print("="*55)
    print(f"{'Symbol':<8} {'Score':>6} {'Signal':<15} {'CallRatio':>10} {'DarkPool':>9} {'PCR':>6}")
    print("─"*55)
    
    for sym in TEST:
        try:
            result = get_composite_flow_score(sym)
            print(f"  {sym:<6}: {result['flow_score']:>5}  "
                  f"{result['signal']:<14}  "
                  f"{result['call_vol_ratio']:>9.1f}x  "
                  f"{result['dark_pool_ratio']:>8.1%}  "
                  f"{result.get('pcr',1.0):>5.2f}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  {sym}: error — {e}")
