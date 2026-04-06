"""
pairs_trader.py — Market-neutral pairs trading
============================================================
Research: Gatev, Goetzmann, Rouwenhorst 1999 — pairs trading
         generates 11% annual excess return, market-neutral

Our validated pairs (from pairs_cointegration.json):
  V/MA:       p=0.003, half-life=29d  ← TRADE NOW (z=+2.62)
  DDOG/ZS:    p=0.005, half-life=37d
  XOM/VLO:    cointegrated
  LOW/MCD:    cointegrated
  TSLA/GM:    cointegrated

Signal: when spread > 2 std devs → long cheap, short expensive
Exit: when spread reverts to mean (z < 0.5)
Size: $5k per leg ($10k total per pair)
Correlation with SPY: near zero — pure arbitrage
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json, os, sys
from datetime import datetime
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

CACHE_DIR = '/Users/rick/ai_trading_bot_v2/cache_pairs'
os.makedirs(CACHE_DIR, exist_ok=True)

PAIRS = [
    {'sym1': 'V',    'sym2': 'MA',   'half_life': 29},
    {'sym1': 'DDOG', 'sym2': 'ZS',   'half_life': 37},
    {'sym1': 'XOM',  'sym2': 'VLO',  'half_life': 45},
    {'sym1': 'LOW',  'sym2': 'MCD',  'half_life': 60},
]

def compute_spread(sym1: str, sym2: str, lookback: int = 120) -> dict:
    """Compute current z-score of pairs spread."""
    try:
        p1 = yf.Ticker(sym1).history(period=f'{lookback+10}d')['Close']
        p2 = yf.Ticker(sym2).history(period=f'{lookback+10}d')['Close']
        
        p1.index = p1.index.tz_localize(None)
        p2.index = p2.index.tz_localize(None)
        
        combined = pd.DataFrame({'p1': p1, 'p2': p2}).dropna()
        if len(combined) < 60:
            return {'error': 'insufficient data'}
        
        # OLS hedge ratio
        from numpy.linalg import lstsq
        X = np.column_stack([np.ones(len(combined)), combined['p2'].values])
        coeffs, _, _, _ = lstsq(X, combined['p1'].values, rcond=None)
        hedge_ratio = coeffs[1]
        
        # Spread
        spread = combined['p1'] - hedge_ratio * combined['p2']
        
        # Z-score
        spread_mean = spread.rolling(60).mean()
        spread_std  = spread.rolling(60).std()
        z_score = float((spread.iloc[-1] - spread_mean.iloc[-1]) / spread_std.iloc[-1])
        
        # Current prices
        price1 = float(combined['p1'].iloc[-1])
        price2 = float(combined['p2'].iloc[-1])
        
        # Signal
        if z_score > 2.0:
            signal = f'SHORT {sym1} / LONG {sym2}'
            action = 'trade'
        elif z_score < -2.0:
            signal = f'LONG {sym1} / SHORT {sym2}'
            action = 'trade'
        elif abs(z_score) < 0.5:
            signal = 'CLOSE — spread reverted'
            action = 'close'
        else:
            signal = 'NEUTRAL — wait'
            action = 'wait'
        
        return {
            'sym1':        sym1,
            'sym2':        sym2,
            'z_score':     round(z_score, 3),
            'hedge_ratio': round(hedge_ratio, 4),
            'spread':      round(float(spread.iloc[-1]), 4),
            'price1':      round(price1, 2),
            'price2':      round(price2, 2),
            'signal':      signal,
            'action':      action,
        }
    except Exception as e:
        return {'sym1': sym1, 'sym2': sym2, 'error': str(e)}

def scan_pairs():
    print("="*60)
    print(f"PAIRS TRADING SCANNER — {datetime.now().strftime('%Y-%m-%d')}")
    print("="*60)
    print("Market-neutral arbitrage — ~zero SPY correlation\n")
    
    results = []
    for pair in PAIRS:
        r = compute_spread(pair['sym1'], pair['sym2'])
        results.append(r)
        if 'error' in r:
            print(f"  {r['sym1']}/{r['sym2']}: error")
            continue
        
        flag = "🔥 TRADE" if r['action'] == 'trade' else \
               "⚡ CLOSE" if r['action'] == 'close' else "  wait"
        print(f"  {r['sym1']}/{r['sym2']:<8}: z={r['z_score']:>+6.2f}  "
              f"hedge={r['hedge_ratio']:.3f}  {flag}")
        if r['action'] == 'trade':
            print(f"    → {r['signal']}")
            print(f"    → {r['sym1']}=${r['price1']}  {r['sym2']}=${r['price2']}")
    
    # Save
    with open(os.path.join(CACHE_DIR, 'signals.json'), 'w') as f:
        json.dump({'date': str(datetime.now().date()), 'pairs': results}, f, indent=2)
    
    tradeable = [r for r in results if r.get('action') == 'trade']
    print(f"\nTradeable pairs today: {len(tradeable)}")
    return results

if __name__ == "__main__":
    scan_pairs()
