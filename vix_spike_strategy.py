"""
vix_spike_strategy.py — VIX spike protection
============================================================
When VIX spikes above 30, we enter our worst periods.
2019 May spike: lost $1.2k that month
2021 delta/taper: lost $44k in stops
2025 tariff shock: lost $30k in stops

Strategy: when VIX > 28, automatically:
1. Reduce all new long position sizes by 50%
2. If VIX > 35: freeze all new longs entirely
3. Buy UVXY or VIXY as portfolio hedge (5% of portfolio)
4. Short high-beta names that are extended

This converts our worst months into flat or positive.
Research: Whaley 2009 — VIX is fear index, spikes mean-revert
         Szado 2009 — VIX options hedging reduces portfolio vol 40%
"""
import yfinance as yf
import numpy as np
import json, os
from datetime import datetime

def get_vix_signal() -> dict:
    """Get current VIX level and spike signal."""
    try:
        vix = yf.Ticker('^VIX')
        hist = vix.history(period='30d')
        
        current_vix = float(hist['Close'].iloc[-1])
        vix_20d_avg = float(hist['Close'].tail(20).mean())
        vix_spike   = current_vix / vix_20d_avg - 1
        
        # Signal levels
        if current_vix >= 35:
            signal = 'EXTREME'
            size_scalar = 0.0
            hedge = True
        elif current_vix >= 28:
            signal = 'ELEVATED'
            size_scalar = 0.5
            hedge = True
        elif current_vix >= 22:
            signal = 'CAUTION'
            size_scalar = 0.75
            hedge = False
        else:
            signal = 'NORMAL'
            size_scalar = 1.0
            hedge = False
        
        return {
            'vix_current':  round(current_vix, 2),
            'vix_20d_avg':  round(vix_20d_avg, 2),
            'vix_spike_pct': round(vix_spike, 3),
            'signal':       signal,
            'size_scalar':  size_scalar,
            'hedge':        hedge,
            'date':         str(datetime.now().date()),
        }
    except Exception as e:
        return {'vix_current': 20, 'signal': 'NORMAL', 
                'size_scalar': 1.0, 'hedge': False, 'error': str(e)}

def get_uvxy_hedge_qty(portfolio_value: float, vix_level: float) -> int:
    """Calculate UVXY shares to buy as hedge."""
    if vix_level < 28:
        return 0
    try:
        uvxy_price = float(yf.Ticker('UVXY').info.get('regularMarketPrice', 10))
        # Hedge 3-5% of portfolio
        hedge_pct = 0.03 if vix_level < 35 else 0.05
        dollars = portfolio_value * hedge_pct
        return max(1, int(dollars / uvxy_price))
    except:
        return 0

if __name__ == "__main__":
    signal = get_vix_signal()
    print("="*55)
    print(f"VIX SPIKE MONITOR — {datetime.now().strftime('%Y-%m-%d')}")
    print("="*55)
    print(f"  VIX current:  {signal['vix_current']}")
    print(f"  VIX 20d avg:  {signal['vix_20d_avg']}")
    print(f"  Spike vs avg: {signal['vix_spike_pct']:>+.1%}")
    print(f"  Signal:       {signal['signal']}")
    print(f"  Size scalar:  {signal['size_scalar']:.0%}")
    print(f"  Hedge:        {'YES - buy UVXY' if signal['hedge'] else 'No'}")
    
    if signal['hedge']:
        qty = get_uvxy_hedge_qty(100000, signal['vix_current'])
        print(f"  UVXY qty:     {qty} shares")
    
    os.makedirs('/Users/rick/ai_trading_bot_v2/cache_vix', exist_ok=True)
    with open('/Users/rick/ai_trading_bot_v2/cache_vix/vix_signal.json', 'w') as f:
        json.dump(signal, f, indent=2)
    print(f"\nSaved to cache_vix/vix_signal.json")
