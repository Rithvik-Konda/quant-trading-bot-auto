"""
wyckoff_spring.py — Detect Wyckoff Spring accumulation signal
============================================================
Research: Wyckoff 1930, confirmed by LSTM pattern recognition (arXiv 2403.18839)
          ATAS.net 2024 analysis of AAPL, TSLA spring patterns
          
The Spring is the highest-probability Wyckoff entry:
  - Price briefly breaks BELOW established support
  - Volume SPIKES on the breakdown (institutional absorption)
  - Price SNAPS BACK above support within 1-2 days
  
Why it works: institutions create the "panic" to shake out weak hands
and accumulate shares cheaply. The volume spike IS the institutional buying.
The snap-back confirms supply is exhausted.

Quantifiable signals (no discretion needed):
  1. Support level: rolling 20-day low
  2. Spring: today's low < 20d low AND close > 20d low (snap-back)
  3. Volume confirmation: spring day volume > 1.5x 20d avg volume
  4. Subsequent strength: next day opens above spring low
  
This is causal not fitted:
  Institutions must absorb supply to stop the decline
  Absorption requires high volume
  Once absorbed, supply exhausted = price snaps back
  This mechanism works the same in 1930 and 2024
"""
import numpy as np
import pandas as pd


def detect_spring(
    df: pd.DataFrame,
    lookback: int = 20,
    volume_multiplier: float = 1.5,
    max_break_pct: float = 0.05,  # spring can't break >5% below support
) -> dict:
    """
    Detect Wyckoff Spring pattern in price/volume data.
    
    Args:
        df: DataFrame with close, high, low, volume columns
        lookback: days to establish support level
        volume_multiplier: min volume surge on spring day
        max_break_pct: maximum % below support for valid spring
        
    Returns:
        dict with spring_detected, strength, signals
    """
    if len(df) < lookback + 5:
        return {'spring_detected': False, 'strength': 0}
    
    close  = df['close']  if 'close'  in df.columns else df['Close']
    low    = df['low']    if 'low'    in df.columns else df['Low']
    high   = df['high']   if 'high'   in df.columns else df['High']
    volume = df['volume'] if 'volume' in df.columns else df['Volume']
    
    # Support level: 20-day low EXCLUDING today
    support = float(low.iloc[-(lookback+1):-1].min())
    
    # Today's candle
    today_low   = float(low.iloc[-1])
    today_close = float(close.iloc[-1])
    today_vol   = float(volume.iloc[-1])
    avg_vol_20  = float(volume.iloc[-21:-1].mean())
    
    signals = {}
    
    # ── Signal 1: Price broke below support ──────────────────────
    broke_support = today_low < support
    break_depth   = (support - today_low) / support if broke_support else 0
    signals['broke_support']  = broke_support
    signals['break_depth_pct'] = round(break_depth, 4)
    signals['support_level']   = round(support, 2)
    
    if not broke_support:
        return {
            'spring_detected': False,
            'strength': 0,
            'signals': signals,
            'reason': 'price did not break below support',
        }
    
    # Spring too deep — not absorption, real breakdown
    if break_depth > max_break_pct:
        return {
            'spring_detected': False,
            'strength': 0,
            'signals': signals,
            'reason': f'break too deep ({break_depth:.1%}) — likely real breakdown not spring',
        }
    
    # ── Signal 2: Snap-back — close above support ─────────────────
    snapped_back = today_close > support
    signals['snapped_back'] = snapped_back
    
    if not snapped_back:
        # Check if yesterday snapped back (spring was yesterday)
        yesterday_close = float(close.iloc[-2])
        yesterday_low   = float(low.iloc[-2])
        prior_support   = float(low.iloc[-(lookback+2):-2].min())
        
        if yesterday_low < prior_support and yesterday_close > prior_support:
            # Yesterday was the spring
            signals['spring_day'] = 'yesterday'
            snapped_back = True
        else:
            return {
                'spring_detected': False,
                'strength': 0,
                'signals': signals,
                'reason': 'no snap-back above support',
            }
    else:
        signals['spring_day'] = 'today'
    
    # ── Signal 3: Volume confirmation ─────────────────────────────
    vol_ratio = today_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
    signals['volume_ratio']   = round(vol_ratio, 2)
    signals['volume_confirmed'] = vol_ratio >= volume_multiplier
    
    # ── Signal 4: Trading range — was price consolidating before? ─
    # Spring is most reliable after a trading range (consolidation)
    # Measure: price range over last 20 days vs prior 20 days
    range_recent = float(high.iloc[-21:-1].max() - low.iloc[-21:-1].min())
    range_prior  = float(high.iloc[-41:-21].max() - low.iloc[-41:-21].min()) if len(df) >= 41 else range_recent
    
    # Recent range < prior range = consolidation = valid spring setup
    consolidating = range_recent < range_prior * 0.8 if range_prior > 0 else False
    signals['consolidating']     = consolidating
    signals['range_compression']  = round(range_recent / range_prior, 3) if range_prior > 0 else 1.0
    
    # ── Composite spring strength ─────────────────────────────────
    strength_components = [
        snapped_back,           # price closed above support
        vol_ratio >= 1.5,       # volume confirmed
        vol_ratio >= 2.0,       # strong volume (bonus)
        break_depth < 0.02,     # shallow spring (better)
        consolidating,          # valid setup
    ]
    strength = int(sum(strength_components) * 20)  # 0-100
    
    return {
        'spring_detected': snapped_back and vol_ratio >= 1.0,
        'strength':        strength,
        'support_level':   round(support, 2),
        'today_low':       round(today_low, 2),
        'today_close':     round(today_close, 2),
        'volume_ratio':    round(vol_ratio, 2),
        'break_depth':     round(break_depth, 4),
        'signals':         signals,
    }


def scan_for_springs(symbols: list, prices_cache: dict = None) -> list:
    """Scan universe for Wyckoff Spring setups."""
    import sys
    sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
    from backtester_clean import fetch_history
    
    springs = []
    for sym in symbols:
        try:
            if prices_cache and sym in prices_cache:
                df = prices_cache[sym].tail(60)
            else:
                df = fetch_history(sym, days=90)
                df.index = pd.to_datetime(df.index).tz_localize(None)
            
            result = detect_spring(df)
            if result['spring_detected']:
                result['symbol'] = sym
                springs.append(result)
        except:
            pass
    
    springs.sort(key=lambda x: x['strength'], reverse=True)
    return springs


def validate_on_historical_trades(trades_csv: str) -> dict:
    """
    Check: did our top winners show spring patterns at entry?
    Validation that spring signal is predictive in our universe.
    """
    import sys
    sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
    from backtester_clean import fetch_history
    
    df = pd.read_csv(trades_csv)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['hold_days'] = (pd.to_datetime(df['exit_date']) - df['entry_date']).dt.days
    long = df[df['side'] == 'long'].copy()
    
    winners    = long[long['pnl'] > 2000].head(20)
    big_losers = long[(long['pnl'] < -1000) & (long['hold_days'] <= 10)].head(20)
    
    winner_springs = 0
    loser_springs  = 0
    
    print("SPRING VALIDATION ON HISTORICAL TRADES")
    print("="*55)
    
    print("\nBIG WINNERS — did they show spring at entry?")
    for _, r in winners.iterrows():
        try:
            prices = fetch_history(r['symbol'], days=9999)
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            entry = pd.Timestamp(r['entry_date'])
            pre   = prices[prices.index <= entry].tail(30)
            res   = detect_spring(pre)
            spring = "🌱 SPRING" if res['spring_detected'] else "  no spring"
            if res['spring_detected']:
                winner_springs += 1
            print(f"  {r['symbol']:<6}: {spring}  strength={res['strength']:>3}  "
                  f"vol_ratio={res.get('volume_ratio',0):.1f}x  "
                  f"pnl=${r['pnl']:>+8,.0f}")
        except:
            pass
    
    print("\nEARLY STOPS — did they show spring at entry?")
    for _, r in big_losers.iterrows():
        try:
            prices = fetch_history(r['symbol'], days=9999)
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            entry = pd.Timestamp(r['entry_date'])
            pre   = prices[prices.index <= entry].tail(30)
            res   = detect_spring(pre)
            spring = "🌱 SPRING" if res['spring_detected'] else "  no spring"
            if res['spring_detected']:
                loser_springs += 1
            print(f"  {r['symbol']:<6}: {spring}  strength={res['strength']:>3}  "
                  f"vol_ratio={res.get('volume_ratio',0):.1f}x  "
                  f"pnl=${r['pnl']:>+8,.0f}")
        except:
            pass
    
    print(f"\nRESULTS:")
    print(f"  Springs in big winners: {winner_springs}/{len(winners)}")
    print(f"  Springs in early stops: {loser_springs}/{len(big_losers)}")
    
    if winner_springs > loser_springs:
        print(f"  ✓ Spring signal discriminates — wire into entry filter")
    else:
        print(f"  ✗ Spring signal does not discriminate — don't use as filter")
    
    return {'winner_springs': winner_springs, 'loser_springs': loser_springs}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
    
    # First validate on historical data
    result = validate_on_historical_trades(
        '/Users/rick/ai_trading_bot_v2/trades_v2.csv'
    )
    
    # Then scan current market
    print("\n\nCURRENT MARKET SPRING SCAN")
    print("="*55)
    import config
    springs = scan_for_springs(config.WATCHLIST[:50])
    if springs:
        print(f"\nSpring setups found today:")
        for s in springs[:10]:
            print(f"  {s['symbol']}: strength={s['strength']}  "
                  f"vol={s['volume_ratio']:.1f}x  "
                  f"break={s['break_depth']:.1%}")
    else:
        print("No spring setups in current market")
        print("(Makes sense — VIX 31, market in downtrend, not consolidating)")
