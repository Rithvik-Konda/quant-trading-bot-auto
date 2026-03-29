"""
accumulation_signal.py — Institutional accumulation detection
============================================================
Based on Wyckoff Method (1930s) + modern academic validation:
  Grinblatt & Titman 1989 — institutional momentum predicts returns
  Sias & Starks 1997 — institutional trading drives momentum
  
Principle: detect whether institutions are accumulating (buying)
or distributing (selling) a stock BEFORE we enter.

We want to enter AFTER accumulation is confirmed — in early markup.
We want to AVOID entering during distribution masquerading as rally.

Signals:
  1. OBV trend — rising OBV = net institutional buying
  2. Volume quality — up-day volume > down-day volume ratio
  3. Price structure — higher lows = uptrend confirmed
  4. Effort vs result — volume expanding with price = genuine demand
  
NOT fitted to backtest — these are causal institutional behavior signals
that work across all markets and time periods (Wyckoff 1930 → today)
"""
import numpy as np
import pandas as pd


def compute_accumulation_score(df: pd.DataFrame) -> dict:
    """
    Compute institutional accumulation score for a stock.
    
    Args:
        df: DataFrame with columns: close, high, low, volume
        
    Returns:
        dict with score 0-100 and component signals
    """
    if len(df) < 30:
        return {'score': 50, 'reason': 'insufficient data'}
    
    close  = df['close'] if 'close' in df.columns else df['Close']
    volume = df['volume'] if 'volume' in df.columns else df['Volume']
    high   = df['high']   if 'high'  in df.columns else df['High']
    low    = df['low']    if 'low'   in df.columns else df['Low']
    
    signals = {}
    
    # ── Signal 1: OBV Trend ────────────────────────────────────
    # On-Balance Volume: accumulates volume on up days, subtracts on down days
    # Rising OBV = institutions buying more than selling
    price_change = close.diff()
    obv = (np.sign(price_change) * volume).fillna(0).cumsum()
    
    # OBV trend over last 20 days vs 60 days
    obv_20d_slope = float(np.polyfit(range(20), obv.iloc[-20:].values, 1)[0])
    obv_60d_slope = float(np.polyfit(range(min(60,len(obv))), 
                                      obv.iloc[-60:].values, 1)[0])
    
    # Normalize by average volume
    avg_vol = float(volume.tail(20).mean())
    obv_20d_norm = obv_20d_slope / avg_vol if avg_vol > 0 else 0
    obv_60d_norm = obv_60d_slope / avg_vol if avg_vol > 0 else 0
    
    signals['obv_rising'] = obv_20d_norm > 0 and obv_60d_norm > 0
    signals['obv_score']  = min(100, max(0, 50 + obv_20d_norm * 500))
    
    # ── Signal 2: Volume Quality ───────────────────────────────
    # Up-day volume vs down-day volume ratio
    # Institutions accumulating: more volume on up days
    last_20 = df.tail(20)
    up_days   = last_20[last_20['close' if 'close' in last_20.columns else 'Close']
                        .diff() > 0]
    down_days = last_20[last_20['close' if 'close' in last_20.columns else 'Close']
                        .diff() < 0]
    
    up_vol   = float(up_days['volume' if 'volume' in up_days.columns else 'Volume'].mean()) if len(up_days) > 0 else avg_vol
    down_vol = float(down_days['volume' if 'volume' in down_days.columns else 'Volume'].mean()) if len(down_days) > 0 else avg_vol
    
    vol_ratio = up_vol / down_vol if down_vol > 0 else 1.0
    signals['vol_quality']       = vol_ratio > 1.1
    signals['up_down_vol_ratio'] = round(vol_ratio, 2)
    
    # ── Signal 3: Price Structure — Higher Lows ───────────────
    # Uptrend confirmed: each pullback stops higher than previous
    # Measure: last 3 local lows are rising
    rolling_low = low.rolling(5).min()
    recent_lows = rolling_low.iloc[-30::5].values  # sample every 5 days
    higher_lows = all(recent_lows[i] <= recent_lows[i+1] 
                      for i in range(len(recent_lows)-1))
    signals['higher_lows'] = higher_lows
    
    # ── Signal 4: Effort vs Result ─────────────────────────────
    # Wyckoff: volume (effort) should match price movement (result)
    # High volume + price progress = genuine demand
    # High volume + no price progress = distribution
    last_10 = df.tail(10)
    c = last_10['close'] if 'close' in last_10.columns else last_10['Close']
    v = last_10['volume'] if 'volume' in last_10.columns else last_10['Volume']
    
    price_change_10d = float(c.iloc[-1] / c.iloc[0] - 1) if float(c.iloc[0]) > 0 else 0
    vol_vs_avg       = float(v.mean()) / avg_vol if avg_vol > 0 else 1.0
    
    # Good: price up AND volume up vs average
    # Bad: price flat/down AND volume up (distribution)
    effort_result_ok = price_change_10d > 0.01 and vol_vs_avg > 0.9
    signals['effort_result'] = effort_result_ok
    signals['price_10d']     = round(price_change_10d, 4)
    signals['vol_vs_avg']    = round(vol_vs_avg, 2)
    
    # ── Signal 5: Accumulation vs Distribution volume ──────────
    # Chaikin Money Flow: measures buying/selling pressure
    # Positive CMF = accumulation, Negative = distribution
    money_flow_mult = ((close - low) - (high - close)) / (high - low).clip(lower=0.001)
    money_flow_vol  = money_flow_mult * volume
    cmf_20 = float(money_flow_vol.tail(20).sum() / volume.tail(20).sum())
    
    signals['cmf_20']         = round(cmf_20, 4)
    signals['accumulating']   = cmf_20 > 0.05  # positive CMF = buying pressure
    
    # ── Composite Score ────────────────────────────────────────
    # Each signal contributes equally — no curve fitting
    score_components = [
        signals['obv_rising'],      # 20 points
        signals['vol_quality'],      # 20 points
        signals['higher_lows'],      # 20 points
        signals['effort_result'],    # 20 points
        signals['accumulating'],     # 20 points
    ]
    
    composite = int(sum(score_components) * 20)
    
    # Bonus: strong OBV trend
    if obv_20d_norm > 0.5:
        composite = min(100, composite + 10)
    
    signals['composite_score'] = composite
    signals['obv_20d_norm']    = round(obv_20d_norm, 4)
    
    return signals


def is_accumulation_confirmed(df: pd.DataFrame, min_score: int = 60) -> bool:
    """
    Simple boolean: is institutional accumulation confirmed?
    Use this as entry filter — only enter if True.
    
    min_score=60 means 3/5 signals must be positive.
    Principled threshold — majority of signals must agree.
    """
    signals = compute_accumulation_score(df)
    return signals.get('composite_score', 0) >= min_score


if __name__ == "__main__":
    # Test on current market conditions
    import yfinance as yf
    import sys
    sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
    from backtester_clean import fetch_history
    
    TEST_STOCKS = ['NVDA', 'APP', 'PLTR', 'SMCI', 'VRT',  # recent winners
                   'SHOP', 'AVAV', 'LLY',                  # mixed
                   'INTC', 'AAPL', 'MSFT']                  # control
    
    print("ACCUMULATION SIGNAL — CURRENT MARKET SCAN")
    print("="*60)
    print(f"{'Symbol':<8} {'Score':>6} {'OBV':>5} {'VolQ':>5} {'HiLo':>5} "
          f"{'Eff':>5} {'CMF':>6} {'Accum':>7}")
    print("─"*60)
    
    for sym in TEST_STOCKS:
        try:
            prices = fetch_history(sym, days=200)
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            sig = compute_accumulation_score(prices)
            
            score  = sig.get('composite_score', 0)
            flag   = "✓ BUY" if score >= 60 else "✗ skip"
            
            print(f"  {sym:<6}: {score:>5}  "
                  f"{'✓' if sig.get('obv_rising') else '✗':>5}  "
                  f"{'✓' if sig.get('vol_quality') else '✗':>5}  "
                  f"{'✓' if sig.get('higher_lows') else '✗':>5}  "
                  f"{'✓' if sig.get('effort_result') else '✗':>5}  "
                  f"{sig.get('cmf_20',0):>+6.3f}  "
                  f"{flag}")
        except Exception as e:
            print(f"  {sym:<6}: error — {e}")
