"""
morning_update.py — Run at 9:00am ET before market open
Refreshes regime, writes live_regime.txt, fires options on regime flip.
Cron: 0 9 * * 1-5 /opt/homebrew/bin/python3.11 ~/ai_trading_bot_v2/scripts/morning_update.py
"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/ai_trading_bot_v2'))
sys.path.insert(0, os.path.expanduser('~/ai_trading_bot_v2/v2'))
import pandas as pd
from datetime import datetime
from regime_classifier import RegimeClassifier, compute_signals, load_macro_data

REGIME_FILE = os.path.expanduser('~/ai_trading_bot_v2/live_regime.txt')

# Load previous regime
prev_regime = None
if os.path.exists(REGIME_FILE):
    with open(REGIME_FILE) as f:
        lines = f.read().strip().split('\n')
        prev_regime = lines[0] if lines else None

# Compute new regime
cache_dir = os.path.expanduser('~/ai_trading_bot_v2/cache_prices')
spy_macro, hyg_macro, vix_macro = load_macro_data(cache_dir=cache_dir)
clf = RegimeClassifier()
regime = "CHOPPY"
for date in spy_macro.index[-60:]:
    signals = compute_signals(spy_macro, hyg_macro, vix_macro, as_of_date=date)
    if signals:
        regime = clf.update(date, signals)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Regime: {regime} (prev: {prev_regime})")

# Write new regime
with open(REGIME_FILE, 'w') as f:
    f.write(f"{regime}\n{pd.Timestamp.now().date()}\n")

# Fire options on BEAR → TRENDING_BULL flip
if prev_regime == "BEAR" and regime == "TRENDING_BULL":
    print(f"[options] REGIME FLIP detected: BEAR → TRENDING_BULL — buying SPY calls")
    try:
        from strategy_options import regime_flip_call
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            os.environ.get("ALPACA_API_KEY"),
            os.environ.get("ALPACA_SECRET_KEY"),
            paper=True
        )
        acct = client.get_account()
        portfolio_value = float(acct.equity)
        regime_flip_call(portfolio_value=portfolio_value, capital_pct=0.03, days_out=21)
    except Exception as e:
        print(f"[options] Error: {e}")
else:
    print(f"[options] No regime flip — no options action")

print("Done")
