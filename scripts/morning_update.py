"""
morning_update.py — Run at 9:00am ET before market open
Refreshes regime and writes to live_regime.txt for live trader.
Cron: 0 9 * * 1-5 /opt/homebrew/bin/python3.11 ~/ai_trading_bot_v2/scripts/morning_update.py
"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/ai_trading_bot_v2'))
sys.path.insert(0, os.path.expanduser('~/ai_trading_bot_v2/v2'))
import pandas as pd
from datetime import datetime
from regime_classifier import RegimeClassifier, compute_signals, load_macro_data

cache_dir = os.path.expanduser('~/ai_trading_bot_v2/cache_prices')
spy_macro, hyg_macro, vix_macro = load_macro_data(cache_dir=cache_dir)
clf = RegimeClassifier()
regime = "CHOPPY"
for date in spy_macro.index[-60:]:
    signals = compute_signals(spy_macro, hyg_macro, vix_macro, as_of_date=date)
    if signals:
        regime = clf.update(date, signals)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Regime: {regime}")
with open(os.path.expanduser('~/ai_trading_bot_v2/live_regime.txt'), 'w') as f:
    f.write(f"{regime}\n{pd.Timestamp.now().date()}\n")
print("Done")
