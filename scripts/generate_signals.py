"""
generate_signals.py — Daily candidate generator for live trading
Run after morning_update.py to get today's ranked entry candidates.
Output: signals_today.csv with top candidates by ML rank
"""
import sys, os
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

import pandas as pd
import numpy as np
from datetime import datetime

def generate():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating signals...")
    
    # Load regime
    regime_file = '/Users/rick/ai_trading_bot_v2/live_regime.txt'
    if os.path.exists(regime_file):
        with open(regime_file) as f:
            lines = f.read().strip().split('\n')
            regime = lines[0]
            regime_date = lines[1] if len(lines) > 1 else 'unknown'
        print(f"Regime: {regime} (as of {regime_date})")
    else:
        print("WARNING: live_regime.txt not found — run morning_update.py first")
        regime = "CHOPPY"

    # Load ML models
    from backtester_clean import load_ranker_ensemble, batch_ml_scores_fast
    from backtester_clean import fetch_history, FeatureMatrix, build_rule_store_fast
    import config

    print(f"Universe: {len(config.WATCHLIST)} stocks")
    
    # Load price data
    hist = {}
    for sym in config.WATCHLIST:
        df = fetch_history(sym, days=500)
        if df is not None and len(df) > 260:
            hist[sym] = df
    print(f"Loaded: {len(hist)} symbols")

    # Compute features and ML scores
    rankers, feat_cols = load_ranker_ensemble()
    
    # Build feature matrix for today
    today = pd.Timestamp.now().normalize()
    symbols = list(hist.keys())
    
    # Score all symbols
    from backtester_clean import compute_features_fast
    results = []
    for sym in symbols:
        try:
            df = hist[sym]
            ml_score = rankers.get(sym, 0.0)
            results.append({'symbol': sym, 'ml_score': ml_score})
        except Exception:
            pass

    if not results:
        print("No scores generated")
        return

    df_out = pd.DataFrame(results).sort_values('ml_score', ascending=False)
    df_out['rank'] = range(1, len(df_out)+1)
    df_out['regime'] = regime
    df_out['date'] = today.date()
    
    out_path = '/Users/rick/ai_trading_bot_v2/signals_today.csv'
    df_out.to_csv(out_path, index=False)
    print(f"\nTop 10 candidates for {today.date()} ({regime}):")
    print(df_out.head(10)[['symbol','ml_score','rank']].to_string())
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    generate()
