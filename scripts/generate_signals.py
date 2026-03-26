"""
generate_signals.py — Daily candidate generator for live trading
Run after morning_update.py. Scores all 447 stocks with the ML ensemble.
Output: signals_today.csv with full ranked list + regime-filtered candidates
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
        print(f"Regime: {regime}")
    else:
        print("WARNING: run morning_update.py first")
        regime = "CHOPPY"

    import config
    from backtester_clean import fetch_history
    from strategy_core import load_ranker_ensemble
    from ml_model import compute_features

    # Load ML ensemble
    rankers = load_ranker_ensemble()
    feat_cols = sorted(set(
        list(rankers[3]["features"]) +
        list(rankers[5]["features"]) +
        list(rankers[7]["features"])
    ))

    # Regime thresholds
    ml_min = {"TRENDING_BULL": 0.85, "CHOPPY": 0.93, "BEAR": 0.95}.get(regime, 0.85)

    today = pd.Timestamp.now().normalize()
    results = []

    print(f"Scoring {len(config.WATCHLIST)} stocks...")
    for i, sym in enumerate(config.WATCHLIST):
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(config.WATCHLIST)}", end="\r", flush=True)
        try:
            df = fetch_history(sym, days=400)
            if df is None or len(df) < 260:
                continue
            feat = compute_features(df, symbol=sym)
            if feat is None or len(feat) == 0:
                continue

            # Get most recent feature row
            row = feat.iloc[-1]

            # Score with each horizon model, average
            scores = []
            for horizon in [3, 5, 7]:
                bundle = rankers[horizon]
                model  = bundle["model"]
                scaler = bundle["scaler"]
                cols   = bundle["features"]
                X = row.reindex(cols).fillna(0).values.reshape(1, -1)
                X_scaled = scaler.transform(X)
                scores.append(float(model.predict(X_scaled)[0]))

            ml_score = float(np.mean(scores))

            # Get current price and vol
            px = float(df['close'].iloc[-1])
            vol60 = float(df['close'].pct_change().dropna().tail(60).std() * (252**0.5))

            results.append({
                'symbol':     sym,
                'ml_score':   round(ml_score, 4),
                'px':         round(px, 2),
                'vol60':      round(vol60, 3),
                'candidate':  ml_score >= ml_min and vol60 <= 0.70,
                'regime':     regime,
                'date':       today.date(),
            })
        except Exception as e:
            pass

    print(f"\nScored {len(results)} symbols")

    df_out = pd.DataFrame(results).sort_values('ml_score', ascending=False).reset_index(drop=True)
    df_out['rank'] = range(1, len(df_out)+1)

    # Save full list
    out_path = '/Users/rick/ai_trading_bot_v2/signals_today.csv'
    df_out.to_csv(out_path, index=False)

    # Print top candidates for today's regime
    candidates = df_out[df_out['candidate'] == True]
    print(f"\n=== TOP CANDIDATES ({regime}, ML≥{ml_min}, vol≤0.70) ===")
    print(candidates.head(15)[['symbol','ml_score','px','vol60']].to_string())
    print(f"\nTotal candidates: {len(candidates)}")
    print(f"Saved: {out_path}")
    return df_out

if __name__ == "__main__":
    generate()
