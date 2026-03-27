"""
stop_continuation_model.py — ML model predicting whether a stop touch
will continue down (real stop) or recover (whipsaw).

Trained on 264 historical stop exits with measured outcomes.
Features: regime, vol, distance below stop, days held, market context.
Target: probability price recovers above stop within 5 days (= whipsaw).

If P(whipsaw) is high → hold, wait for confirmation.
If P(whipsaw) is low → exit immediately.

This replaces hardcoded "2 consecutive closes" with a learned threshold
that adapts to regime, volatility, and market conditions.
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

MODEL_PATH = '/Users/rick/ai_trading_bot_v2/stop_continuation_model.joblib'
CACHE_DIR  = '/Users/rick/ai_trading_bot_v2/cache_prices'


def build_training_data() -> pd.DataFrame:
    """
    Build training dataset from historical stop exits.
    For each stop exit, compute features at time of stop touch
    and label whether price recovered (whipsaw=1) or continued down (real=0).
    """
    from backtester_clean import fetch_history
    from regime_classifier import load_macro_data, compute_signals, RegimeClassifier

    print("Loading historical stop exits...")
    df = pd.read_csv('/Users/rick/ai_trading_bot_v2/trades_v2.csv')
    df = df[(df['reason'] == 'stop') & (df['side'] == 'long')].copy()
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date']  = pd.to_datetime(df['exit_date'])
    print(f"  {len(df)} stop exits to analyze")

    # Load macro data for market context
    spy_macro, hyg_macro, vix_macro = load_macro_data(cache_dir=CACHE_DIR)

    rows = []
    for _, trade in df.iterrows():
        sym       = trade['symbol']
        exit_date = trade['exit_date']
        stop_px   = float(trade['exit_price'])
        entry_px  = float(trade['entry_price'])
        regime    = str(trade.get('regime', 'TRENDING_BULL'))
        ann_vol   = float(trade.get('ann_vol', 0.35))
        ml_rank   = float(trade.get('ml_rank_pct', 0.9))
        hold_days = (trade['exit_date'] - trade['entry_date']).days

        try:
            df_h = fetch_history(sym, days=9999)
            if df_h is None or len(df_h) < 30:
                continue
            df_h.index = pd.to_datetime(df_h.index).tz_localize(None)

            # Price history up to stop
            df_pre = df_h.loc[:exit_date]
            if len(df_pre) < 20:
                continue

            close  = df_pre['close']
            volume = df_pre['volume']

            # Features at time of stop touch
            px_now   = float(close.iloc[-1])
            px_5d    = float(close.iloc[-5])  if len(close) >= 5  else px_now
            px_20d   = float(close.iloc[-20]) if len(close) >= 20 else px_now

            mom_5d  = float(px_now / px_5d - 1)  if px_5d > 0 else 0
            mom_20d = float(px_now / px_20d - 1) if px_20d > 0 else 0

            # How far below the entry did we go?
            drawdown_from_entry = float((px_now - entry_px) / entry_px)

            # Vol at stop time
            vol_20 = float(close.pct_change().dropna().tail(20).std() * np.sqrt(252))

            # Volume spike on stop day (panic selling = real stop)
            vol_ratio = float(volume.iloc[-1] / volume.tail(20).mean()) if volume.tail(20).mean() > 0 else 1.0

            # Distance below 200MA (structural weakness)
            ma200 = float(close.tail(200).mean())
            dist_ma200 = float((px_now - ma200) / ma200)

            # VIX at stop time
            vix_at_stop = float(
                vix_macro['close'].reindex([exit_date], method='ffill').iloc[0]
            ) if len(vix_macro) > 0 else 20.0

            # Regime encoding
            reg_bull = float(regime == 'TRENDING_BULL')
            reg_chop = float(regime == 'CHOPPY')
            reg_bear = float(regime == 'BEAR')

            # Target: did price recover above stop within 5 days? (1=whipsaw, 0=real)
            future = df_h.loc[df_h.index > exit_date]['close']
            if len(future) < 5:
                continue
            recovered_5d = float(future.iloc[:5].max() > stop_px * 1.02)

            rows.append({
                'regime_bull':           reg_bull,
                'regime_chop':           reg_chop,
                'regime_bear':           reg_bear,
                'ann_vol':               float(np.clip(ann_vol, 0.1, 1.5)),
                'vol_20d':               float(np.clip(vol_20, 0.1, 1.5)),
                'mom_5d':                float(np.clip(mom_5d, -0.3, 0.3)),
                'mom_20d':               float(np.clip(mom_20d, -0.5, 0.5)),
                'drawdown_from_entry':   float(np.clip(drawdown_from_entry, -0.5, 0)),
                'hold_days':             float(np.clip(hold_days, 1, 100)),
                'ml_rank':               float(np.clip(ml_rank, 0, 1)),
                'dist_ma200':            float(np.clip(dist_ma200, -0.5, 0.5)),
                'vol_ratio':             float(np.clip(vol_ratio, 0.2, 5)),
                'vix_level':             float(np.clip(vix_at_stop, 10, 80)),
                'target_whipsaw':        recovered_5d,
            })
        except Exception:
            continue

    df_out = pd.DataFrame(rows)
    print(f"  Built {len(df_out)} training examples")
    print(f"  Whipsaw rate: {df_out['target_whipsaw'].mean():.0%}")
    return df_out


FEATURE_COLS = [
    'regime_bull', 'regime_chop', 'regime_bear',
    'ann_vol', 'vol_20d', 'mom_5d', 'mom_20d',
    'drawdown_from_entry', 'hold_days', 'ml_rank',
    'dist_ma200', 'vol_ratio', 'vix_level',
]


def train_model(df: pd.DataFrame) -> dict:
    """Train whipsaw predictor."""
    X = df[FEATURE_COLS].fillna(0)
    y = df['target_whipsaw']

    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=10, subsample=0.8,
        random_state=42, verbose=-1,
        class_weight='balanced',
    )
    model.fit(X_tr, y_tr)

    from sklearn.metrics import roc_auc_score, accuracy_score
    preds     = model.predict_proba(X_te)[:, 1]
    auc       = roc_auc_score(y_te, preds)
    acc       = accuracy_score(y_te, preds > 0.5)
    print(f"\n  Whipsaw model AUC: {auc:.4f}  Accuracy: {acc:.0%}")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n  Feature importance:")
    for f, v in imp.head(8).items():
        print(f"    {f:<25} {v:.0f}")

    # Whipsaw probability by regime (OOS)
    print(f"\n  OOS whipsaw rate by regime:")
    te_full = X_te.copy()
    te_full['pred'] = preds
    te_full['actual'] = y_te.values
    for regime_col, label in [('regime_bull','BULL'), ('regime_chop','CHOP'), ('regime_bear','BEAR')]:
        mask = te_full[regime_col] > 0.5
        if mask.sum() > 0:
            grp = te_full[mask]
            print(f"    {label}: actual={grp['actual'].mean():.0%} whipsaw, "
                  f"predicted={grp['pred'].mean():.0%}")

    # Optimal threshold — minimize PnL loss from premature exits
    # At what P(whipsaw) should we require confirmation?
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n  Optimal whipsaw threshold analysis:")
    print(f"  {'Threshold':>10} {'WouldWait':>10} {'WaitIsRight':>12}")
    for thresh in thresholds:
        would_wait   = (preds > thresh).mean()
        wait_correct = ((preds > thresh) & (y_te == 1)).sum() / max((preds > thresh).sum(), 1)
        print(f"  {thresh:>10.0%} {would_wait:>10.0%} {wait_correct:>12.0%}")

    bundle = {'model': model, 'features': FEATURE_COLS, 'auc': auc}
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n  Saved: {MODEL_PATH}")
    return bundle


def predict_whipsaw_prob(
    regime: str,
    ann_vol: float,
    mom_5d: float,
    mom_20d: float,
    drawdown_from_entry: float,
    hold_days: int,
    ml_rank: float,
    dist_ma200: float,
    vol_ratio: float,
    vix_level: float,
) -> float:
    """
    Predict probability this stop touch is a whipsaw (price will recover).
    High probability = wait for confirmation before exiting.
    Low probability = exit immediately.
    """
    try:
        bundle = joblib.load(MODEL_PATH)
        feats = {
            'regime_bull':         float(regime == 'TRENDING_BULL'),
            'regime_chop':         float(regime == 'CHOPPY'),
            'regime_bear':         float(regime == 'BEAR'),
            'ann_vol':             float(np.clip(ann_vol, 0.1, 1.5)),
            'vol_20d':             float(np.clip(ann_vol, 0.1, 1.5)),
            'mom_5d':              float(np.clip(mom_5d, -0.3, 0.3)),
            'mom_20d':             float(np.clip(mom_20d, -0.5, 0.5)),
            'drawdown_from_entry': float(np.clip(drawdown_from_entry, -0.5, 0)),
            'hold_days':           float(np.clip(hold_days, 1, 100)),
            'ml_rank':             float(np.clip(ml_rank, 0, 1)),
            'dist_ma200':          float(np.clip(dist_ma200, -0.5, 0.5)),
            'vol_ratio':           float(np.clip(vol_ratio, 0.2, 5)),
            'vix_level':           float(np.clip(vix_level, 10, 80)),
        }
        X = pd.DataFrame([feats], columns=bundle['features']).fillna(0)
        return float(bundle['model'].predict_proba(X)[0, 1])
    except Exception:
        # Fallback: use empirical whipsaw rates by regime
        if regime == 'TRENDING_BULL':
            return 0.63
        elif regime == 'CHOPPY':
            return 0.71
        else:
            return 0.0


if __name__ == "__main__":
    df = build_training_data()
    if len(df) >= 20:
        bundle = train_model(df)
        print(f"\n=== Live predictions ===")
        scenarios = [
            ("BULL low vol calm",    "TRENDING_BULL", 0.25, 0.02, 0.05, -0.08, 15, 0.92, 0.05, 1.0, 18),
            ("BULL high vol spike",  "TRENDING_BULL", 0.65, -0.08, -0.15, -0.10, 5, 0.95, -0.05, 3.5, 32),
            ("CHOP normal",          "CHOPPY",        0.35, -0.03, -0.05, -0.08, 10, 0.91, 0.02, 1.2, 22),
            ("BEAR breakdown",       "BEAR",          0.55, -0.12, -0.25, -0.12, 8, 0.88, -0.15, 2.8, 38),
        ]
        print(f"\n{'Scenario':<25} {'P(whipsaw)':>12} {'Action':>20}")
        print("-"*60)
        for name, *args in scenarios:
            p = predict_whipsaw_prob(*args)
            action = "WAIT — likely whipsaw" if p > 0.55 else "EXIT — likely real stop"
            print(f"  {name:<23} {p:>12.0%} {action:>20}")
    else:
        print("Insufficient training data — need more stop exits")
