"""
stop_width_model.py — ML model to predict optimal stop width per position

Trained on synthetic entries across full price history (up to 64 years).
Features: vol, proper ATR, ADX, price vs 200MA, momentum, beta
No hardcoded multipliers — model learns what works from data.
"""
import os, sys, json, traceback
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

CACHE_DIR = '/Users/rick/ai_trading_bot_v2/cache_prices'
MODEL_PATH = '/Users/rick/ai_trading_bot_v2/stop_width_ranker.joblib'
FEATURE_COLS = ['ann_vol', 'true_atr', 'adx', 'price_vs_200ma', 'mom_60', 'vol_regime', 'rsi_14']


def compute_true_atr_pct(df, period=14):
    """Proper ATR as % of price using True Range (handles gaps)."""
    if len(df) < period + 1:
        return 0.025
    high  = df['high']
    low   = df['low']
    close = df['close']
    prev  = close.shift(1)
    tr = pd.concat([
        (high - low) / close,
        (high - prev).abs() / close,
        (low  - prev).abs() / close,
    ], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if not np.isnan(val) else 0.025


def compute_adx(df, period=14):
    """ADX trend strength indicator."""
    if len(df) < period * 2:
        return pd.Series([20.0] * len(df), index=df.index)
    high  = df['high']
    low   = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr   = tr.rolling(period).mean()
    up    = high.diff()
    down  = -low.diff()
    plus  = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    plus_di  = 100 * plus.rolling(period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus.rolling(period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def compute_rsi(close, period=14):
    delta = close.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    val   = rsi.iloc[-1] if len(rsi) else 50.0
    return float(val) if not np.isnan(val) else 50.0


def fetch_max_history(sym):
    """Load max-history pickle, fall back to standard fetch."""
    path = os.path.join(CACHE_DIR, f"{sym}_max.pkl")
    if os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except Exception:
            pass
    from backtester_clean import fetch_history
    return fetch_history(sym, days=9000)


def extract_features(df_pre):
    """Extract stop-width features from price history up to entry."""
    if len(df_pre) < 20 or 'high' not in df_pre.columns:
        return None
    try:
        ann_vol    = float(df_pre['close'].pct_change().dropna().tail(60).std() * (252**0.5))
        true_atr   = compute_true_atr_pct(df_pre, period=14)
        adx_val    = float(compute_adx(df_pre).iloc[-1])
        adx_val    = adx_val if not np.isnan(adx_val) else 20.0
        ma200      = df_pre['close'].rolling(200).mean().iloc[-1] if len(df_pre) >= 200 else df_pre['close'].iloc[-1]
        p200       = float(df_pre['close'].iloc[-1] / ma200 - 1) if ma200 > 0 else 0.0
        mom60      = float(df_pre['close'].iloc[-1] / df_pre['close'].iloc[-61] - 1) if len(df_pre) >= 61 else 0.0
        vol5       = df_pre['close'].pct_change().dropna().tail(5).std()
        vol60      = df_pre['close'].pct_change().dropna().tail(60).std()
        vol_regime = float(vol5 / vol60) if vol60 > 0 else 1.0
        rsi        = compute_rsi(df_pre['close'])
        return {
            'ann_vol':        float(np.clip(ann_vol, 0, 3)),
            'true_atr':       float(np.clip(true_atr, 0, 0.15)),
            'adx':            float(np.clip(adx_val, 0, 100)),
            'price_vs_200ma': float(np.clip(p200, -0.5, 1.0)),
            'mom_60':         float(np.clip(mom60, -0.8, 3.0)),
            'vol_regime':     float(np.clip(vol_regime, 0, 5)),
            'rsi_14':         float(np.clip(rsi, 0, 100)),
        }
    except Exception:
        return None


def build_training_data():
    """Build (features, optimal_stop_width) pairs from full price history."""
    import config
    rows = []

    print(f"Building synthetic training data from {len(config.WATCHLIST)} symbols...")
    sym_ok = 0
    for i, sym in enumerate(config.WATCHLIST):
        try:
            df = fetch_max_history(sym)
            if df is None or len(df) < 300:
                continue
            if 'high' not in df.columns or 'low' not in df.columns:
                continue

            # Sample entry points every 15 trading days
            for idx in range(260, len(df) - 31, 15):
                df_pre   = df.iloc[:idx]
                entry_px = float(df.iloc[idx]['close'])
                df_hold  = df.iloc[idx:idx+30]

                if len(df_hold) < 16:
                    continue

                feats = extract_features(df_pre)
                if feats is None:
                    continue

                max_dd = float((entry_px - df_hold['low'].min()) / entry_px)
                max_dd = max(max_dd, 0.001)

                fwd_15 = float(df_hold['close'].iloc[14] / entry_px - 1)

                # Optimal stop:
                # Recovered after 15 days → stop was too tight → widen it
                # Kept falling → stop at max_dd was appropriate
                if fwd_15 > 0.05:
                    optimal = float(np.clip(max_dd * 1.2, 0.03, 0.28))
                elif fwd_15 < -0.05:
                    optimal = float(np.clip(max_dd * 0.85, 0.03, 0.28))
                else:
                    optimal = float(np.clip(max_dd * 1.05, 0.03, 0.28))

                row = feats.copy()
                row['optimal_stop'] = optimal
                row['fwd_15']       = fwd_15
                rows.append(row)

            sym_ok += 1
            if sym_ok % 50 == 0:
                print(f"  {sym_ok}/{len(config.WATCHLIST)} symbols, {len(rows)} rows", end='\r', flush=True)

        except Exception:
            continue

    print(f"\nDone: {sym_ok} symbols, {len(rows)} training rows")
    return pd.DataFrame(rows)


def train_stop_model(df):
    """Train LightGBM on (features → optimal_stop_width)."""
    X = df[FEATURE_COLS].fillna(0).values
    y = df['optimal_stop'].clip(0.03, 0.28).values

    # Walk-forward split: train on first 80%, test on last 20%
    split  = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=20,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae   = float(np.abs(preds - y_te).mean())
    print(f"OOS MAE: {mae:.4f} ({mae*100:.1f}%)")
    print(f"Pred range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"True range: [{y_te.min():.3f}, {y_te.max():.3f}]")

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importance:")
    for f, v in imp.items():
        print(f"  {f:<20} {v:.0f}")

    # Retrain on all data
    model.fit(X, y)
    joblib.dump({'model': model, 'features': FEATURE_COLS}, MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH}")
    return model


def predict_stop_width(symbol, df_pre, days_since_earnings=60, earnings_streak=0.5):
    """Predict optimal stop width. Falls back to 2x true ATR if model unavailable."""
    try:
        bundle = joblib.load(MODEL_PATH)
        feats  = extract_features(df_pre)
        if feats is None:
            raise ValueError("feature extraction failed")
        X = pd.DataFrame([feats], columns=bundle['features']).fillna(0)
        pred = float(bundle['model'].predict(X)[0])
        return float(np.clip(pred, 0.04, 0.25))
    except Exception:
        atr = compute_true_atr_pct(df_pre)
        return float(np.clip(2.0 * atr, 0.04, 0.20))


if __name__ == "__main__":
    df = build_training_data()

    if len(df) < 200:
        print(f"ERROR: only {len(df)} rows — check price data")
        sys.exit(1)

    print(f"\nOptimal stop distribution:\n{df['optimal_stop'].describe().round(3)}")
    print(f"Recovery rate (fwd_15>5%): {(df['fwd_15']>0.05).mean():.1%}")

    model = train_stop_model(df)

    # Validate on 2025 worst trades
    print("\n=== Stop width predictions for 2025 worst trades ===")
    cases = [
        ('MP',   '2025-06-10'),
        ('RKLB', '2025-11-06'),
        ('VRT',  '2025-07-08'),
        ('NU',   '2025-02-06'),
        ('AVGO', '2025-02-20'),
        ('COIN', '2024-11-13'),
    ]
    for sym, entry in cases:
        df_h = fetch_max_history(sym)
        if df_h is not None:
            df_pre = df_h.loc[:entry]
            if len(df_pre) >= 20:
                ml   = predict_stop_width(sym, df_pre)
                atr  = compute_true_atr_pct(df_pre) * 2.0
                print(f"  {sym} {entry}: ML={ml:.2%}  2xTrueATR={atr:.2%}")
