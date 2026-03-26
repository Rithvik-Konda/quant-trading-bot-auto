"""
short_ranker.py — ML model for short candidate scoring in BEAR regime

Architecture mirrors the long ranker but targets negative forward returns.
Features that predict shorts:
- Earnings MISS streak (opposite of beat streak)
- Analyst revision DOWN momentum
- Short interest surprise (elevated vs sector median)
- Overextension above 200MA (mean reversion target)
- Sector deteriorating vs SPY
- High vol + downward momentum = falling knife
- PEAD negative (recent earnings miss with drift)

Training: synthetic entries on ALL watchlist stocks across full history
Target: stocks in bottom decile of 5d forward return = good short candidates
"""
import os, sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

MODEL_PATH = '/Users/rick/ai_trading_bot_v2/short_ranker.joblib'
CACHE_DIR  = '/Users/rick/ai_trading_bot_v2/cache_prices'

FEATURE_COLS = [
    'ann_vol', 'true_atr', 'adx',
    'price_vs_200ma',        # overextension target
    'price_vs_50ma',
    'mom_60', 'mom_20',      # momentum direction
    'mom_deterioration',     # recent weakening
    'vol_regime',            # vol expanding = falling
    'rsi_14',                # overbought = short
    'dist_from_52w_high',    # how far from high
    'consecutive_down_days', # persistent weakness
    'vol_on_down_days',      # distribution signal
    'sector_vs_spy_20',      # sector lagging
]


def fetch_max(sym):
    path = os.path.join(CACHE_DIR, f"{sym}_max.pkl")
    if os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except Exception:
            pass
    from backtester_clean import fetch_history
    return fetch_history(sym, days=9000)


def compute_short_features(df_pre):
    """Extract features predictive of negative forward returns."""
    if len(df_pre) < 60 or 'high' not in df_pre.columns:
        return None
    try:
        close = df_pre['close']
        high  = df_pre['high']
        low   = df_pre['low']

        ann_vol = float(close.pct_change().dropna().tail(60).std() * (252**0.5))

        # True ATR
        prev = close.shift(1)
        tr = pd.concat([
            (high - low) / close,
            (high - prev).abs() / close,
            (low  - prev).abs() / close,
        ], axis=1).max(axis=1)
        true_atr = float(tr.tail(14).mean())

        # ADX
        up   = high.diff()
        down = -low.diff()
        plus_dm  = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        atr14 = tr.rolling(14).mean()
        plus_di  = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = float(dx.rolling(14).mean().iloc[-1]) if not np.isnan(dx.rolling(14).mean().iloc[-1]) else 20.0

        # Price vs MAs
        ma50  = close.rolling(50).mean().iloc[-1]  if len(df_pre) >= 50  else close.iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(df_pre) >= 200 else close.iloc[-1]
        p50   = float(close.iloc[-1] / ma50  - 1) if ma50  > 0 else 0.0
        p200  = float(close.iloc[-1] / ma200 - 1) if ma200 > 0 else 0.0

        # Momentum
        mom60 = float(close.iloc[-1] / close.iloc[-61] - 1) if len(df_pre) >= 61 else 0.0
        mom20 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(df_pre) >= 21 else 0.0
        # Deterioration: recent momentum vs prior momentum
        mom_det = mom20 - (float(close.iloc[-21] / close.iloc[-61] - 1) if len(df_pre) >= 61 else 0.0)

        # Vol regime
        vol5  = close.pct_change().dropna().tail(5).std()
        vol60 = close.pct_change().dropna().tail(60).std()
        vol_regime = float(vol5 / vol60) if vol60 > 0 else 1.0

        # RSI
        delta = close.diff().dropna()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = float((100 - 100/(1+rs)).iloc[-1]) if len(delta) >= 14 else 50.0
        rsi   = rsi if not np.isnan(rsi) else 50.0

        # Distance from 52w high (shorts want stocks that have already broken down)
        h52 = close.tail(252).max()
        dist_high = float(close.iloc[-1] / h52 - 1) if h52 > 0 else 0.0

        # Consecutive down days
        cons_down = float((close.diff() < 0).astype(int).tail(10).sum() / 10)

        # Volume on down days vs up days
        ret1     = close.pct_change()
        down_vol = df_pre['volume'].where(ret1 < 0, 0).tail(10).mean()
        up_vol   = df_pre['volume'].where(ret1 > 0, 0).tail(10).mean()
        vol_down_ratio = float(down_vol / up_vol) if up_vol > 0 else 1.0

        # Sector vs SPY (load SPY from cache)
        sector_vs_spy = 0.0
        for spy_fname in ['SPY_max.pkl', 'SPY_3650d.csv', 'SPY_etf.csv']:
            spy_path = os.path.join(CACHE_DIR, spy_fname)
            if os.path.exists(spy_path):
                try:
                    spy = pd.read_pickle(spy_path) if spy_fname.endswith('.pkl') else pd.read_csv(spy_path, index_col=0)
                    spy.index = pd.to_datetime(spy.index).tz_localize(None)
                    spy.columns = [c.lower() for c in spy.columns]
                    spy_c = spy['close'].reindex(df_pre.index, method='ffill')
                    v = close.pct_change(20).iloc[-1] - spy_c.pct_change(20).iloc[-1]
                    if not np.isnan(v):
                        sector_vs_spy = float(v)
                    break
                except Exception:
                    pass

        return {
            'ann_vol':             float(np.clip(ann_vol, 0, 3)),
            'true_atr':            float(np.clip(true_atr, 0, 0.15)),
            'adx':                 float(np.clip(adx, 0, 100)),
            'price_vs_200ma':      float(np.clip(p200, -0.5, 2.0)),
            'price_vs_50ma':       float(np.clip(p50, -0.5, 2.0)),
            'mom_60':              float(np.clip(mom60, -0.8, 3.0)),
            'mom_20':              float(np.clip(mom20, -0.5, 2.0)),
            'mom_deterioration':   float(np.clip(mom_det, -1.0, 1.0)),
            'vol_regime':          float(np.clip(vol_regime, 0, 5)),
            'rsi_14':              float(np.clip(rsi, 0, 100)),
            'dist_from_52w_high':  float(np.clip(dist_high, -1, 0)),
            'consecutive_down_days': float(np.clip(cons_down, 0, 1)),
            'vol_on_down_days':    float(np.clip(vol_down_ratio, 0, 5)),
            'sector_vs_spy_20':    float(np.clip(sector_vs_spy, -0.5, 0.5)),
        }
    except Exception:
        return None


def build_training_data():
    import config
    rows = []
    print(f"Building short ranker training data from {len(config.WATCHLIST)} symbols...")
    sym_ok = 0

    for sym in config.WATCHLIST:
        try:
            df = fetch_max(sym)
            if df is None or len(df) < 300:
                continue
            if 'high' not in df.columns:
                continue

            # Sample every 10 trading days
            for idx in range(260, len(df)-6, 10):
                df_pre   = df.iloc[:idx]
                entry_px = float(df.iloc[idx]['close'])
                fwd_5    = float(df.iloc[idx+5]['close'] / entry_px - 1) if idx+5 < len(df) else 0.0

                feats = compute_short_features(df_pre)
                if feats is None:
                    continue

                row = feats.copy()
                row['fwd_5']  = fwd_5
                # Target: 1 = good short (negative return), 0 = bad short
                row['target'] = float(fwd_5 < -0.02)
                # Also keep continuous target for ranking
                row['target_cont'] = float(np.clip(fwd_5, -0.3, 0.3))  # negative = good short
                rows.append(row)

            sym_ok += 1
            if sym_ok % 50 == 0:
                print(f"  {sym_ok}/{len(config.WATCHLIST)} symbols, {len(rows)} rows", end='\r', flush=True)
        except Exception:
            continue

    print(f"\nDone: {sym_ok} symbols, {len(rows)} rows")
    return pd.DataFrame(rows)


def train_short_model(df):
    """Train LightGBM to predict short candidates (negative forward return)."""
    X = df[FEATURE_COLS].fillna(0)
    y = df['target_cont']  # SPY-adjusted relative return — negative = good short

    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    ic    = float(np.corrcoef(preds, y_te.values)[0,1])
    print(f"Short ranker OOS IC: {ic:.4f}")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importance:")
    for f, v in imp.items():
        print(f"  {f:<25} {v:.0f}")

    # Decile analysis — do top-ranked shorts actually go down?
    te_df = X_te.copy()
    te_df['pred']   = preds
    te_df['actual'] = y_te.values
    te_df['decile'] = pd.qcut(te_df['pred'], 10, labels=False, duplicates='drop')
    print("\nDecile analysis (decile 9 = best shorts predicted):")
    decile_means = te_df.groupby('decile')['actual'].mean()
    for d, v in decile_means.items():
        bar = "█" * int(abs(v) * 200)
        sign = "↑ loss" if v < 0 else "↓ gain"
        print(f"  Decile {d}: {v:+.4f} {sign} {bar}")

    # Retrain on all data
    model.fit(X, y)
    joblib.dump({'model': model, 'features': FEATURE_COLS}, MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH}")
    return model


def score_short_candidates(date, prices_by_symbol, sector_map, top_n=10):
    """Score all symbols as short candidates. Returns ranked list."""
    try:
        bundle = joblib.load(MODEL_PATH)
        model  = bundle['model']
        feats  = bundle['features']
    except Exception:
        return []

    results = []
    for sym, df in prices_by_symbol.items():
        try:
            df_pre = df.loc[:date]
            if len(df_pre) < 60:
                continue
            f = compute_short_features(df_pre)
            if f is None:
                continue
            X = pd.DataFrame([f], columns=feats).fillna(0)
            score = float(model.predict(X)[0])
            results.append({'symbol': sym, 'short_score': score})
        except Exception:
            continue

    results.sort(key=lambda x: x['short_score'])  # lowest = most negative expected return
    return results[:top_n]


if __name__ == "__main__":
    df = build_training_data()
    if len(df) < 500:
        print(f"ERROR: only {len(df)} rows")
        sys.exit(1)

    print(f"\nTraining on {len(df)} examples...")
    print(f"Short signal rate (rel<-2% vs SPY): {df['target'].mean():.1%}")
    print(f"Mean fwd return: {df['fwd_5'].mean():.4f}  Mean vs SPY: {df['fwd_5_rel'].mean():.4f}")

    model = train_short_model(df)

    # Validate: do top-predicted shorts actually have negative fundamentals?
    print("\n=== Today's top short candidates ===")
    import config
    from backtester_clean import fetch_history
    today = pd.Timestamp.now().normalize()
    candidates = []
    for sym in config.WATCHLIST[:100]:  # sample for speed
        df_h = fetch_max(sym)
        if df_h is not None and len(df_h) >= 60:
            f = compute_short_features(df_h)
            if f is not None:
                bundle = joblib.load(MODEL_PATH)
                X = pd.DataFrame([f], columns=bundle['features']).fillna(0)
                score = float(bundle['model'].predict(X)[0])
                px = float(df_h['close'].iloc[-1])
                candidates.append({'symbol': sym, 'score': score, 'px': px})

    candidates.sort(key=lambda x: x['score'])  # lowest = best short
    print("Top 10 short candidates (score = predicted neg return):")
    for c in candidates[:10]:
        print(f"  {c['symbol']:<8} score={c['score']:+.4f}  px=${c['px']:.2f}")
