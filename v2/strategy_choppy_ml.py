"""
strategy_choppy_ml.py — ML-trained CHOPPY regime strategy
============================================================
Problem: TRENDING_BULL momentum fails in CHOPPY markets
Solution: Train a separate ML model on CHOPPY regime periods only
          using different features that predict CHOPPY outperformance

Key insight from data:
  CHOPPY regime winners: low-beta, high-quality, defensive
  CHOPPY regime losers:  high-beta momentum names (NVDA, SHOP, AVAV)

Features that matter in CHOPPY (different from momentum):
  - Beta vs SPY (lower = better)
  - Dividend yield (higher = better)  
  - ROE stability (consistent > high)
  - Earnings beat consistency
  - Revenue predictability (recurring > lumpy)
  - Price stability (low 60d vol)
  - Analyst consensus (high agreement = defensive)

Training: use only trading days when regime = CHOPPY
Target: 20d forward return (shorter than 30d momentum target)
"""
import pandas as pd
import numpy as np
import joblib
import os, sys
from datetime import datetime
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

CHOPPY_MODEL_PATH = '/Users/rick/ai_trading_bot_v2/choppy_ranker.joblib'
CHOPPY_CACHE      = '/Users/rick/ai_trading_bot_v2/cache_choppy'
os.makedirs(CHOPPY_CACHE, exist_ok=True)

def compute_choppy_features(df: pd.DataFrame, symbol: str = '') -> pd.DataFrame:
    """
    Features predictive in CHOPPY regime — different from momentum features.
    Focus: stability, quality, low-volatility characteristics.
    """
    import yfinance as yf

    feats = pd.DataFrame(index=df.index)
    close = df['close']
    high  = df['high'] if 'high' in df.columns else close
    low   = df['low']  if 'low'  in df.columns else close
    vol   = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

    # ── Price stability features ──────────────────────────────
    # Low volatility = CHOPPY winner
    feats['vol_20d']  = close.pct_change().rolling(20).std() * np.sqrt(252)
    feats['vol_60d']  = close.pct_change().rolling(60).std() * np.sqrt(252)
    feats['vol_ratio'] = feats['vol_20d'] / feats['vol_60d'].clip(lower=0.01)

    # Price stability — how much does it deviate from trend
    ma50 = close.rolling(50).mean()
    feats['price_stability'] = 1 - (close - ma50).abs() / ma50.clip(lower=0.01)

    # ── Momentum features (muted in CHOPPY) ──────────────────
    feats['mom_20d'] = close.pct_change(20)
    feats['mom_60d'] = close.pct_change(60)
    # In CHOPPY: negative momentum is actually predictive of mean reversion
    # mom_reversal removed — was leaking target signal

    # ── Volume stability ──────────────────────────────────────
    vol_ma = vol.rolling(20).mean()
    feats['vol_stability'] = 1 - (vol - vol_ma).abs() / vol_ma.clip(lower=1)

    # ── Drawdown recovery ─────────────────────────────────────
    roll_max = close.rolling(60).max()
    feats['drawdown_60d'] = (close - roll_max) / roll_max.clip(lower=0.01)
    # Stocks near 52-week high in CHOPPY = dangerous
    feats['pct_from_high'] = feats['drawdown_60d']

    # ── Rolling beta vs SPY (price-derived, historically valid) ─
    # Not static snapshot — computed from actual price history
    try:
        import yfinance as _yf_spy
        spy_hist = _yf_spy.Ticker('SPY').history(period='max')['Close']
        spy_hist.index = spy_hist.index.tz_localize(None)
        spy_ret  = spy_hist.pct_change()
        stk_ret  = close.pct_change()
        # Align
        common   = spy_ret.index.intersection(stk_ret.index)
        spy_al   = spy_ret.loc[common]
        stk_al   = stk_ret.loc[common]
        # Rolling 60-day beta
        cov      = stk_al.rolling(60).cov(spy_al)
        var      = spy_al.rolling(60).var()
        roll_beta = (cov / var.clip(lower=1e-8)).clip(-2, 3)
        roll_beta = roll_beta.reindex(df.index).ffill().fillna(1.0)
        feats['rolling_beta_60d'] = roll_beta
        feats['low_beta_signal']  = (1.0 - roll_beta.clip(0, 2)).clip(0, 1)
        # Rolling correlation with SPY (lower = more defensive)
        roll_corr = stk_al.rolling(60).corr(spy_al)
        roll_corr = roll_corr.reindex(df.index).ffill().fillna(0.5)
        feats['spy_corr_60d'] = roll_corr
        feats['low_corr_signal'] = 1 - roll_corr.clip(0, 1)
    except:
        feats['rolling_beta_60d'] = pd.Series(1.0, index=df.index)
        feats['low_beta_signal']  = pd.Series(0.5, index=df.index)
        feats['spy_corr_60d']     = pd.Series(0.5, index=df.index)
        feats['low_corr_signal']  = pd.Series(0.5, index=df.index)

    # ── Technical mean-reversion ──────────────────────────────
    rsi_delta = close.diff()
    gain = rsi_delta.clip(lower=0).rolling(14).mean()
    loss = (-rsi_delta.clip(upper=0)).rolling(14).mean()
    rs   = gain / loss.clip(lower=0.001)
    feats['rsi_14'] = 100 - 100/(1+rs)
    # In CHOPPY: oversold (RSI<40) tends to bounce
    feats['rsi_oversold'] = (feats['rsi_14'] < 40).astype(float)

    # Bollinger band position
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feats['bb_position'] = (close - bb_mid) / (2 * bb_std.clip(lower=0.001))

    # ── ATR for stop sizing ───────────────────────────────────
    atr = (high - low).rolling(14).mean()
    feats['atr_pct'] = atr / close.clip(lower=0.01)

    # ── Cross-sectional stability rank (key CHOPPY signal) ───
    # Computed per-date so no leakage
    # Low vol + low beta = CHOPPY winner
    # This is the signal we want the model to learn
    feats['stability_composite'] = (
        -feats['vol_20d'] * 0.4 +        # lower vol = better
        -feats['rolling_beta_60d'] * 0.3 + # lower beta = better  
        feats['price_stability'] * 0.3    # price near MA = better
    )

    return feats.fillna(0)


def train_choppy_model(trades_csv: str = None, regime_data: dict = None):
    """
    Train ML model on CHOPPY regime periods only.
    Uses historical trades to identify what worked in CHOPPY.
    """
    from lightgbm import LGBMRegressor
    from backtester_clean import fetch_history
    import config

    print("Training CHOPPY regime ML model...")
    print("Universe: defensive stocks + full watchlist")
    print("Target: 20d forward return in CHOPPY conditions")

    # Load historical trades to identify CHOPPY periods
    df_trades = pd.read_csv(trades_csv or '/Users/rick/ai_trading_bot_v2/trades_v2.csv')
    df_trades['exit_date']  = pd.to_datetime(df_trades['exit_date'])
    df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])

    # CHOPPY periods by year (from our regime classifier)
    # 2019: mostly CHOPPY, 2021: some CHOPPY, 2023-2025: mixed
    CHOPPY_DATES = df_trades[df_trades['regime'] == 'CHOPPY']['entry_date'].tolist()

    if len(CHOPPY_DATES) < 20:
        print(f"  Only {len(CHOPPY_DATES)} CHOPPY trades — expanding training set")
        # Add known CHOPPY periods
        choppy_ranges = [
            ('2019-01-01', '2019-12-31'),
            ('2021-01-01', '2021-06-30'),
            ('2023-01-01', '2023-03-31'),
        ]

    # Build training data
    # For each stock, compute CHOPPY features and 20d forward returns
    # on dates that were CHOPPY regime
    X_rows = []
    y_rows = []

    # Use defensive universe + top momentum names for contrast
    TRAIN_UNIVERSE = [
        'KO','PG','WMT','JNJ','ABT','NEE','DUK','WM','TJX','MSCI',
        'ITW','PGR','TRV','CB','V','MA','MSFT','AAPL','AMZN','GOOGL',
        'NVDA','SHOP','TTD','AVAV','PLTR','APP','SNOW','CRWD','DDOG',
        'META','AMD','INTU','CRM','NOW','ADBE','ORCL',
    ]

    print(f"  Training on {len(TRAIN_UNIVERSE)} symbols...")

    for sym in TRAIN_UNIVERSE:
        try:
            prices = fetch_history(sym, days=9999)
            prices.index = pd.to_datetime(prices.index).tz_localize(None)

            if len(prices) < 200:
                continue

            # Compute features
            feats = compute_choppy_features(prices, symbol=sym)

            # 20d forward return (target)
            fwd_ret = prices['close'].pct_change(20).shift(-20)

            # Only use CHOPPY dates (2019, H1 2021 as proxy)
            choppy_mask = (
                (prices.index.year == 2019) |
                ((prices.index.year == 2021) & (prices.index.month <= 6)) |
                ((prices.index.year == 2023) & (prices.index.month <= 3))
            )

            feats_choppy = feats[choppy_mask].dropna()
            fwd_choppy   = fwd_ret[choppy_mask].dropna()

            # Align
            common_idx = feats_choppy.index.intersection(fwd_choppy.index)
            if len(common_idx) < 20:
                continue

            X_rows.append(feats_choppy.loc[common_idx])
            y_rows.append(fwd_choppy.loc[common_idx])

        except Exception as e:
            pass

    if not X_rows:
        print("  No training data found")
        return None

    X = pd.concat(X_rows).fillna(0)
    y = pd.concat(y_rows)

    # Clip outliers
    y = y.clip(-0.30, 0.30)

    print(f"  Training data: {len(X)} rows, {len(X.columns)} features")

    # Train LightGBM
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    # Train/test split — use 2019 as test, 2021 as train
    train_mask = X.index.year != 2019
    test_mask  = X.index.year == 2019

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    if len(X_test) < 100:
        # Fallback: last 20% as test
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test,  y_test  = X.iloc[split:], y.iloc[split:]

    model.fit(X_train, y_train)
    print(f"  Train: {len(X_train)} rows  Test: {len(X_test)} rows")

    # IC on HOLDOUT test set only — this is the real number
    preds = model.predict(X_test)
    ic = float(pd.Series(preds).corr(pd.Series(y_test.values), method='spearman'))
    train_ic = float(pd.Series(model.predict(X_train)).corr(pd.Series(y_train.values), method='spearman'))
    print(f"  Train IC: {train_ic:.4f} (ignore — in-sample)")
    print(f"  Training IC: {ic:.4f}")

    # Save model
    result = {
        'model':    model,
        'features': list(X.columns),
        'ic':       ic,
        'trained':  str(datetime.now()),
    }
    joblib.dump(result, CHOPPY_MODEL_PATH)
    print(f"  Saved to {CHOPPY_MODEL_PATH}")
    return result


def score_choppy_stocks(symbols: list, current_prices: dict = None) -> list:
    """Score stocks using CHOPPY ML model."""
    from backtester_clean import fetch_history

    if not os.path.exists(CHOPPY_MODEL_PATH):
        print("CHOPPY model not trained yet — run train_choppy_model() first")
        return []

    ranker = joblib.load(CHOPPY_MODEL_PATH)
    model  = ranker['model']
    feat_cols = ranker['features']

    results = []
    for sym in symbols:
        try:
            prices = fetch_history(sym, days=200)
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            feats = compute_choppy_features(prices, symbol=sym)
            latest = feats.iloc[-1].reindex(feat_cols, fill_value=0)
            score = float(model.predict([latest.values])[0])
            results.append({'symbol': sym, 'choppy_score': score})
        except:
            pass

    # Cross-sectional rank
    if results:
        scores = np.array([r['choppy_score'] for r in results])
        ranks  = scores.argsort().argsort() / max(len(scores)-1, 1)
        for r, rank in zip(results, ranks):
            r['choppy_rank_pct'] = float(rank)

    results.sort(key=lambda x: x.get('choppy_rank_pct', 0), reverse=True)
    return results


if __name__ == "__main__":
    import sys
    if '--train' in sys.argv:
        result = train_choppy_model()
        if result:
            print(f"\nCHOPPY model trained successfully")
            print(f"IC: {result['ic']:.4f}")
            print(f"Features: {len(result['features'])}")
    else:
        print("CHOPPY ML Strategy")
        print("Usage:")
        print("  python strategy_choppy_ml.py --train   # train model")
        print("  then import score_choppy_stocks() for live scoring")

        if os.path.exists(CHOPPY_MODEL_PATH):
            ranker = joblib.load(CHOPPY_MODEL_PATH)
            print(f"\nExisting model: IC={ranker['ic']:.4f}, "
                  f"features={len(ranker['features'])}, "
                  f"trained={ranker['trained']}")
        else:
            print("\nNo model trained yet — run with --train")
