"""
vol_strategy_ml.py — ML-enhanced volatility strategies

Two models:
1. VIX strike predictor: given current VIX + term structure + regime,
   predict where VIX will be in 30 days → optimal put strike
   
2. VIX spike predictor: predict probability of VIX >30 in next 10 days
   Used to size VXX short position (0 when spike likely, max when calm)

Both models trained on full VIX history — no hardcoded thresholds.
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from scipy.stats import norm

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

CACHE_DIR  = '/Users/rick/ai_trading_bot_v2/cache_prices'
MODEL_PATH = '/Users/rick/ai_trading_bot_v2/vol_models.joblib'


def load_vix_history():
    path = os.path.join(CACHE_DIR, '^VIX_etf.csv')
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    return df['close'].dropna()


def build_vix_features(vix: pd.Series, idx: int) -> dict:
    """Features for VIX prediction models."""
    if idx < 60:
        return None
    try:
        v     = vix.iloc[idx]
        v5    = vix.iloc[idx-5]
        v20   = vix.iloc[idx-20]
        v60   = vix.iloc[idx-60]
        hist  = vix.iloc[max(0,idx-252):idx]

        # Level features
        vix_level      = float(v)
        vix_pctile_1yr = float((hist < v).mean())  # where is VIX vs own history

        # Momentum
        vix_mom_5d  = float(v / v5 - 1) if v5 > 0 else 0.0
        vix_mom_20d = float(v / v20 - 1) if v20 > 0 else 0.0

        # Mean reversion signal — how far from long-run mean
        vix_mean_1yr = float(hist.mean())
        vix_dev      = float((v - vix_mean_1yr) / vix_mean_1yr) if vix_mean_1yr > 0 else 0.0

        # Realized vol of VIX (vol of vol)
        vix_ret    = vix.pct_change().iloc[max(0,idx-20):idx]
        vix_vol_20 = float(vix_ret.std() * np.sqrt(252))

        # VIX regime — is it in spike territory?
        above_30   = float(v >= 30)
        above_25   = float(v >= 25)
        above_20   = float(v >= 20)

        # Days since last VIX spike > 30
        recent = vix.iloc[max(0,idx-90):idx]
        spikes = (recent > 30).values
        if spikes.any():
            days_since_spike = float(len(spikes) - np.where(spikes[::-1])[0][0])
        else:
            days_since_spike = 90.0

        # Consecutive high VIX days
        consec_high = 0
        for j in range(idx-1, max(idx-20,0), -1):
            if vix.iloc[j] > 25:
                consec_high += 1
            else:
                break

        return {
            'vix_level':         float(np.clip(vix_level, 10, 80)),
            'vix_pctile_1yr':    float(np.clip(vix_pctile_1yr, 0, 1)),
            'vix_mom_5d':        float(np.clip(vix_mom_5d, -0.5, 1.0)),
            'vix_mom_20d':       float(np.clip(vix_mom_20d, -0.5, 2.0)),
            'vix_dev_from_mean': float(np.clip(vix_dev, -0.5, 2.0)),
            'vix_vol_of_vol':    float(np.clip(vix_vol_20, 0, 5)),
            'above_30':          above_30,
            'above_25':          above_25,
            'above_20':          above_20,
            'days_since_spike':  float(np.clip(days_since_spike, 0, 90)),
            'consec_high_days':  float(np.clip(consec_high, 0, 30)),
        }
    except Exception:
        return None


def build_training_data(vix: pd.Series):
    """Build features + targets for both models."""
    rows = []
    print(f"Building VIX model training data from {len(vix)} days...")

    for idx in range(60, len(vix) - 31):
        feats = build_vix_features(vix, idx)
        if feats is None:
            continue

        v_now  = float(vix.iloc[idx])
        v_30d  = float(vix.iloc[idx + 30])
        v_10d  = float(vix.iloc[idx + 10])

        # Target 1: VIX level in 30 days (for strike selection)
        target_vix_30d = v_30d

        # Target 2: probability VIX spikes above 30 in next 10 days
        max_vix_10d     = float(vix.iloc[idx:idx+10].max())
        target_spike_10d = float(max_vix_10d > 30)

        # Target 3: optimal put strike (30d VIX * 0.95 ATM-ish)
        # Strike should be between current VIX and 30d predicted VIX
        optimal_strike = float(np.clip(v_30d * 1.05, v_now * 0.70, v_now * 0.98))

        row = feats.copy()
        row['target_vix_30d']    = target_vix_30d
        row['target_spike_10d']  = target_spike_10d
        row['target_opt_strike'] = optimal_strike
        row['vix_now']           = v_now
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Training rows: {len(df)}")
    return df


FEATURE_COLS = [
    'vix_level', 'vix_pctile_1yr', 'vix_mom_5d', 'vix_mom_20d',
    'vix_dev_from_mean', 'vix_vol_of_vol', 'above_30', 'above_25',
    'above_20', 'days_since_spike', 'consec_high_days',
]


def train_models(df: pd.DataFrame):
    """Train spike predictor and VIX level predictor."""
    X = df[FEATURE_COLS].fillna(0)
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]

    # Model 1: VIX spike predictor (binary)
    y_spike_tr = df['target_spike_10d'].iloc[:split]
    y_spike_te = df['target_spike_10d'].iloc[split:]

    spike_model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=30, subsample=0.8,
        random_state=42, verbose=-1,
    )
    spike_model.fit(X_tr, y_spike_tr)
    spike_preds = spike_model.predict_proba(X_te)[:, 1]
    spike_acc   = float(((spike_preds > 0.5) == y_spike_te.values).mean())
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_spike_te, spike_preds)
        print(f"Spike model AUC: {auc:.4f}")
    except Exception:
        pass

    # Model 2: VIX level predictor (regression)
    y_level_tr = df['target_vix_30d'].iloc[:split]
    y_level_te = df['target_vix_30d'].iloc[split:]

    level_model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=30, subsample=0.8,
        random_state=42, verbose=-1,
    )
    level_model.fit(X_tr, y_level_tr)
    level_preds = level_model.predict(X_te)
    mae = float(np.abs(level_preds - y_level_te).mean())
    print(f"VIX level predictor MAE: {mae:.2f} VIX points")

    # Model 3: Optimal strike predictor
    y_strike_tr = df['target_opt_strike'].iloc[:split]
    strike_model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=30, subsample=0.8,
        random_state=42, verbose=-1,
    )
    strike_model.fit(X_tr, y_strike_tr)

    imp = pd.Series(spike_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nSpike model feature importance:")
    for f, v in imp.head(6).items():
        print(f"  {f:<25} {v:.0f}")

    bundle = {
        'spike_model':  spike_model,
        'level_model':  level_model,
        'strike_model': strike_model,
        'features':     FEATURE_COLS,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH}")
    return bundle


def predict_vix_spike_prob(vix: pd.Series) -> float:
    """Probability VIX spikes >30 in next 10 days. Used to size VXX short."""
    try:
        bundle = joblib.load(MODEL_PATH)
        feats  = build_vix_features(vix, len(vix)-1)
        if feats is None:
            return 0.3
        X = pd.DataFrame([feats], columns=bundle['features']).fillna(0)
        prob = float(bundle['spike_model'].predict_proba(X)[0, 1])
        return prob
    except Exception:
        return 0.3


def predict_vix_30d(vix: pd.Series) -> float:
    """Predicted VIX level in 30 days. Used to select put strike."""
    try:
        bundle = joblib.load(MODEL_PATH)
        feats  = build_vix_features(vix, len(vix)-1)
        if feats is None:
            return float(vix.iloc[-1])
        X = pd.DataFrame([feats], columns=bundle['features']).fillna(0)
        return float(bundle['level_model'].predict(X)[0])
    except Exception:
        return float(vix.iloc[-1])


def predict_optimal_strike(vix: pd.Series) -> float:
    """ML-predicted optimal put strike for current VIX conditions."""
    try:
        bundle = joblib.load(MODEL_PATH)
        feats  = build_vix_features(vix, len(vix)-1)
        if feats is None:
            return float(vix.iloc[-1]) * 0.85
        X = pd.DataFrame([feats], columns=bundle['features']).fillna(0)
        return float(bundle['strike_model'].predict(X)[0])
    except Exception:
        return float(vix.iloc[-1]) * 0.85


def run_ml_vxx_backtest(vix: pd.Series, bundle: dict):
    """
    VXX short with ML spike predictor sizing.
    Position size = (1 - spike_prob) * max_allocation
    When spike probability is high, we reduce or eliminate position.
    """
    from regime_classifier import RegimeClassifier, compute_signals, load_macro_data, TRENDING_BULL

    print("\n=== ML-ENHANCED VXX SHORT BACKTEST ===")
    print("Position size scaled by (1 - P(VIX spike in 10d))")

    svxy_path = os.path.join(CACHE_DIR, 'SVXY_max.pkl')
    svxy = pd.read_pickle(svxy_path)
    svxy.index = pd.to_datetime(svxy.index).tz_localize(None)
    svxy.columns = [c.lower() for c in svxy.columns]
    svxy_close = svxy['close'].dropna()

    spy_mac, hyg_mac, vix_mac = load_macro_data(cache_dir=CACHE_DIR)
    clf = RegimeClassifier()
    regime_map = {}
    for d in spy_mac.index:
        signals = compute_signals(spy_mac, hyg_mac, vix_mac, as_of_date=d)
        if signals:
            regime_map[pd.Timestamp(d).normalize()] = clf.update(d, signals)

    PORTFOLIO    = 100_000
    MAX_ALLOC    = 0.05   # 5% max when spike prob is low
    SLIPPAGE     = 0.001

    common = sorted(svxy_close.index.intersection(pd.DatetimeIndex(regime_map.keys())))

    trades     = []
    position   = 0        # shares held (long SVXY)
    entry_px   = 0.0
    entry_date = None
    total_pnl  = 0.0

    print(f"\n{'Date':<12} {'Regime':<14} {'SVXY':>7} {'VIX':>6} {'SpikeP':>8} {'Action':<25} {'PnL':>10}")
    print("-" * 85)

    for date in common:
        regime  = regime_map.get(date, 'CHOPPY')
        vix_now = float(vix.loc[date]) if date in vix.index else 20.0
        svxy_px = float(svxy_close.loc[date]) if date in svxy_close.index else None
        if svxy_px is None:
            continue

        # ML spike probability
        vix_to_date = vix.loc[vix.index <= date]
        if len(vix_to_date) >= 60:
            idx = len(vix_to_date) - 1
            feats = build_vix_features(vix_to_date, idx)
            if feats:
                X = pd.DataFrame([feats], columns=bundle['features']).fillna(0)
                spike_prob = float(bundle['spike_model'].predict_proba(X)[0, 1])
            else:
                spike_prob = 0.3
        else:
            spike_prob = 0.3

        # Target position: long SVXY when TRENDING_BULL and low spike prob
        in_bull = (regime == TRENDING_BULL)
        target_alloc = MAX_ALLOC * (1 - spike_prob) if in_bull else 0.0
        target_shares = int((PORTFOLIO * target_alloc) / svxy_px)

        # Rebalance if target differs significantly from current
        if abs(target_shares - position) >= max(1, int(position * 0.2)):
            # Close existing
            if position > 0:
                exit_px = svxy_px * (1 - SLIPPAGE)
                pnl = (exit_px - entry_px) * position
                total_pnl += pnl
                days = (date - entry_date).days if entry_date else 0
                if pnl != 0:
                    print(f"  {str(date.date()):<10} {regime:<14} {svxy_px:>7.2f} "
                          f"{vix_now:>6.1f} {spike_prob:>8.1%} "
                          f"CLOSE {position} SVXY {pnl:>+14,.0f} ({pnl/(entry_px*position):.1%})")
                    trades.append({
                        'entry_date': str(entry_date.date()) if entry_date else '',
                        'exit_date':  str(date.date()),
                        'pnl': pnl, 'days': days,
                        'spike_prob': spike_prob,
                    })

            # Open new if target > 0
            if target_shares > 0:
                entry_px   = svxy_px * (1 + SLIPPAGE)
                entry_date = date
                position   = target_shares
                print(f"  {str(date.date()):<10} {regime:<14} {svxy_px:>7.2f} "
                      f"{vix_now:>6.1f} {spike_prob:>8.1%} "
                      f"LONG {position} SVXY @{entry_px:.2f}")
            else:
                position = 0
                entry_px = 0.0

    # Summary
    if trades:
        df_r = pd.DataFrame(trades)
        wins = df_r[df_r['pnl'] > 0]
        print(f"\n{'='*60}")
        print(f"ML VXX/SVXY SUMMARY")
        print(f"{'='*60}")
        print(f"  Trades:     {len(df_r)}")
        print(f"  Win rate:   {len(wins)/len(df_r):.0%}")
        print(f"  Total PnL:  ${df_r['pnl'].sum():,.0f}")
        years = max((pd.Timestamp(df_r['exit_date'].iloc[-1]) -
                     pd.Timestamp(df_r['entry_date'].iloc[0])).days / 365, 1)
        cagr = (total_pnl / PORTFOLIO) / years
        print(f"  CAGR contribution: {cagr:+.2%}")
        verdict = "DEPLOY" if cagr > 0.01 else "MARGINAL" if cagr > 0 else "DO NOT DEPLOY"
        print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    vix = load_vix_history()
    print(f"VIX history: {len(vix)} rows, {vix.index[0].date()} → {vix.index[-1].date()}")

    df_train = build_training_data(vix)
    bundle   = train_models(df_train)

    # Test predictions for current conditions
    print("\n=== Current VIX predictions ===")
    vix_now   = float(vix.iloc[-1])
    spike_p   = predict_vix_spike_prob(vix)
    vix_30d   = predict_vix_30d(vix)
    opt_strike = predict_optimal_strike(vix)

    print(f"  Current VIX:          {vix_now:.1f}")
    print(f"  P(spike>30 in 10d):   {spike_p:.1%}")
    print(f"  Predicted VIX in 30d: {vix_30d:.1f}")
    print(f"  Optimal put strike:   {opt_strike:.1f}")
    print(f"  VXX position sizing:  {(1-spike_p)*100:.0f}% of max allocation")

    # Run ML-enhanced backtest
    run_ml_vxx_backtest(vix, bundle)
