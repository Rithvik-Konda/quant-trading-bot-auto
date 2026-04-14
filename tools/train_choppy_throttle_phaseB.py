#!/usr/bin/env python3.11
"""
Phase B: Train logistic regression for CHOPPY concentration throttle.

Loads /tmp/choppy_throttle_train.csv (IS) and /tmp/choppy_throttle_oos.csv (OOS),
trains a 3-feature logreg, validates on OOS, and simulates threshold impact.

Decision criteria (must ALL pass to proceed to integration):
1. IS 5-fold CV AUC >= 0.60
2. OOS AUC within 0.08 of IS AUC (no severe overfitting)
3. At top-25% threshold: blocks more actual stops than actual winners on OOS
4. Asymmetric block ratio: stops_blocked / winners_blocked >= 1.5
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: sklearn not installed. Run: pip3.11 install scikit-learn --break-system-packages")
    sys.exit(1)

import joblib

print("="*70)
print("  PHASE B: Train CHOPPY throttle logreg")
print("="*70)
print()

# Load data
TRAIN = Path('/tmp/choppy_throttle_train.csv')
OOS = Path('/tmp/choppy_throttle_oos.csv')

# Phase A v2 saved with the +1 bday fix; if user ran old phase A, regenerate
if not TRAIN.exists():
    print(f"ERROR: {TRAIN} not found. Run phase A first.")
    sys.exit(1)

train = pd.read_csv(TRAIN)
oos = pd.read_csv(OOS)

# Phase A may have saved the broken 162-row version. Detect and regenerate if so.
if len(train) < 250:
    print(f"[warn] train has only {len(train)} rows (expected ~360)")
    print(f"[warn] regenerating with +1 bday fix...")
    
    REPO = Path(__file__).resolve().parent.parent
    lc = pd.read_csv(REPO / 'cache_backtester' / 'position_lifecycle.csv')
    lc['date'] = pd.to_datetime(lc['date'])
    tr = pd.read_csv(REPO / 'trades_v2.csv')
    tr['entry_date'] = pd.to_datetime(tr['entry_date'])
    
    entries = lc[lc['days_held'] == 0].copy()
    choppy = entries[entries['regime'] == 'CHOPPY'].copy()
    is_ch = choppy[(choppy['date'] >= '2017-01-01') & (choppy['date'] < '2022-01-01')].copy()
    oos_ch = choppy[choppy['date'] >= '2022-01-01'].copy()
    
    def join_with_offset(lc_df, tr_df):
        lc_df = lc_df.copy()
        lc_df['date_plus1bday'] = lc_df['date'] + pd.tseries.offsets.BDay(1)
        tr_slim = tr_df[['symbol','entry_date','reason','pnl']].copy()
        j1 = lc_df.merge(
            tr_slim.rename(columns={'entry_date':'date'}),
            on=['date','symbol'], how='left'
        )
        j2 = j1.merge(
            tr_slim.rename(columns={'entry_date':'date_plus1bday','reason':'reason_p1','pnl':'pnl_p1'}),
            on=['date_plus1bday','symbol'], how='left'
        )
        j2['reason'] = j2['reason'].fillna(j2['reason_p1'])
        j2['pnl'] = j2['pnl'].fillna(j2['pnl_p1'])
        return j2.drop(columns=['date_plus1bday','reason_p1','pnl_p1'])
    
    train = join_with_offset(is_ch, tr).dropna(subset=['reason']).copy()
    oos = join_with_offset(oos_ch, tr).dropna(subset=['reason']).copy()
    train['stopped'] = (train['reason'] == 'stop').astype(int)
    oos['stopped'] = (oos['reason'] == 'stop').astype(int)
    
    cols = ['date','symbol','portfolio_size','portfolio_corr','vix_now','stopped','reason','pnl']
    train[cols].to_csv(TRAIN, index=False)
    oos[cols].to_csv(OOS, index=False)
    print(f"[ok] regenerated: train={len(train)}, oos={len(oos)}")

print(f"IS train: {len(train)} rows, {train['stopped'].sum()} stops ({train['stopped'].mean()*100:.1f}%)")
print(f"OOS eval: {len(oos)} rows, {oos['stopped'].sum()} stops ({oos['stopped'].mean()*100:.1f}%)")
print()

FEATURES = ['portfolio_size', 'portfolio_corr', 'vix_now']

X_train = train[FEATURES].values
y_train = train['stopped'].values
X_oos = oos[FEATURES].values
y_oos = oos['stopped'].values

# Drop rows with any NaN
mask_train = ~np.isnan(X_train).any(axis=1)
mask_oos = ~np.isnan(X_oos).any(axis=1)
X_train, y_train = X_train[mask_train], y_train[mask_train]
X_oos, y_oos = X_oos[mask_oos], y_oos[mask_oos]
train_clean = train[mask_train].reset_index(drop=True)
oos_clean = oos[mask_oos].reset_index(drop=True)

print(f"After NaN drop: train={len(X_train)}, oos={len(X_oos)}")
print()

# ─── Standardize features ───
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_oos_scaled = scaler.transform(X_oos)

# ─── Train logreg ───
print("─── Training logistic regression ───")
model = LogisticRegression(
    C=1.0,
    class_weight='balanced',  # handle 10.8% positive rate
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ─── Feature importance ───
print("\n─── Feature coefficients (standardized) ───")
print(f"  {'feature':18s}  {'coef':>10s}  {'sign means':>30s}")
print(f"  {'-'*18}  {'-'*10}  {'-'*30}")
for f, coef in zip(FEATURES, model.coef_[0]):
    direction = "higher → MORE likely to stop" if coef > 0 else "higher → LESS likely to stop"
    print(f"  {f:18s}  {coef:+10.4f}  {direction}")
print(f"  {'intercept':18s}  {model.intercept_[0]:+10.4f}")
print()

# ─── 5-fold CV on IS ───
print("─── 5-fold CV on IS ───")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
is_auc = cv_scores.mean()
is_auc_std = cv_scores.std()
print(f"  Per-fold AUC: {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean AUC: {is_auc:.3f} (std {is_auc_std:.3f})")
print()

# ─── OOS AUC ───
oos_proba = model.predict_proba(X_oos_scaled)[:, 1]
oos_auc = roc_auc_score(y_oos, oos_proba)
print(f"─── OOS AUC: {oos_auc:.3f} ───")
print(f"  IS - OOS gap: {is_auc - oos_auc:+.3f}")
print()

# ─── Threshold simulation ───
print("─── Threshold impact simulation (OOS) ───")
print()

# Get IS predictions to calibrate thresholds
is_proba = model.predict_proba(X_train_scaled)[:, 1]

# For each top-N% threshold, compute on OOS:
# - How many ACTUAL stops we would have blocked
# - How many ACTUAL winners we would have blocked
# - Net P&L impact (sum of pnl of blocked trades — we WANT this to be negative)
print(f"{'top%':>6s}  {'P>':>7s}  {'blocked':>8s}  {'stops':>7s}  {'wins':>6s}  {'losers':>7s}  {'ratio':>7s}  {'blocked_pnl':>13s}")
print(f"{'-'*6}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*13}")

results = []
for top_pct in [10, 15, 20, 25, 33, 40, 50]:
    # Threshold = (100-top_pct)th percentile of IS predictions
    threshold = np.percentile(is_proba, 100 - top_pct)
    
    # Apply to OOS
    blocked_mask = oos_proba >= threshold
    n_blocked = blocked_mask.sum()
    
    blocked_trades = oos_clean[blocked_mask]
    n_stops_blocked = (blocked_trades['stopped'] == 1).sum()
    n_winners_blocked = ((blocked_trades['stopped'] == 0) & (blocked_trades['pnl'] > 0)).sum()
    n_losers_blocked = ((blocked_trades['stopped'] == 0) & (blocked_trades['pnl'] <= 0)).sum()
    blocked_pnl = blocked_trades['pnl'].sum()
    
    # Asymmetry ratio: stops blocked vs winners blocked
    ratio = n_stops_blocked / max(n_winners_blocked, 1)
    
    results.append({
        'top_pct': top_pct,
        'threshold': threshold,
        'blocked': n_blocked,
        'stops_blocked': n_stops_blocked,
        'winners_blocked': n_winners_blocked,
        'losers_blocked': n_losers_blocked,
        'blocked_pnl': blocked_pnl,
        'ratio': ratio
    })
    
    print(f"{top_pct:>5d}%  {threshold:.4f}  {n_blocked:>8d}  {n_stops_blocked:>7d}  "
          f"{n_winners_blocked:>6d}  {n_losers_blocked:>7d}  {ratio:>7.2f}  ${blocked_pnl:>+12,.0f}")
print()

# ─── Decision criteria ───
print("="*70)
print("  DECISION CRITERIA")
print("="*70)
ok_is_auc = is_auc >= 0.60
ok_oos_auc = (is_auc - oos_auc) <= 0.08
best_result = max(results, key=lambda r: r['ratio'] if r['blocked_pnl'] < 0 else 0)
ok_asymmetric = best_result['ratio'] >= 1.5 and best_result['blocked_pnl'] < 0

print(f"  IS CV AUC >= 0.60:          {is_auc:.3f}  {'✅' if ok_is_auc else '❌'}")
print(f"  OOS AUC gap <= 0.08:        gap={is_auc-oos_auc:+.3f}  {'✅' if ok_oos_auc else '❌'}")
print(f"  Best asymmetry >= 1.5:      ratio={best_result['ratio']:.2f}  {'✅' if best_result['ratio']>=1.5 else '❌'}")
print(f"  Blocked PnL negative:       ${best_result['blocked_pnl']:+,.0f}  {'✅' if best_result['blocked_pnl']<0 else '❌'}")
print()
print(f"  Best threshold: top {best_result['top_pct']}% (P > {best_result['threshold']:.4f})")
print(f"    blocks {best_result['blocked']} trades, saves ${-best_result['blocked_pnl']:,.0f} in PnL")
print()

if ok_is_auc and ok_oos_auc and ok_asymmetric:
    print("  >>> PROCEED to backtest integration with this model")
    # Save model and threshold for integration
    cache = Path('cache_backtester')
    cache.mkdir(exist_ok=True)
    artifact = {
        'model': model,
        'scaler': scaler,
        'features': FEATURES,
        'threshold': float(best_result['threshold']),
        'top_pct': best_result['top_pct'],
        'is_auc': float(is_auc),
        'oos_auc': float(oos_auc),
    }
    joblib.dump(artifact, cache / 'choppy_throttle.joblib')
    print(f"  Model saved to: cache_backtester/choppy_throttle.joblib")
else:
    print("  >>> DO NOT PROCEED — model doesn't meet criteria")
    print("  Try: Option 5 (XGBoost), Option 7 (more features), or Option 6 (different target)")
