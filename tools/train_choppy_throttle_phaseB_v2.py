#!/usr/bin/env python3.11
"""
Phase B v2: Expanded feature set, logreg + XGBoost.

Adds: ml_rank_entry, portfolio_rank_mean, portfolio_rank_vel, ann_vol
to the original 3 features. Trains both logreg and small XGBoost.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_GBM = True
except ImportError:
    HAS_GBM = False
    print("[warn] sklearn.ensemble not available")

print("="*70)
print("  PHASE B v2: Expanded features, logreg + GBM")
print("="*70)
print()

# ─── Regenerate from lifecycle CSV with expanded features ───
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
    j1 = lc_df.merge(tr_slim.rename(columns={'entry_date':'date'}), on=['date','symbol'], how='left')
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

# ─── EXPANDED feature set ───
FEATURES = [
    'portfolio_size',     # original
    'portfolio_corr',     # original
    'vix_now',            # original
    'ml_rank_entry',      # NEW: how strongly ranker liked this entry
    'portfolio_rank_mean',# NEW: avg ranker score of held positions
    'portfolio_rank_vel', # NEW: rate of change of portfolio rank
    'ann_vol',            # NEW: annualized vol of this position
]

print(f"IS train: {len(train)} rows, {train['stopped'].sum()} stops ({train['stopped'].mean()*100:.1f}%)")
print(f"OOS eval: {len(oos)} rows, {oos['stopped'].sum()} stops ({oos['stopped'].mean()*100:.1f}%)")
print(f"Features: {len(FEATURES)} ({', '.join(FEATURES)})")
print()

# Show feature availability
print("─── Feature NaN counts ───")
for f in FEATURES:
    nan_train = train[f].isna().sum() if f in train.columns else 'MISSING'
    nan_oos = oos[f].isna().sum() if f in oos.columns else 'MISSING'
    print(f"  {f:22s}  train_nan={nan_train}  oos_nan={nan_oos}")
print()

X_train = train[FEATURES].values
y_train = train['stopped'].values
X_oos = oos[FEATURES].values
y_oos = oos['stopped'].values

mask_train = ~np.isnan(X_train).any(axis=1)
mask_oos = ~np.isnan(X_oos).any(axis=1)
X_train, y_train = X_train[mask_train], y_train[mask_train]
X_oos, y_oos = X_oos[mask_oos], y_oos[mask_oos]
train_clean = train[mask_train].reset_index(drop=True)
oos_clean = oos[mask_oos].reset_index(drop=True)

print(f"After NaN drop: train={len(X_train)} ({y_train.sum()} stops), oos={len(X_oos)} ({y_oos.sum()} stops)")
print()

# ─── Standardize ───
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_oos_scaled = scaler.transform(X_oos)


def evaluate_model(name, model, X_tr, y_tr, X_oo, y_oo, oos_df, is_proba_for_thresholds=None):
    """Train, CV, OOS test, threshold sweep."""
    print(f"\n{'='*70}")
    print(f"  MODEL: {name}")
    print(f"{'='*70}")
    
    # Fit
    model.fit(X_tr, y_tr)
    
    # Feature importance
    if hasattr(model, 'coef_'):
        print("\n  Coefficients (standardized):")
        for f, c in zip(FEATURES, model.coef_[0]):
            print(f"    {f:22s}  {c:+.4f}")
    elif hasattr(model, 'feature_importances_'):
        print("\n  Feature importances:")
        for f, imp in zip(FEATURES, model.feature_importances_):
            print(f"    {f:22s}  {imp:.4f}")
    
    # CV AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='roc_auc')
    is_auc = cv_scores.mean()
    print(f"\n  IS 5-fold CV AUC: {is_auc:.3f} (per fold: {[f'{s:.3f}' for s in cv_scores]})")
    
    # OOS AUC
    oos_proba = model.predict_proba(X_oo)[:, 1]
    oos_auc = roc_auc_score(y_oo, oos_proba)
    gap = is_auc - oos_auc
    print(f"  OOS AUC: {oos_auc:.3f}")
    print(f"  Gap: {gap:+.3f}")
    
    # Threshold sweep
    is_proba = model.predict_proba(X_tr)[:, 1]
    
    print(f"\n  Threshold sweep on OOS:")
    print(f"  {'top%':>6s}  {'P>':>7s}  {'blocked':>8s}  {'stops':>7s}  {'wins':>6s}  {'losers':>7s}  {'ratio':>7s}  {'blocked_pnl':>13s}")
    
    best = None
    for top_pct in [10, 15, 20, 25, 33, 40, 50]:
        threshold = np.percentile(is_proba, 100 - top_pct)
        blocked = oos_proba >= threshold
        n_blocked = blocked.sum()
        bt = oos_df[blocked]
        n_stops = (bt['stopped'] == 1).sum()
        n_wins = ((bt['stopped'] == 0) & (bt['pnl'] > 0)).sum()
        n_losers = ((bt['stopped'] == 0) & (bt['pnl'] <= 0)).sum()
        bp = bt['pnl'].sum()
        ratio = n_stops / max(n_wins, 1)
        
        print(f"  {top_pct:>5d}%  {threshold:.4f}  {n_blocked:>8d}  {n_stops:>7d}  {n_wins:>6d}  {n_losers:>7d}  {ratio:>7.2f}  ${bp:>+12,.0f}")
        
        if bp < 0 and (best is None or ratio > best['ratio']):
            best = {'top_pct': top_pct, 'threshold': threshold, 'ratio': ratio, 
                    'blocked_pnl': bp, 'n_blocked': n_blocked, 'n_stops': n_stops, 'n_wins': n_wins}
    
    # Decision
    print(f"\n  Decision criteria:")
    ok_is = is_auc >= 0.60
    ok_gap = gap <= 0.08
    ok_best = best is not None and best['ratio'] >= 1.5
    
    print(f"    IS AUC >= 0.60:        {is_auc:.3f}  {'✅' if ok_is else '❌'}")
    print(f"    Gap <= 0.08:           {gap:+.3f}  {'✅' if ok_gap else '❌'}")
    if best:
        print(f"    Best ratio >= 1.5:     {best['ratio']:.2f}  {'✅' if best['ratio']>=1.5 else '❌'}")
        print(f"    Best blocked PnL neg:  ${best['blocked_pnl']:+,.0f}  ✅")
    else:
        print(f"    Best ratio >= 1.5:     (no profitable threshold)  ❌")
        print(f"    Best blocked PnL neg:  (no profitable threshold)  ❌")
    
    passed = ok_is and ok_gap and ok_best
    print(f"\n  >>> {'PROCEED' if passed else 'FAIL'}")
    return {'name': name, 'is_auc': is_auc, 'oos_auc': oos_auc, 'gap': gap, 
            'best': best, 'passed': passed, 'model': model}


# Evaluate logreg
results = []
results.append(evaluate_model(
    "Logistic Regression (balanced, expanded features)",
    LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
    X_train_scaled, y_train, X_oos_scaled, y_oos, oos_clean
))

# Evaluate GBM
if HAS_GBM:
    results.append(evaluate_model(
        "Gradient Boosting (max_depth=2, n_est=50)",
        GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, 
                                    subsample=0.8, random_state=42),
        X_train_scaled, y_train, X_oos_scaled, y_oos, oos_clean
    ))

# ─── Final summary ───
print(f"\n{'='*70}")
print(f"  PHASE B v2 SUMMARY")
print(f"{'='*70}")
for r in results:
    status = '✅ PASS' if r['passed'] else '❌ FAIL'
    print(f"  {r['name']:50s}  {status}")
    print(f"    IS AUC: {r['is_auc']:.3f}  OOS AUC: {r['oos_auc']:.3f}  Gap: {r['gap']:+.3f}")
    if r['best']:
        print(f"    Best: top {r['best']['top_pct']}% blocks {r['best']['n_blocked']} trades "
              f"(stops={r['best']['n_stops']}, wins={r['best']['n_wins']}, ratio={r['best']['ratio']:.2f})")
print()

passing = [r for r in results if r['passed']]
if passing:
    best_model = max(passing, key=lambda r: r['best']['ratio'])
    print(f"  >>> BEST MODEL: {best_model['name']}")
    print(f"  >>> Save to: cache_backtester/choppy_throttle.joblib")
else:
    print(f"  >>> BOTH MODELS FAILED")
    print(f"  >>> Next: Option 6 (different target variable)")
