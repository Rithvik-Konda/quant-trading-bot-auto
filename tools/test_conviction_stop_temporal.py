#!/usr/bin/env python3.11
"""
Temporal validation of the conviction-stop pattern.

Tests whether ml_rank_pct positively correlates with stop probability
on increasingly restrictive time windows. If the pattern holds on
2017-2021 alone (original IS period, completely held out from 2025),
then a fix to the meanrev ml_boost is justifiable without backfitting.
"""
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRADES = REPO / "trades_v2.csv"

print("="*72)
print("  TEMPORAL VALIDATION: conviction-stop pattern across time windows")
print("="*72)
print()

tr = pd.read_csv(TRADES)
tr['entry_date'] = pd.to_datetime(tr['entry_date'])
tr['stopped'] = (tr['reason'] == 'stop').astype(int)
tr['year'] = tr['entry_date'].dt.year

def cohen_d(stopped_vals, not_stopped_vals):
    s = stopped_vals.dropna()
    n = not_stopped_vals.dropna()
    if len(s) < 5 or len(n) < 5:
        return None
    pooled = np.sqrt(((len(s)-1)*s.var() + (len(n)-1)*n.var()) / (len(s)+len(n)-2))
    return (s.mean() - n.mean()) / pooled if pooled > 0 else 0

def analyze_window(label, df):
    n_total = len(df)
    n_stops = df['stopped'].sum()
    
    print(f"─── {label} ──────────────────────────────────────────")
    print(f"  trades: {n_total}, stops: {n_stops} ({n_stops/max(n_total,1)*100:.1f}%)")
    print()
    print(f"  {'feature':22s}  {'stop_mean':>10s}  {'no_mean':>10s}  {'cohen_d':>10s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for feat in ['ml_rank_pct', 'rule_score', 'combined_score']:
        if feat not in df.columns:
            continue
        s_vals = df[df['stopped']==1][feat]
        n_vals = df[df['stopped']==0][feat]
        d = cohen_d(s_vals, n_vals)
        if d is None:
            print(f"  {feat:22s}  insufficient data")
            continue
        s_mean = s_vals.mean()
        n_mean = n_vals.mean()
        marker = ' ★' if abs(d) >= 0.5 else (' ·' if abs(d) >= 0.2 else '')
        print(f"  {feat:22s}  {s_mean:>10.4f}  {n_mean:>10.4f}  {d:>+10.3f}{marker}")
    print()

# Window 1: 2017-2021 only (the original IS period)
w1 = tr[tr['year'] <= 2021].copy()
analyze_window("WINDOW 1: 2017-2021 only (held out from 2022+)", w1)

# Window 2: 2017-2024 (excludes the disaster year)
w2 = tr[tr['year'] <= 2024].copy()
analyze_window("WINDOW 2: 2017-2024 (excludes 2025 disaster)", w2)

# Window 3: 2017-2026 (full sample, what we already saw)
w3 = tr.copy()
analyze_window("WINDOW 3: 2017-2026 (full sample)", w3)

# Also check meanrev specifically
print("="*72)
print("  MEANREV-ONLY ANALYSIS (the engine we're considering fixing)")
print("="*72)
print()

mr = tr[tr['engine'] == 'meanrev'].copy()
mr1 = mr[mr['year'] <= 2021]
mr2 = mr[mr['year'] <= 2024]
mr3 = mr

analyze_window("MEANREV 2017-2021", mr1)
analyze_window("MEANREV 2017-2024", mr2)
analyze_window("MEANREV 2017-2026", mr3)

# ─── DECISION ───────────────────────────────────────────────────────
print("="*72)
print("  DECISION")
print("="*72)
print()

# Test on 2017-2021 ALL trades
all_2017_2021 = tr[tr['year'] <= 2021].copy()
ml_d_all = cohen_d(
    all_2017_2021[all_2017_2021['stopped']==1]['ml_rank_pct'],
    all_2017_2021[all_2017_2021['stopped']==0]['ml_rank_pct']
)
rule_d_all = cohen_d(
    all_2017_2021[all_2017_2021['stopped']==1]['rule_score'],
    all_2017_2021[all_2017_2021['stopped']==0]['rule_score']
)

# Test on 2017-2021 MEANREV only
mr_2017_2021 = mr[mr['year'] <= 2021].copy()
ml_d_mr = cohen_d(
    mr_2017_2021[mr_2017_2021['stopped']==1]['ml_rank_pct'],
    mr_2017_2021[mr_2017_2021['stopped']==0]['ml_rank_pct']
) if (mr_2017_2021['stopped'].sum() >= 5) else None

print(f"On 2017-2021 ALL trades:")
print(f"  ml_rank_pct conviction-stop d = {ml_d_all:+.3f}" if ml_d_all is not None else "  ml_rank_pct: insufficient data")
print(f"  rule_score  conviction-stop d = {rule_d_all:+.3f}" if rule_d_all is not None else "  rule_score: insufficient data")
print()
if mr_2017_2021['stopped'].sum() >= 5:
    print(f"On 2017-2021 MEANREV only ({len(mr_2017_2021)} trades, {mr_2017_2021['stopped'].sum()} stops):")
    print(f"  ml_rank_pct conviction-stop d = {ml_d_mr:+.3f}" if ml_d_mr is not None else "  insufficient data")
else:
    print(f"On 2017-2021 MEANREV only: only {mr_2017_2021['stopped'].sum()} stops — too few for reliable d")
print()

print("Decision logic:")
print("  - If d >= +0.20 on 2017-2021 ALL trades: ml_boost fix is JUSTIFIED")
print("    (the conviction-stop pattern exists in held-out data)")
print("  - If d < +0.20 on 2017-2021 ALL trades: no clean fix, ship null result")
print()

if ml_d_all is not None and ml_d_all >= 0.20:
    print(f"  >>> JUSTIFIED ({ml_d_all:+.3f} >= +0.20)")
    print(f"  >>> Mechanism fix to meanrev ml_boost is non-backfit")
elif ml_d_all is not None:
    print(f"  >>> NOT JUSTIFIED ({ml_d_all:+.3f} < +0.20)")
    print(f"  >>> Conviction-stop pattern not strong enough on 2017-2021 alone")
    print(f"  >>> Ship Step 5 as null result")
