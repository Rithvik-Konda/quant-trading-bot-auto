#!/usr/bin/env python3.11
"""
Phase A: Build training data for CHOPPY concentration throttle.

Loads cache_backtester/position_lifecycle.csv and trades_v2.csv,
joins them, filters to CHOPPY entries in the IS period (2017-2021),
and prints a summary so we can verify the data is sane before training.

Outputs the prepared training set to /tmp/choppy_throttle_train.csv
for use by phase B.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIFECYCLE = REPO / "cache_backtester" / "position_lifecycle.csv"
TRADES = REPO / "trades_v2.csv"
OUT = Path("/tmp/choppy_throttle_train.csv")

print("="*70)
print("  PHASE A: Build CHOPPY throttle training data")
print("="*70)
print()

# ─── Load lifecycle ──────────────────────────────────────────────────
print(f"[load] {LIFECYCLE}")
lc = pd.read_csv(LIFECYCLE)
lc['date'] = pd.to_datetime(lc['date'])
print(f"  rows: {len(lc)}")
print(f"  date range: {lc['date'].min().date()} to {lc['date'].max().date()}")

# ─── Filter to entries (days_held == 0) ──────────────────────────────
entries = lc[lc['days_held'] == 0].copy()
print(f"\n[filter] entries only (days_held=0): {len(entries)} rows")

# ─── Filter to CHOPPY regime ─────────────────────────────────────────
choppy = entries[entries['regime'] == 'CHOPPY'].copy()
print(f"[filter] CHOPPY regime only: {len(choppy)} rows")

# ─── Filter to IS period (2017-01-01 to 2022-01-01) ──────────────────
is_choppy = choppy[
    (choppy['date'] >= '2017-01-01') & 
    (choppy['date'] < '2022-01-01')
].copy()
print(f"[filter] IS period 2017-2021: {len(is_choppy)} rows")

oos_choppy = choppy[choppy['date'] >= '2022-01-01'].copy()
print(f"[info]  OOS held-out 2022-2026: {len(oos_choppy)} rows")
print()

# ─── Load trades for exit reasons ────────────────────────────────────
print(f"[load] {TRADES}")
trades = pd.read_csv(TRADES)
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
print(f"  rows: {len(trades)}")
print(f"  columns: {list(trades.columns)}")

# Show sample row to understand structure
print()
print("[trades sample]")
print(trades[['symbol','entry_date','exit_date','reason','pnl']].head(3).to_string(index=False))
print()

# ─── Join: lifecycle entries + trade exit reasons ────────────────────
# Match on (symbol, date == entry_date)
print("[join] matching lifecycle entries to trades on (symbol, date)")
join_cols_lc = ['date', 'symbol']
join_cols_tr = ['entry_date', 'symbol']

# Rename for join
trades_join = trades[['symbol', 'entry_date', 'reason', 'pnl']].rename(
    columns={'entry_date': 'date'}
)
trades_join['date'] = pd.to_datetime(trades_join['date'])

# IS join
is_train = is_choppy.merge(trades_join, on=['date', 'symbol'], how='left')
print(f"  IS rows after join: {len(is_train)}")
print(f"  IS rows with exit reason: {is_train['reason'].notna().sum()}")
print(f"  IS rows missing exit reason: {is_train['reason'].isna().sum()}")

# OOS join (for phase C validation later)
oos_eval = oos_choppy.merge(trades_join, on=['date', 'symbol'], how='left')
print(f"  OOS rows after join: {len(oos_eval)}")
print(f"  OOS rows with exit reason: {oos_eval['reason'].notna().sum()}")
print()

# ─── Investigate missing exit reasons ────────────────────────────────
missing = is_train[is_train['reason'].isna()]
if len(missing) > 0:
    print(f"[warn] {len(missing)} IS entries have no matched trade")
    print("  This usually means the position was still open at backtest end")
    print("  or the trade got closed by a different code path")
    print(f"  Sample missing: {missing[['date','symbol']].head(5).to_string(index=False)}")
    print()

# Drop unmatched rows for training
is_train = is_train.dropna(subset=['reason']).copy()
oos_eval = oos_eval.dropna(subset=['reason']).copy()
print(f"[clean] IS training rows after dropping unmatched: {len(is_train)}")
print(f"[clean] OOS eval rows after dropping unmatched: {len(oos_eval)}")
print()

# ─── Create binary label ─────────────────────────────────────────────
is_train['stopped'] = (is_train['reason'] == 'stop').astype(int)
oos_eval['stopped'] = (oos_eval['reason'] == 'stop').astype(int)

print("─── LABEL DISTRIBUTION ────────────────────────────────────────────")
print(f"IS  stop rate: {is_train['stopped'].mean()*100:.1f}% ({is_train['stopped'].sum()} of {len(is_train)})")
print(f"OOS stop rate: {oos_eval['stopped'].mean()*100:.1f}% ({oos_eval['stopped'].sum()} of {len(oos_eval)})")
print()

# ─── Exit reason breakdown ───────────────────────────────────────────
print("─── IS EXIT REASON BREAKDOWN ──────────────────────────────────────")
reason_counts = is_train['reason'].value_counts()
for reason, count in reason_counts.items():
    pct = count / len(is_train) * 100
    avg_pnl = is_train[is_train['reason'] == reason]['pnl'].mean()
    print(f"  {reason:20s}  {count:4d}  ({pct:5.1f}%)  avg_pnl=${avg_pnl:+8.0f}")
print()

# ─── Feature distribution check ──────────────────────────────────────
print("─── FEATURE DISTRIBUTIONS (IS, CHOPPY entries) ────────────────────")
features = ['portfolio_size', 'portfolio_corr', 'vix_now']
for f in features:
    if f not in is_train.columns:
        print(f"  [ERROR] feature '{f}' not in lifecycle CSV!")
        continue
    s = is_train[f].dropna()
    print(f"  {f:18s}  n={len(s):4d}  mean={s.mean():7.3f}  std={s.std():7.3f}  "
          f"min={s.min():7.3f}  max={s.max():7.3f}")
print()

# ─── Univariate signal check ─────────────────────────────────────────
print("─── UNIVARIATE: feature mean for STOPPED vs NOT STOPPED ───────────")
print(f"  {'feature':18s}  {'stopped':>10s}  {'not_stopped':>12s}  {'cohen_d':>8s}")
print(f"  {'-'*18}  {'-'*10}  {'-'*12}  {'-'*8}")
for f in features:
    if f not in is_train.columns:
        continue
    s_stop = is_train[is_train['stopped'] == 1][f].dropna()
    s_no   = is_train[is_train['stopped'] == 0][f].dropna()
    if len(s_stop) == 0 or len(s_no) == 0:
        print(f"  {f:18s}  insufficient data")
        continue
    pooled_std = np.sqrt(((len(s_stop)-1)*s_stop.var() + (len(s_no)-1)*s_no.var()) / (len(s_stop)+len(s_no)-2))
    cohen_d = (s_stop.mean() - s_no.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"  {f:18s}  {s_stop.mean():10.3f}  {s_no.mean():12.3f}  {cohen_d:+8.3f}")
print()

# ─── Save training data for Phase B ──────────────────────────────────
keep_cols = ['date', 'symbol', 'portfolio_size', 'portfolio_corr', 'vix_now', 'stopped', 'reason', 'pnl']
is_train[keep_cols].to_csv(OUT, index=False)
print(f"[save] IS training data -> {OUT}")
print(f"       {len(is_train)} rows, {len(keep_cols)} columns")

# Also save OOS for Phase C
oos_out = Path("/tmp/choppy_throttle_oos.csv")
oos_eval[keep_cols].to_csv(oos_out, index=False)
print(f"[save] OOS eval data -> {oos_out}")
print(f"       {len(oos_eval)} rows")
print()

print("="*70)
print("  PHASE A COMPLETE")
print("="*70)
print()
print("Decision criteria for proceeding to Phase B:")
print("  - IS training set has at least 200 rows")
print("  - IS stop rate is between 15% and 50% (not too rare or too common)")
print("  - At least one feature has |Cohen's d| > 0.20")
print("  - OOS eval set is similar size for validation")
