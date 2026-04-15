#!/usr/bin/env python3.11
"""
Step 5 diagnostic: why does BEAR meanrev outperform CHOPPY meanrev?

Splits meanrev trades by entry-time regime and compares every available
dimension to find structural differences.
"""
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRADES = REPO / "trades_v2.csv"

print("="*72)
print("  STEP 5 DIAGNOSTIC: CHOPPY meanrev vs BEAR meanrev")
print("="*72)
print()

tr = pd.read_csv(TRADES)
tr['entry_date'] = pd.to_datetime(tr['entry_date'])
tr['exit_date'] = pd.to_datetime(tr['exit_date'])
tr['hold_days'] = (tr['exit_date'] - tr['entry_date']).dt.days

# Filter to meanrev only
mr = tr[tr['engine'] == 'meanrev'].copy()
print(f"Total meanrev trades: {len(mr)}")
print(f"  by regime:")
for r, count in mr['regime'].value_counts().items():
    print(f"    {r:14s}  {count}")
print()

# Split CHOPPY vs BEAR
choppy = mr[mr['regime'] == 'CHOPPY'].copy()
bear = mr[mr['regime'] == 'BEAR'].copy()

print(f"CHOPPY meanrev: {len(choppy)} trades")
print(f"BEAR   meanrev: {len(bear)} trades")
print()

# ─── HEADLINE PERFORMANCE ──────────────────────────────────────────────
print("─── HEADLINE PERFORMANCE ──────────────────────────────────────────")
print(f"{'metric':24s}  {'CHOPPY':>14s}  {'BEAR':>14s}  {'diff':>14s}")
print(f"{'-'*24}  {'-'*14}  {'-'*14}  {'-'*14}")

def stat(name, c_val, b_val, fmt="{:.3f}"):
    diff = b_val - c_val
    fc = fmt.format(c_val)
    fb = fmt.format(b_val)
    fd = fmt.format(diff)
    if diff > 0:
        fd = "+" + fd
    print(f"{name:24s}  {fc:>14s}  {fb:>14s}  {fd:>14s}")

stat("trades", len(choppy), len(bear), "{:.0f}")
stat("win_rate", (choppy['pnl']>0).mean(), (bear['pnl']>0).mean(), "{:.1%}")
stat("avg_pnl", choppy['pnl'].mean(), bear['pnl'].mean(), "${:,.0f}")
stat("median_pnl", choppy['pnl'].median(), bear['pnl'].median(), "${:,.0f}")
stat("total_pnl", choppy['pnl'].sum(), bear['pnl'].sum(), "${:,.0f}")
stat("std_pnl", choppy['pnl'].std(), bear['pnl'].std(), "${:,.0f}")
print()

# ─── EXIT REASON BREAKDOWN ────────────────────────────────────────────
print("─── EXIT REASON BREAKDOWN ────────────────────────────────────────")
print(f"{'reason':18s}  {'CHOPPY n':>10s}  {'CHOPPY %':>10s}  {'BEAR n':>10s}  {'BEAR %':>10s}")
print(f"{'-'*18}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
all_reasons = set(choppy['reason'].unique()) | set(bear['reason'].unique())
for reason in sorted(all_reasons):
    c_n = (choppy['reason'] == reason).sum()
    b_n = (bear['reason'] == reason).sum()
    c_p = c_n / max(len(choppy), 1) * 100
    b_p = b_n / max(len(bear), 1) * 100
    print(f"{reason:18s}  {c_n:>10d}  {c_p:>9.1f}%  {b_n:>10d}  {b_p:>9.1f}%")
print()

# ─── EXIT REASON × AVERAGE PNL ────────────────────────────────────────
print("─── PNL BY EXIT REASON ───────────────────────────────────────────")
print(f"{'reason':18s}  {'CHOPPY avg_pnl':>16s}  {'BEAR avg_pnl':>16s}")
print(f"{'-'*18}  {'-'*16}  {'-'*16}")
for reason in sorted(all_reasons):
    c_pnl = choppy[choppy['reason'] == reason]['pnl'].mean()
    b_pnl = bear[bear['reason'] == reason]['pnl'].mean()
    c_str = f"${c_pnl:+,.0f}" if not pd.isna(c_pnl) else "n/a"
    b_str = f"${b_pnl:+,.0f}" if not pd.isna(b_pnl) else "n/a"
    print(f"{reason:18s}  {c_str:>16s}  {b_str:>16s}")
print()

# ─── HOLD DAYS DISTRIBUTION ───────────────────────────────────────────
print("─── HOLD DAYS DISTRIBUTION ───────────────────────────────────────")
for label, df in [("CHOPPY", choppy), ("BEAR", bear)]:
    h = df['hold_days']
    print(f"  {label:6s}  mean={h.mean():.1f}  median={h.median():.0f}  "
          f"p25={h.quantile(0.25):.0f}  p75={h.quantile(0.75):.0f}  "
          f"min={h.min():.0f}  max={h.max():.0f}")
print()

# ─── ENTRY-TIME FEATURE COMPARISON ────────────────────────────────────
print("─── ENTRY FEATURE DISTRIBUTIONS ───────────────────────────────────")
print(f"{'feature':18s}  {'CHOPPY mean':>14s}  {'BEAR mean':>14s}  {'cohen_d':>10s}")
print(f"{'-'*18}  {'-'*14}  {'-'*14}  {'-'*10}")

for feat in ['ml_rank_pct', 'rule_score', 'combined_score', 'ann_vol']:
    if feat not in mr.columns:
        continue
    c_vals = choppy[feat].dropna()
    b_vals = bear[feat].dropna()
    if len(c_vals) == 0 or len(b_vals) == 0:
        continue
    pooled = np.sqrt(((len(c_vals)-1)*c_vals.var() + (len(b_vals)-1)*b_vals.var()) / 
                     (len(c_vals)+len(b_vals)-2))
    d = (b_vals.mean() - c_vals.mean()) / pooled if pooled > 0 else 0
    print(f"{feat:18s}  {c_vals.mean():>14.4f}  {b_vals.mean():>14.4f}  {d:>+10.3f}")
print()
print("(cohen_d > 0 = BEAR has higher value, < 0 = CHOPPY has higher value)")
print()

# ─── PER-TRADE RETURN BREAKDOWN ───────────────────────────────────────
print("─── PER-TRADE RETURN PERCENTAGE ──────────────────────────────────")
mr['return_pct'] = (mr['exit_price'] - mr['entry_price']) / mr['entry_price']
choppy_ret = mr[mr['regime'] == 'CHOPPY']['return_pct']
bear_ret = mr[mr['regime'] == 'BEAR']['return_pct']

for label, r in [("CHOPPY", choppy_ret), ("BEAR", bear_ret)]:
    print(f"  {label:6s}  mean_ret={r.mean()*100:+.2f}%  median={r.median()*100:+.2f}%  "
          f"std={r.std()*100:.2f}%  min={r.min()*100:+.2f}%  max={r.max()*100:+.2f}%")
print()

# ─── WINNERS VS LOSERS BREAKDOWN ──────────────────────────────────────
print("─── WINNERS VS LOSERS ─────────────────────────────────────────────")
for label, df in [("CHOPPY", choppy), ("BEAR", bear)]:
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    print(f"  {label}:")
    print(f"    winners: n={len(wins):4d}  avg=${wins['pnl'].mean():+,.0f}  "
          f"median=${wins['pnl'].median():+,.0f}")
    print(f"    losers:  n={len(losses):4d}  avg=${losses['pnl'].mean():+,.0f}  "
          f"median=${losses['pnl'].median():+,.0f}")
    if len(wins) > 0 and len(losses) > 0:
        rr = abs(wins['pnl'].mean() / losses['pnl'].mean())
        print(f"    win/loss ratio: {rr:.2f}")
print()

# ─── YEARLY BREAKDOWN ──────────────────────────────────────────────────
print("─── YEARLY PERFORMANCE ────────────────────────────────────────────")
mr['year'] = mr['entry_date'].dt.year
print(f"{'year':6s}  {'CHOPPY n':>10s}  {'CHOPPY pnl':>12s}  {'BEAR n':>8s}  {'BEAR pnl':>12s}")
print(f"{'-'*6}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*12}")
for year in sorted(mr['year'].unique()):
    yc = mr[(mr['year'] == year) & (mr['regime'] == 'CHOPPY')]
    yb = mr[(mr['year'] == year) & (mr['regime'] == 'BEAR')]
    print(f"{year:6.0f}  {len(yc):>10d}  ${yc['pnl'].sum():>+11,.0f}  "
          f"{len(yb):>8d}  ${yb['pnl'].sum():>+11,.0f}")
print()

# ─── DIAGNOSTIC SUMMARY ────────────────────────────────────────────────
print("="*72)
print("  DIAGNOSTIC SUMMARY")
print("="*72)
print()

choppy_wr = (choppy['pnl']>0).mean()
bear_wr = (bear['pnl']>0).mean()
choppy_avg = choppy['pnl'].mean()
bear_avg = bear['pnl'].mean()

print(f"CHOPPY meanrev: {len(choppy)} trades, WR {choppy_wr*100:.1f}%, avg ${choppy_avg:+,.0f}")
print(f"BEAR   meanrev: {len(bear)} trades, WR {bear_wr*100:.1f}%, avg ${bear_avg:+,.0f}")
print()

# Detect dominant cause
choppy_stop_rate = (choppy['reason']=='stop').mean()
bear_stop_rate = (bear['reason']=='stop').mean()
choppy_tp_rate = (choppy['reason']=='take_profit').mean()
bear_tp_rate = (bear['reason']=='take_profit').mean()

print("Likely root causes of underperformance:")
print()

if choppy_stop_rate > bear_stop_rate * 1.3:
    print(f"  [HIGH] CHOPPY stop rate {choppy_stop_rate*100:.1f}% vs BEAR {bear_stop_rate*100:.1f}%")
    print(f"         CHOPPY meanrev gets stopped MORE — entry filter too loose")
elif choppy_stop_rate < bear_stop_rate * 0.8:
    print(f"  [LOW]  CHOPPY stop rate is similar/lower than BEAR — not a stop issue")

if bear_tp_rate > choppy_tp_rate * 1.3:
    print(f"  [HIGH] BEAR take_profit rate {bear_tp_rate*100:.1f}% vs CHOPPY {choppy_tp_rate*100:.1f}%")
    print(f"         BEAR hits take-profit MORE — CHOPPY has weaker upside")

if abs(choppy['hold_days'].mean() - bear['hold_days'].mean()) > 2:
    print(f"  [HOLD] Hold days differ: CHOPPY {choppy['hold_days'].mean():.1f} vs BEAR {bear['hold_days'].mean():.1f}")

print()
print("Next: read the breakdown above and decide if there's a clean fix.")
