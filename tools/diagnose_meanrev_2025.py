#!/usr/bin/env python3.11
"""
Forensic investigation of 2025 CHOPPY meanrev disaster.

19 trades produced -$27,703 in 2025 vs roughly breakeven in all prior years.
Find what was different about these specific trades.
"""
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRADES = REPO / "trades_v2.csv"

print("="*78)
print("  2025 CHOPPY MEANREV FORENSICS")
print("="*78)
print()

tr = pd.read_csv(TRADES)
tr['entry_date'] = pd.to_datetime(tr['entry_date'])
tr['exit_date'] = pd.to_datetime(tr['exit_date'])
tr['hold_days'] = (tr['exit_date'] - tr['entry_date']).dt.days
tr['return_pct'] = (tr['exit_price'] - tr['entry_price']) / tr['entry_price']
tr['year'] = tr['entry_date'].dt.year
tr['month'] = tr['entry_date'].dt.to_period('M').astype(str)

mr_choppy = tr[(tr['engine'] == 'meanrev') & (tr['regime'] == 'CHOPPY')].copy()

# Split: 2025 vs prior
y2025 = mr_choppy[mr_choppy['year'] == 2025].copy()
prior = mr_choppy[mr_choppy['year'] < 2025].copy()

print(f"2025 CHOPPY meanrev: {len(y2025)} trades, total PnL ${y2025['pnl'].sum():+,.0f}")
print(f"Prior years (2017-2024): {len(prior)} trades, total PnL ${prior['pnl'].sum():+,.0f}")
print()

# ─── Every 2025 trade ────────────────────────────────────────────────
print("─── ALL 19 TRADES IN 2025 (sorted by PnL) ───────────────────────────")
print(f"{'symbol':>8s}  {'entry':>10s}  {'exit':>10s}  {'hold':>4s}  "
      f"{'ent_px':>8s}  {'exit_px':>8s}  {'ret%':>7s}  {'pnl':>10s}  {'reason':>14s}")
print("-" * 100)
y2025_sorted = y2025.sort_values('pnl')
for _, r in y2025_sorted.iterrows():
    print(f"{r['symbol']:>8s}  {r['entry_date'].strftime('%Y-%m-%d')}  "
          f"{r['exit_date'].strftime('%Y-%m-%d')}  {int(r['hold_days']):>4d}  "
          f"{r['entry_price']:>8.2f}  {r['exit_price']:>8.2f}  "
          f"{r['return_pct']*100:>+6.2f}%  ${r['pnl']:>+9,.0f}  {r['reason']:>14s}")
print()

# ─── Time clustering ─────────────────────────────────────────────────
print("─── 2025 ENTRIES BY MONTH ───────────────────────────────────────────")
monthly = y2025.groupby('month').agg(
    n=('pnl', 'count'),
    pnl=('pnl', 'sum'),
    avg_ret=('return_pct', 'mean'),
).round(4)
print(monthly.to_string())
print()

# ─── Symbol concentration ────────────────────────────────────────────
print("─── 2025 SYMBOL FREQUENCY ───────────────────────────────────────────")
sym_counts = y2025.groupby('symbol').agg(
    trades=('pnl', 'count'),
    total_pnl=('pnl', 'sum'),
    avg_ret=('return_pct', 'mean'),
).sort_values('total_pnl').round(4)
print(sym_counts.to_string())
print()

# ─── Compare 2025 vs prior years on every dimension ─────────────────
print("─── 2025 vs PRIOR YEARS COMPARISON ─────────────────────────────────")
print(f"{'metric':22s}  {'2025':>14s}  {'prior':>14s}  {'change':>14s}")
print(f"{'-'*22}  {'-'*14}  {'-'*14}  {'-'*14}")

def row(name, a, b, fmt="{:.3f}"):
    fa = fmt.format(a) if not pd.isna(a) else "n/a"
    fb = fmt.format(b) if not pd.isna(b) else "n/a"
    diff = a - b if not pd.isna(a) and not pd.isna(b) else None
    fd = ("+" + fmt.format(diff)) if diff is not None and diff > 0 else (fmt.format(diff) if diff is not None else "n/a")
    print(f"{name:22s}  {fa:>14s}  {fb:>14s}  {fd:>14s}")

row("trades", len(y2025), len(prior), "{:.0f}")
row("win_rate", (y2025['pnl']>0).mean(), (prior['pnl']>0).mean(), "{:.1%}")
row("avg_pnl", y2025['pnl'].mean(), prior['pnl'].mean(), "${:,.0f}")
row("median_pnl", y2025['pnl'].median(), prior['pnl'].median(), "${:,.0f}")
row("avg_return_pct", y2025['return_pct'].mean()*100, prior['return_pct'].mean()*100, "{:+.2f}%")
row("median_return_pct", y2025['return_pct'].median()*100, prior['return_pct'].median()*100, "{:+.2f}%")
row("std_return_pct", y2025['return_pct'].std()*100, prior['return_pct'].std()*100, "{:.2f}%")
row("max_loss_pct", y2025['return_pct'].min()*100, prior['return_pct'].min()*100, "{:+.2f}%")
row("max_gain_pct", y2025['return_pct'].max()*100, prior['return_pct'].max()*100, "{:+.2f}%")
row("avg_hold_days", y2025['hold_days'].mean(), prior['hold_days'].mean(), "{:.1f}")
row("ml_rank_pct mean", y2025['ml_rank_pct'].mean(), prior['ml_rank_pct'].mean(), "{:.3f}")
row("ann_vol mean", y2025['ann_vol'].mean(), prior['ann_vol'].mean(), "{:.3f}")
print()

# ─── Position size proxy (qty * entry_price) ─────────────────────────
y2025['notional'] = y2025['qty'] * y2025['entry_price']
prior['notional'] = prior['qty'] * prior['entry_price']
print("─── POSITION SIZE PROXY (qty * entry_price) ─────────────────────────")
row("avg notional", y2025['notional'].mean(), prior['notional'].mean(), "${:,.0f}")
row("median notional", y2025['notional'].median(), prior['notional'].median(), "${:,.0f}")
row("max notional", y2025['notional'].max(), prior['notional'].max(), "${:,.0f}")
print()

# ─── Loser size in 2025 specifically ─────────────────────────────────
print("─── 2025 LOSERS DEEP DIVE ───────────────────────────────────────────")
losers_2025 = y2025[y2025['pnl'] < 0]
print(f"  2025 losers: {len(losers_2025)} trades, total PnL ${losers_2025['pnl'].sum():+,.0f}")
print(f"  avg loss: ${losers_2025['pnl'].mean():+,.0f}")
print(f"  worst loss: ${losers_2025['pnl'].min():+,.0f}")
print(f"  avg loser return: {losers_2025['return_pct'].mean()*100:+.2f}%")
print(f"  avg loser notional: ${losers_2025['notional'].mean():,.0f}")
print()
print("  Largest 5 losses:")
worst = losers_2025.nsmallest(5, 'pnl')
for _, r in worst.iterrows():
    print(f"    {r['symbol']:>6s}  {r['entry_date'].strftime('%Y-%m-%d')}  "
          f"ret {r['return_pct']*100:+.2f}%  notional ${r['notional']:,.0f}  pnl ${r['pnl']:+,.0f}  {r['reason']}")
print()

# ─── Quarter breakdown for 2025 ──────────────────────────────────────
print("─── 2025 BY QUARTER ─────────────────────────────────────────────────")
y2025['quarter'] = y2025['entry_date'].dt.to_period('Q').astype(str)
qb = y2025.groupby('quarter').agg(
    n=('pnl', 'count'),
    pnl=('pnl', 'sum'),
    wr=('pnl', lambda x: (x>0).mean()),
).round(3)
print(qb.to_string())
print()

# ─── Same for momentum trades in 2025 to compare ─────────────────────
mom_2025 = tr[(tr['engine'] == 'momentum') & (tr['year'] == 2025)]
print("─── COMPARISON: 2025 momentum trades (same year, different engine) ──")
print(f"  trades: {len(mom_2025)}")
print(f"  total PnL: ${mom_2025['pnl'].sum():+,.0f}")
print(f"  win rate: {(mom_2025['pnl']>0).mean()*100:.1f}%")
print(f"  avg return: {((mom_2025['exit_price']-mom_2025['entry_price'])/mom_2025['entry_price']).mean()*100:+.2f}%")
print()

# ─── Regime context check ────────────────────────────────────────────
print("="*78)
print("  KEY QUESTIONS TO ANSWER FROM THIS DATA")
print("="*78)
print()
print("1. Is the loss concentrated in specific months? (clustering = event-driven)")
print("2. Is the loss concentrated in specific symbols? (idiosyncratic = bad picks)")
print("3. Are 2025 positions LARGER than prior years? (sizing change)")
print("4. Are 2025 trades held LONGER? (regime confusion)")
print("5. Did momentum also fail in 2025, or just meanrev? (system-wide vs engine-specific)")
print("6. Are 2025 losers from sectors that mean-reverted differently?")
