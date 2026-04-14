#!/usr/bin/env python3.11
"""
Phase A: Build training data for early-kill classifier.

For each trade in trades_v2.csv that survived to day 2:
  1. Load the symbol's price bars
  2. Compute price-action features at day 2 (3 trading days after entry)
  3. Pull entry-context features from trades CSV
  4. Label: eventually_stopped = (reason == 'stop')

Splits at 2022-01-01 (IS = 2017-2021, OOS = 2022-2026).
Saves to /tmp/early_kill_train.csv and /tmp/early_kill_oos.csv.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRADES = REPO / "trades_v2.csv"
PRICE_DIR = REPO / "cache_prices"
PRICE_FILE_TEMPLATE = "{sym}_3650d.csv"

print("="*70)
print("  PHASE A: Build early-kill training data")
print("="*70)
print()

# ─── Load trades ──────────────────────────────────────────────────────
print(f"[load] {TRADES}")
trades = pd.read_csv(TRADES)
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
trades['exit_date'] = pd.to_datetime(trades['exit_date'])
trades['hold_days'] = (trades['exit_date'] - trades['entry_date']).dt.days

print(f"  total trades: {len(trades)}")
print(f"  total stops:  {(trades['reason']=='stop').sum()}")
print()

# Filter to trades that survived at least 3 calendar days (need day 0, 1, 2 closes)
# We use 3 calendar days as a proxy; we'll require 3 actual trading days from price data
trades = trades[trades['hold_days'] >= 3].copy()
print(f"[filter] hold_days >= 3: {len(trades)} trades remaining")
print(f"         stops in this subset: {(trades['reason']=='stop').sum()}")
print()

# ─── Price loader (cached) ────────────────────────────────────────────
_price_cache = {}

def load_price(symbol):
    if symbol in _price_cache:
        return _price_cache[symbol]
    
    path = PRICE_DIR / PRICE_FILE_TEMPLATE.format(sym=symbol)
    if not path.exists():
        # Try alternate lengths
        for alt_len in ['1500d', '4000d', '6500d', '7300d', '2000d']:
            alt_path = PRICE_DIR / f"{symbol}_{alt_len}.csv"
            if alt_path.exists():
                path = alt_path
                break
        else:
            _price_cache[symbol] = None
            return None
    
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
        df = df.loc[~df.index.isna()].copy()
        df.index = df.index.tz_convert("UTC").tz_localize(None).normalize()
        df.columns = [c.lower() for c in df.columns]
        _price_cache[symbol] = df
        return df
    except Exception as e:
        _price_cache[symbol] = None
        return None

# Pre-flight check: how many symbols have price data
unique_syms = trades['symbol'].unique()
print(f"[pre-flight] checking price data for {len(unique_syms)} unique symbols...")
have_data = sum(1 for s in unique_syms if load_price(s) is not None)
print(f"  symbols with price data: {have_data} / {len(unique_syms)}")
missing = [s for s in unique_syms if load_price(s) is None]
if missing:
    print(f"  missing: {missing[:10]}{'...' if len(missing)>10 else ''}")
print()

# ─── Compute features for each trade ──────────────────────────────────
print("[compute] building feature matrix...")
records = []
skipped = {'no_price': 0, 'no_bars_at_entry': 0, 'insufficient_bars': 0, 'bad_data': 0}

for idx, trade in trades.iterrows():
    sym = trade['symbol']
    entry_date = trade['entry_date'].normalize()
    entry_price = trade['entry_price']
    
    px = load_price(sym)
    if px is None:
        skipped['no_price'] += 1
        continue
    
    # Get bars from entry_date forward (next 5 trading days is enough)
    bars = px.loc[px.index >= entry_date].head(6)
    if len(bars) == 0:
        skipped['no_bars_at_entry'] += 1
        continue
    if len(bars) < 3:
        skipped['insufficient_bars'] += 1
        continue
    
    try:
        # Day indices: 0 = entry day, 1 = first day after, 2 = second day after
        # So bars.iloc[0] = entry day, bars.iloc[1] = day 1, bars.iloc[2] = day 2
        d0 = bars.iloc[0]
        d1 = bars.iloc[1]
        d2 = bars.iloc[2]
        
        # Price-action features at close of day 2
        day1_return = (d1['close'] / entry_price) - 1.0
        day2_return = (d2['close'] / entry_price) - 1.0
        day2_minus_day1 = (d2['close'] / d1['close']) - 1.0
        
        # MAE/MFE over days 0-2 (using lows and highs)
        lows_0_2 = bars.iloc[0:3]['low']
        highs_0_2 = bars.iloc[0:3]['high']
        day2_mae = (lows_0_2.min() / entry_price) - 1.0   # max adverse (negative)
        day2_mfe = (highs_0_2.max() / entry_price) - 1.0  # max favorable (positive)
        
        # How much has price recovered from worst point by day 2 close
        worst_low = lows_0_2.min()
        day2_recovery = (d2['close'] - worst_low) / entry_price  # always >= 0
        
        # Realized vol of intraday moves over days 0-2
        intraday_ranges = (highs_0_2 - lows_0_2) / entry_price
        day2_intraday_vol = intraday_ranges.mean()
        
        records.append({
            'symbol': sym,
            'entry_date': entry_date,
            'exit_date': trade['exit_date'],
            'hold_days': trade['hold_days'],
            
            # price action features
            'day1_return': day1_return,
            'day2_return': day2_return,
            'day2_minus_day1': day2_minus_day1,
            'day2_mae': day2_mae,
            'day2_mfe': day2_mfe,
            'day2_recovery': day2_recovery,
            'day2_intraday_vol': day2_intraday_vol,
            
            # entry context features
            'ml_rank_pct': trade['ml_rank_pct'],
            'rule_score': trade['rule_score'],
            'combined_score': trade['combined_score'],
            'regime': trade['regime'],
            'ann_vol': trade['ann_vol'],
            'engine': trade['engine'],
            
            # outcome
            'reason': trade['reason'],
            'pnl': trade['pnl'],
            'stopped': 1 if trade['reason'] == 'stop' else 0,
        })
    except Exception as e:
        skipped['bad_data'] += 1
        continue

print(f"  computed: {len(records)}")
print(f"  skipped: {skipped}")
print()

if len(records) < 500:
    print(f"[ERROR] too few records ({len(records)}) — investigate before continuing")
    sys.exit(1)

df = pd.DataFrame(records)

# ─── Split IS/OOS ─────────────────────────────────────────────────────
df['entry_date'] = pd.to_datetime(df['entry_date'])
is_df = df[df['entry_date'] < '2022-01-01'].copy()
oos_df = df[df['entry_date'] >= '2022-01-01'].copy()

print(f"─── SPLIT ─────────────────────────────────────────────────")
print(f"  IS  (2017-2021): {len(is_df)} trades, {is_df['stopped'].sum()} stops ({is_df['stopped'].mean()*100:.1f}%)")
print(f"  OOS (2022-2026): {len(oos_df)} trades, {oos_df['stopped'].sum()} stops ({oos_df['stopped'].mean()*100:.1f}%)")
print()

# ─── Stops by regime ──────────────────────────────────────────────────
print("─── IS stops by regime ────────────────────────────────────")
for regime in ['TRENDING_BULL', 'CHOPPY', 'BEAR']:
    sub = is_df[is_df['regime'] == regime]
    n_stops = sub['stopped'].sum()
    print(f"  {regime:14s}  trades={len(sub):4d}  stops={n_stops:3d}  rate={n_stops/max(len(sub),1)*100:.1f}%")
print()

# ─── Univariate Cohen's d for each feature ────────────────────────────
print("─── UNIVARIATE: feature mean for STOPPED vs NOT STOPPED (IS) ───")
print(f"  {'feature':22s}  {'stop_mean':>10s}  {'no_mean':>10s}  {'cohen_d':>10s}")
print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")

price_features = ['day1_return', 'day2_return', 'day2_minus_day1', 'day2_mae', 
                  'day2_mfe', 'day2_recovery', 'day2_intraday_vol',
                  'ml_rank_pct', 'rule_score', 'ann_vol']

for f in price_features:
    s = is_df[is_df['stopped']==1][f].dropna()
    n = is_df[is_df['stopped']==0][f].dropna()
    if len(s) == 0 or len(n) == 0:
        continue
    pooled = np.sqrt(((len(s)-1)*s.var() + (len(n)-1)*n.var()) / (len(s)+len(n)-2))
    d = (s.mean() - n.mean()) / pooled if pooled > 0 else 0
    marker = ' ★' if abs(d) >= 0.5 else (' ·' if abs(d) >= 0.3 else '')
    print(f"  {f:22s}  {s.mean():+10.4f}  {n.mean():+10.4f}  {d:+10.3f}{marker}")
print()

# ─── Save outputs ──────────────────────────────────────────────────────
TRAIN_OUT = Path('/tmp/early_kill_train.csv')
OOS_OUT = Path('/tmp/early_kill_oos.csv')
is_df.to_csv(TRAIN_OUT, index=False)
oos_df.to_csv(OOS_OUT, index=False)
print(f"[save] {TRAIN_OUT}  ({len(is_df)} rows)")
print(f"[save] {OOS_OUT}  ({len(oos_df)} rows)")
print()

# ─── Decision criteria ─────────────────────────────────────────────────
print("="*70)
print("  PHASE A DECISION CRITERIA")
print("="*70)
ok_size = len(is_df) >= 500
ok_events = is_df['stopped'].sum() >= 80
ok_oos_size = len(oos_df) >= 200
biggest_d = max(
    abs((is_df[is_df['stopped']==1][f].mean() - is_df[is_df['stopped']==0][f].mean()) / 
        np.sqrt(((len(is_df[is_df['stopped']==1])-1)*is_df[is_df['stopped']==1][f].var() + 
                 (len(is_df[is_df['stopped']==0])-1)*is_df[is_df['stopped']==0][f].var()) / 
                (len(is_df)-2)))
    for f in price_features
    if is_df[f].notna().all() and is_df[is_df['stopped']==1][f].var() > 0
)

print(f"  IS rows >= 500:        {len(is_df)}     {'✅' if ok_size else '❌'}")
print(f"  IS stops >= 80:        {is_df['stopped'].sum()}     {'✅' if ok_events else '❌'}")
print(f"  OOS rows >= 200:       {len(oos_df)}     {'✅' if ok_oos_size else '❌'}")
print(f"  Max |Cohen's d| > 0.3: {biggest_d:.3f}  {'✅' if biggest_d >= 0.3 else '❌'}")
print()
if ok_size and ok_events and ok_oos_size and biggest_d >= 0.3:
    print("  >>> PROCEED to Phase B (model training)")
else:
    print("  >>> Investigate before training")
