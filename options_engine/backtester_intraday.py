"""
0DTE Options Engine — Intraday Backtester
Entry: 10:30 AM using real chain Greeks
Exit: 3:00 PM or 50% profit or 2x loss
"""
import pandas as pd
import numpy as np
import os
import joblib
import time
from fetch_intraday import get_entry_snapshot, get_exit_snapshot, CACHE_DIR
from range_predictor import build_model_features

STARTING_CAPITAL = 50_000
MAX_RISK = 500
PROFIT_TARGET = 0.50
STOP_MULTIPLE = 2.0
MIN_OTM_PCT = 0.003

def get_spx(snapshot):
    prices = snapshot['underlying_price']
    prices = prices[prices > 100]
    return float(prices.median()) if not prices.empty else None

def find_nearest_strike(snapshot, right, target_strike):
    sub = snapshot[snapshot['right'] == right.upper()].copy()
    if sub.empty:
        return None
    sub = sub[sub['bid'] > 0.05]
    if sub.empty:
        return None
    idx = (sub['strike'] - target_strike).abs().idxmin()
    return sub.loc[idx]

def get_condor_at_entry(entry_snap, spx, pred_range):
    half = pred_range / 100 / 2
    buffer = 0.002
    otm = max(MIN_OTM_PCT, half + buffer)
    
    call_short_target = spx * (1 + otm)
    put_short_target  = spx * (1 - otm)
    
    cs = find_nearest_strike(entry_snap, 'CALL', call_short_target)
    ps = find_nearest_strike(entry_snap, 'PUT',  put_short_target)
    
    if cs is None or ps is None:
        return None
    
    cs_strike = float(cs['strike'])
    ps_strike = float(ps['strike'])
    cl_target = cs_strike + 5
    pl_target = ps_strike - 5
    
    cl = find_nearest_strike(entry_snap, 'CALL', cl_target)
    pl = find_nearest_strike(entry_snap, 'PUT',  pl_target)
    
    if cl is None or pl is None:
        return None
    
    def mid(row): return (float(row['bid']) + float(row['ask'])) / 2
    
    credit = mid(cs) + mid(ps) - mid(cl) - mid(pl)
    if credit <= 0.10:
        return None
    
    wing = abs(float(cs['strike']) - float(cl['strike']))
    max_loss = wing * 100 - credit * 100
    if max_loss <= 0:
        return None
    
    return {
        'credit': credit,
        'max_loss': max_loss,
        'ps_strike': ps_strike,
        'cs_strike': cs_strike,
        'pl_strike': float(pl['strike']),
        'cl_strike': float(cl['strike']),
        'ps_iv': float(ps['implied_vol']),
        'cs_iv': float(cs['implied_vol']),
        'spx_entry': spx,
    }

def get_condor_value_at_exit(exit_snap, trade):
    def mid_at(right, strike):
        sub = exit_snap[exit_snap['right'] == right.upper()]
        if sub.empty: return None
        idx = (sub['strike'] - strike).abs().idxmin()
        row = sub.loc[idx]
        if abs(float(row['strike']) - strike) > 25: return None
        return (float(row['bid']) + float(row['ask'])) / 2
    
    cs_val = mid_at('CALL', trade['cs_strike'])
    ps_val = mid_at('PUT',  trade['ps_strike'])
    cl_val = mid_at('CALL', trade['cl_strike'])
    pl_val = mid_at('PUT',  trade['pl_strike'])
    
    if any(v is None for v in [cs_val, ps_val, cl_val, pl_val]):
        # Options expired worthless — full credit
        return 0.0
    
    return cs_val + ps_val - cl_val - pl_val

def run_backtest():
    df = pd.read_parquet(os.path.join(CACHE_DIR, "daily_features.parquet"))
    df = build_model_features(df).dropna(subset=['day_range_pct','vix','realized_vol_5d']).reset_index(drop=True)
    
    split = int(len(df) * 0.6)
    test_df = df.iloc[split:].reset_index(drop=True)
    
    bundle = joblib.load(os.path.join(CACHE_DIR, "range_predictor.joblib"))
    model, feat_cols = bundle['model'], bundle['features']
    
    print(f"Test: {test_df['date'].iloc[0].date()} to {test_df['date'].iloc[-1].date()} ({len(test_df)} days)")
    
    capital = STARTING_CAPITAL
    trades = []
    skipped = 0
    
    for i, row in test_df.iterrows():
        date_str = row['date'].strftime("%Y%m%d")
        
        if row['vix'] > 35:
            skipped += 1
            continue
        
        entry_snap = get_entry_snapshot(date_str)
        if entry_snap.empty:
            skipped += 1
            continue
        
        spx = get_spx(entry_snap)
        if not spx:
            skipped += 1
            continue
        
        feat_dict = {f: row.get(f, 0) for f in feat_cols}
        X = pd.DataFrame([feat_dict])[feat_cols].fillna(0)
        pred_range = float(model.predict(X)[0])
        
        trade = get_condor_at_entry(entry_snap, spx, pred_range)
        if trade is None:
            skipped += 1
            continue
        
        n_contracts = max(1, int(MAX_RISK / trade['max_loss']))
        
        exit_snap = get_exit_snapshot(date_str)
        if exit_snap.empty:
            exit_value = 0.0
        else:
            exit_value = get_condor_value_at_exit(exit_snap, trade)
            if exit_value is None:
                exit_value = 0.0
        
        pnl_per = (trade['credit'] - exit_value) * 100
        
        # Apply profit target and stop
        max_profit = trade['credit'] * 100
        max_loss_actual = trade['max_loss']
        
        if pnl_per >= max_profit * PROFIT_TARGET:
            pnl_per = max_profit * PROFIT_TARGET
            exit_reason = 'profit_target'
        elif pnl_per < -max_loss_actual * STOP_MULTIPLE:
            pnl_per = -max_loss_actual
            exit_reason = 'stop'
        else:
            exit_reason = 'time_exit'
        
        pnl = pnl_per * n_contracts
        capital += pnl
        
        trades.append({
            'date': row['date'],
            'spx': spx,
            'pred_range': pred_range,
            'actual_range': row['day_range_pct'],
            'credit': trade['credit'],
            'n_contracts': n_contracts,
            'pnl': pnl,
            'capital': capital,
            'vix': row['vix'],
            'exit_reason': exit_reason,
        })
        
        if i % 20 == 0:
            print(f"  {date_str}: spx={spx:.0f} pred={pred_range:.2f}% credit=${trade['credit']:.2f} pnl=${pnl:.0f} cap=${capital:,.0f}")
        
        time.sleep(0.1)
    
    results = pd.DataFrame(trades)
    if results.empty:
        print(f"No trades. Skipped: {skipped}/{len(test_df)}")
        return
    
    n_years = (results['date'].iloc[-1] - results['date'].iloc[0]).days / 365
    cagr = ((capital / STARTING_CAPITAL) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    win_rate = (results['pnl'] > 0).mean() * 100
    
    print(f"\n{'='*55}")
    print(f"  RESULTS: {len(results)} trades, {skipped} skipped")
    print(f"  CAGR: {cagr:.1f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg P&L/trade: ${results['pnl'].mean():.0f}")
    print(f"  Max loss day: ${results['pnl'].min():.0f}")
    print(f"  Final Capital: ${capital:,.0f}")
    print(f"\n  Exit reasons:")
    print(results['exit_reason'].value_counts().to_string())
    print(f"{'='*55}")
    
    results.to_parquet(os.path.join(CACHE_DIR, "backtest_results.parquet"))
    return results

if __name__ == "__main__":
    run_backtest()
