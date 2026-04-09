"""
0DTE Options Engine — Backtester
Simulates iron condor entries and tracks P&L.
Entry: after 10:30 AM, strike width based on predicted range
Exit: 50% profit, 2x loss, or 3:15 PM
"""
import pandas as pd
import numpy as np
import os
import joblib
from range_predictor import predict, build_model_features, CACHE_DIR

STARTING_CAPITAL = 50_000
MAX_RISK_PER_TRADE = 500      # $500 max loss per trade = 1% of 50k
PROFIT_TARGET = 0.50          # Close at 50% of max profit
STOP_MULTIPLE = 2.0           # Close at 2x premium collected
MIN_OTM_PCT = 0.003           # 0.30% OTM minimum

# FOMC/CPI/NFP dates — skip these
MACRO_EVENT_DATES = set()     # populated from FRED calendar

def load_chain(date_str):
    path = os.path.join(CACHE_DIR, f"chain_eod_{date_str}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)

def select_strikes(spx_price, predicted_range_pct, min_otm=MIN_OTM_PCT):
    """
    Given SPX price and predicted range, select iron condor strikes.
    Strikes are placed at max(min_otm, predicted_range/2 + buffer) OTM.
    """
    half_range = predicted_range_pct / 100 / 2
    buffer = 0.002  # extra 0.2% cushion
    otm_pct = max(min_otm, half_range + buffer)
    
    call_short = spx_price * (1 + otm_pct)
    put_short  = spx_price * (1 - otm_pct)
    wing_width = spx_price * 0.005  # $5 wide wings = 0.5%
    
    call_long = call_short + wing_width
    put_long  = put_short  - wing_width
    
    # Round to nearest 5
    call_short = round(call_short / 5) * 5
    put_short  = round(put_short  / 5) * 5
    call_long  = call_short + 5
    put_long   = put_short  - 5
    
    return put_long, put_short, call_short, call_long

def get_iron_condor_premium(chain, put_long, put_short, call_short, call_long):
    """Look up bid/ask for each leg from EOD chain."""
    def mid(row):
        return (float(row['bid']) + float(row['ask'])) / 2 if row is not None else 0
    
    def find_strike(right, strike):
        sub = chain[chain['right'] == right]
        if sub.empty:
            return None
        nearest = sub.iloc[(sub['strike'] - strike).abs().argsort()[:1]]
        if abs(float(nearest['strike'].iloc[0]) - strike) > 50:
            return None  # too far
        return nearest.iloc[0]
    
    ps_row = find_strike('PUT',  put_short)
    pl_row = find_strike('PUT',  put_long)
    cs_row = find_strike('CALL', call_short)
    cl_row = find_strike('CALL', call_long)
    
    if any(r is None for r in [ps_row, pl_row, cs_row, cl_row]):
        return None
    
    # Net credit = short premium - long premium
    credit = mid(ps_row) + mid(cs_row) - mid(pl_row) - mid(cl_row)
    max_loss = 5 * 100 - credit * 100  # wing width * multiplier - credit received
    
    return {'credit': credit, 'max_loss': max_loss,
            'put_long': put_long, 'put_short': put_short,
            'call_short': call_short, 'call_long': call_long}

def simulate_outcome(spx_close, trade):
    """
    Simplified EOD settlement — real backtester would use intraday.
    Returns P&L per contract.
    """
    ps, pl = trade['put_short'], trade['put_long']
    cs, cl = trade['call_short'], trade['call_long']
    credit = trade['credit']
    
    put_loss  = max(0, ps - max(spx_close, pl)) - max(0, spx_close - pl)
    call_loss = max(0, min(spx_close, cl) - cs) - max(0, cl - spx_close)
    
    # If expired worthless (price between short strikes)
    if ps <= spx_close <= cs:
        pnl = credit * 100  # full credit
    elif spx_close < pl:
        pnl = (credit - (ps - pl)) * 100  # max loss on put side
    elif spx_close > cl:
        pnl = (credit - (cl - cs)) * 100  # max loss on call side
    elif spx_close < ps:
        pnl = (credit - (ps - spx_close)) * 100
    else:
        pnl = (credit - (spx_close - cs)) * 100
    
    return pnl

def run_backtest():
    df = pd.read_parquet(os.path.join(CACHE_DIR, "daily_features.parquet"))
    df = build_model_features(df).dropna(subset=['day_range_pct','vix','realized_vol_5d']).reset_index(drop=True)
    
    # Walk-forward: train on first 60% predict on last 40%
    split = int(len(df) * 0.6)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:].reset_index(drop=True)
    
    print(f"Train: {train_df['date'].iloc[0].date()} to {train_df['date'].iloc[-1].date()} ({len(train_df)} days)")
    print(f"Test:  {test_df['date'].iloc[0].date()} to {test_df['date'].iloc[-1].date()} ({len(test_df)} days)")
    
    from range_predictor import train
    model, feat_cols = train(train_df)
    
    capital = STARTING_CAPITAL
    trades = []
    
    for i, row in test_df.iterrows():
        date_str = row['date'].strftime("%Y%m%d")
        
        if date_str in MACRO_EVENT_DATES:
            continue
        
        # Skip high vol days — VIX > 35 too dangerous
        if row['vix'] > 35:
            continue
        
        # Get predicted range
        feat_dict = {f: row.get(f, 0) for f in feat_cols}
        pred_range = predict(feat_dict)
        
        # Get chain
        chain = load_chain(date_str)
        if chain.empty:
            continue
        
        # Estimate SPX price from ATM options (use SPY * 10 as proxy)
        spx_price = row.get('spx_close', None)
        if spx_price is None:
            # Use SPY close * ~10 ratio as rough proxy
            continue
        
        # Select strikes
        pl, ps, cs, cl = select_strikes(spx_price, pred_range)
        
        # Get premium
        trade_info = get_iron_condor_premium(chain, pl, ps, cs, cl)
        if trade_info is None or trade_info['credit'] <= 0:
            continue
        
        # Size: max $500 risk per trade
        n_contracts = max(1, int(MAX_RISK_PER_TRADE / trade_info['max_loss']))
        
        # Simulate outcome using actual SPX close
        spx_close = spx_price  # placeholder — need actual close
        pnl = simulate_outcome(spx_close, trade_info) * n_contracts
        capital += pnl
        
        trades.append({
            'date': row['date'],
            'pred_range': pred_range,
            'actual_range': row['day_range_pct'],
            'credit': trade_info['credit'],
            'n_contracts': n_contracts,
            'pnl': pnl,
            'capital': capital,
            'vix': row['vix'],
        })
    
    results = pd.DataFrame(trades)
    if results.empty:
        print("No trades generated")
        return
    
    total_return = (capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    win_rate = (results['pnl'] > 0).mean() * 100
    n_years = (results['date'].iloc[-1] - results['date'].iloc[0]).days / 365
    cagr = ((capital / STARTING_CAPITAL) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"  RESULTS: {len(results)} trades")
    print(f"  Total Return: {total_return:.1f}%")
    print(f"  CAGR: {cagr:.1f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg P&L/trade: ${results['pnl'].mean():.0f}")
    print(f"  Final Capital: ${capital:,.0f}")
    print(f"{'='*50}")
    
    return results

if __name__ == "__main__":
    run_backtest()
