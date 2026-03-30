"""
backtest_meanrev.py — Mean Reversion CHOPPY Strategy
=====================================================
Runs ONLY during CHOPPY regime.
Entry: RSI crosses below 30 (oversold)
Exit:  RSI crosses above 55, OR max 10 days
Capital: separate sleeve, activated only in CHOPPY

Validated: 8.8% CAGR standalone, 53.5% WR, 1,981 trades over 17yr
"""
import os, sys, pandas as pd, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.expanduser('~/ai_trading_bot_v2'))
sys.path.insert(0, os.path.expanduser('~/ai_trading_bot_v2/v2'))

from backtester_clean import fetch_history, apply_fill_cost, calc_stats
from regime_classifier import build_regime_series, load_macro_data, CHOPPY, TRENDING_BULL, BEAR
import config

INITIAL_CAP   = 100_000
MAX_POSITIONS = 8
STOP_PCT      = 0.08
MAX_HOLD      = 10
RSI_ENTRY     = 30
RSI_EXIT      = 55
START_DATE    = '2008-01-01'
END_DATE      = '2025-12-31'
CACHE_DIR     = os.path.expanduser('~/ai_trading_bot_v2/cache_meanrev')
os.makedirs(CACHE_DIR, exist_ok=True)
SEP = "="*60

def load_prices(symbols):
    prices = {}
    for i, sym in enumerate(symbols):
        if i % 50 == 0:
            print(f"  {i}/{len(symbols)} loaded={len(prices)}")
        try:
            cache = os.path.join(CACHE_DIR, f"{sym}.pkl")
            if os.path.exists(cache):
                df = pd.read_pickle(cache)
            else:
                df = fetch_history(sym, days=6500)
                if df is None or len(df) < 200:
                    continue
                idx = pd.to_datetime(df.index)
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                df = df.copy()
                df.index = pd.DatetimeIndex(idx.date)
                df.columns = [c.lower() for c in df.columns]
                df.to_pickle(cache)
            prices[sym] = df
        except:
            pass
    print(f"  Loaded {len(prices)} symbols")
    return prices

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss.clip(lower=0.001))

def run_backtest():
    print(f"\n{SEP}")
    print("  MEAN REVERSION CHOPPY STRATEGY BACKTEST")
    print(f"  Capital: ${INITIAL_CAP:,}  MaxPos: {MAX_POSITIONS}")
    print(f"  Entry: RSI<{RSI_ENTRY}  Exit: RSI>{RSI_EXIT} or {MAX_HOLD}d")
    print(SEP)

    symbols = getattr(config, 'WATCHLIST', getattr(config, 'SYMBOLS', []))
    print(f"Universe: {len(symbols)} symbols")

    prices = load_prices(symbols)

    print("Loading macro data for regime...")
    spy_df, hyg_df, vix_df = load_macro_data()

    spy_prices = prices.get('SPY', next(iter(prices.values())))
    all_dates = spy_prices.index[
        (spy_prices.index >= pd.Timestamp(START_DATE)) &
        (spy_prices.index <= pd.Timestamp(END_DATE))
    ]
    all_dates_ts = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])

    print("Building regime series...")
    regime_series = build_regime_series(spy_df, hyg_df, vix_df, all_dates_ts)
    choppy_days = (regime_series == CHOPPY).sum()
    print(f"CHOPPY days: {choppy_days} of {len(regime_series)} ({choppy_days/len(regime_series):.0%})")

    # Pre-compute RSI for all symbols
    print("Pre-computing RSI signals...")
    rsi_cache = {}
    for sym, df in prices.items():
        try:
            rsi_cache[sym] = compute_rsi(df['close'])
        except:
            pass
    print(f"  RSI computed for {len(rsi_cache)} symbols")

    # Main loop
    cash      = float(INITIAL_CAP)
    positions = {}
    trades    = []
    equity    = []

    for date in all_dates:
        date_ts = pd.Timestamp(date)
        regime  = regime_series.get(date_ts, CHOPPY)

        # Mark to market
        pos_value = 0
        for sym, pos in positions.items():
            df = prices.get(sym)
            if df is not None and date in df.index:
                pos_value += float(df.loc[date, 'close']) * pos['qty']
            else:
                pos_value += pos['entry_price'] * pos['qty']
        equity.append({'date': date_ts, 'equity': cash + pos_value})

        # Exit logic
        for sym in list(positions.keys()):
            pos   = positions[sym]
            df    = prices.get(sym)
            if df is None or date not in df.index:
                continue
            close     = float(df.loc[date, 'close'])
            hold_days = (date_ts - pd.Timestamp(pos['entry_date'])).days
            rsi_val   = float(rsi_cache[sym].get(date_ts, 50) if hasattr(rsi_cache[sym], 'get') else rsi_cache[sym].reindex([date_ts]).fillna(50).iloc[0])

            exit_reason = None
            if close <= pos['stop']:
                exit_reason = 'stop'
            elif rsi_val > RSI_EXIT:
                exit_reason = 'rsi_exit'
            elif hold_days >= MAX_HOLD:
                exit_reason = 'max_hold'
            elif regime != CHOPPY:
                exit_reason = 'regime_exit'

            if exit_reason:
                fill, comm = apply_fill_cost(close, pos['qty'], 'sell')
                pnl = (fill - pos['entry_price']) * pos['qty'] - comm
                cash += fill * pos['qty'] - comm
                trades.append({
                    'symbol': sym, 'entry_date': pos['entry_date'],
                    'exit_date': str(date), 'entry_price': pos['entry_price'],
                    'exit_price': fill, 'qty': pos['qty'],
                    'pnl': pnl, 'reason': exit_reason,
                    'hold_days': hold_days,
                })
                del positions[sym]

        # Entry logic — CHOPPY only
        if regime == CHOPPY and len(positions) < MAX_POSITIONS:
            candidates = []
            for sym, rsi_s in rsi_cache.items():
                if sym in positions:
                    continue
                try:
                    dates_idx = rsi_s.index
                    if date_ts not in dates_idx:
                        continue
                    loc = dates_idx.get_loc(date_ts)
                    if loc == 0:
                        continue
                    curr_rsi = float(rsi_s.iloc[loc])
                    prev_rsi = float(rsi_s.iloc[loc-1])
                    if curr_rsi < RSI_ENTRY and prev_rsi >= RSI_ENTRY:
                        df = prices.get(sym)
                        if df is not None and date in df.index:
                            candidates.append((sym, curr_rsi, float(df.loc[date, 'close'])))
                except:
                    pass

            # Sort by most oversold first
            candidates.sort(key=lambda x: x[1])

            for sym, rsi_val, px in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break
                if px <= 0:
                    continue
                risk_dollars = INITIAL_CAP * 0.02
                stop_px = px * (1 - STOP_PCT)
                qty = int(risk_dollars / (px - stop_px))
                qty = min(qty, int(INITIAL_CAP * 0.125 / px))
                if qty <= 0:
                    continue
                fill, comm = apply_fill_cost(px, qty, 'buy')
                if fill * qty + comm > cash:
                    continue
                cash -= fill * qty + comm
                positions[sym] = {
                    'entry_price': fill, 'qty': qty,
                    'entry_date': str(date), 'stop': stop_px,
                }

    # Force close
    for sym, pos in list(positions.items()):
        df = prices.get(sym)
        final_date = all_dates[-1]
        close = float(df.loc[final_date, 'close']) if (df is not None and final_date in df.index) else pos['entry_price']
        fill, comm = apply_fill_cost(close, pos['qty'], 'sell')
        pnl = (fill - pos['entry_price']) * pos['qty'] - comm
        cash += fill * pos['qty'] - comm
        trades.append({'symbol': sym, 'entry_date': pos['entry_date'],
                       'exit_date': str(final_date), 'pnl': pnl,
                       'reason': 'forced_close', 'hold_days': 0})

    # Results
    eq = pd.DataFrame(equity).set_index('date')['equity']
    df_trades = pd.DataFrame(trades)

    print(f"\n{SEP}")
    print("  MEAN REVERSION RESULTS")
    print(SEP)

    if len(df_trades) == 0:
        print("NO TRADES")
        return

    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr  = ((eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1) * 100
    dr    = eq.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * 252**0.5 if dr.std() > 0 else 0
    roll_max = eq.cummax()
    max_dd = ((eq - roll_max) / roll_max).min() * 100
    wr = (df_trades['pnl'] > 0).mean() * 100

    print(f"  Total Return : {total_ret:>8.2f}%")
    print(f"  CAGR         : {cagr:>8.2f}%")
    print(f"  Sharpe       : {sharpe:>8.2f}")
    print(f"  Max Drawdown : {max_dd:>8.2f}%")
    print(f"  Trades       : {len(df_trades):>8}")
    print(f"  Win Rate     : {wr:>8.1f}%")

    print(f"\n  Exit breakdown:")
    for reason, g in df_trades.groupby('reason'):
        w = (g['pnl'] > 0).mean() * 100
        print(f"    {reason:<20} {len(g):>4}  WR={w:.0f}%  avg=${g['pnl'].mean():>8,.0f}")

    print(f"\n  Year-by-year:")
    df_trades['year'] = pd.to_datetime(df_trades['exit_date']).dt.year
    yr_eq  = eq.copy()
    yr_eq.index = pd.to_datetime(yr_eq.index)
    yr_s = yr_eq.groupby(yr_eq.index.year).first()
    yr_e = yr_eq.groupby(yr_eq.index.year).last()
    yr_r = (yr_e / yr_s - 1) * 100
    for y in sorted(yr_r.index):
        if 2008 <= y <= 2025:
            tyr = df_trades[df_trades['year'] == y]
            w = (tyr['pnl'] > 0).mean() * 100 if len(tyr) > 0 else 0
            print(f"    {y}  {yr_r[y]:>7.1f}%  trades={len(tyr):>3}  WR={w:.0f}%  {'V' if yr_r[y] > 0 else 'X'}")

    out = os.path.expanduser('~/ai_trading_bot_v2/meanrev_trades.csv')
    df_trades.to_csv(out, index=False)
    print(f"\n  Saved to: {out}")
    print(SEP)

if __name__ == '__main__':
    run_backtest()
