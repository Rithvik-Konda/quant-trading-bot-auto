"""
backtest_meanrev.py — RSI(2) Mean Reversion Standalone Backtest
==============================================================
Tests RSI(2) strategy on the full watchlist across all years.
Finds optimal:
- RSI threshold (5, 10, 15, 20)
- Universe filter (all stocks, above 200MA only, defensive sectors)
- Capital allocation vs momentum (0%, 25%, 50%, 75%, 100%)

Run: python v2/backtest_meanrev.py
"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/ai_trading_bot_v2'))
import pandas as pd
import numpy as np
import yfinance as yf
import config

CACHE_DIR = os.path.expanduser('~/ai_trading_bot_v2/cache_prices')

def load_prices(sym):
    path = os.path.join(CACHE_DIR, f"{sym}_3650d.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
        df = df.loc[~df.index.isna()]
        df.index = df.index.tz_convert('UTC').tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        return df[['open','high','low','close','volume']].dropna()
    return None

def compute_rsi2(close, period=2):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def backtest_rsi2(symbols, rsi_threshold=10, capital=100000,
                  max_positions=5, start='2015-01-01', end='2026-01-01',
                  preloaded=None):
    """
    Backtest RSI(2) mean reversion strategy.
    Entry:  RSI(2) < threshold AND price > SMA200
    Exit:   price closes above SMA5  OR  10-day time stop
    """
    if preloaded is not None:
        price_data = {s: preloaded[s] for s in symbols if s in preloaded}
    else:
        price_data = {}
        for sym in symbols:
            df = load_prices(sym)
            if df is not None and len(df) > 250:
                price_data[sym] = df

    # Build daily cross-sectional signals
    all_dates = pd.date_range(start, end, freq='B')
    
    cash     = capital
    holdings = {}  # sym -> {qty, entry_price, entry_date, entry_rsi}
    trades   = []
    equity   = []

    for date in all_dates:
        # Mark to market
        port_val = cash
        for sym, h in holdings.items():
            if sym in price_data:
                df = price_data[sym]
                idx = df.index.asof(date)
                if idx in df.index:
                    port_val += h['qty'] * float(df.loc[idx, 'close'])
        equity.append((date, port_val))

        # Check exits
        to_exit = []
        for sym, h in holdings.items():
            if sym not in price_data: continue
            df  = price_data[sym]
            idx = df.index.asof(date)
            if idx not in df.index: continue
            close = float(df.loc[idx, 'close'])
            sma5  = float(df['close'].loc[:idx].tail(5).mean())
            days_held = (date - h['entry_date']).days
            if close > sma5 or days_held >= 10:
                cash += h['qty'] * close
                pnl   = (close - h['entry_price']) * h['qty']
                trades.append({
                    'sym': sym, 'entry': h['entry_price'],
                    'exit': close, 'pnl': pnl,
                    'days': days_held,
                    'year': date.year,
                    'exit_reason': 'sma5' if close > sma5 else 'timeout'
                })
                to_exit.append(sym)
        for sym in to_exit:
            del holdings[sym]

        # Check entries
        slots = max_positions - len(holdings)
        if slots <= 0: continue

        candidates = []
        for sym in symbols:
            if sym in holdings: continue
            if sym not in price_data: continue
            df  = price_data[sym]
            idx = df.index.asof(date)
            if idx not in df.index: continue
            if len(df.loc[:idx]) < 210: continue
            close  = float(df.loc[idx, 'close'])
            sma200 = float(df['close'].loc[:idx].tail(200).mean())
            rsi2   = compute_rsi2(df['close'].loc[:idx].tail(20))
            if len(rsi2.dropna()) == 0: continue
            rsi_val = float(rsi2.iloc[-1])

            # Entry: RSI(2) oversold AND above 200d MA
            if rsi_val < rsi_threshold and close > sma200:
                candidates.append((sym, rsi_val, close))

        # Sort by most oversold first
        candidates.sort(key=lambda x: x[1])
        for sym, rsi_val, close in candidates[:slots]:
            position_size = (cash * 0.95) / max(slots, 1)
            qty = int(position_size / close)
            if qty < 1: continue
            cash -= qty * close
            holdings[sym] = {
                'qty': qty, 'entry_price': close,
                'entry_date': date, 'entry_rsi': rsi_val
            }

    # Final equity
    eq_df = pd.DataFrame(equity, columns=['date','equity'])
    eq_df = eq_df.set_index('date')
    eq_df['ret'] = eq_df['equity'].pct_change()

    total_ret  = (eq_df['equity'].iloc[-1] / capital - 1) * 100
    years      = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    cagr       = ((eq_df['equity'].iloc[-1] / capital) ** (1/years) - 1) * 100
    sharpe     = eq_df['ret'].mean() / eq_df['ret'].std() * np.sqrt(252) if eq_df['ret'].std() > 0 else 0
    max_dd     = ((eq_df['equity'] / eq_df['equity'].cummax()) - 1).min() * 100

    tr_df   = pd.DataFrame(trades)
    wr      = (tr_df['pnl'] > 0).mean() * 100 if len(tr_df) > 0 else 0
    avg_pnl = tr_df['pnl'].mean() if len(tr_df) > 0 else 0

    # Per-year returns
    eq_df['year'] = eq_df.index.year
    yearly = {}
    for yr, grp in eq_df.groupby('year'):
        if len(grp) > 1:
            yr_ret = (grp['equity'].iloc[-1] / grp['equity'].iloc[0] - 1) * 100
            yearly[yr] = yr_ret

    return {
        'cagr': cagr, 'sharpe': sharpe, 'max_dd': max_dd,
        'total_ret': total_ret, 'trades': len(trades),
        'win_rate': wr, 'avg_pnl': avg_pnl,
        'yearly': yearly, 'equity': eq_df
    }


if __name__ == '__main__':
    symbols = list(config.WATCHLIST)

    # Sector map for universe filtering
    sector_map = {}
    for etf, syms in getattr(config, 'SECTOR_ETFS', {}).items():
        for s in syms:
            sector_map[s] = etf

    DEFENSIVE_SECTORS = {'XLU', 'XLP', 'XLV', 'XLF'}
    ALL_SYMS          = symbols
    DEFENSIVE_SYMS    = [s for s in symbols if sector_map.get(s) in DEFENSIVE_SECTORS]

    print("=" * 65)
    print("RSI(2) MEAN REVERSION BACKTEST — PARAMETER SWEEP")
    print("=" * 65)

    results = []

    # Load prices ONCE — reuse across all parameter combinations
    print("Loading all prices once...")
    import sys
    all_price_data = {}
    for sym in ALL_SYMS:
        df = load_prices(sym)
        if df is not None and len(df) > 250:
            all_price_data[sym] = df
        sys.stdout.write(f"\r  Loaded {len(all_price_data)}/{len(ALL_SYMS)}")
        sys.stdout.flush()
    print(f"\nLoaded {len(all_price_data)} symbols")

    for rsi_thresh in [5, 10, 15, 20]:
        for universe, uni_name in [
            (ALL_SYMS, 'ALL'),
            (DEFENSIVE_SYMS, 'DEFENSIVE'),
        ]:
            for max_pos in [3, 5, 8]:
                print(f"\nTesting RSI<{rsi_thresh} | {uni_name} ({len(universe)} syms) | max_pos={max_pos}", flush=True)
                r = backtest_rsi2(
                    universe,
                    rsi_threshold=rsi_thresh,
                    max_positions=max_pos,
                    preloaded=all_price_data,
                )
                r['params'] = f"RSI<{rsi_thresh} {uni_name} pos={max_pos}"
                results.append(r)
                print(f"  CAGR={r['cagr']:.1f}% Sharpe={r['sharpe']:.2f} "
                      f"MaxDD={r['max_dd']:.1f}% WR={r['win_rate']:.0f}% "
                      f"Trades={r['trades']}")
                for yr in [2019, 2021, 2022, 2025]:
                    yr_ret = r['yearly'].get(yr, None)
                    if yr_ret is not None:
                        print(f"    {yr}: {yr_ret:+.1f}%")

    print("\n" + "=" * 65)
    print("TOP 5 CONFIGURATIONS BY SHARPE:")
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    for r in results[:5]:
        print(f"  {r['params']:<35} CAGR={r['cagr']:.1f}% "
              f"Sharpe={r['sharpe']:.2f} MaxDD={r['max_dd']:.1f}%")

    print("\nTOP 5 BY CHOPPY YEAR PERFORMANCE (2019+2021 combined):")
    for r in results:
        r['choppy_score'] = r['yearly'].get(2019, -99) + r['yearly'].get(2021, -99)
    results.sort(key=lambda x: x['choppy_score'], reverse=True)
    for r in results[:5]:
        print(f"  {r['params']:<35} 2019={r['yearly'].get(2019,0):+.1f}% "
              f"2021={r['yearly'].get(2021,0):+.1f}% "
              f"choppy_total={r['choppy_score']:+.1f}%")
