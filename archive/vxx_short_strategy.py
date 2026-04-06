"""
vxx_short_strategy.py — Systematic VXX short in TRENDING_BULL regime

Research basis:
- Simon (2014): VIX futures roll yield -5 to -7% per month in contango
- Eraker & Wu (2017): Short VIX futures earns risk premium over full cycle
- Our regime classifier: 52% of backtest days = TRENDING_BULL

Strategy:
  - In TRENDING_BULL regime: maintain short VXX position (3% of portfolio)
  - Exit when regime flips to BEAR or VIX > 30
  - No leverage, defined max loss = full position value (VXX can't go above ~100)

VXX loses value from two sources:
  1. VIX futures contango roll: -5 to -7% per month mechanically
  2. VIX mean reversion when elevated
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

CACHE_DIR   = '/Users/rick/ai_trading_bot_v2/cache_prices'
OUTPUT_PATH = '/Users/rick/ai_trading_bot_v2/vxx_short_results.json'


def load_price(sym):
    path = os.path.join(CACHE_DIR, f"{sym}_max.pkl")
    if os.path.exists(path):
        df = pd.read_pickle(path)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        return df['close'].dropna()
    # Fall back to ETF cache
    for ext in ['_etf.csv', '_3650d.csv']:
        path2 = os.path.join(CACHE_DIR, f"{sym}{ext}")
        if os.path.exists(path2):
            df = pd.read_csv(path2, index_col=0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
            return df['close'].dropna()
    return None


def run_vxx_short_backtest():
    from regime_classifier import RegimeClassifier, compute_signals, load_macro_data, TRENDING_BULL, BEAR

    print("=== VXX SHORT BACKTEST ===")
    print("Strategy: Short VXX in TRENDING_BULL regime. Exit on BEAR or VIX>30.")
    print()

    # Load data
    vxx = load_price('VXX')
    vix = load_price('^VIX')
    spy_mac, hyg_mac, vix_mac = load_macro_data(cache_dir=CACHE_DIR)

    if vxx is None:
        print("ERROR: VXX data not found — fetching now...")
        import yfinance as yf
        df = yf.Ticker('VXX').history(period='max', auto_adjust=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        df.to_pickle(os.path.join(CACHE_DIR, 'VXX_max.pkl'))
        vxx = df['close'].dropna()
        print(f"VXX fetched: {len(vxx)} rows")

    if vix is None:
        vix_path = os.path.join(CACHE_DIR, '^VIX_etf.csv')
        df = pd.read_csv(vix_path, index_col=0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        vix = df['close'].dropna()

    # Build regime series
    print("Building regime history...")
    clf = RegimeClassifier()
    regime_map = {}
    for d in spy_mac.index:
        signals = compute_signals(spy_mac, hyg_mac, vix_mac, as_of_date=d)
        if signals:
            regime_map[pd.Timestamp(d).normalize()] = clf.update(d, signals)

    print(f"Regime map: {len(regime_map)} dates")
    print(f"VXX: {len(vxx)} rows, {vxx.index[0].date()} → {vxx.index[-1].date()}")

    PORTFOLIO  = 100_000
    ALLOCATION = 0.03   # 3% of portfolio
    SLIPPAGE   = 0.001  # 10bps round trip

    trades     = []
    in_short   = False
    entry_px   = None
    entry_date = None
    entry_val  = 0.0
    shares     = 0
    total_pnl  = 0.0

    print(f"\n{'Date':<12} {'Regime':<16} {'VXX':>7} {'VIX':>6} {'Action':<20} {'PnL':>10}")
    print("-" * 75)

    # Align dates
    common = vxx.index.intersection(pd.DatetimeIndex(regime_map.keys()))
    common = sorted(common)

    for date in common:
        regime = regime_map.get(date, 'CHOPPY')
        vxx_px = float(vxx.loc[date]) if date in vxx.index else None
        vix_px = float(vix.loc[date]) if vix is not None and date in vix.index else 20.0

        if vxx_px is None:
            continue

        should_be_short = (regime == TRENDING_BULL and vix_px < 30)

        # Entry
        if should_be_short and not in_short:
            entry_val  = PORTFOLIO * ALLOCATION
            shares     = int(entry_val / (vxx_px * (1 + SLIPPAGE)))
            entry_px   = vxx_px * (1 + SLIPPAGE)
            entry_date = date
            in_short   = True
            print(f"  {str(date.date()):<10} {regime:<16} {vxx_px:>7.2f} {vix_px:>6.1f} "
                  f"SHORT {shares} shares @{entry_px:.2f}")

        # Exit
        elif in_short and not should_be_short:
            exit_px = vxx_px * (1 - SLIPPAGE)
            pnl = (entry_px - exit_px) * shares
            total_pnl += pnl
            ret = pnl / entry_val if entry_val > 0 else 0
            reason = f"{'BEAR' if regime==BEAR else 'VIX>30' if vix_px>=30 else regime}"

            print(f"  {str(date.date()):<10} {regime:<16} {vxx_px:>7.2f} {vix_px:>6.1f} "
                  f"COVER ({reason}) {pnl:>+10,.0f} ({ret:+.1%})")

            trades.append({
                'entry_date': str(entry_date.date()),
                'exit_date':  str(date.date()),
                'entry_px':   entry_px,
                'exit_px':    exit_px,
                'shares':     shares,
                'pnl':        pnl,
                'ret':        ret,
                'days':       (date - entry_date).days,
                'exit_reason': reason,
            })
            in_short   = False
            entry_px   = None

    # Close any open position
    if in_short and len(vxx) > 0:
        last_date = common[-1]
        last_px   = float(vxx.iloc[-1]) * (1 - SLIPPAGE)
        pnl = (entry_px - last_px) * shares
        total_pnl += pnl
        trades.append({
            'entry_date': str(entry_date.date()),
            'exit_date':  str(last_date.date()),
            'entry_px':   entry_px,
            'exit_px':    last_px,
            'shares':     shares,
            'pnl':        pnl,
            'ret':        pnl / entry_val,
            'days':       (last_date - entry_date).days,
            'exit_reason': 'OPEN',
        })

    # Summary
    print("\n" + "="*60)
    print("VXX SHORT SUMMARY")
    print("="*60)

    if trades:
        df_r  = pd.DataFrame(trades)
        wins  = df_r[df_r['pnl'] > 0]
        total = len(df_r)

        print(f"  Trades:              {total}")
        print(f"  Win rate:            {len(wins)/total:.0%}")
        print(f"  Avg PnL per trade:   ${df_r['pnl'].mean():,.0f}")
        print(f"  Total PnL:           ${df_r['pnl'].sum():,.0f}")
        print(f"  Best trade:          ${df_r['pnl'].max():,.0f}")
        print(f"  Worst trade:         ${df_r['pnl'].min():,.0f}")
        print(f"  Avg hold (days):     {df_r['days'].mean():.0f}")
        print(f"  Avg return/trade:    {df_r['ret'].mean():+.1%}")

        years = (pd.Timestamp(trades[-1]['exit_date']) -
                 pd.Timestamp(trades[0]['entry_date'])).days / 365.0
        cagr = (total_pnl / PORTFOLIO) / max(years, 1)
        print(f"\n  Annualized contribution: {cagr:+.2%}")

        df_r['year'] = pd.to_datetime(df_r['entry_date']).dt.year
        print(f"\n  {'Year':<6} {'Trades':>7} {'PnL':>10} {'WR':>7}")
        print("  " + "-"*32)
        for yr, grp in df_r.groupby('year'):
            wr = (grp['pnl'] > 0).mean()
            print(f"  {yr:<6} {len(grp):>7} ${grp['pnl'].sum():>9,.0f} {wr:>7.0%}")

        verdict = "DEPLOY" if cagr > 0.01 else "MARGINAL" if cagr > 0 else "DO NOT DEPLOY"
        print(f"\n  VERDICT: {verdict}")

        import json
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(trades, f, indent=2, default=str)
        print(f"  Saved: {OUTPUT_PATH}")
    else:
        print("  No completed trades")

    return trades


if __name__ == "__main__":
    run_vxx_short_backtest()
