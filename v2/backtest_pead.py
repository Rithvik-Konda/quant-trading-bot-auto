"""
backtest_pead.py — Standalone PEAD strategy backtest
Post-Earnings Announcement Drift (Bernard & Thomas 1989)
============================================================
Completely independent from core momentum strategy.
Own capital allocation: $15,000 (15% of portfolio)
Own entry/exit rules
Own backtest
"""
import pandas as pd
import numpy as np
import yfinance as yf
import json, os, sys
from datetime import datetime, timedelta
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
from backtester_clean import fetch_history, apply_fill_cost, calc_stats

PEAD_CAPITAL    = 15_000   # $15k allocation
MIN_SURPRISE    = 0.05     # 5% earnings beat minimum
HOLD_DAYS       = 7        # hold 7 days after earnings
STOP_PCT        = 0.04     # tight 4% stop — if drift doesn't materialize bail fast
MAX_POSITIONS   = 3        # 3 concurrent PEAD trades
POSITION_SIZE   = 5_000    # $5k per trade

def get_earnings_history(ticker: str) -> pd.DataFrame:
    """Get historical earnings surprises."""
    try:
        tick = yf.Ticker(ticker)
        hist = tick.earnings_history
        if hist is None or len(hist) == 0:
            return pd.DataFrame()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist = hist.sort_index()
        hist['surprise_pct'] = (hist['epsActual'] - hist['epsEstimate']) / hist['epsEstimate'].abs().clip(lower=0.01)
        return hist
    except:
        return pd.DataFrame()

def run_pead_backtest(watchlist: list, days: int = 3650):
    """Run PEAD strategy backtest independently."""
    print("="*55)
    print("PEAD STRATEGY BACKTEST")
    print(f"Capital: ${PEAD_CAPITAL:,}")
    print(f"Universe: {len(watchlist)} stocks")
    print(f"Signal: earnings beat >{MIN_SURPRISE:.0%}, enter next day, hold {HOLD_DAYS}d")
    print("="*55)

    # Load earnings history for all symbols
    print("\nLoading earnings histories...")
    earnings_db = {}
    for sym in watchlist[:50]:  # limit for speed
        hist = get_earnings_history(sym)
        if len(hist) > 0:
            earnings_db[sym] = hist

    print(f"Loaded: {len(earnings_db)} symbols with earnings data")

    # Build all PEAD signals
    pead_signals = []
    for sym, hist in earnings_db.items():
        for date, row in hist.iterrows():
            surprise = float(row.get('surprise_pct', 0) or 0)
            if surprise >= MIN_SURPRISE:
                entry_date = date + pd.Timedelta(days=1)
                pead_signals.append({
                    'symbol':     sym,
                    'earnings_date': date,
                    'entry_date': entry_date,
                    'surprise':   surprise,
                    'eps_est':    float(row.get('epsEstimate', 0) or 0),
                    'eps_act':    float(row.get('epsActual', 0) or 0),
                })

    pead_signals.sort(key=lambda x: x['entry_date'])
    print(f"PEAD signals: {len(pead_signals)} over 9 years")

    # Simulate trades
    cash       = float(PEAD_CAPITAL)
    positions  = {}  # sym -> {entry_price, qty, entry_date, stop}
    trades     = []
    equity     = [(pd.Timestamp('2017-01-01'), cash)]

    for signal in pead_signals:
        sym        = signal['symbol']
        entry_date = signal['entry_date']
        surprise   = signal['surprise']

        # Skip if already in position
        if sym in positions:
            continue
        if len(positions) >= MAX_POSITIONS:
            continue

        # Filter to our backtest period
        if entry_date.year < 2017 or entry_date.year > 2025:
            continue

        # Get price data
        try:
            prices = fetch_history(sym, days=9999)
            prices.index = pd.to_datetime(prices.index).tz_localize(None)

            # Find entry price (next day open)
            future = prices[prices.index > entry_date]
            if len(future) == 0:
                continue

            entry_price = float(future.iloc[0]['open'])
            if entry_price <= 0:
                continue

            qty = int(POSITION_SIZE / entry_price)
            if qty <= 0:
                continue

            cost = entry_price * qty
            if cost > cash:
                continue

            cash -= cost
            stop = entry_price * (1 - STOP_PCT)

            positions[sym] = {
                'entry_price': entry_price,
                'qty':         qty,
                'entry_date':  entry_date,
                'stop':        stop,
                'surprise':    surprise,
                'hold_days':   HOLD_DAYS,
                'target_exit': entry_date + pd.Timedelta(days=HOLD_DAYS),
            }
        except:
            continue

        # Check exits for all positions
        current_date = entry_date
        for held_sym in list(positions.keys()):
            pos = positions[held_sym]
            try:
                prices_held = fetch_history(held_sym, days=9999)
                prices_held.index = pd.to_datetime(prices_held.index).tz_localize(None)

                # Get prices during hold period
                hold_prices = prices_held[
                    (prices_held.index > pos['entry_date']) &
                    (prices_held.index <= current_date)
                ]

                exit_reason = exit_price = None

                # Check stop
                for _, bar in hold_prices.iterrows():
                    if bar['low'] <= pos['stop']:
                        exit_reason = 'stop'
                        exit_price  = pos['stop']
                        break

                # Check time exit
                if exit_reason is None and current_date >= pos['target_exit']:
                    exit_reason = 'time_exit'
                    exit_price  = float(prices_held[prices_held.index <= current_date].iloc[-1]['close'])

                if exit_reason and exit_price:
                    pnl = (exit_price - pos['entry_price']) * pos['qty']
                    cash += exit_price * pos['qty']
                    trades.append({
                        'symbol':      held_sym,
                        'entry_date':  str(pos['entry_date'].date()),
                        'exit_date':   str(current_date.date()),
                        'entry_price': pos['entry_price'],
                        'exit_price':  exit_price,
                        'qty':         pos['qty'],
                        'pnl':         pnl,
                        'reason':      exit_reason,
                        'surprise':    pos['surprise'],
                        'year':        pos['entry_date'].year,
                    })
                    del positions[held_sym]

            except:
                pass

    # Close remaining positions
    for sym, pos in positions.items():
        pnl = 0
        cash += pos['entry_price'] * pos['qty']
        trades.append({
            'symbol':   sym,
            'pnl':      pnl,
            'reason':   'open',
            'year':     pos['entry_date'].year,
        })

    # Results
    trade_df = pd.DataFrame(trades)
    if len(trade_df) == 0:
        print("No trades generated")
        return

    closed = trade_df[trade_df['reason'] != 'open']
    if len(closed) == 0:
        print("No closed trades")
        return

    wr      = (closed['pnl'] > 0).mean()
    avg_pnl = closed['pnl'].mean()
    total   = closed['pnl'].sum()
    n       = len(closed)
    cagr    = (1 + total/PEAD_CAPITAL) ** (1/9) - 1

    print(f"\nRESULTS:")
    print(f"  Trades:     {n}")
    print(f"  Win Rate:   {wr:.0%}")
    print(f"  Avg PnL:    ${avg_pnl:,.0f}")
    print(f"  Total PnL:  ${total:,.0f}")
    print(f"  On $15k:    {total/PEAD_CAPITAL:.0%} total return")
    print(f"  CAGR:       {cagr:.1%}")

    print(f"\n  By year:")
    for yr in range(2017, 2026):
        yr_trades = closed[closed['year'] == yr]
        if len(yr_trades) == 0:
            continue
        yr_pnl = yr_trades['pnl'].sum()
        yr_wr  = (yr_trades['pnl'] > 0).mean()
        print(f"    {yr}: n={len(yr_trades):>2}  WR={yr_wr:.0%}  pnl=${yr_pnl:>+8,.0f}")

    print(f"\n  By exit reason:")
    for reason, grp in closed.groupby('reason'):
        print(f"    {reason:<12}: n={len(grp):>3}  WR={(grp['pnl']>0).mean():.0%}  avg=${grp['pnl'].mean():>+7,.0f}")

    # Add to core strategy
    core_cagr = 0.2256
    combined  = core_cagr + cagr * 0.15  # PEAD at 15% allocation
    print(f"\n  COMBINED with core momentum:")
    print(f"    Core CAGR:  {core_cagr:.1%}")
    print(f"    PEAD adds:  {cagr*0.15:.1%}")
    print(f"    Combined:   {combined:.1%}")

if __name__ == "__main__":
    import config
    run_pead_backtest(config.WATCHLIST)
