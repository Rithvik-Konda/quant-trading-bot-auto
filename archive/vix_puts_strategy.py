"""
vix_puts_strategy.py — Systematic VIX puts on spike >30

Research basis: Whaley (2009) — VIX mean-reverts to 20 within 45 days
with 78% probability after crossing 30. Simon & Campasano (2014) JFE.

Strategy:
  - When VIX crosses 30 from below: buy 30-DTE VIX puts, strike = VIX * 0.80
  - Exit at expiry or when VIX drops below 22 (mean reversion complete)
  - Allocation: 2% of portfolio per trade
  - Never hold more than 2 simultaneous VIX put positions

VIX options pricing note: VIX options price off VIX futures not spot.
We approximate using Black-Scholes on VIX spot with vol-of-vol estimate.
This understates edge slightly (futures trade at premium to spot in backwardation).
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

VIX_CACHE   = '/Users/rick/ai_trading_bot_v2/cache_prices/^VIX_etf.csv'
OUTPUT_PATH = '/Users/rick/ai_trading_bot_v2/vix_puts_results.json'


def bs_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def load_vix():
    df = pd.read_csv(VIX_CACHE, index_col=0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    return df['close'].dropna()


def run_vix_puts_backtest():
    print("=== VIX PUTS BACKTEST ===")
    print("Strategy: Buy 30-DTE puts when VIX crosses 30. Strike=80% of VIX.")
    print()

    vix = load_vix()

    # Load SPY for portfolio value tracking
    spy_path = '/Users/rick/ai_trading_bot_v2/cache_prices/SPY_max.pkl'
    spy = pd.read_pickle(spy_path)
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_close = spy['close']

    PORTFOLIO    = 100_000
    ALLOCATION   = 0.02   # 2% per trade
    DTE          = 30
    STRIKE_PCT   = 0.80   # 20% OTM put (strike = 80% of VIX level)
    EXIT_LEVEL   = 22.0   # exit when VIX drops back to 22
    MAX_POSITIONS= 2
    RISK_FREE    = 0.05
    VOL_OF_VIX   = 1.20   # VIX itself has ~120% annualized vol (Whaley 2009)

    dates = vix.index
    results = []
    active_positions = []
    total_pnl = 0.0
    portfolio_val = PORTFOLIO
    prev_vix = float(vix.iloc[0])

    print(f"{'Date':<12} {'VIX':>6} {'Action':<35} {'Cost':>8} {'PnL':>9}")
    print("-" * 75)

    for i, date in enumerate(dates[1:], 1):
        curr_vix = float(vix.iloc[i])

        # Check exits on active positions
        still_active = []
        for pos in active_positions:
            # Exit conditions: VIX dropped to 22, or expiry reached
            days_held = (date - pos['entry_date']).days
            expired   = days_held >= DTE
            mean_rev  = curr_vix <= EXIT_LEVEL

            if expired or mean_rev:
                # Payoff: put value at exit
                K = pos['strike']
                if expired:
                    payoff_per_unit = max(K - curr_vix, 0)
                    reason = "EXPIRY"
                else:
                    # Early exit — use BS price with remaining time
                    T_rem = max((DTE - days_held) / 365.0, 0.01)
                    payoff_per_unit = bs_put(curr_vix, K, T_rem, RISK_FREE, VOL_OF_VIX)
                    reason = f"MEAN_REV VIX={curr_vix:.1f}"

                total_payoff = payoff_per_unit * pos['units']
                pnl = total_payoff - pos['cost']
                portfolio_val += pnl
                total_pnl += pnl
                ret = pnl / pos['cost'] if pos['cost'] > 0 else 0

                print(f"  {str(date.date()):<10} {curr_vix:>6.1f} "
                      f"EXIT {reason:<28} ${total_payoff:>7,.0f} ${pnl:>+8,.0f} ({ret:+.0%})")

                results.append({
                    'entry_date': str(pos['entry_date'].date()),
                    'exit_date':  str(date.date()),
                    'entry_vix':  pos['entry_vix'],
                    'exit_vix':   curr_vix,
                    'strike':     pos['strike'],
                    'cost':       pos['cost'],
                    'payoff':     total_payoff,
                    'pnl':        pnl,
                    'reason':     reason,
                    'days_held':  days_held,
                })
            else:
                still_active.append(pos)

        active_positions = still_active

        # Entry: VIX just crossed 30 from below
        crossed_30 = prev_vix < 30 and curr_vix >= 30
        if crossed_30 and len(active_positions) < MAX_POSITIONS:
            K = curr_vix * STRIKE_PCT  # 20% OTM put
            T = DTE / 365.0

            # Price the put
            put_price = bs_put(curr_vix, K, T, RISK_FREE, VOL_OF_VIX)
            if put_price <= 0:
                prev_vix = curr_vix
                continue

            budget = portfolio_val * ALLOCATION
            # VIX options: 1 contract = 100 multiplier
            units  = max(1, int(budget / (put_price * 100)))
            cost   = units * put_price * 100

            active_positions.append({
                'entry_date': date,
                'entry_vix':  curr_vix,
                'strike':     K,
                'put_price':  put_price,
                'units':      units,
                'cost':       cost,
            })
            portfolio_val -= cost

            print(f"  {str(date.date()):<10} {curr_vix:>6.1f} "
                  f"BUY {units} puts K={K:.1f} @${put_price:.2f}  "
                  f"${cost:>7,.0f}")

        prev_vix = curr_vix

    # Summary
    print("\n" + "="*60)
    print("VIX PUTS SUMMARY")
    print("="*60)

    if results:
        df_r = pd.DataFrame(results)
        wins  = df_r[df_r['pnl'] > 0]
        total = len(df_r)

        print(f"  Trades:              {total}")
        print(f"  Win rate:            {len(wins)/total:.0%}")
        print(f"  Avg PnL:             ${df_r['pnl'].mean():,.0f}")
        print(f"  Total PnL:           ${df_r['pnl'].sum():,.0f}")
        print(f"  Best trade:          ${df_r['pnl'].max():,.0f}")
        print(f"  Worst trade:         ${df_r['pnl'].min():,.0f}")
        print(f"  Avg VIX at entry:    {df_r['entry_vix'].mean():.1f}")
        print(f"  Avg VIX at exit:     {df_r['exit_vix'].mean():.1f}")
        print(f"  Mean reversion exits:{(df_r['reason'].str.contains('MEAN')).sum()}")

        years = (dates[-1] - dates[0]).days / 365.0
        cagr_contribution = (df_r['pnl'].sum() / PORTFOLIO) / years
        print(f"\n  Annualized contribution: {cagr_contribution:+.2%}")

        df_r['year'] = pd.to_datetime(df_r['entry_date']).dt.year
        print(f"\n  {'Year':<6} {'Trades':>7} {'PnL':>10} {'WR':>7}")
        print("  " + "-"*32)
        for yr, grp in df_r.groupby('year'):
            wr = (grp['pnl'] > 0).mean()
            print(f"  {yr:<6} {len(grp):>7} ${grp['pnl'].sum():>9,.0f} {wr:>7.0%}")

        verdict = "DEPLOY" if cagr_contribution > 0.015 else "MARGINAL" if cagr_contribution > 0 else "DO NOT DEPLOY"
        print(f"\n  VERDICT: {verdict}")

        import json
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved: {OUTPUT_PATH}")
    else:
        print("  No completed trades in backtest period")

    return results


if __name__ == "__main__":
    run_vix_puts_backtest()
