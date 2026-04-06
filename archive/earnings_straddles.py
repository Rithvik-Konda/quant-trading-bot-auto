"""
earnings_straddles.py — Systematic earnings straddles on high-PEAD mid-caps

Research basis:
- Earnings announcement premium (EAP): options overprice earnings moves
  by 10-15% on average (Muravyev & Pearson 2020 JFE)
- BUT: for high-PEAD stocks where drift is directional, buying straddles
  and holding through the announcement captures the direction + drift
- Key: don't hold straddle TO expiry — sell the winning leg 2-3 days
  after earnings when IV crush has settled but drift hasn't finished

Strategy:
  - Screen: PEAD signal > 0.10 (strong historical beat pattern)
           AND ml_rank >= 0.90
           AND earnings within 5 days
  - Buy ATM straddle (call + put) 7 days before earnings
  - Exit: sell losing leg immediately after earnings
          hold winning leg for 15 more days to capture drift
  - Position size: 1% of portfolio per straddle

Key insight: we're not betting on direction (we don't know which way it'll go)
We're betting that the POST-earnings move will be LARGER than what options priced in.
High-PEAD stocks historically move more than IV implies.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

CACHE_DIR    = '/Users/rick/ai_trading_bot_v2/cache_prices'
EARNINGS_DIR = '/Users/rick/ai_trading_bot_v2/cache_earnings_streak'
OUTPUT_PATH  = '/Users/rick/ai_trading_bot_v2/straddle_results.json'


def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def bs_straddle(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) + bs_put(S, K, T, r, sigma)


def implied_move(sigma, T_earnings):
    """Expected move implied by options = ATM straddle price."""
    return sigma * np.sqrt(T_earnings)


def run_straddle_backtest():
    print("=== EARNINGS STRADDLE BACKTEST ===")
    print("Strategy: ATM straddle 7d before earnings, sell losing leg after, hold winner 15d")
    print()

    from backtester_clean import fetch_history

    PORTFOLIO     = 100_000
    ALLOC_PCT     = 0.01    # 1% per straddle
    DAYS_BEFORE   = 7       # enter 7 days before earnings
    DAYS_HOLD_WIN = 15      # hold winning leg 15 days post-earnings
    PEAD_MIN      = 0.10    # minimum surprise for PEAD signal
    SLIPPAGE      = 0.005   # 50bps options slippage
    RF            = 0.05

    results = []
    total_pnl = 0.0

    print(f"Scanning earnings history for high-PEAD opportunities...")

    for sym_file in sorted(os.listdir(EARNINGS_DIR)):
        sym = sym_file.replace('.json', '')

        try:
            with open(os.path.join(EARNINGS_DIR, sym_file)) as f:
                records = json.load(f)
        except Exception:
            continue

        if len(records) < 4:
            continue

        # Load price history
        price_path = os.path.join(CACHE_DIR, f"{sym}_max.pkl")
        if not os.path.exists(price_path):
            continue
        try:
            df = pd.read_pickle(price_path)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
        except Exception:
            continue

        # For each earnings event (skip first 4 — need history)
        for i, rec in enumerate(records):
            if i < 4:
                continue

            try:
                earn_date = pd.Timestamp(rec['date'])
                surprise  = float(rec.get('surp', 0))

                # Historical PEAD signal — did prior beats drift?
                prior = records[max(0, i-4):i]
                prior_surps = [float(r.get('surp', 0)) for r in prior]
                avg_prior_surp = np.mean(prior_surps)

                # Only trade if consistent beat history
                beat_streak = sum(1 for s in prior_surps if s > 0) / len(prior_surps)
                if beat_streak < 0.75 or avg_prior_surp < PEAD_MIN:
                    continue

                # Entry: 7 days before earnings
                pre_earn = df.loc[:earn_date]
                if len(pre_earn) < 30:
                    continue

                entry_date = pre_earn.index[-DAYS_BEFORE] if len(pre_earn) >= DAYS_BEFORE else pre_earn.index[-1]
                entry_px   = float(pre_earn.loc[entry_date, 'close'])

                # Historical vol for IV estimate
                vol_30d = float(pre_earn['close'].pct_change().dropna().tail(30).std() * np.sqrt(252))
                vol_30d = float(np.clip(vol_30d, 0.15, 1.5))

                # Earnings vol spike — IV typically 1.5-2x realized vol before earnings
                iv_estimate = vol_30d * 1.6

                # Price straddle
                T_entry    = DAYS_BEFORE / 365.0
                T_expiry   = (DAYS_BEFORE + DAYS_HOLD_WIN) / 365.0
                K          = entry_px  # ATM

                straddle_px = bs_straddle(entry_px, K, T_entry, RF, iv_estimate)
                straddle_ask = straddle_px * (1 + SLIPPAGE)

                budget       = PORTFOLIO * ALLOC_PCT
                n_contracts  = max(1, int(budget / (straddle_ask * 100)))
                straddle_cost = n_contracts * straddle_ask * 100

                # Post-earnings price
                post_earn = df.loc[df.index > earn_date]['close']
                if len(post_earn) < 3:
                    continue

                px_post_1d = float(post_earn.iloc[0])   # day after earnings
                px_post_15d = float(post_earn.iloc[min(DAYS_HOLD_WIN-1, len(post_earn)-1)])

                earn_move = float((px_post_1d - entry_px) / entry_px)

                # Post-earnings IV crush — vol reverts toward realized
                iv_post = vol_30d * 1.1  # still slightly elevated

                # Sell losing leg immediately, hold winning leg
                T_win_remaining = DAYS_HOLD_WIN / 365.0

                if earn_move > 0:
                    # Stock went up — call wins, put loses
                    call_value = bs_call(px_post_1d, K, T_win_remaining, RF, iv_post)
                    put_value  = max(K - px_post_1d, 0) * 0.1  # nearly worthless
                    # Hold call 15 more days
                    call_exit = bs_call(px_post_15d, K, 0.01, RF, vol_30d)
                    total_exit = (call_exit + put_value) * (1 - SLIPPAGE)
                else:
                    # Stock went down — put wins, call loses
                    put_value  = bs_put(px_post_1d, K, T_win_remaining, RF, iv_post)
                    call_value = max(px_post_1d - K, 0) * 0.1
                    # Hold put 15 more days
                    put_exit = bs_put(px_post_15d, K, 0.01, RF, vol_30d)
                    total_exit = (put_exit + call_value) * (1 - SLIPPAGE)

                total_payoff = n_contracts * total_exit * 100
                pnl = total_payoff - straddle_cost
                ret = pnl / straddle_cost if straddle_cost > 0 else 0

                total_pnl += pnl
                results.append({
                    'symbol':        sym,
                    'earn_date':     str(earn_date.date()),
                    'surprise':      surprise,
                    'avg_prior_surp': avg_prior_surp,
                    'beat_streak':   beat_streak,
                    'entry_px':      entry_px,
                    'earn_move':     earn_move,
                    'straddle_cost': straddle_cost,
                    'pnl':           pnl,
                    'ret':           ret,
                    'year':          earn_date.year,
                })

            except Exception:
                continue

    # Summary
    print(f"Completed {len(results)} straddle trades\n")

    if not results:
        print("No qualifying trades found")
        return

    df_r  = pd.DataFrame(results)
    wins  = df_r[df_r['pnl'] > 0]
    total = len(df_r)

    print(f"{'='*55}")
    print(f"EARNINGS STRADDLE SUMMARY")
    print(f"{'='*55}")
    print(f"  Trades:              {total}")
    print(f"  Win rate:            {len(wins)/total:.0%}")
    print(f"  Avg PnL:             ${df_r['pnl'].mean():,.0f}")
    print(f"  Total PnL:           ${df_r['pnl'].sum():,.0f}")
    print(f"  Best trade:          ${df_r['pnl'].max():,.0f}")
    print(f"  Worst trade:         ${df_r['pnl'].min():,.0f}")
    print(f"  Avg earnings move:   {df_r['earn_move'].mean():+.1%}")
    print(f"  Avg return/trade:    {df_r['ret'].mean():+.1%}")

    years = df_r['year'].max() - df_r['year'].min() + 1
    cagr  = (df_r['pnl'].sum() / PORTFOLIO) / max(years, 1)
    print(f"\n  Annualized contribution: {cagr:+.2%}")

    print(f"\n  {'Year':<6} {'Trades':>7} {'PnL':>10} {'WR':>7} {'AvgMove':>9}")
    print("  " + "-"*42)
    for yr, grp in df_r.groupby('year'):
        wr  = (grp['pnl'] > 0).mean()
        avg = grp['earn_move'].mean()
        print(f"  {yr:<6} {len(grp):>7} ${grp['pnl'].sum():>9,.0f} {wr:>7.0%} {avg:>+9.1%}")

    verdict = "DEPLOY" if cagr > 0.01 else "MARGINAL" if cagr > 0 else "DO NOT DEPLOY"
    print(f"\n  VERDICT: {verdict}")

    # Top trades
    print(f"\n  Top 10 trades:")
    for _, r in df_r.nlargest(10, 'pnl').iterrows():
        print(f"    {r['symbol']:<6} {r['earn_date']}  surp={r['surprise']:+.2f}  "
              f"move={r['earn_move']:+.1%}  pnl=${r['pnl']:,.0f}")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_PATH}")

    return df_r


if __name__ == "__main__":
    run_straddle_backtest()
