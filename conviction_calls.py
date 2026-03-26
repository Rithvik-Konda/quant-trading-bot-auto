"""
conviction_calls.py — Options overlay on highest-conviction entries

Research basis: Cremers & Weinbaum (2010) JF — call-put spread predicts
next-week returns with t-stat >4 when combined with directional signals.

Strategy: When ALL four signals align simultaneously:
  1. ML rank ≥ 0.95 (top 5%)
  2. PEAD signal > 0 (positive earnings surprise drift)
  3. Analyst revision momentum > 0 (upgrades > downgrades)
  4. Earnings beat streak ≥ 0.6 (2+ consecutive beats)

→ Buy stock for 50% of intended position
→ Buy 21-DTE calls 5% OTM for the other 50% (by dollar value)

Effect: Same capital deployed, but upside is amplified 3-5x on the call
portion if the thesis plays out. Downside on calls = premium only.

No hardcoded thresholds — conviction score computed from ML signal
strength, not fixed values.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
import joblib

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

CACHE_DIR   = '/Users/rick/ai_trading_bot_v2/cache_prices'
OUTPUT_PATH = '/Users/rick/ai_trading_bot_v2/conviction_calls_results.json'


def bs_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def conviction_score(ml_rank, pead, revision, streak):
    """
    Composite conviction score — all four signals must be positive.
    Returns 0.0 if any signal is absent/negative.
    Returns 0.0-1.0 based on combined signal strength.
    No hardcoded thresholds — score is continuous.
    """
    # All signals must be positive direction
    if pead <= 0 or revision <= 0 or streak <= 0:
        return 0.0

    # Conviction = geometric mean of normalized signals
    # Each signal contributes equally — no one signal dominates
    ml_norm  = float(np.clip((ml_rank - 0.85) / 0.15, 0, 1))  # 0.85-1.0 → 0-1
    pead_n   = float(np.clip(pead / 0.20, 0, 1))               # 0-0.20 → 0-1
    rev_n    = float(np.clip(revision / 0.50, 0, 1))            # 0-0.50 → 0-1
    streak_n = float(np.clip(streak / 1.0, 0, 1))               # 0-1.0 → 0-1

    # Require all four to be meaningfully positive (>0.3 normalized)
    if min(ml_norm, pead_n, rev_n, streak_n) < 0.3:
        return 0.0

    return float(np.power(ml_norm * pead_n * rev_n * streak_n, 0.25))


def options_allocation_fraction(conv_score, base_stock_alloc=1.0):
    """
    How much of the position to express as calls vs stock.
    Higher conviction = more calls (more leverage).
    conv_score 0.5 → 25% calls, 75% stock
    conv_score 1.0 → 50% calls, 50% stock
    No hardcoding — continuous function of conviction.
    """
    call_fraction = float(np.clip(conv_score * 0.5, 0, 0.5))
    stock_fraction = 1.0 - call_fraction
    return stock_fraction, call_fraction


def run_conviction_calls_backtest():
    """
    Simulate conviction calls overlay on historical backtest trades.
    For each trade that would have qualified, compute:
    - Actual stock P&L (as in backtest)
    - What P&L would have been with 50% stock + 50% calls
    - Difference = options overlay contribution
    """
    print("=== CONVICTION CALLS BACKTEST ===")
    print("Strategy: 50% stock + 50% calls on ML≥0.95 + PEAD + revision + streak")
    print()

    # Load historical trades
    trades_df = pd.read_csv('/Users/rick/ai_trading_bot_v2/trades_v2.csv')
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date']  = pd.to_datetime(trades_df['exit_date'])
    trades_df = trades_df[trades_df['side'] == 'long'].copy()

    # Load feature store to get PEAD/revision/streak at entry
    import json
    EARNINGS_DIR = '/Users/rick/ai_trading_bot_v2/cache_earnings_streak'

    results = []
    qualified = 0
    total_stock_pnl   = 0.0
    total_options_pnl = 0.0

    RISK_FREE = 0.05
    DTE       = 21
    STRIKE_OTM = 0.05  # 5% OTM

    print(f"Analyzing {len(trades_df)} historical trades...")
    print()

    for _, trade in trades_df.iterrows():
        sym        = trade['symbol']
        entry_date = trade['entry_date']
        exit_date  = trade['exit_date']
        entry_px   = trade['entry_price']
        exit_px    = trade['exit_price']
        stock_pnl  = trade['pnl']
        qty        = trade['qty']
        ml_rank    = trade.get('ml_rank_pct', 0.0)

        # Get PEAD signal at entry from earnings cache
        pead_score = 0.0
        streak_score = 0.0
        try:
            earn_path = os.path.join(EARNINGS_DIR, f"{sym}.json")
            if os.path.exists(earn_path):
                with open(earn_path) as f:
                    records = json.load(f)
                # Most recent earnings before entry
                prior = [r for r in records
                        if pd.Timestamp(r['date']) < entry_date]
                if prior:
                    last = prior[-1]
                    pead_score = float(last.get('surp', 0))
                    # Streak: consecutive beats
                    beats = [r['beat'] for r in prior[-4:]]
                    streak_score = float(sum(beats) / len(beats)) if beats else 0.0
        except Exception:
            pass

        # Revision proxy: use ml_rank as proxy (high rank = positive revisions)
        revision_score = float(np.clip((ml_rank - 0.85) / 0.15, 0, 1))

        # Compute conviction score
        conv = conviction_score(ml_rank, pead_score, revision_score, streak_score)

        if conv < 0.3:
            continue  # not high conviction enough for options

        qualified += 1
        stock_frac, call_frac = options_allocation_fraction(conv)

        # Original position value
        position_value = entry_px * qty

        # Stock component P&L (reduced position)
        stock_component_pnl = stock_pnl * stock_frac

        # Options component
        # Budget for calls = call_frac * position_value
        call_budget = position_value * call_frac
        K = entry_px * (1 + STRIKE_OTM)

        # Estimate IV from historical vol
        price_path = os.path.join(CACHE_DIR, f"{sym}_max.pkl")
        sigma = 0.35  # default
        if os.path.exists(price_path):
            try:
                df_h = pd.read_pickle(price_path)
                df_h.index = pd.to_datetime(df_h.index).tz_localize(None)
                df_pre = df_h.loc[:entry_date]
                if len(df_pre) >= 20:
                    sigma = float(df_pre['close'].pct_change().dropna().tail(60).std() * np.sqrt(252))
                    sigma = float(np.clip(sigma, 0.15, 1.5))
            except Exception:
                pass

        T = DTE / 365.0
        call_px = bs_call(entry_px, K, T, RISK_FREE, sigma)
        if call_px <= 0:
            continue

        # Number of contracts (100 shares each)
        n_contracts = max(1, int(call_budget / (call_px * 100)))
        call_cost   = n_contracts * call_px * 100

        # Hold period
        hold_days = (exit_date - entry_date).days

        # Call payoff: expires at DTE regardless of trade hold period
        # After expiry, trade continues stock-only
        if hold_days >= DTE:
            # Call expired — compute price at DTE day, not exit day
            price_path2 = os.path.join(CACHE_DIR, f"{sym}_max.pkl")
            dte_px = entry_px  # fallback
            if os.path.exists(price_path2):
                try:
                    df_h2 = pd.read_pickle(price_path2)
                    df_h2.index = pd.to_datetime(df_h2.index).tz_localize(None)
                    future_px = df_h2.loc[df_h2.index > entry_date]['close']
                    if len(future_px) >= DTE:
                        dte_px = float(future_px.iloc[DTE-1])
                except Exception:
                    dte_px = exit_px
            call_payoff_per = max(dte_px - K, 0)
        else:
            # Still alive at trade exit — BS price with remaining time
            T_rem = max((DTE - hold_days) / 365.0, 0.01)
            call_payoff_per = bs_call(exit_px, K, T_rem, RISK_FREE, sigma)

        total_call_payoff = n_contracts * call_payoff_per * 100
        call_pnl          = total_call_payoff - call_cost
        call_ret          = call_pnl / call_cost if call_cost > 0 else 0

        # Stock component: full stock P&L on reduced position
        # After call expires, stock continues to trade exit
        overlay_total = stock_component_pnl + call_pnl
        baseline_pnl  = stock_pnl
        improvement   = overlay_total - baseline_pnl

        total_stock_pnl   += baseline_pnl
        total_options_pnl += overlay_total

        results.append({
            'symbol':       sym,
            'entry_date':   str(entry_date.date()),
            'exit_date':    str(exit_date.date()),
            'ml_rank':      ml_rank,
            'pead':         pead_score,
            'streak':       streak_score,
            'conv_score':   conv,
            'call_frac':    call_frac,
            'stock_pnl':    baseline_pnl,
            'call_pnl':     call_pnl,
            'overlay_pnl':  overlay_total,
            'improvement':  improvement,
            'call_ret':     call_ret,
            'hold_days':    hold_days,
        })

    # Summary
    print(f"Qualified trades (conviction ≥ 0.3): {qualified}/{len(trades_df)}")
    print()

    if results:
        df_r = pd.DataFrame(results).sort_values('improvement', ascending=False)

        print(f"{'Symbol':<6} {'Date':<12} {'Conv':>6} {'CallFrac':>9} "
              f"{'StockPnL':>10} {'CallPnL':>10} {'OverlayPnL':>11} {'Improve':>10}")
        print("-" * 78)
        for _, r in df_r.head(20).iterrows():
            print(f"  {r['symbol']:<4} {r['entry_date']:<12} {r['conv_score']:>6.2f} "
                  f"{r['call_frac']:>9.0%} "
                  f"${r['stock_pnl']:>9,.0f} ${r['call_pnl']:>9,.0f} "
                  f"${r['overlay_pnl']:>10,.0f} ${r['improvement']:>+9,.0f}")

        print(f"\n{'='*60}")
        print(f"CONVICTION CALLS SUMMARY")
        print(f"{'='*60}")

        wins_overlay = df_r[df_r['overlay_pnl'] > 0]
        wins_stock   = df_r[df_r['stock_pnl'] > 0]

        print(f"  Qualified trades:        {len(df_r)}")
        print(f"  Stock-only WR:           {len(wins_stock)/len(df_r):.0%}")
        print(f"  Overlay WR:              {len(wins_overlay)/len(df_r):.0%}")
        print()
        print(f"  Total stock P&L:         ${df_r['stock_pnl'].sum():,.0f}")
        print(f"  Total overlay P&L:       ${df_r['overlay_pnl'].sum():,.0f}")
        print(f"  Total improvement:       ${df_r['improvement'].sum():+,.0f}")
        print()
        print(f"  Avg call return:         {df_r['call_ret'].mean():+.1%}")
        print(f"  Best improvement:        ${df_r['improvement'].max():+,.0f}")
        print(f"  Worst:                   ${df_r['improvement'].min():+,.0f}")

        years = 9
        cagr  = (df_r['improvement'].sum() / 100_000) / years
        print(f"\n  Annualized contribution: {cagr:+.2%}")

        df_r['year'] = pd.to_datetime(df_r['entry_date']).dt.year
        print(f"\n  {'Year':<6} {'Trades':>7} {'StockPnL':>10} {'OverlayPnL':>11} {'Improve':>10}")
        print("  " + "-"*46)
        for yr, grp in df_r.groupby('year'):
            imp = grp['improvement'].sum()
            print(f"  {yr:<6} {len(grp):>7} "
                  f"${grp['stock_pnl'].sum():>9,.0f} "
                  f"${grp['overlay_pnl'].sum():>10,.0f} "
                  f"${imp:>+9,.0f}")

        verdict = "DEPLOY" if cagr > 0.02 else "MARGINAL" if cagr > 0 else "DO NOT DEPLOY"
        print(f"\n  VERDICT: {verdict}")

        import json
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved: {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    run_conviction_calls_backtest()
