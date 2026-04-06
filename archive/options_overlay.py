"""
options_overlay.py — Full options overlay suite

All strategies use BS pricing with realized vol estimates.
LIVE TESTING will validate which actually work — BS is directionally
correct but IV crush, bid-ask, and liquidity vary per stock.

Strategies:
1. Covered calls — sell calls on positions up >15%
2. Cash-secured puts — sell puts for CHOPPY entries  
3. Protective puts — buy puts on concentrated positions >10% portfolio
4. PEAD calls 90-DTE — buy calls on post-earnings drift candidates

All tracked separately in options_trades.json for live validation.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

CACHE_DIR  = '/Users/rick/ai_trading_bot_v2/cache_prices'
EARN_DIR   = '/Users/rick/ai_trading_bot_v2/cache_earnings_streak'
OUTPUT     = '/Users/rick/ai_trading_bot_v2/options_trades.json'
RISK_FREE  = 0.05
SLIP       = 0.005  # 50bps per leg


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


def bs_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return 1.0 if option_type == 'call' else -1.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    if option_type == 'call':
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1)


def get_vol(sym, df, days=30):
    """Annualized realized vol from price history."""
    if df is None or len(df) < days:
        return 0.35
    return float(np.clip(
        df['close'].pct_change().dropna().tail(days).std() * np.sqrt(252),
        0.10, 2.0
    ))


# ─────────────────────────────────────────────────────────────
# STRATEGY 1: Covered Calls
# Sell OTM calls on positions up >15% to collect premium
# Strike: 5-10% OTM, DTE: 30 days
# ─────────────────────────────────────────────────────────────

def covered_call_signal(
    sym: str,
    current_px: float,
    entry_px: float,
    df: pd.DataFrame,
    portfolio_value: float,
    position_value: float,
) -> dict:
    """
    Returns covered call trade if conditions met.
    Only sell when position is up >15% — letting winners run first.
    Strike = current_px * 1.07 (7% OTM) — enough premium, room to run.
    """
    unrealized = (current_px - entry_px) / entry_px
    if unrealized < 0.15:
        return {}  # not enough profit cushion

    sigma = get_vol(sym, df, 30)
    # Earnings IV premium — don't sell calls into earnings
    K   = current_px * 1.07  # 7% OTM
    T   = 30 / 365.0
    px  = bs_call(current_px, K, T, RISK_FREE, sigma)
    bid = px * (1 - SLIP)  # collect at bid

    if bid < 0.01:
        return {}

    # Shares already owned — sell 1 contract per 100 shares
    shares = int(position_value / current_px)
    n_contracts = max(1, shares // 100)

    premium = n_contracts * bid * 100

    return {
        'strategy':    'covered_call',
        'symbol':      sym,
        'stock_px':    current_px,
        'strike':      K,
        'dte':         30,
        'sigma':       sigma,
        'premium':     premium,
        'n_contracts': n_contracts,
        'unrealized':  unrealized,
        'breakeven':   entry_px,  # already profitable
        'max_gain_at': K,
        'notes':       f"Sell {n_contracts} calls at ${bid:.2f} each — position up {unrealized:.0%}",
    }


# ─────────────────────────────────────────────────────────────
# STRATEGY 2: Cash-Secured Puts
# Sell puts at target entry price in CHOPPY regime
# Collect premium if stock stays flat, own it cheaper if it drops
# ─────────────────────────────────────────────────────────────

def cash_secured_put_signal(
    sym: str,
    current_px: float,
    ml_rank: float,
    df: pd.DataFrame,
    portfolio_value: float,
    regime: str,
) -> dict:
    """
    In CHOPPY regime, instead of buying stock directly:
    Sell 30-DTE put at 5% below current price.
    - If stock stays flat or rises: keep premium (free yield)
    - If stock falls 5%: buy it at target price (what we wanted anyway)
    - Max loss: stock goes to zero (same as buying stock, but 5% cheaper)
    """
    if regime != 'CHOPPY':
        return {}
    if ml_rank < 0.90:
        return {}

    sigma = get_vol(sym, df, 30)
    K   = current_px * 0.95  # 5% OTM put = target entry price
    T   = 30 / 365.0
    px  = bs_put(current_px, K, T, RISK_FREE, sigma)
    bid = px * (1 - SLIP)

    if bid < 0.01:
        return {}

    # Cash needed to secure put
    cash_required = K * 100  # per contract
    budget        = portfolio_value * 0.03  # 3% of portfolio
    n_contracts   = max(1, int(budget / cash_required))
    premium       = n_contracts * bid * 100

    return {
        'strategy':      'cash_secured_put',
        'symbol':        sym,
        'stock_px':      current_px,
        'strike':        K,
        'dte':           30,
        'sigma':         sigma,
        'premium':       premium,
        'n_contracts':   n_contracts,
        'cash_required': n_contracts * cash_required,
        'effective_cost': K - bid,  # net cost if assigned
        'notes':         f"Sell {n_contracts} puts at ${bid:.2f} — effective entry ${K-bid:.2f} vs ${current_px:.2f}",
    }


# ─────────────────────────────────────────────────────────────
# STRATEGY 3: Protective Puts
# Buy puts on positions >10% of portfolio
# Caps catastrophic loss on concentrated winners
# ─────────────────────────────────────────────────────────────

def protective_put_signal(
    sym: str,
    current_px: float,
    entry_px: float,
    df: pd.DataFrame,
    portfolio_value: float,
    position_value: float,
) -> dict:
    """
    When a position grows >10% of portfolio, buy puts to cap downside.
    Strike: 10% OTM put — protects against catastrophic loss.
    Cost: small relative to position size.
    Only buy when position has unrealized gain > cost of put.
    """
    position_weight = position_value / portfolio_value
    if position_weight < 0.10:
        return {}  # not concentrated enough

    sigma = get_vol(sym, df, 30)
    K   = current_px * 0.90  # 10% OTM — catastrophe protection
    T   = 90 / 365.0         # 90 DTE — covers a full quarter
    px  = bs_put(current_px, K, T, RISK_FREE, sigma)
    ask = px * (1 + SLIP)    # pay the ask

    shares      = int(position_value / current_px)
    n_contracts = max(1, shares // 100)
    cost        = n_contracts * ask * 100

    # Only worth it if we have unrealized gain to fund the put
    unrealized = (current_px - entry_px) * shares
    if cost > unrealized * 0.20:  # don't spend more than 20% of gains
        return {}

    return {
        'strategy':         'protective_put',
        'symbol':           sym,
        'stock_px':         current_px,
        'strike':           K,
        'dte':              90,
        'sigma':            sigma,
        'cost':             cost,
        'n_contracts':      n_contracts,
        'position_weight':  position_weight,
        'max_loss_without': position_value * 0.10,  # 10% drop unprotected
        'max_loss_with':    cost + position_value * 0.10,  # limited to strike
        'notes':            f"Buy {n_contracts} puts @${ask:.2f} — protects {position_weight:.0%} position",
    }


# ─────────────────────────────────────────────────────────────
# STRATEGY 4: PEAD Calls 90-DTE
# Buy calls on stocks with strong post-earnings drift signal
# 90-DTE gives full drift window without needing perfect timing
# ─────────────────────────────────────────────────────────────

def pead_call_signal(
    sym: str,
    current_px: float,
    df: pd.DataFrame,
    portfolio_value: float,
    ml_rank: float,
) -> dict:
    """
    After an earnings beat with strong PEAD history:
    Buy 90-DTE call at 5% OTM.
    Captures the post-earnings drift without needing to time entry.
    IC=0.043 confirmed — drift persists 60-90 days for large caps.
    """
    # Check PEAD signal from earnings cache
    earn_path = os.path.join(EARN_DIR, f"{sym}.json")
    if not os.path.exists(earn_path):
        return {}

    try:
        with open(earn_path) as f:
            records = json.load(f)
    except Exception:
        return {}

    if len(records) < 4:
        return {}

    # Most recent earnings
    latest = records[-1]
    surp   = float(latest.get('surp', 0))
    if surp < 0.05:
        return {}  # not a strong enough beat

    # Historical drift — did prior beats drift?
    prior  = records[-5:-1]
    surps  = [float(r.get('surp', 0)) for r in prior]
    beats  = [s > 0 for s in surps]
    if sum(beats) < 3:
        return {}  # not consistent enough

    sigma = get_vol(sym, df, 60)
    K     = current_px * 1.05  # 5% OTM
    T     = 90 / 365.0
    px    = bs_call(current_px, K, T, RISK_FREE, sigma)
    ask   = px * (1 + SLIP)

    if ask < 0.01:
        return {}

    budget      = portfolio_value * 0.005  # 0.5% of portfolio
    n_contracts = max(1, int(budget / (ask * 100)))
    cost        = n_contracts * ask * 100

    # Expected payoff at 90 days if drift = IC × sigma × sqrt(90/252)
    expected_drift = 0.043 * np.sqrt(90/252)  # from IC research
    S_expected     = current_px * (1 + expected_drift)
    expected_payoff = max(S_expected - K, 0) * n_contracts * 100
    expected_ret    = (expected_payoff - cost) / cost if cost > 0 else 0

    return {
        'strategy':       'pead_call',
        'symbol':         sym,
        'stock_px':       current_px,
        'strike':         K,
        'dte':            90,
        'sigma':          sigma,
        'cost':           cost,
        'n_contracts':    n_contracts,
        'latest_surp':    surp,
        'beat_rate':      sum(beats)/len(beats),
        'expected_drift': expected_drift,
        'expected_ret':   expected_ret,
        'notes':          f"Buy {n_contracts} calls @${ask:.2f} — PEAD drift {expected_drift:.1%} expected",
    }


# ─────────────────────────────────────────────────────────────
# BACKTEST ALL STRATEGIES
# ─────────────────────────────────────────────────────────────

def run_options_backtest():
    """
    Retrospective test on historical trades.
    For each completed trade, compute what options overlay would have done.
    """
    from backtester_clean import fetch_history
    import config

    print("=== OPTIONS OVERLAY BACKTEST ===")
    print("Testing all 4 strategies on historical trades\n")

    df_trades = pd.read_csv('/Users/rick/ai_trading_bot_v2/trades_v2.csv')
    df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
    df_trades['exit_date']  = pd.to_datetime(df_trades['exit_date'])
    df_long = df_trades[(df_trades['side'] == 'long')].copy()

    PORTFOLIO = 100_000
    results = {
        'covered_call':     [],
        'cash_secured_put': [],
        'protective_put':   [],
        'pead_call':        [],
    }

    print(f"Analyzing {len(df_long)} trades...")

    for _, trade in df_long.iterrows():
        sym      = trade['symbol']
        entry_px = float(trade['entry_price'])
        exit_px  = float(trade['exit_price'])
        entry_dt = trade['entry_date']
        exit_dt  = trade['exit_date']
        ml_rank  = float(trade.get('ml_rank_pct', 0.9))
        regime   = str(trade.get('regime', 'TRENDING_BULL'))
        pnl_base = float(trade['pnl'])

        try:
            df_h = fetch_history(sym, days=9999)
            if df_h is None or len(df_h) < 30:
                continue
            df_h.index = pd.to_datetime(df_h.index).tz_localize(None)
            df_h.columns = [c.lower() for c in df_h.columns]
            df_pre = df_h.loc[:entry_dt]

            hold_days      = (exit_dt - entry_dt).days
            position_value = entry_px * max(1, int(3500 / entry_px))

            # Strategy 1: Covered call — check at entry if position up >15%
            unrealized_at_entry = (entry_px - entry_px) / entry_px  # = 0 at entry
            # Check mid-hold when position is up
            mid_dt = entry_dt + pd.Timedelta(days=hold_days//2)
            df_mid = df_h.loc[:mid_dt]
            if len(df_mid) > 0:
                mid_px = float(df_mid['close'].iloc[-1])
                unr    = (mid_px - entry_px) / entry_px
                if unr >= 0.15:
                    sig = covered_call_signal(sym, mid_px, entry_px, df_mid, PORTFOLIO, position_value)
                    if sig:
                        # Payoff: if exit_px > strike, call exercised = cap gain there
                        K_cc = sig['strike']
                        T_rem = max((exit_dt - mid_dt).days, 1) / 365.0
                        sigma = sig['sigma']
                        if exit_px >= K_cc:
                            # Call exercised — premium kept, gain capped at K_cc
                            cc_pnl = sig['premium']  # keep full premium
                        else:
                            # Call expires worthless — keep full premium
                            cc_pnl = sig['premium']
                        results['covered_call'].append({
                            'symbol': sym, 'year': entry_dt.year,
                            'premium': sig['premium'], 'pnl': cc_pnl,
                            'base_pnl': pnl_base,
                        })

            # Strategy 2: Cash-secured put
            csp = cash_secured_put_signal(sym, entry_px, ml_rank, df_pre, PORTFOLIO, regime)
            if csp:
                # If stock at exit > strike: keep premium (win)
                # If stock at exit < strike: assigned at strike (effectively bought stock)
                K_csp = csp['strike']
                if exit_px >= K_csp:
                    csp_pnl = csp['premium']  # keep premium
                else:
                    # Assigned — own stock at K_csp, now worth exit_px
                    assigned_pnl = (exit_px - K_csp) * csp['n_contracts'] * 100 + csp['premium']
                    csp_pnl = assigned_pnl
                results['cash_secured_put'].append({
                    'symbol': sym, 'year': entry_dt.year,
                    'premium': csp['premium'], 'pnl': csp_pnl,
                    'base_pnl': pnl_base,
                })

            # Strategy 3: Protective put — only for large positions
            pp_pos_val = min(position_value * 3, PORTFOLIO * 0.15)  # simulate larger position
            pp = protective_put_signal(sym, entry_px, entry_px, df_pre, PORTFOLIO, pp_pos_val)
            if pp:
                K_pp  = pp['strike']
                cost  = pp['cost']
                # Payoff: if stock drops below strike, put pays
                if exit_px < K_pp:
                    put_payoff = (K_pp - exit_px) * pp['n_contracts'] * 100
                    pp_pnl = put_payoff - cost
                else:
                    pp_pnl = -cost  # put expires worthless, paid insurance
                results['protective_put'].append({
                    'symbol': sym, 'year': entry_dt.year,
                    'cost': cost, 'pnl': pp_pnl,
                    'base_pnl': pnl_base,
                })

            # Strategy 4: PEAD calls
            pead = pead_call_signal(sym, entry_px, df_pre, PORTFOLIO, ml_rank)
            if pead:
                K_pead   = pead['strike']
                cost     = pead['cost']
                T_90     = 90 / 365.0
                sigma    = pead['sigma']
                # Find price 90 days after entry
                fut_90   = df_h.loc[df_h.index > entry_dt]['close']
                if len(fut_90) >= 90:
                    px_90    = float(fut_90.iloc[89])
                    payoff   = max(px_90 - K_pead, 0) * pead['n_contracts'] * 100
                    pead_pnl = payoff * (1 - SLIP) - cost
                else:
                    pead_pnl = -cost
                results['pead_call'].append({
                    'symbol': sym, 'year': entry_dt.year,
                    'cost': cost, 'pnl': pead_pnl,
                    'base_pnl': pnl_base,
                })

        except Exception:
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"OPTIONS OVERLAY RESULTS")
    print(f"{'='*60}")

    all_good = True
    for strat, trades in results.items():
        if not trades:
            print(f"\n  {strat}: no qualifying trades")
            continue

        df_r    = pd.DataFrame(trades)
        total   = len(df_r)
        wins    = (df_r['pnl'] > 0).sum()
        tot_pnl = df_r['pnl'].sum()
        avg_pnl = df_r['pnl'].mean()
        years   = df_r['year'].max() - df_r['year'].min() + 1
        cagr    = (tot_pnl / PORTFOLIO) / max(years, 1)

        verdict = "DEPLOY" if cagr > 0.005 else "MARGINAL" if cagr > 0 else "SKIP"
        print(f"\n  {strat.upper()}")
        print(f"    Trades:    {total}")
        print(f"    Win rate:  {wins/total:.0%}")
        print(f"    Total PnL: ${tot_pnl:,.0f}")
        print(f"    Avg PnL:   ${avg_pnl:,.0f}")
        print(f"    CAGR contribution: {cagr:+.2%}")
        print(f"    Verdict:   {verdict}")

        by_year = df_r.groupby('year')['pnl'].sum()
        for yr, pnl in by_year.items():
            print(f"      {yr}: ${pnl:,.0f}")

    # Save all signals for live tracking
    with open(OUTPUT, 'w') as f:
        json.dump({k: v[:20] for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT}")
    return results


def generate_live_signals(portfolio_positions: dict, portfolio_value: float, regime: str):
    """
    Generate today's options signals for live trading.
    portfolio_positions: {sym: {'entry_px': X, 'current_px': Y, 'value': Z}}
    """
    from backtester_clean import fetch_history
    import config

    signals = []
    print(f"\n=== TODAY'S OPTIONS SIGNALS (regime={regime}) ===\n")

    for sym, pos in portfolio_positions.items():
        try:
            df = fetch_history(sym, days=120)
            if df is None:
                continue
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = [c.lower() for c in df.columns]

            entry_px = pos['entry_px']
            curr_px  = pos.get('current_px', entry_px)
            pos_val  = pos.get('value', curr_px * 100)
            ml_rank  = pos.get('ml_rank', 0.90)

            # Check each strategy
            cc  = covered_call_signal(sym, curr_px, entry_px, df, portfolio_value, pos_val)
            pp  = protective_put_signal(sym, curr_px, entry_px, df, portfolio_value, pos_val)
            pead= pead_call_signal(sym, curr_px, df, portfolio_value, ml_rank)
            csp = cash_secured_put_signal(sym, curr_px, ml_rank, df, portfolio_value, regime)

            for sig in [cc, pp, pead, csp]:
                if sig:
                    sig['symbol'] = sym
                    signals.append(sig)
                    print(f"  {sig['strategy'].upper():<20} {sym:<6} {sig['notes']}")

        except Exception:
            continue

    if not signals:
        print("  No options signals today")

    return signals


if __name__ == "__main__":
    # Run backtest
    results = run_options_backtest()

    # Show live signals for current portfolio (example)
    print(f"\n=== EXAMPLE LIVE SIGNALS ===")
    example_positions = {
        'NVDA': {'entry_px': 120.0, 'current_px': 178.0, 'value': 17800, 'ml_rank': 0.97},
        'PLTR': {'entry_px': 80.0,  'current_px': 148.0, 'value': 14800, 'ml_rank': 0.95},
        'APP':  {'entry_px': 280.0, 'current_px': 396.0, 'value': 39600, 'ml_rank': 0.96},
    }
    generate_live_signals(example_positions, 100_000, 'TRENDING_BULL')
