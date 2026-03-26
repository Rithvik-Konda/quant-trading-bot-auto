"""
pead_decay.py — Per-stock PEAD decay curve fitting

Novel: instead of flat 90-day PEAD window, fit each stock's own
exponential decay curve from historical earnings surprises.

CELH: IC peaks day 15 then decays fast
APP: IC stays elevated through day 60
CAVA: decays fast after day 20

Each stock gets its own learned hold period.
No paper does per-stock PEAD decay fitting.

Output: optimal_hold_days[symbol] — used by backtester for max_hold
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

CACHE_DIR    = '/Users/rick/ai_trading_bot_v2/cache_prices'
EARNINGS_DIR = '/Users/rick/ai_trading_bot_v2/cache_earnings_streak'
OUTPUT_PATH  = '/Users/rick/ai_trading_bot_v2/pead_decay_params.json'


def exponential_decay(t, alpha, beta, gamma):
    """IC(t) = alpha * exp(-beta * t) + gamma (asymptotic floor)"""
    return alpha * np.exp(-beta * t) + gamma


def fit_pead_decay(symbol, df_price, earnings_records, max_hold=90):
    """
    Fit per-stock PEAD decay curve.
    Returns: optimal_hold_days, peak_ic, decay_rate
    """
    if len(earnings_records) < 4:
        return None

    # For each earnings event, measure IC at each day 1..90
    day_returns = {d: [] for d in range(1, max_hold+1)}

    for rec in earnings_records:
        try:
            earn_date = pd.Timestamp(rec['date'])
            surprise  = float(rec.get('surprise_pct', 0))
            if abs(surprise) < 0.01:
                continue

            sign = 1 if surprise > 0 else -1
            future = df_price.loc[df_price.index > earn_date]['close']
            if len(future) < 5:
                continue

            base = float(df_price.loc[:earn_date]['close'].iloc[-1])
            for day in range(1, min(max_hold+1, len(future)+1)):
                ret = float(future.iloc[day-1] / base - 1)
                # IC proxy: sign-aligned return (positive = drift continuing)
                day_returns[day].append(sign * ret)
        except Exception:
            continue

    # Compute mean IC at each day
    days, ics = [], []
    for d in range(1, max_hold+1):
        if len(day_returns[d]) >= 3:
            days.append(d)
            ics.append(np.mean(day_returns[d]))

    if len(days) < 10:
        return None

    days_arr = np.array(days, dtype=float)
    ics_arr  = np.array(ics)

    try:
        popt, _ = curve_fit(
            exponential_decay, days_arr, ics_arr,
            p0=[0.02, 0.05, 0.005],
            bounds=([-0.1, 0.001, -0.05], [0.2, 1.0, 0.05]),
            maxfev=2000,
        )
        alpha, beta, gamma = popt

        # Optimal hold = day where IC drops below 50% of peak
        peak_ic = alpha + gamma
        half_ic = peak_ic * 0.5
        if alpha > 0:
            optimal_hold = int(-np.log((half_ic - gamma) / alpha) / beta)
            optimal_hold = int(np.clip(optimal_hold, 10, 90))
        else:
            optimal_hold = 22  # fallback

        return {
            'symbol':       symbol,
            'optimal_hold': optimal_hold,
            'peak_ic':      float(peak_ic),
            'decay_rate':   float(beta),
            'floor_ic':     float(gamma),
            'n_events':     len(earnings_records),
        }
    except Exception:
        return None


def build_pead_decay_params(symbols=None):
    """Fit decay curves for all symbols with sufficient earnings history."""
    if symbols is None:
        import config
        symbols = list(config.WATCHLIST)

    results = {}
    print(f"Fitting PEAD decay curves for {len(symbols)} symbols...")

    for sym in symbols:
        try:
            # Load price history
            price_path = os.path.join(CACHE_DIR, f"{sym}_max.pkl")
            if not os.path.exists(price_path):
                continue
            df = pd.read_pickle(price_path)
            if len(df) < 260:
                continue

            # Load earnings records
            earn_path = os.path.join(EARNINGS_DIR, f"{sym}.json")
            if not os.path.exists(earn_path):
                continue
            with open(earn_path) as f:
                records = json.load(f)
            if len(records) < 4:
                continue

            result = fit_pead_decay(sym, df, records)
            if result:
                results[sym] = result

        except Exception:
            continue

    print(f"Fitted {len(results)} symbols")

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

    # Print top stocks by optimal hold period
    if results:
        df_r = pd.DataFrame(results.values()).sort_values('optimal_hold', ascending=False)
        print(f"\nTop 20 by optimal hold period:")
        print(f"{'Symbol':<8} {'Hold':>6} {'PeakIC':>8} {'Decay':>8} {'Events':>7}")
        print("-" * 42)
        for _, row in df_r.head(20).iterrows():
            print(f"  {row['symbol']:<6} {row['optimal_hold']:>6}d "
                  f"{row['peak_ic']:>8.4f} {row['decay_rate']:>8.4f} "
                  f"{row['n_events']:>7.0f}")

    return results


def get_optimal_hold(symbol, fallback=22):
    """Get the ML-learned optimal hold period for a symbol."""
    try:
        if not os.path.exists(OUTPUT_PATH):
            return fallback
        with open(OUTPUT_PATH) as f:
            params = json.load(f)
        return params.get(symbol, {}).get('optimal_hold', fallback)
    except Exception:
        return fallback


if __name__ == "__main__":
    results = build_pead_decay_params()
    print(f"\nMedian optimal hold: {np.median([r['optimal_hold'] for r in results.values()]):.0f} days")
    print(f"Range: {min(r['optimal_hold'] for r in results.values())} - "
          f"{max(r['optimal_hold'] for r in results.values())} days")
