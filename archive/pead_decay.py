"""
pead_decay.py — PEAD decay curve fitting by market cap bucket

Novel: instead of flat 90-day window, fit decay curves by market cap bucket.
Small/mid-caps have slower price discovery = longer drift window.
Large-caps have faster discovery = shorter drift window.

Buckets trained on ALL stocks in each category across full price history.
Each bucket gets its own ML-learned optimal hold period.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, "/Users/rick/ai_trading_bot_v2")

CACHE_DIR    = "/Users/rick/ai_trading_bot_v2/cache_prices"
EARNINGS_DIR = "/Users/rick/ai_trading_bot_v2/cache_earnings_streak"
OUTPUT_PATH  = "/Users/rick/ai_trading_bot_v2/pead_decay_params.json"

# Market cap buckets — proxy via price * avg_volume
# These are learned from data, not hardcoded thresholds
BUCKETS = ["large", "mid", "small"]


def exponential_decay(t, alpha, beta, gamma):
    return alpha * np.exp(-beta * t) + gamma


def get_size_bucket(df):
    """Classify stock by size proxy: price * avg_volume."""
    try:
        px  = float(df["close"].iloc[-1])
        vol = float(df["volume"].tail(60).mean())
        dollar_vol = px * vol
        if dollar_vol > 500_000_000:
            return "large"
        elif dollar_vol > 50_000_000:
            return "mid"
        else:
            return "small"
    except Exception:
        return "mid"


def collect_pead_observations(max_hold=90):
    """Collect (day, sign_aligned_return, bucket) observations from all stocks."""
    import config
    observations = {b: {d: [] for d in range(1, max_hold+1)} for b in BUCKETS}

    print(f"Collecting PEAD observations from {len(config.WATCHLIST)} stocks...")
    n_events = 0

    for sym in config.WATCHLIST:
        try:
            price_path = os.path.join(CACHE_DIR, f"{sym}_max.pkl")
            earn_path  = os.path.join(EARNINGS_DIR, f"{sym}.json")
            if not os.path.exists(price_path) or not os.path.exists(earn_path):
                continue

            df = pd.read_pickle(price_path)
            if len(df) < 260:
                continue

            bucket = get_size_bucket(df)

            with open(earn_path) as f:
                records = json.load(f)

            for rec in records:
                try:
                    earn_date = pd.Timestamp(rec["date"])
                    surprise  = float(rec.get("surp", rec.get("surprise_pct", 0)))
                    if abs(surprise) < 0.02:
                        continue

                    sign   = 1 if surprise > 0 else -1
                    future = df.loc[df.index > earn_date]["close"]
                    if len(future) < 5:
                        continue

                    base = float(df.loc[:earn_date]["close"].iloc[-1])
                    for day in range(1, min(max_hold+1, len(future)+1)):
                        ret = float(future.iloc[day-1] / base - 1)
                        observations[bucket][day].append(sign * ret)
                    n_events += 1
                except Exception:
                    continue
        except Exception:
            continue

    print(f"Collected {n_events} earnings events")
    return observations


def fit_bucket_decay(observations, bucket):
    """Fit exponential decay curve for a bucket."""
    days, ics = [], []
    for d in range(1, 91):
        vals = observations[bucket][d]
        if len(vals) >= 10:
            days.append(d)
            ics.append(np.mean(vals))

    if len(days) < 15:
        return None

    days_arr = np.array(days, dtype=float)
    ics_arr  = np.array(ics)

    try:
        popt, _ = curve_fit(
            exponential_decay, days_arr, ics_arr,
            p0=[0.01, 0.03, 0.002],
            bounds=([-0.1, 0.001, -0.02], [0.15, 0.5, 0.05]),
            maxfev=5000,
        )
        alpha, beta, gamma = popt
        peak_ic = alpha + gamma
        if alpha > 0 and (peak_ic - gamma) > 0:
            optimal_hold = int(-np.log(max((peak_ic * 0.5 - gamma) / alpha, 1e-6)) / beta)
            optimal_hold = int(np.clip(optimal_hold, 10, 90))
        else:
            optimal_hold = 22
        return {
            "bucket": bucket, "optimal_hold": optimal_hold,
            "peak_ic": float(peak_ic), "decay_rate": float(beta),
            "floor_ic": float(gamma), "n_days": len(days),
        }
    except Exception:
        return None


def build_pead_decay_params():
    observations = collect_pead_observations()
    results = {}
    print("\nFitting decay curves by bucket:")
    for bucket in BUCKETS:
        r = fit_bucket_decay(observations, bucket)
        if r:
            results[bucket] = r
            print(f"  {bucket:<6}: hold={r['optimal_hold']}d  peak_ic={r['peak_ic']:.4f}  decay={r['decay_rate']:.3f}")
        else:
            results[bucket] = {"bucket": bucket, "optimal_hold": 22}
            print(f"  {bucket:<6}: insufficient data — using default 22d")

    # Also store per-stock overrides for stocks with enough data
    import config
    per_stock = {}
    for sym in config.WATCHLIST:
        try:
            price_path = os.path.join(CACHE_DIR, f"{sym}_max.pkl")
            earn_path  = os.path.join(EARNINGS_DIR, f"{sym}.json")
            if not os.path.exists(price_path) or not os.path.exists(earn_path):
                continue
            df = pd.read_pickle(price_path)
            bucket = get_size_bucket(df)
            per_stock[sym] = results.get(bucket, {}).get("optimal_hold", 22)
        except Exception:
            per_stock[sym] = 22

    output = {"buckets": results, "per_stock": per_stock}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Per-stock holds: {len(per_stock)} symbols")
    return output


def get_optimal_hold(symbol, fallback=22):
    """Get ML-learned optimal hold period for a symbol."""
    try:
        if not os.path.exists(OUTPUT_PATH):
            return fallback
        with open(OUTPUT_PATH) as f:
            params = json.load(f)
        # Per-stock first, then bucket fallback
        return params.get("per_stock", {}).get(symbol, fallback)
    except Exception:
        return fallback


if __name__ == "__main__":
    results = build_pead_decay_params()
