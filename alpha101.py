"""
alpha101.py - WorldQuant 101 Formulaic Alphas (Kakushadze 2016)
30 cross-sectional factors from OHLCV. All rank-normalized daily.
"""
import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

def rank(x): return x.rank(pct=True)
def delay(x, d): return x.shift(d)
def delta(x, d): return x.diff(d)
def ts_rank(x, d): return x.rolling(d).rank(pct=True)
def ts_min(x, d): return x.rolling(d).min()
def ts_max(x, d): return x.rolling(d).max()
def ts_sum(x, d): return x.rolling(d).sum()
def ts_mean(x, d): return x.rolling(d).mean()
def ts_std(x, d): return x.rolling(d).std()
def correlation(x, y, d): return x.rolling(d).corr(y)
def covariance(x, y, d): return x.rolling(d).cov(y)
def sign(x): return np.sign(x)
def log(x): return np.log(x.clip(lower=1e-8))
def stddev(x, d): return x.rolling(d).std()

def compute_alpha101_features(df):
    if len(df) < 20:
        return pd.DataFrame()
    try:
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        v = df["volume"].astype(float).replace(0, np.nan)
        vwap = (h + l + c) / 3
        returns = c.pct_change()
        adv20 = v.rolling(20).mean()
        features = {}

        # Alpha 2
        try: features["a101_2"] = -1 * correlation(rank(delta(log(v), 2)), rank((c - o) / o.replace(0, np.nan)), 6)
        except: features["a101_2"] = 0.0

        # Alpha 3
        try: features["a101_3"] = -1 * correlation(rank(o), rank(v), 10)
        except: features["a101_3"] = 0.0

        # Alpha 6
        try: features["a101_6"] = -1 * correlation(o, v, 10)
        except: features["a101_6"] = 0.0

        # Alpha 7
        try:
            cond = adv20 < v
            val1 = -1 * ts_rank(delta(c, 7).abs(), 60) * sign(delta(c, 7))
            features["a101_7"] = pd.Series(np.where(cond, val1, -1), index=c.index)
        except: features["a101_7"] = 0.0

        # Alpha 9
        try:
            d1 = delta(c, 1)
            features["a101_9"] = pd.Series(np.where(ts_min(d1,5)>0, d1, np.where(ts_max(d1,5)<0, d1, -d1)), index=c.index)
        except: features["a101_9"] = 0.0

        # Alpha 11
        try:
            vc = vwap - c
            features["a101_11"] = (rank(ts_max(vc,3)) + rank(ts_min(vc,3))) * rank(delta(v,3))
        except: features["a101_11"] = 0.0

        # Alpha 12
        try: features["a101_12"] = sign(delta(v,1)) * (-1 * delta(c,1))
        except: features["a101_12"] = 0.0

        # Alpha 13
        try: features["a101_13"] = -1 * rank(covariance(rank(c), rank(v), 5))
        except: features["a101_13"] = 0.0

        # Alpha 14
        try: features["a101_14"] = -1 * rank(delta(returns,3)) * correlation(o, v, 10)
        except: features["a101_14"] = 0.0

        # Alpha 15
        try: features["a101_15"] = -1 * ts_sum(rank(correlation(rank(h), rank(v), 3)), 3)
        except: features["a101_15"] = 0.0

        # Alpha 16
        try: features["a101_16"] = -1 * rank(covariance(rank(h), rank(v), 5))
        except: features["a101_16"] = 0.0

        # Alpha 17
        try: features["a101_17"] = ((-1*rank(ts_rank(c,10)))*rank(delta(delta(c,1),1)))*rank(ts_rank(v/adv20,5))
        except: features["a101_17"] = 0.0

        # Alpha 18
        try: features["a101_18"] = -1*rank(stddev((c-o).abs(),5)+(c-o)+correlation(c,o,10))
        except: features["a101_18"] = 0.0

        # Alpha 20
        try: features["a101_20"] = ((-1*rank(o-delay(h,1)))*rank(o-delay(c,1)))*rank(o-delay(l,1))
        except: features["a101_20"] = 0.0

        # Alpha 22
        try: features["a101_22"] = -1*delta(correlation(h,v,5),5)*rank(stddev(c,20))
        except: features["a101_22"] = 0.0

        # Alpha 23
        try:
            cond = ts_mean(h,20) < h
            features["a101_23"] = pd.Series(np.where(cond, -1*delta(h,2), 0), index=c.index)
        except: features["a101_23"] = 0.0

        # Alpha 25
        try: features["a101_25"] = rank((-1*returns)*adv20*vwap*(h-c))
        except: features["a101_25"] = 0.0

        # Alpha 26
        try: features["a101_26"] = -1*ts_max(correlation(ts_rank(v,5),ts_rank(h,5),5),3)
        except: features["a101_26"] = 0.0

        # Alpha 33
        try: features["a101_33"] = rank(-1*(1-o/c.replace(0,np.nan)))
        except: features["a101_33"] = 0.0

        # Alpha 34
        try: features["a101_34"] = rank((1-rank(stddev(returns,2)/stddev(returns,5).replace(0,np.nan)))+(1-rank(delta(c,1))))
        except: features["a101_34"] = 0.0

        result = pd.DataFrame(features, index=df.index)
        return result.replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        return pd.DataFrame()


def get_alpha101_row(df):
    feat_df = compute_alpha101_features(df)
    if feat_df is None or len(feat_df) == 0:
        return {}
    row = feat_df.iloc[-1]
    return {k: float(v) if not np.isnan(float(v) if not isinstance(v, float) else v) else 0.0 for k, v in row.items()}


if __name__ == "__main__":
    import config
    from backtester_clean import fetch_history
    df = fetch_history("NVDA", days=500)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    feats = get_alpha101_row(df)
    print(f"Alpha101 features for NVDA: {len(feats)} factors")
    for k, v in sorted(feats.items()):
        print(f"  {k}: {v:.4f}")

    print("\nComputing ICs across 50 symbols...")
    ics = {}
    for sym in config.WATCHLIST[:50]:
        try:
            df_s = fetch_history(sym, days=500)
            if df_s is None or len(df_s) < 100: continue
            df_s.index = pd.to_datetime(df_s.index).tz_localize(None)
            feat_df = compute_alpha101_features(df_s)
            if feat_df is None or len(feat_df) < 6: continue
            fwd = df_s["close"].pct_change(5).shift(-5)
            aligned = feat_df.join(fwd.rename("fwd")).dropna()
            for col in [c for c in aligned.columns if c.startswith("a101")]:
                ic = aligned[col].corr(aligned["fwd"])
                if not np.isnan(ic):
                    ics.setdefault(col, []).append(ic)
        except Exception:
            continue

    print(f"\n{'Factor':<12} {'Mean IC':>10} {'Stability':>10}")
    print("-"*35)
    for factor, vals in sorted(ics.items()):
        if vals:
            mean_ic = np.mean(vals)
            stability = np.mean(np.array(vals) > 0)
            if abs(mean_ic) > 0.01:
                print(f"  {factor:<10} {mean_ic:>+10.4f} {stability:>10.0%}")
