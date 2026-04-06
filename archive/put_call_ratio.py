"""put_call_ratio.py — CBOE Put/Call Ratio contrarian signal"""
import os, sys, json, time
import numpy as np
import pandas as pd
import requests
sys.path.insert(0, "/Users/rick/ai_trading_bot_v2")

CACHE_DIR = "/Users/rick/ai_trading_bot_v2/cache_pcr"
os.makedirs(CACHE_DIR, exist_ok=True)
PCR_CACHE = os.path.join(CACHE_DIR, "pcr_history.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; quant-research)"}


def fetch_cboe_pcr():
    if os.path.exists(PCR_CACHE):
        mtime = os.path.getmtime(PCR_CACHE)
        if time.time() - mtime < 86400:
            return pd.read_csv(PCR_CACHE, index_col=0, parse_dates=True)
    urls = [
        "https://cdn.cboe.com/api/global/us_indices/daily_prices/PC_RATIO.csv",
    ]
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                if len(df) > 10:
                    df.to_csv(PCR_CACHE)
                    return df
        except Exception:
            continue
    # Fallback via VIX
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX").history(period="2y")
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        pcr_approx = 0.5 + (vix["Close"] - 15) / 50
        df_approx = pd.DataFrame({"pcr_total": pcr_approx.clip(0.4, 1.5)})
        df_approx.to_csv(PCR_CACHE)
        print("Using VIX-based P/C approximation")
        return df_approx
    except Exception:
        return pd.DataFrame()


def compute_pcr_features():
    defaults = {"pcr_total": 0.80, "pcr_equity": 0.65, "pcr_5d_ma": 0.80,
                "pcr_fear_signal": 0.0, "pcr_complacency": 0.0, "pcr_extreme": 0.0}
    try:
        df = fetch_cboe_pcr()
        if df is None or len(df) == 0:
            return defaults
        pcr_col = [c for c in df.columns if "total" in c.lower() or "pcr" in c.lower()]
        if not pcr_col:
            pcr_col = [df.columns[0]]
        pcr = df[pcr_col[0]].dropna()
        if len(pcr) == 0:
            return defaults
        pcr_now = float(pcr.iloc[-1])
        pcr_5d  = float(pcr.tail(5).mean())
        pcr_20d = float(pcr.tail(20).mean())
        pcr_std = float(pcr.tail(60).std()) if len(pcr) >= 60 else 0.1
        return {
            "pcr_total":       float(np.clip(pcr_now, 0.3, 2.0)),
            "pcr_equity":      0.65,
            "pcr_5d_ma":       float(np.clip(pcr_5d, 0.3, 2.0)),
            "pcr_fear_signal": float(pcr_now > 1.10),
            "pcr_complacency": float(pcr_now < 0.65),
            "pcr_extreme":     float(abs(pcr_now - pcr_20d) > 2 * pcr_std),
        }
    except Exception:
        return defaults


if __name__ == "__main__":
    print("=== PUT/CALL RATIO ===")
    feats = compute_pcr_features()
    for k, v in feats.items():
        print(f"  {k}: {v:.3f}")
    with open(os.path.join(CACHE_DIR, "pcr_current.json"), "w") as f:
        json.dump(feats, f)
    print("Saved pcr_current.json")
