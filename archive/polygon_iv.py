
import os, sys, json, time
import numpy as np
import pandas as pd
import requests

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")
BASE        = "https://api.polygon.io"
CACHE_DIR   = "/Users/rick/ai_trading_bot_v2/cache_options_iv"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_iv_snapshot(symbol):
    cache_path = os.path.join(CACHE_DIR, symbol + "_iv.json")
    if os.path.exists(cache_path):
        if time.time() - os.path.getmtime(cache_path) < 86400:
            with open(cache_path) as f:
                return json.load(f)
    try:
        url  = BASE + "/v3/snapshot/options/" + symbol
        params = {"limit": 50, "apiKey": POLYGON_KEY}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return {}
        results = resp.json().get("results", [])
        if not results:
            return {}

        # Extract IV and Greeks from ATM options
        calls = [r for r in results if r.get("details", {}).get("contract_type") == "call"]
        puts  = [r for r in results if r.get("details", {}).get("contract_type") == "put"]

        def get_atm(opts, spot):
            if not opts or not spot:
                return None
            return min(opts, key=lambda x: abs(x.get("details", {}).get("strike_price", 9999) - spot))

        # Get spot price from first result
        spot = 0.0
        if results:
            spot = float(results[0].get("underlying_asset", {}).get("price", 0))

        atm_call = get_atm(calls, spot)
        atm_put  = get_atm(puts, spot)

        iv_call = float(atm_call.get("implied_volatility", 0)) if atm_call else 0.0
        iv_put  = float(atm_put.get("implied_volatility", 0)) if atm_put else 0.0
        iv_avg  = (iv_call + iv_put) / 2 if (iv_call + iv_put) > 0 else 0.0

        # IV rank — need historical IV for this, approximate from recent range
        # Will be properly computed once we have historical data
        result = {
            "symbol":   symbol,
            "spot":     spot,
            "iv_call":  iv_call,
            "iv_put":   iv_put,
            "iv_atm":   iv_avg,
            "iv_skew":  float(iv_put - iv_call),  # put IV > call IV = bearish skew
            "n_calls":  len(calls),
            "n_puts":   len(puts),
            "pc_ratio": len(puts) / max(len(calls), 1),
        }
        with open(cache_path, "w") as f:
            json.dump(result, f)
        return result
    except Exception as e:
        return {"error": str(e)}

def compute_iv_features(symbol):
    defaults = {
        "iv_atm": 0.0, "iv_skew": 0.0,
        "iv_high": 0.0, "iv_low": 0.0, "pc_ratio_options": 0.0
    }
    snap = fetch_iv_snapshot(symbol)
    if not snap or "error" in snap or snap.get("iv_atm", 0) == 0:
        return defaults
    iv = snap["iv_atm"]
    return {
        "iv_atm":           float(np.clip(iv, 0, 2.0)),
        "iv_skew":          float(np.clip(snap.get("iv_skew", 0), -0.5, 0.5)),
        "iv_high":          float(iv > 0.60),   # high IV = sell premium
        "iv_low":           float(iv < 0.25),   # low IV = buy options
        "pc_ratio_options": float(np.clip(snap.get("pc_ratio", 1), 0, 3)),
    }

if __name__ == "__main__":
    import config
    if not POLYGON_KEY:
        print("ERROR: POLYGON_API_KEY not set")
        print("Run: export POLYGON_API_KEY=your_key_here")
        exit(1)

    print("Fetching IV snapshots from Polygon...")
    results = []
    for sym in config.WATCHLIST[:50]:
        snap = fetch_iv_snapshot(sym)
        feats = compute_iv_features(sym)
        feats["symbol"] = sym
        results.append(feats)
        if snap.get("iv_atm", 0) > 0:
            print("  " + sym + ": IV=" + str(round(snap["iv_atm"]*100,1)) + "% skew=" + str(round(snap.get("iv_skew",0)*100,1)) + "% pcr=" + str(round(snap.get("pc_ratio",0),2)))
        time.sleep(0.3)

    df = pd.DataFrame(results)
    high_iv = df[df["iv_high"] > 0]
    low_iv  = df[df["iv_low"] > 0]
    print("High IV (>60%, sell premium): " + str(list(high_iv["symbol"])))
    print("Low IV (<25%, buy options):   " + str(list(low_iv["symbol"])))

    with open(os.path.join(CACHE_DIR, "iv_signals.json"), "w") as f:
        json.dump(results, f)
    print("Saved iv_signals.json")
