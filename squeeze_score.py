
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/Users/rick/ai_trading_bot_v2")

CACHE_DIR = "/Users/rick/ai_trading_bot_v2/cache_si"

def compute_squeeze_score(symbol, price_df=None):
    """
    Combine SI velocity + momentum + days-to-cover into squeeze_score.
    High score = potential short squeeze setup.
    """
    defaults = {"squeeze_score": 0.0, "squeeze_alert": 0.0}
    try:
        si_path = os.path.join(CACHE_DIR, symbol + ".json")
        if not os.path.exists(si_path):
            return defaults
        with open(si_path) as f:
            si_data = json.load(f)
        if not si_data:
            return defaults

        si_pct   = float(si_data.get("si_pct", 0))
        si_chg   = float(si_data.get("si_change_4w", 0))
        dtc      = float(si_data.get("days_to_cover", 0))

        # Momentum from price (if available)
        mom = 0.0
        if price_df is not None and len(price_df) >= 20:
            px = price_df["close"]
            mom = float((px.iloc[-1] / px.iloc[-20] - 1))

        # Squeeze score: high SI + rising SI + rising price + high DTC
        # Each component 0-1, weighted sum
        si_score  = float(np.clip(si_pct / 30.0, 0, 1))          # >30% SI = max
        dtc_score = float(np.clip(dtc / 10.0, 0, 1))              # >10 DTC = max
        chg_score = float(np.clip(si_chg / 5.0, 0, 1))           # rising SI
        mom_score = float(np.clip(mom / 0.20, 0, 1))              # price up 20%

        squeeze = 0.35 * si_score + 0.25 * dtc_score + 0.25 * chg_score + 0.15 * mom_score

        return {
            "squeeze_score": float(np.clip(squeeze, 0, 1)),
            "squeeze_alert": float(squeeze > 0.65),
        }
    except Exception:
        return defaults

if __name__ == "__main__":
    import config
    from backtester_clean import fetch_history
    print("Computing squeeze scores...")
    results = []
    for sym in config.WATCHLIST:
        try:
            df = fetch_history(sym, days=60)
            if df is not None:
                df.index = pd.to_datetime(df.index).tz_localize(None)
        except Exception:
            df = None
        score = compute_squeeze_score(sym, df)
        score["symbol"] = sym
        results.append(score)
    df_r = pd.DataFrame(results)
    alerts = df_r[df_r["squeeze_alert"] > 0].sort_values("squeeze_score", ascending=False)
    print("SQUEEZE ALERTS (score > 0.65):")
    for _, r in alerts.iterrows():
        print("  " + r["symbol"] + ": score=" + str(round(r["squeeze_score"],3)))
    with open("/Users/rick/ai_trading_bot_v2/cache_si/squeeze_scores.json", "w") as f:
        json.dump(results, f)
    print("Saved squeeze_scores.json")
