
import os, sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, "/Users/rick/ai_trading_bot_v2")

CACHE_DIR = "/Users/rick/ai_trading_bot_v2/cache_insider"

def compute_insider_sequence(symbol):
    """
    Sequence signal: buy AFTER a cluster of insider purchases ENDS.
    Alpha Architect: long-short earns 22-32% annualized alpha.
    Logic: insiders buy over several months, then stop = stock ready to move.
    """
    cache_path = os.path.join(CACHE_DIR, symbol + "_form4.json")
    if not os.path.exists(cache_path):
        return {"insider_seq_buy": 0.0, "insider_seq_sell": 0.0, "insider_seq_strength": 0.0}
    with open(cache_path) as f:
        txns = json.load(f)
    if not txns:
        return {"insider_seq_buy": 0.0, "insider_seq_sell": 0.0, "insider_seq_strength": 0.0}

    df = pd.DataFrame(txns)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    now = pd.Timestamp.now()
    buys  = df[df["is_buy"] == True].copy()
    sells = df[df["is_sell"] == True].copy()

    # Sequence: group buys by 30-day windows
    # A sequence = 2+ buys within 90 days
    # Sequence END = no buys in last 30 days after a sequence
    seq_buy = 0.0
    seq_strength = 0.0
    if len(buys) >= 2:
        recent_90d = buys[buys["date"] >= now - pd.Timedelta(days=90)]
        recent_30d = buys[buys["date"] >= now - pd.Timedelta(days=30)]
        prev_90d   = buys[(buys["date"] >= now - pd.Timedelta(days=180)) &
                          (buys["date"] < now - pd.Timedelta(days=90))]
        # Sequence ending: had buys 90-180 days ago, none in last 30 days
        if len(prev_90d) >= 2 and len(recent_30d) == 0:
            seq_buy = 1.0
            seq_strength = float(min(len(prev_90d) / 5.0, 1.0))
        # Active sequence: buys in last 90 days, continuing
        elif len(recent_90d) >= 2:
            seq_buy = 0.5  # partial signal — sequence ongoing
            seq_strength = float(min(len(recent_90d) / 5.0, 0.8))

    seq_sell = 0.0
    if len(sells) >= 2:
        recent_30d_s = sells[sells["date"] >= now - pd.Timedelta(days=30)]
        prev_90d_s   = sells[(sells["date"] >= now - pd.Timedelta(days=180)) &
                             (sells["date"] < now - pd.Timedelta(days=90))]
        if len(prev_90d_s) >= 2 and len(recent_30d_s) == 0:
            seq_sell = 1.0

    return {
        "insider_seq_buy":      seq_buy,
        "insider_seq_sell":     seq_sell,
        "insider_seq_strength": seq_strength,
    }


if __name__ == "__main__":
    import config
    results = []
    for sym in config.WATCHLIST:
        r = compute_insider_sequence(sym)
        r["symbol"] = sym
        results.append(r)
    df = pd.DataFrame(results)
    seqs = df[df["insider_seq_buy"] > 0].sort_values("insider_seq_strength", ascending=False)
    print("Insider sequence BUY signals (sequence ended = enter now):")
    for _, r in seqs.iterrows():
        label = "SEQUENCE ENDED" if r["insider_seq_buy"] == 1.0 else "SEQUENCE ACTIVE"
        print("  " + r["symbol"] + ": strength=" + str(round(r["insider_seq_strength"],2)) + " [" + label + "]")
    with open(os.path.join(CACHE_DIR, "insider_sequence.json"), "w") as f:
        json.dump(results, f)
    print("Saved insider_sequence.json")
