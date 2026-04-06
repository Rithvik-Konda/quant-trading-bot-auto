"""congressional_flow.py — Congressional stock trades from STOCK Act disclosures"""
import os, sys, json, time
import numpy as np
import pandas as pd
import requests
sys.path.insert(0, "/Users/rick/ai_trading_bot_v2")

CACHE_DIR = "/Users/rick/ai_trading_bot_v2/cache_congress"
os.makedirs(CACHE_DIR, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; quant-research-bot)", "Accept": "application/json"}


def fetch_congress_trades(lookback_days=90):
    cache_path = os.path.join(CACHE_DIR, "congress_trades.json")
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime < 86400:
            with open(cache_path) as f:
                return pd.DataFrame(json.load(f))
    trades = []
    try:
        # Quiver Quantitative free API
        url = "https://api.quiverquant.com/beta/live/congresstrading"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for t in (data[:500] if isinstance(data, list) else []):
                try:
                    trades.append({
                        "date":    t.get("Date", t.get("TransactionDate", "")),
                        "member":  t.get("Representative", t.get("Senator", "")),
                        "party":   t.get("Party", ""),
                        "chamber": t.get("Chamber", ""),
                        "ticker":  t.get("Ticker", ""),
                        "type":    t.get("Transaction", ""),
                        "amount":  t.get("Range", ""),
                    })
                except Exception:
                    continue
    except Exception as e:
        print(f"Congressional API: {e}")
    if trades:
        with open(cache_path, "w") as f:
            json.dump(trades, f)
    return pd.DataFrame(trades) if trades else pd.DataFrame()


def compute_congress_features(symbol, df_trades=None):
    defaults = {"congress_buy_count": 0.0, "congress_sell_count": 0.0,
                "congress_cluster": 0.0, "congress_net_buying": 0.0, "congress_committee": 0.0}
    if df_trades is None or len(df_trades) == 0:
        return defaults
    try:
        sym_trades = df_trades[df_trades["ticker"].str.upper() == symbol.upper()].copy()
        if len(sym_trades) == 0:
            return defaults
        sym_trades["date"] = pd.to_datetime(sym_trades["date"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
        sym_trades = sym_trades[sym_trades["date"] >= cutoff]
        if len(sym_trades) == 0:
            return defaults
        buys  = sym_trades[sym_trades["type"].str.upper().str.contains("BUY|PURCHASE", na=False)]
        sells = sym_trades[sym_trades["type"].str.upper().str.contains("SELL|SALE", na=False)]
        buy_count = len(buys)
        sell_count = len(sells)
        unique_buyers = buys["member"].nunique() if len(buys) > 0 else 0
        cluster = float(unique_buyers >= 3)
        net = buy_count - sell_count
        total = buy_count + sell_count
        net_ratio = float(net / total) if total > 0 else 0.0
        return {"congress_buy_count": float(np.clip(buy_count,0,20)),
                "congress_sell_count": float(np.clip(sell_count,0,20)),
                "congress_cluster": cluster,
                "congress_net_buying": float(np.clip(net_ratio,-1,1)),
                "congress_committee": float(unique_buyers >= 2)}
    except Exception:
        return defaults


def scan_congress_signals(watchlist):
    print("Fetching congressional trades...")
    df_trades = fetch_congress_trades()
    print(f"  {len(df_trades)} trades fetched")
    rows = []
    for sym in watchlist:
        feats = compute_congress_features(sym, df_trades)
        feats["symbol"] = sym
        rows.append(feats)
    df = pd.DataFrame(rows)
    clusters = df[df["congress_cluster"] > 0]
    print(f"\nClusters found (3+ members buying): {len(clusters)}")
    for _, r in clusters.sort_values("congress_buy_count", ascending=False).iterrows():
        print(f"  {r['symbol']}: buys={r['congress_buy_count']:.0f} net={r['congress_net_buying']:+.1%}")
    with open(os.path.join(CACHE_DIR, "congress_signals.json"), "w") as f:
        json.dump(df.to_dict("records"), f)
    print("Saved congress_signals.json")
    return df


if __name__ == "__main__":
    import config
    scan_congress_signals(config.WATCHLIST)
