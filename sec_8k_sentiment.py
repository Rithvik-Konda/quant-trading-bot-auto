
import os, sys, json, time, re
import numpy as np
import pandas as pd
import requests
sys.path.insert(0, "/Users/rick/ai_trading_bot_v2")

CACHE_DIR = "/Users/rick/ai_trading_bot_v2/cache_8k"
os.makedirs(CACHE_DIR, exist_ok=True)
HEADERS = {"User-Agent": "quant-trading-bot admin@example.com"}

# Loughran-McDonald positive/negative wordlists (abbreviated — most impactful words)
LM_POSITIVE = set(["achievement","achieve","achieved","benefited","beneficial","benefit",
    "breakthrough","efficient","excellent","favorable","improve","improved","improvement",
    "increase","increased","innovative","leading","momentum","opportunities","opportunity",
    "outstanding","positive","profitable","record","revenue","strong","stronger","success",
    "successful","superior","growth","accelerat","expand","expansion","exceed"])

LM_NEGATIVE = set(["adverse","adversely","below","challenging","concern","crisis","decline",
    "decreased","deficiency","delay","difficult","difficulty","disappoint","downgrade",
    "failure","harm","impair","impairment","impaired","inability","inadequate","loss",
    "losses","miss","missed","negative","negligible","penalty","problem","restructur",
    "risk","shortage","slowdown","uncertainty","unfavorable","volatility","weakness",
    "writedown","writeoff","alleged","allege","investigation","lawsuit","litigation"])

def fetch_8k_text(cik, symbol, lookback_days=90):
    cache_path = os.path.join(CACHE_DIR, symbol + "_8k.json")
    if os.path.exists(cache_path):
        if time.time() - os.path.getmtime(cache_path) < 86400 * 3:
            with open(cache_path) as f:
                return json.load(f)
    texts = []
    try:
        url = "https://data.sec.gov/submissions/CIK" + cik + ".json"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        filings = data["filings"]["recent"]
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        for i, form in enumerate(filings["form"]):
            if form != "8-K":
                continue
            if pd.Timestamp(filings["filingDate"][i]) < cutoff:
                continue
            acc    = filings["accessionNumber"][i]
            acc_nd = acc.replace("-", "")
            cik_i  = int(cik)
            idx_url = "https://www.sec.gov/Archives/edgar/data/" + str(cik_i) + "/" + acc_nd + "/" + acc + "-index.htm"
            try:
                idx_r = requests.get(idx_url, headers=HEADERS, timeout=8)
                # Find .htm filing document
                htm_links = re.findall(r'/Archives/edgar/data/[^\']*\.htm', idx_r.text)
                doc_links  = [l for l in htm_links if "index" not in l.lower() and "xsl" not in l.lower()]
                if doc_links:
                    doc_url = "https://www.sec.gov" + doc_links[0]
                    doc_r   = requests.get(doc_url, headers=HEADERS, timeout=10)
                    # Strip HTML tags
                    text = re.sub(r"<[^>]+>", " ", doc_r.text)
                    text = re.sub(r"\s+", " ", text).strip()
                    texts.append({"date": filings["filingDate"][i], "text": text[:5000]})
                time.sleep(0.2)
            except Exception:
                continue
            if len(texts) >= 5:  # limit to 5 most recent 8-Ks
                break
    except Exception as e:
        pass
    with open(cache_path, "w") as f:
        json.dump(texts, f)
    return texts

def score_8k_sentiment(texts):
    if not texts:
        return {"8k_sentiment": 0.0, "8k_positive": 0.0, "8k_negative": 0.0, "8k_count": 0.0}
    total_pos = total_neg = total_words = 0
    for item in texts:
        words = item["text"].lower().split()
        for w in words:
            w_clean = re.sub(r"[^a-z]", "", w)
            if not w_clean:
                continue
            # Check if word is in positive set OR starts with a positive stem
            is_pos = w_clean in LM_POSITIVE or any(w_clean.startswith(p) for p in LM_POSITIVE)
            is_neg = w_clean in LM_NEGATIVE or any(w_clean.startswith(n) for n in LM_NEGATIVE)
            if is_pos:
                total_pos += 1
            if is_neg:
                total_neg += 1
            total_words += 1
    if total_words == 0:
        return {"8k_sentiment": 0.0, "8k_positive": 0.0, "8k_negative": 0.0, "8k_count": 0.0}
    pos_rate = total_pos / total_words
    neg_rate = total_neg / total_words
    # Net sentiment: positive - negative, normalized
    net = pos_rate - neg_rate
    denom = pos_rate + neg_rate + 1e-8
    sentiment = net / denom
    return {
        "8k_sentiment": float(np.clip(sentiment, -1, 1)),
        "8k_positive":  float(np.clip(pos_rate * 100, 0, 10)),
        "8k_negative":  float(np.clip(neg_rate * 100, 0, 10)),
        "8k_count":     float(min(len(texts), 10)),
    }

def compute_8k_features(symbol, cik):
    texts = fetch_8k_text(cik, symbol)
    return score_8k_sentiment(texts)

if __name__ == "__main__":
    import config
    from insider_flow import get_cik
    print("Fetching 8-K sentiment for watchlist...")
    results = []
    for sym in config.WATCHLIST:
        cik = get_cik(sym)
        if not cik:
            continue
        feats = compute_8k_features(sym, cik)
        feats["symbol"] = sym
        results.append(feats)
        if feats["8k_count"] > 0:
            print("  " + sym + ": sentiment=" + str(round(feats["8k_sentiment"],3)) + " pos=" + str(round(feats["8k_positive"],2)) + " neg=" + str(round(feats["8k_negative"],2)) + " n=" + str(int(feats["8k_count"])))
        time.sleep(0.1)
    df = pd.DataFrame(results)
    with open(os.path.join(CACHE_DIR, "8k_sentiment.json"), "w") as f:
        json.dump(results, f)
    print("Saved 8k_sentiment.json")
    top_pos = df[df["8k_count"]>0].nlargest(5, "8k_sentiment")[["symbol","8k_sentiment","8k_count"]]
    top_neg = df[df["8k_count"]>0].nsmallest(5, "8k_sentiment")[["symbol","8k_sentiment","8k_count"]]
    print("Most positive 8-K language: " + str(list(top_pos["symbol"])))
    print("Most negative 8-K language: " + str(list(top_neg["symbol"])))
