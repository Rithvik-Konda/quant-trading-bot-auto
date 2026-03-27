
import os, sys, json, time, re
import numpy as np
import pandas as pd
import requests
sys.path.insert(0, "/Users/rick/ai_trading_bot_v2")

CACHE_DIR = "/Users/rick/ai_trading_bot_v2/cache_fda"
os.makedirs(CACHE_DIR, exist_ok=True)
HEADERS = {"User-Agent": "quant-trading-bot admin@example.com"}

# Biotech in our watchlist
BIOTECH = ["VRTX","REGN","ALNY","INCY","NBIX","BIIB","MRNA","BNTX",
           "RARE","IONS","HALO","ACAD","PRGO","PCRX","HOLX","DXCM","PODD"]

def fetch_fda_pdufa(lookback_days=180):
    cache_path = os.path.join(CACHE_DIR, "pdufa_dates.json")
    if os.path.exists(cache_path):
        if time.time() - os.path.getmtime(cache_path) < 86400 * 7:
            with open(cache_path) as f:
                return json.load(f)
    dates = []
    try:
        # FDA action dates from public calendar
        url = "https://www.fda.gov/patients/drug-development-process/novel-drug-approvals-fda"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            # Parse approval dates and drug names from FDA page
            matches = re.findall(r"(\w+ \d{1,2}, \d{4})[^<]*<[^>]*>([^<]+)</", resp.text)
            for date_str, drug in matches[:50]:
                try:
                    dt = pd.Timestamp(date_str)
                    dates.append({"date": str(dt.date()), "drug": drug.strip()})
                except Exception:
                    continue
    except Exception as e:
        pass
    with open(cache_path, "w") as f:
        json.dump(dates, f)
    return dates

def compute_fda_features(symbol):
    defaults = {"fda_catalyst_near": 0.0, "fda_catalyst_days": 999.0}
    if symbol not in BIOTECH:
        return defaults
    # For now flag all BIOTECH as having potential catalyst
    # In live trading we will fetch actual PDUFA dates from Briefing.com or BioPharma Catalyst
    return {"fda_catalyst_near": 1.0, "fda_catalyst_days": 90.0}

if __name__ == "__main__":
    print("FDA Calendar (biotech event risk flags):")
    for sym in BIOTECH:
        f = compute_fda_features(sym)
        print("  " + sym + ": catalyst_near=" + str(f["fda_catalyst_near"]))
    # Save
    results = [{"symbol": s, **compute_fda_features(s)} for s in BIOTECH]
    with open(os.path.join(CACHE_DIR, "fda_signals.json"), "w") as f:
        json.dump(results, f)
    print("Saved fda_signals.json")
