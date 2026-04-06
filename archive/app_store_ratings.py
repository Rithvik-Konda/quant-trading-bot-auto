
import os, sys, json, time
import numpy as np
import pandas as pd
import requests
sys.path.insert(0, "/Users/rick/ai_trading_bot_v2")

CACHE_DIR = "/Users/rick/ai_trading_bot_v2/cache_appstore"
os.makedirs(CACHE_DIR, exist_ok=True)

# Apple App IDs for our watchlist consumer tech companies
APP_IDS = {
    "UBER": "368677368",
    "ABNB": "401626263",
    "RDDT": "1064216828",
    "DUOL": "570060128",
    "COIN": "886427730",
    "SHOP": "462180Check",
    "SNAP": "447188370",
    "PINS": "778981118",
    "NFLX": "363590051",
    "LYFT": "529379082",
    "DASH": "1445056731",
    "HOOD": "1569179249",
    "SOFI": "1350498058",
    "PTON": "1187428213",
    "DKNG": "1175266940",
}

def fetch_app_rating(app_id, symbol):
    cache_path = os.path.join(CACHE_DIR, symbol + "_rating.json")
    if os.path.exists(cache_path):
        if time.time() - os.path.getmtime(cache_path) < 86400 * 7:
            with open(cache_path) as f:
                return json.load(f)
    result = {"symbol": symbol, "rating": 0.0, "rating_count": 0, "error": None}
    try:
        url = "https://itunes.apple.com/lookup?id=" + app_id + "&country=us"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("resultCount", 0) > 0:
                app = data["results"][0]
                result["rating"]       = float(app.get("averageUserRating", 0))
                result["rating_count"] = int(app.get("userRatingCount", 0))
                result["rating_5star"] = float(app.get("averageUserRatingForCurrentVersion", 0))
        time.sleep(0.3)
    except Exception as e:
        result["error"] = str(e)
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result

def compute_app_features(symbol):
    defaults = {"app_rating": 0.0, "app_rating_high": 0.0, "app_rating_low": 0.0}
    app_id = APP_IDS.get(symbol)
    if not app_id:
        return defaults
    r = fetch_app_rating(app_id, symbol)
    if r.get("error") or r["rating"] == 0:
        return defaults
    rating = r["rating"]
    return {
        "app_rating":      float(np.clip(rating / 5.0, 0, 1)),
        "app_rating_high": float(rating >= 4.5),
        "app_rating_low":  float(rating < 3.5),
    }

if __name__ == "__main__":
    print("Fetching App Store ratings...")
    results = []
    for sym, app_id in APP_IDS.items():
        r = fetch_app_rating(app_id, sym)
        feats = compute_app_features(sym)
        feats["symbol"] = sym
        feats["raw_rating"] = r.get("rating", 0)
        feats["count"] = r.get("rating_count", 0)
        results.append(feats)
        print("  " + sym + ": rating=" + str(r.get("rating", "ERR")) + " n=" + str(r.get("rating_count", 0)))
    df = pd.DataFrame(results)
    with open(os.path.join(CACHE_DIR, "app_ratings.json"), "w") as f:
        json.dump(results, f)
    print("Saved app_ratings.json")
    print("High rating (>=4.5): " + str(list(df[df["app_rating_high"]>0]["symbol"])))
    print("Low rating (<3.5):   " + str(list(df[df["app_rating_low"]>0]["symbol"])))
