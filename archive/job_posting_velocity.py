import os, sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
CACHE_DIR = '/Users/rick/ai_trading_bot_v2/cache_jobs'
os.makedirs(CACHE_DIR, exist_ok=True)
COMPANY_MAP = {'NVDA':'NVIDIA','MSFT':'Microsoft','META':'Meta Platforms','GOOGL':'Google','AMZN':'Amazon','AAPL':'Apple','TSLA':'Tesla','NOW':'ServiceNow','SNOW':'Snowflake','DDOG':'Datadog','NET':'Cloudflare','CRWD':'CrowdStrike','PLTR':'Palantir','APP':'AppLovin','UBER':'Uber','ABNB':'Airbnb','SHOP':'Shopify','COIN':'Coinbase','AXON':'Axon','RDDT':'Reddit','DUOL':'Duolingo','TTD':'Trade Desk','HUBS':'HubSpot','WDAY':'Workday','ADBE':'Adobe','ORCL':'Oracle','CRM':'Salesforce','PANW':'Palo Alto Networks'}

def fetch_trends_batch(symbols, batch_size=5):
    from pytrends.request import TrendReq
    pytrends = TrendReq(hl='en-US', tz=360)
    results = {}
    syms = [s for s in symbols if s in COMPANY_MAP]
    for i in range(0, len(syms), batch_size):
        batch = syms[i:i+batch_size]
        kw_list = [COMPANY_MAP[s] + ' jobs' for s in batch]
        try:
            pytrends.build_payload(kw_list, timeframe='today 3-m', geo='US')
            df = pytrends.interest_over_time()
            if len(df) == 0:
                continue
            for j, sym in enumerate(batch):
                col = kw_list[j]
                if col in df.columns:
                    series = df[col].dropna()
                    if len(series) >= 8:
                        recent   = float(series.tail(4).mean())
                        baseline = float(series.head(len(series)//2).mean())
                        velocity = (recent - baseline) / baseline if baseline > 0 else 0.0
                        results[sym] = {
                            'recent': recent, 'baseline': baseline,
                            'velocity': float(np.clip(velocity, -1, 1)),
                            'job_expanding':   float(velocity > 0.15),
                            'job_contracting': float(velocity < -0.15),
                        }
            time.sleep(1.5)
        except Exception as e:
            print('Batch error: ' + str(e))
            time.sleep(5.0)
    return results

def compute_job_features(symbol, trends_data=None):
    defaults = {'job_velocity': 0.0, 'job_expanding': 0.0, 'job_contracting': 0.0}
    if trends_data is None:
        cache = os.path.join(CACHE_DIR, 'trends_cache.json')
        if os.path.exists(cache):
            with open(cache) as f:
                trends_data = json.load(f)
        else:
            return defaults
    d = trends_data.get(symbol, {})
    return {
        'job_velocity':    float(d.get('velocity', 0)),
        'job_expanding':   float(d.get('job_expanding', 0)),
        'job_contracting': float(d.get('job_contracting', 0)),
    }

if __name__ == '__main__':
    import config
    syms = [s for s in config.WATCHLIST if s in COMPANY_MAP]
    print('Fetching Google Trends for ' + str(len(syms)) + ' symbols...')
    data = fetch_trends_batch(syms)
    print('Got data for ' + str(len(data)) + ' symbols')
    with open(os.path.join(CACHE_DIR, 'trends_cache.json'), 'w') as f:
        json.dump(data, f, indent=2)
    rows = []
    for sym in syms:
        feats = compute_job_features(sym, data)
        feats['symbol'] = sym
        rows.append(feats)
    with open(os.path.join(CACHE_DIR, 'job_signals.json'), 'w') as f:
        json.dump(rows, f)
    print('Top hiring signals:')
    for sym, d in sorted(data.items(), key=lambda x: x[1].get('velocity',0), reverse=True)[:10]:
        print('  ' + sym + ': vel=' + str(round(d.get('velocity',0)*100,1)) + '% recent=' + str(round(d.get('recent',0),1)))
    print('Top layoff signals:')
    for sym, d in sorted(data.items(), key=lambda x: x[1].get('velocity',0))[:5]:
        print('  ' + sym + ': vel=' + str(round(d.get('velocity',0)*100,1)) + '%')
