import requests
import pandas as pd
import os
from io import StringIO
import time

CACHE_DIR = os.path.expanduser("~/ai_trading_bot_v2/cache_options")
THETA_BASE = "http://127.0.0.1:25503/v3"

def fetch_chain_at_time(date, interval="30m"):
    """Fetch full SPXW 0DTE chain at 30-min intervals for a given date."""
    cache_path = os.path.join(CACHE_DIR, f"chain_intraday_{date}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    
    r = requests.get(f"{THETA_BASE}/option/history/greeks/first_order",
        params={"symbol":"SPXW","expiration":date,"start_date":date,
                "end_date":date,"strike":"*","interval":interval},
        timeout=180)
    
    if r.status_code != 200:
        print(f"  {date}: {r.status_code} {r.text[:100]}")
        return pd.DataFrame()
    
    df = pd.read_csv(StringIO(r.text))
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.to_parquet(cache_path)
    return df

def get_entry_snapshot(date, entry_hour=10, entry_minute=30):
    """Get chain snapshot at entry time."""
    df = fetch_chain_at_time(date)
    if df.empty:
        return pd.DataFrame()
    target = df[
        (df['timestamp'].dt.hour == entry_hour) & 
        (df['timestamp'].dt.minute >= entry_minute)
    ]
    if target.empty:
        # fallback to 10:00
        target = df[df['timestamp'].dt.hour == 10]
    if target.empty:
        return pd.DataFrame()
    # Take earliest matching timestamp
    t = target['timestamp'].min()
    return df[df['timestamp'] == t]

def get_exit_snapshot(date, exit_hour=15, exit_minute=0):
    """Get chain snapshot at exit time."""
    df = fetch_chain_at_time(date)
    if df.empty:
        return pd.DataFrame()
    target = df[
        (df['timestamp'].dt.hour == exit_hour) &
        (df['timestamp'].dt.minute >= exit_minute)
    ]
    if target.empty:
        target = df[df['timestamp'].dt.hour == 14]
    if target.empty:
        return pd.DataFrame()
    t = target['timestamp'].min()
    return df[df['timestamp'] == t]

if __name__ == "__main__":
    # Test on a few dates
    import yfinance as yf
    spy = yf.download("SPY", start="2024-06-01", end="2024-06-30", progress=False, auto_adjust=True)
    dates = [d.strftime("%Y%m%d") for d in spy.index[:5]]
    
    for date in dates:
        entry = get_entry_snapshot(date)
        if entry.empty:
            print(f"{date}: no entry snapshot")
            continue
        spx = entry['underlying_price'].iloc[0]
        n_strikes = entry['strike'].nunique()
        print(f"{date}: SPX={spx:.1f} at entry, {n_strikes} strikes available")
        time.sleep(1)
