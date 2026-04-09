import requests
import pandas as pd
import numpy as np
import os
import time
import yfinance as yf
from io import StringIO

CACHE_DIR = os.path.expanduser("~/ai_trading_bot_v2/cache_options")
os.makedirs(CACHE_DIR, exist_ok=True)
THETA_BASE = "http://127.0.0.1:25503/v3"

def dl(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def fetch_spxw_chain_eod(date):
    cache_path = os.path.join(CACHE_DIR, f"chain_eod_{date}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    r = requests.get(f"{THETA_BASE}/option/history/eod",
        params={"symbol":"SPXW","expiration":date,"start_date":date,"end_date":date,"strike":"*"}, timeout=60)
    if r.status_code != 200:
        return pd.DataFrame()
    df = pd.read_csv(StringIO(r.text))
    if not df.empty:
        df.to_parquet(cache_path)
    return df

def build_daily_features(start="2022-01-03", end="2025-12-31"):
    print(f"Building daily features {start} to {end}...")
    spy = dl("SPY", start, end)
    vix = dl("^VIX", start, end)['Close']
    vix1d = dl("^VIX1D", start, end)['Close']
    skew = dl("^SKEW", start, end)['Close']

    spy_returns = spy['Close'].pct_change()
    dates = [d.strftime("%Y%m%d") for d in spy.index]
    rows = []

    for i, date in enumerate(dates):
        dt = spy.index[i]
        try:
            chain = fetch_spxw_chain_eod(date)
            if chain.empty:
                continue

            spy_open  = spy.loc[dt, 'Open']
            spy_high  = spy.loc[dt, 'High']
            spy_low   = spy.loc[dt, 'Low']
            day_range_pct = (spy_high - spy_low) / spy_open * 100

            if i > 0:
                prev_close = spy['Close'].iloc[i-1]
                overnight_gap_pct = (spy_open - prev_close) / prev_close * 100
            else:
                overnight_gap_pct = 0.0

            vix_val  = float(vix.get(dt, np.nan))
            vix1d_val = float(vix1d.get(dt, np.nan)) if dt in vix1d.index else np.nan
            skew_val  = float(skew.get(dt, np.nan))  if dt in skew.index  else np.nan

            prior_rets = spy_returns.iloc[max(0,i-5):i]
            realized_vol_5d = float(prior_rets.std() * np.sqrt(252) * 100) if len(prior_rets) >= 3 else np.nan

            rows.append({
                'date': dt,
                'day_range_pct': float(day_range_pct),
                'overnight_gap_pct': float(overnight_gap_pct),
                'abs_gap': abs(float(overnight_gap_pct)),
                'vix': vix_val,
                'vix1d': vix1d_val,
                'skew': skew_val,
                'realized_vol_5d': realized_vol_5d,
                'vix_term': vix1d_val - vix_val if not np.isnan(vix1d_val) else np.nan,
                'dow': dt.dayofweek,
            })

            if i % 50 == 0:
                print(f"  {date}: range={day_range_pct:.2f}% vix={vix_val:.1f} vix1d={vix1d_val:.1f}")

        except Exception as e:
            print(f"  {date}: ERROR {e}")
        time.sleep(0.02)

    df = pd.DataFrame(rows)
    out = os.path.join(CACHE_DIR, "daily_features.parquet")
    df.to_parquet(out)
    print(f"Saved {len(df)} rows to {out}")
    return df

if __name__ == "__main__":
    df = build_daily_features()
    print(df.describe())

def get_vix1d():
    import requests
    from io import StringIO
    r = requests.get('https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX1D_History.csv')
    df = pd.read_csv(StringIO(r.text))
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')['CLOSE']
    df.index = df.index.normalize()
    return df

def get_skew():
    import requests
    from io import StringIO
    r = requests.get('https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv')
    df = pd.read_csv(StringIO(r.text))
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')['CLOSE']
    df.index = df.index.normalize()
    return df
