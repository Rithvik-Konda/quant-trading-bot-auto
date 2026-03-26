"""
pead_midcap_backtest.py — Standalone PEAD signal backtest on mid-caps only
Large-cap PEAD: 0.2% OOS (analyst coverage too dense)
Mid-cap hypothesis: information diffuses slower → 45-60 day drift persists
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
import config
from backtester_clean import fetch_history

MIDCAP_WATCHLIST = [
    'AXON','WING','CAVA','APP','DUOL','DECK','TXRH','SHAK',
    'PCTY','PAYC','ANF','BOOT','CROX','DKNG','RDDT','RKLB',
    'PLNT','WRBY','PTON','CELH','TTD','MNDY','TOST','BRZE'
]

def compute_pead_signal(df, lookback=20):
    close = df['close']
    volume = df['volume']
    ret1 = close.pct_change()
    sector_ret = ret1.rolling(5).mean()
    idio_ret = ret1 - sector_ret
    vol_threshold = volume.rolling(20).mean() * 2.0
    large_idio = idio_ret.abs() > 0.03
    high_vol = volume > vol_threshold
    earnings_day = (large_idio & high_vol).astype(float)
    earnings_surprise = idio_ret * earnings_day
    pead_45 = earnings_surprise.rolling(45).sum()
    return pead_45

results = []
for sym in MIDCAP_WATCHLIST:
    df = fetch_history(sym, days=1500)
    if df is None or len(df) < 300:
        continue
    signal = compute_pead_signal(df)
    fwd_ret = df['close'].pct_change(21).shift(-21)
    combined = pd.DataFrame({'signal': signal, 'fwd': fwd_ret}).dropna()
    if len(combined) < 100:
        continue
    long_sig = combined[combined['signal'] > 0.02]
    short_sig = combined[combined['signal'] < -0.02]
    neutral = combined[combined['signal'].abs() <= 0.02]
    results.append({
        'symbol': sym,
        'long_mean': long_sig['fwd'].mean(),
        'short_mean': short_sig['fwd'].mean(),
        'neutral_mean': neutral['fwd'].mean(),
        'long_n': len(long_sig),
        'ic': combined['signal'].corr(combined['fwd'])
    })

df_r = pd.DataFrame(results).sort_values('ic', ascending=False)
print("=== MID-CAP PEAD RESULTS ===")
print(df_r.round(4).to_string())
print(f"\nMean IC: {df_r['ic'].mean():.4f}")
print(f"Stocks with IC>0: {(df_r['ic']>0).sum()}/{len(df_r)}")
