"""
thematic_scanner.py — Detects emerging investment themes before they're obvious
============================================================
Principle: when a macro theme emerges (AI, energy transition, reshoring),
ALL stocks in that theme benefit. Identify the theme first, then find
the best-positioned stocks within it.

Method: monitor "theme ETF" performance vs market
When a theme ETF outperforms SPY by >15% over 60 days → theme is hot
Then score all stocks in that theme by fundamental quality + momentum

Not backfitting: themes are identified by ETF outperformance BEFORE
we look at individual stocks. ETF performance is real-time, not hindsight.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json, os, sys, time
from datetime import datetime
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

# Theme ETFs — when these outperform, the theme is real
THEMES = {
    'AI Infrastructure':    {'etf': 'BOTZ', 'stocks': ['NVDA','AMD','AVGO','ANET','VRT','SMCI','CEG','VST']},
    'Data Centers':         {'etf': 'DTCR', 'stocks': ['EQIX','DLR','ORCL','VRT','DELL','HPE']},
    'Energy Transition':    {'etf': 'ICLN', 'stocks': ['ENPH','FSLR','CEG','VST','NEE','PLUG']},
    'Defense/Aerospace':    {'etf': 'ITA',  'stocks': ['LMT','RTX','NOC','GD','KTOS','RKLB','AXON']},
    'Reshoring/Mfg':        {'etf': 'MADE', 'stocks': ['CAT','DE','ETN','PWR','MTZ','STRL']},
    'Memory/Storage':       {'etf': 'SOXQ', 'stocks': ['MU','WDC','SNDK','STX','NAND']},
    'Biotech Breakthrough': {'etf': 'ARKG', 'stocks': ['VRTX','REGN','ALNY','MRNA','NBIX']},
    'Nuclear Energy':       {'etf': 'NLR',  'stocks': ['CEG','CCJ','VST','NRG']},
}

def get_etf_rs(etf: str, days: int = 60) -> float:
    """Get ETF return vs SPY over last N days."""
    try:
        etf_hist = yf.Ticker(etf).history(period=f'{days+10}d')
        spy_hist = yf.Ticker('SPY').history(period=f'{days+10}d')
        if len(etf_hist) < days or len(spy_hist) < days:
            return 0.0
        etf_ret = float(etf_hist['Close'].iloc[-1] / etf_hist['Close'].iloc[-days] - 1)
        spy_ret = float(spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-days] - 1)
        return etf_ret - spy_ret
    except:
        return 0.0

def score_theme_stock(ticker: str) -> dict:
    """Score a stock within a hot theme."""
    try:
        info = yf.Ticker(ticker).info
        return {
            'symbol':          ticker,
            'rev_growth':      info.get('revenueGrowth', 0) or 0,
            'eps_growth':      info.get('earningsGrowth', 0) or 0,
            'gross_margin':    info.get('grossMargins', 0) or 0,
            'recommendation':  info.get('recommendationMean', 3) or 3,
            'target_upside':   (info.get('targetMeanPrice',0) - info.get('currentPrice',1)) /
                               max(info.get('currentPrice',1), 1),
        }
    except:
        return {'symbol': ticker}

def scan_themes():
    print("="*60)
    print(f"THEMATIC MOMENTUM SCANNER — {datetime.now().strftime('%Y-%m-%d')}")
    print("="*60)
    print("Identifies hot investment themes BEFORE individual stocks run")
    print()

    hot_themes = []
    all_results = {}

    for theme, data in THEMES.items():
        etf = data['etf']
        rs_60d = get_etf_rs(etf, 60)
        rs_20d = get_etf_rs(etf, 20)
        is_hot = rs_60d > 0.10  # outperforming SPY by >10% over 60 days
        status = "🔥 HOT" if rs_60d > 0.15 else "✓ warm" if is_hot else "  cold"
        print(f"  {theme:<22}: 60d_rs={rs_60d:>+6.1%}  20d_rs={rs_20d:>+6.1%}  {status}")

        if is_hot:
            hot_themes.append(theme)
            all_results[theme] = {
                'etf': etf,
                'rs_60d': round(rs_60d, 4),
                'rs_20d': round(rs_20d, 4),
                'stocks': data['stocks'],
            }
        time.sleep(0.3)

    print(f"\n── HOT THEMES TODAY ──")
    if hot_themes:
        for theme in hot_themes:
            print(f"\n  {theme} (ETF: {all_results[theme]['etf']}  RS={all_results[theme]['rs_60d']:>+.1%})")
            print(f"  Focus stocks: {all_results[theme]['stocks']}")
    else:
        print("  No themes in breakout mode today")

    # Save
    os.makedirs('/Users/rick/ai_trading_bot_v2/cache_themes', exist_ok=True)
    with open('/Users/rick/ai_trading_bot_v2/cache_themes/hot_themes.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Build combined watchlist of hot theme stocks
    hot_stocks = []
    for theme, data in all_results.items():
        hot_stocks.extend(data['stocks'])
    hot_stocks = list(set(hot_stocks))

    with open('/Users/rick/ai_trading_bot_v2/cache_themes/hot_stocks.json', 'w') as f:
        json.dump({'date': str(datetime.now().date()), 'stocks': hot_stocks}, f)

    print(f"\nHot theme stocks to prioritize: {hot_stocks}")
    print(f"Saved to cache_themes/")
    return all_results

if __name__ == "__main__":
    scan_themes()
