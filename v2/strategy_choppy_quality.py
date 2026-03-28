"""
strategy_choppy_quality.py — Low-volatility quality strategy for CHOPPY regime
============================================================
Problem: In CHOPPY markets we barely trade (4 trades/yr) while SPY
         is fully invested in defensive stocks that lead in choppy markets.

Solution: In CHOPPY regime, rotate to low-volatility quality stocks
          instead of reducing momentum trades.

Research:
  - MSCI: low-vol stocks have IR=0.92, best risk-adjusted over 50 years
  - CFA: low-vol outperforms in downturns and choppy markets
  - Frazzini & Pedersen 2014: betting against beta — low-beta stocks
    earn higher risk-adjusted returns than high-beta stocks

Stocks: dividend growers, utilities, consumer staples, REITs, healthcare
Signal: quality score + dividend consistency + low beta vs SPY
Hold: 20-40 days, trailing stop 5% (tighter — less volatile stocks)
Universe: separate from momentum universe, no overlap
"""

# CHOPPY regime quality universe — these lead when momentum fails
CHOPPY_QUALITY_UNIVERSE = {
    # Utilities — reliable cash flows, regulated monopolies
    'utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'PEG', 'EXC',
                  'WEC', 'ES', 'CMS', 'NI'],

    # Consumer Staples — recession-proof demand
    'staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'CL',
                'KMB', 'GIS', 'KR', 'SFM'],

    # Healthcare — non-cyclical demand
    'healthcare': ['JNJ', 'ABT', 'MDT', 'TMO', 'DHR', 'ZTS', 'BSX',
                   'SYK', 'ISRG', 'CI', 'UNH'],

    # REITs — income, low correlation to growth
    'reits': ['O', 'PLD', 'PSA', 'AMT', 'EQIX', 'DLR', 'WELL',
              'VTR', 'EQR', 'AVB'],

    # Quality industrials — wide moat, consistent earnings
    'industrials': ['WM', 'RSG', 'FAST', 'GWW', 'ITW', 'EMR'],

    # Insurance — stable underwriting income
    'insurance': ['PGR', 'TRV', 'CB', 'AFL', 'AIG'],
}

# Flatten to list
ALL_CHOPPY_STOCKS = []
for sector, stocks in CHOPPY_QUALITY_UNIVERSE.items():
    ALL_CHOPPY_STOCKS.extend(stocks)

# Entry criteria for CHOPPY quality trades
CHOPPY_ENTRY_RULES = {
    'min_dividend_yield': 0.01,      # at least 1% dividend
    'max_beta': 0.80,                # low beta vs SPY
    'min_roe': 0.12,                 # 12%+ return on equity
    'min_gross_margin': 0.20,        # decent margins
    'max_pe': 30,                    # not wildly overvalued
    'min_analyst_upside': 0.05,      # 5%+ analyst upside
    'stop_pct': 0.05,                # tighter stop — 5% vs 8%
    'max_hold_days': 30,             # shorter holds
    'max_positions': 4,              # 4 positions in CHOPPY quality
}

def get_choppy_candidates(regime: str) -> list:
    """Return quality stocks appropriate for CHOPPY regime."""
    if regime != 'CHOPPY':
        return []
    return ALL_CHOPPY_STOCKS

def score_choppy_stock(ticker: str) -> dict:
    """Score a stock for CHOPPY regime quality entry."""
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
        price    = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
        beta     = info.get('beta', 1.0) or 1.0
        div_yld  = info.get('dividendYield', 0) or 0
        roe      = info.get('returnOnEquity', 0) or 0
        gm       = info.get('grossMargins', 0) or 0
        pe       = info.get('trailingPE', 99) or 99
        target   = info.get('targetMeanPrice', price) or price
        upside   = (target - price) / price if price > 0 else 0
        rec      = info.get('recommendationMean', 3) or 3

        # Quality score
        score = 0
        if div_yld >= CHOPPY_ENTRY_RULES['min_dividend_yield']: score += 1
        if beta    <= CHOPPY_ENTRY_RULES['max_beta']:            score += 2
        if roe     >= CHOPPY_ENTRY_RULES['min_roe']:             score += 1
        if gm      >= CHOPPY_ENTRY_RULES['min_gross_margin']:    score += 1
        if pe      <= CHOPPY_ENTRY_RULES['max_pe']:              score += 1
        if upside  >= CHOPPY_ENTRY_RULES['min_analyst_upside']:  score += 1
        if rec     <= 2.0:                                        score += 1

        return {
            'symbol':    ticker,
            'score':     score,
            'beta':      round(beta, 2),
            'div_yld':   round(div_yld, 3),
            'roe':       round(roe, 3),
            'upside':    round(upside, 3),
            'pe':        round(pe, 1),
            'price':     round(price, 2),
            'tradeable': score >= 5 and beta <= 0.80,
        }
    except Exception as e:
        return {'symbol': ticker, 'score': 0, 'tradeable': False, 'error': str(e)}

if __name__ == "__main__":
    import time
    print("CHOPPY QUALITY STRATEGY — Candidate Scan")
    print("="*55)
    results = []
    for sym in ALL_CHOPPY_STOCKS[:20]:
        r = score_choppy_stock(sym)
        results.append(r)
        if r.get('tradeable'):
            print(f"  ✓ {sym:<6}: score={r['score']}  beta={r['beta']}  "
                  f"div={r['div_yld']:.1%}  upside={r['upside']:>+.1%}")
        time.sleep(0.3)

    tradeable = [r for r in results if r.get('tradeable')]
    tradeable.sort(key=lambda x: x['score'], reverse=True)
    print(f"\nTradeable: {len(tradeable)}/{len(results)}")
    for r in tradeable[:10]:
        print(f"  {r['symbol']}: score={r['score']}  beta={r['beta']}  "
              f"upside={r['upside']:>+.0%}  div={r['div_yld']:.1%}")
