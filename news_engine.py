"""
news_engine.py — Multi-source news + sentiment engine
======================================================
Sources:
1. Finnhub    — real-time company news, earnings calendar, SEC filings
2. yfinance   — analyst recommendations, earnings history, upgrades/downgrades
3. Yahoo RSS  — general market news feed
4. Reddit RSS — WSB/investing sentiment (free, no auth)
5. SEC EDGAR  — insider filings (Form 4), institutional (13F)

Used by daily_scanner and live_trader to:
- Block entries on stocks with bad news
- Flag earnings risk (avoid within 5 days)
- Size down on macro risk days
- Add news_sentiment as live ML feature
"""
import requests, json, os, ssl, re
import numpy as np
from datetime import date, timedelta, datetime
from typing import Dict, List, Optional
import urllib.request

FINNHUB_KEY = 'd77aia1r01qp6afkt9s0d77aia1r01qp6afkt9sg'
CACHE_DIR   = '/Users/rick/ai_trading_bot_v2/cache_news'
os.makedirs(CACHE_DIR, exist_ok=True)

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode    = ssl.CERT_NONE

ANTHROPIC_KEY = os.environ.get('ANTHROPIC_KEY', '')


def claude_sentiment(headlines: list, context: str = '') -> list:
    """Score headlines using Claude Haiku."""
    if not headlines:
        return []
    numbered = chr(10).join(f'{i+1}. {h}' for i, h in enumerate(headlines))
    ctx = f"Context: evaluating impact on {context}." if context else ""
    prompt = (f"{ctx} Score each headline for a momentum stock trader. "
              f"For each output BULLISH|score, BEARISH|score, or NEUTRAL|score "
              f"where score is -1.0 to 1.0. One per line, no other text." + chr(10) + 
              f"Headlines:" + chr(10) + numbered)
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json",
                     "x-api-key": ANTHROPIC_KEY,
                     "anthropic-version": "2023-06-01"},
            json={"model": "claude-haiku-4-5-20251001",
                  "max_tokens": len(headlines) * 15,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=10
        )
        lines = resp.json()['content'][0]['text'].strip().split(chr(10))
        results = []
        for line in lines:
            line = line.strip()
            if '|' in line:
                parts = line.split('|')
                label = parts[0].strip().upper()
                try:
                    score = float(parts[1].strip().replace('\u2212', '-').replace('\xe2\x88\x92', '-'))
                except:
                    score = 0.1 if label == 'BULLISH' else -0.1 if label == 'BEARISH' else 0.0
                results.append({'label': label, 'score': score})
            else:
                results.append({'label': 'NEUTRAL', 'score': 0.0})
        return results
    except Exception as e:
        return [{'label': 'NEUTRAL', 'score': _score_text(h)} for h in headlines]



BEARISH_HARD = [
    'fraud', 'investigation', 'sec probe', 'bankruptcy', 'default',
    'restatement', 'delisted', 'accounting irregularity', 'class action',
    'criminal', 'indicted', 'ponzi', 'going concern'
]
BEARISH_SOFT = [
    'downgrade', 'cut', 'lower', 'reduce', 'sell', 'underperform',
    'miss', 'below', 'warning', 'guidance cut', 'layoff', 'resign',
    'tariff', 'sanction', 'ban', 'recall', 'lawsuit', 'short seller',
    'decline', 'drop', 'fall', 'plunge', 'slump', 'concern', 'risk'
]
BULLISH = [
    'upgrade', 'raise', 'buy', 'overweight', 'outperform', 'beat',
    'above', 'record', 'partnership', 'contract', 'buyback', 'dividend',
    'breakout', 'approval', 'fda approved', 'guidance raise', 'surge',
    'rally', 'strong', 'positive', 'growth', 'expansion'
]
MACRO_RISK_WORDS = [
    'fomc', 'federal reserve', 'interest rate', 'cpi', 'inflation',
    'nfp', 'non-farm', 'jobs report', 'gdp', 'tariff', 'trade war',
    'iran', 'war', 'strait of hormuz', 'oil embargo', 'recession',
    'banking crisis', 'debt ceiling', 'shutdown', 'sanctions'
]


def _score_text(text: str) -> float:
    t = text.lower()
    score = 0.0
    for kw in BEARISH_HARD:
        if kw in t: score -= 0.5
    for kw in BEARISH_SOFT:
        if kw in t: score -= 0.15
    for kw in BULLISH:
        if kw in t: score += 0.15
    return float(np.clip(score, -1.0, 1.0))


# ── SOURCE 1: Finnhub company news ────────────────────────────────────────────

def _finnhub_company_news(symbol: str, days_back: int = 3) -> list:
    today     = date.today().isoformat()
    from_date = (date.today() - timedelta(days=days_back)).isoformat()
    try:
        r = requests.get('https://finnhub.io/api/v1/company-news',
            params={'symbol': symbol, 'from': from_date, 'to': today, 'token': FINNHUB_KEY},
            timeout=5)
        return r.json() if isinstance(r.json(), list) else []
    except: return []


def _finnhub_market_news() -> list:
    try:
        r = requests.get('https://finnhub.io/api/v1/news',
            params={'category': 'general', 'token': FINNHUB_KEY}, timeout=5)
        return r.json()[:30] if isinstance(r.json(), list) else []
    except: return []


def _finnhub_earnings(symbols: list, days_ahead: int = 7) -> dict:
    today    = date.today().isoformat()
    end_date = (date.today() + timedelta(days=days_ahead)).isoformat()
    try:
        r = requests.get('https://finnhub.io/api/v1/calendar/earnings',
            params={'from': today, 'to': end_date, 'token': FINNHUB_KEY}, timeout=5)
        cal = r.json().get('earningsCalendar', [])
        sym_set = set(s.upper() for s in symbols)
        return {e['symbol'].upper(): {
            'date': e.get('date'),
            'eps_estimate': e.get('epsEstimate'),
            'days_away': (datetime.strptime(e['date'], '%Y-%m-%d').date() - date.today()).days
        } for e in cal if e.get('symbol','').upper() in sym_set}
    except: return {}


def _finnhub_recommendation(symbol: str) -> dict:
    """Get analyst recommendation trend."""
    try:
        r = requests.get('https://finnhub.io/api/v1/stock/recommendation',
            params={'symbol': symbol, 'token': FINNHUB_KEY}, timeout=5)
        data = r.json()
        if data and isinstance(data, list):
            latest = data[0]
            total = (latest.get('strongBuy',0) + latest.get('buy',0) +
                     latest.get('hold',0) + latest.get('sell',0) + latest.get('strongSell',0))
            if total > 0:
                bull = (latest.get('strongBuy',0) + latest.get('buy',0)) / total
                bear = (latest.get('sell',0) + latest.get('strongSell',0)) / total
                return {'bull_pct': round(bull,2), 'bear_pct': round(bear,2),
                        'total': total, 'period': latest.get('period','')}
    except: pass
    return {}


def _finnhub_sentiment(symbol: str) -> dict:
    """Get news sentiment score from Finnhub."""
    try:
        r = requests.get('https://finnhub.io/api/v1/news-sentiment',
            params={'symbol': symbol, 'token': FINNHUB_KEY}, timeout=5)
        d = r.json()
        if d and 'companyNewsScore' in d:
            return {
                'company_score': d.get('companyNewsScore', 0),
                'sector_score': d.get('sectorAverageNewsScore', 0),
                'buzz': d.get('buzz', {}).get('buzz', 0),
                'articles_last_week': d.get('buzz', {}).get('weeklyAverage', 0),
            }
    except: pass
    return {}


# ── SOURCE 2: yfinance ────────────────────────────────────────────────────────

def _yfinance_upgrades(symbol: str) -> list:
    """Get recent analyst upgrades/downgrades."""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        upgrades = t.upgrades_downgrades
        if upgrades is not None and len(upgrades) > 0:
            # Get last 7 days
            upgrades.index = pd.to_datetime(upgrades.index) if hasattr(upgrades.index, 'to_pydatetime') else upgrades.index
            recent = upgrades.head(5)
            result = []
            for idx, row in recent.iterrows():
                result.append({
                    'date': str(idx)[:10],
                    'firm': row.get('Firm', ''),
                    'action': row.get('Action', ''),
                    'from_grade': row.get('From Grade', ''),
                    'to_grade': row.get('To Grade', ''),
                })
            return result
    except: pass
    return []


def _yfinance_news(symbol: str) -> list:
    """Get yfinance news items."""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        news = t.news
        if news:
            return [{'headline': n.get('title',''), 'source': n.get('publisher','')}
                    for n in news[:10]]
    except: pass
    return []


def _yfinance_calendar(symbol: str) -> dict:
    """Get next earnings date."""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        cal = t.calendar
        if cal is not None and len(cal) > 0:
            return {'next_earnings': str(cal.get('Earnings Date', [''])[0])[:10]}
    except: pass
    return {}


# ── SOURCE 3: Yahoo Finance RSS ────────────────────────────────────────────────

def _yahoo_rss_market() -> list:
    """Parse Yahoo Finance general news RSS."""
    try:
        url = 'https://finance.yahoo.com/news/rss'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=5, context=SSL_CTX)
        content = resp.read().decode('utf-8', errors='ignore')
        titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', content)
        titles += re.findall(r'<title>(.*?)</title>', content)
        return [t.strip() for t in titles if len(t) > 20][:20]
    except: return []


# ── SOURCE 4: Reddit WSB RSS (free, no auth) ────────────────────────────────

def _reddit_sentiment() -> dict:
    """Parse WSB and investing subreddits for market sentiment."""
    results = {'wsb_mentions': {}, 'sentiment': 0.0, 'hot_tickers': []}
    try:
        url = 'https://www.reddit.com/r/wallstreetbets/hot.json?limit=25'
        req = urllib.request.Request(url, headers={'User-Agent': 'TradingBot/1.0'})
        resp = urllib.request.urlopen(req, timeout=5, context=SSL_CTX)
        data = json.loads(resp.read())
        posts = data.get('data', {}).get('children', [])
        ticker_re = re.compile(r'\b([A-Z]{2,5})\b')
        scores = []
        mentions = {}
        for p in posts:
            post_data = p.get('data', {})
            title = post_data.get('title', '')
            score = _score_text(title)
            scores.append(score)
            for ticker in ticker_re.findall(title):
                if ticker not in ('THE','AND','FOR','ARE','BUT','NOT','YOU','ALL'):
                    mentions[ticker] = mentions.get(ticker, 0) + 1
        results['wsb_mentions']  = dict(sorted(mentions.items(), key=lambda x: x[1], reverse=True)[:10])
        results['sentiment']     = float(np.mean(scores)) if scores else 0.0
        results['hot_tickers']   = list(results['wsb_mentions'].keys())[:5]
    except: pass
    return results


# ── MAIN FUNCTIONS ─────────────────────────────────────────────────────────────

def get_stock_news(symbol: str, days_back: int = 3) -> dict:
    """
    Comprehensive stock news from all sources.
    Returns: score, block, headlines, analyst_action, earnings_soon
    """
    all_headlines = []
    raw_headlines = []
    scores = []

    # Finnhub news
    fh_news = _finnhub_company_news(symbol, days_back)
    for a in fh_news[:10]:
        h = a.get('headline', '')
        all_headlines.append(f"[FH] {h[:80]}")
        raw_headlines.append(h)

    # Finnhub sentiment score
    fh_sent = _finnhub_sentiment(symbol)

    # yfinance news
    yf_news = _yfinance_news(symbol)
    for a in yf_news[:5]:
        h = a.get('headline', '')
        if h:
            all_headlines.append(f"[YF] {h[:80]}")
            raw_headlines.append(h)

    # yfinance upgrades/downgrades
    upgrades = _yfinance_upgrades(symbol)
    analyst_action = None
    for u in upgrades[:2]:
        action = u.get('action', '').lower()
        grade  = u.get('to_grade', '').lower()
        headline = f"{u.get('firm','')} {action} to {u.get('to_grade','')} from {u.get('from_grade','')}"
        all_headlines.append(f"[ANALYST] {headline[:80]}")
        if action in ('downgraded', 'reiterated') and grade in ('sell','underperform','underweight'):
            scores.append(-0.5)
            analyst_action = 'downgrade'
        elif action == 'upgraded' and grade in ('buy','outperform','overweight','strong buy'):
            scores.append(0.5)
            analyst_action = 'upgrade'

    # Earnings risk
    yf_cal  = _yfinance_calendar(symbol)
    earnings_soon = False
    if yf_cal.get('next_earnings'):
        try:
            next_earn = datetime.strptime(yf_cal['next_earnings'][:10], '%Y-%m-%d').date()
            days_to   = (next_earn - date.today()).days
            if 0 <= days_to <= 7:
                earnings_soon = True
                all_headlines.append(f"[EARNINGS] {symbol} reports in {days_to} days")
        except: pass

    # Score all headlines with Claude — batch call, ~$0.001 per stock
    if raw_headlines:
        claude_scores = claude_sentiment(raw_headlines, context=symbol)
        scores = [s['score'] for s in claude_scores]
    
    avg_score   = float(np.mean(scores)) if scores else 0.0
    hard_block  = any(kw in ' '.join(all_headlines).lower() for kw in BEARISH_HARD)
    bearish_flags = [h for h in all_headlines if _score_text(h) < -0.3]

    # Blend with Finnhub sentiment if available
    if fh_sent.get('company_score'):
        avg_score = avg_score * 0.6 + (fh_sent['company_score'] - 0.5) * 0.4

    return {
        'symbol':         symbol,
        'score':          round(avg_score, 3),
        'block':          hard_block or avg_score < -0.45,
        'hard_block':     hard_block,
        'earnings_soon':  earnings_soon,
        'analyst_action': analyst_action,
        'headlines':      all_headlines[:8],
        'bearish_flags':  bearish_flags[:3],
        'n_articles':     len(fh_news) + len(yf_news),
        'finnhub_buzz':   fh_sent.get('buzz', 0),
        'analyst_bull':   _finnhub_recommendation(symbol).get('bull_pct', 0.5),
    }


def get_market_sentiment() -> dict:
    """
    Full market sentiment from all sources.
    Returns: score, macro_risk, reddit_sentiment, top_headlines
    """
    all_headlines = []
    scores        = []
    macro_risk    = False

    # Finnhub market news
    fh_news = _finnhub_market_news()
    for a in fh_news[:20]:
        h = a.get('headline', '')
        all_headlines.append(h[:80])
        scores.append(_score_text(h))
        if any(kw in h.lower() for kw in MACRO_RISK_WORDS):
            macro_risk = True

    # Yahoo RSS
    yahoo_titles = _yahoo_rss_market()
    for t in yahoo_titles[:10]:
        all_headlines.append(f"[YFR] {t[:80]}")
        scores.append(_score_text(t))
        if any(kw in t.lower() for kw in MACRO_RISK_WORDS):
            macro_risk = True

    # Reddit
    reddit = _reddit_sentiment()

    avg_score = float(np.mean(scores)) if scores else 0.0

    return {
        'score':           round(avg_score, 3),
        'macro_risk':      macro_risk,
        'headlines':       all_headlines[:10],
        'reddit_score':    reddit.get('sentiment', 0.0),
        'hot_tickers':     reddit.get('hot_tickers', []),
        'wsb_mentions':    reddit.get('wsb_mentions', {}),
        'n_sources':       3,
    }


def get_earnings_risk(symbols: list, days_ahead: int = 7) -> dict:
    """Get earnings risk from Finnhub + yfinance combined."""
    fh_earnings = _finnhub_earnings(symbols, days_ahead)

    # Also check yfinance for any missed
    for sym in symbols[:20]:  # rate limit — check top candidates only
        if sym not in fh_earnings:
            cal = _yfinance_calendar(sym)
            if cal.get('next_earnings'):
                try:
                    next_earn = datetime.strptime(cal['next_earnings'][:10], '%Y-%m-%d').date()
                    days_to   = (next_earn - date.today()).days
                    if 0 <= days_to <= days_ahead:
                        fh_earnings[sym] = {'date': cal['next_earnings'][:10],
                                            'days_away': days_to, 'source': 'yfinance'}
                except: pass

    return fh_earnings


def run_news_scan(watchlist: list = None) -> dict:
    """Full daily news scan — run by daily_scanner."""
    import sys; sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
    if watchlist is None:
        import config
        watchlist = list(config.WATCHLIST)

    print("\n[NEWS] Multi-source market scan...")

    # Market sentiment
    market = get_market_sentiment()
    print(f"  Market score:    {market['score']:+.2f}  macro_risk={market['macro_risk']}")
    print(f"  Reddit WSB:      {market['reddit_score']:+.2f}  hot={market['hot_tickers']}")
    for h in market['headlines'][:3]:
        print(f"    {h[:70]}")

    # Earnings risk
    print(f"\n[NEWS] Earnings risk next 7 days...")
    earnings = get_earnings_risk(watchlist)
    if earnings:
        print(f"  ⚠️  Earnings: {list(earnings.keys())}")

    result = {
        'date':             date.today().isoformat(),
        'timestamp':        datetime.now().isoformat(),
        'market_sentiment': market,
        'earnings_risk':    earnings,
    }
    json.dump(result, open(f'{CACHE_DIR}/news_scan.json', 'w'), indent=2, default=str)
    print(f"\n  ✓ News scan saved")
    return result


if __name__ == '__main__':
    import sys, pandas as pd
    sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
    import config

    print("="*60)
    print("NEWS ENGINE — FULL TEST")
    print("="*60)

    # Market
    print("\n[1] MARKET SENTIMENT")
    mkt = get_market_sentiment()
    print(f"  Score: {mkt['score']:+.2f}  Macro risk: {mkt['macro_risk']}")
    print(f"  Reddit: {mkt['reddit_score']:+.2f}  Hot: {mkt['hot_tickers']}")
    print("  Top headlines:")
    for h in mkt['headlines'][:5]:
        print(f"    {h[:70]}")

    # Current positions
    print("\n[2] POSITION NEWS")
    for sym in ['ASAN', 'MAS']:
        news = get_stock_news(sym)
        print(f"\n  {sym}: score={news['score']:+.2f}  block={news['block']}  "
              f"earnings_soon={news['earnings_soon']}  analyst={news['analyst_action']}")
        for h in news['headlines'][:3]:
            print(f"    {h[:70]}")

    # Earnings
    print("\n[3] EARNINGS RISK (next 7 days)")
    sample = list(config.WATCHLIST)[:50]
    earnings = get_earnings_risk(sample)
    for sym, info in list(earnings.items())[:10]:
        print(f"  {sym}: {info.get('date','')} ({info.get('days_away','')}d away)")

    # Full scan
    print("\n[4] FULL SCAN")
    run_news_scan(list(config.WATCHLIST)[:30])
