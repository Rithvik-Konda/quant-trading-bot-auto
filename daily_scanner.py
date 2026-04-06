"""
daily_scanner.py — Morning pre-market scan
==========================================
Run at 9:00am ET before market open.
Produces today's watchlist with all signals for live trading.

Output:
  - Ranked candidates with all signals
  - News sentiment filter (Polygon)
  - Insider sequence alerts
  - Congressional cluster buys
  - PCR fear/complacency
  - Squeeze alerts
  - FDA catalyst flags
  - Current regime
  - Breadth reading
"""
import os, sys, json, time
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

import config
from backtester_clean import fetch_history
from regime_classifier import RegimeClassifier, compute_signals, load_macro_data

def run_daily_scan():
    print("="*60)
    print(f"DAILY SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    # ── 1. Regime ─────────────────────────────────────────────
    print("\n[1] Computing regime...")
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache_prices')
    spy_macro, hyg_macro, vix_macro = load_macro_data(cache_dir=cache_dir)
    import joblib as _jl, os as _os
    _streak_store  = _jl.load("streak_store.joblib")  if _os.path.exists("streak_store.joblib")  else {}
    _insider_store = _jl.load("insider_store.joblib") if _os.path.exists("insider_store.joblib") else {}
    clf     = RegimeClassifier()
    today   = pd.Timestamp.now().normalize()
    signals = compute_signals(spy_macro, hyg_macro, vix_macro, as_of_date=today)
    regime  = clf.update(today, signals) if signals else "UNKNOWN"
    print(f"  Current regime: {regime}")

    # ── 2. Market breadth ────────────────────────────────────
    print("\n[2] Computing market breadth...")
    above = total = 0
    for sym in config.WATCHLIST[:150]:
        try:
            df = fetch_history(sym, days=60)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if len(df) >= 50:
                ma50 = df['close'].tail(50).mean()
                if df['close'].iloc[-1] > ma50:
                    above += 1
                total += 1
        except:
            pass
    breadth = above / total if total > 0 else 0.5
    breadth_ok = breadth >= 0.50
    print(f"  Breadth: {breadth:.0%} ({'✓ OK' if breadth_ok else '✗ BLOCKED — no new longs'})")

    # ── 3. HYG credit check ──────────────────────────────────
    print("\n[3] Credit stress check (HYG)...")
    hyg_ok = True
    try:
        hyg_close = hyg_macro['close'] if 'close' in hyg_macro.columns else hyg_macro.iloc[:,0]
        hyg_10d   = float(hyg_close.iloc[-1] / hyg_close.iloc[-10] - 1)
        hyg_ok    = hyg_10d > -0.02
        print(f"  HYG 10d: {hyg_10d:+.1%} ({'✓ OK' if hyg_ok else '✗ CREDIT STRESS — no new longs'})")
    except Exception as e:
        print(f"  HYG check failed: {e}")

    # ── 4. News sentiment ────────────────────────────────────
    print("\n[4] Loading news sentiment...")
    bearish_news = []
    try:
        from polygon_news import get_bearish_symbols, scan_watchlist
        news_cache = '/Users/rick/ai_trading_bot_v2/cache_news/sentiment.json'
        if os.path.exists(news_cache):
            mtime = os.path.getmtime(news_cache)
            if time.time() - mtime > 6*3600:
                print("  Refreshing news cache...")
                scan_watchlist(config.WATCHLIST)
        bearish_news = get_bearish_symbols()
        print(f"  Bearish news: {bearish_news if bearish_news else 'None'}")
    except Exception as e:
        print(f"  News scan failed: {e}")

    # ── 5. Insider sequences ─────────────────────────────────
    print("\n[5] Insider sequences...")
    insider_buys = []
    try:
        cache = json.load(open('/Users/rick/ai_trading_bot_v2/cache_insider/insider_sequence.json'))
        insider_buys = [r['symbol'] for r in cache
                       if r.get('insider_seq_buy', 0) > 0 and r.get('insider_seq_strength', 0) >= 0.6]
        print(f"  Active buy sequences: {insider_buys[:10]}")
    except Exception as e:
        print(f"  Insider data failed: {e}")

    # ── 6. Congressional clusters ────────────────────────────
    print("\n[6] Congressional cluster buys...")
    congress_buys = []
    try:
        cache = json.load(open('/Users/rick/ai_trading_bot_v2/cache_congress/congress_trades.json'))
        congress_buys = [r['symbol'] for r in cache if r.get('congress_cluster', 0) > 0]
        print(f"  Cluster buys: {congress_buys[:10]}")
    except Exception as e:
        print(f"  Congress data failed: {e}")

    # ── 7. Squeeze alerts ────────────────────────────────────
    print("\n[7] Squeeze alerts...")
    squeeze_alerts = []
    try:
        cache = json.load(open('/Users/rick/ai_trading_bot_v2/cache_si/squeeze_scores.json'))
        squeeze_alerts = [r['symbol'] for r in cache if r.get('squeeze_alert', 0) > 0]
        print(f"  Squeeze alerts: {squeeze_alerts if squeeze_alerts else 'None'}")
    except Exception as e:
        print(f"  Squeeze data failed: {e}")

    # ── 8. FDA catalysts ────────────────────────────────────
    print("\n[8] FDA catalyst flags...")
    fda_flags = []
    try:
        cache = json.load(open('/Users/rick/ai_trading_bot_v2/cache_fda/fda_signals.json'))
        fda_flags = [r['symbol'] for r in cache if r.get('fda_catalyst_near', 0) > 0]
        print(f"  FDA near-term: {fda_flags[:10]}")
    except Exception as e:
        print(f"  FDA data failed: {e}")

    # ── 9. SUMMARY ───────────────────────────────────────────
    print("\n" + "="*60)
    print("TRADING DAY SUMMARY")
    print("="*60)
    print(f"  Regime:    {regime}")
    print(f"  Breadth:   {breadth:.0%}  {'✓' if breadth_ok else '✗ NO NEW LONGS'}")
    print(f"  Credit:    {'✓ OK' if hyg_ok else '✗ STRESSED — NO NEW LONGS'}")

    can_trade = breadth_ok and hyg_ok and regime in ('TRENDING_BULL', 'CHOPPY')

    if not can_trade:
        print(f"\n  ⚠️  CONDITIONS UNFAVORABLE — HOLD EXISTING, NO NEW ENTRIES")
    else:
        print(f"\n  ✓ CONDITIONS FAVORABLE — SCAN FOR ENTRIES")

    print(f"\n  Avoid today (bearish news):  {bearish_news}")
    print(f"  Favor today (insider buys):  {insider_buys[:5]}")
    print(f"  Favor today (congress):      {congress_buys[:5]}")
    print(f"  Watch (squeeze):             {squeeze_alerts}")
    print(f"  Avoid (FDA binary risk):     {fda_flags[:5]}")

    # ── New listing momentum scan ────────────────────────────
    print("\n[9] Scanning new listings for RS momentum...")
    try:
        from new_listing_scanner import scan_new_listings
        new_listing_results = scan_new_listings()
        tradeable_new = [r['symbol'] for r in new_listing_results if r.get('tradeable')]
        print(f"  Tradeable new listings: {tradeable_new}")
    except Exception as e:
        print(f"  New listing scan failed: {e}")
        tradeable_new = []

    # ── Thematic momentum scan ────────────────────────────────
    print("\n[10] Scanning investment themes...")
    try:
        from thematic_scanner import scan_themes
        hot_themes = scan_themes()
        n_hot = len(hot_themes)
        print(f"  Hot themes: {n_hot}")
    except Exception as e:
        print(f"  Theme scan failed: {e}")

    # ── Pre-score all 447 symbols for trader ─────────────────
    print("\n[9] Pre-scoring ML ranks for all symbols...")
    try:
        from strategy_core import load_ranker_ensemble
        from backtester_clean import fetch_history
        from ml_model import compute_features
        import numpy as np

        rankers = load_ranker_ensemble()
        ml_scores = []
        for sym in config.WATCHLIST:
            try:
                df = fetch_history(sym, days=400)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                feats = compute_features(df, symbol=sym, vix_macro=vix_macro, streak_store=_streak_store, insider_store=_insider_store, regime=regime)
                if feats is None or len(feats) == 0:
                    continue
                latest = feats.iloc[-1]
                scores = []
                for horizon, r in rankers.items():
                    X = pd.DataFrame([latest], columns=feats.columns)
                    X = X.reindex(columns=r['features'], fill_value=0)
                    scores.append(float(r['model'].predict(X)[0]))
                if scores:
                    ml_scores.append({'symbol': sym, 'ml_score': float(np.mean(scores))})
            except:
                pass

        # Convert to cross-sectional percentile rank
        if ml_scores:
            scores_arr = np.array([c['ml_score'] for c in ml_scores])
            ranks = scores_arr.argsort().argsort() / max(len(scores_arr)-1, 1)
            for c, r in zip(ml_scores, ranks):
                c['ml_rank_pct'] = float(r)
            ml_scores.sort(key=lambda x: x['ml_rank_pct'], reverse=True)

        # Save as prev_ml_ranks for confirmation + today's ranks for entry
        prev_ranks = {c['symbol']: c['ml_rank_pct'] for c in ml_scores}
        os.makedirs('/Users/rick/ai_trading_bot_v2/cache_alpaca', exist_ok=True)
        with open('/Users/rick/ai_trading_bot_v2/cache_alpaca/prev_ml_ranks.json', 'w') as f:
            json.dump(prev_ranks, f)
        with open('/Users/rick/ai_trading_bot_v2/cache_alpaca/today_ml_ranks.json', 'w') as f:
            json.dump(ml_scores, f, indent=2)

        top5 = ml_scores[:5]
        print(f"  Scored {len(ml_scores)} symbols")
        print(f"  Top 5: {[c['symbol'] for c in top5]}")
    except Exception as e:
        print(f"  ML pre-scoring failed: {e}")

    # ── News + macro theme integration ──────────────────────────────────
    try:
        from news_engine import get_market_sentiment, get_stock_news
        _mkt_news = get_market_sentiment()
        _macro_risk = _mkt_news.get('macro_risk', False)
        _mkt_score  = _mkt_news.get('score', 0.0)
        print(f"  News sentiment: {_mkt_score:+.2f}  macro_risk={_macro_risk}")
        if _macro_risk:
            print(f"  ⚠️  Macro risk — reducing long slots, boosting shorts")
    except Exception as _ne:
        print(f"  News scan failed: {_ne}")
        _macro_risk = False
        _mkt_score  = 0.0

    # Macro theme — which sectors to favor/avoid based on news
    # Iran war: short consumers/tech, long energy/defense
    # Fed hawkish: short reits/utilities, long financials
    _sector_map = {}
    for _etf, _syms in getattr(config, 'SECTOR_ETFS', {}).items():
        for _s in _syms: _sector_map[_s] = _etf

    _macro_short_sectors = set()
    _macro_long_sectors  = set()
    if _macro_risk:
        import re as _re
        _headlines = ' '.join(_mkt_news.get('headlines', [])).lower()
        if any(w in _headlines for w in ['iran','hormuz','oil embargo','war']):
            _macro_short_sectors = {'XLK','XLY'}
            _macro_long_sectors  = {'XLE','XLB','ITA'}
            print("  Theme: Iran war → long XLE/XLB, short XLK/XLY")
        elif any(w in _headlines for w in ['fed','fomc','rate hike','hawkish']):
            _macro_short_sectors = {'XLRE','XLU'}
            _macro_long_sectors  = {'XLF'}
            print("  Theme: Fed hawkish → long XLF, short XLRE/XLU")
        elif any(w in _headlines for w in ['tariff','trade war','sanction']):
            _macro_short_sectors = {'XLY','XLK'}
            _macro_long_sectors  = {'XLI','XLB'}
            print("  Theme: Tariffs → long domestic, short importers")

    # Generate long and short candidates from ML scores
    long_candidates  = []
    short_candidates = []
    if ml_scores:
        import yfinance as _yf
        for c in ml_scores:
            sym = c['symbol']
            try:
                hist = _yf.Ticker(sym).history(period='3mo')
                if len(hist) < 50: continue
                price  = float(hist['Close'].iloc[-1])
                ma50   = float(hist['Close'].rolling(50).mean().iloc[-1])
                ma200  = float(hist['Close'].rolling(200).mean().iloc[-1])
                ret_5  = float(hist['Close'].pct_change(5).iloc[-1])
                c['price']  = price
                c['ma50']   = ma50
                c['ma200']  = ma200
                c['ret_5d'] = ret_5
                _sector = _sector_map.get(sym, '')

                # ── LONG candidates ──────────────────────────────────────
                # Base: top ML rank + above 200d MA
                _long_ok = c['ml_rank_pct'] >= 0.85 and price > ma200 and can_trade
                # Macro boost: ML strong + macro-favored sector
                _macro_long_boost = (c['ml_rank_pct'] >= 0.80 and
                                     _sector in _macro_long_sectors and
                                     price > ma50)
                if _long_ok or _macro_long_boost:
                    c['macro_confirmed'] = _sector in _macro_long_sectors
                    c['macro_theme']     = 'long' if _sector in _macro_long_sectors else ''
                    long_candidates.append(c)

                # ── SHORT candidates ─────────────────────────────────────
                # Base: bottom ML rank + below MA50 + downtrend
                _short_ok = c['ml_rank_pct'] <= 0.08 and price < ma50 and ret_5 < 0
                # Macro boost: ML weak + macro-hurt sector (even if not below MA50 yet)
                _macro_short_boost = (c['ml_rank_pct'] <= 0.15 and
                                      _sector in _macro_short_sectors and
                                      ret_5 < 0)
                if _short_ok or _macro_short_boost:
                    c['macro_confirmed'] = _sector in _macro_short_sectors
                    c['macro_theme']     = 'short' if _sector in _macro_short_sectors else ''
                    short_candidates.append(c)

            except Exception:
                continue

    # Sort — macro-confirmed candidates first
    long_candidates.sort(key=lambda x: (x.get('macro_confirmed',False), x['ml_rank_pct']), reverse=True)
    short_candidates.sort(key=lambda x: (x.get('macro_confirmed',False), -x['ml_rank_pct']), reverse=True)

    print(f"  Long candidates: {len(long_candidates)}")
    print(f"  Short candidates: {len(short_candidates)}")
    if short_candidates:
        print(f"  Top shorts: {[c['symbol'] for c in short_candidates[:5]]}")

    # Save scan results
    result = {
        'date': str(today.date()),
        'regime': regime,
        'breadth': round(breadth, 3),
        'breadth_ok': breadth_ok,
        'hyg_ok': hyg_ok,
        'can_trade': can_trade,
        'bearish_news': bearish_news,
        'insider_buys': insider_buys,
        'congress_buys': congress_buys,
        'squeeze_alerts': squeeze_alerts,
        'fda_flags': fda_flags,
        'candidates': long_candidates,
        'short_candidates': short_candidates,
        'macro_risk': _macro_risk,
        'news_score': _mkt_score,
        'macro_short_sectors': list(_macro_short_sectors),
        'macro_long_sectors': list(_macro_long_sectors),
    }
    os.makedirs('/Users/rick/ai_trading_bot_v2/cache_scanner', exist_ok=True)
    with open('/Users/rick/ai_trading_bot_v2/cache_scanner/daily_scan.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to cache_scanner/daily_scan.json")
    return result

if __name__ == "__main__":
    run_daily_scan()
