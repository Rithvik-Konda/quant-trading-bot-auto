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
                feats = compute_features(df, symbol=sym)
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
    }
    os.makedirs('/Users/rick/ai_trading_bot_v2/cache_scanner', exist_ok=True)
    with open('/Users/rick/ai_trading_bot_v2/cache_scanner/daily_scan.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to cache_scanner/daily_scan.json")
    return result

if __name__ == "__main__":
    run_daily_scan()
