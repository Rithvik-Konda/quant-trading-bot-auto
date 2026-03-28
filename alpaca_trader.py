"""
alpaca_trader.py — Paper trading execution via Alpaca
======================================================
Run at 9:30am ET after daily_scanner.py completes.
"""
import os, sys, json, time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

ALPACA_KEY    = os.environ.get('ALPACA_KEY', '')
ALPACA_SECRET = os.environ.get('ALPACA_SECRET', '')
BASE_URL      = 'https://paper-api.alpaca.markets'
CACHE_DIR     = '/Users/rick/ai_trading_bot_v2/cache_alpaca'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_api():
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url=BASE_URL)

def get_account(api):
    a = api.get_account()
    return {'equity': float(a.equity), 'cash': float(a.cash),
            'buying_power': float(a.buying_power), 'portfolio_value': float(a.portfolio_value)}

def get_positions(api):
    return {p.symbol: {
        'qty': int(p.qty), 'entry_price': float(p.avg_entry_price),
        'market_value': float(p.market_value), 'unrealized_pnl': float(p.unrealized_pl),
        'unrealized_pct': float(p.unrealized_plpc),
    } for p in api.list_positions()}

def submit_order(api, symbol, qty, side):
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side,
                                  type='market', time_in_force='day')
        print(f"  ORDER: {side.upper()} {qty} {symbol}")
        return order
    except Exception as e:
        print(f"  ORDER FAILED: {symbol} {side} {qty} — {e}")
        return None

def get_ml_scores():
    """Load pre-scored ML ranks from daily_scanner.py cache.
    Scanner runs at 9:00am and pre-scores all 447 symbols.
    Trader reads cache at 9:30am — no rescoring needed.
    """
    cache_file = os.path.join(CACHE_DIR, 'today_ml_ranks.json')
    if os.path.exists(cache_file):
        candidates = json.load(open(cache_file))
        print(f"  Loaded {len(candidates)} pre-scored ranks from scanner cache")
        return candidates

    # Fallback: score live if cache missing
    print("  No pre-scored cache — scoring live (slow)...")
    from strategy_core import load_ranker_ensemble
    from backtester_clean import fetch_history
    from ml_model import compute_features
    import config

    rankers   = load_ranker_ensemble()
    candidates = []
    for i, sym in enumerate(config.WATCHLIST):
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
                candidates.append({'symbol': sym, 'ml_score': float(np.mean(scores))})
        except:
            pass
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(config.WATCHLIST)} scored...", end="\r")

    if candidates:
        scores_arr = np.array([c['ml_score'] for c in candidates])
        ranks = scores_arr.argsort().argsort() / max(len(scores_arr)-1, 1)
        for c, r in zip(candidates, ranks):
            c['ml_rank_pct'] = float(r)
    candidates.sort(key=lambda x: x.get('ml_rank_pct', 0), reverse=True)
    return candidates

def run_daily_execution():
    print("="*60)
    print(f"ALPACA PAPER TRADING — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    if not ALPACA_KEY or not ALPACA_SECRET:
        print("ERROR: Set ALPACA_KEY and ALPACA_SECRET")
        return

    # Market hours check — don't trade on holidays
    try:
        _calendar = api_check.get_clock()
        if not _calendar.is_open and datetime.now().hour < 12:
            print(f"Market closed today — skipping execution")
            return
    except:
        pass

    api     = get_api()
    api_check = api  # alias for clarity
    account = get_account(api)
    print(f"\nPortfolio: ${account['portfolio_value']:,.0f}  Cash: ${account['cash']:,.0f}")

    positions = get_positions(api)
    print(f"Open positions: {len(positions)}")
    for sym, pos in positions.items():
        print(f"  {sym:<6}: {pos['qty']} shares  PnL=${pos['unrealized_pnl']:>+,.0f} ({pos['unrealized_pct']:>+.1%})")

    # Reconcile tracked vs actual Alpaca positions
    pos_file = os.path.join(CACHE_DIR, 'positions.json')
    tracked  = json.load(open(pos_file)) if os.path.exists(pos_file) else {}
    # Add any positions in Alpaca not in tracked
    for sym in positions:
        if sym not in tracked:
            print(f"  RECONCILE: {sym} in Alpaca but not tracked — adding")
            tracked[sym] = {
                "entry_date":  str(datetime.now().date()),
                "entry_price": positions[sym]['entry_price'],
                "stop_pct":    0.08,
                "ml_rank_pct": 0.0,
            }
    # Remove any in tracked but not in Alpaca (already closed)
    for sym in list(tracked.keys()):
        if sym not in positions and 'exit_date' not in tracked[sym]:
            print(f"  RECONCILE: {sym} in tracked but not Alpaca — removing")
            tracked.pop(sym)

    # Load scanner
    scan_file = '/Users/rick/ai_trading_bot_v2/cache_scanner/daily_scan.json'
    if not os.path.exists(scan_file):
        print("No scanner output — run daily_scanner.py first")
        return
    scan      = json.load(open(scan_file))
    regime    = scan.get('regime', 'UNKNOWN')
    can_trade = scan.get('can_trade', False)
    bearish   = scan.get('bearish_news', [])
    fda_flags = scan.get('fda_flags', [])
    print(f"\nRegime: {regime}  Can trade: {can_trade}")

    # tracked already loaded and reconciled above

    # ── EXITS ─────────────────────────────────────────────────
    print(f"\n── DAILY STOP UPDATES ──")
    for sym, pos in positions.items():
        if sym not in tracked:
            continue
        # Recompute trailing stop daily
        try:
            from backtester_clean import fetch_history as _fh3
            _df3 = _fh3(sym, days=60)
            _df3.index = pd.to_datetime(_df3.index).tz_localize(None)
            _atr3 = float((_df3['high'] - _df3['low']).tail(14).mean())
            _price3 = float(_df3['close'].iloc[-1])
            _atr_pct3 = _atr3 / _price3
            _entry3 = float(tracked[sym].get('entry_price', _price3))
            _unreal3 = (_price3 - _entry3) / _entry3
            if _unreal3 >= 0.08:
                _new_stop = max(_entry3, _price3 * (1 - 2 * _atr_pct3))
            else:
                _new_stop = _entry3 * (1 - 0.08)
            _old_stop = tracked[sym].get('stop_price', 0)
            if _new_stop > _old_stop:
                tracked[sym]['stop_price'] = round(_new_stop, 2)
                print(f"  {sym}: stop updated ${_old_stop:.2f} → ${_new_stop:.2f}")
        except:
            pass

    print(f"\n── EXITS ──")
    for sym, pos in positions.items():
        meta         = tracked.get(sym, {})
        entry_date   = pd.Timestamp(meta.get('entry_date', str(datetime.now().date())))
        hold_days    = (pd.Timestamp.now() - entry_date).days
        entry_price  = float(meta.get('entry_price', pos['entry_price']))
        stop_pct     = float(meta.get('stop_pct', 0.08))
        stop_price   = entry_price * (1 - stop_pct)
        current_price= entry_price * (1 + pos['unrealized_pct'])
        exit_reason  = None

        if pos['unrealized_pct'] <= -0.15:
            exit_reason = "emergency_stop (-15%)"
        elif current_price <= stop_price and hold_days >= 10:
            exit_reason = f"stop (hold={hold_days}d)"
        elif sym in bearish:
            exit_reason = "bearish_news"
        elif regime == 'BEAR':
            exit_reason = "regime_bear"
        elif hold_days >= 45:
            exit_reason = f"max_hold ({hold_days}d)"

        if exit_reason:
            print(f"  EXIT {sym}: {exit_reason}  PnL=${pos['unrealized_pnl']:>+,.0f}")
            submit_order(api, sym, pos['qty'], 'sell')
            tracked.pop(sym, None)
        else:
            print(f"  HOLD {sym}: {hold_days}d  PnL=${pos['unrealized_pnl']:>+,.0f}")

    # ── ENTRIES ───────────────────────────────────────────────
    print(f"\n── ENTRIES ──")
    slots = 6 - len([s for s in positions if s not in
                     [k for k,v in tracked.items() if 'exit' in str(v)]])

    if not can_trade or regime == 'BEAR':
        print(f"  Conditions unfavorable ({regime}) — no new entries")
    elif slots <= 0:
        print(f"  Portfolio full (6 positions)")
    else:
        # ── Pre-entry market filters ──────────────────────────
        # SPY 5-day momentum filter
        try:
            spy_hist = yf.Ticker('SPY').history(period='10d')
            spy_5d = float(spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-5] - 1)
        except:
            spy_5d = 0.0
        if spy_5d < -0.015:
            print(f"  SPY 5d={spy_5d:+.1%} — momentum filter blocking entries")
            spy_ok = False
        else:
            spy_ok = True
            print(f"  SPY 5d={spy_5d:+.1%} ✓")

        # VIX level for sizing
        try:
            vix_level = float(yf.Ticker('^VIX').info.get('regularMarketPrice', 20))
        except:
            vix_level = 20.0
        if vix_level >= 35:
            vol_scalar = 0.0
        elif vix_level >= 25:
            vol_scalar = 0.5
        elif vix_level >= 20:
            vol_scalar = 0.75
        else:
            vol_scalar = 1.0
        print(f"  VIX={vix_level:.1f}  size_scalar={vol_scalar}")

        # Earnings calendar — block entries within 10 days of earnings
        earnings_dates = {}
        try:
            import config as _cfg
            for sym in _cfg.WATCHLIST[:100]:
                try:
                    cal = yf.Ticker(sym).get_earnings_dates(limit=8)
                    if cal is not None:
                        earnings_dates[sym] = list(pd.to_datetime(cal.index).tz_localize(None))
                except:
                    pass
        except:
            pass

        if not spy_ok or vol_scalar == 0.0:
            print(f"  Market conditions block entries")
        else:
            print(f"  {slots} slot(s) available — scoring watchlist...")
            candidates = get_ml_scores()
            # Load hot theme stocks — prioritize in entry selection
            hot_stocks = []
            try:
                _theme_file = "/Users/rick/ai_trading_bot_v2/cache_themes/hot_stocks.json"
                _new_file   = "/Users/rick/ai_trading_bot_v2/cache_newlistings/rs_scores.json"
                if os.path.exists(_theme_file):
                    hot_stocks += json.load(open(_theme_file)).get("stocks", [])
                # Load cyclical inflection candidates
                _cyc_file = "/Users/rick/ai_trading_bot_v2/cache_cyclical/inflection_scores.json"
                if os.path.exists(_cyc_file):
                    _cyc = json.load(open(_cyc_file))
                    # Only add score>=4 with positive analyst upside
                    hot_stocks += [r["symbol"] for r in _cyc
                                  if r.get("inflection_score",0) >= 4
                                  and r.get("analyst_upside",0) > 0.10]
                if os.path.exists(_new_file):
                    hot_stocks += [r["symbol"] for r in json.load(open(_new_file)) if r.get("tradeable")]
                hot_stocks = list(set(hot_stocks))
                if hot_stocks:
                    print(f"  Hot theme stocks: {hot_stocks[:8]}")
                    theme_cands = [c for c in candidates if c["symbol"] in hot_stocks]
                    other_cands = [c for c in candidates if c["symbol"] not in hot_stocks]
                    candidates  = theme_cands + other_cands
            except Exception as _e:
                pass
            entered = 0
            for cand in candidates:
                if entered >= slots:
                    break
                sym = cand['symbol']
                if sym in positions or sym in bearish or sym in fda_flags:
                    continue
                if cand.get('ml_rank_pct', 0) < 0.90:
                    continue
                # Earnings filter
                earn = earnings_dates.get(sym, [])
                if earn:
                    days_to_earn = min(abs((pd.Timestamp.now() - d).days) for d in earn)
                    if days_to_earn <= 10:
                        continue
                # Quality gate — filter low quality stocks in TRENDING_BULL
                if regime == 'TRENDING_BULL':
                    try:
                        from ml_model import compute_features
                        from backtester_clean import fetch_history as _fhq
                        _df_q = _fhq(sym, days=200)
                        _df_q.index = pd.to_datetime(_df_q.index).tz_localize(None)
                        _feats_q = compute_features(_df_q, symbol=sym)
                        if _feats_q is not None and 'quality_composite' in _feats_q.columns:
                            _qual = float(_feats_q['quality_composite'].iloc[-1])
                            if _qual < 0.20:
                                continue
                    except:
                        pass

                # Entry confirmation — top decile 2 consecutive days
                import os as _os
                _ranks_file = _os.path.join(CACHE_DIR, "prev_ml_ranks.json")
                _prev = json.load(open(_ranks_file)) if _os.path.exists(_ranks_file) else {}
                if _prev.get(sym, 0) < 0.80:
                    continue
                # Accumulation filter
                try:
                    from entry_filter import is_in_accumulation
                    from backtester_clean import fetch_history as _fh
                    _df_s = _fh(sym, days=200)
                    _df_s.index = pd.to_datetime(_df_s.index).tz_localize(None)
                    if not is_in_accumulation(_df_s, regime):
                        continue
                except:
                    pass
                # Size position
                portfolio   = account["portfolio_value"]
                risk_budget = portfolio * 0.025 * vol_scalar
                try:
                    price = float(yf.Ticker(sym).info.get("regularMarketPrice", 0))
                except:
                    continue
                if price <= 0:
                    continue
                qty = int(risk_budget / (price * 0.08))
                qty = min(qty, int(portfolio * 0.20 / price))
                if qty <= 0:
                    continue
                print(f"  ENTER {sym}: qty={qty}  rank={cand['ml_rank_pct']:.3f}  vix_scalar={vol_scalar}")
                order = submit_order(api, sym, qty, "buy")
                if order:
                    # Wait for fill confirmation — poll up to 10s
                    import time as _t
                    fill_price = price  # fallback to quote
                    for _ in range(10):
                        _t.sleep(1)
                        try:
                            _o = api.get_order(order.id)
                            if _o.status == 'filled':
                                fill_price = float(_o.filled_avg_price)
                                print(f"    Fill confirmed: ${fill_price:.2f} (quote was ${price:.2f})")
                                break
                        except:
                            pass
                    tracked[sym] = {
                        "entry_date":  str(datetime.now().date()),
                        "entry_price": fill_price,
                        "stop_pct":    0.08,
                        "ml_rank_pct": cand["ml_rank_pct"],
                        "order_id":    order.id,
                    }
                    entered += 1
                    _t.sleep(0.5)
    # ── BEAR SHORT ENTRIES ───────────────────────────────────
    if regime == 'BEAR' and not cascade_freeze:
        print(f"\n── BEAR SHORTS ──")
        try:
            from strategy_bear import get_bear_candidates
            bear_candidates = get_bear_candidates()
            short_positions = {s: p for s, p in get_positions(api).items()
                              if int(p['qty']) < 0}
            short_slots = 3 - len(short_positions)
            for sym in bear_candidates[:short_slots]:
                if sym in short_positions:
                    continue
                portfolio = account['portfolio_value']
                try:
                    price = float(yf.Ticker(sym).info.get('regularMarketPrice', 0))
                except:
                    continue
                qty = int(portfolio * 0.015 / max(price, 1))
                if qty > 0:
                    print(f"  SHORT {sym}: qty={qty}")
                    submit_order(api, sym, qty, 'sell')
        except Exception as e:
            print(f"  Bear strategy error: {e}")

    with open(pos_file, 'w') as f:
        json.dump(tracked, f, indent=2)
    print(f"\n✓ Done. Saved positions.json")

if __name__ == "__main__":
    run_daily_execution()
