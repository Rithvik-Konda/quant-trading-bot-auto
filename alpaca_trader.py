"""
alpaca_trader.py — Paper trading execution via Alpaca
======================================================
Run at 9:30am ET after daily_scanner.py completes.
"""
import os, sys, json, time
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
    """Score all watchlist symbols with current ML ranker."""
    from strategy_core import load_ranker_ensemble
    from backtester_clean import fetch_history
    from ml_model import compute_features
    import config

    rankers = load_ranker_ensemble()
    candidates = []
    print(f"  Scoring {len(config.WATCHLIST)} symbols...")
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
            print(f"  {i+1}/{len(config.WATCHLIST)} scored...", end='\r')

    # Convert to cross-sectional percentile rank
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

    api     = get_api()
    account = get_account(api)
    print(f"\nPortfolio: ${account['portfolio_value']:,.0f}  Cash: ${account['cash']:,.0f}")

    positions = get_positions(api)
    print(f"Open positions: {len(positions)}")
    for sym, pos in positions.items():
        print(f"  {sym:<6}: {pos['qty']} shares  PnL=${pos['unrealized_pnl']:>+,.0f} ({pos['unrealized_pct']:>+.1%})")

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

    # Load tracked positions
    pos_file = os.path.join(CACHE_DIR, 'positions.json')
    tracked  = json.load(open(pos_file)) if os.path.exists(pos_file) else {}

    # ── EXITS ─────────────────────────────────────────────────
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
        print(f"  {slots} slot(s) available — scoring watchlist...")
        candidates = get_ml_scores()
        entered = 0
        for cand in candidates:
            if entered >= slots:
                break
            sym = cand['symbol']
            if sym in positions or sym in bearish or sym in fda_flags:
                continue
            if cand.get('ml_rank_pct', 0) < 0.90:
                continue
            # Size position
            portfolio   = account['portfolio_value']
            risk_budget = portfolio * 0.025
            try:
                import yfinance as yf
                price = float(yf.Ticker(sym).info.get('regularMarketPrice', 0))
            except:
                continue
            if price <= 0:
                continue
            qty = int(risk_budget / (price * 0.08))
            qty = min(qty, int(portfolio * 0.20 / price))
            if qty <= 0:
                continue
            print(f"  ENTER {sym}: qty={qty}  rank={cand['ml_rank_pct']:.3f}")
            order = submit_order(api, sym, qty, 'buy')
            if order:
                tracked[sym] = {
                    'entry_date':  str(datetime.now().date()),
                    'entry_price': price,
                    'stop_pct':    0.08,
                    'ml_rank_pct': cand['ml_rank_pct'],
                }
                entered += 1
            time.sleep(0.5)

    with open(pos_file, 'w') as f:
        json.dump(tracked, f, indent=2)
    print(f"\n✓ Done. Saved positions.json")

if __name__ == "__main__":
    run_daily_execution()
