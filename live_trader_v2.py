"""
live_trader_v2.py — Live execution using same logic as backtester_v2.py
=======================================================================
Mirrors backtester_v2.py exactly:
- Same regime classifier
- Same ML ranker
- Same strategy files (trending, choppy, bear)
- Same entry/exit logic
- Same position sizing

Run daily at 9:25am via launchd.
"""
import os, sys, json, time, requests
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

import config
from ml_model import compute_features, fetch_data
from backtester_clean import (
    FeatureMatrix, batch_ml_scores_fast,
)
from strategy_core import (
    load_ranker_ensemble, load_ranker_regime_ensemble,
)
from regime_classifier import (
    RegimeClassifier, compute_signals, load_macro_data,
    TRENDING_BULL, CHOPPY, BEAR,
    classify_choppy_subregime, CHOPPY_BULL, CHOPPY_BEAR,
)
import strategy_trending as strat_bull
import strategy_choppy  as strat_chop
import strategy_bear    as strat_bear

ALPACA_KEY    = os.environ.get('ALPACA_KEY', 'PKKUPJE3L32EXWBQVVEHZG5O7R')
ALPACA_SECRET = os.environ.get('ALPACA_SECRET', 'F7wJNy6qHNfvztDpdBHhE5NT33eo5ckqUZ7b4krk1FpF')
BASE_URL      = 'https://paper-api.alpaca.markets'
CACHE_DIR     = '/Users/rick/ai_trading_bot_v2/cache_live_v2'
os.makedirs(CACHE_DIR, exist_ok=True)

# Sector map for macro theme integration
_sector_map = {}
try:
    import config as _cfg
    for _etf, _syms in getattr(_cfg, 'SECTOR_ETFS', {}).items():
        for _s in _syms: _sector_map[_s] = _etf
except: pass

HEADERS = {
    'APCA-API-KEY-ID': ALPACA_KEY,
    'APCA-API-SECRET-KEY': ALPACA_SECRET,
    'Content-Type': 'application/json'
}


# ── Alpaca helpers ─────────────────────────────────────────────────────────────

def get_account():
    r = requests.get(f'{BASE_URL}/v2/account', headers=HEADERS)
    a = r.json()
    return {
        'equity':   float(a['equity']),
        'cash':     float(a['cash']),
        'portfolio': float(a['portfolio_value']),
    }


def get_positions():
    r = requests.get(f'{BASE_URL}/v2/positions', headers=HEADERS)
    positions = {}
    for p in r.json():
        positions[p['symbol']] = {
            'qty':        int(float(p['qty'])),
            'entry':      float(p['avg_entry_price']),
            'mkt_value':  float(p['market_value']),
            'pnl':        float(p['unrealized_pl']),
            'pnl_pct':    float(p['unrealized_plpc']),
            'side':       'long' if int(float(p['qty'])) > 0 else 'short',
        }
    return positions


def submit_order(symbol, qty, side, order_type='market', limit_price=None, stop_price=None, tif='day'):
    order = {
        'symbol': symbol,
        'qty': str(abs(qty)),
        'side': side,
        'type': order_type,
        'time_in_force': tif,
    }
    if limit_price: order['limit_price'] = str(round(limit_price, 2))
    if stop_price:  order['stop_price']  = str(round(stop_price, 2))
    r = requests.post(f'{BASE_URL}/v2/orders', headers=HEADERS, json=order)
    result = r.json()
    if 'id' in result:
        print(f"  ✓ {side.upper()} {symbol} qty={qty} → {result['status']}")
        return result
    else:
        print(f"  ✗ {side.upper()} {symbol} FAILED: {result.get('message','unknown')}")
        return None


def get_current_price(symbol):
    try:
        r = requests.get(
            f'{BASE_URL}/v2/stocks/{symbol}/quotes/latest',
            headers=HEADERS
        )
        q = r.json().get('quote', {})
        return float(q.get('ap', 0) or q.get('bp', 0) or 0)
    except:
        return 0


def cancel_all_orders():
    requests.delete(f'{BASE_URL}/v2/orders', headers=HEADERS)


# ── Strategy selector ──────────────────────────────────────────────────────────

def get_strategy(regime):
    if regime == TRENDING_BULL: return strat_bull
    if regime == BEAR:          return strat_bear
    return strat_chop


# ── Main execution ─────────────────────────────────────────────────────────────

def run_live_day():
    print("=" * 60)
    print(f"LIVE TRADER V2 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Market hours check — only trade 9:25am to 3:55pm ET
    now = datetime.now()
    market_open  = now.replace(hour=9,  minute=25, second=0)
    market_close = now.replace(hour=15, minute=55, second=0)
    if not (market_open <= now <= market_close):
        print(f"Market closed ({now.strftime('%H:%M')}) — skipping execution")
        return

    # ── Account ────────────────────────────────────────────────────────────────
    account   = get_account()
    portfolio = account['portfolio']
    print(f"Portfolio: ${portfolio:,.0f}  Cash: ${account['cash']:,.0f}")

    # ── Regime ────────────────────────────────────────────────────────────────
    print("\n[1] Regime classification...")
    spy_macro, hyg_macro, vix_macro = load_macro_data()
    signals = compute_signals(spy_macro, hyg_macro, vix_macro)
    if not signals:
        print("  ERROR: could not compute signals")
        return

    # Breadth from cached prices
    breadth_above = 0
    breadth_total = 0
    for sym in config.WATCHLIST[:100]:
        try:
            df = fetch_data(sym)
            if df is None or len(df) < 200: continue
            ma200 = float(df['close'].rolling(200).mean().iloc[-1])
            price = float(df['close'].iloc[-1])
            breadth_above += int(price > ma200)
            breadth_total += 1
        except: continue
    if breadth_total > 0:
        signals['breadth_pct_above_200d'] = breadth_above / breadth_total

    clf = RegimeClassifier()
    # Load persisted regime state
    state_file = os.path.join(CACHE_DIR, 'regime_state.json')
    if os.path.exists(state_file):
        state = json.load(open(state_file))
        clf._current_regime   = state.get('regime', CHOPPY)
        clf._candidate_regime = state.get('candidate', None)
        clf._candidate_days   = state.get('candidate_days', 0)

    regime = clf.update(datetime.now(), signals)
    # Persist regime state
    json.dump({
        'regime': clf._current_regime,
        'candidate': clf._candidate_regime,
        'candidate_days': clf._candidate_days,
    }, open(state_file, 'w'))

    # CHOPPY sub-regime
    choppy_sub = None
    if regime == CHOPPY:
        choppy_sub = classify_choppy_subregime(signals)

    print(f"  Regime: {regime}" + (f" → {choppy_sub}" if choppy_sub else ""))
    print(f"  SPY YTD: {signals.get('spy_ytd',0):.1%}  VIX: {signals.get('vix_level',0):.1f}  Breadth: {signals.get('breadth_pct_above_200d',0):.0%}")

    # ── ML scoring ────────────────────────────────────────────────────────────
    print("\n[2] ML scoring...")
    try:
        # Use today's ML ranks from scanner (already computed for all 446 symbols)
        ranks_file = '/Users/rick/ai_trading_bot_v2/cache_alpaca/today_ml_ranks.json'
        if not os.path.exists(ranks_file):
            print("  ERROR: no ML ranks — run daily_scanner.py first")
            return
        ranks_list = json.load(open(ranks_file))
        ranks = {r['symbol']: r['ml_rank_pct'] for r in ranks_list}
        print(f"  Loaded {len(ranks)} ML ranks from scanner")
    except Exception as e:
        print(f"  ML scoring failed: {e}")
        return

    # ── Get params based on regime ─────────────────────────────────────────────
    strategy = get_strategy(regime)
    params   = strategy.get_params()
    if regime == CHOPPY and choppy_sub == CHOPPY_BULL:
        params.max_positions      = 4
        params.ml_rank_min        = 0.85
        params.max_positions_long = 4

    max_longs  = getattr(params, 'max_positions', getattr(params, 'max_positions_long', 4))
    ml_min     = getattr(params, 'ml_rank_min',   getattr(params, 'ml_rank_min_long',  0.85))
    max_shorts = getattr(params, 'max_positions_short', 3)

    print(f"  max_longs={max_longs}  ml_min={ml_min:.2f}  max_shorts={max_shorts}")

    # ── Current positions ──────────────────────────────────────────────────────
    print("\n[3] Managing existing positions...")
    positions = get_positions()
    long_positions  = {s: p for s, p in positions.items() if p['side'] == 'long'}
    short_positions = {s: p for s, p in positions.items() if p['side'] == 'short'}
    print(f"  Open longs: {len(long_positions)}  Open shorts: {len(short_positions)}")

    # Load tracked positions with entry dates
    tracked_file = os.path.join(CACHE_DIR, 'tracked_positions.json')
    tracked = json.load(open(tracked_file)) if os.path.exists(tracked_file) else {}

    # ── Exit logic ─────────────────────────────────────────────────────────────
    max_hold = getattr(params, 'max_hold_days', 7)
    today_str = str(datetime.now().date())

    for sym, pos in list(long_positions.items()):
        entry_info = tracked.get(sym, {})
        entry_date = entry_info.get('entry_date', today_str)
        age_days   = (datetime.now().date() - datetime.strptime(entry_date, '%Y-%m-%d').date()).days
        price      = get_current_price(sym)
        entry      = pos['entry']

        # Exit if regime changed to BEAR
        if regime == BEAR:
            print(f"  EXIT {sym}: regime → BEAR")
            submit_order(sym, abs(pos['qty']), 'sell')
            tracked.pop(sym, None)
            continue

        # Exit if max hold reached
        if age_days >= max_hold:
            print(f"  EXIT {sym}: max hold {age_days}d")
            submit_order(sym, abs(pos['qty']), 'sell')
            tracked.pop(sym, None)
            continue

        # Stop loss
        stop_pct = entry_info.get('stop_pct', 0.08)
        if price > 0 and price < entry * (1 - stop_pct):
            print(f"  STOP {sym}: price ${price:.2f} below stop ${entry*(1-stop_pct):.2f}")
            submit_order(sym, abs(pos['qty']), 'sell')
            tracked.pop(sym, None)
            continue

        # Take profit
        take_profit = getattr(params, 'take_profit_pct', 0.25)
        if price > 0 and price > entry * (1 + take_profit):
            print(f"  TAKE PROFIT {sym}: +{(price/entry-1):.1%}")
            submit_order(sym, abs(pos['qty']), 'sell')
            tracked.pop(sym, None)
            continue

    # Exit shorts
    for sym, pos in list(short_positions.items()):
        entry_info = tracked.get(sym, {})
        entry_date = entry_info.get('entry_date', today_str)
        age_days   = (datetime.now().date() - datetime.strptime(entry_date, '%Y-%m-%d').date()).days
        price      = get_current_price(sym)
        entry      = pos['entry']

        # Cover if regime improves
        if regime == TRENDING_BULL:
            print(f"  COVER {sym}: regime → TRENDING_BULL")
            submit_order(sym, abs(pos['qty']), 'buy')
            tracked.pop(sym, None)
            continue

        # Max hold on shorts
        short_hold = getattr(params, 'max_hold_days_short', 10)
        if age_days >= short_hold:
            print(f"  COVER {sym}: max hold {age_days}d")
            submit_order(sym, abs(pos['qty']), 'buy')
            tracked.pop(sym, None)
            continue

        # Stop on shorts (price rises 8%)
        if price > 0 and price > entry * 1.08:
            print(f"  STOP SHORT {sym}: price ${price:.2f} above stop")
            submit_order(sym, abs(pos['qty']), 'buy')
            tracked.pop(sym, None)
            continue

        # Take profit on shorts (price falls 8%)
        if price > 0 and price < entry * 0.92:
            print(f"  COVER PROFIT {sym}: -{(1-price/entry):.1%}")
            submit_order(sym, abs(pos['qty']), 'buy')
            tracked.pop(sym, None)
            continue

    # Refresh positions after exits
    positions     = get_positions()
    long_positions  = {s: p for s, p in positions.items() if p['side'] == 'long'}
    short_positions = {s: p for s, p in positions.items() if p['side'] == 'short'}

    # ── New entries ────────────────────────────────────────────────────────────
    print("\n[4] New entries...")
    account = get_account()
    portfolio = account['portfolio']

    # Long entries
    long_slots = max_longs - len(long_positions)
    if long_slots > 0 and regime != BEAR:
        long_candidates = [(sym, rank) for sym, rank in ranks.items()
                          if rank >= ml_min and sym not in long_positions]
        long_candidates.sort(key=lambda x: x[1], reverse=True)
        # Load macro/news signals from scanner
    scan_file = '/Users/rick/ai_trading_bot_v2/cache_scanner/daily_scan.json'
    macro_risk         = False
    news_score         = 0.0
    macro_long_sectors = set()
    macro_short_sectors= set()
    if os.path.exists(scan_file):
        _scan = json.load(open(scan_file))
        macro_risk          = _scan.get('macro_risk', False)
        news_score          = _scan.get('news_score', 0.0)
        macro_long_sectors  = set(_scan.get('macro_long_sectors', []))
        macro_short_sectors = set(_scan.get('macro_short_sectors', []))
        if macro_risk:
            print(f"  ⚠️  Macro risk detected — news_score={news_score:+.2f}")
            print(f"  Long sectors: {macro_long_sectors}  Short sectors: {macro_short_sectors}")

    # Load prev ML ranks for entry confirmation
        prev_ranks_file = '/Users/rick/ai_trading_bot_v2/cache_alpaca/prev_ml_ranks.json'
        prev_ranks = {}
        if os.path.exists(prev_ranks_file):
            pr = json.load(open(prev_ranks_file))
            prev_ranks = {r['symbol']: r['ml_rank_pct'] for r in pr} if isinstance(pr, list) else pr

        # VIX scalar
        vix_level = signals.get('vix_level', 20)
        vol_scalar = 0.0 if vix_level >= 35 else 0.5 if vix_level >= 25 else 0.75 if vix_level >= 20 else 1.0
        if signals.get('spy_5d', 0) < -0.015:
            vol_scalar = 0.0
            print(f"  SPY weak — blocking new longs")
        if vol_scalar == 0.0:
            print(f"  VIX={vix_level:.0f} — no new longs")

        entered = 0
        for sym, rank in long_candidates[:long_slots * 2]:
            if entered >= long_slots: break
            if vol_scalar == 0.0: break
            # Entry confirmation
            if prev_ranks.get(sym, 0) < ml_min:
                continue
            price = get_current_price(sym)
            if price <= 0: continue
            # ATR-based stop
            try:
                import pandas as pd
                df = fetch_data(sym)
                if df is not None and len(df) >= 14:
                    tr = pd.concat([df['high']-df['low'],
                        (df['high']-df['close'].shift()).abs(),
                        (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
                    stop_pct = float(np.clip(2.0 * tr.tail(14).mean() / df['close'].iloc[-1], 0.04, 0.20))
                else:
                    stop_pct = 0.08
            except:
                stop_pct = 0.08
            # Macro confirmation — size up if ML + macro aligned
            _sym_sector = _sector_map.get(sym, '') if '_sector_map' in dir() else ''
            _macro_conf = _sym_sector in macro_long_sectors if macro_long_sectors else False
            _macro_fade = _sym_sector in macro_short_sectors if macro_short_sectors else False
            if _macro_fade:
                print(f"  MACRO FADE {sym} — sector {_sym_sector} in macro short list, skipping")
                continue
            _size_mult = 1.3 if _macro_conf else 1.0
            if macro_risk and not _macro_conf:
                _size_mult = 0.7  # reduce size in macro risk if not confirmed
            risk_budget = portfolio * 0.035 * vol_scalar * _size_mult
            qty = int(risk_budget / (price * stop_pct))
            qty = min(qty, int(portfolio * 0.15 / price))
            if qty < 1: continue
            result = submit_order(sym, qty, 'buy')
            if result:
                tracked[sym] = {
                    'entry_date': today_str,
                    'entry_price': price,
                    'highest_price': price,
                    'stop_pct': stop_pct,
                    'ml_rank': rank,
                    'side': 'long'
                }
                entered += 1

    # Short entries — always run if CHOPPY_BEAR or BEAR
    short_slots = max_shorts - len(short_positions)
    if short_slots > 0 and regime in (BEAR, CHOPPY) and choppy_sub != CHOPPY_BULL:
        short_candidates = [(sym, rank) for sym, rank in ranks.items()
                           if rank <= 0.08 and sym not in short_positions]
        short_candidates.sort(key=lambda x: x[1])

        # Filter: must be below MA50 and in downtrend
        filtered_shorts = []
        for sym, rank in short_candidates[:20]:
            try:
                df = fetch_data(sym)
                if df is None or len(df) < 50: continue
                price  = float(df['close'].iloc[-1])
                ma50   = float(df['close'].rolling(50).mean().iloc[-1])
                ret_5  = float(df['close'].pct_change(5).iloc[-1])
                if price < ma50 and ret_5 < 0:
                    filtered_shorts.append((sym, rank, price))
            except: continue

        entered_shorts = 0
        for sym, rank, price in filtered_shorts[:short_slots]:
            if entered_shorts >= short_slots: break
            stop_pct   = 0.08
            risk_budget = portfolio * 0.025
            qty = int(risk_budget / (price * stop_pct))
            qty = min(qty, int(portfolio * 0.10 / price))
            if qty < 1: continue
            result = submit_order(sym, qty, 'sell')
            if result:
                tracked[sym] = {
                    'entry_date': today_str,
                    'entry_price': price,
                    'stop_pct': stop_pct,
                    'ml_rank': rank,
                    'side': 'short'
                }
                entered_shorts += 1

    # Save tracked positions
    json.dump(tracked, open(tracked_file, 'w'), indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    positions = get_positions()
    account   = get_account()
    print(f"\n{'='*60}")
    print(f"DONE — Portfolio: ${account['portfolio']:,.0f}")
    print(f"Longs: {sum(1 for p in positions.values() if p['side']=='long')}  "
          f"Shorts: {sum(1 for p in positions.values() if p['side']=='short')}")
    for sym, p in positions.items():
        print(f"  {sym:<6}: {'L' if p['side']=='long' else 'S'}  "
              f"qty={abs(p['qty'])}  PnL=${p['pnl']:+,.0f} ({p['pnl_pct']:+.1%})")


if __name__ == "__main__":
    run_live_day()
