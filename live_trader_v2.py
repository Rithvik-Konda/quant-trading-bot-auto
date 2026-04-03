"""
live_trader_v2.py — Live execution with intraday entry timing
=============================================================
Architecture:
  - run_exits()   → called at 9:25am, handles all exits once
  - run_entries() → loops every 5min from 10:00am to 3:30pm
                    waits for intraday confirmation before entering

Intraday confirmation by engine:
  - momentum:      price > VWAP AND volume_ratio > 1.2
  - meanrev:       price < prior_close AND volume_ratio > 1.1 AND RSI(2) still < 5
  - short:         price < VWAP AND price < MA50 AND failed bounce confirmed

Usage:
  python live_trader_v2.py --exits    # run at 9:25am via launchd
  python live_trader_v2.py --entries  # run at 10:00am via launchd (loops until 3:30pm)
  python live_trader_v2.py            # run both (legacy single-shot mode)
"""
import os, sys, json, time, argparse, requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

import config
from ml_model import compute_features, fetch_data
from backtester_clean import FeatureMatrix, batch_ml_scores_fast
from strategy_core import load_ranker_ensemble, load_ranker_regime_ensemble
from regime_classifier import (
    RegimeClassifier, compute_signals, load_macro_data,
    TRENDING_BULL, CHOPPY, BEAR,
    classify_choppy_subregime, CHOPPY_BULL, CHOPPY_BEAR,
)
import strategy_trending as strat_bull
import strategy_choppy   as strat_chop
import strategy_bear     as strat_bear
import strategy_meanrev  as strat_mr

# ── Config ─────────────────────────────────────────────────────────────────────
ALPACA_KEY    = os.environ.get('ALPACA_KEY',    'PKKUPJE3L32EXWBQVVEHZG5O7R')
ALPACA_SECRET = os.environ.get('ALPACA_SECRET', 'F7wJNy6qHNfvztDpdBHhE5NT33eo5ckqUZ7b4krk1FpF')
TRADE_URL     = 'https://paper-api.alpaca.markets'
DATA_URL      = 'https://data.alpaca.markets'
CACHE_DIR     = '/Users/rick/ai_trading_bot_v2/cache_live_v2'
os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {
    'APCA-API-KEY-ID':     ALPACA_KEY,
    'APCA-API-SECRET-KEY': ALPACA_SECRET,
    'Content-Type':        'application/json'
}

# Entry loop config
ENTRY_LOOP_INTERVAL_SEC = 300        # check every 5 minutes
ENTRY_LOOP_START        = (10, 0)    # 10:00am ET
ENTRY_LOOP_END          = (15, 30)   # 3:30pm ET

# Sector map
_sector_map = {}
try:
    for _etf, _syms in getattr(config, 'SECTOR_ETFS', {}).items():
        for _s in _syms: _sector_map[_s] = _etf
except: pass

# ── Alpaca trading helpers ─────────────────────────────────────────────────────
def get_account():
    r = requests.get(f'{TRADE_URL}/v2/account', headers=HEADERS)
    a = r.json()
    return {
        'portfolio': float(a.get('portfolio_value', 0)),
        'cash':      float(a.get('cash', 0)),
    }

def get_positions():
    r = requests.get(f'{TRADE_URL}/v2/positions', headers=HEADERS)
    out = {}
    for p in r.json():
        if not isinstance(p, dict): continue
        sym = p.get('symbol', '')
        out[sym] = {
            'qty':     float(p.get('qty', 0)),
            'side':    p.get('side', 'long'),
            'entry':   float(p.get('avg_entry_price', 0)),
            'pnl':     float(p.get('unrealized_pl', 0)),
            'pnl_pct': float(p.get('unrealized_plpc', 0)),
        }
    return out

def submit_order(symbol, qty, side, order_type='market', limit_price=None, stop_price=None, tif='day'):
    order = {
        'symbol': symbol,
        'qty':    str(abs(qty)),
        'side':   side,
        'type':   order_type,
        'time_in_force': tif,
    }
    if limit_price: order['limit_price'] = str(round(limit_price, 2))
    if stop_price:  order['stop_price']  = str(round(stop_price, 2))
    r = requests.post(f'{TRADE_URL}/v2/orders', headers=HEADERS, json=order)
    result = r.json()
    if 'id' in result:
        print(f"  ✓ {side.upper()} {symbol} qty={qty} → {result['status']}")
        return result
    else:
        print(f"  ✗ {side.upper()} {symbol} FAILED: {result.get('message','unknown')}")
        return None

def get_current_price(symbol):
    try:
        r = requests.get(f'{TRADE_URL}/v2/stocks/{symbol}/quotes/latest', headers=HEADERS)
        q = r.json().get('quote', {})
        return float(q.get('ap', 0) or q.get('bp', 0) or 0)
    except:
        return 0

def cancel_all_orders():
    requests.delete(f'{TRADE_URL}/v2/orders', headers=HEADERS)

# ── Intraday data helpers ──────────────────────────────────────────────────────
def get_intraday_bars(symbol, timeframe='5Min', limit=78):
    """Pull intraday bars from Alpaca data API. Returns DataFrame or None."""
    try:
        r = requests.get(
            f'{DATA_URL}/v2/stocks/{symbol}/bars',
            headers=HEADERS,
            params={
                'timeframe': timeframe,
                'limit':     limit,
                'feed':      'iex',
            }
        )
        bars = r.json().get('bars', [])
        if not bars:
            return None
        df = pd.DataFrame(bars)
        df.columns = [c.lower() for c in df.columns]
        # Rename Alpaca columns to standard names
        rename = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'time'}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        return df
    except:
        return None

def compute_vwap(bars_df):
    """Compute VWAP from intraday bars."""
    try:
        typical = (bars_df['high'] + bars_df['low'] + bars_df['close']) / 3
        return float((typical * bars_df['volume']).sum() / bars_df['volume'].sum())
    except:
        return None

def get_volume_ratio(bars_df, lookback_bars=12):
    """Current bar volume vs average of prior N bars."""
    try:
        if len(bars_df) < 2:
            return 1.0
        recent_vol = float(bars_df['volume'].iloc[-1])
        avg_vol    = float(bars_df['volume'].iloc[:-1].tail(lookback_bars).mean())
        return recent_vol / avg_vol if avg_vol > 0 else 1.0
    except:
        return 1.0

def get_intraday_confirmation(symbol, engine, prior_close=None):
    """
    Returns (confirmed: bool, reason: str) based on intraday price action.

    Momentum:    price > VWAP AND volume_ratio > 1.2
    Mean rev:    price < prior_close AND volume_ratio > 1.1 AND RSI(2) still < 5
    Short:       price < VWAP AND price below MA50 intraday
    """
    bars = get_intraday_bars(symbol)
    if bars is None or len(bars) < 3:
        return False, "no intraday bars"

    price  = float(bars['close'].iloc[-1])
    vwap   = compute_vwap(bars)
    volr   = get_volume_ratio(bars)

    if vwap is None:
        return False, "vwap failed"

    if engine == 'momentum':
        if price <= vwap:
            return False, f"price ${price:.2f} below VWAP ${vwap:.2f}"
        if volr < 1.2:
            return False, f"volume ratio {volr:.2f} < 1.2"
        return True, f"price ${price:.2f} > VWAP ${vwap:.2f}, vol_ratio={volr:.2f}"

    elif engine == 'meanrev':
        # RSI(2) check on intraday bars
        if len(bars) >= 3:
            delta = bars['close'].diff()
            gain  = delta.clip(lower=0).rolling(2).mean()
            loss  = (-delta.clip(upper=0)).rolling(2).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi2  = float(100 - 100 / (1 + rs.iloc[-1]))
        else:
            rsi2 = 50.0

        if prior_close is not None and price >= prior_close:
            return False, f"price ${price:.2f} gapped above prior close ${prior_close:.2f} — edge gone"
        if volr < 1.1:
            return False, f"volume ratio {volr:.2f} < 1.1 — no buyer confirmation"
        if rsi2 >= 15:
            return False, f"RSI(2) {rsi2:.1f} no longer oversold intraday"
        return True, f"price ${price:.2f} < prior_close, vol_ratio={volr:.2f}, RSI2={rsi2:.1f}"

    elif engine == 'short':
        # Need price below VWAP — failed bounce check
        if price >= vwap:
            return False, f"price ${price:.2f} above VWAP ${vwap:.2f} — not failing yet"
        # Check if price bounced and rejected (high > current price by >0.5%)
        session_high = float(bars['high'].max())
        if session_high < price * 1.005:
            return False, "no bounce to reject yet — wait"
        return True, f"failed bounce confirmed: high=${session_high:.2f} rejected, now ${price:.2f} < VWAP ${vwap:.2f}"

    return False, "unknown engine"

# ── Strategy selector ──────────────────────────────────────────────────────────
def get_strategy(regime):
    if regime == TRENDING_BULL: return strat_bull
    if regime == BEAR:          return strat_bear
    return strat_chop

# ── Shared state loader ────────────────────────────────────────────────────────
def load_regime_and_signals():
    """Load macro data, compute signals, classify regime. Returns (regime, choppy_sub, signals)."""
    spy_macro, hyg_macro, vix_macro = load_macro_data()
    signals = compute_signals(spy_macro, hyg_macro, vix_macro)
    if not signals:
        return None, None, None

    # Breadth
    breadth_above = breadth_total = 0
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
    state_file = os.path.join(CACHE_DIR, 'regime_state.json')
    if os.path.exists(state_file):
        state = json.load(open(state_file))
        clf._current_regime   = state.get('regime', CHOPPY)
        clf._candidate_regime = state.get('candidate', None)
        clf._candidate_days   = state.get('candidate_days', 0)

    regime = clf.update(datetime.now(), signals)
    json.dump({
        'regime':         clf._current_regime,
        'candidate':      clf._candidate_regime,
        'candidate_days': clf._candidate_days,
    }, open(state_file, 'w'))

    choppy_sub = classify_choppy_subregime(signals) if regime == CHOPPY else None
    return regime, choppy_sub, signals

def load_ml_ranks():
    """Load today's ML ranks from scanner output."""
    ranks_file = '/Users/rick/ai_trading_bot_v2/cache_alpaca/today_ml_ranks.json'
    if not os.path.exists(ranks_file):
        print("  ERROR: no ML ranks — run daily_scanner.py first")
        return None
    ranks_list = json.load(open(ranks_file))
    return {r['symbol']: r['ml_rank_pct'] for r in ranks_list}

def load_scan_meta():
    """Load macro/news signals from daily scan."""
    scan_file = '/Users/rick/ai_trading_bot_v2/cache_scanner/daily_scan.json'
    if not os.path.exists(scan_file):
        return False, 0.0, set(), set()
    _scan = json.load(open(scan_file))
    return (
        _scan.get('macro_risk', False),
        _scan.get('news_score', 0.0),
        set(_scan.get('macro_long_sectors', [])),
        set(_scan.get('macro_short_sectors', [])),
    )

# ── EXIT MANAGER ──────────────────────────────────────────────────────────────
def run_exits():
    """Run at 9:25am. Handles all exits for existing positions."""
    print("=" * 60)
    print(f"EXIT MANAGER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    regime, choppy_sub, signals = load_regime_and_signals()
    if regime is None:
        print("ERROR: could not compute regime")
        return

    print(f"  Regime: {regime}" + (f" → {choppy_sub}" if choppy_sub else ""))

    strategy  = get_strategy(regime)
    params    = strategy.get_params()
    max_hold  = getattr(params, 'max_hold_days', 7)
    today_str = str(datetime.now().date())

    positions    = get_positions()
    tracked_file = os.path.join(CACHE_DIR, 'tracked_positions.json')
    tracked      = json.load(open(tracked_file)) if os.path.exists(tracked_file) else {}

    long_positions  = {s: p for s, p in positions.items() if p['side'] == 'long'}
    short_positions = {s: p for s, p in positions.items() if p['side'] == 'short'}
    print(f"\n[exits] Longs: {len(long_positions)}  Shorts: {len(short_positions)}")

    # ── Long exits ────────────────────────────────────────────────────────────
    for sym, pos in list(long_positions.items()):
        entry_info = tracked.get(sym, {})
        engine     = entry_info.get('engine', 'momentum')
        entry_date = entry_info.get('entry_date', today_str)
        age_days   = (datetime.now().date() - datetime.strptime(entry_date, '%Y-%m-%d').date()).days
        price      = get_current_price(sym)
        entry      = pos['entry']

        # Regime → BEAR: exit all longs
        if regime == BEAR:
            print(f"  EXIT {sym}: regime → BEAR")
            submit_order(sym, abs(pos['qty']), 'sell')
            tracked.pop(sym, None); continue

        # Mean reversion exits (different logic)
        if engine == 'meanrev':
            bars = get_intraday_bars(sym)
            if bars is not None and len(bars) >= 5:
                sma5 = float(bars['close'].tail(5).mean())
                if price >= sma5:
                    print(f"  EXIT MEANREV {sym}: price ${price:.2f} crossed SMA5 ${sma5:.2f}")
                    submit_order(sym, abs(pos['qty']), 'sell')
                    tracked.pop(sym, None); continue
            if age_days >= 10:
                print(f"  EXIT MEANREV {sym}: 10-day timeout")
                submit_order(sym, abs(pos['qty']), 'sell')
                tracked.pop(sym, None); continue
            # Emergency stop at -10%
            if price > 0 and price < entry * 0.90:
                print(f"  STOP MEANREV {sym}: -10% structural breakdown")
                submit_order(sym, abs(pos['qty']), 'sell')
                tracked.pop(sym, None); continue
            continue

        # Momentum exits
        if age_days >= max_hold:
            print(f"  EXIT {sym}: max hold {age_days}d")
            submit_order(sym, abs(pos['qty']), 'sell')
            tracked.pop(sym, None); continue

        stop_pct = entry_info.get('stop_pct', 0.08)
        if price > 0 and price < entry * (1 - stop_pct):
            print(f"  STOP {sym}: ${price:.2f} below stop ${entry*(1-stop_pct):.2f}")
            submit_order(sym, abs(pos['qty']), 'sell')
            tracked.pop(sym, None); continue

        take_profit = getattr(params, 'take_profit_pct', 0.25)
        if price > 0 and price > entry * (1 + take_profit):
            print(f"  TAKE PROFIT {sym}: +{(price/entry-1):.1%}")
            submit_order(sym, abs(pos['qty']), 'sell')
            tracked.pop(sym, None); continue

    # ── Short exits ───────────────────────────────────────────────────────────
    short_hold = getattr(params, 'max_hold_days_short', 10)
    for sym, pos in list(short_positions.items()):
        entry_info = tracked.get(sym, {})
        entry_date = entry_info.get('entry_date', today_str)
        age_days   = (datetime.now().date() - datetime.strptime(entry_date, '%Y-%m-%d').date()).days
        price      = get_current_price(sym)
        entry      = pos['entry']

        if regime == TRENDING_BULL:
            print(f"  COVER {sym}: regime → TRENDING_BULL")
            submit_order(sym, abs(pos['qty']), 'buy')
            tracked.pop(sym, None); continue

        if age_days >= short_hold:
            print(f"  COVER {sym}: max hold {age_days}d")
            submit_order(sym, abs(pos['qty']), 'buy')
            tracked.pop(sym, None); continue

        if price > 0 and price > entry * 1.08:
            print(f"  STOP SHORT {sym}: ${price:.2f} above stop")
            submit_order(sym, abs(pos['qty']), 'buy')
            tracked.pop(sym, None); continue

        if price > 0 and price < entry * 0.92:
            print(f"  COVER PROFIT {sym}: -{(1-price/entry):.1%}")
            submit_order(sym, abs(pos['qty']), 'buy')
            tracked.pop(sym, None); continue

    json.dump(tracked, open(tracked_file, 'w'), indent=2)
    print(f"\n[exits] Done — {datetime.now().strftime('%H:%M')}")

# ── ENTRY MANAGER ─────────────────────────────────────────────────────────────
def run_entry_scan(regime, choppy_sub, signals, ranks, tracked, portfolio):
    """
    Single entry scan pass. Called every 5 minutes by the entry loop.
    Returns number of new entries made.
    """
    strategy = get_strategy(regime)
    params   = strategy.get_params()
    if regime == CHOPPY and choppy_sub == CHOPPY_BULL:
        params.max_positions   = 4
        params.ml_rank_min     = 0.85

    max_longs  = getattr(params, 'max_positions',       getattr(params, 'max_positions_long', 4))
    ml_min     = getattr(params, 'ml_rank_min',         getattr(params, 'ml_rank_min_long',  0.85))
    max_shorts = getattr(params, 'max_positions_short', 3)

    positions      = get_positions()
    long_positions  = {s: p for s, p in positions.items() if p['side'] == 'long'}
    short_positions = {s: p for s, p in positions.items() if p['side'] == 'short'}

    macro_risk, news_score, macro_long_sectors, macro_short_sectors = load_scan_meta()
    today_str = str(datetime.now().date())

    # VIX gate
    vix_level  = signals.get('vix_level', 20)
    vol_scalar = 0.0 if vix_level >= 35 else 0.5 if vix_level >= 25 else 0.75 if vix_level >= 20 else 1.0
    if signals.get('spy_5d', 0) < -0.015:
        vol_scalar = 0.0

    entered = 0

    # ── Long entries (momentum) ────────────────────────────────────────────────
    long_slots = max_longs - len(long_positions)
    if long_slots > 0 and regime != BEAR and vol_scalar > 0:
        candidates = [(s, r) for s, r in ranks.items()
                      if r >= ml_min and s not in long_positions and s not in tracked]
        candidates.sort(key=lambda x: x[1], reverse=True)

        for sym, rank in candidates[:long_slots * 3]:
            if entered >= long_slots: break

            # Macro filter
            sector = _sector_map.get(sym, '')
            if sector in macro_short_sectors:
                continue
            macro_conf = sector in macro_long_sectors if macro_long_sectors else False
            size_mult  = 1.3 if macro_conf else (0.7 if macro_risk else 1.0)

            # Intraday confirmation
            confirmed, reason = get_intraday_confirmation(sym, 'momentum')
            if not confirmed:
                print(f"  SKIP {sym} momentum: {reason}")
                continue

            price = get_current_price(sym)
            if price <= 0: continue

            # ATR stop
            try:
                df = fetch_data(sym)
                if df is not None and len(df) >= 14:
                    tr = pd.concat([
                        df['high'] - df['low'],
                        (df['high'] - df['close'].shift()).abs(),
                        (df['low']  - df['close'].shift()).abs()
                    ], axis=1).max(axis=1)
                    stop_pct = float(np.clip(2.0 * tr.tail(14).mean() / df['close'].iloc[-1], 0.04, 0.20))
                else:
                    stop_pct = 0.08
            except:
                stop_pct = 0.08

            risk_budget = portfolio * 0.035 * vol_scalar * size_mult
            qty = int(risk_budget / (price * stop_pct))
            qty = min(qty, int(portfolio * 0.15 / price))
            if qty < 1: continue

            result = submit_order(sym, qty, 'buy')
            if result:
                print(f"  ✓ ENTRY momentum {sym}: {reason}")
                tracked[sym] = {
                    'entry_date':    today_str,
                    'entry_price':   price,
                    'highest_price': price,
                    'stop_pct':      stop_pct,
                    'ml_rank':       rank,
                    'engine':        'momentum',
                    'side':          'long',
                }
                entered += 1

    # ── Mean reversion entries (CHOPPY only) ───────────────────────────────────
    if regime == CHOPPY and vol_scalar > 0:
        mr_positions = [s for s, t in tracked.items() if t.get('engine') == 'meanrev']
        mr_slots     = strat_mr.MAX_POSITIONS - len(mr_positions)

        if mr_slots > 0:
            # Get oversold candidates from full universe
            mr_candidates = []
            for sym in config.WATCHLIST:
                if sym in long_positions or sym in tracked: continue
                try:
                    df = fetch_data(sym)
                    if df is None or len(df) < 210: continue
                    snap = strat_mr.compute_snapshot(sym, df)
                    if snap is None: continue
                    params_mr = strat_mr.MeanRevParams()
                    ok, _ = strat_mr.can_enter_meanrev(snap, params_mr)
                    if ok:
                        mr_candidates.append((sym, snap.rsi2, float(df['close'].iloc[-2])))  # prior close
                except: continue

            mr_candidates.sort(key=lambda x: x[1])  # lowest RSI first

            for sym, rsi2, prior_close in mr_candidates[:mr_slots * 3]:
                if len([s for s, t in tracked.items() if t.get('engine') == 'meanrev']) >= strat_mr.MAX_POSITIONS:
                    break

                confirmed, reason = get_intraday_confirmation(sym, 'meanrev', prior_close=prior_close)
                if not confirmed:
                    print(f"  SKIP {sym} meanrev: {reason}")
                    continue

                price = get_current_price(sym)
                if price <= 0: continue

                df = fetch_data(sym)
                snap = strat_mr.compute_snapshot(sym, df) if df is not None else None
                if snap is None: continue

                ml_rank = ranks.get(sym, 0.5)
                qty     = strat_mr.size_meanrev_position(capital=portfolio, price=price, params=strat_mr.MeanRevParams(), n_open_positions=len(mr_positions), snap=snap, ml_rank=ml_rank)
                if qty < 1: continue

                result = submit_order(sym, qty, 'buy')
                if result:
                    print(f"  ✓ ENTRY meanrev {sym} RSI2={rsi2:.1f}: {reason}")
                    tracked[sym] = {
                        'entry_date':  today_str,
                        'entry_price': price,
                        'stop_pct':    0.10,
                        'ml_rank':     ml_rank,
                        'engine':      'meanrev',
                        'side':        'long',
                        'rsi2':        rsi2,
                    }
                    entered += 1

    # ── Short entries ──────────────────────────────────────────────────────────
    short_slots = max_shorts - len(short_positions)
    if short_slots > 0 and regime in (BEAR, CHOPPY) and choppy_sub != CHOPPY_BULL:
        short_candidates = [(s, r) for s, r in ranks.items()
                            if r <= 0.08 and s not in short_positions and s not in tracked]
        short_candidates.sort(key=lambda x: x[1])

        for sym, rank in short_candidates[:20]:
            if len(short_positions) >= max_shorts: break
            try:
                df = fetch_data(sym)
                if df is None or len(df) < 50: continue
                price = float(df['close'].iloc[-1])
                ma50  = float(df['close'].rolling(50).mean().iloc[-1])
                ret5  = float(df['close'].pct_change(5).iloc[-1])
                if not (price < ma50 and ret5 < 0): continue
            except: continue

            confirmed, reason = get_intraday_confirmation(sym, 'short')
            if not confirmed:
                print(f"  SKIP {sym} short: {reason}")
                continue

            price = get_current_price(sym)
            if price <= 0: continue

            stop_pct    = 0.08
            risk_budget = portfolio * 0.025
            qty = int(risk_budget / (price * stop_pct))
            qty = min(qty, int(portfolio * 0.10 / price))
            if qty < 1: continue

            result = submit_order(sym, qty, 'sell')
            if result:
                print(f"  ✓ ENTRY short {sym}: {reason}")
                tracked[sym] = {
                    'entry_date':  today_str,
                    'entry_price': price,
                    'stop_pct':    stop_pct,
                    'ml_rank':     rank,
                    'engine':      'short',
                    'side':        'short',
                }
                entered += 1
                short_positions[sym] = {'side': 'short', 'qty': -qty, 'entry': price, 'pnl': 0, 'pnl_pct': 0}

    return entered

def run_entries():
    """
    Entry loop — runs from 10:00am to 3:30pm, scanning every 5 minutes.
    Waits for intraday confirmation before entering any position.
    """
    print("=" * 60)
    print(f"ENTRY MANAGER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    now   = datetime.now()
    start = now.replace(hour=ENTRY_LOOP_START[0], minute=ENTRY_LOOP_START[1], second=0)
    end   = now.replace(hour=ENTRY_LOOP_END[0],   minute=ENTRY_LOOP_END[1],   second=0)

    if now > end:
        print("Past 3:30pm — entry window closed")
        return

    # Wait until 10:00am if launched early
    if now < start:
        wait = (start - now).seconds
        print(f"Waiting {wait//60}min until 10:00am...")
        time.sleep(wait)

    # Load shared state once at start of loop
    regime, choppy_sub, signals = load_regime_and_signals()
    if regime is None:
        print("ERROR: could not compute regime")
        return

    ranks = load_ml_ranks()
    if ranks is None:
        return

    print(f"  Regime: {regime}" + (f" → {choppy_sub}" if choppy_sub else ""))
    print(f"  VIX: {signals.get('vix_level',0):.1f}  SPY 5d: {signals.get('spy_5d',0):.1%}")
    print(f"  Loaded {len(ranks)} ML ranks")
    print(f"  Scanning every 5min until 3:30pm...\n")

    tracked_file = os.path.join(CACHE_DIR, 'tracked_positions.json')
    total_entered = 0
    loop = 0

    while datetime.now() <= end:
        loop += 1
        now_str = datetime.now().strftime('%H:%M')
        print(f"─── Scan #{loop} at {now_str} ───")

        # Reload tracked each loop (exits may have removed positions)
        tracked = json.load(open(tracked_file)) if os.path.exists(tracked_file) else {}
        account  = get_account()
        portfolio = account['portfolio']

        n = run_entry_scan(regime, choppy_sub, signals, ranks, tracked, portfolio)
        total_entered += n

        # Save tracked after each scan
        json.dump(tracked, open(tracked_file, 'w'), indent=2)

        if n > 0:
            print(f"  → {n} entries this scan, {total_entered} total today")
        else:
            print(f"  → no entries (waiting for confirmation)")

        # Check if all slots filled
        positions = get_positions()
        strategy  = get_strategy(regime)
        params    = strategy.get_params()
        max_longs = getattr(params, 'max_positions', getattr(params, 'max_positions_long', 4))
        longs = sum(1 for p in positions.values() if p['side'] == 'long')
        if longs >= max_longs:
            print(f"\nAll {max_longs} long slots filled — entry loop done")
            break

        time.sleep(ENTRY_LOOP_INTERVAL_SEC)

    print(f"\n[entries] Done — {total_entered} total entries today")

    # Print summary
    positions = get_positions()
    account   = get_account()
    print(f"\n{'='*60}")
    print(f"Portfolio: ${account['portfolio']:,.0f}  Cash: ${account['cash']:,.0f}")
    for sym, p in positions.items():
        print(f"  {sym:<6}: {'L' if p['side']=='long' else 'S'}  "
              f"qty={abs(p['qty'])}  PnL=${p['pnl']:+,.0f} ({p['pnl_pct']:+.1%})")

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exits',   action='store_true', help='Run exit manager only (9:25am)')
    parser.add_argument('--entries', action='store_true', help='Run entry loop only (10:00am-3:30pm)')
    args = parser.parse_args()

    if args.exits:
        run_exits()
    elif args.entries:
        run_entries()
    else:
        # Legacy mode — run exits then entries
        run_exits()
        run_entries()
