"""
stop_manager.py — Continuous stop management
=============================================
Runs every 5 minutes during market hours.
Checks all positions against stops and takes profit targets.
Scheduled via launchd to run 9:30am-4:00pm weekdays.
"""
import os, sys, json, requests, time
from datetime import datetime

KEY    = os.environ.get('ALPACA_KEY', 'PKKUPJE3L32EXWBQVVEHZG5O7R')
SECRET = os.environ.get('ALPACA_SECRET', 'F7wJNy6qHNfvztDpdBHhE5NT33eo5ckqUZ7b4krk1FpF')
BASE   = 'https://paper-api.alpaca.markets'
HEADERS = {'APCA-API-KEY-ID': KEY, 'APCA-API-SECRET-KEY': SECRET}

CACHE_DIR    = '/Users/rick/ai_trading_bot_v2/cache_live_v2'
STOP_PCT     = 0.08   # 8% stop
TP_LONG_PCT  = 0.25   # 25% take profit on longs
TP_SHORT_PCT = 0.08   # 8% take profit on shorts (price decline)
MAX_HOLD     = 7      # days


def get_positions():
    r = requests.get(f'{BASE}/v2/positions', headers=HEADERS)
    return r.json()


def close_position(symbol, qty, side):
    order_side = 'sell' if side == 'long' else 'buy'
    order = {
        'symbol': symbol,
        'qty': str(abs(qty)),
        'side': order_side,
        'type': 'market',
        'time_in_force': 'day'
    }
    r = requests.post(f'{BASE}/v2/orders', headers=HEADERS, json=order)
    return r.json()


def run():
    now = datetime.now()
    print(f"[{now.strftime('%H:%M')}] Stop manager running...")

    positions = get_positions()
    if not positions:
        print("  No positions")
        return

    tracked_file = os.path.join(CACHE_DIR, 'tracked_positions.json')
    tracked = json.load(open(tracked_file)) if os.path.exists(tracked_file) else {}

    for p in positions:
        sym   = p['symbol']
        qty   = int(float(p['qty']))
        entry = float(p['avg_entry_price'])
        price = float(p['current_price'])
        side  = 'long' if qty > 0 else 'short'
        pnl_pct = float(p['unrealized_plpc'])

        entry_info = tracked.get(sym, {})
        entry_date = entry_info.get('entry_date', str(now.date()))
        age_days = (now.date() - datetime.strptime(entry_date, '%Y-%m-%d').date()).days

        reason = None

        if side == 'long':
            if price < entry * (1 - STOP_PCT):
                reason = f"STOP LOSS {pnl_pct:.1%}"
            elif price > entry * (1 + TP_LONG_PCT):
                reason = f"TAKE PROFIT {pnl_pct:.1%}"
            elif age_days >= MAX_HOLD:
                reason = f"MAX HOLD {age_days}d"

        elif side == 'short':
            if price > entry * (1 + STOP_PCT):
                reason = f"STOP LOSS {pnl_pct:.1%}"
            elif price < entry * (1 - TP_SHORT_PCT):
                reason = f"TAKE PROFIT {pnl_pct:.1%}"
            elif age_days >= MAX_HOLD:
                reason = f"MAX HOLD {age_days}d"

        if reason:
            print(f"  CLOSING {sym} ({side}): {reason}  price=${price:.2f}  entry=${entry:.2f}")
            result = close_position(sym, qty, side)
            print(f"    → {result.get('status', 'ERROR')}")
            tracked.pop(sym, None)
        else:
            print(f"  HOLD {sym} ({side}): price=${price:.2f}  PnL={pnl_pct:+.1%}  age={age_days}d")

    json.dump(tracked, open(tracked_file, 'w'), indent=2)


if __name__ == "__main__":
    run()
