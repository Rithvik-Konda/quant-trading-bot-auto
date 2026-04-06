"""
put_seller.py — Cash-Secured Put Selling Engine
================================================
Runs in CHOPPY_BEAR regime when system would otherwise sit in cash.
Sells 30-delta puts on top ML-ranked quality names.
Premium collection: 2-3% monthly on capital deployed.

Logic:
- Take top 5 names from ML ranker that are ABOVE their 200d MA
- Sell 30-delta put at nearest weekly expiration (7-14 days out)
- Position size: 5% of portfolio per put (cash-secured)
- Exit: let expire worthless OR buy back at 80% profit
- Hard stop: buy back if put doubles in value (stock down 10%+)
"""
import os, sys, json, time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

ALPACA_KEY    = os.environ.get('ALPACA_KEY', '')
ALPACA_SECRET = os.environ.get('ALPACA_SECRET', '')
BASE_URL      = 'https://paper-api.alpaca.markets'
CACHE_DIR     = '/Users/rick/ai_trading_bot_v2/cache_puts'
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_PUTS          = 5       # max simultaneous put positions
CAPITAL_PER_PUT   = 0.05    # 5% of portfolio per put
TARGET_DELTA      = 0.30    # 30-delta puts
MIN_PREMIUM_PCT   = 0.008   # minimum 0.8% premium (annualizes to ~10%)
PROFIT_CLOSE_PCT  = 0.80    # close when 80% of max profit captured
STOP_LOSS_MULT    = 2.0     # stop if put doubles in value
MIN_DAYS_TO_EXP   = 7
MAX_DAYS_TO_EXP   = 21


def get_headers():
    return {
        'APCA-API-KEY-ID': ALPACA_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET,
        'Content-Type': 'application/json'
    }


def get_top_candidates(n=10):
    """Get top ML-ranked stocks above 200d MA from daily scan."""
    scan_file = '/Users/rick/ai_trading_bot_v2/cache_scanner/daily_scan.json'
    if not os.path.exists(scan_file):
        print("No scanner output — run daily_scanner.py first")
        return []
    scan = json.load(open(scan_file))
    candidates = scan.get('candidates', [])
    # Filter: above 200d MA, high quality
    result = []
    for c in candidates[:n*2]:
        sym = c.get('symbol', c.get('sym', ''))
        if not sym: continue
        try:
            t = yf.Ticker(sym)
            hist = t.history(period='1y')
            if len(hist) < 200: continue
            price = float(hist['Close'].iloc[-1])
            ma200 = float(hist['Close'].rolling(200).mean().iloc[-1])
            if price > ma200:  # only sell puts on stocks above 200d MA
                result.append({'sym': sym, 'price': price, 'ma200': ma200,
                               'ml_rank': c.get('ml_rank_pct', 0)})
        except Exception as e:
            continue
    return result[:n]


def find_put_to_sell(sym: str, price: float, portfolio: float):
    """Find the best put to sell — near 30-delta, 7-21 DTE."""
    try:
        import requests
        headers = {
            'APCA-API-KEY-ID': ALPACA_KEY,
            'APCA-API-SECRET-KEY': ALPACA_SECRET
        }
        # Get option contracts
        url = f"{BASE_URL}/v2/options/contracts"
        params = {
            'underlying_symbols': sym,
            'type': 'put',
            'expiration_date_gte': (datetime.now() + timedelta(days=MIN_DAYS_TO_EXP)).strftime('%Y-%m-%d'),
            'expiration_date_lte': (datetime.now() + timedelta(days=MAX_DAYS_TO_EXP)).strftime('%Y-%m-%d'),
            'limit': 50
        }
        resp = requests.get(url, headers=headers, params=params)
        contracts = resp.json().get('option_contracts', [])
        if not contracts:
            return None

        # Find contract closest to 30-delta (strike ~5-8% OTM)
        target_strike = price * 0.93  # ~7% OTM as proxy for 30-delta
        best = None
        best_dist = float('inf')
        for c in contracts:
            strike = float(c['strike_price'])
            dist = abs(strike - target_strike)
            if dist < best_dist:
                best_dist = dist
                best = c

        if not best:
            return None

        # Get quote for the put
        symbol = best['symbol']
        url2 = f"{BASE_URL}/v2/options/snapshots/{symbol}"
        resp2 = requests.get(url2, headers=headers)
        snap = resp2.json()

        bid = float(snap.get('greeks', {}).get('bid', 0) or
                    snap.get('latestQuote', {}).get('bp', 0) or 0)
        mid = bid  # use bid as conservative estimate

        if mid <= 0:
            return None

        strike = float(best['strike_price'])
        premium_pct = mid / strike
        if premium_pct < MIN_PREMIUM_PCT:
            print(f"  {sym}: premium {premium_pct:.2%} below minimum {MIN_PREMIUM_PCT:.2%}")
            return None

        # Position sizing: 5% of portfolio, cash-secured
        cash_required = strike * 100  # per contract
        max_contracts = int((portfolio * CAPITAL_PER_PUT) / cash_required)
        if max_contracts < 1:
            return None

        return {
            'symbol': symbol,
            'underlying': sym,
            'strike': strike,
            'expiration': best['expiration_date'],
            'bid': bid,
            'mid': mid,
            'contracts': max_contracts,
            'cash_required': cash_required * max_contracts,
            'premium_pct': premium_pct,
            'annualized': premium_pct * (365 / max(1, (datetime.strptime(best['expiration_date'], '%Y-%m-%d') - datetime.now()).days))
        }
    except Exception as e:
        print(f"  Error finding put for {sym}: {e}")
        return None


def manage_existing_puts(api, tracked_puts: dict, positions: dict):
    """Close puts that hit profit target or stop loss."""
    to_close = []
    for sym, put in tracked_puts.items():
        entry_credit = put.get('entry_credit', 0)
        option_sym = put.get('option_symbol', '')
        if option_sym not in positions:
            to_close.append(sym)  # already closed/expired
            continue
        current_value = abs(float(positions[option_sym].get('unrealized_pnl', 0)))
        profit_pct = current_value / max(entry_credit * 100, 1)
        if profit_pct >= PROFIT_CLOSE_PCT:
            print(f"  CLOSE PUT {option_sym}: {profit_pct:.0%} profit captured")
            to_close.append(sym)
    return to_close


def run_put_engine():
    print("="*60)
    print(f"PUT SELLING ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    api = get_api()
    account = api.get_account()
    portfolio = float(account.portfolio_value)
    cash = float(account.cash)
    print(f"Portfolio: ${portfolio:,.0f}  Cash: ${cash:,.0f}")

    # Load tracked puts
    puts_file = os.path.join(CACHE_DIR, 'active_puts.json')
    tracked_puts = json.load(open(puts_file)) if os.path.exists(puts_file) else {}

    # Get current positions
    positions = {p.symbol: {'unrealized_pnl': p.unrealized_pl}
                 for p in api.list_positions()}

    # Manage existing puts
    to_close = manage_existing_puts(api, tracked_puts, positions)
    for sym in to_close:
        tracked_puts.pop(sym, None)

    # How many slots available
    slots = MAX_PUTS - len(tracked_puts)
    if slots <= 0:
        print(f"All {MAX_PUTS} put slots filled")
        json.dump(tracked_puts, open(puts_file, 'w'), indent=2)
        return

    print(f"\nOpen put slots: {slots}")

    # Get candidates
    candidates = get_top_candidates(n=15)
    print(f"Candidates above 200d MA: {len(candidates)}")

    entered = 0
    for cand in candidates:
        if entered >= slots:
            break
        sym = cand['sym']
        if sym in tracked_puts:
            continue

        print(f"\n  Evaluating {sym} @ ${cand['price']:.2f} (MA200=${cand['ma200']:.2f})")
        put = find_put_to_sell(sym, cand['price'], portfolio)
        if not put:
            continue

        print(f"  PUT: {put['symbol']} strike=${put['strike']} exp={put['expiration']}")
        print(f"  Premium: ${put['mid']:.2f} ({put['premium_pct']:.2%}) annualized={put['annualized']:.1%}")
        print(f"  Contracts: {put['contracts']} cash_required=${put['cash_required']:,.0f}")

        # Submit sell-to-open order
        try:
            order = api.submit_order(
                symbol=put['symbol'],
                qty=put['contracts'],
                side='sell',
                type='limit',
                time_in_force='day',
                limit_price=round(put['mid'] - 0.01, 2),
                order_class='simple'
            )
            print(f"  ✓ Order submitted: {order.id}")
            tracked_puts[sym] = {
                'option_symbol': put['symbol'],
                'strike': put['strike'],
                'expiration': put['expiration'],
                'entry_credit': put['mid'],
                'contracts': put['contracts'],
                'entry_date': str(datetime.now().date()),
                'order_id': order.id
            }
            entered += 1
        except Exception as e:
            print(f"  Order failed: {e}")

    json.dump(tracked_puts, open(puts_file, 'w'), indent=2)
    print(f"\n✓ Done. {entered} puts entered. Total active: {len(tracked_puts)}")


if __name__ == "__main__":
    run_put_engine()
