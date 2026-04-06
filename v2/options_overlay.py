"""
options_overlay.py — 30-DTE Call Overlay on TRENDING_BULL Momentum Entries
==========================================================================
Standalone module. Not wired into live_trader_v2.py yet.

Usage:
  python v2/options_overlay.py           # print open calls status
  python v2/options_overlay.py --help    # show help
"""
from __future__ import annotations

import os
import sys
import json
import time
import math
import requests
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))
sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2/v2"))

THETADATA_BASE = "http://127.0.0.1:25503"
OPEN_CALLS_PATH = os.path.expanduser("~/ai_trading_bot_v2/cache_live_v2/open_calls.json")


# ═══════════════════════════════════════════════════════════════════════════════
#  1. EXPIRATION SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_nearest_expiration(entry_date: str, target_dte: int = 30) -> str:
    """
    Returns nearest monthly expiration ~30 DTE from entry_date.
    Monthly expirations are 3rd Friday of each month.

    Args:
        entry_date: 'YYYY-MM-DD' or 'YYYYMMDD'
        target_dte: target days to expiration (default 30)

    Returns:
        'YYYYMMDD' string of the nearest monthly expiration
    """
    if len(entry_date) == 8:
        dt = datetime.strptime(entry_date, "%Y%m%d").date()
    else:
        dt = datetime.strptime(entry_date, "%Y-%m-%d").date()

    target = dt + timedelta(days=target_dte)

    # Find 3rd Friday of target month and adjacent months
    candidates = []
    for month_offset in range(-1, 3):
        y = target.year + (target.month + month_offset - 1) // 12
        m = (target.month + month_offset - 1) % 12 + 1
        third_friday = _third_friday(y, m)
        if third_friday > dt:  # must be in the future
            candidates.append(third_friday)

    if not candidates:
        # Fallback: 30 days out
        fallback = dt + timedelta(days=target_dte)
        return fallback.strftime("%Y%m%d")

    # Pick the one closest to target DTE
    best = min(candidates, key=lambda d: abs((d - dt).days - target_dte))
    return best.strftime("%Y%m%d")


def _third_friday(year: int, month: int) -> date:
    """Returns the 3rd Friday of the given month."""
    # First day of month
    first = date(year, month, 1)
    # Find first Friday: weekday() 4 = Friday
    days_until_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_until_friday)
    # Third Friday = first Friday + 14 days
    return first_friday + timedelta(days=14)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. STRIKE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_nearest_strike(stock_price: float) -> str:
    """
    Round to nearest standard strike.
    >$50: round to nearest $5
    <=$50: round to nearest $1

    Returns string with 3 decimal places e.g. "480.000"
    """
    if stock_price <= 0:
        return "0.000"
    if stock_price > 50:
        strike = round(stock_price / 5) * 5
    else:
        strike = round(stock_price)
    return f"{strike:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
#  3. THETADATA OPTION PRICING
# ═══════════════════════════════════════════════════════════════════════════════

def get_option_price(
    symbol: str,
    expiration: str,
    strike: str,
    right: str = "call",
    query_date: Optional[str] = None,
) -> Optional[float]:
    """
    Query ThetaData REST API for option EOD price.

    Args:
        symbol: underlying symbol e.g. 'NVDA'
        expiration: 'YYYYMMDD'
        strike: string with 3 decimals e.g. '480.000'
        right: 'call' or 'put'
        query_date: 'YYYYMMDD' date to query price for.
                    Defaults to expiration - 1 day.

    Returns:
        Mid price (bid+ask)/2 or None if no data.
    """
    if query_date is None:
        # Day before expiry
        exp_dt = datetime.strptime(expiration, "%Y%m%d").date()
        qd = exp_dt - timedelta(days=1)
        # Skip weekends
        while qd.weekday() >= 5:
            qd -= timedelta(days=1)
        query_date = qd.strftime("%Y%m%d")

    url = f"{THETADATA_BASE}/v3/option/history/eod"
    params = {
        "root": symbol,
        "exp": expiration,
        "strike": strike,
        "right": right[0].upper(),  # 'C' or 'P'
        "start_date": query_date,
        "end_date": query_date,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        time.sleep(3)  # rate limit

        if resp.status_code != 200:
            return None

        data = resp.json()
        rows = data.get("response", [])
        if not rows:
            return None

        # EOD format: look for bid/ask or close
        row = rows[-1] if isinstance(rows, list) else rows
        if isinstance(row, dict):
            bid = row.get("bid", 0) or 0
            ask = row.get("ask", 0) or 0
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            close = row.get("close", 0) or 0
            if close > 0:
                return float(close)

        return None

    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  4. POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════════

def size_call_position(
    portfolio_value: float,
    option_price: float,
    max_premium_pct: float = 0.01,
) -> int:
    """
    Size call position so total premium <= max_premium_pct of portfolio.

    Args:
        portfolio_value: total portfolio value in dollars
        option_price: per-share option price (multiply by 100 for per-contract)
        max_premium_pct: max fraction of portfolio to spend on premium

    Returns:
        Number of contracts (1-5), or 0 if option_price invalid.
    """
    if option_price is None or option_price <= 0 or portfolio_value <= 0:
        return 0

    max_premium = portfolio_value * max_premium_pct
    cost_per_contract = option_price * 100
    if cost_per_contract <= 0:
        return 0

    contracts = int(max_premium / cost_per_contract)
    contracts = max(contracts, 1)  # minimum 1
    contracts = min(contracts, 5)  # maximum 5

    # Final check: don't exceed budget
    if contracts * cost_per_contract > max_premium * 1.5:
        contracts = max(1, int(max_premium / cost_per_contract))

    return min(contracts, 5)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. ENTRY FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def should_buy_call(regime: str, ml_rank: float, stock_price: float) -> bool:
    """
    Returns True only if all conditions met:
    - regime == 'TRENDING_BULL'
    - ml_rank >= 0.85
    - stock_price >= 20 (liquid enough for options)
    """
    if regime != "TRENDING_BULL":
        return False
    if ml_rank < 0.85:
        return False
    if stock_price < 20:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  6-7. PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def load_open_calls(path: str = OPEN_CALLS_PATH) -> Dict:
    """Load open call positions from JSON. Returns empty dict if not found."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_open_calls(open_calls: Dict, path: str = OPEN_CALLS_PATH) -> None:
    """Save open calls dict to JSON. Creates directory if needed."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(open_calls, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  8. ADD POSITION
# ═══════════════════════════════════════════════════════════════════════════════

def add_call_position(
    symbol: str,
    expiration: str,
    strike: str,
    contracts: int,
    entry_price: float,
    entry_date: str,
    equity_entry_price: float,
) -> None:
    """
    Add a call position to open_calls.json.

    Args:
        symbol: underlying e.g. 'NVDA'
        expiration: 'YYYYMMDD'
        strike: e.g. '480.000'
        contracts: number of contracts
        entry_price: per-share option price paid
        entry_date: 'YYYY-MM-DD'
        equity_entry_price: stock price at time of equity entry
    """
    open_calls = load_open_calls()
    open_calls[symbol] = {
        "expiration": expiration,
        "strike": strike,
        "contracts": contracts,
        "entry_price": entry_price,
        "entry_date": entry_date,
        "equity_entry_price": equity_entry_price,
    }
    save_open_calls(open_calls)


# ═══════════════════════════════════════════════════════════════════════════════
#  9. EXIT LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def check_call_exits(
    open_calls: Dict,
    current_equity_positions: Dict[str, dict],
    current_date: str,
) -> List[str]:
    """
    Check which open calls should be closed.

    Returns list of symbols to close. Reasons:
    - Equity position no longer exists (equity was sold)
    - Days to expiration <= 5 (avoid pin risk)

    Args:
        open_calls: dict from load_open_calls()
        current_equity_positions: {symbol: {side, qty, ...}} from get_positions()
        current_date: 'YYYY-MM-DD'
    """
    to_close = []

    if len(current_date) == 8:
        today = datetime.strptime(current_date, "%Y%m%d").date()
    else:
        today = datetime.strptime(current_date, "%Y-%m-%d").date()

    for sym, call in open_calls.items():
        # Equity position closed — close the call too
        if sym not in current_equity_positions:
            to_close.append(sym)
            continue

        # Equity position is short or zero — not a long anymore
        eq = current_equity_positions[sym]
        if eq.get("side") != "long" or float(eq.get("qty", 0)) <= 0:
            to_close.append(sym)
            continue

        # Days to expiration <= 5 — avoid pin risk
        try:
            exp_dt = datetime.strptime(call["expiration"], "%Y%m%d").date()
            dte = (exp_dt - today).days
            if dte <= 5:
                to_close.append(sym)
                continue
        except (ValueError, KeyError):
            to_close.append(sym)
            continue

    return to_close


# ═══════════════════════════════════════════════════════════════════════════════
#  10. MAIN — print open calls status
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Print all open calls with current status."""
    open_calls = load_open_calls()

    print("=" * 70)
    print("  OPTIONS OVERLAY — Open Calls Status")
    print("=" * 70)

    if not open_calls:
        print("  No open call positions.")
        print(f"  File: {OPEN_CALLS_PATH}")
        return

    today = date.today()
    total_premium = 0.0

    print(f"\n  {'SYM':6s} {'EXP':>10s} {'K':>8s} {'Qty':>4s} {'Entry':>7s} "
          f"{'Premium':>9s} {'DTE':>4s} {'EqEntry':>8s} {'Status'}")
    print(f"  {'-' * 68}")

    for sym, call in sorted(open_calls.items()):
        exp = call.get("expiration", "")
        strike = call.get("strike", "")
        contracts = call.get("contracts", 0)
        entry_px = call.get("entry_price", 0)
        entry_dt = call.get("entry_date", "")
        eq_entry = call.get("equity_entry_price", 0)

        premium = entry_px * contracts * 100
        total_premium += premium

        try:
            exp_dt = datetime.strptime(exp, "%Y%m%d").date()
            dte = (exp_dt - today).days
        except ValueError:
            dte = -1

        if dte < 0:
            status = "EXPIRED"
        elif dte <= 5:
            status = "CLOSE (pin risk)"
        else:
            status = "OPEN"

        exp_fmt = f"{exp[:4]}-{exp[4:6]}-{exp[6:]}" if len(exp) == 8 else exp
        print(f"  {sym:6s} {exp_fmt:>10s} {strike:>8s} {contracts:>4d} "
              f"${entry_px:>6.2f} ${premium:>8.0f} {dte:>4d}d "
              f"${eq_entry:>7.2f} {status}")

    print(f"\n  Total premium deployed: ${total_premium:,.0f}")
    print(f"  Positions: {len(open_calls)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
