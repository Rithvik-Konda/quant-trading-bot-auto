"""
intraday_monitor.py — 30-Minute Position Monitor During Market Hours
====================================================================
Standalone script. Checks open positions against stop levels and
profit targets. Submits exit orders when thresholds are breached.

Schedule via launchd (every 30 minutes, 9:30am-4:00pm ET weekdays):

    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
      "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
      <key>Label</key>
      <string>com.tradingbot.intraday_monitor</string>
      <key>ProgramArguments</key>
      <array>
        <string>/usr/bin/python3</string>
        <string>/Users/rick/ai_trading_bot_v2/v2/intraday_monitor.py</string>
      </array>
      <key>StartInterval</key>
      <integer>1800</integer>
      <key>StandardOutPath</key>
      <string>/Users/rick/ai_trading_bot_v2/logs/intraday_monitor_stdout.log</string>
      <key>StandardErrorPath</key>
      <string>/Users/rick/ai_trading_bot_v2/logs/intraday_monitor_stderr.log</string>
    </dict>
    </plist>

    Save to: ~/Library/LaunchAgents/com.tradingbot.intraday_monitor.plist
    Load:    launchctl load ~/Library/LaunchAgents/com.tradingbot.intraday_monitor.plist

Usage:
    python v2/intraday_monitor.py          # run one check
    python v2/intraday_monitor.py --force  # ignore market hours check
"""
from __future__ import annotations

import os
import sys
import json
import logging
import requests
from datetime import datetime, time as dt_time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))

ALPACA_KEY    = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
TRADE_URL     = "https://paper-api.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
    "Content-Type":        "application/json",
}

TRACKED_FILE = os.path.expanduser("~/ai_trading_bot_v2/cache_live_v2/tracked_positions.json")
LOG_DIR      = Path(os.path.expanduser("~/ai_trading_bot_v2/logs"))
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "intraday_monitor.log"),
    ],
)
log = logging.getLogger("intraday_monitor")

# Short position thresholds
SHORT_TAKE_PROFIT_PCT = 0.15
SHORT_STOP_LOSS_PCT   = 0.08


def is_market_hours() -> bool:
    """True if US market is open (9:30-16:00 ET, weekdays)."""
    try:
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        import pytz
        et = datetime.now(pytz.timezone("America/New_York"))
    if et.weekday() >= 5:
        return False
    return dt_time(9, 30) <= et.time() <= dt_time(16, 0)


def get_positions() -> list:
    """Get all open positions from Alpaca."""
    try:
        resp = requests.get(f"{TRADE_URL}/v2/positions", headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            log.error(f"Failed to get positions: {resp.status_code} {resp.text[:200]}")
            return []
        return resp.json()
    except Exception as e:
        log.error(f"Exception getting positions: {e}")
        return []


def get_current_price(symbol: str) -> float:
    """Get latest quote from Alpaca."""
    try:
        resp = requests.get(
            f"{TRADE_URL}/v2/stocks/{symbol}/quotes/latest",
            headers=HEADERS, timeout=10,
        )
        q = resp.json().get("quote", {})
        return float(q.get("ap", 0) or q.get("bp", 0) or 0)
    except Exception:
        return 0.0


def submit_order(symbol: str, qty: int, side: str) -> bool:
    """Submit market order. Returns True on success."""
    order = {
        "symbol":        symbol,
        "qty":           str(abs(qty)),
        "side":          side,
        "type":          "market",
        "time_in_force": "day",
    }
    try:
        resp = requests.post(f"{TRADE_URL}/v2/orders", headers=HEADERS, json=order, timeout=10)
        result = resp.json()
        if "id" in result:
            log.info(f"  Order submitted: {side.upper()} {symbol} qty={abs(qty)} -> {result.get('status','')}")
            return True
        else:
            log.error(f"  Order failed: {side.upper()} {symbol}: {result.get('message', resp.text[:100])}")
            return False
    except Exception as e:
        log.error(f"  Order exception: {side.upper()} {symbol}: {e}")
        return False


def load_tracked() -> dict:
    """Load tracked positions with entry metadata."""
    if not os.path.exists(TRACKED_FILE):
        return {}
    try:
        with open(TRACKED_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_tracked(tracked: dict) -> None:
    """Save tracked positions."""
    os.makedirs(os.path.dirname(TRACKED_FILE), exist_ok=True)
    with open(TRACKED_FILE, "w") as f:
        json.dump(tracked, f, indent=2)


def run_monitor():
    """Main monitoring loop — one pass."""
    log.info("=" * 50)
    log.info("INTRADAY MONITOR CHECK")
    log.info("=" * 50)

    positions = get_positions()
    tracked = load_tracked()
    modified = False

    if not positions:
        log.info("No open positions.")
        return

    log.info(f"Open positions: {len(positions)}")

    for pos in positions:
        if not isinstance(pos, dict):
            continue

        sym   = pos.get("symbol", "")
        side  = pos.get("side", "")
        qty   = abs(float(pos.get("qty", 0)))
        entry = float(pos.get("avg_entry_price", 0))

        if not sym or qty == 0 or entry == 0:
            continue

        price = get_current_price(sym)
        if price <= 0:
            log.warning(f"  {sym}: could not get price, skipping")
            continue

        # ── Long momentum positions ──────────────────────────────────────
        if side == "long":
            info = tracked.get(sym, {})
            engine = info.get("engine", "momentum")
            stop_pct = info.get("stop_pct", 0.08)

            unrealized_pct = (price - entry) / entry
            stop_price = entry * (1 - stop_pct)

            log.info(f"  {sym} LONG: entry=${entry:.2f} now=${price:.2f} "
                     f"P&L={unrealized_pct:+.1%} stop=${stop_price:.2f} ({engine})")

            if price < stop_price:
                log.warning(f"  STOP HIT {sym} at ${price:.2f} "
                            f"(stop=${stop_price:.2f}, loss={unrealized_pct:+.1%})")
                if submit_order(sym, int(qty), "sell"):
                    tracked.pop(sym, None)
                    modified = True

        # ── Short positions ──────────────────────────────────────────────
        elif side == "short":
            unrealized_pct = (entry - price) / entry  # positive = profit for shorts

            log.info(f"  {sym} SHORT: entry=${entry:.2f} now=${price:.2f} "
                     f"P&L={unrealized_pct:+.1%}")

            if unrealized_pct >= SHORT_TAKE_PROFIT_PCT:
                log.warning(f"  SHORT TAKE PROFIT {sym} at ${price:.2f} "
                            f"(gain={unrealized_pct:+.1%})")
                if submit_order(sym, int(qty), "buy"):
                    tracked.pop(sym, None)
                    modified = True

            elif unrealized_pct <= -SHORT_STOP_LOSS_PCT:
                log.warning(f"  SHORT STOP HIT {sym} at ${price:.2f} "
                            f"(loss={unrealized_pct:+.1%})")
                if submit_order(sym, int(qty), "buy"):
                    tracked.pop(sym, None)
                    modified = True

    if modified:
        save_tracked(tracked)
        log.info("Tracked positions updated.")

    log.info("Monitor check complete.\n")


if __name__ == "__main__":
    force = "--force" in sys.argv

    if not force and not is_market_hours():
        print("Market closed.")
        sys.exit(0)

    if not ALPACA_KEY or not ALPACA_SECRET:
        log.error("Missing ALPACA_KEY or ALPACA_SECRET environment variables.")
        sys.exit(1)

    run_monitor()
