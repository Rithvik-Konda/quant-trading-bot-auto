"""
realtime_engine.py -- Real-Time Event Ingestion & Alert Engine
==============================================================
Monitors volume spikes, news sentiment, VIX levels, and options flow
during market hours.  Alerts only -- no order placement.

Run:
    python v2/realtime_engine.py              # normal loop (market hours)
    python v2/realtime_engine.py --test       # single cycle then exit
    python v2/realtime_engine.py --dry-run    # no file writes
"""
from __future__ import annotations

import sys, os, time, json, logging, argparse, traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))
sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2/v2"))

import pandas as pd
import numpy as np
import requests

import config

# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

ROOT = Path(os.path.expanduser("~/ai_trading_bot_v2"))
CACHE_PRICES  = ROOT / "cache_prices"
CACHE_SCANNER = ROOT / "cache_scanner"
CACHE_ALPACA  = ROOT / "cache_alpaca"
LOG_DIR       = ROOT / "logs"

CACHE_SCANNER.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

ALERTS_FILE        = CACHE_SCANNER / "realtime_alerts.json"
SIGNAL_ENGINE_FILE = CACHE_SCANNER / "signal_engine_latest.json"
MAX_ALERTS_KEPT    = 500

ALPACA_BASE = "https://paper-api.alpaca.markets"
ALPACA_DATA = "https://data.alpaca.markets"

# Sentiment keywords
BULLISH_WORDS = {"beat", "surge", "surges", "surging", "upgrade", "upgraded",
                 "record", "breakout", "rally", "raises", "exceeds", "soars",
                 "outperform", "buy", "bullish", "profit", "growth", "strong",
                 "record-high", "blowout", "accelerate", "boost"}
BEARISH_WORDS = {"miss", "misses", "downgrade", "downgraded", "cut", "cuts",
                 "layoff", "layoffs", "recall", "warning", "warns", "weak",
                 "disappoints", "loss", "decline", "declines", "sell",
                 "bearish", "underperform", "slash", "slashes", "fraud",
                 "investigation", "lawsuit", "bankruptcy", "default"}

# VIX alert thresholds (ascending)
VIX_THRESHOLDS = [25, 30, 35]

# Volume spike multiplier
VOLUME_SPIKE_MULT = 3.0

# Options flow: volume vs OI multiplier
OPTIONS_FLOW_MULT = 3.0

# Polling intervals (seconds)
POLL_VIX     = 30
POLL_VOLUME  = 60
POLL_OPTIONS = 120
POLL_NEWS    = 120

# ═════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "realtime_engine.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("realtime_engine")

# ═════════════════════════════════════════════════════════════════════════════
#  ALERT DATA MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Alert:
    timestamp: str
    alert_type: str
    symbol: str
    score: float
    message: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ═════════════════════════════════════════════════════════════════════════════
#  COLOR HELPERS (terminal output)
# ═════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "VOLUME_SPIKE": "\033[93m",   # yellow
    "NEWS_ALERT":   "\033[96m",   # cyan
    "VIX_ALERT":    "\033[91m",   # red
    "OPTIONS_FLOW": "\033[95m",   # magenta
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"


def _color_print(alert: Alert) -> None:
    color = _COLORS.get(alert.alert_type, "")
    ts = alert.timestamp[:19]
    print(f"{_DIM}{ts}{_RESET}  {color}{_BOLD}[{alert.alert_type}]{_RESET}  {alert.message}")


# ═════════════════════════════════════════════════════════════════════════════
#  MARKET HOURS CHECK
# ═════════════════════════════════════════════════════════════════════════════

def _now_et() -> datetime:
    """Current time in US/Eastern."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def is_market_hours() -> bool:
    """True if 9:30-16:00 ET on a weekday."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


# ═════════════════════════════════════════════════════════════════════════════
#  API SESSION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _alpaca_session() -> Optional[requests.Session]:
    key = os.environ.get("ALPACA_KEY", "")
    secret = os.environ.get("ALPACA_SECRET", "")
    if not key or not secret:
        log.warning("ALPACA_KEY / ALPACA_SECRET not set -- Alpaca monitors disabled")
        return None
    sess = requests.Session()
    sess.headers.update({
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    })
    return sess


def _polygon_key() -> Optional[str]:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        log.warning("POLYGON_API_KEY not set -- news monitor disabled")
        return None
    return key


# ═════════════════════════════════════════════════════════════════════════════
#  REALTIME ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class RealtimeEngine:
    def __init__(self, symbols: List[str], *, dry_run: bool = False):
        self.symbols: List[str] = symbols
        self.symbol_set: Set[str] = set(symbols)
        self.dry_run = dry_run

        # state
        self.alerts: List[dict] = []
        self.vix_level: Optional[float] = None
        self.vix_crossed: Set[int] = set()          # thresholds already alerted
        self.position_scalar: float = 1.0
        self.avg_volumes: Dict[str, float] = {}     # sym -> 20d avg daily vol
        self.ml_top50: List[str] = []

        # timers (epoch seconds of last poll)
        self._last_vix: float = 0.0
        self._last_volume: float = 0.0
        self._last_news: float = 0.0
        self._last_options: float = 0.0

        # sessions
        self._alpaca: Optional[requests.Session] = _alpaca_session()
        self._polygon_key: Optional[str] = _polygon_key()

    # ─────────────────────────────────────────────────────────────────────
    #  PRECOMPUTE
    # ─────────────────────────────────────────────────────────────────────

    def precompute_avg_volumes(self) -> None:
        """Load 20-day average daily volume from cached price CSVs.

        The cache files contain daily bars.  We convert daily volume
        to an approximate per-5-minute volume by dividing by 78
        (6.5 hours * 12 bars/hour).
        """
        BARS_PER_DAY = 78  # 6.5 hrs * 12 five-min bars
        loaded = 0
        for sym in self.symbols:
            path = CACHE_PRICES / f"{sym}_3650d.csv"
            if not path.exists():
                path = CACHE_PRICES / f"{sym}_1500d.csv"
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, usecols=["volume"])
                if len(df) < 20:
                    continue
                avg_daily = df["volume"].iloc[-20:].mean()
                self.avg_volumes[sym] = avg_daily / BARS_PER_DAY
                loaded += 1
            except Exception:
                continue
        log.info(f"Precomputed avg 5-min volumes for {loaded}/{len(self.symbols)} symbols")

    def load_ml_top50(self) -> None:
        """Load top-50 ML-ranked names for options flow monitoring."""
        path = CACHE_ALPACA / "today_ml_ranks.json"
        if not path.exists():
            log.warning("today_ml_ranks.json not found -- options flow disabled")
            return
        try:
            with open(path) as f:
                ranks = json.load(f)
            ranks.sort(key=lambda r: r.get("ml_rank_pct", 0), reverse=True)
            self.ml_top50 = [r["symbol"] for r in ranks[:50]]
            log.info(f"Loaded {len(self.ml_top50)} ML top-ranked symbols for options flow")
        except Exception as e:
            log.warning(f"Failed to load ML ranks: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  ALERT MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────

    def _fire_alert(self, alert: Alert) -> None:
        """Record alert, print to terminal, persist to disk."""
        d = alert.to_dict()
        self.alerts.append(d)

        # terminal
        _color_print(alert)

        if self.dry_run:
            return

        # append to realtime_alerts.json (keep last MAX_ALERTS_KEPT)
        try:
            existing: List[dict] = []
            if ALERTS_FILE.exists():
                with open(ALERTS_FILE) as f:
                    existing = json.load(f)
            existing.append(d)
            existing = existing[-MAX_ALERTS_KEPT:]
            with open(ALERTS_FILE, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            log.error(f"Failed to write alerts file: {e}")

        # merge into signal_engine_latest.json
        try:
            sig_data: dict = {}
            if SIGNAL_ENGINE_FILE.exists():
                with open(SIGNAL_ENGINE_FILE) as f:
                    sig_data = json.load(f)
            rt_key = "realtime_alerts"
            if rt_key not in sig_data:
                sig_data[rt_key] = []
            sig_data[rt_key].append(d)
            sig_data[rt_key] = sig_data[rt_key][-100:]
            sig_data["realtime_position_scalar"] = self.position_scalar
            sig_data["realtime_vix"] = self.vix_level
            sig_data["realtime_updated"] = _now_et().isoformat()
            with open(SIGNAL_ENGINE_FILE, "w") as f:
                json.dump(sig_data, f, indent=2)
        except Exception as e:
            log.error(f"Failed to update signal engine file: {e}")

    def _make_alert(
        self,
        alert_type: str,
        symbol: str,
        score: float,
        message: str,
        **meta,
    ) -> Alert:
        return Alert(
            timestamp=_now_et().isoformat(),
            alert_type=alert_type,
            symbol=symbol,
            score=score,
            message=message,
            meta=meta,
        )

    # ─────────────────────────────────────────────────────────────────────
    #  MONITOR 1: VOLUME SPIKES
    # ─────────────────────────────────────────────────────────────────────

    def poll_volume(self) -> None:
        if not self._alpaca:
            return
        if not self.avg_volumes:
            return

        # Batch symbols in groups of 50 for the bars endpoint
        batch_size = 50
        sym_list = [s for s in self.symbols if s in self.avg_volumes]

        for i in range(0, len(sym_list), batch_size):
            batch = sym_list[i : i + batch_size]
            symbols_param = ",".join(batch)
            try:
                resp = self._alpaca.get(
                    f"{ALPACA_DATA}/v2/stocks/bars",
                    params={
                        "symbols": symbols_param,
                        "timeframe": "5Min",
                        "limit": 1,
                    },
                    timeout=15,
                )
                if resp.status_code != 200:
                    log.warning(f"Volume poll HTTP {resp.status_code}")
                    continue
                data = resp.json()
                bars = data.get("bars", {})
            except Exception as e:
                log.warning(f"Volume poll error: {e}")
                continue

            for sym, bar_list in bars.items():
                if not bar_list:
                    continue
                bar = bar_list[-1] if isinstance(bar_list, list) else bar_list
                vol = bar.get("v", 0)
                price = bar.get("c", 0)
                open_price = bar.get("o", 0)

                avg_5m = self.avg_volumes.get(sym, 0)
                if avg_5m <= 0:
                    continue

                ratio = vol / avg_5m
                if ratio >= VOLUME_SPIKE_MULT:
                    pct_change = ((price - open_price) / open_price * 100) if open_price else 0
                    msg = (
                        f"VOLUME_SPIKE: {sym} {ratio:.1f}x avg, "
                        f"price={price:.2f}, change={pct_change:+.1f}%"
                    )
                    alert = self._make_alert(
                        "VOLUME_SPIKE", sym, ratio,
                        msg,
                        volume=vol, avg_5min=avg_5m, price=price,
                        pct_change=round(pct_change, 2),
                    )
                    self._fire_alert(alert)

    # ─────────────────────────────────────────────────────────────────────
    #  MONITOR 2: NEWS SENTIMENT
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_headline(headline: str) -> float:
        """Simple keyword sentiment.  Returns score in [-1, 1]."""
        words = set(headline.lower().split())
        bull = len(words & BULLISH_WORDS)
        bear = len(words & BEARISH_WORDS)
        total = bull + bear
        if total == 0:
            return 0.0
        return (bull - bear) / total

    def poll_news(self) -> None:
        if not self._polygon_key:
            return
        try:
            resp = requests.get(
                "https://api.polygon.io/v2/reference/news",
                params={"limit": 50, "apiKey": self._polygon_key},
                timeout=15,
            )
            if resp.status_code != 200:
                log.warning(f"News poll HTTP {resp.status_code}")
                return
            articles = resp.json().get("results", [])
        except Exception as e:
            log.warning(f"News poll error: {e}")
            return

        for article in articles:
            tickers = article.get("tickers", [])
            headline = article.get("title", "")
            if not headline:
                continue

            matched = [t for t in tickers if t in self.symbol_set]
            if not matched:
                continue

            score = self._score_headline(headline)
            if abs(score) < 0.5:
                continue

            for sym in matched:
                msg = f'NEWS_ALERT: {sym} sentiment={score:+.2f} "{headline[:120]}"'
                alert = self._make_alert(
                    "NEWS_ALERT", sym, score,
                    msg,
                    headline=headline,
                    source=article.get("publisher", {}).get("name", ""),
                    article_url=article.get("article_url", ""),
                )
                self._fire_alert(alert)

    # ─────────────────────────────────────────────────────────────────────
    #  MONITOR 3: VIX LEVEL
    # ─────────────────────────────────────────────────────────────────────

    def poll_vix(self) -> None:
        """Poll VIX via Alpaca latest bar for VIXY as proxy, or cached CSV."""
        vix_val: Optional[float] = None

        # Strategy 1: try Alpaca snapshot for a VIX proxy ETF (VIXY)
        if self._alpaca:
            try:
                resp = self._alpaca.get(
                    f"{ALPACA_DATA}/v2/stocks/bars",
                    params={"symbols": "VIXY", "timeframe": "1Min", "limit": 1},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json().get("bars", {})
                    vixy_bars = data.get("VIXY", [])
                    if vixy_bars:
                        bar = vixy_bars[-1] if isinstance(vixy_bars, list) else vixy_bars
                        # VIXY is NOT VIX -- fall through to CSV
            except Exception:
                pass

        # Strategy 2: read cached VIX CSV (updated by other processes)
        vix_csv = CACHE_PRICES / "^VIX_etf.csv"
        if vix_csv.exists():
            try:
                df = pd.read_csv(vix_csv)
                if len(df) > 0:
                    vix_val = float(df["close"].iloc[-1])
            except Exception as e:
                log.warning(f"VIX CSV read error: {e}")

        if vix_val is None:
            return

        prev = self.vix_level
        self.vix_level = vix_val

        # Update position sizing scalar
        if vix_val > 35:
            self.position_scalar = 0.25
        elif vix_val > 30:
            self.position_scalar = 0.5
        else:
            self.position_scalar = 1.0

        # Check threshold crossings (only fire once per cross)
        if prev is not None:
            for level in VIX_THRESHOLDS:
                if prev < level <= vix_val and level not in self.vix_crossed:
                    self.vix_crossed.add(level)
                    msg = (
                        f"VIX_ALERT: VIX crossed {level} "
                        f"(now {vix_val:.1f}), sizing_scalar={self.position_scalar}"
                    )
                    alert = self._make_alert(
                        "VIX_ALERT", "^VIX", float(level),
                        msg,
                        vix=vix_val, threshold=level,
                        position_scalar=self.position_scalar,
                    )
                    self._fire_alert(alert)
                elif vix_val < level and level in self.vix_crossed:
                    # Reset so we can alert again if it crosses back up
                    self.vix_crossed.discard(level)

    # ─────────────────────────────────────────────────────────────────────
    #  MONITOR 4: OPTIONS FLOW
    # ─────────────────────────────────────────────────────────────────────

    def poll_options(self) -> None:
        """Check options snapshots for unusual volume vs OI on ML top-50."""
        if not self._alpaca:
            return
        if not self.ml_top50:
            return

        for sym in self.ml_top50:
            try:
                resp = self._alpaca.get(
                    f"{ALPACA_DATA}/v1beta1/options/snapshots/{sym}",
                    params={"feed": "indicative", "limit": 20},
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                snapshots = resp.json().get("snapshots", {})
            except Exception:
                continue

            for contract_id, snap in snapshots.items():
                latest_trade = snap.get("latestTrade", {})
                latest_quote = snap.get("latestQuote", {})
                greeks       = snap.get("greeks", {})

                vol = latest_trade.get("s", 0)  # size / volume
                oi  = snap.get("openInterest", 0)
                if oi <= 0 or vol <= 0:
                    continue

                ratio = vol / oi
                if ratio < OPTIONS_FLOW_MULT:
                    continue

                # Parse contract details from contract_id
                # Format: SYM   YYMMDDCNNNNN  or similar
                opt_type = "CALL" if "C" in contract_id[len(sym):] else "PUT"
                strike = latest_trade.get("p", 0)

                msg = (
                    f"OPTIONS_FLOW: {sym} {opt_type} K={strike:.0f} "
                    f"vol={vol} vs OI={oi} ({ratio:.1f}x)"
                )
                alert = self._make_alert(
                    "OPTIONS_FLOW", sym, ratio,
                    msg,
                    contract=contract_id,
                    option_type=opt_type,
                    strike=strike,
                    volume=vol,
                    open_interest=oi,
                    ratio=round(ratio, 1),
                )
                self._fire_alert(alert)

    # ─────────────────────────────────────────────────────────────────────
    #  MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────

    def run_once(self) -> List[dict]:
        """Execute one polling cycle for all monitors.  Returns new alerts."""
        start_count = len(self.alerts)
        now = time.time()

        # VIX -- every 30s
        if now - self._last_vix >= POLL_VIX:
            try:
                self.poll_vix()
            except Exception as e:
                log.error(f"VIX monitor error: {e}\n{traceback.format_exc()}")
            self._last_vix = now

        # Volume -- every 60s
        if now - self._last_volume >= POLL_VOLUME:
            try:
                self.poll_volume()
            except Exception as e:
                log.error(f"Volume monitor error: {e}\n{traceback.format_exc()}")
            self._last_volume = now

        # Options -- every 120s
        if now - self._last_options >= POLL_OPTIONS:
            try:
                self.poll_options()
            except Exception as e:
                log.error(f"Options monitor error: {e}\n{traceback.format_exc()}")
            self._last_options = now

        # News -- every 120s
        if now - self._last_news >= POLL_NEWS:
            try:
                self.poll_news()
            except Exception as e:
                log.error(f"News monitor error: {e}\n{traceback.format_exc()}")
            self._last_news = now

        return self.alerts[start_count:]

    def run(self) -> None:
        """Main loop -- polls during market hours, sleeps otherwise."""
        log.info("=" * 60)
        log.info("Realtime Engine starting")
        log.info(f"  Symbols:  {len(self.symbols)}")
        log.info(f"  Avg vols: {len(self.avg_volumes)}")
        log.info(f"  ML top50: {len(self.ml_top50)}")
        log.info(f"  Alpaca:   {'OK' if self._alpaca else 'DISABLED'}")
        log.info(f"  Polygon:  {'OK' if self._polygon_key else 'DISABLED'}")
        log.info(f"  Dry-run:  {self.dry_run}")
        log.info("=" * 60)

        while True:
            if not is_market_hours():
                next_check = 60
                now = _now_et()
                if now.hour < 9 or (now.hour == 9 and now.minute < 25):
                    # Before market open -- sleep until 9:25
                    target = now.replace(hour=9, minute=25, second=0, microsecond=0)
                    next_check = max(30, (target - now).total_seconds())
                elif now.hour >= 16:
                    log.info("Market closed for today.  Exiting.")
                    break
                elif now.weekday() >= 5:
                    log.info("Weekend.  Exiting.")
                    break
                log.info(f"Outside market hours.  Sleeping {next_check:.0f}s ...")
                time.sleep(min(next_check, 300))
                continue

            new_alerts = self.run_once()
            if new_alerts:
                log.info(f"Cycle produced {len(new_alerts)} alert(s)")

            # Sleep the minimum polling interval before next cycle
            time.sleep(POLL_VIX)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime alert engine for trading bot")
    parser.add_argument("--test", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to files")
    args = parser.parse_args()

    symbols = config.WATCHLIST
    log.info(f"Universe: {len(symbols)} symbols")

    engine = RealtimeEngine(symbols, dry_run=args.dry_run)

    # Precompute
    log.info("Precomputing average volumes ...")
    engine.precompute_avg_volumes()
    engine.load_ml_top50()

    if args.test:
        log.info("--- TEST MODE: running single cycle ---")
        alerts = engine.run_once()
        log.info(f"Test cycle complete.  {len(alerts)} alert(s) fired.")
        if alerts:
            for a in alerts:
                print(json.dumps(a, indent=2))
        return

    engine.run()


if __name__ == "__main__":
    main()
