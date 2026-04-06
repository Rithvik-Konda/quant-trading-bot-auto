"""
strategy_leadership_puts.py — Sector Leadership + Put Selling Engine
=====================================================================
Strategy 5: Sell cash-secured puts on stocks in leading sectors during
CHOPPY regime. Combines sector leadership scoring with ML conviction
to identify high-quality put-selling opportunities.

Live mode:
    python v2/strategy_leadership_puts.py              # full engine
    python v2/strategy_leadership_puts.py --scan-only  # signal scan only
    python v2/strategy_leadership_puts.py --backtest   # historical backtest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))
sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2/v2"))

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from config import SECTOR_ETFS, WATCHLIST
from sector_leadership import LeadershipAdapter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT_DIR   = os.path.expanduser("~/ai_trading_bot_v2")
CACHE_DIR  = os.path.join(ROOT_DIR, "cache_puts")
PRICE_DIR  = os.path.join(ROOT_DIR, "cache_prices")
SCAN_FILE  = os.path.join(ROOT_DIR, "cache_scanner", "daily_scan.json")
ML_RANKS_FILE = os.path.join(ROOT_DIR, "cache_alpaca", "today_ml_ranks.json")

POSITIONS_FILE = os.path.join(CACHE_DIR, "leadership_puts.json")
LOG_FILE       = os.path.join(CACHE_DIR, "leadership_puts_log.json")

os.makedirs(CACHE_DIR, exist_ok=True)

ALPACA_KEY    = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
BASE_URL      = "https://paper-api.alpaca.markets"

# Position sizing
CAPITAL_PER_PUT       = 0.05     # 5% of portfolio per put (cash-secured)
MAX_PUTS              = 3        # max simultaneous put positions
MAX_OPTIONS_EXPOSURE  = 0.15     # 15% total options exposure cap

# Signal detection
ML_RANK_MIN           = 0.80     # minimum ML rank percentile
ANN_VOL_MAX           = 0.55     # max annualized volatility
REGIME_REQUIRED       = "CHOPPY" # only sell puts in CHOPPY

# Options selection
MIN_DTE               = 21       # minimum days to expiration
MAX_DTE               = 30       # maximum days to expiration
TARGET_DELTA_STRIKE   = 0.93     # strike ~ 7% OTM (30-delta proxy)
MIN_PREMIUM_PCT       = 0.008    # minimum premium as % of strike (0.8%)

# Exit management
PROFIT_CLOSE_PCT      = 0.80     # close at 80% of max profit
STOP_LOSS_MULT        = 2.0      # close if put doubles in value
ML_RANK_EXIT          = 0.65     # close if ML rank drops below this
MIN_DTE_EXIT          = 3        # close if < 3 DTE remaining


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Timestamp string for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_action(action: str, details: dict) -> None:
    """Append an action to the trade log (append-only JSON lines)."""
    entry = {"ts": _ts(), "action": action, **details}
    try:
        log = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                log = json.load(f)
        log.append(entry)
        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=2, default=str)
    except Exception as e:
        print(f"  [WARN] Failed to write log: {e}")


def _get_headers() -> dict:
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
    }


def _load_prices(sym: str) -> Optional[pd.DataFrame]:
    """Load cached OHLCV from cache_prices/{SYM}_3650d.csv."""
    path = os.path.join(PRICE_DIR, f"{sym}_3650d.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns:
            return None
        return df.dropna(subset=["close"])
    except Exception:
        return None


def _load_ml_ranks() -> Dict[str, float]:
    """Load current ML ranks from cache file. Returns {symbol: ml_rank_pct}."""
    if not os.path.exists(ML_RANKS_FILE):
        return {}
    try:
        with open(ML_RANKS_FILE, "r") as f:
            data = json.load(f)
        return {
            item["symbol"]: item.get("ml_rank_pct", item.get("ml_score", 0))
            for item in data
            if "symbol" in item
        }
    except Exception:
        return {}


def _load_scan() -> dict:
    """Load daily scanner output."""
    if not os.path.exists(SCAN_FILE):
        return {}
    try:
        with open(SCAN_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_positions() -> dict:
    """Load tracked put positions."""
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_positions(positions: dict) -> None:
    """Save tracked put positions."""
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def _compute_ann_vol(df: pd.DataFrame, window: int = 60) -> float:
    """Compute annualized volatility from daily returns."""
    if len(df) < window + 1:
        return 999.0
    rets = df["close"].pct_change().dropna().iloc[-window:]
    if len(rets) < 20:
        return 999.0
    return float(rets.std() * np.sqrt(252))


def _compute_sma(df: pd.DataFrame, window: int) -> Optional[float]:
    """Compute simple moving average of close prices."""
    if len(df) < window:
        return None
    return float(df["close"].iloc[-window:].mean())


def _get_support_level(df: pd.DataFrame) -> float:
    """Find support level: lower of SMA20 and 10-day low."""
    sma20 = _compute_sma(df, 20)
    low_10d = float(df["low"].iloc[-10:].min()) if len(df) >= 10 else None
    candidates = [x for x in [sma20, low_10d] if x is not None]
    return min(candidates) if candidates else 0.0


# ---------------------------------------------------------------------------
# 1. SIGNAL DETECTION
# ---------------------------------------------------------------------------

def scan_put_candidates(
    regime: str,
    ml_ranks: dict,
    portfolio_value: float,
) -> List[dict]:
    """
    Scan for put selling candidates in leading sectors.

    Returns ranked list of candidates with conviction scores.
    Only activates in CHOPPY regime.
    """
    if regime != REGIME_REQUIRED:
        print(f"  Regime is {regime}, not {REGIME_REQUIRED} -- skipping put scan")
        return []

    # Load sector leadership state
    adapter = LeadershipAdapter(update_frequency_days=1)
    hist: Dict[str, pd.DataFrame] = {}

    # Load prices for all watchlist symbols to compute leadership
    for sym in WATCHLIST:
        df = _load_prices(sym)
        if df is not None and len(df) >= 210:
            hist[sym] = df

    # Also load SPY for regime classification
    spy_df = _load_prices("SPY")
    if spy_df is not None:
        hist["SPY"] = spy_df

    if len(hist) < 50:
        print(f"  Only {len(hist)} symbols with price data -- need at least 50")
        return []

    # Compute leadership scores
    today = pd.Timestamp.now().normalize()
    adapter.update(today, hist, force=True)
    state = adapter.get_state()
    if state is None:
        print("  Failed to compute leadership state")
        return []

    leader_sectors = state.leader_sectors
    print(f"  Leader sectors: {sorted(leader_sectors)}")

    if not leader_sectors:
        print("  No sectors in leadership -- expanding to top 3")
        leader_sectors = set(state.top_sectors[:3])

    # Get constituent stocks from leader sectors
    leader_stocks = set()
    for etf in leader_sectors:
        constituents = SECTOR_ETFS.get(etf, [])
        leader_stocks.update(constituents)

    print(f"  Stocks in leader sectors: {len(leader_stocks)}")

    # Filter and score candidates
    candidates = []
    for sym in leader_stocks:
        # ML rank filter
        ml_rank = ml_ranks.get(sym, 0)
        if ml_rank < ML_RANK_MIN:
            continue

        # Load price data
        df = _load_prices(sym)
        if df is None or len(df) < 210:
            continue

        price = float(df["close"].iloc[-1])

        # Price > SMA200 filter
        sma200 = _compute_sma(df, 200)
        if sma200 is None or price <= sma200:
            continue

        # Annualized vol filter
        ann_vol = _compute_ann_vol(df)
        if ann_vol >= ANN_VOL_MAX:
            continue

        # Compute conviction score
        leadership_mult = adapter.score_multiplier(sym)
        conviction = ml_rank * leadership_mult * (1.0 - ann_vol)

        candidates.append({
            "symbol": sym,
            "price": round(price, 2),
            "ml_rank": round(ml_rank, 4),
            "leadership_mult": round(leadership_mult, 3),
            "ann_vol": round(ann_vol, 4),
            "conviction": round(conviction, 4),
            "sma200": round(sma200, 2),
            "sector": adapter.scorer.symbol_to_sector.get(sym, "unknown"),
        })

    # Sort by conviction descending
    candidates.sort(key=lambda c: c["conviction"], reverse=True)
    print(f"  Put candidates after filtering: {len(candidates)}")
    for c in candidates[:10]:
        print(
            f"    {c['symbol']:<6} price=${c['price']:>8.2f}  "
            f"ML={c['ml_rank']:.2f}  lead={c['leadership_mult']:.2f}  "
            f"vol={c['ann_vol']:.2f}  conv={c['conviction']:.3f}"
        )

    return candidates


# ---------------------------------------------------------------------------
# 2. OPTIONS SELECTION
# ---------------------------------------------------------------------------

def select_put_contract(
    sym: str,
    price: float,
    portfolio: float,
    conviction: float = 0.7,
) -> Optional[dict]:
    """
    Query Alpaca REST for put contracts and select the best one.

    Targets ~30-delta put (strike ~ price * 0.93), 21-30 DTE.
    Prefers strikes near technical support levels.
    """
    if not ALPACA_KEY or not ALPACA_SECRET:
        print("  [WARN] Alpaca API keys not set -- cannot query options")
        return None

    if requests is None:
        print("  [WARN] requests library not available")
        return None

    headers = _get_headers()

    try:
        url = f"{BASE_URL}/v2/options/contracts"
        params = {
            "underlying_symbols": sym,
            "type": "put",
            "expiration_date_gte": (datetime.now() + timedelta(days=MIN_DTE)).strftime("%Y-%m-%d"),
            "expiration_date_lte": (datetime.now() + timedelta(days=MAX_DTE)).strftime("%Y-%m-%d"),
            "limit": 50,
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"  [WARN] Alpaca contracts API returned {resp.status_code}")
            return None

        contracts = resp.json().get("option_contracts", [])
        if not contracts:
            print(f"  No put contracts found for {sym}")
            return None

    except Exception as e:
        print(f"  [ERR] Failed to query contracts for {sym}: {e}")
        return None

    # Find target strike near 30-delta proxy and support
    target_strike = price * TARGET_DELTA_STRIKE  # ~7% OTM

    # Check for support level to prefer
    df = _load_prices(sym)
    support = _get_support_level(df) if df is not None else 0.0
    # If support is within 2% of target_strike, bias toward it
    if support > 0 and abs(support - target_strike) / target_strike < 0.02:
        target_strike = support

    best = None
    best_dist = float("inf")
    for c in contracts:
        strike = float(c["strike_price"])
        # Only consider strikes below current price (OTM puts)
        if strike >= price:
            continue
        dist = abs(strike - target_strike)
        if dist < best_dist:
            best_dist = dist
            best = c

    if not best:
        print(f"  No suitable OTM put for {sym}")
        return None

    # Get quote for the selected contract
    option_symbol = best["symbol"]
    try:
        url2 = f"{BASE_URL}/v2/options/snapshots/{option_symbol}"
        resp2 = requests.get(url2, headers=headers, timeout=10)
        if resp2.status_code != 200:
            print(f"  [WARN] Snapshot API returned {resp2.status_code} for {option_symbol}")
            return None
        snap = resp2.json()
    except Exception as e:
        print(f"  [ERR] Failed to get snapshot for {option_symbol}: {e}")
        return None

    # Extract bid price (conservative -- we sell at bid)
    bid = float(
        snap.get("latestQuote", {}).get("bp", 0)
        or snap.get("greeks", {}).get("bid", 0)
        or 0
    )
    if bid <= 0:
        print(f"  No bid available for {option_symbol}")
        return None

    strike = float(best["strike_price"])
    premium_pct = bid / strike

    if premium_pct < MIN_PREMIUM_PCT:
        print(f"  {sym}: premium {premium_pct:.2%} below minimum {MIN_PREMIUM_PCT:.2%}")
        return None

    # Position sizing: 5% of portfolio, scaled by conviction
    cash_required_per = strike * 100  # cash required per contract
    base_contracts = int((portfolio * CAPITAL_PER_PUT) / cash_required_per)
    if base_contracts < 1:
        return None

    # Scale by conviction: higher conviction -> more contracts (up to 1.3x)
    scale = min(conviction / 0.7, 1.3)
    contracts_count = max(1, int(base_contracts * scale))

    expiration = best["expiration_date"]
    dte = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).days

    return {
        "option_symbol": option_symbol,
        "underlying": sym,
        "strike": strike,
        "expiration": expiration,
        "dte": dte,
        "bid": round(bid, 2),
        "premium_pct": round(premium_pct, 4),
        "contracts": contracts_count,
        "cash_required": round(cash_required_per * contracts_count, 2),
        "conviction": round(conviction, 4),
    }


# ---------------------------------------------------------------------------
# 3. POSITION SIZING (integrated into select_put_contract above)
# ---------------------------------------------------------------------------

def _check_exposure_limits(
    tracked: dict,
    new_cash_required: float,
    portfolio_value: float,
    available_cash: float,
) -> Tuple[bool, str]:
    """
    Validate that adding a new position stays within limits.
    Returns (ok, reason).
    """
    # Max simultaneous positions
    if len(tracked) >= MAX_PUTS:
        return False, f"Already at max {MAX_PUTS} put positions"

    # Total options exposure cap
    existing_exposure = sum(
        p.get("cash_required", 0) for p in tracked.values()
    )
    new_total = existing_exposure + new_cash_required
    if new_total > portfolio_value * MAX_OPTIONS_EXPOSURE:
        return False, (
            f"Total exposure ${new_total:,.0f} exceeds "
            f"{MAX_OPTIONS_EXPOSURE:.0%} cap (${portfolio_value * MAX_OPTIONS_EXPOSURE:,.0f})"
        )

    # Never exceed available cash
    if new_cash_required > available_cash:
        return False, (
            f"Cash required ${new_cash_required:,.0f} > available ${available_cash:,.0f}"
        )

    return True, "OK"


# ---------------------------------------------------------------------------
# 4. EXIT MANAGEMENT
# ---------------------------------------------------------------------------

def manage_positions(
    tracked: dict,
    ml_ranks: dict,
) -> List[dict]:
    """
    Check every tracked position against exit rules.

    Returns list of {symbol, option_symbol, reason} for positions to close.
    """
    to_close = []

    # Load current regime from scanner
    scan = _load_scan()
    current_regime = scan.get("regime", "UNKNOWN")

    headers = _get_headers()
    has_api = bool(ALPACA_KEY and ALPACA_SECRET and requests)

    for sym, pos in tracked.items():
        option_sym = pos.get("option_symbol", "")
        entry_credit = pos.get("entry_credit", 0)
        expiration = pos.get("expiration", "")

        # --- Rule 1: Regime shift to BEAR ---
        if current_regime == "BEAR":
            to_close.append({
                "symbol": sym,
                "option_symbol": option_sym,
                "reason": f"Regime shifted to {current_regime}",
            })
            continue

        # --- Rule 2: ML rank dropped below threshold ---
        ml_rank = ml_ranks.get(sym, 1.0)
        if ml_rank < ML_RANK_EXIT:
            to_close.append({
                "symbol": sym,
                "option_symbol": option_sym,
                "reason": f"ML rank dropped to {ml_rank:.2f} (< {ML_RANK_EXIT})",
            })
            continue

        # --- Rule 3: DTE check ---
        if expiration:
            try:
                exp_date = datetime.strptime(expiration, "%Y-%m-%d")
                dte = (exp_date - datetime.now()).days
                if dte < MIN_DTE_EXIT:
                    to_close.append({
                        "symbol": sym,
                        "option_symbol": option_sym,
                        "reason": f"Only {dte} DTE remaining (< {MIN_DTE_EXIT})",
                    })
                    continue
            except ValueError:
                pass

        # --- Rules 4 & 5: Profit target and stop loss (need live quote) ---
        if has_api and option_sym and entry_credit > 0:
            try:
                url = f"{BASE_URL}/v2/options/snapshots/{option_sym}"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    snap = resp.json()
                    ask = float(
                        snap.get("latestQuote", {}).get("ap", 0)
                        or snap.get("greeks", {}).get("ask", 0)
                        or 0
                    )
                    if ask > 0:
                        # We sold at entry_credit, current buyback cost is ask
                        profit_captured = (entry_credit - ask) / entry_credit

                        # Close at 80% max profit
                        if profit_captured >= PROFIT_CLOSE_PCT:
                            to_close.append({
                                "symbol": sym,
                                "option_symbol": option_sym,
                                "reason": (
                                    f"Profit target: {profit_captured:.0%} captured "
                                    f"(entry=${entry_credit:.2f}, ask=${ask:.2f})"
                                ),
                            })
                            continue

                        # Stop: put doubled in value
                        if ask >= entry_credit * STOP_LOSS_MULT:
                            to_close.append({
                                "symbol": sym,
                                "option_symbol": option_sym,
                                "reason": (
                                    f"Stop loss: put value ${ask:.2f} >= "
                                    f"{STOP_LOSS_MULT}x entry ${entry_credit:.2f}"
                                ),
                            })
                            continue
            except Exception as e:
                print(f"  [WARN] Failed to check exit for {option_sym}: {e}")

    return to_close


# ---------------------------------------------------------------------------
# 5. EXECUTION
# ---------------------------------------------------------------------------

def run_leadership_puts() -> None:
    """
    Main execution function:
    1. Get account info from Alpaca
    2. Load tracked positions
    3. Run exit management
    4. If slots available AND regime=CHOPPY: scan -> select -> place orders
    5. Log every action
    6. Save updated positions
    """
    print("=" * 60)
    print(f"LEADERSHIP PUT ENGINE -- {_ts()}")
    print("=" * 60)

    if not ALPACA_KEY or not ALPACA_SECRET:
        print("[ERR] ALPACA_KEY and ALPACA_SECRET not set in environment")
        print("  Set them and re-run. Exiting.")
        return

    if requests is None:
        print("[ERR] requests library not available")
        return

    headers = _get_headers()

    # --- Step 1: Get account info ---
    try:
        resp = requests.get(f"{BASE_URL}/v2/account", headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"[ERR] Account API returned {resp.status_code}")
            return
        account = resp.json()
        portfolio_value = float(account.get("portfolio_value", 0))
        cash = float(account.get("cash", 0))
        print(f"  Portfolio: ${portfolio_value:,.0f}  Cash: ${cash:,.0f}")
    except Exception as e:
        print(f"[ERR] Failed to get account: {e}")
        return

    # --- Step 2: Load tracked positions ---
    tracked = _load_positions()
    print(f"  Active put positions: {len(tracked)}")
    for sym, pos in tracked.items():
        print(
            f"    {sym}: {pos.get('option_symbol', '?')} "
            f"strike=${pos.get('strike', 0)} exp={pos.get('expiration', '?')}"
        )

    # --- Step 3: Exit management ---
    ml_ranks = _load_ml_ranks()
    exits = manage_positions(tracked, ml_ranks)

    for ex in exits:
        sym = ex["symbol"]
        option_sym = ex["option_symbol"]
        reason = ex["reason"]
        print(f"\n  CLOSE: {sym} ({option_sym}) -- {reason}")

        # Submit buy-to-close order
        pos = tracked.get(sym, {})
        contracts = pos.get("contracts", 1)
        try:
            order_data = {
                "symbol": option_sym,
                "qty": str(contracts),
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
            }
            resp = requests.post(
                f"{BASE_URL}/v2/orders",
                headers=headers,
                json=order_data,
                timeout=10,
            )
            if resp.status_code in (200, 201):
                order = resp.json()
                print(f"    Buy-to-close order submitted: {order.get('id', '?')}")
                _log_action("CLOSE", {
                    "symbol": sym,
                    "option_symbol": option_sym,
                    "reason": reason,
                    "contracts": contracts,
                    "order_id": order.get("id"),
                })
            else:
                print(f"    [WARN] Close order failed: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"    [ERR] Close order failed: {e}")

        tracked.pop(sym, None)

    # --- Step 4: New entries ---
    scan = _load_scan()
    regime = scan.get("regime", "UNKNOWN")
    print(f"\n  Current regime: {regime}")

    slots = MAX_PUTS - len(tracked)
    if slots <= 0:
        print(f"  All {MAX_PUTS} put slots filled -- no new entries")
        _save_positions(tracked)
        return

    if regime != REGIME_REQUIRED:
        print(f"  Regime is {regime}, not {REGIME_REQUIRED} -- no new entries")
        _save_positions(tracked)
        return

    print(f"\n  Open slots: {slots}  Scanning for candidates...")
    candidates = scan_put_candidates(regime, ml_ranks, portfolio_value)

    entered = 0
    for cand in candidates:
        if entered >= slots:
            break

        sym = cand["symbol"]
        if sym in tracked:
            continue

        print(f"\n  Evaluating {sym} @ ${cand['price']:.2f} (conviction={cand['conviction']:.3f})")
        put = select_put_contract(sym, cand["price"], portfolio_value, cand["conviction"])
        if put is None:
            continue

        # Check exposure limits
        ok, reason = _check_exposure_limits(tracked, put["cash_required"], portfolio_value, cash)
        if not ok:
            print(f"    Skipped: {reason}")
            continue

        print(
            f"    PUT: {put['option_symbol']} strike=${put['strike']} "
            f"exp={put['expiration']} DTE={put['dte']}"
        )
        print(
            f"    Premium: ${put['bid']:.2f} ({put['premium_pct']:.2%})  "
            f"Contracts: {put['contracts']}  Cash: ${put['cash_required']:,.0f}"
        )

        # Submit sell-to-open order
        try:
            order_data = {
                "symbol": put["option_symbol"],
                "qty": str(put["contracts"]),
                "side": "sell",
                "type": "limit",
                "time_in_force": "day",
                "limit_price": str(round(put["bid"] - 0.01, 2)),
            }
            resp = requests.post(
                f"{BASE_URL}/v2/orders",
                headers=headers,
                json=order_data,
                timeout=10,
            )
            if resp.status_code in (200, 201):
                order = resp.json()
                print(f"    Sell-to-open order submitted: {order.get('id', '?')}")

                tracked[sym] = {
                    "option_symbol": put["option_symbol"],
                    "underlying": sym,
                    "strike": put["strike"],
                    "expiration": put["expiration"],
                    "entry_credit": put["bid"],
                    "contracts": put["contracts"],
                    "cash_required": put["cash_required"],
                    "conviction": put["conviction"],
                    "entry_date": str(datetime.now().date()),
                    "order_id": order.get("id"),
                }
                cash -= put["cash_required"]
                entered += 1

                _log_action("OPEN", {
                    "symbol": sym,
                    "option_symbol": put["option_symbol"],
                    "strike": put["strike"],
                    "expiration": put["expiration"],
                    "entry_credit": put["bid"],
                    "contracts": put["contracts"],
                    "cash_required": put["cash_required"],
                    "conviction": put["conviction"],
                    "order_id": order.get("id"),
                })
            else:
                print(f"    [WARN] Order failed: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"    [ERR] Order failed: {e}")

    # --- Step 5 & 6: Save ---
    _save_positions(tracked)
    print(f"\n  Done. {entered} new puts entered. Total active: {len(tracked)}")
    _log_action("RUN_COMPLETE", {
        "entered": entered,
        "active": len(tracked),
        "exits": len(exits),
        "regime": regime,
    })


# ---------------------------------------------------------------------------
# 6. BACKTEST
# ---------------------------------------------------------------------------

def backtest_leadership_puts(
    start: str = "2023-01-01",
    end: str = "2025-12-31",
) -> dict:
    """
    Backtest the leadership put selling strategy.

    Reconstructs regime from SPY/HYG/VIX. During CHOPPY periods, identifies
    candidates using momentum proxy and models put P&L via Black-Scholes
    approximation.
    """
    print("=" * 60)
    print("BACKTEST: Leadership Put Selling")
    print(f"Period: {start} to {end}")
    print("=" * 60)

    # --- Load macro data for regime ---
    try:
        from regime_classifier import (
            RegimeClassifier,
            build_regime_series,
            compute_signals,
            load_macro_data,
            CHOPPY,
        )
    except ImportError:
        print("[ERR] Cannot import regime_classifier -- ensure v2/ is in path")
        return {}

    try:
        spy_df, hyg_df, vix_df = load_macro_data()
    except Exception as e:
        print(f"[ERR] Failed to load macro data: {e}")
        return {}

    # --- Build regime series ---
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    # Filter to backtest range (with lookback buffer)
    buffer_start = start_dt - pd.Timedelta(days=300)
    spy_bt = spy_df.loc[buffer_start:end_dt].copy()
    hyg_bt = hyg_df.loc[buffer_start:end_dt].copy()
    vix_bt = vix_df.loc[buffer_start:end_dt].copy()

    dates_in_range = spy_bt.loc[start_dt:end_dt].index
    if len(dates_in_range) == 0:
        print("[ERR] No dates in backtest range")
        return {}

    print(f"  Building regime series for {len(dates_in_range)} trading days...")
    regime_series = build_regime_series(spy_bt, hyg_bt, vix_bt, dates_in_range)

    choppy_dates = regime_series[regime_series == CHOPPY].index
    n_choppy = len(choppy_dates)
    print(f"  CHOPPY days: {n_choppy}/{len(dates_in_range)} ({n_choppy/len(dates_in_range):.0%})")

    # --- Load prices for all symbols ---
    print("  Loading prices...")
    all_prices: Dict[str, pd.DataFrame] = {}
    for sym in WATCHLIST:
        df = _load_prices(sym)
        if df is not None and len(df) >= 210:
            all_prices[sym] = df

    spy_prices = _load_prices("SPY")
    if spy_prices is not None:
        all_prices["SPY"] = spy_prices

    print(f"  Loaded {len(all_prices)} symbols with sufficient history")

    # --- Build leadership adapter ---
    adapter = LeadershipAdapter(update_frequency_days=5)

    # --- Simulate trades ---
    trades = []
    active_puts: Dict[str, dict] = {}  # sym -> put info
    portfolio = 100_000.0
    trade_interval = 21  # check for new trades every 21 days in CHOPPY
    last_trade_date = None

    print("  Running simulation...")

    for date in sorted(dates_in_range):
        regime = regime_series.get(date, "UNKNOWN")

        # Update leadership every 5 days
        hist_slice = {
            sym: df.loc[:date]
            for sym, df in all_prices.items()
            if len(df.loc[:date]) >= 210
        }
        adapter.update(date, hist_slice)

        # --- Check active puts for expiration/outcome ---
        expired = []
        for sym, put in active_puts.items():
            exp_date = pd.Timestamp(put["expiration"])
            if date >= exp_date:
                # Put expired -- check if assigned
                df = all_prices.get(sym)
                if df is None:
                    expired.append(sym)
                    continue

                prices_at_exp = df.loc[:exp_date]
                if len(prices_at_exp) == 0:
                    expired.append(sym)
                    continue

                exit_price = float(prices_at_exp["close"].iloc[-1])
                strike = put["strike"]
                premium = put["premium"]

                if exit_price >= strike:
                    # Expired worthless -- full premium captured
                    pnl = premium * 100 * put["contracts"]
                    outcome = "expired_worthless"
                else:
                    # Assigned -- loss
                    loss_per_share = strike - exit_price
                    pnl = (premium - loss_per_share) * 100 * put["contracts"]
                    outcome = "assigned"

                ret = pnl / put["cash_required"] if put["cash_required"] > 0 else 0
                trades.append({
                    "symbol": sym,
                    "entry_date": str(put["entry_date"]),
                    "exit_date": str(date.date()),
                    "strike": strike,
                    "premium": premium,
                    "entry_price": put["entry_price"],
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "return": round(ret, 4),
                    "outcome": outcome,
                    "contracts": put["contracts"],
                    "cash_required": put["cash_required"],
                })
                portfolio += pnl
                expired.append(sym)

        for sym in expired:
            active_puts.pop(sym, None)

        # --- Close on regime shift to BEAR ---
        if regime == "BEAR" and active_puts:
            for sym in list(active_puts.keys()):
                put = active_puts[sym]
                df = all_prices.get(sym)
                if df is None:
                    active_puts.pop(sym)
                    continue
                prices_now = df.loc[:date]
                if len(prices_now) == 0:
                    active_puts.pop(sym)
                    continue

                current_price = float(prices_now["close"].iloc[-1])
                strike = put["strike"]
                premium = put["premium"]
                dte_remaining = (pd.Timestamp(put["expiration"]) - date).days

                # Estimate buyback cost via BS approximation
                ann_vol = put.get("ann_vol", 0.30)
                T = max(dte_remaining / 365.0, 1 / 365.0)
                intrinsic = max(strike - current_price, 0)
                time_value = 0.4 * current_price * ann_vol * math.sqrt(T) * 0.5
                buyback_cost = intrinsic + time_value

                pnl = (premium - buyback_cost) * 100 * put["contracts"]
                ret = pnl / put["cash_required"] if put["cash_required"] > 0 else 0

                trades.append({
                    "symbol": sym,
                    "entry_date": str(put["entry_date"]),
                    "exit_date": str(date.date()),
                    "strike": strike,
                    "premium": premium,
                    "entry_price": put["entry_price"],
                    "exit_price": current_price,
                    "pnl": round(pnl, 2),
                    "return": round(ret, 4),
                    "outcome": "bear_exit",
                    "contracts": put["contracts"],
                    "cash_required": put["cash_required"],
                })
                portfolio += pnl
                active_puts.pop(sym)

        # --- Enter new puts in CHOPPY ---
        if regime != CHOPPY:
            continue
        if len(active_puts) >= MAX_PUTS:
            continue
        if last_trade_date is not None and (date - last_trade_date).days < trade_interval:
            continue

        state = adapter.get_state()
        if state is None:
            continue

        leader_sectors = state.leader_sectors
        if not leader_sectors:
            leader_sectors = set(state.top_sectors[:3])

        leader_stocks = set()
        for etf in leader_sectors:
            leader_stocks.update(SECTOR_ETFS.get(etf, []))

        # Score candidates using momentum proxy (no ML in backtest)
        bt_candidates = []
        for sym in leader_stocks:
            if sym in active_puts:
                continue
            df = all_prices.get(sym)
            if df is None:
                continue

            prices_to_date = df.loc[:date]
            if len(prices_to_date) < 210:
                continue

            price = float(prices_to_date["close"].iloc[-1])
            sma200 = float(prices_to_date["close"].iloc[-200:].mean())

            # Price > SMA200
            if price <= sma200:
                continue

            # 60-day momentum as ML proxy
            if len(prices_to_date) >= 61:
                mom_60 = float(prices_to_date["close"].iloc[-1] / prices_to_date["close"].iloc[-60] - 1)
            else:
                continue
            if mom_60 <= 0:
                continue

            # Annualized vol
            rets = prices_to_date["close"].pct_change().dropna().iloc[-60:]
            if len(rets) < 20:
                continue
            ann_vol = float(rets.std() * np.sqrt(252))
            if ann_vol >= ANN_VOL_MAX:
                continue

            # ML proxy: normalize 60d momentum to 0-1 rank
            ml_proxy = min(mom_60 / 0.30, 1.0)  # 30% momentum = rank 1.0
            if ml_proxy < 0.50:
                continue

            leadership_mult = adapter.score_multiplier(sym)
            conviction = ml_proxy * leadership_mult * (1.0 - ann_vol)

            bt_candidates.append({
                "symbol": sym,
                "price": price,
                "ann_vol": ann_vol,
                "conviction": conviction,
                "ml_proxy": ml_proxy,
                "leadership_mult": leadership_mult,
            })

        bt_candidates.sort(key=lambda c: c["conviction"], reverse=True)

        # Enter top candidates
        slots = MAX_PUTS - len(active_puts)
        for cand in bt_candidates[:slots]:
            sym = cand["symbol"]
            price = cand["price"]
            ann_vol = cand["ann_vol"]
            conviction = cand["conviction"]

            # Model put: strike at S * 0.93
            strike = round(price * TARGET_DELTA_STRIKE, 2)

            # Black-Scholes premium approximation for 30-delta put
            T = 21.0 / 365.0
            premium = 0.4 * price * ann_vol * math.sqrt(T) * 0.5
            premium = round(premium, 2)

            premium_pct = premium / strike if strike > 0 else 0
            if premium_pct < MIN_PREMIUM_PCT:
                continue

            # Position sizing
            cash_per = strike * 100
            base_contracts = int((portfolio * CAPITAL_PER_PUT) / cash_per)
            if base_contracts < 1:
                continue
            scale = min(conviction / 0.7, 1.3)
            contracts = max(1, int(base_contracts * scale))
            cash_required = cash_per * contracts

            if cash_required > portfolio * MAX_OPTIONS_EXPOSURE:
                continue

            # Set expiration 21 calendar days out
            exp_date = date + pd.Timedelta(days=21)

            active_puts[sym] = {
                "strike": strike,
                "premium": premium,
                "entry_price": price,
                "entry_date": date.date(),
                "expiration": str(exp_date.date()),
                "contracts": contracts,
                "cash_required": cash_required,
                "ann_vol": ann_vol,
                "conviction": conviction,
            }
            last_trade_date = date

    # --- Close any remaining active puts at end of backtest ---
    for sym, put in active_puts.items():
        df = all_prices.get(sym)
        if df is None:
            continue
        prices_end = df.loc[:end_dt]
        if len(prices_end) == 0:
            continue
        exit_price = float(prices_end["close"].iloc[-1])
        strike = put["strike"]
        premium = put["premium"]
        if exit_price >= strike:
            pnl = premium * 100 * put["contracts"]
            outcome = "expired_worthless"
        else:
            pnl = (premium - (strike - exit_price)) * 100 * put["contracts"]
            outcome = "assigned"
        ret = pnl / put["cash_required"] if put["cash_required"] > 0 else 0
        trades.append({
            "symbol": sym,
            "entry_date": str(put["entry_date"]),
            "exit_date": str(end_dt.date()),
            "strike": strike,
            "premium": premium,
            "entry_price": put["entry_price"],
            "exit_price": exit_price,
            "pnl": round(pnl, 2),
            "return": round(ret, 4),
            "outcome": outcome,
            "contracts": put["contracts"],
            "cash_required": put["cash_required"],
        })
        portfolio += pnl

    # --- Report ---
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    if not trades:
        print("  No trades generated.")
        return {"n_trades": 0}

    df_trades = pd.DataFrame(trades)
    n_trades = len(df_trades)
    wins = (df_trades["pnl"] > 0).sum()
    losses = (df_trades["pnl"] <= 0).sum()
    win_rate = wins / n_trades if n_trades > 0 else 0

    total_pnl = df_trades["pnl"].sum()
    avg_ret = df_trades["return"].mean()
    avg_win = df_trades.loc[df_trades["pnl"] > 0, "return"].mean() if wins > 0 else 0
    avg_loss = df_trades.loc[df_trades["pnl"] <= 0, "return"].mean() if losses > 0 else 0

    assigned = (df_trades["outcome"] == "assigned").sum()
    assignment_rate = assigned / n_trades if n_trades > 0 else 0

    # Annual return and Sharpe
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    if years > 0 and total_pnl > -100_000:
        annual_return = ((100_000 + total_pnl) / 100_000) ** (1 / years) - 1
    else:
        annual_return = 0

    rets = df_trades["return"]
    sharpe = (rets.mean() / rets.std()) * np.sqrt(n_trades / years) if rets.std() > 0 and years > 0 else 0

    print(f"  Total trades:     {n_trades}")
    print(f"  Win rate:         {win_rate:.1%}")
    print(f"  Assignment rate:  {assignment_rate:.1%}")
    print(f"  Total P&L:        ${total_pnl:,.0f}")
    print(f"  Avg return/trade: {avg_ret:.2%}")
    print(f"  Avg win:          {avg_win:.2%}")
    print(f"  Avg loss:         {avg_loss:.2%}")
    print(f"  Annual return:    {annual_return:.1%}")
    print(f"  Sharpe ratio:     {sharpe:.2f}")
    print(f"  Final portfolio:  ${portfolio:,.0f}")

    # Outcome breakdown
    print(f"\n  Outcomes:")
    for outcome, count in df_trades["outcome"].value_counts().items():
        avg_r = df_trades.loc[df_trades["outcome"] == outcome, "return"].mean()
        print(f"    {outcome:<20} n={count:>3}  avg_ret={avg_r:+.2%}")

    # Save trades
    output_path = os.path.join(CACHE_DIR, "backtest_leadership_puts.csv")
    df_trades.to_csv(output_path, index=False)
    print(f"\n  Trades saved to {output_path}")

    stats = {
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "assignment_rate": round(assignment_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_return": round(avg_ret, 4),
        "annual_return": round(annual_return, 4),
        "sharpe": round(sharpe, 2),
        "final_portfolio": round(portfolio, 2),
    }
    return stats


# ---------------------------------------------------------------------------
# 7. DAILY SCANNER INTEGRATION
# ---------------------------------------------------------------------------

def add_leadership_puts_to_scan(scan_result: dict) -> dict:
    """
    Integrate leadership put candidates into the daily scanner output.

    If regime=CHOPPY, runs scan_put_candidates and adds results
    to the scan_result dict under 'put_candidates' key.
    """
    regime = scan_result.get("regime", "UNKNOWN")
    if regime != REGIME_REQUIRED:
        scan_result["put_candidates"] = []
        scan_result["put_candidates_note"] = f"Regime {regime} != {REGIME_REQUIRED}"
        return scan_result

    # Get ML ranks from scan candidates or from cache
    ml_ranks = {}
    for c in scan_result.get("candidates", []):
        sym = c.get("symbol", c.get("sym", ""))
        if sym:
            ml_ranks[sym] = c.get("ml_rank_pct", 0)

    # Supplement with cached ML ranks
    cached_ranks = _load_ml_ranks()
    for sym, rank in cached_ranks.items():
        if sym not in ml_ranks:
            ml_ranks[sym] = rank

    # Use default portfolio value for scanning (actual sizing happens at execution)
    portfolio_value = 100_000

    try:
        candidates = scan_put_candidates(regime, ml_ranks, portfolio_value)
        scan_result["put_candidates"] = candidates[:10]  # top 10
        scan_result["put_candidates_count"] = len(candidates)
    except Exception as e:
        print(f"  [WARN] Leadership put scan failed: {e}")
        scan_result["put_candidates"] = []
        scan_result["put_candidates_error"] = str(e)

    return scan_result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leadership Put Selling Engine")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    parser.add_argument("--scan-only", action="store_true", help="Scan for candidates only")
    parser.add_argument("--start", default="2023-01-01", help="Backtest start date")
    parser.add_argument("--end", default="2025-12-31", help="Backtest end date")
    args = parser.parse_args()

    if args.backtest:
        backtest_leadership_puts(start=args.start, end=args.end)
    elif args.scan_only:
        scan = _load_scan()
        regime = scan.get("regime", "UNKNOWN")
        ml_ranks = _load_ml_ranks()
        print(f"Regime: {regime}")
        candidates = scan_put_candidates(regime, ml_ranks, portfolio_value=100_000)
        if candidates:
            print(f"\nTop {min(10, len(candidates))} candidates:")
            for c in candidates[:10]:
                print(
                    f"  {c['symbol']:<6} conv={c['conviction']:.3f}  "
                    f"ML={c['ml_rank']:.2f}  lead={c['leadership_mult']:.2f}  "
                    f"vol={c['ann_vol']:.2f}  price=${c['price']:.2f}"
                )
        else:
            print("No candidates found.")
    else:
        run_leadership_puts()


# ---------------------------------------------------------------------------
# INTEGRATION INSTRUCTIONS
# ---------------------------------------------------------------------------
#
# In daily_scanner.py, add before the final save (around line 320):
#
#     from v2.strategy_leadership_puts import add_leadership_puts_to_scan
#     result = add_leadership_puts_to_scan(result)
#
# This will add 'put_candidates' key to the daily scan JSON output,
# which the leadership put engine reads during its execution run.
#
# Scheduling (crontab):
#     # Scan at 9:25am ET
#     25 9 * * 1-5 cd ~/ai_trading_bot_v2 && python v2/strategy_leadership_puts.py --scan-only
#     # Full engine run at 9:35am ET
#     35 9 * * 1-5 cd ~/ai_trading_bot_v2 && python v2/strategy_leadership_puts.py
# ---------------------------------------------------------------------------
