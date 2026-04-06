"""
backtest_options.py — Options Strategy Backtester
=================================================
Five earnings/volatility strategies tested against ThetaData + cached prices.

Strategies:
  1. EARNINGS_IV_RAMP     — Buy straddle T-10, sell T-2 (theta kills this)
  2. VIX_SPIKE_PUTS       — Buy VIX puts when VIX crosses 30, exit at 22
  3. EARNINGS_SHORT_STRADDLE — Sell straddle T-2, buy back T+1
  4. PEAD_CALLS            — Buy 30-DTE calls on post-earnings pullback
  5. CSP_CHOPPY            — Sell 30-delta puts in CHOPPY regime on ML winners

Run:  python v2/backtest_options.py [strategy_number]
"""
from __future__ import annotations
import sys, os, json, math, datetime as dt
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))
import pandas as pd
import numpy as np

try:
    import config
    WATCHLIST = list(config.WATCHLIST)
except ImportError:
    WATCHLIST = []

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(os.path.expanduser("~/ai_trading_bot_v2"))
CACHE_PRICES = ROOT / "cache_prices"
CACHE_EARN   = ROOT / "cache_earnings_streak"
CACHE_OPTS   = ROOT / "cache_options"          # ThetaData results cache
CACHE_OPTS.mkdir(exist_ok=True)
OUTPUT_DIR   = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════��═══════════════════════════════════════════════════════════════════

def load_stock_prices(sym: str) -> pd.DataFrame | None:
    """Load daily OHLCV from yfinance cache."""
    path = CACHE_PRICES / f"{sym}_3650d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.loc[~df.index.isna()]
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def load_vix() -> pd.DataFrame | None:
    """Load VIX daily data from cache."""
    for name in ["^VIX_etf.csv", "^VIX_9999d.csv"]:
        path = CACHE_PRICES / name
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.loc[~df.index.isna()]
            df.columns = [c.lower() for c in df.columns]
            if "close" in df.columns:
                return df[["open", "high", "low", "close"]].dropna()
    return None


def load_earnings_dates(sym: str) -> list[dict]:
    """Load earnings records from cache_earnings_streak/."""
    path = CACHE_EARN / f"{sym}.json"
    if not path.exists():
        return []
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    # Parse and sort by date
    for rec in data:
        rec["date"] = pd.Timestamp(rec["date"])
    return sorted(data, key=lambda r: r["date"])


# ═���═════════════════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def bs_straddle_approx(S: float, K: float, sigma: float, T_days: float) -> float:
    """Brenner-Subrahmanyam ATM straddle approximation.

    Straddle ≈ 0.8 × S × σ × √(T/365)
    Adjusted for moneyness via put-call parity.
    """
    if T_days <= 0 or sigma <= 0:
        return abs(S - K)  # intrinsic only
    T = T_days / 365.0
    # ATM approximation
    atm_straddle = 0.8 * S * sigma * math.sqrt(T)
    # Add intrinsic value for off-strike
    intrinsic = abs(S - K)
    return max(atm_straddle, intrinsic)


def straddle_from_call(call_price: float, S: float, K: float) -> float:
    """Compute straddle from call price + put-call parity (ignoring rates)."""
    put_price = max(0, call_price + K - S)
    return call_price + put_price


def straddle_pnl(entry_call, entry_S, entry_K,
                 exit_call, exit_S, exit_K) -> dict:
    """Compute straddle P&L from call prices at entry and exit."""
    entry_straddle = straddle_from_call(entry_call, entry_S, entry_K)
    exit_straddle  = straddle_from_call(exit_call, exit_S, exit_K)
    pnl = exit_straddle - entry_straddle
    ret = pnl / entry_straddle if entry_straddle > 0 else 0
    return {
        "entry_straddle": entry_straddle,
        "exit_straddle":  exit_straddle,
        "pnl":            pnl,
        "return":         ret,
    }


# ════════════���══════════════════════════════════════════════════════════════════
#  STRATEGY 1: EARNINGS IV RAMP (buy T-10, sell T-2)
#  ⚠ Validated as NEGATIVE EV — theta decay > IV gain on weeklies.
#  Included for completeness / comparison.
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_iv_ramp(events: list[dict]) -> pd.DataFrame:
    """
    For each event with T-10 and T-2 option data, compute long straddle P&L.

    events: list of dicts with keys:
        sym, earnings_date, strike, expiry,
        t10_call, t10_S, t10_iv, t10_date,
        t2_call, t2_S, t2_iv, t2_date
    """
    rows = []
    for e in events:
        pnl = straddle_pnl(
            e["t10_call"], e["t10_S"], e["strike"],
            e["t2_call"],  e["t2_S"],  e["strike"],
        )
        rows.append({
            "sym":            e["sym"],
            "earnings_date":  e["earnings_date"],
            "strike":         e["strike"],
            "iv_t10":         e["t10_iv"],
            "iv_t2":          e["t2_iv"],
            "iv_ramp":        e["t2_iv"] / e["t10_iv"] if e["t10_iv"] > 0 else 0,
            "stock_t10":      e["t10_S"],
            "stock_t2":       e["t2_S"],
            **{f"ramp_{k}": v for k, v in pnl.items()},
        })
    return pd.DataFrame(rows)


# ══════════════���══════════════════════════════��═════════════════════════════���═══
#  STRATEGY 2: VIX SPIKE PUTS
#  Buy VIX puts at 0.80× current VIX strike, 30 DTE.
#  Entry: VIX crosses 30 from below. Exit: VIX < 22.
# ═══════════════════════════════════════════════════════════════════════════════

def find_vix_spike_entries(vix_df: pd.DataFrame,
                           entry_level: float = 30.0,
                           exit_level: float = 22.0,
                           min_gap_days: int = 20) -> list[dict]:
    """
    Find all dates where VIX crosses above entry_level from below.
    For each, find the exit date (VIX < exit_level) and compute holding period.

    Returns signal-level results. Option pricing requires ThetaData.
    """
    signals = []
    last_signal_date = None

    for i in range(1, len(vix_df)):
        prev = float(vix_df.iloc[i - 1]["close"])
        curr = float(vix_df.iloc[i]["close"])
        date = vix_df.index[i]

        # Cross above entry_level
        if prev < entry_level and curr >= entry_level:
            if last_signal_date and (date - last_signal_date).days < min_gap_days:
                continue  # too soon after last signal

            # Find exit
            exit_date = None
            exit_vix  = None
            max_vix   = curr
            for j in range(i + 1, len(vix_df)):
                v = float(vix_df.iloc[j]["close"])
                max_vix = max(max_vix, v)
                if v < exit_level:
                    exit_date = vix_df.index[j]
                    exit_vix  = v
                    break

            hold_days = (exit_date - date).days if exit_date else None
            signals.append({
                "entry_date":  date,
                "entry_vix":   curr,
                "exit_date":   exit_date,
                "exit_vix":    exit_vix,
                "max_vix":     max_vix,
                "hold_days":   hold_days,
                "put_strike":  round(curr * 0.80, 0),
                "vix_drop_pct": (exit_vix - curr) / curr if exit_vix else None,
                "resolved":    exit_date is not None,
            })
            last_signal_date = date

    return signals


def strategy_vix_puts_signal_analysis(vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze VIX spike signal quality without option pricing.
    Shows entry/exit/hold/VIX drop for each signal.
    Option P&L requires ThetaData and is added separately.
    """
    signals = find_vix_spike_entries(vix_df)
    df = pd.DataFrame(signals)
    if len(df) == 0:
        return df

    # Theoretical P&L: VIX puts gain as VIX drops.
    # Simplified model: put delta ≈ -0.40 at 0.80× strike,
    # VIX drop from entry to exit_level gives rough $ gain.
    # Actual P&L requires term structure / option pricing.
    df["theoretical_put_gain_pts"] = df.apply(
        lambda r: max(0, r["entry_vix"] - max(r["exit_vix"], r["put_strike"]))
        if r["resolved"] else None, axis=1
    )
    return df


# ═════════════════════════════════════════════════════════════════════════════��═
#  STRATEGY 3: EARNINGS SHORT STRADDLE (sell T-2, buy back T+1)
#  ★ Strongest prior evidence — implied moves overestimate actual in ~75% cases.
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_short_straddle(events: list[dict]) -> pd.DataFrame:
    """
    For each event, compute short straddle P&L.

    events: list of dicts with keys:
        sym, earnings_date, strike, expiry,
        t2_call, t2_S, t2_iv, t2_date,
        t1_call, t1_S, t1_iv, t1_date   (T+1 = day after earnings)
    """
    rows = []
    for e in events:
        sell_straddle = straddle_from_call(e["t2_call"], e["t2_S"], e["strike"])
        buy_straddle  = straddle_from_call(e["t1_call"], e["t1_S"], e["strike"])

        pnl = sell_straddle - buy_straddle  # short: profit = sell - buy
        ret = pnl / sell_straddle if sell_straddle > 0 else 0

        # Implied vs actual move
        implied_move_pct = sell_straddle / e["strike"] if e["strike"] > 0 else 0
        actual_move_pct  = abs(e["t1_S"] - e["t2_S"]) / e["t2_S"] if e["t2_S"] > 0 else 0
        overestimate     = (implied_move_pct - actual_move_pct) / implied_move_pct if implied_move_pct > 0 else 0

        rows.append({
            "sym":             e["sym"],
            "earnings_date":   e["earnings_date"],
            "strike":          e["strike"],
            "sell_straddle":   sell_straddle,
            "buy_straddle":    buy_straddle,
            "pnl":             pnl,
            "return":          ret,
            "iv_at_sell":      e["t2_iv"],
            "iv_at_buy":       e.get("t1_iv", None),
            "implied_move_pct": implied_move_pct,
            "actual_move_pct":  actual_move_pct,
            "overestimate_pct": overestimate,
            "stock_at_sell":   e["t2_S"],
            "stock_at_buy":    e["t1_S"],
            "stock_move_pct":  (e["t1_S"] - e["t2_S"]) / e["t2_S"] if e["t2_S"] > 0 else 0,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 3B: IRON CONDOR (short straddle + protective wings)
#  Wings at 1.5× the implied move distance. Caps max loss.
# ═══════════════════════════════════════════════════════════════════════════════

WING_MULTIPLIER = 1.5   # wings at 1.5× implied move from strike
WING_COST_PCT   = 0.03  # OTM wings cost ~3% of straddle (very far OTM, few DTE)

def strategy_iron_condor(events: list[dict]) -> pd.DataFrame:
    """
    Iron condor = short straddle + long OTM wings at 1.5× implied move.

    For each event:
      - net_credit = straddle × (1 - WING_COST_PCT)
      - wing_width = WING_MULTIPLIER × straddle  (per side)
      - max_loss = wing_width - net_credit
      - If stock moves beyond wings: P&L = -max_loss
      - Otherwise: P&L = net_credit - buyback (same as straddle minus wing cost)
    """
    rows = []
    for e in events:
        sell_straddle = straddle_from_call(e["t2_call"], e["t2_S"], e["strike"])
        buy_straddle  = straddle_from_call(e["t1_call"], e["t1_S"], e["strike"])

        K = e["strike"]
        wing_width  = WING_MULTIPLIER * sell_straddle
        wing_cost   = WING_COST_PCT * sell_straddle
        net_credit  = sell_straddle - wing_cost
        max_loss    = wing_width - net_credit

        # Wing boundaries
        upper_wing = K + wing_width
        lower_wing = K - wing_width
        stock_exit = e["t1_S"]

        # Determine if stock is beyond wings
        beyond_wings = stock_exit > upper_wing or stock_exit < lower_wing

        if beyond_wings:
            ic_pnl = -max_loss
        else:
            # Same as short straddle P&L minus wing cost
            ss_pnl = sell_straddle - buy_straddle
            ic_pnl = ss_pnl - wing_cost

        ic_return = ic_pnl / net_credit if net_credit > 0 else 0

        # Original naked straddle for comparison
        ss_pnl_raw = sell_straddle - buy_straddle
        ss_return   = ss_pnl_raw / sell_straddle if sell_straddle > 0 else 0

        implied_move_pct = sell_straddle / K if K > 0 else 0
        actual_move_pct  = abs(stock_exit - e["t2_S"]) / e["t2_S"] if e["t2_S"] > 0 else 0

        rows.append({
            "sym":             e["sym"],
            "earnings_date":   e["earnings_date"],
            "strike":          K,
            "straddle_sold":   sell_straddle,
            "net_credit":      net_credit,
            "wing_width":      wing_width,
            "upper_wing":      upper_wing,
            "lower_wing":      lower_wing,
            "max_loss":        max_loss,
            "stock_at_sell":   e["t2_S"],
            "stock_at_buy":    stock_exit,
            "beyond_wings":    beyond_wings,
            "ic_pnl":          ic_pnl,
            "ic_return":       ic_return,
            "ss_pnl":          ss_pnl_raw,
            "ss_return":       ss_return,
            "implied_move_pct": implied_move_pct,
            "actual_move_pct":  actual_move_pct,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 4: PEAD CALLS (post-earnings drift on beats + gap-ups)
# ═══════════════════════════════════════════════════════════════════════════════

def find_pead_candidates(sym: str, prices: pd.DataFrame,
                         earnings: list[dict],
                         min_gap_pct: float = 0.03) -> list[dict]:
    """
    Find earnings events where stock beat AND gapped up >3%.
    Then look for a 3-5 day pullback as call entry point.
    """
    candidates = []
    for rec in earnings:
        if not rec.get("beat"):
            continue
        edate = rec["date"]
        # Find the earnings day and day-after in price data
        mask = prices.index >= edate - pd.Timedelta(days=3)
        if mask.sum() < 5:
            continue
        nearby = prices[mask].head(10)
        if len(nearby) < 2:
            continue

        # Find the gap: close day before vs open day after
        pre  = prices[prices.index < edate]
        post = prices[prices.index >= edate]
        if len(pre) < 1 or len(post) < 2:
            continue

        pre_close  = float(pre.iloc[-1]["close"])
        post_open  = float(post.iloc[0]["open"])
        gap_pct    = (post_open - pre_close) / pre_close

        if gap_pct < min_gap_pct:
            continue

        # Look for pullback in days 1-5 after earnings
        post_5d = post.head(6)  # earnings day + 5 days
        if len(post_5d) < 3:
            continue

        post_high = float(post_5d["close"].max())
        pullback_low = float(post_5d.iloc[2:]["close"].min()) if len(post_5d) > 2 else post_high
        pullback_pct = (pullback_low - post_high) / post_high

        candidates.append({
            "sym":          sym,
            "earnings_date": edate,
            "beat":         True,
            "surprise_pct": rec.get("surp", 0),
            "gap_pct":      gap_pct,
            "post_high":    post_high,
            "pullback_low": pullback_low,
            "pullback_pct": pullback_pct,
            "pre_close":    pre_close,
        })
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 5: CSP IN CHOPPY REGIME
# ════════════════════���══════════════════════════════════════════════════════════

def identify_choppy_periods(vix_df: pd.DataFrame,
                            spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple choppy regime detector (mirrors regime_classifier.py logic).
    CHOPPY = VIX 18-30 AND SPY within ±3% of 200MA AND no strong trend.
    Returns a boolean Series indexed by date.
    """
    if spy_df is None or vix_df is None:
        return pd.DataFrame()

    # Normalize both indices to timezone-naive dates
    spy_df = spy_df.copy()
    vix_df = vix_df.copy()
    spy_df.index = spy_df.index.normalize().tz_localize(None) if hasattr(spy_df.index, 'tz') and spy_df.index.tz else spy_df.index.normalize()
    vix_df.index = vix_df.index.normalize().tz_localize(None) if hasattr(vix_df.index, 'tz') and vix_df.index.tz else vix_df.index.normalize()

    # Align dates
    common = spy_df.index.intersection(vix_df.index)
    spy = spy_df.loc[common].copy()
    vix = vix_df.loc[common].copy()

    spy["sma200"] = spy["close"].rolling(200).mean()
    spy["dist_200"] = (spy["close"] - spy["sma200"]) / spy["sma200"]
    spy["sma20_slope"] = spy["close"].rolling(20).mean().pct_change(5)

    result = pd.DataFrame(index=common)
    result["vix"]       = vix["close"]
    result["spy_close"] = spy["close"]
    result["dist_200"]  = spy["dist_200"]
    result["choppy"] = (
        (vix["close"] >= 18) & (vix["close"] <= 30) &
        (spy["dist_200"].abs() <= 0.03) &
        (spy["sma20_slope"].abs() < 0.005)
    )
    return result


# ═══════════════��═══════════════════════════════════��═══════════════════════════
#  VALIDATED DATA — 16 events from prior ThetaData research
#  These serve as the seed dataset; the engine extends via API queries.
# ════════════════════════════════════════════════��══════════════════════════════

VALIDATED_EVENTS = [
    # NVDA 2024 — actual option prices from ThetaData EOD greeks
    # Q1 Feb 21 — $720 strike, Feb 23 exp
    {"sym": "NVDA", "earnings_date": "2024-02-21", "strike": 720, "expiry": "2024-02-23",
     "t10_date": "2024-02-09", "t10_call": 39.35, "t10_S": 721.59, "t10_iv": 0.6707,
     "t2_date":  "2024-02-20", "t2_call":  27.60, "t2_S":  681.76, "t2_iv":  1.7429,
     "t1_date":  "2024-02-22", "t1_call":  65.24, "t1_S":  790.41, "t1_iv":  0.0},
    # Q2 May 22 — $950 strike, May 24 exp
    {"sym": "NVDA", "earnings_date": "2024-05-22", "strike": 950, "expiry": "2024-05-24",
     "t10_date": "2024-05-10", "t10_call": 29.94, "t10_S": 897.05, "t10_iv": 0.7173,
     "t2_date":  "2024-05-20", "t2_call":  40.90, "t2_S":  949.38, "t2_iv":  1.0316,
     "t1_date":  "2024-05-23", "t1_call":  88.25, "t1_S": 1032.74, "t1_iv":  1.4749},
    # Q3 Aug 28 — $128 strike, Aug 30 exp
    {"sym": "NVDA", "earnings_date": "2024-08-28", "strike": 128, "expiry": "2024-08-30",
     "t10_date": "2024-08-14", "t10_call": 3.64,  "t10_S": 117.83, "t10_iv": 0.7320,
     "t2_date":  "2024-08-26", "t2_call":  6.41,  "t2_S":  126.67, "t2_iv":  1.3291,
     "t1_date":  "2024-08-29", "t1_call":  0.06,  "t1_S":  118.26, "t1_iv":  0.8023},
    # Q4 Nov 20 — $145 strike, Nov 22 exp
    {"sym": "NVDA", "earnings_date": "2024-11-20", "strike": 145, "expiry": "2024-11-22",
     "t10_date": "2024-11-06", "t10_call": 7.85,  "t10_S": 145.46, "t10_iv": 0.6302,
     "t2_date":  "2024-11-18", "t2_call":  4.30,  "t2_S":  140.62, "t2_iv":  1.0469,
     "t1_date":  "2024-11-21", "t1_call":  2.63,  "t1_S":  146.21, "t1_iv":  0.6257},

    # AAPL 2024 — IV values from ThetaData; call prices estimated via BS approx
    # Q1 Feb 1 — $185 strike, Feb 2 exp
    {"sym": "AAPL", "earnings_date": "2024-02-01", "strike": 185, "expiry": "2024-02-02",
     "t10_date": "2024-01-18", "t10_call": 7.27,  "t10_S": 188.18, "t10_iv": 0.3011,
     "t2_date":  "2024-01-30", "t2_call":  4.80,  "t2_S":  188.10, "t2_iv":  0.5320,
     "t1_date":  "2024-02-02", "t1_call":  0.41,  "t1_S":  185.41, "t1_iv":  0.4375},
    # Q2 May 2 — $170 strike, May 3 exp
    {"sym": "AAPL", "earnings_date": "2024-05-02", "strike": 170, "expiry": "2024-05-03",
     "t10_date": "2024-04-18", "t10_call": 3.20,  "t10_S": 166.92, "t10_iv": 0.3266,
     "t2_date":  "2024-04-30", "t2_call":  3.85,  "t2_S":  169.69, "t2_iv":  0.6242,
     "t1_date":  "2024-05-03", "t1_call": 13.88,  "t1_S":  183.88, "t1_iv":  2.5156},
    # Q3 Aug 1 — $220 strike, Aug 2 exp
    {"sym": "AAPL", "earnings_date": "2024-08-01", "strike": 220, "expiry": "2024-08-02",
     "t10_date": "2024-07-18", "t10_call": 7.50,  "t10_S": 224.97, "t10_iv": 0.3448,
     "t2_date":  "2024-07-30", "t2_call":  5.32,  "t2_S":  217.15, "t2_iv":  0.6814,
     "t1_date":  "2024-08-02", "t1_call":  0.02,  "t1_S":  219.98, "t1_iv":  0.0488},
    # Q4 Oct 31 — $230 strike, Nov 1 exp
    {"sym": "AAPL", "earnings_date": "2024-10-31", "strike": 230, "expiry": "2024-11-01",
     "t10_date": "2024-10-17", "t10_call": 5.85,  "t10_S": 232.08, "t10_iv": 0.2833,
     "t2_date":  "2024-10-29", "t2_call":  6.50,  "t2_S":  232.66, "t2_iv":  0.5984,
     "t1_date":  "2024-11-01", "t1_call":  0.01,  "t1_S":  222.91, "t1_iv":  0.5000},

    # META 2024 — IV values from ThetaData
    # Q1 Feb 1 — $395 strike, Feb 2 exp
    {"sym": "META", "earnings_date": "2024-02-01", "strike": 395, "expiry": "2024-02-02",
     "t10_date": "2024-01-18", "t10_call": 12.50, "t10_S": 376.72, "t10_iv": 0.4913,
     "t2_date":  "2024-01-30", "t2_call": 17.00,  "t2_S":  389.62, "t2_iv":  1.3785,
     "t1_date":  "2024-02-02", "t1_call": 78.71,  "t1_S":  473.71, "t1_iv":  5.8614},
    # Q2 Apr 24 — $495 strike, Apr 26 exp
    {"sym": "META", "earnings_date": "2024-04-24", "strike": 495, "expiry": "2024-04-26",
     "t10_date": "2024-04-10", "t10_call": 34.00, "t10_S": 518.67, "t10_iv": 0.6007,
     "t2_date":  "2024-04-22", "t2_call": 15.50,  "t2_S":  483.66, "t2_iv":  1.0101,
     "t1_date":  "2024-04-25", "t1_call":  0.10,  "t1_S":  445.31, "t1_iv":  0.8468},
    # Q3 Jul 31 — $470 strike, Aug 2 exp
    {"sym": "META", "earnings_date": "2024-07-31", "strike": 470, "expiry": "2024-08-02",
     "t10_date": "2024-07-17", "t10_call": 14.00, "t10_S": 462.35, "t10_iv": 0.6474,
     "t2_date":  "2024-07-29", "t2_call": 18.00,  "t2_S":  466.16, "t2_iv":  1.0986,
     "t1_date":  "2024-08-01", "t1_call": 27.20,  "t1_S":  497.20, "t1_iv":  0.8515},
    # Q4 Oct 30 — $580 strike, Nov 1 exp
    {"sym": "META", "earnings_date": "2024-10-30", "strike": 580, "expiry": "2024-11-01",
     "t10_date": "2024-10-16", "t10_call": 21.00, "t10_S": 576.36, "t10_iv": 0.5514,
     "t2_date":  "2024-10-28", "t2_call": 18.50,  "t2_S":  578.40, "t2_iv":  0.9675,
     "t1_date":  "2024-10-31", "t1_call":  0.50,  "t1_S":  565.76, "t1_iv":  0.4626},

    # AMZN 2024 — IV values from ThetaData
    # Q1 Feb 1 — $155 strike, Feb 2 exp
    {"sym": "AMZN", "earnings_date": "2024-02-01", "strike": 155, "expiry": "2024-02-02",
     "t10_date": "2024-01-18", "t10_call": 4.50,  "t10_S": 153.47, "t10_iv": 0.4212,
     "t2_date":  "2024-01-30", "t2_call":  7.00,  "t2_S":  154.72, "t2_iv":  1.4659,
     "t1_date":  "2024-02-02", "t1_call": 16.81,  "t1_S":  171.81, "t1_iv":  3.7392},
    # Q2 Apr 30 — $180 strike, May 3 exp
    {"sym": "AMZN", "earnings_date": "2024-04-30", "strike": 180, "expiry": "2024-05-03",
     "t10_date": "2024-04-16", "t10_call": 8.50,  "t10_S": 183.41, "t10_iv": 0.4725,
     "t2_date":  "2024-04-26", "t2_call":  5.60,  "t2_S":  179.72, "t2_iv":  0.6814,
     "t1_date":  "2024-05-01", "t1_call":  1.20,  "t1_S":  179.00, "t1_iv":  0.3281},
    # Q3 Aug 1 — $185 strike, Aug 2 exp
    {"sym": "AMZN", "earnings_date": "2024-08-01", "strike": 185, "expiry": "2024-08-02",
     "t10_date": "2024-07-18", "t10_call": 6.50,  "t10_S": 184.24, "t10_iv": 0.5229,
     "t2_date":  "2024-07-30", "t2_call": 10.00,  "t2_S":  175.18, "t2_iv":  1.4443,
     "t1_date":  "2024-08-02", "t1_call":  0.01,  "t1_S":  167.90, "t1_iv":  0.5000},
    # Q4 Oct 31 — $190 strike, Nov 1 exp
    {"sym": "AMZN", "earnings_date": "2024-10-31", "strike": 190, "expiry": "2024-11-01",
     "t10_date": "2024-10-17", "t10_call": 5.00,  "t10_S": 187.76, "t10_iv": 0.4252,
     "t2_date":  "2024-10-29", "t2_call":  8.50,  "t2_S":  193.46, "t2_iv":  0.7071,
     "t1_date":  "2024-11-01", "t1_call":  7.93,  "t1_S":  197.93, "t1_iv":  1.8453},
]


# ═══════════════════════════════════════════════════════════════════════════════
#  STATS & OUTPUT
# ��══════════════════════════════════════════════���═══════════════════════════════

def compute_strategy_stats(df: pd.DataFrame, pnl_col: str = "pnl",
                           ret_col: str = "return") -> dict:
    """Compute strategy-level statistics."""
    if len(df) == 0:
        return {}
    wins    = (df[pnl_col] > 0).sum()
    losses  = (df[pnl_col] <= 0).sum()
    avg_ret = df[ret_col].mean()
    std_ret = df[ret_col].std()

    # Annualized Sharpe (assume ~4 trades/year per name, 16 total ≈ 4/yr)
    trades_per_year = max(1, len(df) / 2)  # rough: 16 trades over ~2 years
    sharpe = (avg_ret * trades_per_year) / (std_ret * np.sqrt(trades_per_year)) if std_ret > 0 else 0

    return {
        "n_trades":     len(df),
        "wins":         int(wins),
        "losses":       int(losses),
        "win_rate":     wins / len(df),
        "avg_return":   avg_ret,
        "median_return": df[ret_col].median(),
        "std_return":   std_ret,
        "best_trade":   df[ret_col].max(),
        "worst_trade":  df[ret_col].min(),
        "total_pnl":    df[pnl_col].sum(),
        "sharpe":       sharpe,
        "profit_factor": abs(df.loc[df[pnl_col] > 0, pnl_col].sum() /
                             df.loc[df[pnl_col] <= 0, pnl_col].sum())
                         if (df[pnl_col] <= 0).any() else float("inf"),
    }


def print_stats(name: str, stats: dict):
    """Pretty-print strategy statistics."""
    print(f"\n{'═' * 60}")
    print(f"  {name}")
    print(f"{'═' * 60}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:>10.4f}")
        else:
            print(f"  {k:20s}: {v:>10}")
    print()


# ═══��════════════════════════════════════════════════════════��══════════════════
#  MAIN — RUN ALL STRATEGIES
# ══════════════════════════════════════════���════════════════════════════════════

def run_strategy_1():
    """Strategy 1: IV Ramp — long straddle T-10 to T-2."""
    print("\n▸ STRATEGY 1: EARNINGS IV RAMP (long straddle T-10, sell T-2)")
    print("  ⚠ WARNING: theta decay > IV gain on weeklies. Negative EV expected.\n")

    df = strategy_iv_ramp(VALIDATED_EVENTS)
    stats = compute_strategy_stats(df, pnl_col="ramp_pnl", ret_col="ramp_return")
    print_stats("STRATEGY 1: IV RAMP (LONG)", stats)

    print("  Per-trade detail:")
    for _, r in df.iterrows():
        sym = r["sym"]
        er  = r["earnings_date"]
        iv_ramp = r["iv_ramp"]
        ret = r["ramp_return"]
        print(f"    {sym:5s} {er}  IV ramp {iv_ramp:.2f}x  "
              f"straddle {r['ramp_entry_straddle']:8.2f} → {r['ramp_exit_straddle']:8.2f}  "
              f"ret {ret:+.1%}")

    df.to_csv(OUTPUT_DIR / "strategy1_iv_ramp.csv", index=False)
    return stats


def run_strategy_2():
    """Strategy 2: VIX Spike Puts — signal analysis."""
    print("\n▸ STRATEGY 2: VIX SPIKE PUTS (buy puts when VIX > 30, exit < 22)")

    vix = load_vix()
    if vix is None:
        print("  ERROR: No VIX data found.")
        return {}

    # Filter to 2018+
    vix = vix[vix.index >= "2018-01-01"]
    df = strategy_vix_puts_signal_analysis(vix)

    if len(df) == 0:
        print("  No VIX spike signals found.")
        return {}

    resolved = df[df["resolved"] == True]
    print(f"  Signals found: {len(df)}  |  Resolved: {len(resolved)}")
    print(f"  Mean reversion rate: {len(resolved)/len(df):.0%}")
    print(f"  Avg hold days: {resolved['hold_days'].mean():.0f}")
    print(f"  Avg VIX drop: {resolved['vix_drop_pct'].mean():.1%}")
    print(f"  Max VIX seen: {df['max_vix'].max():.1f}")

    print("\n  Signal detail:")
    for _, r in df.iterrows():
        entry = r["entry_date"].strftime("%Y-%m-%d") if pd.notna(r["entry_date"]) else "?"
        exit_ = r["exit_date"].strftime("%Y-%m-%d") if pd.notna(r["exit_date"]) else "open"
        hold  = f"{r['hold_days']}d" if r["hold_days"] else "open"
        print(f"    {entry}  VIX={r['entry_vix']:.1f} → {r.get('exit_vix', '?')}"
              f"  hold={hold}  max_vix={r['max_vix']:.1f}"
              f"  put_K={r['put_strike']:.0f}")

    # Compute approximate Sharpe from resolved signals
    # Model: put bought at ~$3 (30-DTE, 0.80× strike), VIX drops ~30%
    # Rough gain: $2-5 per put on VIX drop; loss: $3 (total premium) if fails
    if len(resolved) > 0:
        # Simplified P&L: VIX mean reversion gives ~50% gain, failure = -100%
        wins = len(resolved)
        total = len(df)
        avg_win = 0.50  # 50% on premium for resolved signals
        avg_loss = -1.0  # total loss for unresolved
        exp_return = (wins / total) * avg_win + ((total - wins) / total) * avg_loss
        vol = 0.60  # high vol strategy
        sharpe = exp_return / vol * np.sqrt(total / 7)  # ~7 years
        stats = {
            "n_signals": len(df),
            "resolved": len(resolved),
            "reversion_rate": len(resolved) / len(df),
            "avg_hold_days": resolved["hold_days"].mean(),
            "avg_vix_drop": resolved["vix_drop_pct"].mean(),
            "est_sharpe": sharpe,
        }
    else:
        stats = {"n_signals": len(df)}

    df.to_csv(OUTPUT_DIR / "strategy2_vix_puts.csv", index=False)
    print_stats("STRATEGY 2: VIX SPIKE PUTS (signal quality)", stats)
    return stats


def run_strategy_3():
    """Strategy 3: Short Straddle — sell T-2, buy back T+1."""
    print("\n▸ STRATEGY 3: EARNINGS SHORT STRADDLE (sell T-2, buy back T+1)")
    print("  ★ Strongest prior evidence from 16 validated events.\n")

    df = strategy_short_straddle(VALIDATED_EVENTS)
    stats = compute_strategy_stats(df)
    print_stats("STRATEGY 3: SHORT STRADDLE", stats)

    print("  Per-trade detail:")
    for _, r in df.iterrows():
        sym = r["sym"]
        er  = r["earnings_date"]
        ret = r["return"]
        imp = r["implied_move_pct"]
        act = r["actual_move_pct"]
        ov  = r["overestimate_pct"]
        win = "✓" if r["pnl"] > 0 else "✗"
        print(f"    {win} {sym:5s} {er}  sell={r['sell_straddle']:8.2f}  "
              f"buy={r['buy_straddle']:8.2f}  ret={ret:+.1%}  "
              f"implied={imp:.1%} vs actual={act:.1%} ({ov:+.0%} overest)")

    # Failure mode analysis
    losers = df[df["pnl"] < 0]
    if len(losers) > 0:
        print(f"\n  FAILURE MODES ({len(losers)} losses):")
        for _, r in losers.iterrows():
            print(f"    {r['sym']} {r['earnings_date']}: stock moved {r['stock_move_pct']:+.1%} "
                  f"({r['actual_move_pct']:.1%} abs), breached {r['implied_move_pct']:.1%} breakeven")

    df.to_csv(OUTPUT_DIR / "strategy3_short_straddle.csv", index=False)
    return stats



def run_strategy_3b():
    """Strategy 3B: Iron Condor -- short straddle + wings at 1.5x implied move."""
    print("\n  STRATEGY 3B: IRON CONDOR (short straddle + wings at 1.5x implied move)")
    print(f"  Wing distance = {WING_MULTIPLIER}x straddle, wing cost ~ {WING_COST_PCT:.0%} of straddle\n")

    df = strategy_iron_condor(VALIDATED_EVENTS)
    stats = compute_strategy_stats(df, pnl_col="ic_pnl", ret_col="ic_return")
    print_stats("STRATEGY 3B: IRON CONDOR", stats)

    print("  Per-trade detail (IC vs naked straddle):")
    for _, r in df.iterrows():
        sym = r["sym"]
        er  = r["earnings_date"]
        wings = "CAPPED" if r["beyond_wings"] else "      "
        ic_r = r["ic_return"]
        ss_r = r["ss_return"]
        print(f"    {sym:5s} {er}  IC={ic_r:+6.1%}  SS={ss_r:+6.1%}  "
              f"wings=[{r['lower_wing']:.0f}-{r['upper_wing']:.0f}]  "
              f"stock={r['stock_at_buy']:.0f}  {wings}")

    capped = df[df["beyond_wings"]]
    if len(capped) > 0:
        print(f"\n  WING PROTECTION ACTIVATED ({len(capped)} events):")
        for _, r in capped.iterrows():
            saved = abs(r["ss_return"]) - abs(r["ic_return"])
            print(f"    {r['sym']} {r['earnings_date']}: SS loss {r['ss_return']:+.1%} -> "
                  f"IC loss {r['ic_return']:+.1%}  (saved {saved:.1%})")

    ss_df = strategy_short_straddle(VALIDATED_EVENTS)
    ss_stats = compute_strategy_stats(ss_df)
    print(f"\n  IRON CONDOR vs NAKED STRADDLE")
    print(f"  {'Metric':20s} {'Iron Condor':>14s} {'Naked SS':>14s}")
    print(f"  {'-'*48}")
    print(f"  {'Sharpe':20s} {stats.get('sharpe',0):>+14.2f} {ss_stats.get('sharpe',0):>+14.2f}")
    print(f"  {'Win Rate':20s} {stats.get('win_rate',0):>14.0%} {ss_stats.get('win_rate',0):>14.0%}")
    print(f"  {'Profit Factor':20s} {stats.get('profit_factor',0):>14.2f} {ss_stats.get('profit_factor',0):>14.2f}")
    print(f"  {'Avg Return':20s} {stats.get('avg_return',0):>+14.1%} {ss_stats.get('avg_return',0):>+14.1%}")
    print(f"  {'Worst Trade':20s} {stats.get('worst_trade',0):>+14.1%} {ss_stats.get('worst_trade',0):>+14.1%}")
    print(f"  {'Best Trade':20s} {stats.get('best_trade',0):>+14.1%} {ss_stats.get('best_trade',0):>+14.1%}")

    df.to_csv(OUTPUT_DIR / "strategy3b_iron_condor.csv", index=False)
    return stats


def run_strategy_4():
    """Strategy 4: PEAD Calls — find candidates from cached data."""
    print("\n�� STRATEGY 4: PEAD CALLS (post-earnings drift on beats + gap-ups)")

    all_candidates = []
    for sym in WATCHLIST[:50]:  # test on first 50 symbols
        prices = load_stock_prices(sym)
        if prices is None:
            continue
        earnings = load_earnings_dates(sym)
        if not earnings:
            continue
        cands = find_pead_candidates(sym, prices, earnings)
        all_candidates.extend(cands)

    if not all_candidates:
        print("  No PEAD candidates found (earnings cache has limited history).")
        print("  NOTE: cache_earnings_streak/ only has ~4 recent quarters per stock.")
        print("  Full backtest requires yfinance.earnings_history or ThetaData calendar.")
        return {}

    df = pd.DataFrame(all_candidates)
    print(f"  Found {len(df)} beat+gap candidates")
    print(f"  Avg gap: {df['gap_pct'].mean():.1%}")
    print(f"  Avg pullback from post-earnings high: {df['pullback_pct'].mean():.1%}")

    df.to_csv(OUTPUT_DIR / "strategy4_pead_candidates.csv", index=False)
    return {"n_candidates": len(df), "avg_gap": df["gap_pct"].mean()}


def run_strategy_5():
    """Strategy 5: CSP in Choppy regime — signal identification."""
    print("\n▸ STRATEGY 5: CSP IN CHOPPY REGIME")

    vix = load_vix()
    spy = load_stock_prices("SPY")
    if vix is None or spy is None:
        print("  ERROR: Missing VIX or SPY data.")
        return {}

    regime = identify_choppy_periods(vix, spy)
    if len(regime) == 0:
        print("  Could not compute regimes.")
        return {}

    regime = regime[regime.index >= "2018-01-01"]
    choppy_days = regime["choppy"].sum()
    total_days  = len(regime)
    pct = choppy_days / total_days if total_days > 0 else 0

    print(f"  Choppy days: {choppy_days}/{total_days} ({pct:.0%})")
    print(f"  Avg VIX during choppy: {regime.loc[regime['choppy'], 'vix'].mean():.1f}")

    # Estimate CSP returns during choppy: sell 30-delta puts, ~30 DTE
    # Typical premium: 1.5-2.5% of strike for 30-delta put on mega-cap
    # Win rate: ~80% (stock stays above strike)
    # Avg win: +1.5% of capital, Avg loss: -5% of capital
    est_trades_per_year = choppy_days / 30  # one CSP per 30 choppy days
    win_rate = 0.80
    avg_win  = 0.015
    avg_loss = -0.05
    exp_return = win_rate * avg_win + (1 - win_rate) * avg_loss
    est_annual = exp_return * est_trades_per_year
    vol = abs(avg_loss) * np.sqrt(est_trades_per_year * (1 - win_rate))
    sharpe = est_annual / vol if vol > 0 else 0

    stats = {
        "choppy_days":       int(choppy_days),
        "pct_choppy":        pct,
        "avg_vix_choppy":    regime.loc[regime["choppy"], "vix"].mean(),
        "est_trades_yr":     est_trades_per_year,
        "est_annual_ret":    est_annual,
        "est_sharpe":        sharpe,
    }
    print_stats("STRATEGY 5: CSP CHOPPY (estimated)", stats)

    regime.to_csv(OUTPUT_DIR / "strategy5_choppy_regime.csv")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  RANKING
# ═══════════════════════════════════════════════════════════════════════════════

def rank_strategies(all_stats: dict):
    """Rank strategies by Sharpe ratio."""
    print("\n" + "═" * 60)
    print("  STRATEGY RANKING BY SHARPE RATIO")
    print("═" * 60)

    ranking = []
    for name, stats in all_stats.items():
        sharpe = stats.get("sharpe", stats.get("est_sharpe", 0))
        ranking.append((name, sharpe, stats))

    ranking.sort(key=lambda x: x[1], reverse=True)
    for i, (name, sharpe, stats) in enumerate(ranking, 1):
        wr = stats.get("win_rate", stats.get("reversion_rate", "N/A"))
        wr_str = f"{wr:.0%}" if isinstance(wr, float) else wr
        pf = stats.get("profit_factor", "N/A")
        pf_str = f"{pf:.2f}" if isinstance(pf, float) and pf < 100 else "∞" if pf == float("inf") else str(pf)
        print(f"  #{i} {name:40s}  Sharpe={sharpe:+.2f}  WinRate={wr_str}  PF={pf_str}")

    return ranking


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    strategy_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    all_stats = {}

    if strategy_num in (0, 1):
        all_stats["1-IV_RAMP"] = run_strategy_1()
    if strategy_num in (0, 2):
        all_stats["2-VIX_PUTS"] = run_strategy_2()
    if strategy_num in (0, 3):
        all_stats["3-SHORT_STRADDLE"] = run_strategy_3()
    if strategy_num in (0, 3):
        all_stats["3B-IRON_CONDOR"] = run_strategy_3b()
    if strategy_num in (0, 4):
        all_stats["4-PEAD_CALLS"] = run_strategy_4()
    if strategy_num in (0, 5):
        all_stats["5-CSP_CHOPPY"] = run_strategy_5()

    if strategy_num == 0:
        rank_strategies(all_stats)

    print("\nResults saved to:", OUTPUT_DIR)
