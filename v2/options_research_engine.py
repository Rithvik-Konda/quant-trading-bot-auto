"""
options_research_engine.py — Systematic Options Strategy Sweep
==============================================================
Parameterized backtesting framework that tests ANY combination of:
  - Entry signals (ML rank, regime, IV percentile, earnings, insider, etc.)
  - Options structures (calls, puts, spreads, condors, straddles, etc.)
  - Expirations (7-90 DTE)
  - Strikes (ATM, 10-50 delta, ITM/OTM 10%)
  - Exit rules (profit targets, stop losses, time-based)

Uses cached price data + ThetaData for options pricing.
Results cached to results/options_sweep_results.csv for resumable sweeps.

Run:
  python v2/options_research_engine.py                # full sweep
  python v2/options_research_engine.py --top 20       # show top 20 results
  python v2/options_research_engine.py --resume       # resume interrupted sweep
"""
from __future__ import annotations

import sys, os, json, math, hashlib, itertools, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))
sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2/v2"))

import pandas as pd
import numpy as np

try:
    import config
    WATCHLIST = list(config.WATCHLIST)
    SECTOR_ETFS = getattr(config, "SECTOR_ETFS", {})
except ImportError:
    WATCHLIST = []
    SECTOR_ETFS = {}

ROOT     = Path(os.path.expanduser("~/ai_trading_bot_v2"))
RESULTS  = ROOT / "results"
CACHE    = ROOT / "cache_options"
RESULTS.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

SWEEP_FILE = RESULTS / "options_sweep_results.csv"


# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETER SPACE
# ═══════════════════════════════════════════════════════════════════════════════

# Entry signal filters
ENTRY_SIGNALS = {
    "ml_decile":           [1, 2, 3, 5, 8, 9, 10],  # 1=bottom, 10=top
    "regime":              ["TRENDING_BULL", "CHOPPY", "BEAR", "ANY"],
    "iv_pct":              ["LOW", "MID", "HIGH", "ANY"],  # vs own history
    "earnings_proximity":  ["NONE", "PRE_15", "PRE_5", "POST_5"],
    "sector_leadership":   ["LEADER", "LAGGARD", "ANY"],
}

# Options structures
STRUCTURES = [
    "long_call", "long_put",
    "short_call", "short_put",
    "bull_call_spread", "bear_put_spread",
    "iron_condor", "straddle", "strangle",
]

# Expirations
DTES = [7, 14, 21, 30, 45, 60]

# Strike selection
STRIKES = ["ATM", "D30", "D20", "D10", "OTM_10"]

# Exit rules
EXIT_RULES = [
    {"profit_pct": 0.50, "stop_pct": 1.00, "label": "50P/100S"},
    {"profit_pct": 0.75, "stop_pct": 1.00, "label": "75P/100S"},
    {"profit_pct": 0.80, "stop_pct": 2.00, "label": "80P/200S"},
    {"profit_pct": 1.00, "stop_pct": 2.00, "label": "100P/200S"},
    {"profit_pct": None,  "stop_pct": None,  "label": "HOLD_TO_EXP"},
]


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & SIGNAL RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def load_price(sym: str) -> Optional[pd.DataFrame]:
    for suffix in ["_3650d.csv", "_9999d.csv", "_etf.csv"]:
        p = ROOT / "cache_prices" / f"{sym}{suffix}"
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            return df[["open", "high", "low", "close", "volume"]].dropna()
    return None


_price_cache: Dict[str, pd.DataFrame] = {}

def get_price(sym: str) -> Optional[pd.DataFrame]:
    if sym not in _price_cache:
        _price_cache[sym] = load_price(sym)
    return _price_cache.get(sym)


def load_trades() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "trades_v2.csv")
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    return df


def load_earnings_store() -> Dict[str, pd.DataFrame]:
    store = {}
    d = ROOT / "cache_earnings_streak"
    if not d.exists():
        return store
    for f in d.glob("*.json"):
        sym = f.stem.upper()
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list) and data:
                edf = pd.DataFrame(data)
                if "date" in edf.columns:
                    edf["date"] = pd.to_datetime(edf["date"])
                store[sym] = edf
        except Exception:
            pass
    return store


def reconstruct_regime() -> pd.Series:
    """Daily regime series from SPY/HYG/VIX."""
    spy = get_price("SPY")
    vix = get_price("^VIX")
    hyg = get_price("HYG")
    if spy is None or vix is None:
        return pd.Series(dtype=str)

    common = spy.index.intersection(vix.index)
    if hyg is not None:
        common = common.intersection(hyg.index)
    common = common.sort_values()

    spy_c = spy.loc[common, "close"]
    vix_c = vix.loc[common, "close"]
    hyg_c = hyg.loc[common, "close"] if hyg is not None else pd.Series(0, index=common)

    sma200 = spy_c.rolling(200).mean()
    spy_vs_200 = (spy_c - sma200) / sma200
    spy_20d = spy_c.pct_change(20)
    spy_60d = spy_c.pct_change(60)
    vix_20d = vix_c.pct_change(20)
    hyg_20d = hyg_c.pct_change(20)
    spy_slope = spy_c.rolling(20).mean().pct_change(5)

    regimes = pd.Series("CHOPPY", index=common)
    bear = (spy_vs_200 < -0.02) | (hyg_20d < -0.025) | (vix_c > 30) | (spy_60d < -0.05)
    bull = (spy_slope > 0) & (spy_20d > 0) & (vix_20d < 0.15) & (hyg_20d > -0.01) & (spy_vs_200 > 0.03)
    regimes[bear] = "BEAR"
    regimes[bull & ~bear] = "TRENDING_BULL"

    # 5-day hysteresis
    current = "CHOPPY"
    hold = 0
    out = []
    for d in common:
        raw = regimes.loc[d]
        if raw == current:
            hold = 0
            out.append(current)
        else:
            hold += 1
            if hold >= 5:
                current = raw
                hold = 0
            out.append(current)
    return pd.Series(out, index=common)


def compute_iv_percentile(sym: str, date: pd.Timestamp, vix: pd.Series) -> str:
    """IV percentile bucket using VIX × beta as proxy."""
    px = get_price(sym)
    if px is None:
        return "MID"
    # Realized vol as IV proxy
    close = px["close"].loc[:date]
    if len(close) < 60:
        return "MID"
    rv_20 = close.pct_change().tail(20).std() * np.sqrt(252)
    rv_1y = close.pct_change().tail(252).std() * np.sqrt(252)
    if rv_1y == 0:
        return "MID"
    rv_hist = close.pct_change().rolling(20).std().dropna() * np.sqrt(252)
    if len(rv_hist) < 100:
        return "MID"
    pct = (rv_hist < rv_20).mean()
    if pct < 0.33:
        return "LOW"
    elif pct > 0.67:
        return "HIGH"
    return "MID"


def sector_leadership_rank(sym: str, date: pd.Timestamp) -> str:
    """LEADER if stock's sector ETF outperforms SPY over 20d."""
    stock_to_etf = {}
    for etf, syms in SECTOR_ETFS.items():
        for s in syms:
            stock_to_etf[s] = etf
    etf = stock_to_etf.get(sym)
    if not etf:
        return "ANY"
    etf_px = get_price(etf)
    spy_px = get_price("SPY")
    if etf_px is None or spy_px is None:
        return "ANY"
    try:
        etf_ret = etf_px["close"].asof(date) / etf_px["close"].asof(date - pd.Timedelta(days=30)) - 1
        spy_ret = spy_px["close"].asof(date) / spy_px["close"].asof(date - pd.Timedelta(days=30)) - 1
        return "LEADER" if etf_ret > spy_ret else "LAGGARD"
    except Exception:
        return "ANY"


def earnings_proximity(sym: str, date: pd.Timestamp, earnings_store: dict) -> str:
    """Classify date relative to nearest earnings."""
    if sym not in earnings_store:
        return "NONE"
    edf = earnings_store[sym]
    if len(edf) == 0:
        return "NONE"
    diffs = (edf["date"] - date).dt.days
    future = diffs[diffs >= 0]
    past = diffs[diffs < 0]
    if len(future) > 0 and future.min() <= 15:
        if future.min() <= 5:
            return "PRE_5"
        return "PRE_15"
    if len(past) > 0 and abs(past.max()) <= 5:
        return "POST_5"
    return "NONE"


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIONS PRICING (Black-Scholes approximation)
# ═══════════════════════════════════════════════════════════════════════════════

def _N(x):
    """Standard normal CDF approximation."""
    a = 0.2316419
    b = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
    if x >= 0:
        t = 1.0 / (1.0 + a * x)
        poly = t * (b[0] + t * (b[1] + t * (b[2] + t * (b[3] + t * b[4]))))
        return 1.0 - 0.3989422804 * math.exp(-0.5 * x * x) * poly
    else:
        return 1.0 - _N(-x)


def bs_call(S, K, T, sigma, r=0.04):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N(d1) - K * math.exp(-r * T) * _N(d2)


def bs_put(S, K, T, sigma, r=0.04):
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0)
    return bs_call(S, K, T, sigma, r) - S + K * math.exp(-r * T)


def strike_from_delta(S, sigma, T, target_delta, right="C"):
    """Find strike for a given delta target."""
    if T <= 0 or sigma <= 0:
        return S
    # For calls: delta = N(d1), so d1 = N_inv(delta)
    # Approximate: K = S * exp(-d1 * sigma * sqrt(T) + 0.5 * sigma^2 * T)
    from scipy.stats import norm
    if right == "C":
        d1 = norm.ppf(target_delta)
    else:  # put delta is negative
        d1 = norm.ppf(1 + target_delta)  # target_delta is negative like -0.30
    K = S * math.exp(-d1 * sigma * math.sqrt(T) + 0.5 * sigma**2 * T)
    return round(K, 2)


def get_strike(S, sigma, T, strike_type):
    """Convert strike type to actual strike price."""
    try:
        from scipy.stats import norm
        has_scipy = True
    except ImportError:
        has_scipy = False

    if strike_type == "ATM":
        return round(S, 2)
    elif strike_type == "OTM_10":
        return round(S * 0.90, 2)  # 10% OTM put / call
    elif strike_type.startswith("D") and has_scipy:
        delta = int(strike_type[1:]) / 100.0
        # For puts, return strike below S
        d1 = norm.ppf(1.0 - delta)
        K = S * math.exp(-d1 * sigma * math.sqrt(T) + 0.5 * sigma**2 * T)
        return round(K, 2)
    elif strike_type.startswith("D"):
        # No scipy fallback: approximate
        delta = int(strike_type[1:]) / 100.0
        return round(S * (1 - delta * 0.5 * sigma * math.sqrt(T)), 2)
    return round(S, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY PRICING — compute entry/exit P&L for each structure
# ═══════════════════════════════════════════════════════════════════════════════

def price_structure(structure: str, S_entry: float, S_exit: float,
                    sigma_entry: float, sigma_exit: float,
                    dte_entry: int, dte_exit: int, K: float,
                    wing_width: float = 0) -> dict:
    """
    Compute entry cost, exit value, and P&L for an options structure.
    Returns dict with: entry_cost, exit_value, pnl, max_loss
    """
    T_entry = dte_entry / 365.0
    T_exit  = max(dte_exit, 0) / 365.0

    if structure == "long_call":
        entry = bs_call(S_entry, K, T_entry, sigma_entry)
        exit_v = bs_call(S_exit, K, T_exit, sigma_exit)
        return {"entry_cost": entry, "exit_value": exit_v,
                "pnl": exit_v - entry, "max_loss": entry}

    elif structure == "long_put":
        entry = bs_put(S_entry, K, T_entry, sigma_entry)
        exit_v = bs_put(S_exit, K, T_exit, sigma_exit)
        return {"entry_cost": entry, "exit_value": exit_v,
                "pnl": exit_v - entry, "max_loss": entry}

    elif structure == "short_call":
        entry = bs_call(S_entry, K, T_entry, sigma_entry)
        exit_v = bs_call(S_exit, K, T_exit, sigma_exit)
        return {"entry_cost": -entry, "exit_value": -exit_v,
                "pnl": entry - exit_v, "max_loss": S_entry * 0.5}  # capped for modeling

    elif structure == "short_put":
        entry = bs_put(S_entry, K, T_entry, sigma_entry)
        exit_v = bs_put(S_exit, K, T_exit, sigma_exit)
        return {"entry_cost": -entry, "exit_value": -exit_v,
                "pnl": entry - exit_v, "max_loss": K}

    elif structure == "bull_call_spread":
        K_hi = K * 1.05  # 5% wide
        entry = bs_call(S_entry, K, T_entry, sigma_entry) - bs_call(S_entry, K_hi, T_entry, sigma_entry)
        exit_v = bs_call(S_exit, K, T_exit, sigma_exit) - bs_call(S_exit, K_hi, T_exit, sigma_exit)
        return {"entry_cost": entry, "exit_value": exit_v,
                "pnl": exit_v - entry, "max_loss": entry}

    elif structure == "bear_put_spread":
        K_lo = K * 0.95  # 5% wide
        entry = bs_put(S_entry, K, T_entry, sigma_entry) - bs_put(S_entry, K_lo, T_entry, sigma_entry)
        exit_v = bs_put(S_exit, K, T_exit, sigma_exit) - bs_put(S_exit, K_lo, T_exit, sigma_exit)
        return {"entry_cost": entry, "exit_value": exit_v,
                "pnl": exit_v - entry, "max_loss": entry}

    elif structure == "iron_condor":
        w = max(wing_width, K * 0.05)
        # Sell straddle + buy wings
        sell_call = bs_call(S_entry, K, T_entry, sigma_entry)
        sell_put  = bs_put(S_entry, K, T_entry, sigma_entry)
        buy_call  = bs_call(S_entry, K + w, T_entry, sigma_entry)
        buy_put   = bs_put(S_entry, K - w, T_entry, sigma_entry)
        credit = (sell_call + sell_put) - (buy_call + buy_put)

        exit_sell_call = bs_call(S_exit, K, T_exit, sigma_exit)
        exit_sell_put  = bs_put(S_exit, K, T_exit, sigma_exit)
        exit_buy_call  = bs_call(S_exit, K + w, T_exit, sigma_exit)
        exit_buy_put   = bs_put(S_exit, K - w, T_exit, sigma_exit)
        exit_cost = (exit_sell_call + exit_sell_put) - (exit_buy_call + exit_buy_put)

        pnl = credit - exit_cost
        max_loss = w - credit
        return {"entry_cost": -credit, "exit_value": -exit_cost,
                "pnl": pnl, "max_loss": max(max_loss, 0)}

    elif structure == "straddle":
        entry = bs_call(S_entry, K, T_entry, sigma_entry) + bs_put(S_entry, K, T_entry, sigma_entry)
        exit_v = bs_call(S_exit, K, T_exit, sigma_exit) + bs_put(S_exit, K, T_exit, sigma_exit)
        return {"entry_cost": entry, "exit_value": exit_v,
                "pnl": exit_v - entry, "max_loss": entry}

    elif structure == "strangle":
        K_call = K * 1.05
        K_put  = K * 0.95
        entry = bs_call(S_entry, K_call, T_entry, sigma_entry) + bs_put(S_entry, K_put, T_entry, sigma_entry)
        exit_v = bs_call(S_exit, K_call, T_exit, sigma_exit) + bs_put(S_exit, K_put, T_exit, sigma_exit)
        return {"entry_cost": entry, "exit_value": exit_v,
                "pnl": exit_v - entry, "max_loss": entry}

    return {"entry_cost": 0, "exit_value": 0, "pnl": 0, "max_loss": 1}


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY CONFIG & GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyConfig:
    ml_decile: int
    regime: str
    iv_pct: str
    earnings_prox: str
    sector_lead: str
    structure: str
    dte: int
    strike: str
    exit_label: str
    profit_pct: Optional[float]
    stop_pct: Optional[float]

    @property
    def key(self) -> str:
        return f"{self.structure}|ML{self.ml_decile}|{self.regime}|IV{self.iv_pct}|" \
               f"ER{self.earnings_prox}|SEC{self.sector_lead}|{self.dte}d|K{self.strike}|{self.exit_label}"

    def __hash__(self):
        return hash(self.key)


def generate_strategy_configs() -> List[StrategyConfig]:
    """Generate viable strategy combinations, filtering nonsensical ones."""
    configs = []
    for struct in STRUCTURES:
        for ml in ENTRY_SIGNALS["ml_decile"]:
            for regime in ENTRY_SIGNALS["regime"]:
                for iv in ENTRY_SIGNALS["iv_pct"]:
                    for earn in ENTRY_SIGNALS["earnings_proximity"]:
                        for sec in ENTRY_SIGNALS["sector_leadership"]:
                            for dte in DTES:
                                for strike in STRIKES:
                                    for exit_rule in EXIT_RULES:
                                        # ── Filter nonsensical combos ────
                                        # Long calls in BEAR with low ML
                                        if struct == "long_call" and regime == "BEAR" and ml <= 3:
                                            continue
                                        # Long puts in TRENDING_BULL with high ML
                                        if struct == "long_put" and regime == "TRENDING_BULL" and ml >= 8:
                                            continue
                                        # Short options without clear signal
                                        if struct.startswith("short_") and ml in [4, 5, 6]:
                                            continue
                                        # Iron condors need CHOPPY or ANY
                                        if struct == "iron_condor" and regime == "BEAR":
                                            continue
                                        # Straddles: only near earnings
                                        if struct in ("straddle", "strangle") and earn == "NONE":
                                            continue
                                        # Reduce strike space for spreads
                                        if "spread" in struct and strike not in ("ATM", "D30"):
                                            continue
                                        # Short DTE only for weeklies
                                        if dte == 7 and struct in ("bull_call_spread", "bear_put_spread"):
                                            continue
                                        # Very long DTE only for directional
                                        if dte >= 60 and struct in ("iron_condor", "straddle"):
                                            continue

                                        configs.append(StrategyConfig(
                                            ml_decile=ml, regime=regime,
                                            iv_pct=iv, earnings_prox=earn,
                                            sector_lead=sec, structure=struct,
                                            dte=dte, strike=strike,
                                            exit_label=exit_rule["label"],
                                            profit_pct=exit_rule["profit_pct"],
                                            stop_pct=exit_rule["stop_pct"],
                                        ))
    return configs


# ═══════════════════════════════════════════════════════════════════════════════
#  EVENT GENERATION — find historical dates matching signal conditions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeEvent:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    ml_rank_pct: float
    regime: str
    ann_vol: float
    iv_pct: str
    earn_prox: str
    sector_lead: str


def build_event_universe(trades: pd.DataFrame, regime_series: pd.Series,
                         vix: pd.DataFrame, earnings_store: dict) -> List[TradeEvent]:
    """
    Build universe of trade events from trades_v2.csv (1056 entries with ML ranks)
    enriched with IV percentile, earnings proximity, and sector leadership.
    """
    events = []
    vix_c = vix["close"] if "close" in vix.columns else vix.iloc[:, 0]

    for _, t in trades.iterrows():
        sym = t["symbol"]
        edate = t["entry_date"]
        xdate = t["exit_date"]

        # Enrich with signal data
        iv = compute_iv_percentile(sym, edate, vix_c)
        earn = earnings_proximity(sym, edate, earnings_store)
        sec = sector_leadership_rank(sym, edate)

        events.append(TradeEvent(
            symbol=sym, entry_date=edate, exit_date=xdate,
            entry_price=t["entry_price"], exit_price=t["exit_price"],
            ml_rank_pct=t["ml_rank_pct"], regime=t["regime"],
            ann_vol=t.get("ann_vol", 0.35),
            iv_pct=iv, earn_prox=earn, sector_lead=sec,
        ))
    return events


def filter_events(events: List[TradeEvent], cfg: StrategyConfig) -> List[TradeEvent]:
    """Filter events matching strategy config signal conditions."""
    filtered = []
    for e in events:
        # ML decile
        ml_decile = min(10, max(1, int(e.ml_rank_pct * 10) + 1))
        if cfg.ml_decile != ml_decile and cfg.ml_decile not in (ml_decile,):
            # Allow ±1 decile tolerance for sample size
            if abs(ml_decile - cfg.ml_decile) > 1:
                continue

        # Regime
        if cfg.regime != "ANY" and e.regime != cfg.regime:
            continue

        # IV percentile
        if cfg.iv_pct != "ANY" and e.iv_pct != cfg.iv_pct:
            continue

        # Earnings
        if cfg.earnings_prox != "NONE" and e.earn_prox != cfg.earnings_prox:
            continue
        if cfg.earnings_prox == "NONE" and e.earn_prox != "NONE":
            continue  # explicitly wants non-earnings

        # Sector
        if cfg.sector_lead != "ANY" and e.sector_lead != cfg.sector_lead:
            continue

        filtered.append(e)
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTEST A SINGLE STRATEGY CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_strategy(cfg: StrategyConfig, events: List[TradeEvent]) -> dict:
    """
    Backtest a strategy config against matching events.
    Returns stats dict.
    """
    matching = filter_events(events, cfg)
    if len(matching) < 3:
        return {"key": cfg.key, "n_trades": len(matching), "skip": True}

    returns = []
    pnls = []

    for e in matching:
        S = e.entry_price
        sigma = max(e.ann_vol, 0.15)  # annualized vol
        T = cfg.dte / 365.0

        # Strike
        K = get_strike(S, sigma, T, cfg.strike)

        # IV at exit: model regime-dependent IV change
        # CHOPPY: IV roughly stable. BEAR: IV rises. BULL: IV falls.
        hold_days = min(cfg.dte, (e.exit_date - e.entry_date).days)
        if hold_days <= 0:
            hold_days = min(cfg.dte // 2, 5)

        sigma_exit = sigma
        if e.regime == "BEAR":
            sigma_exit = sigma * 1.15
        elif e.regime == "TRENDING_BULL":
            sigma_exit = sigma * 0.90

        # If near earnings, model IV crush
        if e.earn_prox in ("PRE_5", "PRE_15") and "POST" not in e.earn_prox:
            sigma_exit = sigma * 0.65  # post-earnings crush

        dte_exit = max(cfg.dte - hold_days, 0)

        # Exit price from actual trade data
        S_exit = e.exit_price

        # Apply exit rules via intermediate P&L check
        result = price_structure(
            cfg.structure, S, S_exit, sigma, sigma_exit,
            cfg.dte, dte_exit, K, wing_width=K * 0.08
        )

        entry_cost = result["entry_cost"]
        raw_pnl = result["pnl"]
        max_loss = result["max_loss"]

        # Apply exit rules
        if entry_cost == 0:
            continue
        raw_ret = raw_pnl / abs(entry_cost) if entry_cost != 0 else 0

        # For short structures, return is on credit received
        if cfg.structure.startswith("short_") or cfg.structure == "iron_condor":
            credit = abs(entry_cost)
            if credit == 0:
                continue
            raw_ret = raw_pnl / credit

            # Profit target: close early if we've captured X% of max profit
            if cfg.profit_pct is not None and raw_ret > 0:
                raw_ret = min(raw_ret, cfg.profit_pct)
            # Stop loss
            if cfg.stop_pct is not None and raw_ret < 0:
                raw_ret = max(raw_ret, -cfg.stop_pct)
        else:
            # Long structures
            if cfg.profit_pct is not None and raw_ret > cfg.profit_pct:
                raw_ret = cfg.profit_pct
            if cfg.stop_pct is not None and raw_ret < -cfg.stop_pct:
                raw_ret = -cfg.stop_pct

        returns.append(raw_ret)
        pnls.append(raw_pnl)

    if len(returns) < 3:
        return {"key": cfg.key, "n_trades": len(returns), "skip": True}

    rets = np.array(returns)
    wins = (rets > 0).sum()
    losses = (rets <= 0).sum()
    win_rets = rets[rets > 0]
    loss_rets = rets[rets <= 0]

    # Sharpe: annualize assuming ~4 trades/year per config
    avg_ret = rets.mean()
    std_ret = rets.std()
    trades_per_year = max(1, len(rets) / 7)  # ~7 years of data
    sharpe = (avg_ret * trades_per_year) / (std_ret * np.sqrt(trades_per_year)) if std_ret > 0 else 0

    pf = abs(win_rets.sum() / loss_rets.sum()) if len(loss_rets) > 0 and loss_rets.sum() != 0 else (
        float("inf") if len(win_rets) > 0 else 0)

    # Flag proprietary signals
    uses_proprietary = (cfg.ml_decile >= 8 or cfg.ml_decile <= 2 or
                        cfg.sector_lead != "ANY" or cfg.regime != "ANY")

    return {
        "key": cfg.key,
        "structure": cfg.structure,
        "ml_decile": cfg.ml_decile,
        "regime": cfg.regime,
        "iv_pct": cfg.iv_pct,
        "earnings": cfg.earnings_prox,
        "sector": cfg.sector_lead,
        "dte": cfg.dte,
        "strike": cfg.strike,
        "exit_rule": cfg.exit_label,
        "n_trades": len(rets),
        "win_rate": wins / len(rets),
        "avg_return": avg_ret,
        "median_return": np.median(rets),
        "std_return": std_ret,
        "sharpe": sharpe,
        "profit_factor": min(pf, 99),
        "best_trade": rets.max(),
        "worst_trade": rets.min(),
        "total_pnl": sum(pnls),
        "uses_proprietary": uses_proprietary,
        "skip": False,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SWEEP ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_completed_keys() -> set:
    """Load keys of already-computed strategies for resume capability."""
    if SWEEP_FILE.exists():
        df = pd.read_csv(SWEEP_FILE)
        return set(df["key"].tolist())
    return set()


def save_result(result: dict):
    """Append a single result to the sweep CSV."""
    df = pd.DataFrame([result])
    if SWEEP_FILE.exists():
        df.to_csv(SWEEP_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(SWEEP_FILE, index=False)


def run_sweep(resume: bool = False):
    """Run the full parameter sweep."""
    print("=" * 70)
    print("  OPTIONS RESEARCH ENGINE — Systematic Strategy Sweep")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    trades = load_trades()
    print(f"  Trades: {len(trades)} (2017-2026)")

    print("  Reconstructing regime history...")
    regime_series = reconstruct_regime()
    print(f"  Regime days: {dict(regime_series.value_counts())}")

    vix = get_price("^VIX")
    earnings_store = load_earnings_store()
    print(f"  Earnings: {len(earnings_store)} symbols")
    print(f"  VIX: {len(vix)} days")

    print("\nBuilding enriched event universe...")
    events = build_event_universe(trades, regime_series, vix, earnings_store)
    print(f"  Events: {len(events)}")

    # Generate strategies
    configs = generate_strategy_configs()
    print(f"\nStrategy configs generated: {len(configs)}")

    # Resume support
    completed = load_completed_keys() if resume else set()
    if completed:
        print(f"  Resuming: {len(completed)} already computed")
        configs = [c for c in configs if c.key not in completed]
        print(f"  Remaining: {len(configs)}")

    # Run sweep
    print(f"\nRunning sweep...")
    results = []
    tested = 0
    skipped = 0
    t0 = time.time()

    for i, cfg in enumerate(configs):
        result = backtest_strategy(cfg, events)
        if result.get("skip"):
            skipped += 1
            continue

        tested += 1
        results.append(result)
        save_result(result)

        if tested % 500 == 0:
            elapsed = time.time() - t0
            rate = tested / elapsed if elapsed > 0 else 0
            remaining = (len(configs) - i) / rate if rate > 0 else 0
            print(f"  [{tested:,}/{len(configs):,}] {rate:.0f}/s "
                  f"~{remaining / 60:.0f}min remaining "
                  f"(skipped {skipped:,} low-sample)")

    elapsed = time.time() - t0
    print(f"\nSweep complete: {tested:,} strategies tested, "
          f"{skipped:,} skipped (low sample), {elapsed:.0f}s")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  RANKING & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def report_results(top_n: int = 20, min_trades: int = 10):
    """Load sweep results and print ranked tables."""
    if not SWEEP_FILE.exists():
        print("No sweep results found. Run the sweep first.")
        return

    df = pd.read_csv(SWEEP_FILE)
    df = df[df["skip"] != True] if "skip" in df.columns else df
    print(f"\nTotal strategies evaluated: {len(df)}")

    # Filter statistically meaningful
    meaningful = df[df["n_trades"] >= min_trades].copy()
    print(f"Strategies with >= {min_trades} trades: {len(meaningful)}")

    def print_table(title, subset, sort_col):
        print(f"\n{'=' * 110}")
        print(f"  {title} (n >= {min_trades} trades)")
        print(f"{'=' * 110}")
        top = subset.nlargest(top_n, sort_col)
        print(f"  {'#':>2} {'Structure':12s} {'ML':>3} {'Regime':12s} {'IV':>4} "
              f"{'DTE':>3} {'K':>5} {'Exit':>10} {'N':>4} {'WR':>6} {'AvgRet':>8} "
              f"{'Sharpe':>7} {'PF':>6} {'Prop':>4}")
        print(f"  {'-' * 105}")
        for i, (_, r) in enumerate(top.iterrows(), 1):
            prop = "***" if r.get("uses_proprietary") else ""
            print(f"  {i:2d} {r['structure']:12s} {r['ml_decile']:3.0f} "
                  f"{r.get('regime',''):12s} {r.get('iv_pct',''):>4s} "
                  f"{r['dte']:3.0f} {r.get('strike',''):>5s} {r.get('exit_rule',''):>10s} "
                  f"{r['n_trades']:4.0f} {r['win_rate']:6.0%} {r['avg_return']:+8.1%} "
                  f"{r['sharpe']:+7.2f} {r['profit_factor']:6.2f} {prop}")

    if len(meaningful) > 0:
        print_table("TOP 20 BY SHARPE RATIO", meaningful, "sharpe")
        print_table("TOP 20 BY WIN RATE", meaningful, "win_rate")
        print_table("TOP 20 BY PROFIT FACTOR", meaningful, "profit_factor")

    # Proprietary signal strategies
    prop = meaningful[meaningful["uses_proprietary"] == True] if "uses_proprietary" in meaningful.columns else meaningful
    if len(prop) > 0:
        print_table("TOP 20 PROPRIETARY SIGNAL STRATEGIES", prop, "sharpe")

    # Multi-regime robust strategies
    multi = meaningful.copy()
    multi_keys = multi.groupby(["structure", "ml_decile", "strike", "exit_rule"]).agg(
        regimes=("regime", "nunique"),
        avg_sharpe=("sharpe", "mean"),
        total_trades=("n_trades", "sum"),
    ).reset_index()
    multi_regime = multi_keys[multi_keys["regimes"] >= 2].nlargest(top_n, "avg_sharpe")
    if len(multi_regime) > 0:
        print(f"\n{'=' * 90}")
        print(f"  MULTI-REGIME ROBUST STRATEGIES (work in >= 2 regimes)")
        print(f"{'=' * 90}")
        print(f"  {'Structure':12s} {'ML':>3} {'K':>5} {'Exit':>10} {'Regimes':>8} "
              f"{'AvgSharpe':>10} {'TotalN':>7}")
        print(f"  {'-' * 65}")
        for _, r in multi_regime.iterrows():
            print(f"  {r['structure']:12s} {r['ml_decile']:3.0f} {r['strike']:>5s} "
                  f"{r['exit_rule']:>10s} {r['regimes']:8.0f} "
                  f"{r['avg_sharpe']:+10.2f} {r['total_trades']:7.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: OPTIONS OVERLAY ANALYSIS
#  For every trade in trades_v2.csv, price what would have happened with options
# ═══════════════════════════════════════════════════════════════════════════════

def run_overlay_analysis():
    """
    Retroactively price options overlays on all 1056 equity trades:
    - momentum long  → 30-delta call, 30 DTE
    - meanrev long   → ATM call, 21 DTE
    - short          → 30-delta put, 30 DTE
    Compare options P&L vs equity P&L for each trade.
    """
    print("\n" + "=" * 80)
    print("  PHASE 4: OPTIONS OVERLAY ANALYSIS")
    print("  Pricing options overlays on all 1056 equity trades")
    print("=" * 80)

    trades = load_trades()
    vix_df = get_price("^VIX")
    if vix_df is None:
        print("  ERROR: No VIX data")
        return pd.DataFrame()

    vix_c = vix_df["close"]
    results = []

    for _, t in trades.iterrows():
        sym = t["symbol"]
        S = t["entry_price"]
        S_exit = t["exit_price"]
        engine = t.get("engine", "momentum")
        side = t.get("side", "long")
        hold_days = max(1, (t["exit_date"] - t["entry_date"]).days)
        eq_pnl = t["pnl"]
        eq_ret = (S_exit - S) / S if S > 0 else 0

        # Get VIX at entry as IV proxy
        vix_val = vix_c.asof(t["entry_date"])
        if pd.isna(vix_val):
            continue
        # Stock vol ≈ VIX × beta (use 1.3x for typical stocks)
        sigma = float(vix_val) / 100.0 * 1.3

        # Select structure based on engine/side
        if side == "long" and engine == "momentum":
            # 30-delta call, 30 DTE
            dte = 30
            T = dte / 365.0
            K = get_strike(S, sigma, T, "D30")
            opt_entry = bs_call(S, K, T, sigma)
            T_exit = max(dte - hold_days, 0) / 365.0
            sigma_exit = sigma * (0.95 if t["regime"] == "TRENDING_BULL" else 1.05)
            opt_exit = bs_call(S_exit, K, T_exit, sigma_exit)
            opt_type = "30d_call_30DTE"

        elif side == "long" and engine == "meanrev":
            # ATM call, 21 DTE (shorter for mean reversion)
            dte = 21
            T = dte / 365.0
            K = round(S, 2)
            opt_entry = bs_call(S, K, T, sigma)
            T_exit = max(dte - hold_days, 0) / 365.0
            sigma_exit = sigma * 0.90  # mean reversion = vol compression
            opt_exit = bs_call(S_exit, K, T_exit, sigma_exit)
            opt_type = "ATM_call_21DTE"

        elif side == "short":
            # 30-delta put, 30 DTE (profit from decline)
            dte = 30
            T = dte / 365.0
            K = get_strike(S, sigma, T, "D30")
            # For shorts we BUY puts
            opt_entry = bs_put(S, K, T, sigma)
            T_exit = max(dte - hold_days, 0) / 365.0
            sigma_exit = sigma * 1.15  # vol rises on declines
            opt_exit = bs_put(S_exit, K, T_exit, sigma_exit)
            opt_type = "30d_put_30DTE"
        else:
            continue

        if opt_entry <= 0:
            continue

        opt_pnl = opt_exit - opt_entry
        opt_ret = opt_pnl / opt_entry

        # Leverage: how much capital for same notional?
        # Equity: buy $S per share. Options: pay opt_entry per share.
        leverage = S / opt_entry if opt_entry > 0 else 0

        # Capital-adjusted comparison: $1000 equity vs $1000 options
        capital = 1000
        eq_shares = capital / S if S > 0 else 0
        opt_contracts_equiv = capital / (opt_entry * 100) if opt_entry > 0 else 0
        eq_dollar_pnl = eq_shares * (S_exit - S) if side == "long" else eq_shares * (S - S_exit)
        opt_dollar_pnl = opt_pnl * 100 * opt_contracts_equiv

        results.append({
            "symbol": sym,
            "entry_date": t["entry_date"].strftime("%Y-%m-%d"),
            "engine": engine,
            "side": side,
            "regime": t["regime"],
            "hold_days": hold_days,
            "ml_rank": t["ml_rank_pct"],
            "opt_type": opt_type,
            "strike": K,
            "sigma": sigma,
            "entry_price": S,
            "exit_price": S_exit,
            "equity_return": eq_ret if side == "long" else -eq_ret,
            "options_return": opt_ret,
            "leverage": leverage,
            "eq_pnl_1k": eq_dollar_pnl,
            "opt_pnl_1k": opt_dollar_pnl,
            "options_better": opt_dollar_pnl > eq_dollar_pnl,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No results.")
        return df

    # Analysis
    print(f"\n  Total trades analyzed: {len(df)}")
    for eng in df["engine"].unique():
        for side in df[df["engine"] == eng]["side"].unique():
            sub = df[(df["engine"] == eng) & (df["side"] == side)]
            print(f"\n  {eng.upper()} {side.upper()} ({len(sub)} trades):")
            print(f"    Equity:  avg={sub['equity_return'].mean():+.2%}, "
                  f"win={(sub['equity_return'] > 0).mean():.0%}, "
                  f"$1k P&L avg=${sub['eq_pnl_1k'].mean():+.2f}")
            print(f"    Options: avg={sub['options_return'].mean():+.2%}, "
                  f"win={(sub['options_return'] > 0).mean():.0%}, "
                  f"$1k P&L avg=${sub['opt_pnl_1k'].mean():+.2f}")
            print(f"    Avg leverage: {sub['leverage'].mean():.1f}x")
            print(f"    Options better: {sub['options_better'].mean():.0%} of trades")

    # By regime
    print(f"\n  BY REGIME:")
    for regime in ["TRENDING_BULL", "CHOPPY", "BEAR"]:
        sub = df[df["regime"] == regime]
        if len(sub) == 0:
            continue
        eq_win = (sub["equity_return"] > 0).mean()
        opt_win = (sub["options_return"] > 0).mean()
        print(f"    {regime:15s}  N={len(sub):4d}  "
              f"EQ win={eq_win:.0%} avg={sub['eq_pnl_1k'].mean():+6.1f}  "
              f"OPT win={opt_win:.0%} avg={sub['opt_pnl_1k'].mean():+6.1f}  "
              f"better={sub['options_better'].mean():.0%}")

    # Breakeven analysis
    print(f"\n  BREAKEVEN ANALYSIS:")
    # What equity win rate is needed for options overlay to be net positive?
    for eng_side in [("momentum", "long"), ("meanrev", "long"), ("momentum", "short")]:
        sub = df[(df["engine"] == eng_side[0]) & (df["side"] == eng_side[1])]
        if len(sub) < 10:
            continue
        winners = sub[sub["equity_return"] > 0]
        losers = sub[sub["equity_return"] <= 0]
        if len(winners) == 0 or len(losers) == 0:
            continue
        avg_opt_win = winners["opt_pnl_1k"].mean()
        avg_opt_loss = losers["opt_pnl_1k"].mean()
        if avg_opt_win - avg_opt_loss == 0:
            continue
        breakeven_wr = abs(avg_opt_loss) / (avg_opt_win + abs(avg_opt_loss))
        actual_wr = len(winners) / len(sub)
        edge = actual_wr - breakeven_wr
        print(f"    {eng_side[0]:10s} {eng_side[1]:5s}: "
              f"breakeven WR={breakeven_wr:.0%}, actual WR={actual_wr:.0%}, "
              f"edge={edge:+.0%} {'<-- PROFITABLE' if edge > 0 else '<-- UNPROFITABLE'}")

    df.to_csv(RESULTS / "phase4_options_overlay.csv", index=False)
    print(f"\n  Saved to results/phase4_options_overlay.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Options Strategy Sweep")
    parser.add_argument("--top", type=int, default=20, help="Show top N results")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted sweep")
    parser.add_argument("--report-only", action="store_true", help="Just show results")
    parser.add_argument("--overlay", action="store_true", help="Run Phase 4 overlay analysis")
    parser.add_argument("--min-trades", type=int, default=10, help="Min trades for ranking")
    args = parser.parse_args()

    if args.overlay:
        run_overlay_analysis()
    elif args.report_only:
        report_results(top_n=args.top, min_trades=args.min_trades)
    else:
        run_sweep(resume=args.resume)
        report_results(top_n=args.top, min_trades=args.min_trades)
