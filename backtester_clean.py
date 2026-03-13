from __future__ import annotations

# Suppress sklearn feature-name warnings — we pass numpy arrays intentionally
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import config
from ml_model import compute_features
from risk_manager import Position
from sector_leadership import LeadershipAdapter, apply_leadership_to_snapshots
from strategy_core import (
    RegimeState,
    SignalSnapshot,
    adaptive_stop_pct,
    compute_atr_pct,
    compute_rule_score,
    load_ranker_ensemble,
    market_regime,
    normalize_ohlcv,
    select_top_candidates,
    trend_bullish,
)

CACHE_DIR = "cache_prices"
os.makedirs(CACHE_DIR, exist_ok=True)

CORR_REFRESH_DAYS = 10


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:         str
    entry_date:     str
    exit_date:      str
    entry_price:    float
    exit_price:     float
    qty:            int
    pnl:            float
    reason:         str
    ml_rank_pct:    float
    rule_score:     float
    combined_score: float
    side:           str = "long"   # "long" or "short"


# ─────────────────────────────────────────────────────────────────────────────
# Cache / data loading
# ─────────────────────────────────────────────────────────────────────────────

def cache_path(symbol: str, days: int) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_{days}d.csv")


def fetch_history(symbol: str, days: int, refresh: bool = False) -> pd.DataFrame:
    path = cache_path(symbol, days)
    if (not refresh) and os.path.exists(path):
        df  = pd.read_csv(path, index_col=0)
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        df  = df.loc[~idx.isna()].copy()
        idx = idx[~idx.isna()].tz_convert("UTC").tz_localize(None)
        df.index   = idx
        df.columns = [str(c).lower() for c in df.columns]
        df = normalize_ohlcv(df)
        print(f"  [cache] {symbol}: {len(df)} rows", end="\r", flush=True)
        return df

    years = max(2, int(np.ceil(days / 365)))
    print(f"  [download] {symbol} ({years}y)...      ", end="\r", flush=True)
    df = yf.Ticker(symbol).history(period=f"{years}y", interval="1d", auto_adjust=False)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.columns = [str(c).lower() for c in df.columns]
    for c in ["adj close", "dividends", "stock splits", "capital gains"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    df = normalize_ohlcv(df)
    df.to_csv(path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────────────────

def apply_fill_cost(price: float, qty: int, side: str) -> Tuple[float, float]:
    slip = price * (config.SLIPPAGE_BPS / 10_000)
    fill = price + slip if side == "buy" else price - slip
    comm = min(qty * config.COMMISSION_PER_SHARE, price * qty * config.COMMISSION_MAX_PCT)
    return float(max(fill, 0.01)), float(comm)


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

def calc_stats(equity_curve: pd.Series, trades: List[Trade]) -> dict:
    rets = equity_curve.pct_change().dropna()
    if len(equity_curve) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "trades": len(trades), "win_rate": 0.0}

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years  = max(len(equity_curve) / 252, 1 / 252)
    cagr   = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
    sharpe = 0.0 if rets.std() == 0 else (rets.mean() / rets.std()) * np.sqrt(252)

    roll_max = equity_curve.cummax()
    dd       = equity_curve / roll_max - 1
    max_dd   = float(dd.min()) if len(dd) else 0.0

    wins     = [t for t in trades if t.pnl > 0]
    win_rate = 0.0 if not trades else len(wins) / len(trades)

    # Per-year breakdown
    annual: Dict[int, dict] = {}
    for yr in equity_curve.index.year.unique():
        eq_yr = equity_curve[equity_curve.index.year == yr]
        if len(eq_yr) < 2:
            continue
        tr_yr  = [t for t in trades if str(t.exit_date)[:4] == str(yr)]
        yr_ret = eq_yr.iloc[-1] / eq_yr.iloc[0] - 1
        yr_rets = eq_yr.pct_change().dropna()
        yr_sh   = 0.0 if yr_rets.std() == 0 else (yr_rets.mean() / yr_rets.std()) * np.sqrt(252)
        yr_dd   = float((eq_yr / eq_yr.cummax() - 1).min())
        wins_yr = sum(1 for t in tr_yr if t.pnl > 0)
        annual[yr] = {
            "cagr":         float(yr_ret),
            "sharpe":       float(yr_sh),
            "max_drawdown": float(yr_dd),
            "n_trades":     len(tr_yr),
            "win_rate":     wins_yr / len(tr_yr) if tr_yr else 0.0,
        }

    return {
        "total_return": float(total_return),
        "cagr":         float(cagr),
        "sharpe":       float(sharpe),
        "max_drawdown": float(max_dd),
        "trades":       len(trades),
        "n_trades":     len(trades),
        "win_rate":     float(win_rate),
        "annual":       annual,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised rule score
# ─────────────────────────────────────────────────────────────────────────────

def compute_rule_score_vectorised(df: pd.DataFrame) -> pd.Series:
    if len(df) < 220:
        return pd.Series(dtype=float)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    above_200 = (close > ma200).astype(float)
    above_50  = (close > ma50).astype(float)
    golden    = (ma50 > ma200).astype(float)

    ret5  = close.pct_change(5)
    ret20 = close.pct_change(20)
    ret60 = close.pct_change(60)
    ret5_c  = ret5.clip(-0.30, 0.30)
    ret20_c = ret20.clip(-0.40, 0.40)
    ret60_c = ret60.clip(-0.60, 0.60)

    vol_ma20  = volume.rolling(20).mean()
    vol_ratio = (volume / vol_ma20.replace(0, np.nan)).fillna(1.0).clip(0, 5)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14  = tr.rolling(14).mean()
    high20 = close.rolling(20).max().shift(1)
    breakout = ((close - high20) / atr14.replace(0, np.nan)).clip(-2, 2).fillna(0)
    accel    = (ret5 - ret20 / 4).clip(-0.15, 0.15).fillna(0)

    raw = (
        0.20 * above_200 +
        0.10 * above_50  +
        0.10 * golden    +
        0.25 * ret20_c / 0.40 +
        0.15 * ret60_c / 0.60 +
        0.10 * (vol_ratio - 1.0) / 2.0 +
        0.05 * breakout / 2.0 +
        0.05 * accel / 0.15
    )

    scores = raw.clip(-1.0, 1.0).fillna(0.0)
    scores.iloc[:219] = 0.0
    return scores


def build_rule_store_fast(
    symbols: List[str],
    prices: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    rule_store: Dict[str, pd.Series] = {}
    total = len(symbols)
    for i, s in enumerate(symbols):
        print(f"  [rule] {i+1}/{total} {s}              ", end="\r", flush=True)
        try:
            rule_store[s] = compute_rule_score_vectorised(prices[s])
        except Exception:
            rule_store[s] = pd.Series(dtype=float)
    print(f"  [rule] done ({total} symbols)            ", flush=True)
    return rule_store


# ─────────────────────────────────────────────────────────────────────────────
# Pre-stacked feature matrix
# ─────────────────────────────────────────────────────────────────────────────

class FeatureMatrix:
    def __init__(
        self,
        symbols: List[str],
        feature_store: Dict[str, pd.DataFrame],
        feat_cols: List[str],
        all_dates: pd.DatetimeIndex,
    ):
        self.symbols   = symbols
        self.feat_cols = feat_cols
        self.all_dates = all_dates
        self.n_dates   = len(all_dates)
        self.n_syms    = len(symbols)
        self.n_feats   = len(feat_cols)
        self.sym_idx   = {s: i for i, s in enumerate(symbols)}
        self.date_idx  = {d: i for i, d in enumerate(all_dates)}

        print("  [matrix] pre-stacking feature arrays...", flush=True)
        self.matrix = np.full((self.n_syms, self.n_dates, self.n_feats),
                              fill_value=np.nan, dtype=np.float32)
        self.valid  = np.zeros((self.n_syms, self.n_dates), dtype=bool)

        for si, sym in enumerate(symbols):
            feat_df = feature_store.get(sym)
            if feat_df is None or len(feat_df) == 0:
                continue
            for col in feat_cols:
                base = col[:-8] if col.endswith("_cs_rank") else col
                if base not in feat_df.columns:
                    continue
                col_j = feat_cols.index(col)
                for date, val in feat_df[base].items():
                    di = self.date_idx.get(date)
                    if di is None:
                        continue
                    if np.isfinite(val):
                        self.matrix[si, di, col_j] = val
                        self.valid[si, di] = True
        print("  [matrix] done.                         ", flush=True)

    def get_panel(
        self,
        date: pd.Timestamp,
        available_symbols: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        di = self.date_idx.get(date)
        if di is None:
            return np.zeros((0, self.n_feats), dtype=np.float32), []
        valid_syms = [s for s in available_symbols if s in self.sym_idx]
        if not valid_syms:
            return np.zeros((0, self.n_feats), dtype=np.float32), []

        X = np.zeros((len(valid_syms), self.n_feats), dtype=np.float32)
        for row_i, sym in enumerate(valid_syms):
            si = self.sym_idx[sym]
            valid_dates = np.where(self.valid[si, :di+1])[0]
            if len(valid_dates) == 0:
                continue
            row = self.matrix[si, valid_dates[-1], :]
            X[row_i] = np.where(np.isfinite(row), row, 0.0)

        cs_rank_cols = [j for j, c in enumerate(self.feat_cols) if c.endswith("_cs_rank")]
        for j_rank in cs_rank_cols:
            base_name = self.feat_cols[j_rank][:-8]
            if base_name in self.feat_cols:
                j_base   = self.feat_cols.index(base_name)
                col_vals = X[:, j_base]
                if col_vals.std() > 0:
                    X[:, j_rank] = pd.Series(col_vals).rank(pct=True).values.astype(np.float32)
                else:
                    X[:, j_rank] = 0.5
        return X, valid_syms


# ─────────────────────────────────────────────────────────────────────────────
# Batch ML scoring
# ─────────────────────────────────────────────────────────────────────────────

def batch_ml_scores_fast(
    X: np.ndarray,
    syms: List[str],
    rankers: dict,
    feat_cols: List[str],
) -> Dict[str, float]:
    if len(syms) == 0 or X.shape[0] == 0:
        return {}

    def score_one(bundle: dict) -> Dict[str, float]:
        col_map  = {c: i for i, c in enumerate(feat_cols)}
        feat_idx = [col_map[c] for c in bundle["features"] if c in col_map]
        if not feat_idx:
            return {}
        Xb = np.nan_to_num(X[:, feat_idx].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        Xs = bundle["scaler"].transform(Xb)
        return {s: float(p) for s, p in zip(syms, bundle["model"].predict(Xs))}

    s3 = score_one(rankers[3])
    s5 = score_one(rankers[5])
    s7 = score_one(rankers[7])

    ensemble: Dict[str, float] = {}
    for sym in sorted(set(s3) | set(s5) | set(s7)):
        vals, wts = [], []
        if sym in s3: vals.append(s3[sym]); wts.append(0.25)
        if sym in s5: vals.append(s5[sym]); wts.append(0.35)
        if sym in s7: vals.append(s7[sym]); wts.append(0.40)
        if wts:
            ensemble[sym] = float(np.average(vals, weights=wts))

    if ensemble:
        return pd.Series(ensemble).rank(pct=True).to_dict()
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Correlation cache
# ─────────────────────────────────────────────────────────────────────────────

class CorrMatrixCache:
    def __init__(self, refresh_days: int = CORR_REFRESH_DAYS):
        self.refresh_days = refresh_days
        self._matrix: Optional[pd.DataFrame] = None
        self._day_count: int = 0

    def get(
        self,
        date: pd.Timestamp,
        available_symbols: List[str],
        hist: Dict[str, pd.DataFrame],
        lookback: int,
    ) -> pd.DataFrame:
        self._day_count += 1
        if self._matrix is None or self._day_count % self.refresh_days == 0:
            self._matrix = self._compute(date, available_symbols, hist, lookback)
        return self._matrix

    @staticmethod
    def _compute(
        date: pd.Timestamp,
        available_symbols: List[str],
        hist: Dict[str, pd.DataFrame],
        lookback: int,
    ) -> pd.DataFrame:
        frames = []
        for s in available_symbols:
            try:
                df  = hist[s].loc[:date]
                if len(df) < lookback + 2:
                    continue
                ret = df["close"].pct_change().tail(lookback).rename(s)
                frames.append(ret)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).dropna(how="all").corr()


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builders
# ─────────────────────────────────────────────────────────────────────────────

def stop_pct_for_symbol(df: pd.DataFrame) -> float:
    atr_pct = compute_atr_pct(df, config.ATR_PERIOD)
    if config.USE_ATR_STOPS:
        raw = atr_pct * config.ATR_STOP_MULTIPLIER
        return float(np.clip(raw, config.STOP_MIN_PCT, config.STOP_MAX_PCT))
    return float(config.FIXED_STOP_LOSS_PCT)


def build_fast_snapshots(
    date: pd.Timestamp,
    available_symbols: List[str],
    hist: Dict[str, pd.DataFrame],
    rule_store: Dict[str, pd.Series],
    ml_scores: Dict[str, float],
) -> Dict[str, SignalSnapshot]:
    snapshots: Dict[str, SignalSnapshot] = {}
    if not ml_scores:
        return snapshots

    for symbol in available_symbols:
        if symbol not in ml_scores:
            continue
        try:
            df = hist[symbol].loc[:date]
            if len(df) < 260:
                continue
            rule     = float(rule_store[symbol].get(date, 0.0))
            ml       = float(ml_scores[symbol])
            bull     = trend_bullish(df)
            atr_pct  = compute_atr_pct(df, config.ATR_PERIOD)
            stop_pct = stop_pct_for_symbol(df)
            combined = 0.75 * ml + 0.15 * ml + 0.10 * rule

            snapshots[symbol] = SignalSnapshot(
                symbol=symbol,
                rule_score=rule,
                ml_score=ml,
                ml_rank_pct=ml,
                combined_score=combined,
                trend_bullish=bull,
                stop_pct=stop_pct,
                atr_pct=atr_pct,
            )
        except Exception:
            continue
    return snapshots


def conviction_multiplier(snap: SignalSnapshot) -> float:
    if snap.ml_rank_pct >= 0.97: return 1.15
    if snap.ml_rank_pct >= 0.93: return 1.00
    if snap.ml_rank_pct >= 0.88: return 0.85
    return 0.70


def current_gross_exposure(
    close_prices: Dict[str, float],
    positions: Dict[str, Position],
) -> float:
    return sum(close_prices.get(s, p.entry_price) * p.qty for s, p in positions.items())


def build_symbol_to_sector_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for etf, syms in config.SECTOR_ETFS.items():
        for s in syms:
            mapping[s] = etf
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Short position tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShortPosition:
    symbol:        str
    qty:           int
    entry_price:   float
    entry_time:    str
    stop_price:    float   # exit if price rises above this
    highest_loss:  float = 0.0  # tracks max adverse move

    def age_days(self, current_date) -> int:
        from datetime import datetime
        try:
            entry = datetime.strptime(self.entry_time, "%Y-%m-%d")
            if hasattr(current_date, "to_pydatetime"):
                current_date = current_date.to_pydatetime()
            return max(0, (current_date - entry).days)
        except Exception:
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# Regime-driven position counts
# ─────────────────────────────────────────────────────────────────────────────

def max_longs_shorts_for_regime(regime: RegimeState) -> Tuple[int, int]:
    """
    Derive max long and short position counts from continuous regime score.
    All limits from config — no hardcoding.
    """
    max_pos       = config.MAX_POSITIONS
    max_short_pos = getattr(config, "SHORT_MAX_POSITIONS", 4)
    score         = regime.score

    bull_thresh    = getattr(config, "REGIME_BULL_THRESHOLD",    0.60)
    neutral_thresh = getattr(config, "REGIME_NEUTRAL_THRESHOLD", 0.40)

    if score >= bull_thresh:
        # Full long, no shorts
        return max_pos, 0
    elif score >= neutral_thresh:
        # Blend linearly: at bull_thresh → (max_pos, 0); at neutral_thresh → (max_pos//2, max_short_pos//2)
        t       = (score - neutral_thresh) / (bull_thresh - neutral_thresh)
        n_long  = max(1, int(round(max_pos * (0.5 + 0.5 * t))))
        n_short = max(0, int(round(max_short_pos * (1 - t) * 0.5)))
        return n_long, n_short
    else:
        # Bear: blend from neutral to full short
        bear_floor = 0.0
        t       = (score - bear_floor) / (neutral_thresh - bear_floor) if neutral_thresh > bear_floor else 0.0
        n_long  = max(0, int(round(max_pos * t * 0.5)))
        n_short = max(1, int(round(max_short_pos * (0.5 + 0.5 * (1 - t)))))
        return n_long, n_short


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest loop
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(days: int = 3650, refresh_cache: bool = False) -> Tuple:
    print("\n[start] long/short ML backtest with continuous regime", flush=True)
    print(f"        universe={len(config.WATCHLIST)} symbols  days={days}\n", flush=True)

    # ── 1. Load price data ────────────────────────────────────────────────
    symbols     = list(config.WATCHLIST)
    all_symbols = symbols + [config.BENCHMARK_SYMBOL]
    hist: Dict[str, pd.DataFrame] = {}
    for s in all_symbols:
        try:
            hist[s] = fetch_history(s, days, refresh=refresh_cache)
        except Exception as e:
            print(f"  [error] {s}: {e}", flush=True)
            hist[s] = pd.DataFrame()
    hist = {k: v for k, v in hist.items() if len(v) > 0}
    print(f"\n[info] loaded {len(hist)} symbols", flush=True)

    if config.BENCHMARK_SYMBOL not in hist:
        raise RuntimeError("Missing benchmark history")

    spy              = hist[config.BENCHMARK_SYMBOL]
    prices_by_symbol = {k: v for k, v in hist.items() if k != config.BENCHMARK_SYMBOL}
    symbols          = list(prices_by_symbol.keys())

    if not symbols:
        raise RuntimeError("No tradable histories loaded")

    # ── 2. Trade date index ───────────────────────────────────────────────
    all_trade_dates = pd.DatetimeIndex([])
    for s in symbols:
        all_trade_dates = all_trade_dates.union(prices_by_symbol[s].index)
    all_trade_dates = all_trade_dates.intersection(spy.index).sort_values()

    print(f"[info] {len(all_trade_dates)} trading dates available", flush=True)
    if len(all_trade_dates) < 300:
        raise RuntimeError(f"Not enough dates: {len(all_trade_dates)}")

    # ── 3. ML ensemble ────────────────────────────────────────────────────
    print("[prep] loading ML ensemble...", flush=True)
    rankers = load_ranker_ensemble()
    feat_cols_union = sorted(set(
        list(rankers[3]["features"]) +
        list(rankers[5]["features"]) +
        list(rankers[7]["features"])
    ))
    print(f"[ok]   ensemble loaded ({len(feat_cols_union)} features)", flush=True)

    # ── 4. Feature computation ────────────────────────────────────────────
    print("[prep] computing features...", flush=True)
    feature_store: Dict[str, pd.DataFrame] = {}
    for i, s in enumerate(symbols):
        print(f"  [feat] {i+1}/{len(symbols)} {s}         ", end="\r", flush=True)
        try:
            f = compute_features(prices_by_symbol[s], symbol=s)
            feature_store[s] = f.replace([np.inf, -np.inf], np.nan) if f is not None and len(f) else pd.DataFrame()
        except Exception:
            feature_store[s] = pd.DataFrame()
    print(f"\n[ok]   features done ({len(feature_store)} symbols)", flush=True)

    # ── 5. Rule scores ────────────────────────────────────────────────────
    print("[prep] computing rule scores (vectorised)...", flush=True)
    rule_store = build_rule_store_fast(symbols, prices_by_symbol)

    # ── 6. Feature matrix ─────────────────────────────────────────────────
    print("[prep] pre-stacking feature matrix...", flush=True)
    feat_matrix = FeatureMatrix(
        symbols=symbols, feature_store=feature_store,
        feat_cols=feat_cols_union, all_dates=all_trade_dates,
    )

    corr_cache = CorrMatrixCache(refresh_days=CORR_REFRESH_DAYS)

    # ── 7. Leadership adapter ─────────────────────────────────────────────
    leadership_enabled = getattr(config, "ADAPTIVE_LEADERSHIP_ENABLED", True)
    leadership_adapter: Optional[LeadershipAdapter] = None
    if leadership_enabled:
        leadership_adapter = LeadershipAdapter(
            update_frequency_days=getattr(config, "LEADERSHIP_UPDATE_FREQ_DAYS", 5),
            leadership_threshold=getattr(config, "LEADERSHIP_THRESHOLD", 0.62),
            top_n_leaders=getattr(config, "LEADERSHIP_TOP_N", 4),
        )
        print("[ok]   adaptive leadership adapter ready", flush=True)

    # ── 8. Main loop ──────────────────────────────────────────────────────
    cash                              = config.INITIAL_CAPITAL
    long_positions:  Dict[str, Position]      = {}
    short_positions: Dict[str, ShortPosition] = {}
    entry_meta:      Dict[str, dict]          = {}
    trades:          List[Trade]              = []
    equity                                    = []
    last_regime_exit_date: Optional[pd.Timestamp] = None
    short_stop_dates: Dict[str, pd.Timestamp]     = {}   # symbol → date of last short_stop
    entries_today:    int                         = 0    # new long entries on current date
    entries_date:     Optional[pd.Timestamp]      = None # date entries_today was last reset
    _regime_candidate: Optional[str]     = None
    _regime_candidate_days: int          = 0
    _current_regime_label:  str          = "bull"
    short_stop_dates: Dict[str, pd.Timestamp] = {}  # symbol → last short_stop date

    lookback    = 260
    total_steps = len(all_trade_dates) - 1 - lookback

    print(f"\n[run] starting main loop ({total_steps} steps)...\n", flush=True)

    for step_idx, i in enumerate(range(lookback, len(all_trade_dates) - 1), start=1):
        date      = all_trade_dates[i]
        next_date = all_trade_dates[i + 1]

        if step_idx % 50 == 0 or step_idx == total_steps:
            pct = step_idx / total_steps * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            n_l = len(long_positions)
            n_s = len(short_positions)
            print(f"\r  [{bar}] {pct:5.1f}%  step {step_idx}/{total_steps}  "
                  f"L={n_l} S={n_s}  trades={len(trades)}  cash=${cash:,.0f}  ",
                  end="", flush=True)

        # ── Prices for today ──────────────────────────────────────────────
        available_symbols  = []
        close_prices:      Dict[str, float] = {}
        high_prices:       Dict[str, float] = {}
        low_prices:        Dict[str, float] = {}
        open_next_prices:  Dict[str, float] = {}

        for s in symbols:
            full_df = prices_by_symbol[s]
            if date not in full_df.index:
                continue
            df_now = full_df.loc[:date]
            if len(df_now) < lookback:
                continue
            available_symbols.append(s)
            close_prices[s] = float(df_now["close"].iloc[-1])
            high_prices[s]  = float(df_now["high"].iloc[-1])
            low_prices[s]   = float(df_now["low"].iloc[-1])
            future = full_df.loc[full_df.index > date]
            if len(future) > 0:
                open_next_prices[s] = float(future.iloc[0]["open"])

        if len(available_symbols) < 2:
            port_val = _portfolio_value(cash, close_prices, long_positions, short_positions)
            equity.append((date, port_val))
            continue

        spy_window = spy.loc[:date]
        if len(spy_window) < lookback:
            continue

        # ── Continuous regime ─────────────────────────────────────────────
        regime = market_regime(spy_window, universe_dfs={
            s: prices_by_symbol[s].loc[:date] for s in available_symbols[:40]
        })

        # ── Hysteresis: regime must hold N days before flipping ───────────
        # Stops same-day entry/exit churn during volatile oscillations.
        min_days = getattr(config, "REGIME_MIN_DAYS_BEFORE_FLIP", 5)
        new_label = "bull" if regime.is_bull else ("bear" if regime.is_bear else "neutral")
        if new_label != _current_regime_label:
            if _regime_candidate != new_label:
                _regime_candidate      = new_label
                _regime_candidate_days = 1
            else:
                _regime_candidate_days += 1

            if _regime_candidate_days >= min_days:
                _current_regime_label  = new_label
                _regime_candidate      = None
                _regime_candidate_days = 0
        else:
            _regime_candidate      = None
            _regime_candidate_days = 0

        # Override regime classification with stable (hysteresis-filtered) label
        stable_is_bull    = _current_regime_label == "bull"
        stable_is_bear    = _current_regime_label == "bear"
        stable_is_neutral = _current_regime_label == "neutral"

        # ── ML scoring (runs before exits — available for future exit logic) ──────
        X, valid_syms = feat_matrix.get_panel(date, available_symbols)
        if X.shape[0] == 0:
            port_val = _portfolio_value(cash, close_prices, long_positions, short_positions)
            equity.append((date, port_val))
            continue
        ml_scores = batch_ml_scores_fast(X, valid_syms, rankers, feat_cols_union)

        # ── Exit: longs ───────────────────────────────────────────────────
        for s in list(long_positions.keys()):
            pos   = long_positions[s]
            close = close_prices.get(s)
            low   = low_prices.get(s)
            if close is None or low is None:
                continue

            pos.update_high(close)
            # Adaptive stop — widens in volatile stocks, tightens in profit
            df_s     = prices_by_symbol[s].loc[:date]
            stop_pct = adaptive_stop_pct(df_s, pos.entry_price, close, side="long")
            stop_px  = pos.entry_price * (1 - stop_pct)
            # Also respect any trailing high
            trail_stop = pos.highest_price * (1 - stop_pct)
            stop_px    = max(stop_px, trail_stop)

            hold_d      = pos.age_days(pd.Timestamp(date).to_pydatetime())
            exit_reason = exit_ref = None

            if low <= stop_px:
                exit_reason = "stop"
                exit_ref    = stop_px
            elif close >= pos.entry_price * (1 + config.TAKE_PROFIT_PCT):
                exit_reason = "take_profit"
                exit_ref    = close
            elif hold_d >= config.MAX_HOLD_DAYS:
                exit_reason = "max_hold"
                exit_ref    = close

            # Regime flip: if stable regime is bear, exit longs
            if exit_reason is None and stable_is_bear:
                exit_reason = "regime_exit"
                exit_ref    = close

            if exit_reason is not None:
                fill, comm = apply_fill_cost(exit_ref, pos.qty, "sell")
                pnl  = (fill - pos.entry_price) * pos.qty - comm
                cash += fill * pos.qty - comm
                meta  = entry_meta.get(s, {})
                trades.append(Trade(
                    symbol=s, entry_date=pos.entry_time,
                    exit_date=str(date.date()), entry_price=pos.entry_price,
                    exit_price=fill, qty=pos.qty, pnl=pnl, reason=exit_reason,
                    ml_rank_pct=float(meta.get("ml_rank_pct", 0.0)),
                    rule_score=float(meta.get("rule_score", 0.0)),
                    combined_score=float(meta.get("combined_score", 0.0)),
                    side="long",
                ))
                if exit_reason == "regime_exit":
                    last_regime_exit_date = date
                del long_positions[s]
                entry_meta.pop(s, None)

        # ── Exit: shorts ──────────────────────────────────────────────────
        short_tp_pct   = getattr(config, "SHORT_TAKE_PROFIT_PCT", 0.20)
        short_stop_pct = getattr(config, "SHORT_STOP_PCT",        0.08)
        short_hold     = getattr(config, "SHORT_MAX_HOLD_DAYS",   15)

        for s in list(short_positions.keys()):
            pos   = short_positions[s]
            close = close_prices.get(s)
            high  = high_prices.get(s)
            if close is None or high is None:
                continue

            pnl_pct      = (pos.entry_price - close) / pos.entry_price
            exit_reason  = exit_ref = None

            if pnl_pct >= short_tp_pct:
                exit_reason = "short_take_profit"
                exit_ref    = close
            elif high >= pos.entry_price * (1 + short_stop_pct):
                exit_reason = "short_stop"
                exit_ref    = pos.entry_price * (1 + short_stop_pct)
            elif pos.age_days(pd.Timestamp(date).to_pydatetime()) >= short_hold:
                exit_reason = "short_max_hold"
                exit_ref    = close

            # Regime flip to stable bull — cover shorts
            if exit_reason is None and stable_is_bull:
                exit_reason = "regime_cover"
                exit_ref    = close

            if exit_reason is not None:
                # Cover short: buy back at exit_ref
                fill, comm = apply_fill_cost(exit_ref, pos.qty, "buy")
                pnl        = (pos.entry_price - fill) * pos.qty - comm
                cash       += pnl  # net P&L back to cash (margin account)
                meta        = entry_meta.get(f"short_{s}", {})
                trades.append(Trade(
                    symbol=s, entry_date=pos.entry_time,
                    exit_date=str(date.date()), entry_price=pos.entry_price,
                    exit_price=fill, qty=pos.qty, pnl=pnl, reason=exit_reason,
                    ml_rank_pct=float(meta.get("ml_rank_pct", 0.0)),
                    rule_score=float(meta.get("rule_score", 0.0)),
                    combined_score=float(meta.get("combined_score", 0.0)),
                    side="short",
                ))
                # Record short_stop date to block re-entry on same symbol
                if exit_reason == "short_stop":
                    short_stop_dates[s] = date
                del short_positions[s]
                entry_meta.pop(f"short_{s}", None)

        # ── Hard circuit breakers (halts new entries only, exits already processed) ──
        if config.ENABLE_REGIME_FILTER and (regime.spy_crash or regime.vol_halt):
            port_val = _portfolio_value(cash, close_prices, long_positions, short_positions)
            equity.append((date, port_val))
            continue

        # ── Snapshots ─────────────────────────────────────────────────────
        snapshots = build_fast_snapshots(
            date=date, available_symbols=available_symbols,
            hist=prices_by_symbol, rule_store=rule_store, ml_scores=ml_scores,
        )

        if leadership_adapter is not None:
            leadership_adapter.update(date, prices_by_symbol)
            snapshots = apply_leadership_to_snapshots(
                snapshots=snapshots, adapter=leadership_adapter,
                soft_filter=True,
                min_multiplier=getattr(config, "LEADERSHIP_MIN_MULTIPLIER", 0.60),
                boost_cap=getattr(config, "LEADERSHIP_BOOST_CAP", 1.50),
            )

        symbol_to_df_sel = {s: prices_by_symbol[s].loc[:date] for s in snapshots}
        corr_matrix = corr_cache.get(
            date=date, available_symbols=list(snapshots.keys()),
            hist=prices_by_symbol, lookback=config.CORRELATION_LOOKBACK_DAYS,
        )

        # ── Regime-driven position limits ─────────────────────────────────
        # Use stable (hysteresis-filtered) regime for position sizing
        from dataclasses import replace as _dc_replace
        stable_regime = _dc_replace(
            regime,
            is_bull=stable_is_bull,
            is_bear=stable_is_bear,
            is_neutral=stable_is_neutral,
            position_scalar=(1.0 if stable_is_bull else (0.80 if stable_is_neutral else 0.60)),
        )
        max_longs, max_shorts = max_longs_shorts_for_regime(stable_regime)

        # ── Long entries ──────────────────────────────────────────────────
        cooldown_days = getattr(config, "REGIME_EXIT_COOLDOWN_DAYS", 10)
        in_cooldown   = (
            last_regime_exit_date is not None
            and (date - last_regime_exit_date).days < cooldown_days
        )
        # VIX spike: halt new longs even if regime hasn't flipped yet
        vix_halted = regime.vix_spike

        if max_longs > 0 and not in_cooldown and not vix_halted:
            long_candidates = select_top_candidates(
                snapshots=snapshots, symbol_to_df=symbol_to_df_sel,
                current_positions={}, max_names=max_longs,
                corr_matrix=corr_matrix, side="long",
            )

            leader_score = long_candidates[0].combined_score if long_candidates else 0.0

            for idx, snap in enumerate(long_candidates):
                s = snap.symbol
                if s in long_positions or s not in open_next_prices:
                    continue
                if len(long_positions) >= max_longs:
                    break

                # Staggered entries — max N new longs per day
                max_per_day = getattr(config, "MAX_NEW_ENTRIES_PER_DAY", 2)
                if entries_date != date:
                    entries_today = 0
                    entries_date  = date
                if entries_today >= max_per_day:
                    break

                px         = open_next_prices[s]
                stop_pct   = snap.stop_pct
                scalar     = regime.position_scalar
                conviction = conviction_multiplier(snap)

                gross_exp = current_gross_exposure(close_prices, long_positions)
                remaining = max(0.0, config.INITIAL_CAPITAL * config.MAX_TOTAL_EXPOSURE - gross_exp)
                if remaining <= 0:
                    break

                risk_budget    = config.INITIAL_CAPITAL * config.RISK_PER_TRADE * scalar * conviction
                risk_per_share = px * stop_pct
                qty_risk       = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
                max_dollars    = min(
                    config.INITIAL_CAPITAL * config.MAX_POSITION_WEIGHT * scalar * conviction,
                    config.MAX_POSITION_DOLLARS * conviction,
                    cash, remaining,
                )
                qty = min(qty_risk, int(max_dollars / px) if px > 0 else 0)
                if qty <= 0:
                    continue

                fill, comm = apply_fill_cost(px, qty, "buy")
                cost = fill * qty + comm
                if cost > cash:
                    continue

                cash -= cost
                entries_today += 1
                long_positions[s] = Position(
                    symbol=s, qty=qty, entry_price=fill,
                    entry_time=str(next_date.date()),
                    stop_pct=stop_pct,
                    initial_stop=fill * (1 - stop_pct),
                    highest_price=fill, add_count=0,
                )
                entry_meta[s] = {
                    "ml_rank_pct":    snap.ml_rank_pct,
                    "rule_score":     snap.rule_score,
                    "combined_score": snap.combined_score,
                }

        # ── Short entries ─────────────────────────────────────────────────
        if max_shorts > 0:
            # Exclude symbols already in long book
            short_snapshots = {k: v for k, v in snapshots.items() if k not in long_positions}
            short_sym_df    = {s: prices_by_symbol[s].loc[:date] for s in short_snapshots}

            short_candidates = select_top_candidates(
                snapshots=short_snapshots, symbol_to_df=short_sym_df,
                current_positions={}, max_names=max_shorts,
                corr_matrix=corr_matrix, side="short",
            )

            short_risk     = getattr(config, "SHORT_RISK_PER_TRADE", 0.020)
            short_max_dol  = getattr(config, "SHORT_MAX_DOLLARS", 35_000)
            short_stop_pct = getattr(config, "SHORT_STOP_PCT", 0.08)

            for snap in short_candidates:
                s = snap.symbol
                if s in short_positions or s not in open_next_prices:
                    continue
                if len(short_positions) >= max_shorts:
                    break

                # Short reentry cooldown — if this symbol was stopped out recently, skip
                short_cooldown = getattr(config, "SHORT_REENTRY_COOLDOWN_DAYS", 60)
                last_stop = short_stop_dates.get(s)
                if last_stop is not None and (date - last_stop).days < short_cooldown:
                    continue

                px         = open_next_prices[s]
                scalar     = regime.position_scalar
                conviction = 1.0 - conviction_multiplier(snap) / 1.15  # inverse conviction for shorts

                risk_budget    = config.INITIAL_CAPITAL * short_risk * scalar
                risk_per_share = px * short_stop_pct
                qty_risk       = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
                qty            = min(qty_risk, int(short_max_dol / px) if px > 0 else 0)
                if qty <= 0:
                    continue

                # Margin requirement: typically 50% — use 60% to be conservative
                margin_required = px * qty * 0.60
                if margin_required > cash:
                    continue

                stop_px = px * (1 + short_stop_pct)
                short_positions[s] = ShortPosition(
                    symbol=s, qty=qty, entry_price=px,
                    entry_time=str(next_date.date()),
                    stop_price=stop_px,
                )
                entry_meta[f"short_{s}"] = {
                    "ml_rank_pct":    snap.ml_rank_pct,
                    "rule_score":     snap.rule_score,
                    "combined_score": snap.combined_score,
                }

        # ── Debug print ───────────────────────────────────────────────────
        if step_idx % 250 == 0:
            print()
            label = 'bull' if stable_is_bull else ('bear' if stable_is_bear else 'neutral')
            vix_str = " VIX_SPIKE" if regime.vix_spike else ""
            print(f"  [regime] {date.date()} score={regime.score:.2f} "
                  f"({label}){vix_str} "
                  f"breadth={regime.breadth:.0%} vol={regime.realized_vol:.0%} "
                  f"L={max_longs} S={max_shorts}", flush=True)

        port_val = _portfolio_value(cash, close_prices, long_positions, short_positions)
        equity.append((date, port_val))

    print("\n", flush=True)

    # ── Results ───────────────────────────────────────────────────────────
    equity_curve = pd.Series(
        data=[v for _, v in equity],
        index=pd.to_datetime([d for d, _ in equity]),
        name="equity",
    )

    stats = calc_stats(equity_curve, trades)
    _print_results(stats, trades)
    return equity_curve, trades, stats


def _portfolio_value(
    cash: float,
    close_prices: Dict[str, float],
    long_positions: Dict[str, Position],
    short_positions: Dict[str, ShortPosition],
) -> float:
    long_val  = sum(close_prices.get(s, p.entry_price) * p.qty for s, p in long_positions.items())
    # Short P&L: (entry - current) * qty
    short_pnl = sum((p.entry_price - close_prices.get(s, p.entry_price)) * p.qty
                    for s, p in short_positions.items())
    return cash + long_val + short_pnl


def _print_results(stats: dict, trades: List[Trade]) -> None:
    print("\n" + "═" * 50)
    print("  BACKTEST RESULTS")
    print("═" * 50)
    print(f"  Total Return : {stats['total_return']:>10.2%}")
    print(f"  CAGR         : {stats['cagr']:>10.2%}")
    print(f"  Sharpe       : {stats['sharpe']:>10.2f}")
    print(f"  Max Drawdown : {stats['max_drawdown']:>10.2%}")
    print(f"  Trades       : {stats['trades']:>10}")
    print(f"  Win Rate     : {stats['win_rate']:>10.2%}")

    if trades:
        pnl      = pd.Series([t.pnl for t in trades])
        trade_df = pd.DataFrame([t.__dict__ for t in trades])
        print(f"  Avg Trade PnL: {pnl.mean():>10.2f}")
        print(f"  Avg ML Rank% : {trade_df['ml_rank_pct'].mean():>10.2%}")

        long_trades  = trade_df[trade_df["side"] == "long"]
        short_trades = trade_df[trade_df["side"] == "short"]
        if len(long_trades):
            lw = (long_trades["pnl"] > 0).mean()
            print(f"  Long trades  : {len(long_trades):>4}  win={lw:.0%}  "
                  f"avg_pnl={long_trades['pnl'].mean():.1f}")
        if len(short_trades):
            sw = (short_trades["pnl"] > 0).mean()
            print(f"  Short trades : {len(short_trades):>4}  win={sw:.0%}  "
                  f"avg_pnl={short_trades['pnl'].mean():.1f}")

        print("\n  Exit breakdown:")
        for reason, grp in trade_df.groupby("reason"):
            wins = (grp["pnl"] > 0).sum()
            print(f"    {reason:<20} {len(grp):>4} trades  "
                  f"win={wins/len(grp):.0%}  avg_pnl={grp['pnl'].mean():.1f}")

        print("\n  Top symbols by total PnL:")
        sym_pnl = trade_df.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
        for sym, total_pnl in sym_pnl.head(10).items():
            n = (trade_df["symbol"] == sym).sum()
            print(f"    {sym:<8} {total_pnl:>10.1f}  ({n} trades)")

    print("═" * 50 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Legacy sector rotation (kept for A/B testing)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_top_sectors_legacy(
    date: pd.Timestamp,
    hist: Dict[str, pd.DataFrame],
) -> Set[str]:
    lookback      = int(config.SECTOR_LOOKBACK_DAYS)
    sector_scores = []
    for etf, sector_symbols in config.SECTOR_ETFS.items():
        rets = []
        for s in sector_symbols:
            if s not in hist:
                continue
            df = hist[s].loc[:date]
            if len(df) < lookback + 1:
                continue
            try:
                ret = float(df["close"].iloc[-1] / df["close"].iloc[-lookback] - 1.0)
                rets.append(ret)
            except Exception:
                continue
        if len(rets) >= 3:
            sector_scores.append((etf, float(np.median(rets))))
    if not sector_scores:
        return set()
    sector_scores.sort(key=lambda x: x[1], reverse=True)
    return {etf for etf, _ in sector_scores[:int(config.TOP_SECTORS_TO_TRADE)]}


# ─────────────────────────────────────────────────────────────────────────────
# OOS validation
# ─────────────────────────────────────────────────────────────────────────────

def run_oos_test() -> None:
    print("=" * 60)
    print("  OUT-OF-SAMPLE VALIDATION")
    print("  IN-SAMPLE : 2015-2021  (model trained here)")
    print("  OOS       : 2022-2025  (blind — model never saw this)")
    print("=" * 60 + "\n")

    equity_full, trades_full, _ = run_backtest(days=3650)
    if equity_full is None or len(equity_full) < 2:
        print("[error] Not enough data")
        return

    oos_cutoff  = pd.Timestamp("2022-01-01")
    eq_in       = equity_full[equity_full.index <  oos_cutoff]
    eq_oos      = equity_full[equity_full.index >= oos_cutoff]
    eq_oos_norm = eq_oos / eq_oos.iloc[0] * config.INITIAL_CAPITAL
    trades_in   = [t for t in trades_full if str(t.exit_date) <  "2022-01-01"]
    trades_oos  = [t for t in trades_full if str(t.exit_date) >= "2022-01-01"]
    stats_in    = calc_stats(eq_in,       trades_in)
    stats_oos   = calc_stats(eq_oos_norm, trades_oos)

    def print_result(label: str, stats: dict) -> None:
        print(f"\n  -- {label} --")
        print(f"  CAGR         : {stats['cagr']:>10.2%}")
        print(f"  Sharpe       : {stats['sharpe']:>10.2f}")
        print(f"  Max Drawdown : {stats['max_drawdown']:>10.2%}")
        print(f"  Win Rate     : {stats['win_rate']:>10.2%}")
        print(f"  Trades       : {stats['trades']:>10d}")
        if stats.get("annual"):
            print(f"\n  {'Year':<6} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8}")
            print("  " + "-" * 48)
            for yr in sorted(stats["annual"]):
                a = stats["annual"][yr]
                print(f"  {yr:<6} {a['cagr']:>7.1%}  {a['sharpe']:>6.2f}  "
                      f"{a['max_drawdown']:>7.1%}  {a['n_trades']:>6d}  {a['win_rate']:>7.1%}")

    print_result("IN-SAMPLE 2015-2021",     stats_in)
    print_result("OUT-OF-SAMPLE 2022-2025", stats_oos)

    oos_sh   = stats_oos["sharpe"]
    oos_cagr = stats_oos["cagr"]
    oos_dd   = abs(stats_oos["max_drawdown"])

    if   oos_cagr >= 0.20 and oos_dd <= 0.18 and oos_sh >= 1.5:
        verdict = "STRONG EDGE — OOS holds up. Ready to scale."
    elif oos_cagr >= 0.12 and oos_dd <= 0.25 and oos_sh >= 1.0:
        verdict = "MODERATE EDGE — Some decay. Investigate weak years."
    elif oos_cagr >= 0.00:
        verdict = "WEAK / UNCERTAIN — Edge is thin. Do not scale yet."
    else:
        verdict = "NO EDGE — OOS is negative."

    print(f"\n{'='*60}\n  VERDICT: {verdict}\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Long/short ML backtest with continuous regime")
    parser.add_argument("--days",          type=int,  default=3650)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-leadership", action="store_true")
    parser.add_argument("--corr-refresh",  type=int,  default=10)
    parser.add_argument("--oos",           action="store_true",
                        help="Out-of-sample test: IS 2015-2021 vs OOS 2022-2025")
    args = parser.parse_args()

    if args.no_leadership:
        config.ADAPTIVE_LEADERSHIP_ENABLED = False
    if args.corr_refresh:
        global CORR_REFRESH_DAYS
        CORR_REFRESH_DAYS = args.corr_refresh

    if args.oos:
        run_oos_test()
    else:
        run_backtest(days=args.days, refresh_cache=args.refresh_cache)


if __name__ == "__main__":
    main()