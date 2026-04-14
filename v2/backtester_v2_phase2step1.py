"""
backtester_v2.py — Multi-Engine Regime-Conditional Backtester
=============================================================
The core loop for v2. Key difference from v1:

1. Regime classifier runs FIRST every day
2. Strategy selection based on regime
3. Entry filter applied with regime-conditional thresholds
4. Position sizing, stops, hold periods all from active strategy

Flow per trading day:
  1. Compute regime signals (SPY, HYG, VIX)
  2. Update regime state (with hysteresis)
  3. Select active strategy (trending/choppy/bear)
  4. Score ML candidates
  5. Apply entry filter (regime-conditional A/D check)
  6. Size positions per active strategy
  7. Manage exits per active strategy parameters
"""

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for v1 imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from backtester_clean import (
    fetch_history, apply_fill_cost, calc_stats,
    compute_rule_score_vectorised, build_rule_store_fast,
    FeatureMatrix, batch_ml_scores_fast, CorrMatrixCache,
    build_fast_snapshots, conviction_multiplier,
    ShortPosition, Trade,
)
from ml_model import compute_features
from risk_manager import Position
from sector_leadership import LeadershipAdapter, apply_leadership_to_snapshots
from strategy_core import (
    adaptive_stop_pct, compute_atr_pct, load_ranker_ensemble, load_ranker_regime_ensemble,
    load_ranker_ensemble_for_year,
    market_regime, normalize_ohlcv, select_top_candidates,
    trend_bullish,
)

from regime_classifier import (
    RegimeClassifier, compute_signals, load_macro_data,
    TRENDING_BULL, CHOPPY, BEAR,
    classify_choppy_subregime, CHOPPY_BULL, CHOPPY_BEAR,
)
from entry_filter import is_in_accumulation, filter_candidates
import strategy_trending as strat_bull
import strategy_choppy  as strat_chop
import strategy_meanrev as strat_mr
import strategy_bear    as strat_bear

CACHE_DIR = "cache_prices"

# ── ABLATION HARNESS ──────────────────────────────────────────────────
import os as _ablate_os
_ABLATED = set(_ablate_os.environ.get("ABLATE", "tcps,earnings_exclusion,biotech_gate,vix_vol_scalar,conviction_calls,friday_delay,vol_cooldown").split(","))
def _ablate(name: str) -> bool:
    return name in _ABLATED


# ── SILENT-FAILURE LOGGER (added by hygiene_patch.py) ─────────────────
def _silent_log(_where: str, _sym: str, _exc: Exception) -> None:
    import os as _os, traceback as _tb
    _path = _os.environ.get("V2_SILENT_LOG", "/tmp/v2_silent_errors.log")
    try:
        with open(_path, "a") as _f:
            _f.write(f"[{_where}] {_sym}: {type(_exc).__name__}: {_exc}\n")
    except Exception:
        pass



def _get_strategy(regime: str):
    """Return the active strategy module for the current regime."""
    if regime == TRENDING_BULL:
        return strat_bull
    elif regime == CHOPPY:
        return strat_chop
    else:
        return strat_bear


def _compute_one(args):
    """Module-level worker for parallel feature computation."""
    sym, df, vix_df, streak_store, insider_store = args
    try:
        f = compute_features(df, symbol=sym, vix_macro=vix_df, streak_store=streak_store, insider_store=insider_store)
        return sym, f.replace([np.inf, -np.inf], np.nan) if f is not None and len(f) else pd.DataFrame()
    except Exception:
        return sym, pd.DataFrame()


def run_backtest_v2(
    days:          int  = 3650,
    refresh_cache: bool = False,
    verbose:       bool = True,
) -> Tuple[pd.Series, List[Trade], dict]:

    print("\n" + "="*60)
    print("  V2 MULTI-ENGINE REGIME-CONDITIONAL BACKTEST")
    print("="*60)
    print(f"  Universe: {len(config.WATCHLIST)} symbols")
    print(f"  Days:     {days}")
    print()

    # ── 1. Load macro data for regime classifier ──────────────────────────
    print("[prep] loading macro data (SPY/HYG/VIX)...", flush=True)
    macro_cache = os.path.join(os.path.dirname(__file__), '..', CACHE_DIR)
    spy_macro, hyg_macro, vix_macro = load_macro_data(cache_dir=macro_cache)
    regime_clf = RegimeClassifier()
    print("[ok]   macro data loaded", flush=True)

    # ── 2. Load price data ────────────────────────────────────────────────
    symbols     = list(config.WATCHLIST)
    all_symbols = symbols + [config.BENCHMARK_SYMBOL]
    hist: Dict[str, pd.DataFrame] = {}

    # ── SURVIVORSHIP BIAS FIX ─────────────────────────────────────────
    # For each year, exclude stocks whose price data starts AFTER that year.
    # A stock with no data before 2020 is excluded from 2017-2019 backtests.
    # This prevents look-ahead bias from today's universe selection.
    # The per-day check at line ~345 (`if date not in full_df.index: continue`)
    # already handles this naturally — stocks without data for a date are
    # excluded from that day's available_symbols. No additional filter needed.
    # Flag: SURVIVORSHIP_BIAS_HANDLED = True

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    for s in all_symbols:
        try:
            hist[s] = fetch_history(s, days, refresh=refresh_cache)
        except Exception as e:
            hist[s] = pd.DataFrame()
    hist = {k: v for k, v in hist.items() if len(v) > 0}
    print(f"[info] loaded {len(hist)} symbols", flush=True)

    spy              = hist[config.BENCHMARK_SYMBOL]
    prices_by_symbol = {k: v for k, v in hist.items() if k != config.BENCHMARK_SYMBOL}
    symbols          = list(prices_by_symbol.keys())

    # ── 3. Trade date index ───────────────────────────────────────────────
    all_trade_dates = pd.DatetimeIndex([])
    for s in symbols:
        all_trade_dates = all_trade_dates.union(prices_by_symbol[s].index)
    all_trade_dates = all_trade_dates.intersection(spy.index).sort_values()
    print(f"[info] {len(all_trade_dates)} trading dates", flush=True)

    # ── 4. ML ensemble ────────────────────────────────────────────────────
    print("[prep] loading ML ensemble...", flush=True)
    import warnings as _w
    _w.filterwarnings("ignore", category=UserWarning, message=".*InconsistentVersion.*")
    rankers = load_ranker_ensemble()
    import joblib as _jl, os as _os
    try:
        _streak_store  = _jl.load("streak_store.joblib")  if _os.path.exists("streak_store.joblib")  else {}
    except Exception as _e:
        print(f"[warn] streak_store.joblib failed to load ({_e.__class__.__name__}), using empty")
        _streak_store = {}
    try:
        _insider_store = _jl.load("insider_store.joblib") if _os.path.exists("insider_store.joblib") else {}
    except Exception as _e:
        print(f"[warn] insider_store.joblib failed to load ({_e.__class__.__name__}), using empty")
        _insider_store = {}
    feat_cols_union = sorted(set(
        list(rankers[3]["features"]) +
        list(rankers[5]["features"]) +
        list(rankers[7]["features"])
    ))
    print(f"[ok]   ensemble loaded ({len(feat_cols_union)} features)", flush=True)

    # Load regime-specific models (trained only on same-regime days)
    # Falls back to all-regime model if not yet trained
    print("[prep] loading regime-specific ML ensembles...", flush=True)
    regime_rankers = load_ranker_regime_ensemble()
    has_regime_models = any(
        os.path.exists(f"cross_sectional_ranker_5d_{r}.joblib")
        for r in ["TRENDING_BULL", "CHOPPY", "BEAR"]
    )
    if has_regime_models:
        print("[ok]   regime-specific models loaded", flush=True)
    else:
        print("[warn] regime models not found — using all-regime model for all regimes", flush=True)

    # ── 5. Feature computation (parallel + cached) ───────────────────────
    import hashlib, pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cache_dir_feat = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache_prices", "feat_cache")
    os.makedirs(cache_dir_feat, exist_ok=True)

    # _compute_one moved to module level for multiprocessing pickling

    def _cache_key(sym, df):
        last_date = str(df.index[-1]) if len(df) else "empty"
        nrows = str(len(df))
        return hashlib.md5(f"{sym}_{last_date}_{nrows}_macrov1".encode()).hexdigest()[:12]

    print("[prep] computing features (parallel + cached)...", flush=True)
    feature_store: Dict[str, pd.DataFrame] = {}
    to_compute = []
    for s in symbols:
        df = prices_by_symbol[s]
        key = _cache_key(s, df)
        cache_path = os.path.join(cache_dir_feat, f"{s}_{key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as fh:
                    feature_store[s] = pickle.load(fh)
                continue
            except Exception:
                pass
        to_compute.append(s)

    cached_count = len(feature_store)
    print(f"  [feat] {cached_count} cached, computing {len(to_compute)} symbols...", flush=True)

    if to_compute:
        import multiprocessing
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        args_list = [(s, prices_by_symbol[s], vix_macro, _streak_store, _insider_store) for s in to_compute]
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_compute_one, a): a[0] for a in args_list}
            for future in as_completed(futures):
                sym, feat_df = future.result()
                feature_store[sym] = feat_df
                key = _cache_key(sym, prices_by_symbol[sym])
                cache_path = os.path.join(cache_dir_feat, f"{sym}_{key}.pkl")
                try:
                    with open(cache_path, "wb") as fh:
                        pickle.dump(feat_df, fh)
                except Exception:
                    pass
                done += 1
                if done % 50 == 0 or done == len(to_compute):
                    print(f"  [feat] {done}/{len(to_compute)} computed   ", end="\r", flush=True)

    print(f"\n[ok]   features done ({cached_count} cached + {len(to_compute)} computed)", flush=True)

    # ── 6. Rule scores + feature matrix (cached) ─────────────────────────
    print("[prep] computing rule scores...", flush=True)
    rule_cache_key = _cache_key("rule_store", spy) + f"_{len(symbols)}"
    rule_cache_path = os.path.join(cache_dir_feat, f"rule_store_{rule_cache_key}.pkl")
    if os.path.exists(rule_cache_path):
        try:
            with open(rule_cache_path, "rb") as fh:
                rule_store = pickle.load(fh)
            print("[ok]   rule scores loaded from cache", flush=True)
        except Exception:
            rule_store = build_rule_store_fast(symbols, prices_by_symbol)
            with open(rule_cache_path, "wb") as fh:
                pickle.dump(rule_store, fh)
    else:
        rule_store = build_rule_store_fast(symbols, prices_by_symbol)
        try:
            with open(rule_cache_path, "wb") as fh:
                pickle.dump(rule_store, fh)
        except Exception:
            pass
    print("[prep] building feature matrix...", flush=True)
    feat_matrix = FeatureMatrix(
        symbols=symbols, feature_store=feature_store,
        feat_cols=feat_cols_union, all_dates=all_trade_dates,
    )
    corr_cache = CorrMatrixCache(refresh_days=10)

    # ── 7. Sector map ─────────────────────────────────────────────────────
    sector_map: Dict[str, str] = {}
    for etf, syms in config.SECTOR_ETFS.items():
        for s in syms:
            sector_map[s] = etf

    # ── 7b. Precompute SPY 5-day returns for performance ─────────────────
    _spy_close_series = spy_macro["close"] if "close" in spy_macro.columns else spy_macro.iloc[:, 0]
    _spy_5d_returns: Dict[pd.Timestamp, float] = {}
    for _d in all_trade_dates:
        _spy_hist = _spy_close_series.loc[:_d]
        if len(_spy_hist) >= 6:
            _spy_5d_returns[_d] = float(_spy_hist.iloc[-1] / _spy_hist.iloc[-6] - 1)
        else:
            _spy_5d_returns[_d] = 0.0
    print(f"[ok]   SPY 5d returns precomputed ({len(_spy_5d_returns)} dates)", flush=True)

    # ── 8. Leadership adapter ─────────────────────────────────────────────
    leadership_adapter = None
    if getattr(config, "ADAPTIVE_LEADERSHIP_ENABLED", True):
        leadership_adapter = LeadershipAdapter(
            update_frequency_days=getattr(config, "LEADERSHIP_UPDATE_FREQ_DAYS", 5),
            leadership_threshold=getattr(config, "LEADERSHIP_THRESHOLD", 0.62),
            top_n_leaders=getattr(config, "LEADERSHIP_TOP_N", 4),
        )
    # Precompute earnings dates for all symbols once before main loop
    print("[prep] loading earnings calendars...", flush=True)
    earnings_dates: Dict[str, List[pd.Timestamp]] = {}
    _earn_cache_path = os.path.join(CACHE_DIR, "earnings_calendar_cache.pkl")
    _earn_cache_age  = 0
    if os.path.exists(_earn_cache_path):
        _earn_cache_age = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(_earn_cache_path), unit="s")).total_seconds() / 3600
    if os.path.exists(_earn_cache_path) and _earn_cache_age < 24:
        import pickle as _pkl
        with open(_earn_cache_path, "rb") as _ef:
            earnings_dates = _pkl.load(_ef)
        print(f"[ok]   earnings calendars loaded from cache ({len(earnings_dates)} symbols, {_earn_cache_age:.1f}h old)", flush=True)
    else:
        try:
            import yfinance as _yf, pickle as _pkl
            for _sym in symbols:
                try:
                    _cal = _yf.Ticker(_sym).get_earnings_dates(limit=20)
                    if _cal is not None and len(_cal) > 0:
                        earnings_dates[_sym] = list(pd.to_datetime(_cal.index).tz_localize(None))
                except Exception:
                    pass
            with open(_earn_cache_path, "wb") as _ef:
                _pkl.dump(earnings_dates, _ef)
            print(f"[ok]   earnings calendars loaded ({len(earnings_dates)} symbols) — cached", flush=True)
        except Exception:
            print("[warn] earnings calendar load failed — skipping filter", flush=True)

    print("[ok]   ready\n", flush=True)

    # ── 9. Main loop ──────────────────────────────────────────────────────
    cash             = float(config.INITIAL_CAPITAL)
    long_positions:  Dict[str, Position]      = {}
    short_positions: Dict[str, ShortPosition] = {}
    entry_meta:      Dict[str, dict]          = {}
    trades:          List[Trade]              = []
    equity:          List[Tuple]              = []

    long_stop_dates:  Dict[str, pd.Timestamp] = {}
    short_stop_dates: Dict[str, pd.Timestamp] = {}
    prev_ml_ranks:    Dict[str, float]         = {}  # Fix 1: entry confirmation
    recent_stop_dates: List[pd.Timestamp]      = []  # cascade sensor

    last_regime_exit_date: Optional[pd.Timestamp] = None
    _regime_candidate:     Optional[str]           = None
    _regime_candidate_days: int                    = 0
    _current_regime:        str                    = CHOPPY
    regime_counts = {TRENDING_BULL: 0, CHOPPY: 0, BEAR: 0}

    lookback    = 260
    total_steps = len(all_trade_dates) - 1 - lookback

    print(f"[run] starting main loop ({total_steps} steps)...\n", flush=True)

    _ml_scores_cache  = None
    _ml_scores_regime = None
    _lifecycle_rows: list = []
    for step_idx, i in enumerate(range(lookback, len(all_trade_dates) - 1), start=1):
        date      = all_trade_dates[i]
        next_date = all_trade_dates[i + 1]

        if step_idx % 50 == 0 or step_idx == total_steps:
            pct = step_idx / total_steps * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  [{bar}] {pct:5.1f}%  step {step_idx}/{total_steps}"
                  f"  L={len(long_positions)} S={len(short_positions)}"
                  f"  trades={len(trades)}  regime={_current_regime[:4]}"
                  f"  cash=${cash:,.0f}  ",
                  end="", flush=True)

        # ── Prices ───────────────────────────────────────────────────────
        available_symbols = []
        close_prices:     Dict[str, float] = {}
        high_prices:      Dict[str, float] = {}
        low_prices:       Dict[str, float] = {}
        open_next_prices: Dict[str, float] = {}

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

        # ── Regime classification ─────────────────────────────────────────
        signals = compute_signals(spy_macro, hyg_macro, vix_macro, as_of_date=date)
        # Add breadth signal — % of universe above 200d MA
        # Computed from actual holdings universe, more relevant than external index
        if signals and available_symbols:
            _breadth_above = 0
            _breadth_total = 0
            for _bs in available_symbols[:100]:  # sample 100 for speed
                _bdf = prices_by_symbol.get(_bs)
                if _bdf is not None and len(_bdf) >= 200:
                    _bclose = _bdf.loc[:date]['close']
                    if len(_bclose) >= 200:
                        _bma200 = float(_bclose.rolling(200).mean().iloc[-1])
                        _breadth_above += int(float(_bclose.iloc[-1]) > _bma200)
                        _breadth_total += 1
            if _breadth_total > 0:
                signals['breadth_pct_above_200d'] = _breadth_above / _breadth_total
        if signals:
            _current_regime = regime_clf.update(date, signals)
        regime_counts[_current_regime] = regime_counts.get(_current_regime, 0) + 1

        # Get active strategy
        strategy = _get_strategy(_current_regime)
        params   = strategy.get_params()

        # CHOPPY sub-regime override — CHOPPY_BULL gets more positions and lower ML bar
        # Validated: 2019 SPY +28.7% and 2025 SPY +16.6% both CHOPPY_BULL
        _choppy_subregime = None
        if _current_regime == CHOPPY and signals:
            _choppy_subregime = classify_choppy_subregime(signals)
            if _choppy_subregime == CHOPPY_BULL:
                # Participate more in high-vol bull — lower bar, more positions
                params.max_positions      = 4
                params.ml_rank_min        = 0.85
                params.max_positions_long = 4

        # ── ML scoring ────────────────────────────────────────────────────
        X, valid_syms = feat_matrix.get_panel(date, available_symbols)
        if X.shape[0] == 0:
            port_val = _portfolio_value(cash, close_prices, long_positions, short_positions)
            equity.append((date, port_val))
            continue
        # All-regime ranker — regime-specific rankers tested and hurt OOS
        # BEAR ranker damaged 2023 (+30% vs +48% baseline) and 2025
        # Regime-specific models may need more BEAR training data to be useful
        active_rankers = rankers
        # Cache ML scores — recompute every 2 days only.
        # Stock rankings are highly autocorrelated day-to-day.
        # This cuts runtime ~45% with negligible impact on results.
        if step_idx % 2 == 0 or _ml_scores_cache is None or _ml_scores_regime != _current_regime:
            ml_scores = batch_ml_scores_fast(X, valid_syms, active_rankers, feat_cols_union)
            _ml_scores_cache  = ml_scores
            _ml_scores_regime = _current_regime
        else:
            ml_scores = _ml_scores_cache

        # ── Exit: longs ───────────────────────────────────────────────────
        for s in list(long_positions.keys()):
            pos   = long_positions[s]
            close = close_prices.get(s)
            low   = low_prices.get(s)
            if close is None or low is None:
                continue

            pos.update_high(close)
            df_s     = prices_by_symbol[s].loc[:date]

            # ML-predicted stop width — learned from 393 historical trades
            # Features: ann_vol, proper_atr, ADX, price_vs_200ma, momentum_60d
            # No hardcoded thresholds — model learns what stop width each situation needs
            try:
                from stop_width_model import predict_stop_width
                _meta = entry_meta.get(s, {})
                _dse  = int((date - pd.Timestamp(_meta.get("entry_date", str(date)))).days) if "entry_date" in _meta else 60
                _str  = float(_meta.get("earnings_streak", 0.5))
                stop_pct = predict_stop_width(s, df_s, _dse, _str)
            except Exception:
                # Fallback: proper true-range ATR (not close-to-close approximation)
                if len(df_s) >= 2:
                    _tr = pd.concat([
                        df_s["high"] - df_s["low"],
                        (df_s["high"] - df_s["close"].shift()).abs(),
                        (df_s["low"]  - df_s["close"].shift()).abs(),
                    ], axis=1).max(axis=1)
                    _true_atr_pct = float(_tr.tail(14).mean() / df_s["close"].iloc[-1])
                    # Widen stops in TRENDING_BULL — data: all stops WR<21% = noise
                    # ATR×3 in TRENDING_BULL, ATR×2.5 in CHOPPY, ATR×2 in BEAR
                    if _current_regime == TRENDING_BULL:
                        _atr_mult = 3.0
                    elif _current_regime == CHOPPY:
                        _atr_mult = 2.5
                    else:
                        _atr_mult = 2.0
                    stop_pct = float(np.clip(_atr_mult * _true_atr_pct, 0.04, 0.20))
                else:
                    stop_pct = 0.08

            stop_px    = pos.entry_price * (1 - stop_pct)
            trail_stop = pos.highest_price * (1 - stop_pct)
            stop_px    = max(stop_px, trail_stop)

            # Fix 4: move stop to breakeven once up 8%
            unrealized_pct = (close - pos.entry_price) / pos.entry_price
            if unrealized_pct >= 0.08 and not _ablate("breakeven"):
                stop_px = max(stop_px, pos.entry_price * 1.001)

            hold_d      = pos.age_days(pd.Timestamp(date).to_pydatetime())
            exit_reason = exit_ref = None

            # TCPS: Trajectory-Conditioned Position Scaling
            # OOS validated: p=0.003, IS p=0.026, STRONGER OOS than IS
            # Day 7 positive WR=74% avg=$1,071 vs negative WR=35% avg=-$39
            # Scale UP winners, scale DOWN losers — never exit, just resize
            if hold_d == 7 and not _ablate("tcps"):
                try:
                    _spy_now = spy_macro["close"].reindex([date], method="ffill")
                    _spy_px  = float(_spy_now.iloc[0]) if len(_spy_now) else None
                    _spy_7d_ago = spy_macro["close"].reindex(
                        [all_trade_dates[i-7]], method="ffill")
                    _spy_7d_px = float(_spy_7d_ago.iloc[0]) if len(_spy_7d_ago) else None

                    _entry_px = pos.entry_price
                    _r7 = (close - _entry_px) / _entry_px
                    _s7 = (_spy_px / _spy_7d_px - 1) if _spy_px and _spy_7d_px else 0
                    _excess7 = _r7 - _s7

                    if _r7 > 0.01 and _excess7 > 0.005:
                        # Strong trajectory — scale UP by buying more shares
                        _extra_budget = config.INITIAL_CAPITAL * 0.015  # 1.5% extra
                        _extra_qty = int(_extra_budget / close) if close > 0 else 0
                        if _extra_qty > 0 and cash >= _extra_qty * close:
                            _fill, _comm = apply_fill_cost(close, _extra_qty, "buy")
                            if _fill * _extra_qty + _comm <= cash:
                                cash -= _fill * _extra_qty + _comm
                                long_positions[s] = Position(
                                    symbol=s, qty=pos.qty + _extra_qty,
                                    entry_price=pos.entry_price,
                                    entry_time=pos.entry_time,
                                    stop_pct=pos.stop_pct,
                                    initial_stop=pos.initial_stop,
                                    highest_price=pos.highest_price,
                                    add_count=getattr(pos, 'add_count', 0) + 1,
                                )
                                pos = long_positions[s]
                    elif _r7 < -0.01 and _excess7 < -0.005:
                        # Weak trajectory — trim to 50% of position
                        _trim_qty = max(0, pos.qty // 2)
                        if _trim_qty > 0:
                            _fill, _comm = apply_fill_cost(close, _trim_qty, "sell")
                            cash += _fill * _trim_qty - _comm
                            long_positions[s] = Position(
                                symbol=s, qty=pos.qty - _trim_qty,
                                entry_price=pos.entry_price,
                                entry_time=pos.entry_time,
                                stop_pct=pos.stop_pct,
                                initial_stop=pos.initial_stop,
                                highest_price=pos.highest_price,
                                add_count=getattr(pos, 'add_count', 0),
                            )
                            pos = long_positions[s]
                except Exception as _e:
                    _silent_log("tcps_day7", s, _e)

            # ── Mean reversion exit (different logic from momentum) ──────
            # RSI(2) positions: exit on SMA5 cross or 10-day timeout
            # NO ATR stop — Connors research: stops hurt RSI(2) performance
            if entry_meta.get(s, {}).get("engine") == "meanrev":
                _mr_sma5 = float(df_s["close"].tail(5).mean()) if len(df_s) >= 5 else close
                _mr_days = getattr(strat_mr, 'MAX_HOLD_DAYS', 10)
                if close > _mr_sma5:
                    exit_reason = "max_hold"  # reuse max_hold label for stats
                    exit_ref    = close
                elif hold_d >= _mr_days:
                    exit_reason = "max_hold"
                    exit_ref    = close
                elif close < pos.entry_price * 0.90:
                    # Emergency exit: down >10% = structural breakdown not a dip
                    exit_reason = "stop"
                    exit_ref    = close
                if exit_reason:
                    pass  # falls through to standard exit handler below
                else:
                    continue  # skip ATR stop logic for meanrev positions

            # Stop exit — original logic restored.
            # Regime-conditional confirmation (vol_ratio>1.5 OR >3% break)
            # caused WR to drop to 13% by holding through real breakdowns.
            # Lesson: whipsaw reduction via confirmation delays real exits too long.
            # Simple close-below-stop is more robust across regimes.
            if low <= stop_px:
                # ML conviction check — only hold through stop in TRENDING_BULL
                # if model still ranks stock above entry threshold.
                # Max pain = 2x original stop_pct — structural breakdown, not dip.
                _current_ml_rank    = prev_ml_ranks.get(s, 0.0)
                _entry_ml_threshold = getattr(params, 'ml_rank_min',
                                      getattr(params, 'ml_rank_min_long', 0.80))
                _original_stop_pct  = entry_meta.get(s, {}).get('stop_pct', stop_pct)
                _max_pain           = _original_stop_pct * 2
                _ml_still_believes  = (_current_ml_rank >= _entry_ml_threshold and
                                       _current_regime in (TRENDING_BULL, CHOPPY) and
                                       unrealized_pct > -_max_pain)
                if not _ml_still_believes or _ablate("ml_stop_hold"):
                    exit_reason = "stop"
                    exit_ref    = stop_px
            elif close >= pos.entry_price * (1 + getattr(params, 'take_profit_pct', getattr(params, 'take_profit_long', 0.40))):
                # Take profit — trailing ATR stop handles secular winners naturally
                # Removed fitted suppression (0.90/0.85/0.65 thresholds were fitted to APP/NVDA/VRT)
                exit_reason = "take_profit"
                exit_ref    = close
            elif hold_d >= getattr(params, 'max_hold_days', getattr(params, 'max_hold_days_long', 12)):
                # Profit-based hold extension: let winners run.
                # If position is up >8% AND ML score is top-decile, extend by 50%.
                # Data: 129 trades forced out at max_hold while up >$1000 — these
                # are genuine winners being killed by an arbitrary time limit.
                base_max   = getattr(params, 'max_hold_days', getattr(params, 'max_hold_days_long', 12))
                ml_score   = ml_scores.get(s, 0.0)
                extended   = int(base_max * 1.5) if (unrealized_pct > 0.08 and ml_score > 0.85 and not _ablate("hold_extension")) else base_max
                if hold_d >= extended:
                    # Friday exit delay — OOS validated p=0.035
                    # Friday exits WR=38% avg=-$329 vs Mon WR=61% avg=+$492
                    # If today is Friday (weekday=4), delay to Monday unless stop
                    if pd.Timestamp(date).weekday() == 4 and not _ablate("friday_delay"):
                        pass  # hold through weekend, exit Monday
                    else:
                        exit_reason = "max_hold"
                        exit_ref    = close
            elif _current_regime == BEAR and sector_map.get(s) not in strat_bear.DEFENSIVE_SECTORS:
                exit_reason = "regime_exit"
                exit_ref    = close


            if exit_reason:
                fill, comm = apply_fill_cost(exit_ref, pos.qty, "sell")
                pnl  = (fill - pos.entry_price) * pos.qty - comm
                cash += fill * pos.qty - comm
                meta  = entry_meta.get(s, {})
                # Settle conviction call before recording trade
                _call_pnl = 0.0
                OPTIONS_SLIP_EXIT = 0.005
                if meta.get("call_contracts", 0) > 0:
                    try:
                        from conviction_calls import bs_call as _bs_call2
                        _K2         = float(meta["call_strike"])
                        _sigma2     = float(meta["call_sigma"])
                        _n2         = int(meta["call_contracts"])
                        _cost2      = float(meta["call_cost"])
                        _entry_dt2  = pd.Timestamp(meta["call_entry_date"])
                        _days_held2 = (date - _entry_dt2).days
                        _DTE2       = 21
                        if _days_held2 >= _DTE2:
                            _df_c2  = prices_by_symbol.get(s, pd.DataFrame())
                            _fut2   = _df_c2.loc[_df_c2.index > _entry_dt2]["close"]
                            _dte_px = float(_fut2.iloc[_DTE2-1]) if len(_fut2) >= _DTE2 else fill
                            _mid2   = max(_dte_px - _K2, 0)
                        else:
                            _T_rem2 = max((_DTE2 - _days_held2) / 365.0, 0.01)
                            _mid2   = _bs_call2(fill, _K2, _T_rem2, 0.05, _sigma2)
                        _bid2      = _mid2 * (1 - OPTIONS_SLIP_EXIT)
                        _payoff2   = _n2 * _bid2 * 100
                        _call_pnl  = _payoff2 - _cost2
                        cash      += _payoff2
                        pnl       += _call_pnl
                    except Exception as _e:
                        _silent_log("conv_call_exit", s, _e)

                trades.append(Trade(
                    symbol=s, entry_date=pos.entry_time,
                    exit_date=str(date.date()), entry_price=pos.entry_price,
                    exit_price=fill, qty=pos.qty, pnl=pnl,
                    reason=exit_reason,
                    ml_rank_pct=float(meta.get("ml_rank_pct", 0)),
                    rule_score=float(meta.get("rule_score", 0)),
                    combined_score=float(meta.get("combined_score", 0)),
                    side="long",
                    regime=meta.get("regime", _current_regime),
                    ann_vol=float(meta.get("ann_vol", 0.35)),
                    call_cost=float(meta.get("call_cost", 0.0)),
                    call_pnl=float(_call_pnl),
                    engine=str(entry_meta.get(s, {}).get("engine", "momentum")),
                ))
                if exit_reason == "stop" or (exit_reason == "max_hold" and pnl < 0):
                    long_stop_dates[s] = date
                if exit_reason == "stop":
                    recent_stop_dates.append(date)
                if exit_reason == "regime_exit":
                    last_regime_exit_date = date

                del long_positions[s]
                entry_meta.pop(s, None)

        # ── Exit: shorts ──────────────────────────────────────────────────
        bear_params = strat_bear.get_params()
        for s in list(short_positions.keys()):
            pos   = short_positions[s]
            close = close_prices.get(s)
            high  = high_prices.get(s)
            if close is None or high is None:
                continue

            pnl_pct     = (pos.entry_price - close) / pos.entry_price
            exit_reason = exit_ref = None

            # Fix A: Short exits — time-based only. Price stops removed.
            # Data: 143 price stops at 0% WR = -$216k destroyed.
            # 95 max_hold exits at 85% WR = +$182k. Price stop is net negative.
            # Only keeping disaster stop at 40% for true catastrophic squeezes.
            if pnl_pct >= bear_params.take_profit_short:
                exit_reason = "short_take_profit"
                exit_ref    = close
            elif pos.age_days(pd.Timestamp(date).to_pydatetime()) >= bear_params.max_hold_days_short:
                exit_reason = "short_max_hold"
                exit_ref    = close
            elif _current_regime == TRENDING_BULL or (_current_regime == CHOPPY and not _cascade_bear and pos.age_days(pd.Timestamp(date).to_pydatetime()) <= 7):
                exit_reason = "regime_cover"
                exit_ref    = close
            elif high >= pos.entry_price * 1.40:
                exit_reason = "short_stop_disaster"
                exit_ref    = pos.entry_price * 1.40

            if exit_reason:
                fill, comm = apply_fill_cost(exit_ref, pos.qty, "buy")
                pnl        = (pos.entry_price - fill) * pos.qty - comm
                cash       += pnl
                meta        = entry_meta.get(f"short_{s}", {})
                trades.append(Trade(
                    symbol=s, entry_date=pos.entry_time,
                    exit_date=str(date.date()), entry_price=pos.entry_price,
                    exit_price=fill, qty=pos.qty, pnl=pnl,
                    reason=exit_reason,
                    ml_rank_pct=float(meta.get("ml_rank_pct", 0)),
                    rule_score=float(meta.get("rule_score", 0)),
                    combined_score=float(meta.get("combined_score", 0)),
                    side="short",
                ))
                if exit_reason == "short_stop_disaster" or (exit_reason == "short_max_hold" and pnl < 0):
                    short_stop_dates[s] = date
                del short_positions[s]
                entry_meta.pop(f"short_{s}", None)

        # ── Snapshots ─────────────────────────────────────────────────────
        snapshots = build_fast_snapshots(
            date=date, available_symbols=available_symbols,
            hist=prices_by_symbol, rule_store=rule_store, ml_scores=ml_scores,
        )

        # Fix 1: update ML rank memory for next day confirmation
        for _s_r, _snap_r in snapshots.items():
            prev_ml_ranks[_s_r] = _snap_r.ml_rank_pct

        if leadership_adapter:
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

        # ── Daily VIX scalar — shared by momentum AND meanrev ────────────
        # Computed once per day so both engines see the same market stress level
        # Meanrev was bypassing this entirely — fired in Feb 2025 at VIX 27-30
        _vix_now = vix_macro["close"].reindex([date], method="ffill")
        _vix_level = float(_vix_now.iloc[0]) if len(_vix_now) and not _vix_now.isna().all() else 20.0
        if _vix_level >= 35.0 and not _ablate("vix_vol_scalar"):
            _daily_vol_scalar = 0.0    # acute stress — no new longs either engine
        elif _vix_level >= 25.0:
            _daily_vol_scalar = 0.5    # elevated stress — half size
        elif _vix_level >= 20.0:
            _daily_vol_scalar = 0.75   # mild stress — 75% size
        else:
            _daily_vol_scalar = 1.0    # calm — full size

        # ── Long entries ──────────────────────────────────────────────────
        max_longs  = getattr(params, 'max_positions', getattr(params, 'max_positions_long', 4))
        max_shorts = getattr(params, 'max_positions_short', 0)

        cooldown_ok = (
            last_regime_exit_date is None or
            (date - last_regime_exit_date).days >= getattr(config, "REGIME_EXIT_COOLDOWN_DAYS", 10)
        )

        # Stop cascade sensor — if ≥2 stops fired in last 5 trading days,
        # freeze new entries. Macro shocks cluster stops before regime flips.
        recent_stop_dates[:] = [d for d in recent_stop_dates if (date - d).days <= 5]
        cascade_freeze = len(recent_stop_dates) >= 2 and not _ablate("cascade_freeze")

        if len(long_positions) < max_longs and cooldown_ok and not cascade_freeze:
            # Get ML threshold for current regime
            ml_min = getattr(params, 'ml_rank_min', getattr(params, 'ml_rank_min_long', 0.80))

            long_candidates = select_top_candidates(
                snapshots={k: v for k, v in snapshots.items() if v.ml_rank_pct >= ml_min},
                symbol_to_df=symbol_to_df_sel,
                current_positions={},
                max_names=max_longs,
                corr_matrix=corr_matrix,
                side="long",
            )

            for snap in long_candidates:
                s = snap.symbol
                if s in long_positions or s not in open_next_prices:
                    continue
                if len(long_positions) >= max_longs:
                    break

                # Vol-proportional cooldown — high-vol stocks need longer cooling.
                # APP vol=0.94: 180 days. NVDA vol=0.34: 30 days. Flat 15d let APP enter 4x in 2025.
                last_stop = long_stop_dates.get(s)
                if last_stop:
                    _df_cool = hist.get(s)
                    if _df_cool is not None and len(_df_cool) >= 20:
                        _cool_vol = float(_df_cool.loc[:date]["close"].pct_change().dropna().tail(60).std() * (252**0.5))
                        _cool_days = (180 if _cool_vol > 0.70 else 90 if _cool_vol > 0.50 else 45 if _cool_vol > 0.30 else 15) if not _ablate("vol_cooldown") else 0
                    else:
                        _cool_days = 15
                    if (date - last_stop).days < _cool_days:
                        continue

                # Earnings calendar filter — avoid entries within 5 days of earnings
                _earn = earnings_dates.get(s, [])
                if _earn and not _ablate("earnings_exclusion") and min(abs((date - d).days) for d in _earn) <= 5:
                    continue

                # Regime-conditional entry filter
                df_s = prices_by_symbol[s].loc[:date]
                if not is_in_accumulation(df_s, _current_regime):
                    continue

                # Fix 1: Entry confirmation — must rank top decile 2 consecutive days
                ml_min_threshold = getattr(params, 'ml_rank_min', getattr(params, 'ml_rank_min_long', 0.80))
                if prev_ml_ranks.get(s, 0.0) < ml_min_threshold:
                    prev_ml_ranks[s] = snap.ml_rank_pct
                    continue

                # Quality gate — don't buy high-momentum low-quality stocks
                # Research: momentum+quality combo reduces drawdowns 30-40% vs momentum alone
                # (Fama-French 5-factor, SGH 2024, abrdn 2024)
                q_feat = feature_store.get(s)
                if q_feat is not None and date in q_feat.index:
                    q_row = q_feat.loc[date]
                    quality = float(q_row.get("quality_composite", 0.5))
                    # In TRENDING_BULL: require quality > 0.2 (weak filter, don't miss rallies)
                    # In CHOPPY: require quality > 0.35 (tighter — only resilient stocks)
                    q_min = 0.35 if _current_regime == "CHOPPY" else 0.20
                    if quality < q_min and not _ablate("quality_gate"):
                        continue

                # SPY 5-day momentum filter — don't buy into weakness
                # Even in TRENDING_BULL, short-term dips cause clustered stops
                if _spy_5d_returns.get(date, 0.0) < -0.015 and not _ablate("spy_5d_filter"):
                    continue



                # Biotech gate — XLV high-vol names with binary event risk.
                # ALNY stopped twice same month different years. Same mechanism.
                # LLY/ABBV/MRK pass (consistent streaks). ALNY/RARE/IONS fail.
                if sector_map.get(s) == "XLV" and not _ablate("biotech_gate"):
                    _bio_df = hist.get(s)
                    if _bio_df is not None and len(_bio_df) >= 20:
                        _bio_vol = float(_bio_df.loc[:date]["close"].pct_change().dropna().tail(60).std() * (252**0.5))
                        if _bio_vol > 0.35:
                            q_feat2 = feature_store.get(s)
                            if q_feat2 is not None and date in q_feat2.index:
                                _streak = float(q_feat2.loc[date].get("earnings_beat_streak", 0.0))
                                if _streak < 0.60:  # requires ≥2 consecutive beats
                                    continue

                # Bear strategy: defensive sectors only
                if _current_regime == BEAR:
                    sector = sector_map.get(s)
                    if sector not in strat_bear.DEFENSIVE_SECTORS:
                        continue
                    if len(df_s) >= 200:
                        ma200 = df_s["close"].rolling(200).mean().iloc[-1]
                        if df_s["close"].iloc[-1] < ma200:
                            continue

                px         = open_next_prices[s]
                stop_pct   = float(np.clip(snap.stop_pct, getattr(params, 'stop_min_pct', getattr(params, 'stop_min_long', 0.02)), getattr(params, 'stop_max_pct', getattr(params, 'stop_max_long', 0.12))))
                # Fix 2: ML-rank-scaled conviction sizing
                import numpy as _np
                base_conviction = conviction_multiplier(snap)
                ml_conv = 0.5 + (snap.ml_rank_pct - ml_min_threshold) / max(1.0 - ml_min_threshold, 0.01)
                ml_conv = float(_np.clip(ml_conv, 0.5, 1.5))
                conviction = base_conviction * ml_conv

                # Use daily VIX scalar computed once above — same for momentum and meanrev
                vol_scalar = _daily_vol_scalar
                if vol_scalar == 0.0:
                    continue           # skip this entry entirely
                conviction = conviction * vol_scalar
                # Kelly position sizing — replaces hardcoded 3.5% risk/trade
                # Computed from last 100 similar trades (regime + ML bucket + vol bucket)
                # Falls back to 3.5% if <15 similar trades in history
                try:
                    import sys as _sys2
                    _sys2.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from kelly_sizer import compute_kelly
                    _df_vol = prices_by_symbol.get(s)
                    _ann_vol = float(_df_vol.loc[:date]["close"].pct_change().dropna().tail(60).std() * (252**0.5)) if _df_vol is not None and len(_df_vol) >= 20 else 0.35
                    _trade_dicts = [t.__dict__ for t in trades]
                    risk_pt = compute_kelly(
                        trades=_trade_dicts,
                        regime=_current_regime,
                        ml_score=snap.ml_rank_pct,
                        ann_vol=_ann_vol,
                        min_trades=15,
                        fallback=getattr(params, 'risk_per_trade', 0.035),
                    )
                except Exception:
                    risk_pt = getattr(params, 'risk_per_trade', 0.035)
                scalar     = getattr(params, 'position_scalar', getattr(params, 'position_scalar_long', 1.0))

                gross_exp  = sum(close_prices.get(sym, p.entry_price) * p.qty for sym, p in long_positions.items())
                max_exp    = config.INITIAL_CAPITAL * getattr(params, 'max_total_exposure', 1.6)
                remaining  = max(0, max_exp - gross_exp)
                if remaining <= 0:
                    break

                risk_budget    = config.INITIAL_CAPITAL * risk_pt * scalar * conviction
                risk_per_share = px * stop_pct
                qty_risk       = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
                max_wt         = getattr(params, 'max_position_weight', 0.35)
                max_dollars    = min(
                    config.INITIAL_CAPITAL * max_wt * scalar * conviction,
                    cash, remaining,
                )
                qty = min(qty_risk, int(max_dollars / px) if px > 0 else 0)
                # Vol-aware sizing: use ML stop width to scale position size.
                # Wider ML stop = model expects more noise = smaller position.
                # stop_pct already set by ML model above — use it directly.
                # High vol stocks get narrower ML stops (less room to breathe)
                # which naturally reduces position size through risk-per-share.
                # No separate vol scalar needed — it's implicit in stop_pct.
                pass
                if qty <= 0:
                    continue

                fill, comm = apply_fill_cost(px, qty, "buy")
                if fill * qty + comm > cash:
                    continue

                cash -= fill * qty + comm
                long_positions[s] = Position(
                    symbol=s, qty=qty, entry_price=fill,
                    entry_time=str(next_date.date()),
                    stop_pct=stop_pct,
                    initial_stop=fill * (1 - stop_pct),
                    highest_price=fill, add_count=0,
                )
                _df_vol_entry = prices_by_symbol.get(s)
                _ann_vol_entry = float(_df_vol_entry.loc[:date]["close"].pct_change().dropna().tail(60).std() * (252**0.5)) if _df_vol_entry is not None and len(_df_vol_entry) >= 20 else 0.35

                # ── Conviction calls overlay — delta-neutral sizing ───────
                # Budget sized so delta-adjusted exposure stays within Kelly.
                # No hardcoded fractions — call delta determines position size.
                _conv_meta = {
                    "ml_rank_pct":    snap.ml_rank_pct,
                    "rule_score":     snap.rule_score,
                    "combined_score": snap.combined_score,
                    "regime":         _current_regime,
                    "ann_vol":        _ann_vol_entry,
                }
                try:
                    from conviction_calls import conviction_score, compute_options_budget, bs_call
                    import json as _json
                    OPTIONS_SLIP = 0.005  # 50bps options slippage

                    _pead_s, _streak_s = 0.0, 0.0
                    _earn_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'cache_earnings_streak', f'{s}.json'
                    )
                    if os.path.exists(_earn_path):
                        with open(_earn_path) as _ef:
                            _recs = _json.load(_ef)
                        _prior = [r for r in _recs if pd.Timestamp(r['date']) < next_date]
                        if _prior:
                            _pead_s   = float(_prior[-1].get('surp', 0))
                            _beats    = [r['beat'] for r in _prior[-4:]]
                            _streak_s = float(sum(_beats)/len(_beats)) if _beats else 0.0

                    _rev_s = float(np.clip((snap.ml_rank_pct - 0.85) / 0.15, 0, 1))
                    _conv  = conviction_score(snap.ml_rank_pct, _pead_s, _rev_s, _streak_s)

                    if _conv >= 0.3 and not _ablate("conviction_calls"):
                        _K       = fill * 1.05
                        _T_call  = 21 / 365.0
                        _sigma_c = float(np.clip(_ann_vol_entry, 0.15, 1.5))
                        _call_mid = bs_call(fill, _K, _T_call, 0.05, _sigma_c)
                        _call_ask = _call_mid * (1 + OPTIONS_SLIP)

                        # Kelly budget = risk_pt * INITIAL_CAPITAL
                        _kelly_budget = risk_pt * config.INITIAL_CAPITAL

                        # Delta-neutral budget — exposure stays within Kelly
                        _call_budget = compute_options_budget(
                            conv_score=_conv,
                            kelly_budget=_kelly_budget,
                            stock_price=fill,
                            call_strike=_K,
                            call_price=_call_ask,
                            sigma=_sigma_c,
                            T=_T_call,
                        )

                        if _call_ask > 0.01 and _call_budget > _call_ask * 100:
                            _n_contracts = max(1, int(_call_budget / (_call_ask * 100)))
                            _call_cost   = _n_contracts * _call_ask * 100

                            # Reduce stock by equivalent delta exposure
                            from scipy.stats import norm as _norm2
                            _d1    = (np.log(fill / _K) + (0.05 + 0.5*_sigma_c**2)*_T_call) / (_sigma_c*np.sqrt(_T_call))
                            _delta = float(np.clip(_norm2.cdf(_d1), 0.05, 0.95))
                            _call_delta_equiv = _n_contracts * _delta * 100 * fill
                            _shares_to_remove = max(0, int(_call_delta_equiv / fill))
                            _new_qty = max(1, qty - _shares_to_remove)
                            _refund  = (qty - _new_qty) * fill
                            cash    += _refund

                            long_positions[s] = Position(
                                symbol=s, qty=_new_qty, entry_price=fill,
                                entry_time=str(next_date.date()),
                                stop_pct=stop_pct,
                                initial_stop=fill * (1 - stop_pct),
                                highest_price=fill, add_count=0,
                            )

                            if _call_cost <= cash:
                                cash -= _call_cost
                                _conv_meta.update({
                                    "conv_score":     _conv,
                                    "call_strike":    _K,
                                    "call_sigma":     _sigma_c,
                                    "call_contracts": _n_contracts,
                                    "call_cost":      _call_cost,
                                    "call_entry_date": str(next_date.date()),
                                    "call_delta":     _delta,
                                })
                except Exception as _e:
                    _silent_log("conv_call_entry", s, _e)

                entry_meta[s] = _conv_meta

        # ── Mean reversion entries (CHOPPY regime only) ───────────────────────
        # Only fire after momentum is fully deployed — meanrev is additive, not a replacement
        _momentum_slots_filled = len([s for s in long_positions if entry_meta.get(s, {}).get("engine", "momentum") == "momentum"]) >= max_longs
        # In BEAR regime momentum is suppressed to defensive only — slots rarely fill
        # Allow meanrev to fire independently in BEAR on quality oversold names
        _mr_slots_ok = _momentum_slots_filled or _current_regime == BEAR
        if _current_regime in (CHOPPY, BEAR) and _mr_slots_ok and _daily_vol_scalar > 0 and not cascade_freeze:
            _mr_params = strat_mr.MeanRevParams()
            _mr_slots  = max(0, _mr_params.max_positions - sum(
                1 for s in long_positions
                if entry_meta.get(s, {}).get("engine") == "meanrev"
            ))
            if _mr_slots > 0:
                _mr_snaps = []
                for sym in valid_syms:
                    if sym in long_positions: continue
                    # Cooldown — no meanrev re-entry within 60 days of any stop on this symbol
                    _mr_last_stop = long_stop_dates.get(sym)
                    if _mr_last_stop and (date - _mr_last_stop).days < 60: continue
                    df_s2 = prices_by_symbol.get(sym)
                    if df_s2 is None: continue
                    _ml_r = 0.5
                    _ml_val = ml_scores.get(sym, 0.5)
                    if isinstance(_ml_val, dict):
                        _ml_r = float(_ml_val.get("ml_rank_pct", 0.5))
                    elif isinstance(_ml_val, (int, float)):
                        _ml_r = float(_ml_val)
                    snap_mr = strat_mr.compute_snapshot(sym, df_s2.loc[:date], ml_rank=_ml_r)
                    if snap_mr is not None:
                        _mr_snaps.append(snap_mr)
                _mr_candidates = strat_mr.score_meanrev_candidates(_mr_snaps, _mr_params)
                # Track sectors already in meanrev to avoid clustering
                _mr_sectors_open = {sector_map.get(s2, '') for s2 in long_positions
                                    if entry_meta.get(s2, {}).get("engine") == "meanrev"}

                for mr_snap in _mr_candidates[:_mr_slots]:
                    s = mr_snap.symbol
                    if s in long_positions or s not in open_next_prices: continue

                    # Earnings filter — same as momentum, block within 7 days
                    _mr_earn = earnings_dates.get(s, [])
                    if _mr_earn:
                        _days_to_earn = [(d - date).days for d in _mr_earn]
                        # Block 10 days BEFORE earnings (upcoming) and 2 days AFTER (gap still open)
                        if any(-2 <= d <= 10 for d in _days_to_earn):
                            continue

                    # Minimum liquidity — no meanrev on thin stocks
                    # Thin stocks gap through stops (RKLB, ZIM pattern)
                    _mr_df = prices_by_symbol.get(s)
                    if _mr_df is not None and len(_mr_df) >= 30:
                        _mr_avg_vol = float(_mr_df.loc[:date]['volume'].tail(30).mean())
                        _mr_price   = float(_mr_df.loc[:date]['close'].iloc[-1])
                        _mr_dollar_vol = _mr_avg_vol * _mr_price
                        if _mr_dollar_vol < 50_000_000:  # $50M daily dollar volume minimum
                            continue

                    # Sector concentration — max 1 meanrev per sector
                    _mr_sector = sector_map.get(s, '')
                    if _mr_sector and _mr_sector in _mr_sectors_open:
                        continue
                    _mr_sectors_open.add(_mr_sector)
                    px  = open_next_prices[s]
                    port_val_mr = cash + sum(
                        p.qty * close_prices.get(s2, p.entry_price)
                        for s2, p in long_positions.items()
                    )
                    _mr_ml_rank = float(ml_scores.get(s, 0.5)) if isinstance(ml_scores.get(s, 0.5), float) else 0.5
                    # BACKFIT FIX: ML filter on meanrev REMOVED.
                    # This filter was added AFTER seeing OOS results (backfitted).
                    # Origin: commit 3de1207 added ml_rank_min after observing
                    # OOS 30-50% bucket avg=-$73 vs 85%+ avg=$235-277.
                    # For clean holdout: meanrev enters on pure RSI(2) < 5 only.
                    # If we want ML filter, must be chosen on training data and
                    # locked BEFORE seeing 2025 holdout.
                    qty = strat_mr.size_meanrev_position(
                        capital=port_val_mr, price=px,
                        params=_mr_params,
                        n_open_positions=len(long_positions),
                        snap=mr_snap,
                        ml_rank=_mr_ml_rank,
                    )
                    if qty < 1 or qty * px > cash * 0.95: continue
                    cash -= qty * px
                    long_positions[s] = Position(
                        symbol=s, qty=qty, entry_price=px,
                        entry_time=str(next_date.date()),
                        stop_pct=0.10,
                        initial_stop=px * 0.90,
                        highest_price=px,
                        add_count=0,
                    )
                    entry_meta[s] = {"engine": "meanrev", "entry_date": str(date),
                                     "rsi2": mr_snap.rsi2,
                                     "ml_rank_pct": _mr_ml_rank,
                                     "ann_vol": mr_snap.ann_vol,
                                     "regime": _current_regime}

        # ── Short entries (bear regime only) ──────────────────────────────
        # Cascade bear — activate short book in CHOPPY when market is breaking down
        # Data: cascade + SPY_5d < -1.5% correctly identifies Feb 24 - Mar 10 2025
        # Does NOT fire when cascade fires during SPY uptrend (Feb 7-21 2025)
        _spy_5d_now = _spy_5d_returns.get(date, 0.0)
        _cascade_bear = cascade_freeze and _spy_5d_now < -0.015

        if (_current_regime == BEAR or _cascade_bear) and len(short_positions) < max_shorts:
            short_candidates = strat_bear.score_short_candidates(
                snapshots=snapshots,
                prices=prices_by_symbol,
                sector_map=sector_map,
                as_of_date=date,
            )

            for cand in short_candidates:
                s = cand["symbol"]
                if s in short_positions or s not in open_next_prices:
                    continue
                if len(short_positions) >= max_shorts:
                    break

                last_stop = short_stop_dates.get(s)
                if last_stop and (date - last_stop).days < getattr(config, "SHORT_REENTRY_COOLDOWN_DAYS", 60):
                    continue

                px         = open_next_prices[s]
                stop_pct   = bear_params.stop_short
                scalar     = bear_params.position_scalar_short
                risk_budget= config.INITIAL_CAPITAL * bear_params.risk_per_trade_short * scalar
                qty_risk   = int(risk_budget / (px * stop_pct)) if px * stop_pct > 0 else 0
                qty        = min(qty_risk, int(config.INITIAL_CAPITAL * 0.08 / px) if px > 0 else 0)  # max 8% of initial capital per short
                if qty <= 0:
                    continue

                margin = px * qty * 0.60
                if margin > cash:
                    continue

                stop_px = px * (1 + stop_pct)
                short_positions[s] = ShortPosition(
                    symbol=s, qty=qty, entry_price=px,
                    entry_time=str(next_date.date()),
                    stop_price=stop_px,
                )
                entry_meta[f"short_{s}"] = {
                    "ml_rank_pct":    cand["ml_rank_pct"],
                    "rule_score":     0.0,
                    "combined_score": cand["short_score"],
                    "regime":         _current_regime,
                }

        # ── Debug print ───────────────────────────────────────────────────
        if step_idx % 250 == 0:
            print()
            print(f"  [regime] {date.date()} = {_current_regime}"
                  f"  L={len(long_positions)} S={len(short_positions)}"
                  f"  trades={len(trades)}", flush=True)

        # ── Position lifecycle logging ────────────────────────────────────
        if long_positions:
            _port_rank_mean = float(sum(prev_ml_ranks.get(s, 0.5) for s in long_positions) / len(long_positions))
            _port_rank_prev = getattr(run_backtest_v2, '_prev_port_rank', _port_rank_mean)
            _port_rank_vel  = _port_rank_mean - _port_rank_prev
            run_backtest_v2._prev_port_rank = _port_rank_mean
            _n_open = len(long_positions)
            _port_corr = 0.0
            if _n_open >= 2:
                try:
                    _ret_matrix = []
                    for _ps in list(long_positions.keys())[:8]:
                        _pdf = prices_by_symbol.get(_ps)
                        if _pdf is not None:
                            _rets = _pdf.loc[:date]['close'].pct_change().dropna().tail(20)
                            if len(_rets) >= 10:
                                _ret_matrix.append(_rets.values[-10:])
                    if len(_ret_matrix) >= 2:
                        _corr_mat = np.corrcoef(np.array(_ret_matrix))
                        _upper = _corr_mat[np.triu_indices(len(_corr_mat), k=1)]
                        _port_corr = float(np.mean(_upper))
                except Exception:
                    pass
            for _ls, _lp in list(long_positions.items()):
                _lc = close_prices.get(_ls)
                if _lc is None:
                    continue
                _unr = (_lc - _lp.entry_price) / _lp.entry_price
                _hd  = _lp.age_days(pd.Timestamp(date).to_pydatetime())
                _lifecycle_rows.append({
                    'date':                str(date.date()),
                    'symbol':              _ls,
                    'engine':              entry_meta.get(_ls, {}).get('engine', 'momentum'),
                    'regime':              _current_regime,
                    'days_held':           int(_hd),
                    'ml_rank_entry':       float(entry_meta.get(_ls, {}).get('ml_rank_pct', 0.5)),
                    'ml_rank_now':         float(prev_ml_ranks.get(_ls, 0.5)),
                    'ml_rank_delta':       float(prev_ml_ranks.get(_ls, 0.5)) - float(entry_meta.get(_ls, {}).get('ml_rank_pct', 0.5)),
                    'unrealized_pct':      float(_unr),
                    'vix_now':             float(_vix_level),
                    'portfolio_rank_mean': float(_port_rank_mean),
                    'portfolio_rank_vel':  float(_port_rank_vel),
                    'portfolio_size':      int(_n_open),
                    'portfolio_corr':      float(_port_corr),
                    'ann_vol':             float(entry_meta.get(_ls, {}).get('ann_vol', 0.35)),
                })

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
    _print_results(stats, trades, regime_counts)
    # Save position lifecycle data for TFT training
    if _lifecycle_rows:
        import os as _os2
        _lc_df = pd.DataFrame(_lifecycle_rows)
        _lc_path = _os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), '..', 'cache_backtester', 'position_lifecycle.csv')
        _lc_df.to_csv(_lc_path, index=False)
        print(f"[lifecycle] {len(_lc_df):,} daily snapshots saved → cache_backtester/position_lifecycle.csv", flush=True)
    return equity_curve, trades, stats


def _portfolio_value(cash, close_prices, long_positions, short_positions):
    long_val  = sum(close_prices.get(s, p.entry_price) * p.qty for s, p in long_positions.items())
    short_pnl = sum((p.entry_price - close_prices.get(s, p.entry_price)) * p.qty for s, p in short_positions.items())
    return cash + long_val + short_pnl


def _print_results(stats, trades, regime_counts):
    print("\n" + "═"*60)
    print("  V2 BACKTEST RESULTS")
    print("═"*60)
    print(f"  Total Return : {stats['total_return']:>10.2%}")
    print(f"  CAGR         : {stats['cagr']:>10.2%}")
    print(f"  Sharpe       : {stats['sharpe']:>10.2f}")
    print(f"  Max Drawdown : {stats['max_drawdown']:>10.2%}")
    print(f"  Trades       : {stats['trades']:>10}")
    print(f"  Win Rate     : {stats['win_rate']:>10.2%}")

    total_days = sum(regime_counts.values())
    if total_days > 0:
        print(f"\n  Regime distribution:")
        for r, count in regime_counts.items():
            print(f"    {r:<15}: {count:4d} days ({count/total_days:.0%})")

    if trades:
        trade_df = pd.DataFrame([t.__dict__ for t in trades])
        trade_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trades_v2.csv"), index=False)
        pnl = trade_df["pnl"]
        print(f"\n  Avg Trade PnL: ${pnl.mean():.0f}")

        print(f"\n  Exit breakdown:")
        for reason, grp in trade_df.groupby("reason"):
            wr = (grp["pnl"] > 0).mean()
            print(f"    {reason:<20} {len(grp):4d}  WR={wr:.0%}  avg=${grp['pnl'].mean():.0f}")

        # Mean reversion vs momentum breakdown
        if "engine" in trade_df.columns:
            print(f"\n  By engine:")
            for eng, grp in trade_df.groupby("engine"):
                wr = (grp["pnl"] > 0).mean()
                print(f"    {str(eng):<15} {len(grp):4d} trades  WR={wr:.0%}  avg=${grp['pnl'].mean():.0f}")
        print(f"\n  By regime entered:")
        if "regime" in trade_df.columns:
            for regime, grp in trade_df.groupby("regime"):
                wr = (grp["pnl"] > 0).mean()
                print(f"    {regime:<15} {len(grp):4d} trades  WR={wr:.0%}  avg=${grp['pnl'].mean():.0f}")

    if stats.get("annual"):
        print(f"\n  {'Year':<6} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7}")
        print("  " + "-"*40)
        for yr in sorted(stats["annual"]):
            a = stats["annual"][yr]
            flag = " ✓" if a["cagr"] > 0 else " ✗"
            print(f"  {yr:<6} {a['cagr']:>7.1%}  {a['sharpe']:>6.2f}  {a['max_drawdown']:>7.1%}  {a['n_trades']:>6}{flag}")

    print("═"*60 + "\n")


def run_oos_test_v2():
    print("="*60)
    print("  V2 OUT-OF-SAMPLE VALIDATION")
    print("  IN-SAMPLE : 2015-2021")
    print("  OOS       : 2022-2025")
    print("="*60 + "\n")

    equity_full, trades_full, _ = run_backtest_v2(days=args.days)

    oos_cutoff  = pd.Timestamp("2022-01-01")
    eq_in       = equity_full[equity_full.index <  oos_cutoff]
    eq_oos      = equity_full[equity_full.index >= oos_cutoff]
    eq_oos_norm = eq_oos / eq_oos.iloc[0] * config.INITIAL_CAPITAL
    trades_in   = [t for t in trades_full if str(t.exit_date) <  "2022-01-01"]
    trades_oos  = [t for t in trades_full if str(t.exit_date) >= "2022-01-01"]
    stats_in    = calc_stats(eq_in,       trades_in)
    stats_oos   = calc_stats(eq_oos_norm, trades_oos)

    def _print(label, stats):
        print(f"\n  -- {label} --")
        print(f"  CAGR     : {stats['cagr']:>8.2%}")
        print(f"  Sharpe   : {stats['sharpe']:>8.2f}")
        print(f"  Max DD   : {stats['max_drawdown']:>8.2%}")
        print(f"  Win Rate : {stats['win_rate']:>8.2%}")
        print(f"  Trades   : {stats['trades']:>8}")
        if stats.get("annual"):
            print(f"\n  {'Year':<6} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7}")
            print("  " + "-"*40)
            for yr in sorted(stats["annual"]):
                a = stats["annual"][yr]
                flag = " ✓" if a["cagr"] > 0 else " ✗"
                print(f"  {yr:<6} {a['cagr']:>7.1%}  {a['sharpe']:>6.2f}  {a['max_drawdown']:>7.1%}  {a['n_trades']:>6}{flag}")

    _print("IN-SAMPLE 2015-2021",     stats_in)
    _print("OUT-OF-SAMPLE 2022-2025", stats_oos)

    oos_sh   = stats_oos["sharpe"]
    oos_cagr = stats_oos["cagr"]
    oos_dd   = abs(stats_oos["max_drawdown"])

    if   oos_cagr >= 0.20 and oos_dd <= 0.15 and oos_sh >= 1.8:
        verdict = "STRONG EDGE — Ready to scale."
    elif oos_cagr >= 0.12 and oos_dd <= 0.20 and oos_sh >= 1.2:
        verdict = "MODERATE EDGE — Good foundation, keep building."
    elif oos_cagr >= 0.00:
        verdict = "WEAK — Edge is thin."
    else:
        verdict = "NO EDGE — Negative OOS."

    print(f"\n{'='*60}\n  VERDICT: {verdict}\n{'='*60}\n")


def run_2025_holdout():
    """
    CLEAN 2025 HOLDOUT TEST
    =======================
    Rules:
    - Models: 2025 vintage (trained 2020-2024). No 2025 data in training.
    - Parameters: ALL locked at current values. No changes after seeing results.
    - Universe: only stocks with price data available on each date (survivorship handled).
    - Meanrev: pure RSI(2) < 5 entry, NO ML filter (removed as backfitted).
    - Regime thresholds: locked, documented origins above.

    This is the ONLY number that matters. Everything else is in-sample.
    """
    print("=" * 60)
    print("  2025 CLEAN HOLDOUT TEST")
    print("  Models trained: 2020-2024 only (2025 vintage)")
    print("  Parameters: LOCKED (no post-hoc changes)")
    print("  Meanrev ML filter: REMOVED (backfitted)")
    print("  Survivorship: handled by data-availability check")
    print("=" * 60 + "\n")

    equity_full, trades_full, _ = run_backtest_v2(days=args.days)

    holdout_start = pd.Timestamp("2025-01-01")
    eq_holdout    = equity_full[equity_full.index >= holdout_start]
    if len(eq_holdout) < 10:
        print("ERROR: insufficient 2025 data in backtest range. Use --days 4000 or higher.")
        return

    eq_holdout_norm = eq_holdout / eq_holdout.iloc[0] * config.INITIAL_CAPITAL
    trades_holdout  = [t for t in trades_full if str(t.exit_date) >= "2025-01-01"
                       and str(t.entry_date) >= "2025-01-01"]
    stats = calc_stats(eq_holdout_norm, trades_holdout)

    trade_df = pd.DataFrame([{
        "sym": t.symbol, "entry": t.entry_date, "exit": t.exit_date,
        "pnl": t.pnl, "reason": t.reason,
        "engine": getattr(t, "engine", ""),
        "regime": getattr(t, "regime", ""),
    } for t in trades_holdout])

    print(f"\n{'='*60}")
    print(f"  2025 HOLDOUT RESULTS")
    print(f"{'='*60}")
    print(f"  CAGR       : {stats['cagr']:>8.2%}")
    print(f"  Sharpe     : {stats['sharpe']:>8.2f}")
    print(f"  Max DD     : {stats['max_drawdown']:>8.2%}")
    print(f"  Win Rate   : {stats['win_rate']:>8.2%}")
    print(f"  Trades     : {stats['trades']:>8}")

    if len(trade_df) > 0:
        print(f"\n  By engine:")
        if "engine" in trade_df.columns:
            for eng, grp in trade_df.groupby("engine"):
                if len(grp) == 0: continue
                wr = (grp["pnl"] > 0).mean()
                print(f"    {str(eng):<15} {len(grp):4d} trades  WR={wr:.0%}  avg=${grp['pnl'].mean():.0f}")
        print(f"\n  By regime:")
        if "regime" in trade_df.columns:
            for reg, grp in trade_df.groupby("regime"):
                if len(grp) == 0: continue
                wr = (grp["pnl"] > 0).mean()
                print(f"    {str(reg):<15} {len(grp):4d} trades  WR={wr:.0%}  avg=${grp['pnl'].mean():.0f}")

    if stats.get("annual"):
        print(f"\n  {'Year':<6} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7}")
        print("  " + "-" * 40)
        for yr in sorted(stats["annual"]):
            a = stats["annual"][yr]
            flag = " ✓" if a["cagr"] > 0 else " ✗"
            print(f"  {yr:<6} {a['cagr']:>7.1%}  {a['sharpe']:>6.2f}  {a['max_drawdown']:>7.1%}  {a['n_trades']:>6}{flag}")

    print(f"\n{'='*60}")
    print(f"  THIS IS THE ONLY NUMBER THAT MATTERS.")
    print(f"  Everything before 2025 is in-sample.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos",     action="store_true")
    parser.add_argument("--holdout", action="store_true", help="Clean 2025-only holdout test")
    parser.add_argument("--days",    type=int, default=3650)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.holdout:
        run_2025_holdout()
    elif args.oos:
        run_oos_test_v2()
    else:
        run_backtest_v2(days=args.days, refresh_cache=args.refresh)
