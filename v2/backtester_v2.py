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
    adaptive_stop_pct, compute_atr_pct, load_ranker_ensemble,
    market_regime, normalize_ohlcv, select_top_candidates,
    trend_bullish,
)

from regime_classifier import (
    RegimeClassifier, compute_signals, load_macro_data,
    TRENDING_BULL, CHOPPY, BEAR,
)
from entry_filter import is_in_accumulation, filter_candidates
import strategy_trending as strat_bull
import strategy_choppy  as strat_chop
import strategy_bear    as strat_bear

CACHE_DIR = "cache_prices"


def _get_strategy(regime: str):
    """Return the active strategy module for the current regime."""
    if regime == TRENDING_BULL:
        return strat_bull
    elif regime == CHOPPY:
        return strat_chop
    else:
        return strat_bear


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
    rankers = load_ranker_ensemble()
    feat_cols_union = sorted(set(
        list(rankers[3]["features"]) +
        list(rankers[5]["features"]) +
        list(rankers[7]["features"])
    ))
    print(f"[ok]   ensemble loaded ({len(feat_cols_union)} features)", flush=True)

    # ── 5. Feature computation ────────────────────────────────────────────
    print("[prep] computing features...", flush=True)
    feature_store: Dict[str, pd.DataFrame] = {}
    for i, s in enumerate(symbols):
        print(f"  [feat] {i+1}/{len(symbols)} {s}   ", end="\r", flush=True)
        try:
            f = compute_features(prices_by_symbol[s], symbol=s)
            feature_store[s] = f.replace([np.inf, -np.inf], np.nan) if f is not None and len(f) else pd.DataFrame()
        except Exception:
            feature_store[s] = pd.DataFrame()
    print(f"\n[ok]   features done", flush=True)

    # ── 6. Rule scores + feature matrix ──────────────────────────────────
    print("[prep] computing rule scores...", flush=True)
    rule_store = build_rule_store_fast(symbols, prices_by_symbol)
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

    # ── 8. Leadership adapter ─────────────────────────────────────────────
    leadership_adapter = None
    if getattr(config, "ADAPTIVE_LEADERSHIP_ENABLED", True):
        leadership_adapter = LeadershipAdapter(
            update_frequency_days=getattr(config, "LEADERSHIP_UPDATE_FREQ_DAYS", 5),
            leadership_threshold=getattr(config, "LEADERSHIP_THRESHOLD", 0.62),
            top_n_leaders=getattr(config, "LEADERSHIP_TOP_N", 4),
        )
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

    last_regime_exit_date: Optional[pd.Timestamp] = None
    _regime_candidate:     Optional[str]           = None
    _regime_candidate_days: int                    = 0
    _current_regime:        str                    = CHOPPY
    regime_counts = {TRENDING_BULL: 0, CHOPPY: 0, BEAR: 0}

    lookback    = 260
    total_steps = len(all_trade_dates) - 1 - lookback

    print(f"[run] starting main loop ({total_steps} steps)...\n", flush=True)

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
        if signals:
            _current_regime = regime_clf.update(date, signals)
        regime_counts[_current_regime] = regime_counts.get(_current_regime, 0) + 1

        # Get active strategy
        strategy = _get_strategy(_current_regime)
        params   = strategy.get_params()

        # ── ML scoring ────────────────────────────────────────────────────
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
            df_s     = prices_by_symbol[s].loc[:date]
            stop_pct = adaptive_stop_pct(df_s, pos.entry_price, close, side="long")

            # Use strategy-specific stop bounds
            stop_min = getattr(params, 'stop_min_pct', getattr(params, 'stop_min_long', 0.02))
            stop_max = getattr(params, 'stop_max_pct', getattr(params, 'stop_max_long', 0.12))
            stop_pct = float(np.clip(stop_pct, stop_min, stop_max))
            stop_px  = pos.entry_price * (1 - stop_pct)
            trail_stop = pos.highest_price * (1 - stop_pct)
            stop_px  = max(stop_px, trail_stop)

            hold_d      = pos.age_days(pd.Timestamp(date).to_pydatetime())
            exit_reason = exit_ref = None

            if low <= stop_px:
                exit_reason = "stop"
                exit_ref    = stop_px
            elif close >= pos.entry_price * (1 + getattr(params, 'take_profit_pct', getattr(params, 'take_profit_long', 0.40))):
                exit_reason = "take_profit"
                exit_ref    = close
            elif hold_d >= getattr(params, 'max_hold_days', getattr(params, 'max_hold_days_long', 12)):
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
                trades.append(Trade(
                    symbol=s, entry_date=pos.entry_time,
                    exit_date=str(date.date()), entry_price=pos.entry_price,
                    exit_price=fill, qty=pos.qty, pnl=pnl,
                    reason=exit_reason,
                    ml_rank_pct=float(meta.get("ml_rank_pct", 0)),
                    rule_score=float(meta.get("rule_score", 0)),
                    combined_score=float(meta.get("combined_score", 0)),
                    side="long",
                ))
                if exit_reason == "stop":
                    long_stop_dates[s] = date
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

            if pnl_pct >= bear_params.take_profit_short:
                exit_reason = "short_take_profit"
                exit_ref    = close
            elif high >= pos.entry_price * (1 + bear_params.stop_short):
                exit_reason = "short_stop"
                exit_ref    = pos.entry_price * (1 + bear_params.stop_short)
            elif pos.age_days(pd.Timestamp(date).to_pydatetime()) >= bear_params.max_hold_days_short:
                exit_reason = "short_max_hold"
                exit_ref    = close
            elif _current_regime == TRENDING_BULL:
                exit_reason = "regime_cover"
                exit_ref    = close

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
                if exit_reason == "short_stop":
                    short_stop_dates[s] = date
                del short_positions[s]
                entry_meta.pop(f"short_{s}", None)

        # ── Snapshots ─────────────────────────────────────────────────────
        snapshots = build_fast_snapshots(
            date=date, available_symbols=available_symbols,
            hist=prices_by_symbol, rule_store=rule_store, ml_scores=ml_scores,
        )

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

        # ── Long entries ──────────────────────────────────────────────────
        max_longs  = getattr(params, 'max_positions', getattr(params, 'max_positions_long', 4))
        max_shorts = getattr(params, 'max_positions_short', 0)

        cooldown_ok = (
            last_regime_exit_date is None or
            (date - last_regime_exit_date).days >= getattr(config, "REGIME_EXIT_COOLDOWN_DAYS", 10)
        )

        if len(long_positions) < max_longs and cooldown_ok:
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

                # Long stop cooldown
                last_stop = long_stop_dates.get(s)
                if last_stop and (date - last_stop).days < 15:
                    continue

                # Regime-conditional entry filter
                df_s = prices_by_symbol[s].loc[:date]
                if not is_in_accumulation(df_s, _current_regime):
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
                conviction = conviction_multiplier(snap)
                risk_pt    = getattr(params, 'risk_per_trade', 0.035)
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
                entry_meta[s] = {
                    "ml_rank_pct":    snap.ml_rank_pct,
                    "rule_score":     snap.rule_score,
                    "combined_score": snap.combined_score,
                    "regime":         _current_regime,
                }

        # ── Short entries (bear regime only) ──────────────────────────────
        if _current_regime == BEAR and len(short_positions) < max_shorts:
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
                qty        = min(qty_risk, int(35_000 / px) if px > 0 else 0)
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
        pnl = trade_df["pnl"]
        print(f"\n  Avg Trade PnL: ${pnl.mean():.0f}")

        print(f"\n  Exit breakdown:")
        for reason, grp in trade_df.groupby("reason"):
            wr = (grp["pnl"] > 0).mean()
            print(f"    {reason:<20} {len(grp):4d}  WR={wr:.0%}  avg=${grp['pnl'].mean():.0f}")

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

    equity_full, trades_full, _ = run_backtest_v2(days=3650)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos",     action="store_true")
    parser.add_argument("--days",    type=int, default=3650)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.oos:
        run_oos_test_v2()
    else:
        run_backtest_v2(days=args.days, refresh_cache=args.refresh)
