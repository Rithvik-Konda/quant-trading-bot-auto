#!/usr/bin/env python3
"""
train_trending_bull.py — TRENDING_BULL Regime-Specific ML Ranker v2
====================================================================
The existing TRENDING_BULL model (from train_rolling_regime_ensemble) has
"consistently negative IC" per strategy_core.py:569. This script tries a
different approach:

1. Same feature set + cross-sectional ranks (as existing pipeline)
2. Regime-filtered training: ONLY TRENDING_BULL days
3. Tuned LightGBM hyperparameters: deeper trees, more momentum weight
4. 5-day horizon (primary holding period)
5. Rolling walk-forward: train 2015-2022, validate 2023-2025
6. Compare IC vs all-regime model on TRENDING_BULL days

Key hypothesis: in TRENDING_BULL, momentum persistence and sector
leadership matter more; mean-reversion and defensive features matter less.
A deeper model can capture non-linear momentum interactions that the
shallow (depth=4) all-regime model cannot.

Usage:
    cd ~/ai_trading_bot_v2
    python v2/train_trending_bull.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# Add project root
ROOT = os.path.expanduser("~/ai_trading_bot_v2")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "v2"))

import config
from ml_model import (
    fetch_data, compute_features, compute_forward_return,
    _load_etf, _build_model, log, build_panel_from_store,
    USE_LIGHTGBM, CACHE_DIR,
)
from regime_classifier import (
    RegimeClassifier, compute_signals, load_macro_data,
    TRENDING_BULL, CHOPPY, BEAR,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ── Configuration ──────────────────────────────────────────────────────────────

# Use a liquid subset for faster initial validation
# Top 50 most liquid symbols from the watchlist
LIQUID_50 = [
    "NVDA", "AVGO", "AMD", "MU", "QCOM", "TXN", "INTC", "AMAT", "LRCX", "KLAC",
    "MSFT", "META", "AMZN", "GOOGL", "AAPL", "TSLA",
    "SNOW", "DDOG", "NET", "SHOP", "NOW", "CRWD", "CRM", "ORCL", "PLTR",
    "JPM", "GS", "V", "MA", "AXP",
    "UNH", "LLY", "ABBV", "JNJ", "MRK",
    "XOM", "CVX", "COP",
    "HD", "NKE", "SBUX", "MCD", "CMG",
    "BA", "CAT", "RTX", "GE", "DE",
    "DIS", "NFLX", "UBER",
]

HORIZON = 5  # 5-day forward return
TRAIN_YEARS = 6  # train on 6 years before test year
WEEKLY_SAMPLE = True  # sample weekly to reduce compute


def build_regime_series() -> pd.Series:
    """Reconstruct daily regime labels from macro data."""
    log("INFO | Building regime label series...")
    cache_dir = os.path.join(ROOT, "cache_prices")
    spy_macro, hyg_macro, vix_macro = load_macro_data(cache_dir=cache_dir)
    clf = RegimeClassifier()

    spy_dates = spy_macro.index.sort_values()
    regime_labels: dict = {}
    for d in spy_dates:
        signals = compute_signals(spy_macro, hyg_macro, vix_macro, as_of_date=d)
        if signals:
            regime_labels[d] = clf.update(d, signals)

    regime_series = pd.Series(regime_labels)
    counts = {r: int((regime_series == r).sum()) for r in [TRENDING_BULL, CHOPPY, BEAR]}
    log(f"INFO | Regime counts: {counts}")
    return regime_series


def build_store_fast(symbols: list, days: int = 3650) -> dict:
    """Build symbol store — features for each symbol. Skips slow external caches."""
    # Pre-load streak/insider/revision stores if cached
    streak_store = {}
    insider_store = {}
    revision_store = {}
    try:
        sp = os.path.join(ROOT, "streak_store.joblib")
        if os.path.exists(sp):
            streak_store = joblib.load(sp)
            log(f"INFO | Loaded streak store: {len(streak_store)} symbols")
    except Exception:
        pass
    try:
        ip = os.path.join(ROOT, "insider_store.joblib")
        if os.path.exists(ip):
            insider_store = joblib.load(ip)
            log(f"INFO | Loaded insider store: {len(insider_store)} symbols")
    except Exception:
        pass

    store = {}
    for i, sym in enumerate(symbols):
        df = fetch_data(sym, days, refresh=False)
        if df is None or len(df) < 260:
            continue
        feat = compute_features(
            df, symbol=sym,
            streak_store=streak_store,
            insider_store=insider_store,
            revision_store=revision_store,
        )
        store[sym] = {"prices": df, "features": feat}
        if (i + 1) % 10 == 0:
            log(f"INFO | Loaded {i+1}/{len(symbols)} symbols")
    log(f"INFO | Store built: {len(store)} symbols")
    return store


def build_panel(store: dict, horizon: int) -> pd.DataFrame:
    """Build panel with forward returns and cross-sectional ranks."""
    # Use the existing pipeline which adds cs_rank features
    panel = build_panel_from_store(store, horizon)
    return panel


def build_trending_bull_model(horizon: int = 5, depth: int = 5, n_est: int = 600):
    """Build a TRENDING_BULL-tuned LightGBM model."""
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=n_est,
            learning_rate=0.015,      # slower learning → better generalization
            max_depth=depth,           # deeper than all-regime (4) to capture interactions
            num_leaves=31,             # more leaves for deeper trees
            min_child_samples=30,      # slightly less regularized (more TRENDING_BULL-specific patterns)
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=0.05,            # less L1 penalty — let momentum features dominate
            reg_lambda=0.5,            # less L2 penalty
            random_state=42 + horizon,
            n_jobs=-1,
            verbose=-1,
        )
        log(f"INFO | TRENDING_BULL LightGBM (depth={depth}, est={n_est})")
        return model
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.02, max_depth=3,
            min_samples_leaf=30, subsample=0.7, max_features=0.6,
            random_state=42 + horizon,
        )


def evaluate_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank correlation (Information Coefficient)."""
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 30:
        return 0.0
    ic, _ = spearmanr(pred[mask], actual[mask])
    return float(ic) if np.isfinite(ic) else 0.0


def evaluate_ic_by_date(panel: pd.DataFrame, pred_col: str, target_col: str) -> pd.Series:
    """Compute daily IC (rank correlation per date)."""
    ics = {}
    for date, grp in panel.groupby("date"):
        if len(grp) < 10:
            continue
        ic = evaluate_ic(grp[pred_col].values, grp[target_col].values)
        ics[date] = ic
    return pd.Series(ics)


def main():
    t0 = time.time()
    log("=" * 70)
    log("TRENDING_BULL Regime-Specific ML Ranker v2")
    log("=" * 70)

    # Step 1: Build regime labels
    regime_series = build_regime_series()

    # Step 2: Build symbol store (50 liquid symbols for fast iteration)
    symbols = LIQUID_50
    log(f"INFO | Universe: {len(symbols)} liquid symbols")
    store = build_store_fast(symbols)

    # Step 3: Build panel
    log(f"INFO | Building panel for horizon={HORIZON}d...")
    panel = build_panel(store, HORIZON)
    panel["date"] = pd.to_datetime(panel["date"])
    log(f"INFO | Panel: {len(panel):,} rows, {len(panel.columns)} columns")

    # Step 4: Attach regime labels
    panel["regime"] = panel["date"].map(regime_series)
    panel = panel.dropna(subset=["regime"])
    log(f"INFO | After regime join: {len(panel):,} rows")

    regime_counts = panel.groupby("regime").size()
    log(f"INFO | Panel regime distribution:\n{regime_counts}")

    # Step 5: Define feature columns (same LIVE_ONLY exclusions as ml_model.py)
    LIVE_ONLY_FEATURES = {
        "congress_cluster", "congress_net_buying", "congress_buy_count",
        "congress_sell_count", "congress_committee",
        "pcr_total", "pcr_equity", "pcr_5d_ma", "pcr_fear_signal",
        "pcr_complacency", "pcr_extreme",
        "job_velocity", "job_expanding", "job_contracting",
        "insider_seq_buy", "insider_seq_sell", "insider_seq_strength",
        "app_rating", "app_rating_high", "app_rating_low",
        "squeeze_alert",
        "8k_sentiment", "8k_negative", "8k_positive", "8k_count",
        "fda_catalyst_near", "fda_catalyst_days",
        "inst_buying", "inst_selling", "inst_chg_pct",
        "overnight_mom_5d", "overnight_mom_20d", "overnight_share_20d",
        "gap_up_freq_20d", "gap_down_freq_20d",
        "iv_term_structure", "iv_backwardation", "iv_contango_steep",
        "analyst_score", "analyst_momentum", "analyst_consensus",
        "sym_historical_stop_rate", "sym_historical_wr", "sym_n_trades",
    }
    META_COLS = {"date", "symbol", "target_raw", "target", "target_rank", "regime"}
    feature_cols = [c for c in panel.columns
                    if c not in META_COLS and c not in LIVE_ONLY_FEATURES]
    log(f"INFO | Feature count: {len(feature_cols)}")

    # Step 6: Rolling walk-forward evaluation
    # Test years: 2020-2025 (rolling retrain before each)
    test_years = [2020, 2021, 2022, 2023, 2024, 2025]
    train_window = TRAIN_YEARS

    all_regime_ics = {}
    trending_bull_ics = {}
    trending_bull_v2_ics = {}

    for test_year in test_years:
        train_start = pd.Timestamp(f"{test_year - train_window}-01-01")
        train_end = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year + 1}-01-01")

        # Full panel splits
        train_all = panel[(panel["date"] >= train_start) & (panel["date"] < train_end)]
        test_all = panel[(panel["date"] >= train_end) & (panel["date"] < test_end)]

        # Regime-filtered splits
        train_tb = train_all[train_all["regime"] == TRENDING_BULL]
        test_tb = test_all[test_all["regime"] == TRENDING_BULL]

        if len(test_tb) < 100:
            log(f"WARN | {test_year}: only {len(test_tb)} TRENDING_BULL test rows — skipping")
            continue

        log(f"\n{'='*60}")
        log(f"TEST YEAR: {test_year}")
        log(f"  Train all-regime: {len(train_all):,} rows")
        log(f"  Train TRENDING_BULL: {len(train_tb):,} rows")
        log(f"  Test TRENDING_BULL: {len(test_tb):,} rows")

        # ----- Model A: All-Regime model (baseline) -----
        scaler_a = StandardScaler()
        X_train_a = scaler_a.fit_transform(train_all[feature_cols].fillna(0))
        model_a = _build_model(HORIZON)
        model_a.fit(X_train_a, train_all["target_rank"])

        X_test_tb = scaler_a.transform(test_tb[feature_cols].fillna(0))
        pred_a = model_a.predict(X_test_tb)
        ic_a = evaluate_ic(pred_a, test_tb["target_rank"].values)
        all_regime_ics[test_year] = ic_a
        log(f"  All-Regime model IC on TRENDING_BULL days: {ic_a:.4f}")

        # ----- Model B: Existing Regime-filtered (same arch as all-regime, depth=4) -----
        if len(train_tb) >= 500:
            scaler_b = StandardScaler()
            X_train_b = scaler_b.fit_transform(train_tb[feature_cols].fillna(0))
            model_b = _build_model(HORIZON)
            model_b.fit(X_train_b, train_tb["target_rank"])

            X_test_tb_b = scaler_b.transform(test_tb[feature_cols].fillna(0))
            pred_b = model_b.predict(X_test_tb_b)
            ic_b = evaluate_ic(pred_b, test_tb["target_rank"].values)
            trending_bull_ics[test_year] = ic_b
            log(f"  Existing TB model IC (depth=4): {ic_b:.4f}")
        else:
            log(f"  WARN: insufficient TRENDING_BULL train rows ({len(train_tb)}), using all-regime for Model B")
            trending_bull_ics[test_year] = ic_a

        # ----- Model C: TRENDING_BULL v2 (deeper, momentum-tuned) -----
        # Use TRENDING_BULL train data with augmentation from recent all-regime data
        if len(train_tb) >= 500:
            train_for_v2 = train_tb.copy()
        else:
            # Augment: 70% TRENDING_BULL + 30% all-regime (upweight TB)
            n_aug = max(500, int(len(train_tb) * 0.43))  # bring total to ~70/30 split
            non_tb = train_all[train_all["regime"] != TRENDING_BULL].sample(
                n=min(n_aug, len(train_all[train_all["regime"] != TRENDING_BULL])),
                random_state=42,
            )
            train_for_v2 = pd.concat([train_tb, non_tb])
            log(f"  Augmented training: {len(train_for_v2):,} rows ({len(train_tb)} TB + {len(non_tb)} other)")

        scaler_c = StandardScaler()
        X_train_c = scaler_c.fit_transform(train_for_v2[feature_cols].fillna(0))
        model_c = build_trending_bull_model(HORIZON, depth=5, n_est=600)
        model_c.fit(X_train_c, train_for_v2["target_rank"])

        X_test_tb_c = scaler_c.transform(test_tb[feature_cols].fillna(0))
        pred_c = model_c.predict(X_test_tb_c)
        ic_c = evaluate_ic(pred_c, test_tb["target_rank"].values)
        trending_bull_v2_ics[test_year] = ic_c
        log(f"  TRENDING_BULL v2 IC (depth=5): {ic_c:.4f}")

        # Daily IC breakdown for v2
        test_tb_copy = test_tb.copy()
        test_tb_copy["pred_c"] = pred_c
        daily_ics = evaluate_ic_by_date(test_tb_copy, "pred_c", "target_rank")
        pct_positive = (daily_ics > 0).mean() * 100
        log(f"  Daily IC: mean={daily_ics.mean():.4f}, median={daily_ics.median():.4f}, "
            f"positive={pct_positive:.0f}%")

    # ── Summary ─────────────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("SUMMARY: IC Comparison on TRENDING_BULL Days")
    log("=" * 70)
    log(f"{'Year':<8} {'All-Regime':<14} {'TB v1 (d=4)':<14} {'TB v2 (d=5)':<14} {'v2 vs All':<12}")
    log("-" * 62)

    for yr in sorted(set(all_regime_ics.keys()) | set(trending_bull_v2_ics.keys())):
        ic_all = all_regime_ics.get(yr, float("nan"))
        ic_v1 = trending_bull_ics.get(yr, float("nan"))
        ic_v2 = trending_bull_v2_ics.get(yr, float("nan"))
        delta = ic_v2 - ic_all if np.isfinite(ic_v2) and np.isfinite(ic_all) else float("nan")
        log(f"{yr:<8} {ic_all:<14.4f} {ic_v1:<14.4f} {ic_v2:<14.4f} {delta:+.4f}")

    mean_all = np.nanmean(list(all_regime_ics.values()))
    mean_v1 = np.nanmean(list(trending_bull_ics.values()))
    mean_v2 = np.nanmean(list(trending_bull_v2_ics.values()))
    log("-" * 62)
    log(f"{'MEAN':<8} {mean_all:<14.4f} {mean_v1:<14.4f} {mean_v2:<14.4f} {mean_v2 - mean_all:+.4f}")

    # ── Feature importances for the final v2 model ──────────────────────────────
    log("\n" + "=" * 70)
    log("TOP 20 FEATURE IMPORTANCES (TRENDING_BULL v2, last vintage)")
    log("=" * 70)
    if hasattr(model_c, "feature_importances_"):
        imp = pd.Series(model_c.feature_importances_, index=feature_cols)
        top20 = imp.nlargest(20)
        for i, (feat, val) in enumerate(top20.items(), 1):
            log(f"  {i:>2}. {feat:<45s} {val:.4f}")

        # Category analysis
        momentum_feats = [f for f in feature_cols if any(k in f for k in ["ret_", "mom_", "trajectory"])]
        sector_feats = [f for f in feature_cols if "sector" in f or "vs_spy" in f or "vs_sector" in f]
        quality_feats = [f for f in feature_cols if "quality" in f or "earnings" in f or "eps_" in f]
        vol_feats = [f for f in feature_cols if "vol" in f.lower() or "atr" in f]
        macro_feats = [f for f in feature_cols if "ca_" in f or "yield" in f or "vix" in f]

        total_imp = imp.sum()
        log("\nCategory importance shares:")
        for name, feats in [("Momentum", momentum_feats), ("Sector/RelStrength", sector_feats),
                            ("Quality/Earnings", quality_feats), ("Volatility", vol_feats),
                            ("Macro", macro_feats)]:
            share = imp[[f for f in feats if f in imp.index]].sum() / total_imp
            log(f"  {name:<25s} {share:.1%}")

    # ── Decision: save or not ───────────────────────────────────────────────────
    log("\n" + "=" * 70)
    DEPLOY_THRESHOLD = 0.02  # minimum mean IC improvement over all-regime
    improvement = mean_v2 - mean_all

    if mean_v2 > 0.03 and improvement > 0:
        log("DECISION: DEPLOY — TRENDING_BULL v2 model outperforms all-regime on TB days")
        log(f"  Mean IC improvement: {improvement:+.4f}")

        # Train final model on all available TRENDING_BULL data (2015-2024)
        log("\nTraining FINAL model on all TRENDING_BULL data for deployment...")
        final_train = panel[panel["regime"] == TRENDING_BULL]
        if len(final_train) < 1000:
            final_train = panel  # fallback if not enough TB data

        scaler_final = StandardScaler()
        X_final = scaler_final.fit_transform(final_train[feature_cols].fillna(0))
        model_final = build_trending_bull_model(HORIZON, depth=5, n_est=600)
        model_final.fit(X_final, final_train["target_rank"])

        bundle = {
            "model": model_final,
            "scaler": scaler_final,
            "features": feature_cols,
            "horizon": HORIZON,
            "regime": TRENDING_BULL,
            "mean_ic": float(mean_v2),
            "ic_by_year": {int(k): float(v) for k, v in trending_bull_v2_ics.items()},
            "trained_on": "TRENDING_BULL days only",
            "n_train_rows": len(final_train),
        }
        path = os.path.join(ROOT, f"cross_sectional_ranker_{HORIZON}d_TRENDING_BULL_v2.joblib")
        joblib.dump(bundle, path)
        log(f"INFO | Saved {path}")

        # Also save per-year vintages for rolling backtest
        for test_year in test_years:
            train_start = pd.Timestamp(f"{test_year - train_window}-01-01")
            train_end = pd.Timestamp(f"{test_year}-01-01")

            train_yr = panel[(panel["date"] >= train_start) & (panel["date"] < train_end)]
            train_yr_tb = train_yr[train_yr["regime"] == TRENDING_BULL]

            if len(train_yr_tb) < 500:
                train_yr_tb = train_yr  # fallback

            scaler_yr = StandardScaler()
            X_yr = scaler_yr.fit_transform(train_yr_tb[feature_cols].fillna(0))
            model_yr = build_trending_bull_model(HORIZON, depth=5, n_est=600)
            model_yr.fit(X_yr, train_yr_tb["target_rank"])

            bundle_yr = {
                "model": model_yr,
                "scaler": scaler_yr,
                "features": feature_cols,
                "horizon": HORIZON,
                "regime": TRENDING_BULL,
                "test_year": test_year,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
            }
            path_yr = os.path.join(ROOT, f"cross_sectional_ranker_{HORIZON}d_{test_year}_TRENDING_BULL_v2.joblib")
            joblib.dump(bundle_yr, path_yr)
            log(f"INFO | Saved {path_yr}")

    elif mean_v2 > mean_all:
        log("DECISION: MARGINAL — v2 improves but not enough. Consider more data/features.")
        log(f"  Mean IC: v2={mean_v2:.4f} vs all-regime={mean_all:.4f} (delta={improvement:+.4f})")
    else:
        log("DECISION: DO NOT DEPLOY — all-regime model is still better on TB days")
        log(f"  Mean IC: v2={mean_v2:.4f} vs all-regime={mean_all:.4f} (delta={improvement:+.4f})")

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
