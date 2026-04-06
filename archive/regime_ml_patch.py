"""
Patch to add regime-specific ML training to the quant trading bot.

Changes:
1. ml_model.py     — add train_rolling_regime_ensemble()
2. strategy_core.py — add load_ranker_regime_ensemble()
3. backtester_v2.py — swap rankers based on _current_regime

Run this script from ~/ai_trading_bot_v2 to apply all patches.
"""

import os, sys

BASE = os.path.expanduser("~/ai_trading_bot_v2")

# ── 1. ml_model.py — add train_rolling_regime_ensemble() ─────────────────────

ML_MODEL = os.path.join(BASE, "ml_model.py")

REGIME_TRAIN_FN = '''

def train_rolling_regime_ensemble(symbols: List[str], days: int, refresh: bool = False) -> None:
    """
    Train three separate LightGBM rankers — one per regime — for each
    horizon and each test year.

    Motivation:
      The all-regime model averages conflicting signals across TRENDING_BULL,
      CHOPPY, and BEAR days. IC is near zero in 2021-2023 because the model
      learned momentum features that work in TRENDING_BULL but not CHOPPY.

      A CHOPPY-specific model learns that quality, earnings stability, and
      low-vol matter more than raw momentum. A BEAR model learns defensive
      and short signals. Each model is sharper because it trains on days
      where its signal type actually works.

    Output files (per horizon per year per regime):
      cross_sectional_ranker_5d_2023_TRENDING_BULL.joblib
      cross_sectional_ranker_5d_2023_CHOPPY.joblib
      cross_sectional_ranker_5d_2023_BEAR.joblib
      ... etc for horizons 3d, 7d and years 2020-2025

    Live models (most recent vintage, used by backtester):
      cross_sectional_ranker_5d_TRENDING_BULL.joblib
      cross_sectional_ranker_5d_CHOPPY.joblib
      cross_sectional_ranker_5d_BEAR.joblib
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2"))
    from regime_classifier import RegimeClassifier, compute_signals, load_macro_data
    from regime_classifier import TRENDING_BULL, CHOPPY, BEAR

    log("INFO | === REGIME-SPECIFIC ML TRAINING ===")
    log(f"INFO | Universe: {len(symbols)} stocks")

    # Load regime labels for every trading date
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_prices")
    spy_macro, hyg_macro, vix_macro = load_macro_data(cache_dir=cache_dir)
    clf = RegimeClassifier()

    log("INFO | Building regime label series...")
    import pandas as _pd
    spy_dates = spy_macro.index.sort_values()
    regime_labels: dict = {}
    for d in spy_dates:
        signals = compute_signals(spy_macro, hyg_macro, vix_macro, as_of_date=d)
        if signals:
            regime_labels[d] = clf.update(d, signals)
    regime_series = _pd.Series(regime_labels)
    log(f"INFO | Regime counts: { {r: int((regime_series==r).sum()) for r in [TRENDING_BULL, CHOPPY, BEAR]} }")

    # Build full data store (same as rolling retrain)
    store = build_symbol_store(symbols, days, refresh=refresh)
    log(f"INFO | Loaded {len(store)} symbols")

    regimes    = [TRENDING_BULL, CHOPPY, BEAR]
    test_years = [2020, 2021, 2022, 2023, 2024, 2025]
    train_years = 5

    for horizon in [3, 5, 7]:
        log(f"INFO | === Horizon {horizon}d ===")
        panel = build_panel_from_store(store, horizon)
        panel["date"] = _pd.to_datetime(panel["date"])

        feature_cols = [c for c in panel.columns
                        if c not in {"date", "symbol", "target_raw", "target", "target_rank"}]

        # Attach regime label to each row
        panel["regime"] = panel["date"].map(regime_series)
        panel = panel.dropna(subset=["regime"])

        for test_year in test_years:
            train_start = _pd.Timestamp(f"{test_year - train_years}-01-01")
            train_end   = _pd.Timestamp(f"{test_year}-01-01")
            test_end    = _pd.Timestamp(f"{test_year + 1}-01-01")

            train_all = panel[(panel["date"] >= train_start) & (panel["date"] < train_end)]
            test_df   = panel[(panel["date"] >= train_end)   & (panel["date"] < test_end)]

            if len(train_all) < 1000:
                log(f"WARN | {test_year}: insufficient train data, skipping")
                continue

            for regime in regimes:
                regime_short = regime.replace("_", "")[:4].upper()

                # Train only on days classified as this regime
                train_regime = train_all[train_all["regime"] == regime]

                if len(train_regime) < 500:
                    log(f"WARN | {test_year}/{regime}: only {len(train_regime)} rows — using all-regime fallback")
                    train_regime = train_all  # fallback to all-regime data

                log(f"INFO | {test_year}/{regime}: {len(train_regime):,} train rows")

                scaler  = StandardScaler()
                X_train = scaler.fit_transform(train_regime[feature_cols].fillna(0))
                model   = _build_model(horizon)
                model.fit(X_train, train_regime["target_rank"])

                # Evaluate on regime-specific test days
                test_regime = test_df[test_df["regime"] == regime]
                if len(test_regime) > 50:
                    X_test = scaler.transform(test_regime[feature_cols].fillna(0))
                    pred   = model.predict(X_test)
                    ic     = float(np.corrcoef(pred, test_regime["target_rank"])[0, 1])
                    log(f"INFO | {test_year}/{regime} OOS IC={ic:.4f}")

                bundle = {
                    "model":    model,
                    "scaler":   scaler,
                    "features": feature_cols,
                    "horizon":  horizon,
                    "regime":   regime,
                    "train_start": str(train_start.date()),
                    "test_year":   test_year,
                }
                path = f"cross_sectional_ranker_{horizon}d_{test_year}_{regime}.joblib"
                joblib.dump(bundle, path)
                log(f"INFO | Saved {path}")

        # Save most recent vintage as live model for each regime
        latest_year = max(test_years)
        for regime in regimes:
            src  = f"cross_sectional_ranker_{horizon}d_{latest_year}_{regime}.joblib"
            dst  = f"cross_sectional_ranker_{horizon}d_{regime}.joblib"
            if os.path.exists(src):
                joblib.dump(joblib.load(src), dst)
                log(f"INFO | Updated live model {dst} from {latest_year} vintage")

    log("INFO | Regime ML training complete.")
'''

with open(ML_MODEL, "r") as f:
    ml_content = f.read()

# Insert before def main()
if "train_rolling_regime_ensemble" in ml_content:
    print("ml_model.py: regime function already present, skipping")
else:
    insert_before = "def main():"
    if insert_before in ml_content:
        ml_content = ml_content.replace(insert_before, REGIME_TRAIN_FN + "\n\n" + insert_before)
        with open(ML_MODEL, "w") as f:
            f.write(ml_content)
        print("ml_model.py: regime training function added")
    else:
        print("ERROR: could not find insertion point in ml_model.py")
        sys.exit(1)


# ── 2. strategy_core.py — add load_ranker_regime_ensemble() ──────────────────

STRATEGY_CORE = os.path.join(BASE, "strategy_core.py")

REGIME_LOAD_FN = '''

def load_ranker_regime_ensemble() -> dict:
    """
    Load regime-specific model sets.

    Returns nested dict:
        {
          "TRENDING_BULL": {3: bundle, 5: bundle, 7: bundle},
          "CHOPPY":        {3: bundle, 5: bundle, 7: bundle},
          "BEAR":          {3: bundle, 5: bundle, 7: bundle},
        }

    Falls back to the all-regime model if regime-specific file missing.
    """
    regimes = ["TRENDING_BULL", "CHOPPY", "BEAR"]
    result  = {}
    for regime in regimes:
        regime_rankers = {}
        for h in [3, 5, 7]:
            path = f"cross_sectional_ranker_{h}d_{regime}.joblib"
            fallback = f"cross_sectional_ranker_{h}d.joblib"
            if os.path.exists(path):
                regime_rankers[h] = load_ranker(path)
            elif os.path.exists(fallback):
                log(f"WARN | {path} missing — falling back to all-regime model")
                regime_rankers[h] = load_ranker(fallback)
        result[regime] = regime_rankers
    return result
'''

with open(STRATEGY_CORE, "r") as f:
    sc_content = f.read()

if "load_ranker_regime_ensemble" in sc_content:
    print("strategy_core.py: loader already present, skipping")
else:
    # Insert after load_ranker_ensemble_for_year function
    insert_after = "log(\"INFO | Rolling retrain complete.\")"
    # Find a safe insertion point — after load_ranker_ensemble_for_year
    marker = "def load_ranker_ensemble_for_year"
    if marker in sc_content:
        # Find end of that function by finding next def
        idx = sc_content.find(marker)
        next_def = sc_content.find("\ndef ", idx + 1)
        if next_def > 0:
            sc_content = sc_content[:next_def] + "\n" + REGIME_LOAD_FN + sc_content[next_def:]
            with open(STRATEGY_CORE, "w") as f:
                f.write(sc_content)
            print("strategy_core.py: regime loader added")
        else:
            print("ERROR: could not find insertion point in strategy_core.py")
    else:
        print("ERROR: load_ranker_ensemble_for_year not found in strategy_core.py")


# ── 3. backtester_v2.py — load regime ensemble and swap rankers ───────────────

BACKTESTER = os.path.join(BASE, "v2/backtester_v2.py")

with open(BACKTESTER, "r") as f:
    bt_content = f.read()

# Add import of load_ranker_regime_ensemble
old_import = "from strategy_core import (\n    adaptive_stop_pct, compute_atr_pct, load_ranker_ensemble,"
new_import = "from strategy_core import (\n    adaptive_stop_pct, compute_atr_pct, load_ranker_ensemble, load_ranker_regime_ensemble,"

if "load_ranker_regime_ensemble" not in bt_content:
    if old_import in bt_content:
        bt_content = bt_content.replace(old_import, new_import)
        print("backtester_v2.py: import added")
    else:
        print("WARNING: could not find import block — checking alternative")
        # Try a softer match
        if "load_ranker_ensemble," in bt_content:
            bt_content = bt_content.replace(
                "load_ranker_ensemble,",
                "load_ranker_ensemble, load_ranker_regime_ensemble,"
            )
            print("backtester_v2.py: import added (alternative match)")
else:
    print("backtester_v2.py: import already present")

# Load regime ensemble after regular ensemble
old_ensemble_load = '''    print("[prep] loading ML ensemble...", flush=True)
    rankers = load_ranker_ensemble()
    feat_cols_union = sorted(set(
        list(rankers[3]["features"]) +
        list(rankers[5]["features"]) +
        list(rankers[7]["features"])
    ))
    print(f"[ok]   ensemble loaded ({len(feat_cols_union)} features)", flush=True)'''

new_ensemble_load = '''    print("[prep] loading ML ensemble...", flush=True)
    rankers = load_ranker_ensemble()
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
        print("[warn] regime models not found — using all-regime model for all regimes", flush=True)'''

if "regime_rankers" not in bt_content:
    if old_ensemble_load in bt_content:
        bt_content = bt_content.replace(old_ensemble_load, new_ensemble_load)
        print("backtester_v2.py: regime ensemble loading added")
    else:
        print("WARNING: could not find ensemble load block exactly")

# Swap rankers based on regime in the ML scoring section
old_ml_score = "        ml_scores = batch_ml_scores_fast(X, valid_syms, rankers, feat_cols_union)"
new_ml_score = '''        # Use regime-specific model if available, else fall back to all-regime
        active_rankers = regime_rankers.get(_current_regime, rankers) if regime_rankers else rankers
        ml_scores = batch_ml_scores_fast(X, valid_syms, active_rankers, feat_cols_union)'''

if "active_rankers" not in bt_content:
    if old_ml_score in bt_content:
        bt_content = bt_content.replace(old_ml_score, new_ml_score)
        print("backtester_v2.py: regime-aware scoring added")
    else:
        print("WARNING: could not find ml_scores line exactly")

with open(BACKTESTER, "w") as f:
    bt_content_written = bt_content
    f.write(bt_content)

print("\nAll patches complete.")
print("\nNext steps:")
print("1. Train regime models:  cd ~/ai_trading_bot_v2 && caffeinate -i /opt/homebrew/bin/python3.11 ml_model.py --regime")
print("2. Run backtest:         cd ~/ai_trading_bot_v2 && caffeinate -i /opt/homebrew/bin/python3.11 v2/backtester_v2.py --oos")
