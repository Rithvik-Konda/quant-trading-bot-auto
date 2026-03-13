"""
ml_model.py — Multi-Factor Cross-Sectional Ranker v3
=====================================================
Regime-robust signals. 150-stock universe. LightGBM-ready.

ROOT CAUSE OF 0.56% OOS CAGR:
1. 40 correlated tech stocks — ranking within one sector is noise
2. Pure price features — memorized 2015-2021 bull, failed 2022+
3. ML training contaminated — model had seen OOS data

WHAT WORKS ACROSS REGIMES (academic evidence 1980s-present):
- Momentum + Quality filter (not momentum alone)
- Analyst revision momentum (institutional flow signal)
- Short interest changes (bearish predictor)
- Jegadeesh-Titman 12-1 month momentum (canonical signal)
- Mean reversion 1-5d within momentum trend (Citadel approach)

MODEL:
- sklearn GBM with strong regularization (current)
- LightGBM drop-in after: pip install lightgbm
- Walk-forward split: train pre-2022, test 2022+ (truly blind)
"""

from __future__ import annotations

import argparse
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import config

CACHE_DIR    = "cache_prices"
USE_LIGHTGBM = False   # flip to True after: pip install lightgbm
os.makedirs(CACHE_DIR, exist_ok=True)

SECTOR_ETF_MAP: Dict[str, str] = {}
for _etf, _syms in getattr(config, "SECTOR_ETFS", {}).items():
    for _s in _syms:
        SECTOR_ETF_MAP[_s] = _etf

_etf_cache:   Dict[str, pd.DataFrame] = {}
_fund_cache:  Dict[str, dict]         = {}


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} | {msg}", flush=True)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_etf(ticker: str) -> Optional[pd.DataFrame]:
    if ticker in _etf_cache:
        return _etf_cache[ticker]
    path = os.path.join(CACHE_DIR, f"{ticker}_etf.csv")
    try:
        if os.path.exists(path):
            df  = pd.read_csv(path, index_col=0)
            idx = pd.to_datetime(df.index, utc=True, errors="coerce")
            df  = df.loc[~idx.isna()].copy()
            df.index   = idx[~idx.isna()].tz_convert("UTC").tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
        else:
            import yfinance as yf
            raw = yf.Ticker(ticker).history(period="12y", interval="1d")
            if raw is None or len(raw) == 0:
                return None
            if getattr(raw.index, "tz", None):
                raw.index = raw.index.tz_convert("UTC").tz_localize(None)
            raw.columns = [c.lower() for c in raw.columns]
            df = raw[[c for c in raw.columns if c in ["open","high","low","close","volume"]]]
            df.to_csv(path)
        _etf_cache[ticker] = df
        return df
    except Exception as e:
        log(f"WARN | ETF {ticker}: {e}")
        return None


def fetch_data(symbol: str, days: int = 3650, refresh: bool = False) -> Optional[pd.DataFrame]:
    path = os.path.join(CACHE_DIR, f"{symbol}_{days}d.csv")
    if not refresh and os.path.exists(path):
        df  = pd.read_csv(path, index_col=0)
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        df  = df.loc[~idx.isna()].copy()
        df.index   = idx[~idx.isna()].tz_convert("UTC").tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        log(f"INFO | Loaded {len(df)} cached bars for {symbol}")
        return df[["open","high","low","close","volume"]].dropna()
    import yfinance as yf
    df = yf.Ticker(symbol).history(period="12y", interval="1d")
    if df is None or len(df) == 0:
        return None
    if getattr(df.index, "tz", None):
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open","high","low","close","volume"]].dropna()
    df.to_csv(path)
    return df


def _get_fundamentals(symbol: str) -> dict:
    if symbol in _fund_cache:
        return _fund_cache[symbol]
    result = {}
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        if not isinstance(info, dict):
            info = {}
        for f in ["returnOnEquity","returnOnAssets","grossMargins","profitMargins",
                  "debtToEquity","currentRatio","revenueGrowth","earningsGrowth",
                  "forwardPE","pegRatio","shortPercentOfFloat","shortRatio",
                  "heldPercentInstitutions"]:
            v = info.get(f)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                result[f] = float(v)
    except Exception:
        pass
    _fund_cache[symbol] = result
    return result


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    high, low, close, open_, volume = df["high"], df["low"], df["close"], df["open"], df["volume"]
    feat = pd.DataFrame(index=df.index)

    # 1. Price momentum — multiple horizons
    for p in [1, 3, 5, 10, 20, 40, 60, 90, 120]:
        feat[f"ret_{p}"] = close.pct_change(p).clip(-2, 2)
    feat["mom_accel_5_20"]  = feat["ret_5"]  - feat["ret_20"]
    feat["mom_accel_20_60"] = feat["ret_20"] - feat["ret_60"]
    feat["mom_accel_5_60"]  = feat["ret_5"]  - feat["ret_60"]
    feat["mom_accel_1_5"]   = feat["ret_1"]  - feat["ret_5"]
    # Jegadeesh-Titman canonical momentum: 12-month return ex last month
    feat["mom_12_1"] = (close.pct_change(252) - close.pct_change(21)).clip(-2, 2)

    # 2. Trend structure
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    feat["px_vs_sma20"]     = (close / sma20.replace(0, np.nan)  - 1).clip(-1, 1)
    feat["px_vs_sma50"]     = (close / sma50.replace(0, np.nan)  - 1).clip(-1, 1)
    feat["px_vs_sma200"]    = (close / sma200.replace(0, np.nan) - 1).clip(-1, 1)
    feat["sma20_vs_sma50"]  = (sma20  / sma50.replace(0, np.nan)  - 1).clip(-0.5, 0.5)
    feat["sma50_vs_sma200"] = (sma50  / sma200.replace(0, np.nan) - 1).clip(-0.5, 0.5)
    feat["sma20_slope"]     = sma20.pct_change(5).clip(-0.2, 0.2)
    feat["sma50_slope"]     = sma50.pct_change(10).clip(-0.2, 0.2)
    feat["sma200_slope"]    = sma200.pct_change(20).clip(-0.2, 0.2)

    # 3. Oscillators
    for p in [7, 14, 21]:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(p).mean()
        loss  = (-delta.clip(upper=0)).rolling(p).mean()
        feat[f"rsi_{p}"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).clip(0, 100)
    feat["rsi_14_slope"] = feat["rsi_14"].diff(5)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"]      = (macd - sig)
    feat["macd_hist_norm"] = feat["macd_hist"] / close.replace(0, np.nan)
    feat["macd_hist_slope"]= feat["macd_hist"].diff(3)
    bb_mid  = close.rolling(20).mean()
    bb_std  = close.rolling(20).std()
    bb_rng  = (4 * bb_std).replace(0, np.nan)
    feat["bb_pct"]   = ((close - (bb_mid - 2*bb_std)) / bb_rng).clip(0, 1)
    feat["bb_width"] = (4 * bb_std / bb_mid.replace(0, np.nan)).clip(0, 1)

    # 4. Volatility
    ret1 = close.pct_change()
    feat["realized_vol_10"] = ret1.rolling(10).std() * np.sqrt(252)
    feat["realized_vol_20"] = ret1.rolling(20).std() * np.sqrt(252)
    feat["realized_vol_60"] = ret1.rolling(60).std() * np.sqrt(252)
    feat["vol_expansion"]   = (feat["realized_vol_10"] / feat["realized_vol_20"].replace(0, np.nan)).clip(0, 5)
    feat["vol_contracting"] = (feat["realized_vol_10"] < feat["realized_vol_20"]).astype(float)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    feat["atr_pct"] = (tr.rolling(14).mean() / close.replace(0, np.nan)).clip(0, 0.2)

    # 5. Volume / flow
    vol_ma20 = volume.rolling(20).mean()
    feat["vol_ratio_20"] = (volume  / vol_ma20.replace(0, np.nan)).clip(0, 10)
    feat["vol_ratio_5"]  = (volume.rolling(5).mean() / vol_ma20.replace(0, np.nan)).clip(0, 5)
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    feat["obv_slope_10"] = ((obv - obv.shift(10)) / (obv.shift(10).abs() + 1)).clip(-1, 1)
    feat["obv_slope_20"] = ((obv - obv.shift(20)) / (obv.shift(20).abs() + 1)).clip(-1, 1)
    mf_raw = ((close-low) - (high-close)) / (high-low).replace(0, np.nan) * volume
    feat["money_flow_14"] = (mf_raw.rolling(14).mean() / (vol_ma20 * close).replace(0, np.nan)).clip(-1, 1)

    # 6. Breakout / range
    h52 = close.rolling(252).max()
    l52 = close.rolling(252).min()
    feat["dist_to_52w_high"] = (close / h52.replace(0, np.nan) - 1).clip(-1, 0)
    feat["dist_to_52w_low"]  = (close / l52.replace(0, np.nan) - 1).clip(0, 5)
    feat["range_pos_52w"]    = ((close - l52) / (h52 - l52).replace(0, np.nan)).clip(0, 1)
    feat["dist_to_20d_high"] = (close / close.rolling(20).max().replace(0, np.nan) - 1).clip(-1, 0)
    feat["breakout_quality"] = (feat["dist_to_52w_high"].clip(-0.05,0)*-20 * feat["vol_ratio_20"].clip(1,3)/3).clip(0,1)

    # 7. Mean reversion 1-5d (uncorrelated with 20-60d momentum)
    feat["mean_rev_1d"]    = (-ret1).clip(-0.10, 0.10)
    feat["mean_rev_3d"]    = (-close.pct_change(3)).clip(-0.15, 0.15)
    feat["rsi_7_dip"]      = (50 - feat["rsi_7"]).clip(0, 50) / 50
    feat["bb_lower_touch"] = (1 - feat["bb_pct"]).clip(0, 1)
    feat["dip_in_trend"]   = (feat["ret_20"].clip(0,0.4) * feat["mean_rev_3d"].clip(0,0.15)).clip(0,0.1)

    # 8. Trend quality
    feat["trend_consistency"]    = (close.diff() > 0).rolling(10).mean()
    feat["trend_consistency_20"] = (close.diff() > 0).rolling(20).mean()
    cr = (high - low).replace(0, np.nan)
    feat["body_pct"]      = ((close-open_).abs() / cr).clip(0, 1)
    feat["close_to_high"] = ((high-close) / cr).clip(0, 1)
    feat["close_to_low"]  = ((close-low)  / cr).clip(0, 1)

    # 9. vs SPY
    spy_df = _etf_cache.get("SPY")
    if spy_df is None:
        spy_df = _load_etf("SPY")
    if spy_df is not None and len(spy_df) > 60:
        spy_c = spy_df["close"].reindex(close.index, method="ffill")
        for p in [5, 20, 60, 120]:
            feat[f"ret_{p}_vs_spy"] = (close.pct_change(p) - spy_c.pct_change(p)).clip(-1, 1)
        feat["mom_accel_vs_spy"] = (
            (close.pct_change(5) - spy_c.pct_change(5)) -
            (close.pct_change(20) - spy_c.pct_change(20))
        ).clip(-0.5, 0.5)
        rcov = close.pct_change().rolling(60).cov(spy_c.pct_change())
        rvar = spy_c.pct_change().rolling(60).var()
        feat["beta_60"] = (rcov / rvar.replace(0, np.nan)).clip(-3, 3)

    # 10. vs Sector ETF
    if symbol:
        etf = SECTOR_ETF_MAP.get(symbol.upper())
        if etf:
            etf_df = _load_etf(etf)
            if etf_df is not None and len(etf_df) > 60:
                ec = etf_df["close"].reindex(close.index, method="ffill")
                for p in [5, 20, 60]:
                    feat[f"ret_{p}_vs_sector"] = (close.pct_change(p) - ec.pct_change(p)).clip(-1, 1)
                feat["mom_accel_vs_sector"] = (
                    (close.pct_change(5)-ec.pct_change(5)) -
                    (close.pct_change(20)-ec.pct_change(20))
                ).clip(-0.5, 0.5)

    # 11. Quality fundamentals (regime-robust — institutional accumulation signal)
    if symbol:
        f = _get_fundamentals(symbol)
        mapping = [
            ("returnOnEquity",      "quality_roe",           (-1, 5)),
            ("returnOnAssets",      "quality_roa",           (-1, 2)),
            ("grossMargins",        "quality_gross_margin",  (-1, 1)),
            ("profitMargins",       "quality_profit_margin", (-1, 1)),
            ("debtToEquity",        "quality_debt_equity",   (0, 3)),   # normalized /200
            ("currentRatio",        "quality_current_ratio", (0, 5)),
            ("revenueGrowth",       "quality_rev_growth",    (-1, 3)),
            ("earningsGrowth",      "quality_earn_growth",   (-2, 5)),
        ]
        q_parts = []
        for src, dst, (lo, hi) in mapping:
            v = f.get(src)
            if v is not None:
                scaled = np.clip(v / 200 if src == "debtToEquity" else v, lo, hi)
                feat[dst] = scaled
                if src in ("returnOnEquity", "grossMargins", "revenueGrowth"):
                    q_parts.append(np.clip(v, 0, 1))
        if f.get("pegRatio", 0) > 0:
            feat["quality_peg"] = np.clip(1 / f["pegRatio"], 0, 3)
        if q_parts:
            feat["quality_composite"] = float(np.mean(q_parts))

    # 12. Short interest (bearish signal — rising SI predicts underperformance)
    if symbol:
        f = _get_fundamentals(symbol)
        si = f.get("shortPercentOfFloat")
        sr = f.get("shortRatio")
        if si is not None: feat["short_pct_float"] = np.clip(si, 0, 0.5)
        if sr is not None: feat["short_ratio"]     = np.clip(sr / 20, 0, 1)

    # 13. Regime features
    feat["vol_regime_score"] = (1 - feat["realized_vol_20"].clip(0, 0.8) / 0.8)
    feat["dd_recovery"]      = (close / close.rolling(60).max().replace(0, np.nan)).clip(0.5, 1)
    feat["trend_age"]        = (close > sma20).astype(int).rolling(60).sum() / 60

    # 14. Vol-adjusted momentum — raw return divided by realized vol
    # A 3% move in a 15-vol stock >> 3% move in a 40-vol stock
    # Without this the model can't distinguish signal from noise in momentum
    # vol_base: use realized_vol_10 for short horizons, realized_vol_20 for longer
    _vol_map = {5: "realized_vol_10", 20: "realized_vol_20", 60: "realized_vol_20"}
    for p in [5, 20, 60]:
        vol_base = feat[_vol_map[p]].replace(0, np.nan)
        feat[f"vol_adj_ret_{p}"] = (feat[f"ret_{p}"] / (vol_base / np.sqrt(252) * np.sqrt(p))).clip(-5, 5)

    # 15. Momentum trajectory — is momentum accelerating or decelerating?
    # A stock at rank 85 that was rank 70 last week is accelerating (buy signal)
    # A stock at rank 85 that was rank 95 last week is decelerating (warning)
    # Computed as slope of 5d return over rolling 10/20 day windows
    feat["mom_trajectory_10"] = feat["ret_5"].diff(10).clip(-0.3, 0.3)
    feat["mom_trajectory_20"] = feat["ret_5"].diff(20).clip(-0.5, 0.5)
    feat["mom_vs_spy_trajectory"] = (
        feat.get("ret_5_vs_spy", pd.Series(0, index=feat.index)).diff(10)
    ).clip(-0.3, 0.3)

    # 16. Persistence signals — consecutive days above/below key levels
    # Persistent weakness (20+ days below 200MA) = structural bear, not a dip
    # This is what distinguishes a genuine short from a temporarily weak stock
    feat["days_above_sma20"]  = (close > sma20).astype(int).rolling(20).sum() / 20
    feat["days_above_sma200"] = (close > sma200).astype(int).rolling(60).sum() / 60
    feat["days_below_sma200"] = (close < sma200).astype(int).rolling(60).sum() / 60
    # Persistent weakness: in bottom half of performance for 20 consecutive days
    below_median_20d = (feat["ret_20"] < 0).astype(int)
    feat["persistent_weakness"] = below_median_20d.rolling(20).mean()
    feat["persistent_strength"] = (feat["ret_20"] > 0).astype(int).rolling(20).mean()

    # 17. Sector relative strength — is this stock's sector leading or lagging?
    # Key for short selection: don't short stocks in leading sectors.
    # The model previously had no way to know energy was leading in 2022.
    if symbol:
        etf = SECTOR_ETF_MAP.get(symbol.upper())
        if etf:
            etf_df = _etf_cache.get(etf)
            if etf_df is None:
                etf_df = _load_etf(etf)
            if etf_df is not None and len(etf_df) > 60:
                spy_df2 = _etf_cache.get("SPY")
                if spy_df2 is None:
                    spy_df2 = _load_etf("SPY")
                if spy_df2 is not None:
                    ec  = etf_df["close"].reindex(close.index, method="ffill")
                    sc2 = spy_df2["close"].reindex(close.index, method="ffill")
                    # Sector vs SPY over 20/60 days — positive = sector leading
                    feat["sector_vs_spy_20"]  = (ec.pct_change(20) - sc2.pct_change(20)).clip(-0.5, 0.5)
                    feat["sector_vs_spy_60"]  = (ec.pct_change(60) - sc2.pct_change(60)).clip(-0.5, 0.5)
                    # Is sector in uptrend vs market?
                    feat["sector_leading"]    = (feat["sector_vs_spy_60"] > 0).astype(float)
                    # Stock strength within its own sector
                    feat["stock_vs_sector_20"] = (close.pct_change(20) - ec.pct_change(20)).clip(-0.5, 0.5)

    return feat.replace([np.inf, -np.inf], np.nan)


# ── Forward return ────────────────────────────────────────────────────────────

def compute_forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    return (df["close"].shift(-horizon) / df["open"].shift(-1)) - 1


# ── Panel builder ─────────────────────────────────────────────────────────────

def build_symbol_store(symbols: List[str], days: int, refresh: bool = False) -> dict:
    store = {}
    for sym in symbols:
        df = fetch_data(sym, days, refresh)
        if df is None or len(df) < 260:
            continue
        feat = compute_features(df, symbol=sym)
        store[sym] = {"prices": df, "features": feat}
    return store


def build_panel_from_store(store: dict, horizon: int) -> pd.DataFrame:
    frames    = []
    spy       = fetch_data(config.BENCHMARK_SYMBOL)
    spy_close = spy["close"] if spy is not None else None

    for sym in config.WATCHLIST:
        if sym not in store:
            continue
        df   = store[sym]["prices"]
        feat = store[sym]["features"].copy()
        target_raw = compute_forward_return(df, horizon)

        # Canonical SPY-relative features
        if spy_close is not None:
            common = feat.index.intersection(spy_close.index)
            for p in [5, 20, 60]:
                if f"ret_{p}" in feat.columns:
                    feat.loc[common, f"ret_{p}_vs_spy"] = (
                        feat.loc[common, f"ret_{p}"] -
                        spy_close.loc[common].pct_change(p)
                    )

        merged           = pd.concat([feat, target_raw.rename("target_raw")], axis=1)
        merged["symbol"] = sym
        merged["date"]   = merged.index
        frames.append(merged)

    if not frames:
        raise RuntimeError("No symbols loaded")

    panel = pd.concat(frames)
    panel["target_rank"] = panel.groupby("date")["target_raw"].rank(pct=True)

    feature_cols = [c for c in panel.columns
                    if c not in {"date", "symbol", "target_raw", "target_rank"}]
    for c in feature_cols:
        panel[f"{c}_cs_rank"] = panel.groupby("date")[c].rank(pct=True)

    return panel.dropna().reset_index(drop=True)


# ── Model ─────────────────────────────────────────────────────────────────────

def _build_model(horizon: int):
    if USE_LIGHTGBM:
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.02, max_depth=4,
                num_leaves=15, min_child_samples=50, subsample=0.6,
                colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42+horizon, n_jobs=-1, verbose=-1,
            )
            log(f"INFO | Using LightGBM (horizon={horizon}d)")
            return model
        except ImportError:
            log("WARN | LightGBM not installed — pip install lightgbm")

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=2,
        min_samples_leaf=50, subsample=0.6, max_features=0.5,
        random_state=42+horizon,
    )
    log(f"INFO | Using sklearn GBM (horizon={horizon}d)")
    return model


def train_ranker(panel: pd.DataFrame, horizon: int, oos_only: bool = False):
    feature_cols = [c for c in panel.columns
                    if c not in {"date", "symbol", "target_raw", "target_rank"}]

    panel        = panel.sort_values(["date", "symbol"])
    unique_dates = sorted(panel["date"].unique())
    oos_cutoff   = pd.Timestamp("2022-01-01")
    n_pre        = sum(1 for d in unique_dates if pd.Timestamp(d) < oos_cutoff)

    split_date = oos_cutoff if n_pre >= int(len(unique_dates) * 0.3) else unique_dates[int(len(unique_dates)*0.7)]
    log(f"INFO | Walk-forward split at {pd.Timestamp(split_date).date()}")

    train_df = panel[panel["date"] <  split_date]
    test_df  = panel[panel["date"] >= split_date]
    log(f"INFO | Train={len(train_df):,}  Test={len(test_df):,}")

    # Validate on OOS window
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(train_df[feature_cols].fillna(0))
    X_test_s  = scaler.transform(test_df[feature_cols].fillna(0))

    model = _build_model(horizon)
    model.fit(X_train_s, train_df["target_rank"])

    pred = model.predict(X_test_s)
    rmse = float(np.sqrt(mean_squared_error(test_df["target_rank"], pred)))
    ic   = float(np.corrcoef(pred, test_df["target_rank"])[0, 1])
    log(f"INFO | horizon={horizon}d  RMSE={rmse:.4f}  IC={ic:.4f}")

    if hasattr(model, "feature_importances_"):
        imp   = pd.Series(model.feature_importances_, index=feature_cols)
        top10 = imp.nlargest(10)
        log("INFO | Top features: " + ", ".join(f"{n}={v:.3f}" for n, v in top10.items()))
        cs_share = imp[[c for c in feature_cols if "_cs_rank" in c]].sum()
        log(f"INFO | CS-rank share: {cs_share:.1%} ({'OK' if cs_share < 0.6 else 'HIGH'})")

    if abs(ic) < 0.02:
        log(f"WARN | IC={ic:.4f} is near zero — limited OOS predictive power")

    # CRITICAL: retrain on ALL data for the live model.
    # Walk-forward split above is for validation only.
    # A model frozen at 2021 has never seen 2022-2025 patterns.
    if not oos_only:
        log(f"INFO | Retraining on full dataset ({len(panel):,} rows) for live model...")
        scaler_full   = StandardScaler()
        X_full        = scaler_full.fit_transform(panel[feature_cols].fillna(0))
        model_full    = _build_model(horizon)
        model_full.fit(X_full, panel["target_rank"])
        log(f"INFO | Full retrain complete.")
        return model_full, scaler_full, feature_cols

    return model, scaler, feature_cols


def train_and_save_ensemble(symbols: List[str], days: int, refresh: bool = False) -> None:
    log(f"INFO | Universe: {len(symbols)} stocks")
    log(f"INFO | Model: {'LightGBM' if USE_LIGHTGBM else 'sklearn GBM'}")
    store = build_symbol_store(symbols, days, refresh=refresh)
    log(f"INFO | Loaded {len(store)} symbols")

    for horizon in [3, 5, 7]:
        log(f"INFO | Building panel horizon={horizon}d …")
        panel  = build_panel_from_store(store, horizon)
        n_feat = len([c for c in panel.columns if c not in {"date","symbol","target_raw","target_rank"}])
        log(f"INFO | Panel rows={len(panel):,}  features={n_feat}")

        model, scaler, features = train_ranker(panel, horizon)
        bundle = {"model": model, "scaler": scaler, "features": features, "horizon": horizon}
        path   = f"cross_sectional_ranker_{horizon}d.joblib"
        joblib.dump(bundle, path)
        log(f"INFO | Saved {path}")

    joblib.dump(joblib.load("cross_sectional_ranker_5d.joblib"), "cross_sectional_ranker.joblib")
    log("INFO | Complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",          type=int, default=3650)
    parser.add_argument("--save-ensemble", action="store_true")
    parser.add_argument("--refresh",       action="store_true")
    parser.add_argument("--lightgbm",      action="store_true")
    args = parser.parse_args()

    global USE_LIGHTGBM
    if args.lightgbm:
        USE_LIGHTGBM = True

    if args.save_ensemble:
        train_and_save_ensemble(list(config.WATCHLIST), args.days, args.refresh)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()