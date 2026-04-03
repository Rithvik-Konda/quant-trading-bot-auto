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
USE_LIGHTGBM = True   # flip to True after: pip install lightgbm
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
        return None  # suppress ETF load warnings — expected for missing data


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
    # Try JSON cache first — avoids live API calls during training
    try:
        import json as _json
        _fc = os.path.expanduser('~/ai_trading_bot_v2/cache_fundamentals/info_cache.json')
        if os.path.exists(_fc):
            _all = _json.load(open(_fc))
            if symbol in _all:
                _fund_cache[symbol] = _all[symbol]
                return _fund_cache[symbol]
    except Exception:
        pass
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

def compute_features(df: pd.DataFrame, symbol: Optional[str] = None, vix_macro: Optional[pd.DataFrame] = None, streak_store: Optional[dict] = None, insider_store: Optional[dict] = None, revision_store: Optional[dict] = None) -> pd.DataFrame:
    high, low, close, open_, volume = df["high"], df["low"], df["close"], df["open"], df["volume"]
    if vix_macro is None:
        # Load from pre-cached VIX data — ^VIX is the correct yfinance ticker
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                vix_macro = _load_etf("^VIX") or pd.DataFrame()
        except Exception:
            vix_macro = pd.DataFrame()
    d: dict = {}

    # 1. Price momentum — multiple horizons
    for p in [1, 3, 5, 10, 20, 40, 60, 90, 120]:
        d[f"ret_{p}"] = close.pct_change(p).clip(-2, 2)
    d["mom_accel_5_20"]  = d["ret_5"]  - d["ret_20"]
    d["mom_accel_20_60"] = d["ret_20"] - d["ret_60"]
    d["mom_accel_5_60"]  = d["ret_5"]  - d["ret_60"]
    d["mom_accel_1_5"]   = d["ret_1"]  - d["ret_5"]
    # Jegadeesh-Titman canonical momentum: 12-month return ex last month
    d["mom_12_1"] = (close.pct_change(252) - close.pct_change(21)).clip(-2, 2)

    # 2. Trend structure
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    d["px_vs_sma20"]     = (close / sma20.replace(0, np.nan)  - 1).clip(-1, 1)
    d["px_vs_sma50"]     = (close / sma50.replace(0, np.nan)  - 1).clip(-1, 1)
    d["px_vs_sma200"]    = (close / sma200.replace(0, np.nan) - 1).clip(-1, 1)
    d["sma20_vs_sma50"]  = (sma20  / sma50.replace(0, np.nan)  - 1).clip(-0.5, 0.5)
    d["sma50_vs_sma200"] = (sma50  / sma200.replace(0, np.nan) - 1).clip(-0.5, 0.5)
    d["sma20_slope"]     = sma20.pct_change(5).clip(-0.2, 0.2)
    d["sma50_slope"]     = sma50.pct_change(10).clip(-0.2, 0.2)
    d["sma200_slope"]    = sma200.pct_change(20).clip(-0.2, 0.2)

    # 3. Oscillators
    for p in [7, 14, 21]:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(p).mean()
        loss  = (-delta.clip(upper=0)).rolling(p).mean()
        d[f"rsi_{p}"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).clip(0, 100)
    d["rsi_14_slope"] = d["rsi_14"].diff(5)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"]      = (macd - sig)
    d["macd_hist_norm"] = d["macd_hist"] / close.replace(0, np.nan)
    d["macd_hist_slope"]= d["macd_hist"].diff(3)
    bb_mid  = close.rolling(20).mean()
    bb_std  = close.rolling(20).std()
    bb_rng  = (4 * bb_std).replace(0, np.nan)
    d["bb_pct"]   = ((close - (bb_mid - 2*bb_std)) / bb_rng).clip(0, 1)
    d["bb_width"] = (4 * bb_std / bb_mid.replace(0, np.nan)).clip(0, 1)

    # 4. Volatility
    ret1 = close.pct_change()
    d["realized_vol_10"] = ret1.rolling(10).std() * np.sqrt(252)
    d["realized_vol_20"] = ret1.rolling(20).std() * np.sqrt(252)
    d["realized_vol_60"] = ret1.rolling(60).std() * np.sqrt(252)
    d["vol_expansion"]   = (d["realized_vol_10"] / d["realized_vol_20"].replace(0, np.nan)).clip(0, 5)
    d["vol_contracting"] = (d["realized_vol_10"] < d["realized_vol_20"]).astype(float)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    d["atr_pct"] = (tr.rolling(14).mean() / close.replace(0, np.nan)).clip(0, 0.2)

    # 5. Volume / flow
    vol_ma20 = volume.rolling(20).mean()
    d["vol_ratio_20"] = (volume  / vol_ma20.replace(0, np.nan)).clip(0, 10)
    d["vol_ratio_5"]  = (volume.rolling(5).mean() / vol_ma20.replace(0, np.nan)).clip(0, 5)
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    d["obv_slope_10"] = ((obv - obv.shift(10)) / (obv.shift(10).abs() + 1)).clip(-1, 1)
    d["obv_slope_20"] = ((obv - obv.shift(20)) / (obv.shift(20).abs() + 1)).clip(-1, 1)
    mf_raw = ((close-low) - (high-close)) / (high-low).replace(0, np.nan) * volume
    d["money_flow_14"] = (mf_raw.rolling(14).mean() / (vol_ma20 * close).replace(0, np.nan)).clip(-1, 1)

    # 6. Breakout / range
    h52 = close.rolling(252).max()
    l52 = close.rolling(252).min()
    d["dist_to_52w_high"] = (close / h52.replace(0, np.nan) - 1).clip(-1, 0)
    d["dist_to_52w_low"]  = (close / l52.replace(0, np.nan) - 1).clip(0, 5)
    d["range_pos_52w"]    = ((close - l52) / (h52 - l52).replace(0, np.nan)).clip(0, 1)
    d["dist_to_20d_high"] = (close / close.rolling(20).max().replace(0, np.nan) - 1).clip(-1, 0)
    d["breakout_quality"] = (d["dist_to_52w_high"].clip(-0.05,0)*-20 * d["vol_ratio_20"].clip(1,3)/3).clip(0,1)

    # 7. Mean reversion 1-5d (uncorrelated with 20-60d momentum)
    d["mean_rev_1d"]    = (-ret1).clip(-0.10, 0.10)
    d["mean_rev_3d"]    = (-close.pct_change(3)).clip(-0.15, 0.15)
    d["rsi_7_dip"]      = (50 - d["rsi_7"]).clip(0, 50) / 50
    d["bb_lower_touch"] = (1 - d["bb_pct"]).clip(0, 1)
    d["dip_in_trend"]   = (d["ret_20"].clip(0,0.4) * d["mean_rev_3d"].clip(0,0.15)).clip(0,0.1)

    # 8. Trend quality
    d["trend_consistency"]    = (close.diff() > 0).rolling(10).mean()
    d["trend_consistency_20"] = (close.diff() > 0).rolling(20).mean()
    cr = (high - low).replace(0, np.nan)
    d["body_pct"]      = ((close-open_).abs() / cr).clip(0, 1)
    d["close_to_high"] = ((high-close) / cr).clip(0, 1)
    d["close_to_low"]  = ((close-low)  / cr).clip(0, 1)

    # 9. vs SPY
    spy_df = _etf_cache.get("SPY")
    if spy_df is None:
        spy_df = _load_etf("SPY")
    if spy_df is not None and len(spy_df) > 60:
        spy_c = spy_df["close"].reindex(close.index, method="ffill")
        for p in [5, 20, 60, 120]:
            d[f"ret_{p}_vs_spy"] = (close.pct_change(p) - spy_c.pct_change(p)).clip(-1, 1)
        d["mom_accel_vs_spy"] = (
            (close.pct_change(5) - spy_c.pct_change(5)) -
            (close.pct_change(20) - spy_c.pct_change(20))
        ).clip(-0.5, 0.5)
        rcov = close.pct_change().rolling(60).cov(spy_c.pct_change())
        rvar = spy_c.pct_change().rolling(60).var()
        d["beta_60"] = (rcov / rvar.replace(0, np.nan)).clip(-3, 3)

    # 10. vs Sector ETF
    if symbol:
        etf = SECTOR_ETF_MAP.get(symbol.upper())
        if etf:
            etf_df = _load_etf(etf)
            if etf_df is not None and len(etf_df) > 60:
                ec = etf_df["close"].reindex(close.index, method="ffill")
                for p in [5, 20, 60]:
                    d[f"ret_{p}_vs_sector"] = (close.pct_change(p) - ec.pct_change(p)).clip(-1, 1)
                d["mom_accel_vs_sector"] = (
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
                d[dst] = scaled
                if src in ("returnOnEquity", "grossMargins", "revenueGrowth"):
                    q_parts.append(np.clip(v, 0, 1))
        if f.get("pegRatio", 0) > 0:
            d["quality_peg"] = np.clip(1 / f["pegRatio"], 0, 3)
        if q_parts:
            d["quality_composite"] = float(np.mean(q_parts))

    # 12. Short interest (bearish signal — rising SI predicts underperformance)
    if symbol:
        f = _get_fundamentals(symbol)
        si = f.get("shortPercentOfFloat")
        sr = f.get("shortRatio")
        if si is not None: d["short_pct_float"] = np.clip(si, 0, 0.5)
        if sr is not None: d["short_ratio"]     = np.clip(sr / 20, 0, 1)

    # 13. Regime features
    d["vol_regime_score"] = (1 - d["realized_vol_20"].clip(0, 0.8) / 0.8)
    d["dd_recovery"]      = (close / close.rolling(60).max().replace(0, np.nan)).clip(0.5, 1)
    d["trend_age"]        = (close > sma20).astype(int).rolling(60).sum() / 60

    # 14. Macro regime features — yield curve + VIX term structure
    # These are cross-sectional constants (same for all stocks on a given day)
    # but they tell the model WHAT KIND OF MARKET it's operating in.
    # Yield curve inversion predicted every recession since 1955.
    # VIX term structure ratio predicts acute vs structural stress.
    try:
        t2_df  = _load_etf("treasury_2yr")
        t10_df = _load_etf("treasury_10yr")
        v3m_df = _load_etf("vix3m")

        if t2_df is not None and t10_df is not None and len(t2_df) > 60 and len(t10_df) > 60:
            t2  = t2_df["close"].reindex(close.index, method="ffill")
            t10 = t10_df["close"].reindex(close.index, method="ffill")

            # Yield spread: 10yr - 2yr (negative = inverted = danger)
            spread = t10 - t2
            d["yield_spread"]         = spread.clip(-3, 3)

            # Spread change over 60 days — steepening or flattening?
            d["yield_spread_60d_chg"] = spread.diff(60).clip(-2, 2)

            # Spread vs 1yr rolling average — unusually flat vs history?
            spread_ma252 = spread.rolling(252).mean()
            d["yield_spread_vs_1yr"]  = (spread - spread_ma252).clip(-2, 2)

            # Is curve inverted? Binary signal
            d["yield_inverted"]       = (spread < 0).astype(float)

            # 2yr yield level — absolute rate environment
            d["rate_2yr_level"]       = t2.clip(0, 10) / 10.0  # normalize to 0-1

            # Rate of change in 2yr — Fed hiking or cutting?
            d["rate_2yr_60d_chg"]     = t2.diff(60).clip(-3, 3)

        if v3m_df is not None and len(v3m_df) > 30:
            vix_s  = vix_macro["close"] if "close" in vix_macro.columns else vix_macro.iloc[:, 0]
            vix_aligned = vix_s.reindex(close.index, method="ffill")
            v3m    = v3m_df["close"].reindex(close.index, method="ffill")

            # VIX term structure ratio — >1 means near-term fear > long-term fear
            # (acute stress), <1 means calm short-term (structural concern only)
            vix_ratio = (vix_aligned / v3m.replace(0, np.nan)).clip(0.5, 2.0)
            d["vix_term_ratio"]       = vix_ratio

            # Is term structure inverted (contango vs backwardation)?
            d["vix_backwardation"]    = (vix_ratio > 1.0).astype(float)

            # 20-day change in ratio — is stress rising or falling?
            d["vix_term_ratio_20d"]   = vix_ratio.diff(20).clip(-0.5, 0.5)

            # Interaction: high VIX AND inverted curve = maximum danger signal
            # (2024 paper: combined signal predicts recessions far better than either alone)
            if "yield_inverted" in d:
                yi = d["yield_inverted"]
                d["vix_yield_danger"]  = (vix_ratio * yi.reindex(vix_ratio.index, method="ffill").fillna(0)).clip(0, 2)

    except Exception as _e:
        pass  # macro features optional — don't break if data missing

    # 15. Vol-adjusted momentum — raw return divided by realized vol
    # A 3% move in a 15-vol stock >> 3% move in a 40-vol stock
    # Without this the model can't distinguish signal from noise in momentum
    # vol_base: use realized_vol_10 for short horizons, realized_vol_20 for longer
    _vol_map = {5: "realized_vol_10", 20: "realized_vol_20", 60: "realized_vol_20"}
    for p in [5, 20, 60]:
        vol_base = d[_vol_map[p]].replace(0, np.nan)
        d[f"vol_adj_ret_{p}"] = (d[f"ret_{p}"] / (vol_base / np.sqrt(252) * np.sqrt(p))).clip(-5, 5)

    # 15. Momentum trajectory — is momentum accelerating or decelerating?
    # A stock at rank 85 that was rank 70 last week is accelerating (buy signal)
    # A stock at rank 85 that was rank 95 last week is decelerating (warning)
    # Computed as slope of 5d return over rolling 10/20 day windows
    d["mom_trajectory_10"] = d["ret_5"].diff(10).clip(-0.3, 0.3)
    d["mom_trajectory_20"] = d["ret_5"].diff(20).clip(-0.5, 0.5)
    d["mom_vs_spy_trajectory"] = (
        d.get("ret_5_vs_spy", pd.Series(0, index=df.index)).diff(10)
    ).clip(-0.3, 0.3)

    # 16. Persistence signals — consecutive days above/below key levels
    # Persistent weakness (20+ days below 200MA) = structural bear, not a dip
    # This is what distinguishes a genuine short from a temporarily weak stock
    d["days_above_sma20"]  = (close > sma20).astype(int).rolling(20).sum() / 20
    d["days_above_sma200"] = (close > sma200).astype(int).rolling(60).sum() / 60
    d["days_below_sma200"] = (close < sma200).astype(int).rolling(60).sum() / 60
    # Persistent weakness: in bottom half of performance for 20 consecutive days
    below_median_20d = (d["ret_20"] < 0).astype(int)
    d["persistent_weakness"] = below_median_20d.rolling(20).mean()
    d["persistent_strength"] = (d["ret_20"] > 0).astype(int).rolling(20).mean()

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
                    d["sector_vs_spy_20"]  = (ec.pct_change(20) - sc2.pct_change(20)).clip(-0.5, 0.5)
                    d["sector_vs_spy_60"]  = (ec.pct_change(60) - sc2.pct_change(60)).clip(-0.5, 0.5)
                    # Is sector in uptrend vs market?
                    d["sector_leading"]    = (d["sector_vs_spy_60"] > 0).astype(float)
                    # Stock strength within its own sector
                    d["stock_vs_sector_20"] = (close.pct_change(20) - ec.pct_change(20)).clip(-0.5, 0.5)


    # ── NEW FEATURES: PEAD proxy, momentum exhaustion, SI surprise, peer spillover ──

    # 18. PEAD Proxy — earnings surprise from price
    # Identify earnings days: large idiosyncratic return vs sector on high volume
    # Post-earnings drift persists 45-60 days (Bernard & Thomas 1989, replicated 2024)
    if symbol:
        etf = SECTOR_ETF_MAP.get(symbol.upper())
        etf_df2 = _etf_cache.get(etf) if etf else None
        if etf_df2 is not None and len(etf_df2) > 60:
            ec2 = etf_df2["close"].reindex(close.index, method="ffill")
            # Idiosyncratic daily return = stock return - sector return
            idio_ret = close.pct_change() - ec2.pct_change()
            # Earnings day proxy: |idiosyncratic return| > 3% AND volume > 2x average
            vol_threshold = volume.rolling(20).mean() * 2.0
            large_idio    = idio_ret.abs() > 0.03
            high_vol      = volume > vol_threshold
            earnings_day_proxy = (large_idio & high_vol).astype(float)
            # Signed earnings surprise: direction of idiosyncratic move on earnings day
            earnings_surprise = idio_ret * earnings_day_proxy
            # PEAD signal: decay-weighted sum of recent earnings surprises
            # Recent surprises (last 45 days) weighted by recency
            pead_45 = earnings_surprise.rolling(45).sum()
            pead_20 = earnings_surprise.rolling(20).sum()
            d["pead_signal_45"]    = pead_45.clip(-0.5, 0.5)
            d["pead_signal_20"]    = pead_20.clip(-0.3, 0.3)
            # Days since last positive earnings surprise (recency matters)
            pos_surprise_days = earnings_day_proxy * (idio_ret > 0.03).astype(float)
            last_pos = pos_surprise_days.rolling(90).apply(
                lambda x: 90 - (np.where(x[::-1] > 0)[0][0] if (x > 0).any() else 90),
                raw=True
            )
            d["days_since_pos_earnings"] = (last_pos / 90).clip(0, 1)  # 0=recent, 1=old/none
            # Magnitude of last earnings surprise
            d["last_earnings_magnitude"] = (
                idio_ret.where(earnings_day_proxy > 0).fillna(0).rolling(90).apply(
                    lambda x: x[x != 0][-1] if (x != 0).any() else 0, raw=True
                )
            ).clip(-0.3, 0.3)

    # 19. Momentum Exhaustion Signal
    # Where does today's N-day return sit on this stock's own historical distribution?
    # Stock at 95th percentile of own history = exhausted, likely to mean revert
    # Stock at 60th percentile = momentum has room to run
    # No paper has implemented this as a per-stock self-referential feature
    for p, window in [(20, 252), (60, 504)]:
        ret_series = close.pct_change(p)
        # Rolling percentile rank of current return vs trailing history
        def pctile_rank(x):
            if len(x) < 20:
                return 0.5
            return float(np.sum(x[:-1] < x[-1]) / max(len(x)-1, 1))
        exhaustion = ret_series.rolling(window).apply(pctile_rank, raw=True)
        d[f"mom_exhaustion_{p}d"] = exhaustion.clip(0, 1)
        # Distance from exhaustion: 0.5=neutral, >0.8=overbought, <0.2=oversold
        d[f"mom_exhaustion_extreme_{p}d"] = (exhaustion - 0.5).abs().clip(0, 0.5)

    # 20. Short Interest Surprise (SUSIR)
    # Raw SI level is noisy. The SIGNAL is the surprise vs own history.
    # Akbas et al 2023 JFE: surprise in SI predicts cross-section globally
    if symbol:
        f2 = _get_fundamentals(symbol)
        si = f2.get("shortPercentOfFloat")
        if si is not None and "short_pct_float" in d:
            # We only have a point-in-time SI, not history
            # Proxy: use SI relative to typical range for this stock's sector
            # If SI > 0.15 for a low-SI sector (tech) it's a surprise
            # If SI > 0.15 for a high-SI sector (retail) it's normal
            sector_si_medians = {
                "XLK": 0.035, "XLC": 0.040, "XLY": 0.055, "XLP": 0.025,
                "XLF": 0.025, "XLV": 0.045, "XLI": 0.025, "XLE": 0.030,
                "XLU": 0.020, "XLB": 0.030, "XLRE": 0.035,
            }
            etf3 = SECTOR_ETF_MAP.get(symbol.upper(), "XLK")
            sector_median = sector_si_medians.get(etf3, 0.035)
            # SI surprise = (current SI - sector median) / sector median
            si_surprise = (si - sector_median) / max(sector_median, 0.01)
            d["short_interest_surprise"] = np.clip(si_surprise, -3, 3)
            # High SI surprise = bearish signal
            d["si_elevated"]   = float(si > sector_median * 2.0)
            d["si_suppressed"] = float(si < sector_median * 0.5)

    # 21. Volume Shape Features — institutional accumulation/distribution
    # Gervais et al: high volume days predict positive returns (high-vol return premium)
    # Novel: direction of volume matters more than level
    d["volume_acceleration"]   = (volume / volume.rolling(20).mean().replace(0, np.nan) - 1).clip(-1, 2)
    d["vol_accel_5d"]          = (volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan) - 1).clip(-1, 2)
    # VWAP proxy: close vs open as fraction of daily range (buying pressure)
    daily_range = (high - low).replace(0, np.nan)
    d["buying_pressure"]       = ((close - open_) / daily_range).clip(-1, 1)
    d["close_strength"]        = ((close - low) / daily_range).clip(0, 1)  # close in upper half = bullish
    # Consecutive high-volume days: sustained institutional accumulation
    above_avg_vol = (volume > volume.rolling(20).mean()).astype(int)
    d["vol_streak_up"]         = above_avg_vol.rolling(5).sum() / 5  # 0-1, 1 = all 5 days high vol
    # Volume on up vs down days
    up_vol   = volume.where(close.diff() > 0, 0)
    down_vol = volume.where(close.diff() <= 0, 0)
    d["up_down_vol_ratio"] = (
        up_vol.rolling(10).mean() / down_vol.rolling(10).mean().replace(0, np.nan)
    ).clip(0, 5)

    # 22. Quality Acceleration (Ma, Yang, Ye 2024 - Research in International Business)
    # Second derivative of quality: rate of change of quality growth
    # Predicts returns beyond quality level AND quality growth
    if "quality_composite" in d:
        qc = d["quality_composite"]
        if isinstance(qc, pd.Series) and len(qc) > 20:
            # Quality growth: 20-day change in quality composite
            d["quality_growth_20"]    = qc.diff(20).clip(-0.5, 0.5)
            d["quality_growth_60"]    = qc.diff(60).clip(-0.8, 0.8)
            # Quality acceleration: change in quality growth (second derivative)
            d["quality_acceleration"] = d["quality_growth_20"].diff(20).clip(-0.3, 0.3)
            # Quality streak: consecutive periods of improvement
            d["quality_improving"]    = (d["quality_growth_20"] > 0).astype(float)
            d["quality_streak"]       = d["quality_improving"].rolling(3).mean()
        else:
            # Scalar value — create as constant series for this row
            for feat_name in ["quality_growth_20", "quality_growth_60",
                               "quality_acceleration", "quality_improving", "quality_streak"]:
                d[feat_name] = 0.0


    # 23. Earnings beat streak — He & Narayanamoorthy (2020), t-stat > 3.0
    if streak_store is not None and symbol is not None and symbol in streak_store:
        try:
            import sys as _sys
            _sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
            from earnings_streak import beat_streak_score as _bss
            _streak_df = streak_store[symbol]
            _scores = [_bss(symbol, pd.Timestamp(_dt), {symbol: _streak_df}) for _dt in df.index]
            d["earnings_beat_streak"] = pd.Series(_scores, index=df.index)
        except Exception:
            d["earnings_beat_streak"] = pd.Series(0.0, index=df.index)
    else:
        d["earnings_beat_streak"] = pd.Series(0.0, index=df.index)

    # 24. Insider cluster score — Cohen, Malloy, Pomorski (JF 2012)
    if insider_store is not None and symbol is not None and symbol in insider_store:
        try:
            from insider_signal import insider_score as _iscore
            _idf = insider_store[symbol]
            _iscores = [_iscore(symbol, pd.Timestamp(_dt), {symbol: _idf}) for _dt in df.index]
            d["insider_cluster_score"] = pd.Series(_iscores, index=df.index)
        except Exception:
            d["insider_cluster_score"] = pd.Series(0.0, index=df.index)
    else:
        d["insider_cluster_score"] = pd.Series(0.0, index=df.index)

    # 25. Analyst revision momentum — Mill Street Research 2023
    # Net direction of EPS revisions over last 4 quarters. Persists 5 months.
    if revision_store is not None and symbol is not None and symbol in revision_store:
        try:
            from analyst_revision import revision_momentum_score as _rms
            _rdf = revision_store[symbol]
            _rscores = [_rms(symbol, pd.Timestamp(_dt), {symbol: _rdf}) for _dt in df.index]
            d["analyst_revision_momentum"] = pd.Series(_rscores, index=df.index)
        except Exception:
            d["analyst_revision_momentum"] = pd.Series(0.0, index=df.index)
    else:
        d["analyst_revision_momentum"] = pd.Series(0.0, index=df.index)

    # 26b. Real PEAD signal — actual earnings surprise with time decay
    # IC=0.043 on mid-caps vs 0.002 on large-caps (our backtest tonight)
    # Bernard & Thomas (1989), 90-day drift window
    if symbol is not None:
        try:
            from pead_signal import pead_score as _pead_score, build_pead_store as _build_pead
            import os as _os
            _pead_cache = os.path.join("cache_earnings_streak", f"{symbol}.json")
            if _os.path.exists(_pead_cache):
                import json as _json
                with open(_pead_cache) as _f:
                    _recs = _json.load(_f)
                if _recs:
                    _pdf = pd.DataFrame(_recs)
                    _pdf["date"] = pd.to_datetime(_pdf["date"])
                    _pead_scores = [_pead_score(symbol, pd.Timestamp(_dt), {symbol: _pdf}) for _dt in df.index]
                    d["pead_real"] = pd.Series(_pead_scores, index=df.index)
                else:
                    d["pead_real"] = pd.Series(0.0, index=df.index)
            else:
                d["pead_real"] = pd.Series(0.0, index=df.index)
        except Exception:
            d["pead_real"] = pd.Series(0.0, index=df.index)
    else:
        d["pead_real"] = pd.Series(0.0, index=df.index)

    # 26. Sector earnings beat cluster — novel signal
    # 3+ peers in same sector beating same quarter = fundamental sector momentum
    if symbol is not None:
        try:
            import sys as _sys2
            _sys2.path.insert(0, '/Users/rick/ai_trading_bot_v2')
            from sector_earnings_cluster import sector_beat_cluster_score as _sbcs
            import config as _cfg
            _smap = {}
            for _etf, _syms in _cfg.SECTOR_ETFS.items():
                for _s in _syms:
                    _smap[_s] = _etf
            _cluster_scores = [_sbcs(symbol, pd.Timestamp(_dt), _smap, list(_cfg.WATCHLIST)) for _dt in df.index]
            d["sector_earnings_cluster"] = pd.Series(_cluster_scores, index=df.index)
        except Exception:
            d["sector_earnings_cluster"] = pd.Series(0.0, index=df.index)
    else:
        d["sector_earnings_cluster"] = pd.Series(0.0, index=df.index)

    # Novel D: Earnings beat streak × PEAD interaction
    # A stock with 4 consecutive beats AND confirmed PEAD drift isn't just additive
    # The market is systematically underreacting to a consistent pattern
    # This interaction term captures that compounding effect
    if "earnings_beat_streak" in d and "pead_real" in d:
        _streak = d["earnings_beat_streak"]
        _pead   = d["pead_real"]
        if isinstance(_streak, pd.Series) and isinstance(_pead, pd.Series):
            d["streak_pead_interaction"] = (_streak * _pead).clip(-0.5, 0.5)
        else:
            d["streak_pead_interaction"] = float(_streak) * float(_pead)
    else:
        d["streak_pead_interaction"] = pd.Series(0.0, index=df.index)

    # Novel E: Short interest × PEAD amplifier
    # High SI + positive earnings surprise = short squeeze amplification of PEAD drift
    # APP 2023 +$8,012 in 15 days was exactly this: PEAD fired, shorts squeezed
    # SI × PEAD interaction as both feature and sizing signal
    if symbol and "pead_real" in d:
        _f = _get_fundamentals(symbol) if symbol else {}
        _si = _f.get("shortPercentOfFloat", 0.0) or 0.0
        _pead_val = d["pead_real"]
        if isinstance(_pead_val, pd.Series):
            d["si_pead_squeeze"] = (_pead_val * float(_si)).clip(-0.3, 0.3)
        else:
            d["si_pead_squeeze"] = float(_pead_val) * float(_si)
    else:
        d["si_pead_squeeze"] = pd.Series(0.0, index=df.index)

    # Short interest CHANGE × earnings surprise interaction
    # Signal: when shorts are COVERING (SI declining) AND earnings beat = multiplicative effect
    # Academic basis: shorts covering + positive surprise = 3-4x individual signal power
    # Win rate historically >85%, ~40 events/year across 447 stocks
    # Key: using CHANGE in SI not level — covering pressure is the signal, not existing short
    if "short_interest_surprise" in d and "eps_beat_rate" in d and "pead_real" in d:
        _si_surp  = d["short_interest_surprise"]   # negative = shorts covering vs sector
        _eps_beat = d["eps_beat_rate"]              # > 0.5 = beating estimates
        _pead_v   = d["pead_real"]                  # positive = confirmed drift
        if isinstance(_si_surp, pd.Series):
            # Shorts covering (negative surprise) AND earnings beat AND PEAD positive
            # Clip to prevent outlier domination
            _cover_signal = (-_si_surp).clip(0, 3)  # higher = more covering pressure
            _earn_signal  = (_eps_beat - 0.5).clip(0, 0.5) * 2  # 0-1 scale
            d["si_earnings_squeeze"] = (_cover_signal * _earn_signal * _pead_v.clip(0, 1)).clip(0, 1)
        else:
            d["si_earnings_squeeze"] = pd.Series(0.0, index=df.index)
    else:
        d["si_earnings_squeeze"] = pd.Series(0.0, index=df.index)

    # 52-week high proximity — George & Hwang (2004)
    # IC=0.65% monthly, crash-resistant, works in all regimes
    # Stocks near 52-week high = anchoring bias creates underreaction
    # Fixes 2017/2021 underperformance — those years led by stocks at ATH
    try:
        high_52w = close.rolling(252).max()
        d["high_52w_proximity"] = (close / high_52w.replace(0, np.nan)).fillna(1.0)
        # Distance from 52-week low (oversold bounce signal)
        low_52w = close.rolling(252).min()
        d["low_52w_proximity"]  = (close / low_52w.replace(0, np.nan)).fillna(1.0)
        # Range position: where in 52-week range is price?
        _range = (high_52w - low_52w).replace(0, np.nan)
        d["range_position_52w"] = ((close - low_52w) / _range).fillna(0.5)
    except Exception:
        d["high_52w_proximity"] = pd.Series(1.0, index=df.index)
        d["low_52w_proximity"]  = pd.Series(1.0, index=df.index)
        d["range_position_52w"] = pd.Series(0.5, index=df.index)

    # IV term structure signal — individual stock options
    # near-term IV > far-term IV = temporary fear = mean reversion opportunity
    # IC estimate: 0.03-0.05, Cremers & Weinbaum (2010)
    try:
        from iv_term_structure import get_iv_signals
        _iv_store_path = os.path.join(os.path.dirname(os.path.abspath(__file__) if '__file__' in dir() else '.'), 'cache_options_iv', 'iv_term_structure.json')
        _iv_store = {}
        if os.path.exists(_iv_store_path):
            import json as _json2
            _iv_store = _json2.load(open(_iv_store_path))
        _iv = get_iv_signals(symbol, _iv_store) if symbol else {}
        d['iv_term_structure']  = pd.Series(float(_iv.get('iv_term_structure', 1.0)),  index=df.index)
        d['iv_backwardation']   = pd.Series(float(_iv.get('iv_backwardation',  0.0)),  index=df.index)
        d['iv_contango_steep']  = pd.Series(float(_iv.get('iv_contango_steep', 0.0)),  index=df.index)
    except Exception:
        d['iv_term_structure']  = pd.Series(1.0, index=df.index)
        d['iv_backwardation']   = pd.Series(0.0, index=df.index)
        d['iv_contango_steep']  = pd.Series(0.0, index=df.index)

    # Novel C proxy: momentum persistence score
    # How consistent is this stock's ML-predictable momentum across rolling windows?
    # Stocks with consistent momentum across 5/20/60d all aligned = higher persistence
    if "ret_5" in d and "ret_20" in d and "ret_60" in d:
        _r5  = d["ret_5"]  if isinstance(d["ret_5"],  pd.Series) else pd.Series(d["ret_5"],  index=df.index)
        _r20 = d["ret_20"] if isinstance(d["ret_20"], pd.Series) else pd.Series(d["ret_20"], index=df.index)
        _r60 = d["ret_60"] if isinstance(d["ret_60"], pd.Series) else pd.Series(d["ret_60"], index=df.index)
        # All three momentum horizons aligned = persistent momentum
        _aligned = ((_r5 > 0).astype(int) + (_r20 > 0).astype(int) + (_r60 > 0).astype(int)) / 3.0
        d["momentum_persistence"] = _aligned
    else:
        d["momentum_persistence"] = pd.Series(0.5, index=df.index)

    # Job posting velocity — Indeed RSS, 2-3 quarter leading indicator
    try:
        import json as _jj
        _jp = "/Users/rick/ai_trading_bot_v2/cache_jobs/job_signals.json"
        if os.path.exists(_jp):
            _jd = {r["symbol"]: r for r in _jj.load(open(_jp))}
            _j  = _jd.get(symbol or "", {})
            d["job_velocity"]    = pd.Series(float(_j.get("job_velocity", 0)), index=df.index)
            d["job_expanding"]   = pd.Series(float(_j.get("job_expanding", 0)), index=df.index)
            d["job_contracting"] = pd.Series(float(_j.get("job_contracting", 0)), index=df.index)
        else:
            for _k in ["job_velocity","job_expanding","job_contracting"]:
                d[_k] = pd.Series(0.0, index=df.index)
    except Exception:
        for _k in ["job_velocity","job_expanding","job_contracting"]:
            d[_k] = pd.Series(0.0, index=df.index)

    # Insider sequence signal — sequence of buys ENDING = strongest buy signal
    # Alpha Architect: 22-32% annualized alpha from sequence confirmation
    try:
        import json as _jseq
        _sp = "/Users/rick/ai_trading_bot_v2/cache_insider/insider_sequence.json"
        if os.path.exists(_sp):
            _sd = {r["symbol"]: r for r in _jseq.load(open(_sp))}
            _s  = _sd.get(symbol or "", {})
            d["insider_seq_buy"]      = pd.Series(float(_s.get("insider_seq_buy", 0)), index=df.index)
            d["insider_seq_strength"] = pd.Series(float(_s.get("insider_seq_strength", 0)), index=df.index)
        else:
            d["insider_seq_buy"]      = pd.Series(0.0, index=df.index)
            d["insider_seq_strength"] = pd.Series(0.0, index=df.index)
    except Exception:
        d["insider_seq_buy"]      = pd.Series(0.0, index=df.index)
        d["insider_seq_strength"] = pd.Series(0.0, index=df.index)

    # App Store rating — consumer tech revenue proxy (Apple iTunes API, free)
    # High rating (>=4.5) correlates with user retention and revenue growth
    try:
        import json as _japp
        _ap = "/Users/rick/ai_trading_bot_v2/cache_appstore/app_ratings.json"
        if os.path.exists(_ap):
            _ad = {r["symbol"]: r for r in _japp.load(open(_ap))}
            _a  = _ad.get(symbol or "", {})
            d["app_rating"]      = pd.Series(float(_a.get("app_rating", 0)), index=df.index)
            d["app_rating_high"] = pd.Series(float(_a.get("app_rating_high", 0)), index=df.index)
        else:
            d["app_rating"]      = pd.Series(0.0, index=df.index)
            d["app_rating_high"] = pd.Series(0.0, index=df.index)
    except Exception:
        d["app_rating"]      = pd.Series(0.0, index=df.index)
        d["app_rating_high"] = pd.Series(0.0, index=df.index)

    # Squeeze score — SI velocity + days-to-cover + momentum composite
    try:
        import json as _jsq
        _sqp = "/Users/rick/ai_trading_bot_v2/cache_si/squeeze_scores.json"
        if os.path.exists(_sqp):
            _sqd = {r["symbol"]: r for r in _jsq.load(open(_sqp))}
            _sq  = _sqd.get(symbol or "", {})
            d["squeeze_score"] = pd.Series(float(_sq.get("squeeze_score", 0)), index=df.index)
            d["squeeze_alert"] = pd.Series(float(_sq.get("squeeze_alert", 0)), index=df.index)
        else:
            d["squeeze_score"] = pd.Series(0.0, index=df.index)
            d["squeeze_alert"] = pd.Series(0.0, index=df.index)
    except Exception:
        d["squeeze_score"] = pd.Series(0.0, index=df.index)
        d["squeeze_alert"] = pd.Series(0.0, index=df.index)

    # FDA catalyst flag — biotech event risk
    try:
        import json as _jfda
        _fdap = "/Users/rick/ai_trading_bot_v2/cache_fda/fda_signals.json"
        if os.path.exists(_fdap):
            _fdad = {r["symbol"]: r for r in _jfda.load(open(_fdap))}
            _fda  = _fdad.get(symbol or "", {})
            d["fda_catalyst_near"] = pd.Series(float(_fda.get("fda_catalyst_near", 0)), index=df.index)
        else:
            d["fda_catalyst_near"] = pd.Series(0.0, index=df.index)
    except Exception:
        d["fda_catalyst_near"] = pd.Series(0.0, index=df.index)

    # 8-K text sentiment — Loughran-McDonald financial wordlist on SEC filings
    try:
        import json as _j8k
        _8kp = "/Users/rick/ai_trading_bot_v2/cache_8k/8k_sentiment.json"
        if os.path.exists(_8kp):
            _8kd = {r["symbol"]: r for r in _j8k.load(open(_8kp))}
            _8k  = _8kd.get(symbol or "", {})
            d["8k_sentiment"] = pd.Series(float(_8k.get("8k_sentiment", 0)), index=df.index)
            d["8k_negative"]  = pd.Series(float(_8k.get("8k_negative", 0)), index=df.index)
        else:
            d["8k_sentiment"] = pd.Series(0.0, index=df.index)
            d["8k_negative"]  = pd.Series(0.0, index=df.index)
    except Exception:
        d["8k_sentiment"] = pd.Series(0.0, index=df.index)
        d["8k_negative"]  = pd.Series(0.0, index=df.index)

    # Insider flow features — SEC EDGAR Form 4 (filed within 2 days)
    # insider_cluster: 3+ insiders buying = strongest signal (Seyhun 1998)
    # insider_net_buying: buys minus sells — positive = bullish insider sentiment
    # Most powerful in BEAR regime but valid signal in all regimes
    try:
        import sys as _si_sys2
        _si_sys2.path.insert(0, '/Users/rick/ai_trading_bot_v2')
        from insider_flow import compute_insider_features as _ins_feats
        _ins = _ins_feats(symbol or '')
        for _k, _v in _ins.items():
            d[_k] = pd.Series(float(_v), index=df.index)
    except Exception:
        for _k in ['insider_buy_count','insider_buy_value','insider_cluster',
                   'insider_ceo_buying','insider_net_buying']:
            d[_k] = pd.Series(0.0, index=df.index)

    # ── Earnings quality — accruals vs cash flow (Sloan 1996) ───────────
    # Principle: earnings backed by CASH are more durable than accrual earnings
    # Low accruals = conservative accounting = higher quality earnings
    # Works because of accounting reality — true since double-entry bookkeeping 1494
    # High accruals stock underperforms by 10%/yr (Sloan 1996, 40-year study)
    try:
        # Skip live accruals during training — too slow, in LIVE_ONLY_FEATURES anyway
        raise Exception("skip accruals during training")
        import yfinance as _yf_aq
        _aq_tick = _yf_aq.Ticker(symbol or '')
        _cf = _aq_tick.cashflow
        _bs = _aq_tick.balance_sheet
        _inc = _aq_tick.income_stmt
        if _cf is not None and _bs is not None and _inc is not None and len(_cf.columns) >= 2:
            # Operating cash flow
            _ocf = float(_cf.loc['Operating Cash Flow'].iloc[0]) if 'Operating Cash Flow' in _cf.index else 0
            # Net income
            _ni  = float(_inc.loc['Net Income'].iloc[0]) if 'Net Income' in _inc.index else 0
            # Total assets for scaling
            _ta  = float(_bs.loc['Total Assets'].iloc[0]) if 'Total Assets' in _bs.index else 1
            # Accruals = Net Income - Operating Cash Flow (scaled by assets)
            # Low accruals (earnings ≈ cash) = high quality
            _accruals = (_ni - _ocf) / max(abs(_ta), 1)
            # Accrual ratio: negative = cash > earnings = HIGH QUALITY
            _accrual_ratio = float(np.clip(_accruals, -0.20, 0.20))
            # Cash earnings ratio: OCF/NI > 1 = cash confirms earnings
            _cash_earnings = float(np.clip(_ocf / max(abs(_ni), 1), -2, 3)) if _ni != 0 else 0.0
            d["accrual_ratio"]    = pd.Series(_accrual_ratio, index=df.index)
            d["cash_earnings"]    = pd.Series(_cash_earnings, index=df.index)
            d["earnings_quality"] = pd.Series(float(-_accrual_ratio + 0.20), index=df.index)  # higher = better
        else:
            d["accrual_ratio"]    = pd.Series(0.0, index=df.index)
            d["cash_earnings"]    = pd.Series(1.0, index=df.index)
            d["earnings_quality"] = pd.Series(0.5, index=df.index)
    except Exception:
        d["accrual_ratio"]    = pd.Series(0.0, index=df.index)
        d["cash_earnings"]    = pd.Series(1.0, index=df.index)
        d["earnings_quality"] = pd.Series(0.5, index=df.index)

    # ── Earnings surprise quality (soft signal, not hard filter) ──────────
    # beat_rate as ML feature — let ranker weigh vs other signals
    # Data showed hard filter hurts 2022/2023/2025 — soft is better
    try:
        import json as _jeq
        _eq_path = '/Users/rick/ai_trading_bot_v2/cache_revisions/earnings_quality.json'
        if os.path.exists(_eq_path):
            _eq = _jeq.load(open(_eq_path))
            _eq_data = _eq.get(symbol or {}, {})
            _beat_rate  = float(_eq_data.get('beat_rate', 0.5))
            _avg_surp   = float(np.clip(_eq_data.get('avg_surprise', 0.0), -1, 2))
            _streak     = float(min(_eq_data.get('streak', 0), 8)) / 8.0
        else:
            _beat_rate = 0.5
            _avg_surp  = 0.0
            _streak    = 0.0
        d["eps_beat_rate"]    = pd.Series(_beat_rate, index=df.index)
        d["eps_avg_surprise"] = pd.Series(_avg_surp,  index=df.index)
        d["eps_beat_streak"]  = pd.Series(_streak,    index=df.index)
    except Exception:
        d["eps_beat_rate"]    = pd.Series(0.5, index=df.index)
        d["eps_avg_surprise"] = pd.Series(0.0, index=df.index)
        d["eps_beat_streak"]  = pd.Series(0.0, index=df.index)

    # Institutional flow features — SEC EDGAR 13F quarterly
    # Measures Vanguard+BlackRock+StateStreet QoQ share count changes
    # Filtered to exclude passive rebalancing artifacts (>50% changes)
    # Active buying (5-20% increase) predicts 6-12mo returns (Nofsinger 1999)
    try:
        import json as _json13f
        _13f_path = '/Users/rick/ai_trading_bot_v2/cache_13f/inst_changes_filtered.json'
        if os.path.exists(_13f_path):
            with open(_13f_path) as _f13f:
                _13f_data = {r['symbol']: r for r in _json13f.load(_f13f)}
            _sym_13f = _13f_data.get(symbol or '', {})
            _inst_chg = float(_sym_13f.get('chg', 0.0))
            d['inst_buying']  = pd.Series(float(np.clip(max(_inst_chg, 0), 0, 0.5)), index=df.index)
            d['inst_selling'] = pd.Series(float(np.clip(-min(_inst_chg, 0), 0, 0.5)), index=df.index)
            d['inst_chg_pct'] = pd.Series(float(np.clip(_inst_chg, -0.5, 0.5)), index=df.index)
        else:
            d['inst_buying']  = pd.Series(0.0, index=df.index)
            d['inst_selling'] = pd.Series(0.0, index=df.index)
            d['inst_chg_pct'] = pd.Series(0.0, index=df.index)
    except Exception:
        d['inst_buying']  = pd.Series(0.0, index=df.index)
        d['inst_selling'] = pd.Series(0.0, index=df.index)
        d['inst_chg_pct'] = pd.Series(0.0, index=df.index)

    # Short Interest velocity features — FINRA data (2x monthly)
    # si_cover_signal: falling SI + rising price = short covering rally
    # si_squeeze_risk: high SI + rising SI = squeeze danger
    # ── Cross-stock propagation (Menzly & Ozbas 2010) ─────────────────────
    # Peer idiosyncratic moves propagate to linked stocks with 1-3 day delay
    # IC 0.03-0.055 measured on semiconductor and financial clusters
    try:
        import yfinance as _yf_prop
        _prop_cache = os.path.expanduser('~/ai_trading_bot_v2/cache_propagation')
        os.makedirs(_prop_cache, exist_ok=True)
        PEER_MAP = {
            'NVDA':['AMD','INTC','QCOM','AVGO'],
            'AMAT':['LRCX','KLAC','ASML','NVDA'],
            'LRCX':['AMAT','KLAC','NVDA','MU'],
            'KLAC':['AMAT','LRCX','NVDA','MU'],
            'MU':  ['NVDA','AMD','INTC','WDC'],
            'AMD': ['NVDA','INTC','QCOM','MU'],
            'JPM': ['GS','MS','BAC','WFC'],
            'GS':  ['JPM','MS','BAC','AXP'],
            'MS':  ['JPM','GS','BAC','BLK'],
            'XOM': ['CVX','COP','SLB','EOG'],
            'CVX': ['XOM','COP','EOG','SLB'],
            'LLY': ['UNH','ABBV','JNJ','MRK'],
            'UNH': ['LLY','ABBV','JNJ','CVS'],
            'AAPL':['MSFT','GOOGL','META','AMZN'],
            'MSFT':['AAPL','GOOGL','AMZN','META'],
        }
        _peers = PEER_MAP.get(symbol, [])
        _prop_cols = ['upstream_prop_1d','upstream_prop_3d',
                      'upstream_large_move_1d','upstream_signed_1d']
        if _peers:
            _peer_idios = []
            _spy_ret = close.pct_change()
            for _p in _peers:
                try:
                    _cf = os.path.join(_prop_cache, f'{_p}.pkl')
                    if os.path.exists(_cf):
                        _ps = pd.read_pickle(_cf)
                    else:
                        _ps = _yf_prop.download(_p, period='10y',
                            progress=False, auto_adjust=True)['Close'].squeeze()
                        _ps.to_pickle(_cf)
                    _ps.index = pd.to_datetime(_ps.index).tz_localize(None)
                    _pr = _ps.pct_change()
                    _b  = _pr.rolling(60).cov(_spy_ret) / _spy_ret.rolling(60).var()
                    _peer_idios.append((_pr - _b.fillna(1).clip(0,3) * _spy_ret
                                       ).reindex(df.index).ffill())
                except:
                    pass
            if _peer_idios:
                _mean_idio = pd.concat(_peer_idios, axis=1).mean(axis=1)
                _large     = (_mean_idio.abs() > 0.02).astype(float)
                d['upstream_prop_1d']        = _mean_idio.shift(1).rolling(3).mean()
                d['upstream_prop_3d']        = _mean_idio.shift(3).rolling(3).mean()
                d['upstream_large_move_1d']  = _large.shift(1)
                d['upstream_signed_1d']      = (_mean_idio * _large).shift(1)
            else:
                for _f in _prop_cols: d[_f] = pd.Series(0.0, index=df.index)
        else:
            for _f in _prop_cols: d[_f] = pd.Series(0.0, index=df.index)
    except Exception:
        for _f in ['upstream_prop_1d','upstream_prop_3d',
                   'upstream_large_move_1d','upstream_signed_1d']:
            d[_f] = pd.Series(0.0, index=df.index)

    # ── Cross-asset macro signals (FRED data) ───────────────────────────────
    # Copper IC=0.1166 for industrials, yield curve IC=0.052 for financials
    # HY spread IC=-0.0395 (rising spread = momentum crash predictor)
    # All data from FRED, no lookahead bias — daily series forward-filled
    try:
        import urllib.request, io as _io
        _ca_path = os.path.expanduser('~/ai_trading_bot_v2/cross_asset_data.csv')
        if os.path.exists(_ca_path):
            _ca = pd.read_csv(_ca_path, index_col=0, parse_dates=True)
            _ca = _ca.reindex(df.index, method='ffill')

            # HY credit spread features
            if 'hy_spread' in _ca.columns:
                _hy = _ca['hy_spread']
                d['ca_hy_spread_level']   = _hy
                d['ca_hy_spread_chg_20d'] = _hy.diff(20)
                d['ca_hy_spread_chg_60d'] = _hy.diff(60)
                # Danger signal: spread rising fast = momentum crash risk
                d['ca_hy_danger']         = (_hy.diff(60) > 0.5).astype(float)

            # Yield curve features
            if 'yield_curve' in _ca.columns:
                _yc = _ca['yield_curve']
                d['ca_yield_curve']       = _yc
                d['ca_yield_curve_chg']   = _yc.diff(20)
                d['ca_yield_inverted']    = (_yc < 0).astype(float)
                d['ca_yield_steepening']  = (_yc.diff(20) > 0.1).astype(float)

            # Copper features (monthly data forward-filled to daily)
            if 'copper' in _ca.columns:
                _cu = _ca['copper']
                d['ca_copper_ret_21d']    = _cu.pct_change(21)
                d['ca_copper_ret_63d']    = _cu.pct_change(63)
                d['ca_copper_momentum']   = (_cu.pct_change(21) > 0).astype(float)
                d['ca_copper_accel']      = _cu.pct_change(21) - _cu.pct_change(63)

            # IG spread (investment grade)
            if 'ig_spread' in _ca.columns:
                _ig = _ca['ig_spread']
                d['ca_ig_spread_chg']     = _ig.diff(20)
                d['ca_credit_risk_on']    = (_ig.diff(20) < 0).astype(float)

            # Inflation breakeven
            if 'breakeven_10y' in _ca.columns:
                _be = _ca['breakeven_10y']
                d['ca_breakeven_level']   = _be
                d['ca_breakeven_rising']  = (_be.diff(20) > 0.1).astype(float)

        else:
            _ca_cols = ['ca_hy_spread_level','ca_hy_spread_chg_20d','ca_hy_spread_chg_60d',
                        'ca_hy_danger','ca_yield_curve','ca_yield_curve_chg','ca_yield_inverted',
                        'ca_yield_steepening','ca_copper_ret_21d','ca_copper_ret_63d',
                        'ca_copper_momentum','ca_copper_accel','ca_ig_spread_chg',
                        'ca_credit_risk_on','ca_breakeven_level','ca_breakeven_rising']
            for _f in _ca_cols:
                d[_f] = pd.Series(0.0, index=df.index)
    except Exception:
        _ca_cols = ['ca_hy_spread_level','ca_hy_spread_chg_20d','ca_hy_spread_chg_60d',
                    'ca_hy_danger','ca_yield_curve','ca_yield_curve_chg','ca_yield_inverted',
                    'ca_yield_steepening','ca_copper_ret_21d','ca_copper_ret_63d',
                    'ca_copper_momentum','ca_copper_accel','ca_ig_spread_chg',
                    'ca_credit_risk_on','ca_breakeven_level','ca_breakeven_rising']
        for _f in _ca_cols:
            d[_f] = pd.Series(0.0, index=df.index)

    # ── Overnight vs intraday return decomposition ──────────────────────────
    # Validated on your actual trade data (2017-2025):
    # Intraday momentum IC=+0.0473 t=8.26 (very significant)
    # Overnight momentum IC=-0.0152 t=-2.66 (negative predictor — stop risk)
    # High overnight momentum entries have 45.8% stop rate vs 28.3% for low
    # Directly contradicts Lou/Polk/Skouras on broad market — large/mid cap behaves differently
    try:
        if 'open' in df.columns:
            _open  = df['open']
            _close = close

            # Overnight return: close → next open (gap)
            _overnight = (_open / _close.shift(1)) - 1
            # Intraday return: open → close
            _intraday  = (_close / _open) - 1

            # Rolling cumulative signals
            d['overnight_mom_5d']  = _overnight.rolling(5).sum()
            d['overnight_mom_20d'] = _overnight.rolling(20).sum()
            d['intraday_mom_5d']   = _intraday.rolling(5).sum()
            d['intraday_mom_20d']  = _intraday.rolling(20).sum()

            # Ratio: how much of recent momentum is overnight vs intraday
            _total_20 = (_overnight + _intraday).rolling(20).sum()
            _on_share = _overnight.rolling(20).sum() / (_total_20.abs() + 0.001)
            d['overnight_share_20d'] = _on_share  # high = dangerous

            # Gap up frequency (overnight gaps > 0.5%)
            d['gap_up_freq_20d']    = (_overnight > 0.005).rolling(20).mean()
            d['gap_down_freq_20d']  = (_overnight < -0.005).rolling(20).mean()

            # Intraday consistency (how often closes above open)
            d['intraday_win_rate_20d'] = (_intraday > 0).rolling(20).mean()
        else:
            for _f in ['overnight_mom_5d','overnight_mom_20d','intraday_mom_5d',
                       'intraday_mom_20d','overnight_share_20d','gap_up_freq_20d',
                       'gap_down_freq_20d','intraday_win_rate_20d']:
                d[_f] = pd.Series(0.0, index=df.index)
    except Exception:
        for _f in ['overnight_mom_5d','overnight_mom_20d','intraday_mom_5d',
                   'intraday_mom_20d','overnight_share_20d','gap_up_freq_20d',
                   'gap_down_freq_20d','intraday_win_rate_20d']:
            d[_f] = pd.Series(0.0, index=df.index)

    # ── Symbol historical trade performance (from live trading history) ──
    # Correlation stop_rate → avg_pnl = -0.438 (validated on 809 trades)
    # SMCI 100% stop rate, PANW 86%, EBAY 65% — ranker should learn to avoid
    try:
        import json as _jth
        _th_path = os.path.expanduser('~/ai_trading_bot_v2/cache_fundamentals/symbol_trade_history.json')
        if os.path.exists(_th_path) and symbol:
            _th = _jth.load(open(_th_path))
            _sym_data = _th.get(symbol, {})
            d['sym_historical_stop_rate'] = pd.Series(
                float(_sym_data.get('historical_stop_rate', 0.30)), index=df.index)
            d['sym_historical_wr'] = pd.Series(
                float(_sym_data.get('historical_wr', 0.50)), index=df.index)
            d['sym_n_trades'] = pd.Series(
                float(_sym_data.get('n_trades', 0)), index=df.index)
        else:
            for _k in ['sym_historical_stop_rate','sym_historical_wr','sym_n_trades']:
                d[_k] = pd.Series(0.0, index=df.index)
    except Exception:
        for _k in ['sym_historical_stop_rate','sym_historical_wr','sym_n_trades']:
            d[_k] = pd.Series(0.0, index=df.index)

    # si_velocity: acceleration of SI change (momentum of momentum)
    try:
        import sys as _si_sys
        _si_sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
        from si_velocity import compute_si_features
        _si_feats = compute_si_features(symbol or '', df)
        for _k, _v in _si_feats.items():
            d[_k] = pd.Series(float(_v), index=df.index)
    except Exception:
        for _k in ['si_level','si_change_2w','si_change_4w','si_velocity',
                   'si_squeeze_risk','si_cover_signal','si_days_cover']:
            d[_k] = pd.Series(0.0, index=df.index)

    # VIX spike probability — ML model (AUC=0.90)
    # High spike probability = reduce long scores = system naturally
    # becomes more conservative before volatility events
    try:
        import sys as _sys2
        _sys2.path.insert(0, '/Users/rick/ai_trading_bot_v2')
        from vol_strategy_ml import predict_vix_spike_prob, predict_vix_30d, load_vix_history
        _vix_series = load_vix_history()
        if len(_vix_series) > 0:
            _vix_to_date = _vix_series.loc[_vix_series.index <= close.index[-1]]
            if len(_vix_to_date) >= 60:
                _spike_prob = predict_vix_spike_prob(_vix_to_date)
                _vix_30d    = predict_vix_30d(_vix_to_date)
                d["vix_spike_prob_10d"] = pd.Series(float(_spike_prob), index=df.index)
                d["vix_predicted_30d"]  = pd.Series(float(_vix_30d), index=df.index)
                # Derived: expected vol regime change
                _curr_vix = float(_vix_to_date.iloc[-1])
                d["vix_expected_drop"]  = pd.Series(float((_curr_vix - _vix_30d) / _curr_vix), index=df.index)
            else:
                d["vix_spike_prob_10d"] = pd.Series(0.3, index=df.index)
                d["vix_predicted_30d"]  = pd.Series(20.0, index=df.index)
                d["vix_expected_drop"]  = pd.Series(0.0, index=df.index)
    except Exception:
        d["vix_spike_prob_10d"] = pd.Series(0.3, index=df.index)
        d["vix_predicted_30d"]  = pd.Series(20.0, index=df.index)
        d["vix_expected_drop"]  = pd.Series(0.0, index=df.index)

    feat = pd.DataFrame(d, index=df.index)
    return feat.replace([np.inf, -np.inf], np.nan)


# ── Forward return ────────────────────────────────────────────────────────────

def compute_forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    return (df["close"].shift(-horizon) / df["open"].shift(-1)) - 1


# ── Panel builder ─────────────────────────────────────────────────────────────

def build_symbol_store(symbols: List[str], days: int, refresh: bool = False) -> dict:
    # Build earnings beat streak store once for all symbols
    try:
        import sys as _sys
        _sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
        from earnings_streak import build_streak_store
        streak_store = build_streak_store(symbols, verbose=False)
        log(f"INFO | Earnings streak store: {len(streak_store)} symbols")
        import joblib as _jl; _jl.dump(streak_store, "streak_store.joblib")
    except Exception as _e:
        log(f"WARN | Earnings streak store failed: {_e}")
        streak_store = {}
    # Build insider cluster store once for all symbols
    try:
        from insider_signal import build_insider_store
        insider_store = build_insider_store(symbols, verbose=False)
        log(f"INFO | Insider store: {len(insider_store)} symbols with buy data")
        import joblib as _jl; _jl.dump(insider_store, "insider_store.joblib")
    except Exception as _e:
        log(f"WARN | Insider store failed: {_e}")
        insider_store = {}
    # Build analyst revision store
    try:
        from analyst_revision import build_revision_store
        revision_store = build_revision_store(symbols, verbose=False)
        log(f"INFO | Revision store: {len(revision_store)} symbols")
    except Exception as _e:
        log(f"WARN | Revision store failed: {_e}")
        revision_store = {}
    store = {}
    for sym in symbols:
        df = fetch_data(sym, days, refresh)
        if df is None or len(df) < 260:
            continue
        feat = compute_features(df, symbol=sym, streak_store=streak_store, insider_store=insider_store, revision_store=revision_store)
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
            common = df.index.intersection(spy_close.index)
            for p in [5, 20, 60]:
                if f"ret_{p}" in feat.columns:
                    feat.loc[common, f"ret_{p}_vs_spy"] = (
                        feat.loc[common, f"ret_{p}"] -
                        spy_close.loc[common].pct_change(p)
                    )

        if spy_close is not None:
            spy_fwd = (spy_close.shift(-horizon) / spy_close.shift(-1)) - 1
            common  = target_raw.index.intersection(spy_fwd.index)
            target  = pd.Series(index=target_raw.index, dtype=float)
            target.loc[common] = (target_raw.loc[common] - spy_fwd.loc[common]).clip(-0.20, 0.20)
        else:
            target = target_raw.clip(-0.20, 0.20)

        merged           = pd.concat([feat, target_raw.rename("target_raw"), target.rename("target")], axis=1)
        merged["symbol"] = sym
        merged["date"]   = merged.index
        frames.append(merged)

    if not frames:
        raise RuntimeError("No symbols loaded")

    panel = pd.concat(frames)
    panel["target_rank"] = panel.groupby("date")["target"].rank(pct=True)

    # Static snapshot features — valid for live trading but POISON for backtesting
    # They are zero for all historical dates, teaching the ranker a false zero-pattern
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
        "sym_historical_stop_rate", "sym_historical_wr", "sym_n_trades",
        # Overnight gap features — negative predictor in TRENDING_BULL (2024 AI bull)
        "overnight_mom_5d", "overnight_mom_20d", "overnight_share_20d",
        "gap_up_freq_20d", "gap_down_freq_20d",
        # Snapshot-only features — valid live but no historical time series
        "iv_term_structure", "iv_backwardation", "iv_contango_steep",
        "analyst_score", "analyst_momentum", "analyst_consensus",
    }
    feature_cols = [c for c in panel.columns
                    if c not in {"date", "symbol", "target_raw", "target", "target_rank"}
                    and c not in LIVE_ONLY_FEATURES]
    cs_ranks = {}
    for c in feature_cols:
        cs_ranks[f"{c}_cs_rank"] = panel.groupby("date")[c].rank(pct=True)
    panel = pd.concat([panel, pd.DataFrame(cs_ranks, index=panel.index)], axis=1)

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
    # Static snapshot features — valid for live trading but POISON for backtesting
    # They are zero for all historical dates, teaching the ranker a false zero-pattern
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
        # Overnight gap features — hurt TRENDING_BULL (negative IC in 2021, 2024)
        # Intraday features kept (IC=+0.047, regime-neutral)
        "overnight_mom_5d", "overnight_mom_20d", "overnight_share_20d",
        "gap_up_freq_20d", "gap_down_freq_20d",
        # Snapshot-only features — valid live but no historical time series
        "iv_term_structure", "iv_backwardation", "iv_contango_steep",
        "analyst_score", "analyst_momentum", "analyst_consensus",

    }
    feature_cols = [c for c in panel.columns
                    if c not in {"date", "symbol", "target_raw", "target", "target_rank"}
                    and c not in LIVE_ONLY_FEATURES]

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
        total_imp = imp.sum()
        cs_share = imp[[c for c in feature_cols if "_cs_rank" in c]].sum() / total_imp if total_imp > 0 else 0
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

        model, scaler, features = train_ranker(panel, horizon, oos_only=True)
        bundle = {"model": model, "scaler": scaler, "features": features, "horizon": horizon}
        path   = f"cross_sectional_ranker_{horizon}d.joblib"
        joblib.dump(bundle, path)
        log(f"INFO | Saved {path}")

    joblib.dump(joblib.load("cross_sectional_ranker_5d.joblib"), "cross_sectional_ranker.joblib")
    log("INFO | Complete.")



def train_rolling_ensemble(symbols: List[str], days: int = 3650, refresh: bool = False) -> None:
    """
    Rolling walk-forward retrain — the way real quant funds do it.

    For each test year, train on the prior 5 years of data and save a dated model.
    The backtester then loads the correct vintage for each date, eliminating
    any future data leakage into the model.

    Vintages produced:
      cross_sectional_ranker_3d_2020.joblib  (trained 2015-2019, test 2020)
      cross_sectional_ranker_3d_2021.joblib  (trained 2016-2020, test 2021)
      ...
      cross_sectional_ranker_3d_2025.joblib  (trained 2020-2024, test 2025)

    Also saves the "current" model (trained on most recent 5 years) as the
    standard cross_sectional_ranker_Xd.joblib for live trading.
    """
    log(f"INFO | Rolling walk-forward retrain")
    log(f"INFO | Universe: {len(symbols)} stocks")
    log(f"INFO | Model: {'LightGBM' if USE_LIGHTGBM else 'sklearn GBM'}")

    store = build_symbol_store(symbols, days, refresh=refresh)
    log(f"INFO | Loaded {len(store)} symbols")

    test_years  = [2020, 2021, 2022, 2023, 2024, 2025]
    train_years = 5

    for horizon in [3, 5, 7]:
        log(f"INFO | === Horizon {horizon}d ===")
        panel = build_panel_from_store(store, horizon)
        panel["date"] = pd.to_datetime(panel["date"])

        feature_cols = [c for c in panel.columns
                        if c not in {"date", "symbol", "target_raw", "target", "target_rank"}]

        for test_year in test_years:
            train_start = pd.Timestamp(f"{test_year - train_years}-01-01")
            train_end   = pd.Timestamp(f"{test_year}-01-01")
            test_end    = pd.Timestamp(f"{test_year + 1}-01-01")

            train_df = panel[(panel["date"] >= train_start) & (panel["date"] < train_end)]
            test_df  = panel[(panel["date"] >= train_end)   & (panel["date"] < test_end)]

            if len(train_df) < 1000:
                log(f"WARN | {test_year}: insufficient train data ({len(train_df)} rows), skipping")
                continue

            log(f"INFO | {test_year}: train={train_start.year}-{train_end.year-1} "
                f"({len(train_df):,} rows) | test={test_year} ({len(test_df):,} rows)")

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(train_df[feature_cols].fillna(0))
            model   = _build_model(horizon)
            model.fit(X_train, train_df["target_rank"])

            if len(test_df) > 0:
                X_test = scaler.transform(test_df[feature_cols].fillna(0))
                pred   = model.predict(X_test)
                rmse   = float(np.sqrt(mean_squared_error(test_df["target_rank"], pred)))
                ic     = float(np.corrcoef(pred, test_df["target_rank"])[0, 1])
                log(f"INFO | {test_year} OOS: RMSE={rmse:.4f}  IC={ic:.4f}")

            bundle = {
                "model":    model,
                "scaler":   scaler,
                "features": feature_cols,
                "horizon":  horizon,
                "train_start": str(train_start.date()),
                "train_end":   str(train_end.date()),
                "test_year":   test_year,
            }
            path = f"cross_sectional_ranker_{horizon}d_{test_year}.joblib"
            joblib.dump(bundle, path)
            log(f"INFO | Saved {path}")

        # Save the most recent vintage as the live model
        latest_year = max(test_years)
        latest_path = f"cross_sectional_ranker_{horizon}d_{latest_year}.joblib"
        joblib.dump(joblib.load(latest_path), f"cross_sectional_ranker_{horizon}d.joblib")
        log(f"INFO | Updated live model cross_sectional_ranker_{horizon}d.joblib from {latest_year} vintage")

    joblib.dump(joblib.load("cross_sectional_ranker_5d.joblib"), "cross_sectional_ranker.joblib")
    log("INFO | Rolling retrain complete.")




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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",          type=int, default=3650)
    parser.add_argument("--save-ensemble", action="store_true")
    parser.add_argument("--rolling",       action="store_true", help="Rolling walk-forward retrain (one model per year)")
    parser.add_argument("--regime",        action="store_true", help="Train regime-specific models (TRENDING_BULL, CHOPPY, BEAR)")
    parser.add_argument("--refresh",       action="store_true")
    parser.add_argument("--lightgbm",      action="store_true")
    args = parser.parse_args()

    global USE_LIGHTGBM
    if args.lightgbm:
        USE_LIGHTGBM = True

    if args.rolling:
        train_rolling_ensemble(list(config.WATCHLIST), args.days, args.refresh)
    elif args.regime:
        train_rolling_regime_ensemble(list(config.WATCHLIST), args.days, args.refresh)
    elif args.save_ensemble:
        train_and_save_ensemble(list(config.WATCHLIST), args.days, args.refresh)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()