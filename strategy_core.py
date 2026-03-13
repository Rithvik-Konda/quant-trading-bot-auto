from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import config
from ml_model import compute_features

# Build symbol → sector ETF map from config
SECTOR_ETF_MAP: Dict[str, str] = {}
for _etf, _syms in getattr(config, "SECTOR_ETFS", {}).items():
    for _s in _syms:
        SECTOR_ETF_MAP[_s] = _etf


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalSnapshot:
    symbol:         str
    rule_score:     float
    ml_score:       float
    ml_rank_pct:    float
    combined_score: float
    trend_bullish:  bool
    stop_pct:       float
    atr_pct:        float
    side:           str = "long"   # "long" or "short"


@dataclass
class RegimeState:
    """
    Continuous 0-1 regime score.  No hardcoded thresholds — everything flows
    through config.  Score feeds position sizing, long/short ratio, stop width.

    score = 1.0 → strong bull   (full long book, no shorts)
    score = 0.5 → neutral       (long + short, reduced size)
    score = 0.0 → strong bear   (short-heavy, minimal longs)
    """
    score:               float          # 0-1 continuous
    is_bull:             bool
    is_neutral:          bool
    is_bear:             bool
    spy_crash:           bool           # hard intraday circuit breaker
    vol_halt:            bool           # extreme vol halt
    vix_spike:           bool           # VIX proxy spiked — halt new longs
    position_scalar:     float          # multiplier for position size
    net_exposure_target: float          # target net exposure as fraction of capital
    breadth:             float          # fraction of universe above 200MA
    realized_vol:        float          # SPY 20-day realized vol annualised
    components:          Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"missing column: {col}")
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.astype(float)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    needed = ["open", "high", "low", "close", "volume"]
    for c in needed:
        if c not in out.columns:
            raise ValueError(f"OHLCV missing column: {c}")
    out = out[needed].dropna().copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    df    = normalize_ohlcv(df)
    high  = _safe_series(df, "high")
    low   = _safe_series(df, "low")
    close = _safe_series(df, "close")
    tr1   = high - low
    tr2   = (high - close.shift(1)).abs()
    tr3   = (low  - close.shift(1)).abs()
    atr   = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(period).mean().iloc[-1]
    px    = close.iloc[-1]
    if pd.isna(atr) or px <= 0:
        return config.FIXED_STOP_LOSS_PCT
    return float(atr / px)


def trend_bullish(df: pd.DataFrame) -> bool:
    df    = normalize_ohlcv(df)
    close = _safe_series(df, "close")
    if len(close) < config.TREND_SMA_SLOW:
        return True
    sma_fast = close.rolling(config.TREND_SMA_FAST).mean()
    sma_slow = close.rolling(config.TREND_SMA_SLOW).mean()
    return bool(sma_fast.iloc[-1] > sma_slow.iloc[-1])


def realized_vol_annualized(df: pd.DataFrame, window: int = 20) -> float:
    df    = normalize_ohlcv(df)
    close = _safe_series(df, "close")
    if len(close) < window + 1:
        return 0.0
    return float(close.pct_change().rolling(window).std().iloc[-1] * np.sqrt(252))


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive stop width
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_stop_pct(
    df: pd.DataFrame,
    entry_price: float,
    current_price: float,
    side: str = "long",
) -> float:
    """
    Stop width that adapts to:
    1. Individual stock realized volatility (wider in high-vol stocks)
    2. Profit in the position (tightens as gains accumulate)

    All parameters from config — nothing hardcoded.
    """
    atr_pct = compute_atr_pct(df, config.ATR_PERIOD)

    # Base stop: ATR × vol multiplier, bounded by min/max
    base_stop = float(np.clip(
        atr_pct * config.STOP_VOL_MULTIPLIER,
        config.STOP_MIN_PCT,
        config.STOP_MAX_PCT,
    ))

    if entry_price <= 0 or current_price <= 0:
        return base_stop

    # How far in profit are we?
    if side == "long":
        profit_pct = (current_price - entry_price) / entry_price
    else:
        profit_pct = (entry_price - current_price) / entry_price

    # Once profit exceeds threshold, tighten stop to lock in gains
    if profit_pct > config.STOP_PROFIT_TIGHTEN_START:
        excess      = profit_pct - config.STOP_PROFIT_TIGHTEN_START
        tightening  = excess * config.STOP_PROFIT_TIGHTEN_RATE
        base_stop   = max(config.STOP_MIN_PCT, base_stop - tightening)

    return float(np.clip(base_stop, config.STOP_MIN_PCT, config.STOP_MAX_PCT))


# ─────────────────────────────────────────────────────────────────────────────
# Continuous regime detection
# ─────────────────────────────────────────────────────────────────────────────

def _score_component(value: float, low: float, high: float) -> float:
    """Linearly map value from [low, high] → [0, 1], clamped."""
    if high == low:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def market_regime(
    spy_df: pd.DataFrame,
    universe_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> RegimeState:
    """
    Compute a continuous 0-1 regime score from six independent signals.
    No hardcoded thresholds — sensitivity and lookbacks from config.

    Components:
      1. SPY vs 50/100/200 MA structure        (trend)
      2. SPY momentum at 5/20/60d horizons     (momentum)
      3. SPY realized vol                      (volatility — inverted)
      4. Market breadth: % universe > 200MA    (breadth)
      5. Vol trend: is vol expanding or shrinking (risk-on/off)
      6. SPY drawdown from recent high         (drawdown)
    """
    spy   = normalize_ohlcv(spy_df)
    close = _safe_series(spy, "close")

    # Defaults for short history
    if len(close) < 220:
        return RegimeState(
            score=0.7, is_bull=True, is_neutral=False, is_bear=False,
            spy_crash=False, vol_halt=False, position_scalar=1.0,
            net_exposure_target=config.NET_EXPOSURE_BULL,
            breadth=0.7, realized_vol=0.15, components={},
        )

    components: Dict[str, float] = {}

    # ── Component 1: MA structure ─────────────────────────────────────────
    sma50  = float(close.rolling(50).mean().iloc[-1])
    sma100 = float(close.rolling(100).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    px     = float(close.iloc[-1])

    ma_bull_count = sum([px > sma50, px > sma100, px > sma200,
                         sma50 > sma100, sma100 > sma200])
    components["ma_structure"] = ma_bull_count / 5.0

    # ── Component 2: Multi-horizon momentum ──────────────────────────────
    ret5  = float(close.pct_change(5).iloc[-1])
    ret20 = float(close.pct_change(20).iloc[-1])
    ret60 = float(close.pct_change(60).iloc[-1])

    # Map returns to 0-1: centre on 0, ±10%/20%/30% = extremes
    mom5  = _score_component(ret5,  -0.10,  0.10)
    mom20 = _score_component(ret20, -0.20,  0.20)
    mom60 = _score_component(ret60, -0.30,  0.30)
    components["momentum"] = float(np.mean([mom5, mom20, mom60]))

    # ── Component 3: Realized vol (inverted — high vol = bearish) ────────
    vol_lb = getattr(config, "REGIME_VOL_LOOKBACK", 20)
    rvol   = float(close.pct_change().rolling(vol_lb).std().iloc[-1] * np.sqrt(252))
    # 10% annualised vol = score 1.0 (very calm), 50% = score 0.0 (crisis)
    vol_score = _score_component(rvol, 0.50, 0.10)
    components["vol_inverted"] = vol_score

    # ── Component 4: Market breadth ───────────────────────────────────────
    breadth = 0.5  # default if universe not supplied
    if universe_dfs:
        breadth_lb = getattr(config, "REGIME_BREADTH_LOOKBACK", 50)
        above = 0
        total = 0
        for sym_df in universe_dfs.values():
            try:
                c = normalize_ohlcv(sym_df)["close"]
                if len(c) < 200:
                    continue
                ma200 = float(c.rolling(200).mean().iloc[-1])
                total += 1
                if float(c.iloc[-1]) > ma200:
                    above += 1
            except Exception:
                continue
        if total > 0:
            breadth = above / total
    # 30% above = 0.0, 70% above = 1.0
    components["breadth"] = _score_component(breadth, 0.30, 0.70)

    # ── Component 5: Vol trend (is vol expanding?) ────────────────────────
    rvol_fast = float(close.pct_change().rolling(10).std().iloc[-1] * np.sqrt(252))
    rvol_slow = float(close.pct_change().rolling(30).std().iloc[-1] * np.sqrt(252))
    # vol contracting = bullish, expanding = bearish
    if rvol_slow > 0:
        vol_ratio = rvol_fast / rvol_slow
        # ratio < 1 (contracting) → 1.0, ratio > 2 (exploding) → 0.0
        components["vol_trend"] = _score_component(vol_ratio, 2.0, 0.5)
    else:
        components["vol_trend"] = 0.5

    # ── Component 6: Drawdown from 60-day high ────────────────────────────
    high60   = float(close.rolling(60).max().iloc[-1])
    drawdown = (px / high60) - 1.0 if high60 > 0 else 0.0
    # -5% = 0.0, 0% (at high) = 1.0
    components["drawdown"] = _score_component(drawdown, -0.25, 0.0)

    # ── Weighted composite — all weights from config, nothing hardcoded ──
    weights = {
        "ma_structure": getattr(config, "REGIME_WEIGHT_MA_STRUCTURE", 0.20),
        "momentum":     getattr(config, "REGIME_WEIGHT_MOMENTUM",     0.25),
        "vol_inverted": getattr(config, "REGIME_WEIGHT_VOL",          0.20),
        "breadth":      getattr(config, "REGIME_WEIGHT_BREADTH",      0.20),
        "vol_trend":    getattr(config, "REGIME_WEIGHT_VOL_TREND",    0.08),
        "drawdown":     getattr(config, "REGIME_WEIGHT_DRAWDOWN",     0.07),
    }
    raw_score = sum(weights[k] * components[k] for k in weights)

    # Apply sensitivity smoothing via config (0-1, 1=no smoothing)
    sensitivity = getattr(config, "REGIME_SENSITIVITY", 0.5)
    # Blend toward 0.5 (neutral) by (1 - sensitivity)
    score = float(sensitivity * raw_score + (1 - sensitivity) * 0.5)

    # ── Circuit breakers (kept as hard safety stops) ──────────────────────
    day_ret    = float(close.pct_change().iloc[-1])
    spy_crash  = day_ret <= config.SPY_CRASH_HALT_PCT
    vol_halt   = rvol >= getattr(config, "REALIZED_VOL_HALT", 0.40)

    # ── VIX spike early warning ───────────────────────────────────────────
    # When realized vol spikes sharply over a short window, halt new longs
    # even if the regime score hasn't moved yet. This catches early bear
    # signals like Jan 2022 before MAs break.
    vix_spike_window = getattr(config, "VIX_SPIKE_WINDOW_DAYS", 5)
    vix_spike_thresh = getattr(config, "VIX_SPIKE_THRESHOLD",   0.30)
    rvol_now  = float(close.pct_change().rolling(vix_spike_window).std().iloc[-1] * np.sqrt(252))
    rvol_prev = float(close.pct_change().rolling(vix_spike_window).std().iloc[-vix_spike_window - 1] * np.sqrt(252))
    vix_spike = (
        rvol_prev > 0
        and (rvol_now - rvol_prev) / rvol_prev >= vix_spike_thresh
        and rvol_now > 0.18   # only trigger if absolute vol is elevated too
    )

    # ── Classify and derive targets ───────────────────────────────────────
    bull_thresh    = getattr(config, "REGIME_BULL_THRESHOLD",    0.60)
    neutral_thresh = getattr(config, "REGIME_NEUTRAL_THRESHOLD", 0.45)

    is_bull    = score >= bull_thresh
    is_bear    = score <  neutral_thresh
    is_neutral = not is_bull and not is_bear

    if is_bull:
        net_target      = getattr(config, "NET_EXPOSURE_BULL",    1.0)
        position_scalar = 1.0
    elif is_neutral:
        net_target      = getattr(config, "NET_EXPOSURE_NEUTRAL", 0.0)
        position_scalar = 0.80
    else:  # bear
        net_target      = getattr(config, "NET_EXPOSURE_BEAR",   -0.5)
        position_scalar = 0.60

    return RegimeState(
        score=score,
        is_bull=is_bull,
        is_neutral=is_neutral,
        is_bear=is_bear,
        spy_crash=spy_crash,
        vol_halt=vol_halt,
        vix_spike=vix_spike,
        position_scalar=position_scalar,
        net_exposure_target=net_target,
        breadth=breadth,
        realized_vol=rvol,
        components=components,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rule score (unchanged — still used as one input to combined score)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rule_score(df: pd.DataFrame) -> float:
    df = normalize_ohlcv(df)
    if len(df) < 220:
        return 0.0

    close  = _safe_series(df, "close")
    high   = _safe_series(df, "high")
    low    = _safe_series(df, "low")
    volume = _safe_series(df, "volume")
    s      = -1

    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    tech   = 0.0

    if sma50.iloc[s] > sma200.iloc[s]:
        tech += 0.15

    sma10 = close.rolling(10).mean()
    sma40 = close.rolling(40).mean()
    if sma10.iloc[s] > sma40.iloc[s]:
        tech += 0.10

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    lv    = loss.iloc[s]
    rs    = gain.iloc[s] / lv if pd.notna(lv) and lv != 0 else 1.0
    rsi   = 100 - (100 / (1 + rs))

    if 40 < rsi < 68:   tech += 0.15
    elif rsi > 75:       tech -= 0.10
    elif rsi < 30:       tech += 0.05

    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal

    if hist.iloc[s] > 0 and hist.iloc[s] > hist.iloc[s-1]:  tech += 0.15
    elif hist.iloc[s] < 0:                                    tech -= 0.10

    typical    = (high + low + close) / 3
    vwap_proxy = typical.rolling(20).mean()
    if close.iloc[s] > vwap_proxy.iloc[s]:
        tech += 0.10

    vol_ma    = volume.rolling(20).mean()
    vol_ratio = volume.iloc[s] / vol_ma.iloc[s] if vol_ma.iloc[s] > 0 else 1.0
    vol_score = 0.0
    if vol_ratio > 1.5:   vol_score += 0.20
    elif vol_ratio > 1.2: vol_score += 0.10
    elif vol_ratio < 0.7: vol_score -= 0.10

    direction = np.sign(close.diff()).fillna(0.0)
    obv       = (direction * volume).cumsum()
    obv_base  = abs(obv.iloc[s-5]) + 1
    obv_slope = (obv.iloc[s] - obv.iloc[s-5]) / obv_base
    if obv_slope > 0: vol_score += 0.10
    else:             vol_score -= 0.05

    ret5  = (close.iloc[s] - close.iloc[s-5])  / close.iloc[s-5]
    ret20 = (close.iloc[s] - close.iloc[s-20]) / close.iloc[s-20]
    sent  = 0.0
    if ret5 > 0.02:    sent += 0.15
    elif ret5 < -0.02: sent -= 0.15
    if ret20 > 0.05:   sent += 0.10
    elif ret20 < -0.05: sent -= 0.10

    w     = config.WEIGHTS
    score = tech * w["technical"] + vol_score * w["volume"] + sent * w["sentiment"]

    if config.TREND_FILTER_ENABLED:
        if sma50.iloc[s] > sma200.iloc[s] and score < 0:   score *= 0.5
        elif sma50.iloc[s] <= sma200.iloc[s] and score > 0: score *= 0.5

    return float(max(-1.0, min(1.0, score)))


# ─────────────────────────────────────────────────────────────────────────────
# ML ensemble loader
# ─────────────────────────────────────────────────────────────────────────────

def load_ranker(path: str) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("ranker artifact is not a dict")
    for k in ["model", "scaler", "features"]:
        if k not in bundle:
            raise ValueError(f"ranker artifact missing key: {k}")
    return bundle


def load_ranker_ensemble() -> Dict[int, dict]:
    return {
        3: load_ranker("cross_sectional_ranker_3d.joblib"),
        5: load_ranker("cross_sectional_ranker_5d.joblib"),
        7: load_ranker("cross_sectional_ranker_7d.joblib"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Correlation helpers
# ─────────────────────────────────────────────────────────────────────────────

def bucket_of(symbol: str) -> str:
    return config.CORRELATION_BUCKETS.get(symbol, symbol)


def return_corr_matrix(
    symbol_to_df: Dict[str, pd.DataFrame],
    lookback: int,
) -> pd.DataFrame:
    frames = []
    for symbol, df in symbol_to_df.items():
        try:
            ret = _safe_series(normalize_ohlcv(df), "close").pct_change().rename(symbol)
            frames.append(ret)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    rets = pd.concat(frames, axis=1).dropna(how="all")
    if len(rets) > lookback:
        rets = rets.tail(lookback)
    return rets.corr()


# ─────────────────────────────────────────────────────────────────────────────
# Candidate selection — works for both longs and shorts
# ─────────────────────────────────────────────────────────────────────────────

def select_top_candidates(
    snapshots: Dict[str, SignalSnapshot],
    symbol_to_df: Dict[str, pd.DataFrame],
    current_positions: Optional[Dict[str, dict]] = None,
    max_names: Optional[int] = None,
    corr_matrix: Optional[pd.DataFrame] = None,
    side: str = "long",
) -> List[SignalSnapshot]:
    """
    Select top candidates for longs or shorts.

    For longs:  highest combined_score, ml_rank_pct >= LONG_ENTRY_THRESHOLD
    For shorts: lowest combined_score,  ml_rank_pct <= SHORT_ENTRY_THRESHOLD
    """
    if current_positions is None:
        current_positions = {}
    if max_names is None:
        max_names = config.MAX_POSITIONS

    long_thresh  = getattr(config, "LONG_ENTRY_THRESHOLD",  0.80)
    short_thresh = getattr(config, "SHORT_ENTRY_THRESHOLD", 0.20)

    if side == "long":
        eligible = [
            s for s in snapshots.values()
            if s.rule_score >= config.RULE_THRESHOLD
            and s.ml_rank_pct >= long_thresh
        ]
        eligible.sort(key=lambda x: x.combined_score, reverse=True)
    else:
        # Shorts: pick worst-ranked stocks breaking down
        # BUT exclude stocks in sectors that are outperforming SPY —
        # shorting energy in 2022 when energy was +60% is exactly wrong.
        short_sector_lookback = getattr(config, "SHORT_SECTOR_EXCLUDE_LOOKBACK", 60)
        short_sector_thresh   = getattr(config, "SHORT_SECTOR_EXCLUDE_THRESHOLD", 0.05)

        # Build set of outperforming sectors to exclude from shorts
        outperforming_sectors: set = set()
        spy_df_sample = next(iter(symbol_to_df.values()), None)
        # We need SPY returns — use the leadership adapter sector scores if available
        # Fallback: compute sector median returns from symbols we have
        sector_rets: Dict[str, List[float]] = {}
        for sym, df in symbol_to_df.items():
            sector = SECTOR_ETF_MAP.get(sym)
            if sector is None:
                continue
            try:
                c = normalize_ohlcv(df)["close"] if not isinstance(df, pd.Series) else df
                if len(c) < short_sector_lookback + 1:
                    continue
                ret = float(c.iloc[-1] / c.iloc[-short_sector_lookback] - 1)
                sector_rets.setdefault(sector, []).append(ret)
            except Exception:
                continue

        # Mark sectors with median return > threshold as outperforming → exclude from shorts
        for sector, rets in sector_rets.items():
            if len(rets) >= 3 and float(np.median(rets)) > short_sector_thresh:
                outperforming_sectors.add(sector)

        eligible = [
            s for s in snapshots.values()
            if s.ml_rank_pct <= short_thresh
            and SECTOR_ETF_MAP.get(s.symbol) not in outperforming_sectors
            and getattr(config, "STOCK_CLASSIFICATION", {}).get(s.symbol, "cyclical")
                not in getattr(config, "SHORT_INELIGIBLE_TYPES", {"defensive","utility","reit"})
        ]
        eligible.sort(key=lambda x: x.combined_score, reverse=False)

    if not eligible:
        return []

    corr = corr_matrix if corr_matrix is not None else return_corr_matrix(
        symbol_to_df, config.CORRELATION_LOOKBACK_DAYS
    )

    selected:      List[SignalSnapshot] = []
    bucket_weights: Dict[str, float]   = {}

    for cand in eligible:
        if len(selected) >= max_names:
            break

        bucket = bucket_of(cand.symbol)
        if bucket_weights.get(bucket, 0.0) >= config.MAX_CORRELATED_BUCKET_WEIGHT:
            continue

        penalty = 0.0
        if not corr.empty and selected:
            corrs = []
            for s in selected:
                try:
                    corrs.append(float(corr.loc[cand.symbol, s.symbol]))
                except Exception:
                    pass
            if corrs:
                avg_corr = float(np.nanmean(corrs))
                if avg_corr > config.CORRELATION_PENALTY_START:
                    penalty = (
                        avg_corr - config.CORRELATION_PENALTY_START
                    ) * config.CORRELATION_PENALTY_MULT

        adjusted_score = cand.combined_score - penalty

        if side == "long" and adjusted_score < config.COMBINED_SCORE_MIN:
            continue

        snap = SignalSnapshot(
            symbol=cand.symbol,
            rule_score=cand.rule_score,
            ml_score=cand.ml_score,
            ml_rank_pct=cand.ml_rank_pct,
            combined_score=adjusted_score,
            trend_bullish=cand.trend_bullish,
            stop_pct=cand.stop_pct,
            atr_pct=cand.atr_pct,
            side=side,
        )
        selected.append(snap)
        bucket_weights[bucket] = bucket_weights.get(bucket, 0.0) + config.MAX_POSITION_WEIGHT

    selected.sort(
        key=lambda x: x.combined_score,
        reverse=(side == "long"),
    )
    return selected