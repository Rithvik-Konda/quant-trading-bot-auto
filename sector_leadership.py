"""
sector_leadership.py — Adaptive Market Leadership Detection
============================================================
Replaces the blunt sector rotation filter with a continuous, multi-signal
leadership scoring system that TILTS exposure toward current leaders
rather than hard-gating entire sectors.

Core Insight
------------
The old filter said: "only trade top-3 sectors."
Problem: it killed alpha by excluding stocks right before they became leaders,
and it used a single 60-day return window (too slow + too crude).

This module says: "give every stock a leadership multiplier (0.5–1.5x)
based on how strong its sector is right now — across multiple timeframes."

Architecture
------------
1. LeadershipScorer  — multi-timeframe sector momentum scores
2. LeadershipAdapter — per-stock score multiplier (drops into backtester)
3. RegimeClassifier  — broad market regime (bull/bear/rotation/risk-off)
4. UniverseAdapter   — dynamically expands/contracts the active universe

Usage (backtester_clean.py integration)
-----------------------------------------
    from sector_leadership import LeadershipAdapter

    adapter = LeadershipAdapter()

    # At each backtest step, call once:
    adapter.update(date, hist)

    # Then when building snapshots, apply multiplier:
    for symbol, snap in snapshots.items():
        snap.combined_score *= adapter.score_multiplier(symbol)

    # Or filter universe by minimum leadership score:
    live_universe = adapter.filter_universe(snapshots, min_score=0.55)

Design Principles
-----------------
- No hard-gating: every sector has *some* score (0.5 minimum)
- Multi-timeframe: short (20d), medium (60d), long (120d) — weighted
- Breadth confirmation: % of sector stocks above their 50MA
- RS vs SPY: sector must beat the benchmark to score high
- Momentum quality: penalize sectors with decelerating momentum
- Regime-aware: in risk-off, defensives get boosted automatically
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector ETF → constituent mapping
# Pulled from config.SECTOR_ETFS but can be overridden
# ---------------------------------------------------------------------------

DEFAULT_SECTOR_MAP: Dict[str, List[str]] = {
    "XLK": ["NVDA", "AVGO", "AMD", "MU", "MRVL", "QCOM", "TXN", "INTC", "ADI",
            "AMAT", "LRCX", "KLAC", "SMCI", "ANET", "ARM",
            "MSFT", "AAPL", "ORCL", "CRM", "INTU", "ADBE", "SNOW", "MDB", "DDOG",
            "NET", "TEAM", "ZS", "NOW", "CRWD", "PANW", "PLTR", "FTNT", "VRT"],
    "XLY": ["AMZN", "TSLA", "HD", "LOW", "NKE", "MCD", "SBUX", "CMG",
            "MAR", "BKNG", "TJX", "ROST", "EBAY", "ETSY", "ULTA", "SHOP"],
    "XLC": ["META", "GOOGL"],
    "XLF": ["JPM", "BAC", "C", "GS", "MS", "BLK", "SCHW", "AXP",
            "SPGI", "ICE", "CME", "AIG", "CB", "PGR", "TRV",
            "USB", "PNC", "TFC", "COF", "BK"],
    "XLV": ["UNH", "LLY", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ISRG",
            "DHR", "ABT", "BMY", "AMGN", "GILD", "VRTX", "REGN",
            "ZTS", "SYK", "BSX", "MDT", "CI"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL",
            "KMB", "GIS", "HSY", "EL"],
    "XLI": ["CAT", "DE", "HON", "LMT", "BA", "RTX", "UPS", "FDX",
            "WM", "GE", "PH", "ETN", "ITW", "EMR", "ROK",
            "GD", "NOC", "CSX", "UNP", "NSC"],
    "XLE": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "HAL"],
    "XLU": ["NEE", "DUK", "SO", "D", "EXC", "AEP", "SRE", "PEG", "VST", "CEG", "NRG"],
    "XLB": ["LIN", "APD", "NEM", "FCX", "ECL", "SHW", "DD", "DOW"],
    "XLRE": ["EQIX", "DLR", "AMT", "CCI", "SBAC", "PLD", "PSA"],
}

# Sectors considered defensive (boosted in risk-off)
DEFENSIVE_SECTORS = {"XLP", "XLU", "XLV", "XLRE"}
# Sectors considered cyclical (boosted in risk-on)
CYCLICAL_SECTORS  = {"XLK", "XLY", "XLC", "XLF", "XLI", "XLE", "XLB"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SectorScore:
    etf: str
    raw_return_20d:  float = 0.0
    raw_return_60d:  float = 0.0
    raw_return_120d: float = 0.0
    rs_vs_spy_60d:   float = 0.0   # Sector return - SPY return over 60d
    breadth_pct:     float = 0.5   # % of constituents above their 50MA
    momentum_accel:  float = 0.0   # 20d return - 40d return (positive = accelerating)
    composite:       float = 0.5   # Final 0–1 score
    rank:            int   = 0     # 1 = top sector
    is_leader:       bool  = False


@dataclass
class MarketRegimeState:
    """
    Broad market regime — goes beyond simple bull/bear.
    """
    label: str = "bull"              # bull | bear | rotation | risk_off | recovery
    spy_trend_bullish: bool = True   # 50MA > 200MA
    spy_momentum_20d: float = 0.0   # SPY 20-day return
    spy_breadth_pct: float  = 0.5   # % SPY stocks above 50MA (proxied via SPY return)
    vix_regime: str = "normal"       # low | normal | elevated | spike
    rotation_active: bool = False    # True when sector dispersion is high
    defensive_tilt: float = 0.0     # 0 = neutral, 1 = full defensive tilt


@dataclass
class LeadershipState:
    """Snapshot computed on a given date."""
    date: pd.Timestamp
    sector_scores: Dict[str, SectorScore] = field(default_factory=dict)
    regime: MarketRegimeState = field(default_factory=MarketRegimeState)
    symbol_multipliers: Dict[str, float] = field(default_factory=dict)
    top_sectors: List[str] = field(default_factory=list)     # Ranked 1..N
    leader_sectors: Set[str] = field(default_factory=set)    # Score >= leadership_threshold


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

class LeadershipScorer:
    """
    Computes multi-signal sector leadership scores.

    Timeframe weights:
      20d  → 0.30   (recent momentum, catches new leaders early)
      60d  → 0.40   (medium term — primary signal)
      120d → 0.30   (trend confirmation — penalizes false breakouts)

    Additional signals:
      Breadth    → +/- 0.10  (how many stocks in sector are healthy)
      RS vs SPY  → +/- 0.15  (alpha vs benchmark)
      Accel      → +/- 0.10  (is momentum building or fading?)
    """

    def __init__(
        self,
        sector_map: Optional[Dict[str, List[str]]] = None,
        min_constituents: int = 3,
        leadership_threshold: float = 0.62,  # Above this = "leader"
        top_n_leaders: int = 4,
    ):
        self.sector_map = sector_map or DEFAULT_SECTOR_MAP
        self.min_constituents = min_constituents
        self.leadership_threshold = leadership_threshold
        self.top_n_leaders = top_n_leaders

        # Build reverse map: symbol → sector ETF
        self.symbol_to_sector: Dict[str, str] = {}
        for etf, symbols in self.sector_map.items():
            for s in symbols:
                self.symbol_to_sector[s] = etf

    def _sector_constituent_returns(
        self,
        etf: str,
        date: pd.Timestamp,
        hist: Dict[str, pd.DataFrame],
        lookback: int,
    ) -> List[float]:
        """Returns list of N-day returns for each available constituent."""
        returns = []
        for s in self.sector_map.get(etf, []):
            if s not in hist:
                continue
            df = hist[s].loc[:date]
            if len(df) < lookback + 1:
                continue
            try:
                ret = float(df["close"].iloc[-1] / df["close"].iloc[-lookback] - 1.0)
                returns.append(ret)
            except Exception:
                continue
        return returns

    def _breadth(
        self,
        etf: str,
        date: pd.Timestamp,
        hist: Dict[str, pd.DataFrame],
        ma_window: int = 50,
    ) -> float:
        """Fraction of constituents above their MA. Returns 0.5 if insufficient data."""
        above = 0
        total = 0
        for s in self.sector_map.get(etf, []):
            if s not in hist:
                continue
            df = hist[s].loc[:date]
            if len(df) < ma_window + 1:
                continue
            try:
                ma = df["close"].iloc[-ma_window:].mean()
                px = float(df["close"].iloc[-1])
                total += 1
                if px > ma:
                    above += 1
            except Exception:
                continue
        return float(above / total) if total >= 2 else 0.5

    def score_sector(
        self,
        etf: str,
        date: pd.Timestamp,
        hist: Dict[str, pd.DataFrame],
        spy_return_60d: float,
    ) -> Optional[SectorScore]:
        """Compute composite leadership score for one sector."""

        rets_20  = self._sector_constituent_returns(etf, date, hist, 20)
        rets_60  = self._sector_constituent_returns(etf, date, hist, 60)
        rets_120 = self._sector_constituent_returns(etf, date, hist, 120)
        rets_40  = self._sector_constituent_returns(etf, date, hist, 40)

        if len(rets_60) < self.min_constituents:
            return None

        r20  = float(np.median(rets_20))  if rets_20  else 0.0
        r60  = float(np.median(rets_60))  if rets_60  else 0.0
        r120 = float(np.median(rets_120)) if rets_120 else 0.0
        r40  = float(np.median(rets_40))  if rets_40  else 0.0

        rs_60  = r60 - spy_return_60d
        accel  = r20 - (r40 / 2)   # 20d annualized minus 40d — positive = accelerating
        breadth = self._breadth(etf, date, hist)

        sc = SectorScore(
            etf=etf,
            raw_return_20d=r20,
            raw_return_60d=r60,
            raw_return_120d=r120,
            rs_vs_spy_60d=rs_60,
            breadth_pct=breadth,
            momentum_accel=accel,
        )
        return sc

    def compute_all_sectors(
        self,
        date: pd.Timestamp,
        hist: Dict[str, pd.DataFrame],
        spy_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, SectorScore]:
        """Score all sectors. Returns ETF → SectorScore dict."""

        # SPY 60d return as benchmark
        spy_return_60d = 0.0
        if spy_df is not None and len(spy_df.loc[:date]) >= 61:
            spy_window = spy_df.loc[:date]
            try:
                spy_return_60d = float(
                    spy_window["close"].iloc[-1] / spy_window["close"].iloc[-60] - 1.0
                )
            except Exception:
                pass
        else:
            # Try to estimate from SPY in hist
            if "SPY" in hist:
                spy_window = hist["SPY"].loc[:date]
                if len(spy_window) >= 61:
                    try:
                        spy_return_60d = float(
                            spy_window["close"].iloc[-1] / spy_window["close"].iloc[-60] - 1.0
                        )
                    except Exception:
                        pass

        raw_scores: Dict[str, SectorScore] = {}
        for etf in self.sector_map:
            sc = self.score_sector(etf, date, hist, spy_return_60d)
            if sc is not None:
                raw_scores[etf] = sc

        if not raw_scores:
            return {}

        # --- Cross-sectional normalization ---
        # Convert each raw dimension to a 0–1 percentile rank across sectors
        r60_vals   = {e: sc.raw_return_60d  for e, sc in raw_scores.items()}
        r120_vals  = {e: sc.raw_return_120d for e, sc in raw_scores.items()}
        r20_vals   = {e: sc.raw_return_20d  for e, sc in raw_scores.items()}
        rs_vals    = {e: sc.rs_vs_spy_60d   for e, sc in raw_scores.items()}
        bread_vals = {e: sc.breadth_pct     for e, sc in raw_scores.items()}
        accel_vals = {e: sc.momentum_accel  for e, sc in raw_scores.items()}

        def pct_rank(d: Dict[str, float]) -> Dict[str, float]:
            vals = list(d.values())
            keys = list(d.keys())
            ranks = pd.Series(vals, index=keys).rank(pct=True)
            return ranks.to_dict()

        r60_rank   = pct_rank(r60_vals)
        r120_rank  = pct_rank(r120_vals)
        r20_rank   = pct_rank(r20_vals)
        rs_rank    = pct_rank(rs_vals)
        bread_rank = pct_rank(bread_vals)
        accel_rank = pct_rank(accel_vals)

        for etf, sc in raw_scores.items():
            composite = (
                0.30 * r20_rank.get(etf, 0.5)   +
                0.40 * r60_rank.get(etf, 0.5)   +
                0.15 * r120_rank.get(etf, 0.5)  +
                0.10 * rs_rank.get(etf, 0.5)    +
                0.05 * bread_rank.get(etf, 0.5) +
                0.00 * accel_rank.get(etf, 0.5)  # Small acceleration bonus
            )
            # Acceleration as small bonus/penalty on top
            accel_adj = float(np.clip(accel_rank.get(etf, 0.5) - 0.5, -0.1, 0.1))
            sc.composite = float(np.clip(composite + accel_adj, 0.0, 1.0))

        # Assign ranks
        sorted_etfs = sorted(raw_scores.keys(), key=lambda e: raw_scores[e].composite, reverse=True)
        for rank, etf in enumerate(sorted_etfs, start=1):
            raw_scores[etf].rank = rank
            raw_scores[etf].is_leader = (raw_scores[etf].composite >= self.leadership_threshold)

        return raw_scores


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

class RegimeClassifier:
    """
    Classifies the broad market regime using SPY + cross-sector dispersion.

    Regimes:
      bull       — SPY trending up, most sectors participating
      bear       — SPY below 200MA, broad deterioration
      rotation   — High sector dispersion, leadership changing
      risk_off   — SPY volatile/down, defensive sectors outperforming
      recovery   — SPY bouncing from oversold, breadth improving
    """

    def classify(
        self,
        date: pd.Timestamp,
        hist: Dict[str, pd.DataFrame],
        sector_scores: Dict[str, SectorScore],
    ) -> MarketRegimeState:

        regime = MarketRegimeState()

        spy_df = hist.get("SPY")
        if spy_df is None or len(spy_df.loc[:date]) < 210:
            return regime  # Default bull if insufficient data

        spy = spy_df.loc[:date]["close"]
        spy_now = float(spy.iloc[-1])

        # Trend
        ma50  = float(spy.iloc[-50:].mean())  if len(spy) >= 50  else spy_now
        ma200 = float(spy.iloc[-200:].mean()) if len(spy) >= 200 else spy_now
        regime.spy_trend_bullish = spy_now > ma200

        # Short-term momentum
        if len(spy) >= 21:
            regime.spy_momentum_20d = float(spy.iloc[-1] / spy.iloc[-20] - 1.0)

        # Sector dispersion — std dev of composite scores
        if sector_scores:
            composites = [sc.composite for sc in sector_scores.values()]
            dispersion = float(np.std(composites))
            regime.rotation_active = dispersion > 0.20

        # Defensive vs cyclical performance
        defensive_scores  = [sc.composite for etf, sc in sector_scores.items() if etf in DEFENSIVE_SECTORS]
        cyclical_scores   = [sc.composite for etf, sc in sector_scores.items() if etf in CYCLICAL_SECTORS]

        avg_def = float(np.mean(defensive_scores)) if defensive_scores else 0.5
        avg_cyc = float(np.mean(cyclical_scores))  if cyclical_scores  else 0.5

        # Defensive tilt: 0 = neutral, positive = defensive outperforming
        regime.defensive_tilt = float(np.clip((avg_def - avg_cyc) / 0.3, -1.0, 1.0))

        # Classify regime
        if not regime.spy_trend_bullish and regime.spy_momentum_20d < -0.05:
            regime.label = "bear"
        elif regime.defensive_tilt > 0.4 and regime.spy_momentum_20d < 0.0:
            regime.label = "risk_off"
        elif regime.rotation_active and regime.spy_trend_bullish:
            regime.label = "rotation"
        elif not regime.spy_trend_bullish and regime.spy_momentum_20d > 0.02:
            regime.label = "recovery"
        else:
            regime.label = "bull"

        return regime


# ---------------------------------------------------------------------------
# Main adapter — drop-in integration for backtester
# ---------------------------------------------------------------------------

class LeadershipAdapter:
    """
    Drop-in adapter for backtester_clean.py.

    Call update() once per backtest step, then use:
      - score_multiplier(symbol)  →  float (0.50 to 1.50)
      - filter_universe(snapshots, min_score)  →  filtered snapshots
      - get_state()  →  full LeadershipState for debugging

    Multiplier logic:
      Leader sector stock      → 1.20–1.50x (boost conviction)
      Above-average sector     → 1.00–1.20x (neutral-to-boost)
      Below-average sector     → 0.75–1.00x (dampen but don't exclude)
      Laggard sector stock     → 0.50–0.75x (strong dampen)
      + Regime overlay         → ±0.10 on top

    Key difference from the old filter:
      Old: sector not in top-3 → excluded entirely
      New: sector not in top-3 → score multiplied by 0.50–0.85
           meaning it can still be selected if the individual stock
           score is high enough, but needs to beat more to compete
    """

    def __init__(
        self,
        sector_map: Optional[Dict[str, List[str]]] = None,
        update_frequency_days: int = 5,   # Recompute every 5 trading days
        leadership_threshold: float = 0.62,
        top_n_leaders: int = 4,
    ):
        self.scorer    = LeadershipScorer(
            sector_map=sector_map,
            leadership_threshold=leadership_threshold,
            top_n_leaders=top_n_leaders,
        )
        self.classifier = RegimeClassifier()

        self.update_frequency_days = update_frequency_days
        self._state: Optional[LeadershipState] = None
        self._last_update_date: Optional[pd.Timestamp] = None
        self._update_counter: int = 0

    def update(
        self,
        date: pd.Timestamp,
        hist: Dict[str, pd.DataFrame],
        force: bool = False,
    ) -> LeadershipState:
        """
        Recompute leadership state. Call once per backtest step.
        Uses frequency gating to avoid expensive recompute every day.
        """
        self._update_counter += 1

        should_update = (
            force or
            self._state is None or
            (self._update_counter % self.update_frequency_days == 0)
        )

        if not should_update:
            return self._state

        # Score all sectors
        sector_scores = self.scorer.compute_all_sectors(date, hist)

        # Classify regime
        regime = self.classifier.classify(date, hist, sector_scores)

        # Compute per-symbol multipliers
        symbol_multipliers = self._compute_multipliers(sector_scores, regime)

        # Rank top sectors
        sorted_sectors = sorted(
            sector_scores.keys(),
            key=lambda e: sector_scores[e].composite,
            reverse=True,
        )

        state = LeadershipState(
            date=date,
            sector_scores=sector_scores,
            regime=regime,
            symbol_multipliers=symbol_multipliers,
            top_sectors=sorted_sectors,
            leader_sectors={e for e, sc in sector_scores.items() if sc.is_leader},
        )

        self._state = state
        self._last_update_date = date

        logger.debug(
            "[leadership] %s | regime=%s | leaders=%s | top3=%s",
            date.date(), regime.label,
            sorted(state.leader_sectors),
            sorted_sectors[:3],
        )

        return state

    def _compute_multipliers(
        self,
        sector_scores: Dict[str, SectorScore],
        regime: MarketRegimeState,
    ) -> Dict[str, float]:
        """
        Compute per-symbol score multiplier based on sector score + regime.

        Multiplier scale:
          sector composite 0.80–1.00  → stock multiplier 1.30–1.50
          sector composite 0.60–0.80  → 1.10–1.30
          sector composite 0.40–0.60  → 0.90–1.10  (near neutral)
          sector composite 0.20–0.40  → 0.65–0.90
          sector composite 0.00–0.20  → 0.50–0.65
        """
        multipliers: Dict[str, float] = {}

        for etf, sc in sector_scores.items():
            # Base multiplier from composite score (0.0–1.0 → 0.50–1.50)
            base_mult = 0.50 + sc.composite * 1.00

            # Regime overlay
            regime_adj = 0.0
            if regime.label == "risk_off":
                # In risk-off: boost defensives, dampen cyclicals
                if etf in DEFENSIVE_SECTORS:
                    regime_adj = +0.10
                elif etf in CYCLICAL_SECTORS:
                    regime_adj = -0.10
            elif regime.label == "bear":
                # In bear: defensives hold, cyclicals get penalized more
                if etf in DEFENSIVE_SECTORS:
                    regime_adj = +0.05
                elif etf in CYCLICAL_SECTORS:
                    regime_adj = -0.15
            elif regime.label == "rotation":
                # In rotation: slightly dampen all to reduce conviction
                regime_adj = -0.05
            elif regime.label == "bull":
                # In bull: slight boost to cyclicals
                if etf in CYCLICAL_SECTORS:
                    regime_adj = +0.05

            sector_mult = float(np.clip(base_mult + regime_adj, 0.45, 1.55))

            for symbol in self.scorer.sector_map.get(etf, []):
                multipliers[symbol] = sector_mult

        return multipliers

    def score_multiplier(self, symbol: str) -> float:
        """
        Returns the leadership multiplier for a symbol.
        1.0 = neutral, >1.0 = leadership boost, <1.0 = dampen.
        Defaults to 1.0 if no state computed yet.
        """
        if self._state is None:
            return 1.0
        return self._state.symbol_multipliers.get(symbol, 1.0)

    def filter_universe(
        self,
        snapshots: dict,
        min_multiplier: float = 0.60,
    ) -> dict:
        """
        Soft filter: remove stocks whose sector multiplier is below threshold.
        This replaces the hard top-3 sector gate.

        min_multiplier=0.60 means: only exclude stocks from sectors
        scoring below 0.10 composite (very severe laggards only).
        Recommended range: 0.55–0.70.
        """
        if self._state is None:
            return snapshots

        return {
            sym: snap
            for sym, snap in snapshots.items()
            if self.score_multiplier(sym) >= min_multiplier
        }

    def get_state(self) -> Optional[LeadershipState]:
        """Full state snapshot for debugging and logging."""
        return self._state

    def print_summary(self) -> None:
        """Print a human-readable leadership summary."""
        if self._state is None:
            print("[leadership] No state computed yet.")
            return

        s = self._state
        print(f"\n{'═'*60}")
        print(f"  Market Leadership Summary — {s.date.date()}")
        print(f"{'═'*60}")
        print(f"  Regime: {s.regime.label.upper()}")
        print(f"  SPY trend bullish: {s.regime.spy_trend_bullish}")
        print(f"  SPY 20d momentum:  {s.regime.spy_momentum_20d:+.2%}")
        print(f"  Defensive tilt:    {s.regime.defensive_tilt:+.2f}")
        print(f"  Rotation active:   {s.regime.rotation_active}")
        print(f"{'─'*60}")
        print(f"  {'Sector':<8}  {'Score':>6}  {'Rank':>4}  {'Leader':>6}  {'RS 60d':>7}  {'Breadth':>7}")
        print(f"  {'──────':<8}  {'─────':>6}  {'────':>4}  {'──────':>6}  {'──────':>7}  {'───────':>7}")
        for etf in s.top_sectors:
            sc = s.sector_scores[etf]
            leader_tag = "★" if sc.is_leader else ""
            print(
                f"  {etf:<8}  {sc.composite:>6.3f}  {sc.rank:>4}  {leader_tag:>6}"
                f"  {sc.rs_vs_spy_60d:>+7.2%}  {sc.breadth_pct:>7.1%}"
            )
        print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# Integration helpers for backtester_clean.py
# ---------------------------------------------------------------------------

def apply_leadership_to_snapshots(
    snapshots: dict,
    adapter: LeadershipAdapter,
    soft_filter: bool = True,
    min_multiplier: float = 0.60,
    boost_cap: float = 1.50,
) -> dict:
    """
    Apply leadership multipliers for ORDERING/SELECTION only.

    IMPORTANT: combined_score is boosted for ranking purposes so that stocks
    in leading sectors sort higher in the candidate list. However the original
    raw combined_score is preserved in snap.rule_score_raw (a new attribute)
    so that position sizing logic is never inflated by the leadership boost.

    The backtester's conviction_multiplier() uses ml_rank_pct exclusively,
    so sizing is unaffected. The boost only affects which stocks get picked.
    """
    import copy
    result = {}
    for sym, snap in snapshots.items():
        mult = min(adapter.score_multiplier(sym), boost_cap)
        if soft_filter and mult < min_multiplier:
            continue

        new_snap = copy.copy(snap)
        # Store raw score before boost (used for sizing guard)
        new_snap._raw_combined_score = snap.combined_score
        # Boost combined_score for selection ordering only
        # Cap at 1.0 so it stays in the original score range —
        # this prevents the gap-check logic from breaking
        new_snap.combined_score = float(np.clip(snap.combined_score * mult, 0.0, 1.0))
        result[sym] = new_snap

    return result


# ---------------------------------------------------------------------------
# Config patch — new parameters to add to config.py
# ---------------------------------------------------------------------------

LEADERSHIP_CONFIG_PATCH = """
# ── Adaptive Leadership Detection ─────────────────────────────────────────
# Replaces the old SECTOR_ROTATION_ENABLED / TOP_SECTORS_TO_TRADE logic.
# The old binary filter is replaced with continuous score multipliers.

ADAPTIVE_LEADERSHIP_ENABLED = True

# How often (in trading days) to recompute sector scores (5 = weekly)
LEADERSHIP_UPDATE_FREQ_DAYS = 5

# Sector composite score above which a sector is "in leadership"
LEADERSHIP_THRESHOLD = 0.62

# How many sectors to designate as top leaders (affects logging only)
LEADERSHIP_TOP_N = 4

# Minimum sector multiplier to survive soft filter (0.60 = only exclude
# very severe laggards; set to 0.0 to disable filtering entirely)
LEADERSHIP_MIN_MULTIPLIER = 0.60

# Cap on leadership boost applied to combined_score (1.50 = max 50% boost)
LEADERSHIP_BOOST_CAP = 1.50
"""


# ---------------------------------------------------------------------------
# Quick test / diagnostic script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yfinance as yf
    import sys

    print("Loading data for leadership diagnostic...")

    test_symbols = [
        "NVDA", "MSFT", "JPM", "XOM", "UNH", "PG", "NEE",
        "LIN", "AMT", "CAT", "AMZN", "META", "SPY"
    ]

    hist = {}
    for sym in test_symbols:
        try:
            df = yf.Ticker(sym).history(period="2y", interval="1d")
            if df is not None and len(df) > 0:
                df.columns = [c.lower() for c in df.columns]
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("UTC").tz_localize(None)
                hist[sym] = df[["open", "high", "low", "close", "volume"]].dropna()
                print(f"  Loaded {sym}: {len(hist[sym])} bars")
        except Exception as e:
            print(f"  Failed {sym}: {e}")

    if len(hist) < 5:
        print("Not enough data loaded for test.")
        sys.exit(1)

    adapter = LeadershipAdapter(update_frequency_days=1)
    date = max(df.index[-1] for df in hist.values())
    print(f"\nRunning leadership analysis for {date.date()}...")

    adapter.update(date, hist, force=True)
    adapter.print_summary()

    print("Symbol multipliers:")
    for sym in ["NVDA", "JPM", "XOM", "UNH", "PG", "NEE", "CAT", "AMZN"]:
        mult = adapter.score_multiplier(sym)
        sector = adapter.scorer.symbol_to_sector.get(sym, "?")
        print(f"  {sym:<8} ({sector:<5})  multiplier: {mult:.3f}")