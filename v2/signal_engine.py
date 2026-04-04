"""
signal_engine.py — Unified Real-Time Signal Scanner
====================================================
Integrates all signal sources into a single 60-second loop.
Each signal: symbol, signal_type, score 0-100, action, rationale, timestamp.

Run:  python v2/signal_engine.py              # one-shot scan
      python v2/signal_engine.py --loop       # 60s loop during market hours
      python v2/signal_engine.py --loop 30    # custom interval
      python v2/signal_engine.py --json       # JSON output for downstream
"""
from __future__ import annotations

import sys, os, time, json, logging, traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))
sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2/v2"))

import pandas as pd
import numpy as np

import config

# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    symbol: str
    signal_type: str          # MOMENTUM, MEAN_REVERSION, EARNINGS, INSIDER, etc.
    score: int                # 0-100 (normalized)
    action: str               # BUY, SELL, SHORT, COVER, HOLD, ALERT
    rationale: str
    timestamp: str            # ISO format
    source: str               # provider name
    confidence: float = 0.5   # 0-1
    regime: str = ""
    meta: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(os.path.expanduser("~/ai_trading_bot_v2"))
CACHE_PRICES = ROOT / "cache_prices"
CACHE_EARN   = ROOT / "cache_earnings_streak"
CACHE_INSIDER = ROOT / "cache_insider"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "signal_engine.log"),
    ],
)
log = logging.getLogger("signal_engine")


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def is_market_hours() -> bool:
    """True if US market is open (9:30-16:00 ET, weekdays)."""
    from zoneinfo import ZoneInfo
    et = datetime.now(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    t = et.time()
    from datetime import time as dt_time
    return dt_time(9, 30) <= t <= dt_time(16, 0)


def load_price_df(sym: str) -> Optional[pd.DataFrame]:
    """Load cached OHLCV from yfinance cache."""
    path = CACHE_PRICES / f"{sym}_3650d.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception:
        return None


def load_json_store(directory: Path) -> Dict[str, pd.DataFrame]:
    """Load all JSONs in a cache dir into a dict of DataFrames."""
    store = {}
    if not directory.exists():
        return store
    for f in directory.glob("*.json"):
        sym = f.stem.upper()
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                store[sym] = df
        except Exception:
            pass
    return store


def norm_score(raw: float, lo: float, hi: float) -> int:
    """Normalize raw score from [lo, hi] to 0-100 int."""
    if hi == lo:
        return 50
    return max(0, min(100, int(((raw - lo) / (hi - lo)) * 100)))


def score_to_action(score: int, short: bool = False) -> str:
    """Convert 0-100 score to action label."""
    if short:
        if score >= 80: return "SHORT"
        if score >= 60: return "ALERT"
        return "HOLD"
    if score >= 80: return "BUY"
    if score >= 60: return "ALERT"
    if score <= 20: return "SELL"
    return "HOLD"


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL PROVIDERS — each wraps an existing codebase signal source
# ═══════════════════════════════════════════════════════════════════════════════

class ProviderRegistry:
    """Central registry of signal providers with TTL caching."""

    def __init__(self):
        self._providers: List[dict] = []
        self._cache: Dict[str, tuple] = {}  # key → (signals, expiry_time)

    def register(self, name: str, fn: Callable, ttl: int = 60,
                 signal_type: str = "COMPOSITE", per_symbol: bool = True):
        self._providers.append({
            "name": name, "fn": fn, "ttl": ttl,
            "signal_type": signal_type, "per_symbol": per_symbol,
        })

    def scan_all(self, symbols: List[str], context: dict) -> List[Signal]:
        """Run all providers, return flat list of signals."""
        all_signals = []
        for p in self._providers:
            cache_key = p["name"]
            now = time.time()
            # Check TTL cache
            if cache_key in self._cache:
                cached_sigs, expiry = self._cache[cache_key]
                if now < expiry:
                    all_signals.extend(cached_sigs)
                    continue

            try:
                if p["per_symbol"]:
                    sigs = []
                    for sym in symbols:
                        result = p["fn"](sym, context)
                        if result:
                            sigs.extend(result if isinstance(result, list) else [result])
                    self._cache[cache_key] = (sigs, now + p["ttl"])
                    all_signals.extend(sigs)
                else:
                    sigs = p["fn"](symbols, context)
                    if sigs:
                        result = sigs if isinstance(sigs, list) else [sigs]
                        self._cache[cache_key] = (result, now + p["ttl"])
                        all_signals.extend(result)
            except Exception as e:
                log.warning(f"Provider {p['name']} failed: {e}")

        return all_signals


# ═══════════════════════════════════════════════════════════════════════════════
#  PROVIDER IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Regime ────────────────────────────────────────────────────────────────

def _provide_regime(symbols: List[str], ctx: dict) -> List[Signal]:
    """Market regime classification — runs once, applies to all."""
    try:
        from regime_classifier import RegimeClassifier
    except ImportError:
        return []

    clf = ctx.get("regime_classifier")
    if clf is None:
        clf = RegimeClassifier()
        ctx["regime_classifier"] = clf

    spy_df = load_price_df("SPY")
    vix_df = load_price_df("^VIX") or _load_vix_alt()
    hyg_df = load_price_df("HYG")
    if spy_df is None:
        return []

    today = pd.Timestamp.now().normalize()
    signals_dict = {}
    if spy_df is not None and len(spy_df) > 200:
        spy = spy_df["close"]
        signals_dict["spy_20d"] = float((spy.iloc[-1] / spy.iloc[-20] - 1) if len(spy) > 20 else 0)
        signals_dict["spy_60d"] = float((spy.iloc[-1] / spy.iloc[-60] - 1) if len(spy) > 60 else 0)
        sma200 = float(spy.iloc[-200:].mean())
        signals_dict["spy_vs_200ma"] = float((spy.iloc[-1] - sma200) / sma200)
        signals_dict["spy_slope"] = float(spy.iloc[-20:].mean() - spy.iloc[-25:-5].mean()) / max(sma200, 1)
    if vix_df is not None and len(vix_df) > 20:
        vix = vix_df["close"]
        signals_dict["vix_level"] = float(vix.iloc[-1])
        signals_dict["vix_20d_chg"] = float((vix.iloc[-1] / vix.iloc[-20] - 1) if len(vix) > 20 else 0)
    if hyg_df is not None and len(hyg_df) > 20:
        hyg = hyg_df["close"]
        signals_dict["hyg_20d"] = float((hyg.iloc[-1] / hyg.iloc[-20] - 1) if len(hyg) > 20 else 0)

    regime = clf.update(today, signals_dict)
    ctx["regime"] = regime

    vix_level = signals_dict.get("vix_level", 0)
    score = {"TRENDING_BULL": 80, "CHOPPY": 50, "BEAR": 20}.get(regime, 50)
    return [Signal(
        symbol="SPY", signal_type="REGIME", score=score,
        action="BUY" if regime == "TRENDING_BULL" else "SELL" if regime == "BEAR" else "HOLD",
        rationale=f"Regime={regime}, VIX={vix_level:.1f}, SPY vs 200MA={signals_dict.get('spy_vs_200ma', 0):+.1%}",
        timestamp=now_iso(), source="regime_classifier",
        confidence=0.8, regime=regime,
        meta=signals_dict,
    )]


def _load_vix_alt() -> Optional[pd.DataFrame]:
    for name in ["^VIX_etf.csv", "^VIX_9999d.csv"]:
        path = CACHE_PRICES / name
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            return df[["close"]].dropna()
    return None


# ── 2. Macro Signals ─────────────────────────────────────────────────────────

def _provide_macro(symbols: List[str], ctx: dict) -> List[Signal]:
    """Factor health, VIX spike, IVTS backwardation."""
    try:
        from macro_signal import compute_signals
    except ImportError:
        return []
    try:
        data = compute_signals()
    except Exception as e:
        log.debug(f"macro_signal failed: {e}")
        return []

    sigs = []
    entry_scalar = data.get("entry_scalar", 1.0)
    score = norm_score(entry_scalar, 0, 1)
    parts = []
    if data.get("momentum_dead"):
        parts.append(f"MomDead({data.get('momentum_dead_streak', 0)}d)")
    if data.get("vix_spike_active"):
        parts.append(f"VIXspike({data.get('vix_10d_chg', 0):+.0%})")
    if data.get("backwardation"):
        parts.append(f"IVTS backwardation({data.get('ivts', 0):.2f})")

    rationale = f"EntryScalar={entry_scalar:.2f}"
    if parts:
        rationale += " | " + ", ".join(parts)

    action = "HOLD"
    if entry_scalar < 0.3:
        action = "SELL"
    elif entry_scalar >= 0.9:
        action = "BUY"

    sigs.append(Signal(
        symbol="MACRO", signal_type="MACRO", score=score,
        action=action, rationale=rationale,
        timestamp=now_iso(), source="macro_signal",
        confidence=0.7, regime=ctx.get("regime", ""),
        meta=data,
    ))
    return sigs


# ── 3. PEAD (Post-Earnings Drift) ────────────────────────────────────────────

def _provide_pead(sym: str, ctx: dict) -> List[Signal]:
    """Post-earnings announcement drift signal."""
    try:
        from pead_signal import pead_score
    except ImportError:
        return []

    store = ctx.get("earnings_store")
    if store is None:
        store = load_json_store(CACHE_EARN)
        ctx["earnings_store"] = store

    today = pd.Timestamp.now().normalize()
    try:
        raw = pead_score(sym, today, store)
    except Exception:
        return []

    if abs(raw) < 0.05:
        return []

    score = norm_score(raw, -1, 1)
    action = "BUY" if raw > 0.3 else "SHORT" if raw < -0.3 else "HOLD"
    return [Signal(
        symbol=sym, signal_type="EARNINGS", score=score,
        action=action,
        rationale=f"PEAD={raw:+.2f} ({'beat drift' if raw > 0 else 'miss drift'})",
        timestamp=now_iso(), source="pead_signal",
        confidence=min(abs(raw), 1.0), regime=ctx.get("regime", ""),
    )]


# ── 4. Earnings Beat Streak ──────────────────────────────────────────────────

def _provide_streak(sym: str, ctx: dict) -> List[Signal]:
    """Consecutive earnings beat streak."""
    try:
        from earnings_streak import beat_streak_score
    except ImportError:
        return []

    store = ctx.get("earnings_store")
    if store is None:
        store = load_json_store(CACHE_EARN)
        ctx["earnings_store"] = store

    today = pd.Timestamp.now().normalize()
    try:
        raw = beat_streak_score(sym, today, store)
    except Exception:
        return []

    if raw < 0.25:
        return []

    score = norm_score(raw, 0, 1)
    streaks = {0.3: "1 beat", 0.6: "2 beats", 0.85: "3 beats", 1.0: "4+ beats"}
    label = "4+ beats" if raw >= 0.95 else "3 beats" if raw >= 0.8 else "2 beats" if raw >= 0.55 else "1 beat"
    return [Signal(
        symbol=sym, signal_type="EARNINGS", score=score,
        action="BUY" if raw >= 0.6 else "ALERT",
        rationale=f"BeatStreak={label} (score={raw:.2f})",
        timestamp=now_iso(), source="earnings_streak",
        confidence=raw, regime=ctx.get("regime", ""),
    )]


# ── 5. Analyst Revision Momentum ─────────────────────────────────────────────

def _provide_revision(sym: str, ctx: dict) -> List[Signal]:
    """Analyst estimate revision momentum."""
    try:
        from analyst_revision import revision_momentum_score
    except ImportError:
        return []

    store = ctx.get("earnings_store")
    if store is None:
        store = load_json_store(CACHE_EARN)
        ctx["earnings_store"] = store

    today = pd.Timestamp.now().normalize()
    try:
        raw = revision_momentum_score(sym, today, store)
    except Exception:
        return []

    if abs(raw) < 0.1:
        return []

    score = norm_score(raw, -1, 1)
    return [Signal(
        symbol=sym, signal_type="FUNDAMENTAL", score=score,
        action="BUY" if raw > 0.4 else "SHORT" if raw < -0.4 else "HOLD",
        rationale=f"RevisionMomentum={raw:+.2f}",
        timestamp=now_iso(), source="analyst_revision",
        confidence=min(abs(raw), 1.0), regime=ctx.get("regime", ""),
    )]


# ── 6. Insider Flow ──────────────────────────────────────────────────────────

def _provide_insider(sym: str, ctx: dict) -> List[Signal]:
    """SEC Form 4 insider cluster detection."""
    try:
        from insider_signal import insider_score
    except ImportError:
        return []

    store = ctx.get("insider_store")
    if store is None:
        store = load_json_store(CACHE_INSIDER)
        ctx["insider_store"] = store

    today = pd.Timestamp.now().normalize()
    try:
        raw = insider_score(sym, today, store)
    except Exception:
        return []

    if raw < 0.3:
        return []

    score = norm_score(raw, 0, 1)
    label = "cluster 3+" if raw >= 0.95 else "cluster 2" if raw >= 0.75 else "single buy"
    return [Signal(
        symbol=sym, signal_type="INSIDER", score=score,
        action="BUY" if raw >= 0.8 else "ALERT",
        rationale=f"InsiderBuying={label} (score={raw:.2f})",
        timestamp=now_iso(), source="insider_signal",
        confidence=raw, regime=ctx.get("regime", ""),
    )]


# ── 7. Short Interest Velocity ────────────────────────────────────────────────

def _provide_si(sym: str, ctx: dict) -> List[Signal]:
    """Short interest level and velocity."""
    try:
        from si_velocity import compute_si_features
    except ImportError:
        return []

    df = load_price_df(sym)
    if df is None:
        return []
    try:
        feats = compute_si_features(sym, df)
    except Exception:
        return []

    if not feats:
        return []

    si_level = feats.get("si_level", 0)
    squeeze_risk = feats.get("si_squeeze_risk", 0)

    sigs = []
    if si_level > 15:
        score = norm_score(si_level, 10, 50)
        sigs.append(Signal(
            symbol=sym, signal_type="SHORT_INTEREST", score=score,
            action="ALERT",
            rationale=f"SI={si_level:.1f}%, velocity={feats.get('si_velocity', 0):+.1f}",
            timestamp=now_iso(), source="si_velocity",
            confidence=min(si_level / 40, 1.0), regime=ctx.get("regime", ""),
            meta=feats,
        ))
    if squeeze_risk > 0.5:
        score = norm_score(squeeze_risk, 0, 1)
        sigs.append(Signal(
            symbol=sym, signal_type="SQUEEZE", score=score,
            action="BUY" if squeeze_risk > 0.7 else "ALERT",
            rationale=f"SqueezeRisk={squeeze_risk:.2f}, SI={si_level:.1f}%",
            timestamp=now_iso(), source="si_velocity",
            confidence=squeeze_risk, regime=ctx.get("regime", ""),
        ))
    return sigs


# ── 8. Rule Score (Technical) ────────────────────────────────────────────────

def _provide_rule_score(sym: str, ctx: dict) -> List[Signal]:
    """Technical rule-based composite score."""
    try:
        from backtester_clean import compute_rule_score_vectorised
    except ImportError:
        return []

    df = load_price_df(sym)
    if df is None or len(df) < 220:
        return []

    try:
        scores = compute_rule_score_vectorised(df)
        raw = float(scores.iloc[-1])
    except Exception:
        return []

    if abs(raw) < 0.1:
        return []

    score = norm_score(raw, -1, 1)
    return [Signal(
        symbol=sym, signal_type="MOMENTUM", score=score,
        action=score_to_action(score),
        rationale=f"RuleScore={raw:+.2f} (trend+momentum+volume)",
        timestamp=now_iso(), source="rule_score",
        confidence=min(abs(raw), 1.0), regime=ctx.get("regime", ""),
    )]


# ── 9. Sector Leadership ─────────────────────────────────────────────────────

def _provide_sector(sym: str, ctx: dict) -> List[Signal]:
    """Sector relative strength multiplier."""
    try:
        from sector_leadership import LeadershipScorer
    except ImportError:
        return []

    scorer = ctx.get("leadership_scorer")
    if scorer is None:
        try:
            sector_map = getattr(config, "SECTOR_ETFS", {})
            scorer = LeadershipScorer(sector_map=sector_map)
            ctx["leadership_scorer"] = scorer
        except Exception:
            return []

    try:
        multiplier = scorer.score_multiplier(sym) if hasattr(scorer, "score_multiplier") else 1.0
    except Exception:
        return []

    if abs(multiplier - 1.0) < 0.1:
        return []

    score = norm_score(multiplier, 0.5, 1.5)
    label = "leading" if multiplier > 1.2 else "strong" if multiplier > 1.0 else "lagging"
    return [Signal(
        symbol=sym, signal_type="SECTOR", score=score,
        action="BUY" if multiplier > 1.2 else "SELL" if multiplier < 0.7 else "HOLD",
        rationale=f"SectorLeadership={multiplier:.2f}x ({label})",
        timestamp=now_iso(), source="sector_leadership",
        confidence=abs(multiplier - 1.0), regime=ctx.get("regime", ""),
    )]


# ── 10. Entry Filter (A/D) ───────────────────────────────────────────────────

def _provide_entry_filter(sym: str, ctx: dict) -> List[Signal]:
    """Accumulation/Distribution entry gating."""
    try:
        from entry_filter import is_in_accumulation
    except ImportError:
        return []

    df = load_price_df(sym)
    if df is None or len(df) < 20:
        return []

    regime = ctx.get("regime", "CHOPPY")
    try:
        acc = is_in_accumulation(df, regime)
    except Exception:
        return []

    if acc:
        return []  # Only emit signal when BLOCKED (distribution detected)

    return [Signal(
        symbol=sym, signal_type="FILTER", score=20,
        action="HOLD",
        rationale=f"Distribution detected (regime={regime}) — entry blocked",
        timestamp=now_iso(), source="entry_filter",
        confidence=0.7, regime=regime,
    )]


# ── 11. News Sentiment ───────────────────────────────────────────────────────

def _provide_news(symbols: List[str], ctx: dict) -> List[Signal]:
    """Polygon news sentiment scan."""
    try:
        from polygon_news import scan_watchlist
    except ImportError:
        return []

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        return []

    try:
        results = scan_watchlist(symbols, force=False)
    except Exception as e:
        log.debug(f"polygon_news failed: {e}")
        return []

    sigs = []
    for sym, data in results.items():
        if not data:
            continue
        raw = data.get("score", 0)
        if abs(raw) < 0.2:
            continue
        score = norm_score(raw, -1, 1)
        signal_label = data.get("signal", "NEUTRAL")
        headlines = data.get("headlines", [])[:2]
        headline_str = " | ".join(headlines) if headlines else ""
        sigs.append(Signal(
            symbol=sym, signal_type="SENTIMENT", score=score,
            action="BUY" if signal_label == "BULLISH" else "SHORT" if signal_label == "BEARISH" else "HOLD",
            rationale=f"NewsSentiment={raw:+.2f} ({signal_label}): {headline_str[:120]}",
            timestamp=now_iso(), source="polygon_news",
            confidence=min(abs(raw), 1.0), regime=ctx.get("regime", ""),
        ))
    return sigs


# ── 12. Put/Call Ratio ────────────────────────────────────────────────────────

def _provide_pcr(symbols: List[str], ctx: dict) -> List[Signal]:
    """CBOE put/call ratio contrarian signal."""
    try:
        from put_call_ratio import compute_pcr_features
    except ImportError:
        return []
    try:
        feats = compute_pcr_features()
    except Exception:
        return []

    if not feats:
        return []

    pcr = feats.get("pcr_total", 0)
    fear = feats.get("pcr_fear_signal", False)
    complacency = feats.get("pcr_complacency", False)

    if not fear and not complacency:
        return []

    if fear:
        return [Signal(
            symbol="MARKET", signal_type="SENTIMENT", score=80,
            action="BUY",
            rationale=f"PCR={pcr:.2f} — extreme fear, contrarian bullish",
            timestamp=now_iso(), source="put_call_ratio",
            confidence=0.6, regime=ctx.get("regime", ""),
            meta=feats,
        )]
    else:
        return [Signal(
            symbol="MARKET", signal_type="SENTIMENT", score=20,
            action="SELL",
            rationale=f"PCR={pcr:.2f} — extreme complacency, contrarian bearish",
            timestamp=now_iso(), source="put_call_ratio",
            confidence=0.5, regime=ctx.get("regime", ""),
            meta=feats,
        )]


# ── 13. Congressional Flow ───────────────────────────────────────────────────

def _provide_congress(sym: str, ctx: dict) -> List[Signal]:
    """Congressional stock trade cluster detection."""
    try:
        from congressional_flow import compute_congress_features
    except ImportError:
        return []
    try:
        feats = compute_congress_features(sym)
    except Exception:
        return []

    if not feats:
        return []

    cluster = feats.get("congress_cluster", False)
    net = feats.get("congress_net_buying", 0)

    if not cluster and abs(net) < 0.3:
        return []

    score = norm_score(net, -1, 1)
    return [Signal(
        symbol=sym, signal_type="INSTITUTIONAL", score=score,
        action="BUY" if cluster and net > 0 else "ALERT",
        rationale=f"Congress={'CLUSTER' if cluster else 'active'}, net={net:+.2f}",
        timestamp=now_iso(), source="congressional_flow",
        confidence=0.5 if not cluster else 0.7, regime=ctx.get("regime", ""),
        meta=feats,
    )]


# ── 14. Squeeze Score ─────────────────────────────────────────────────────────

def _provide_squeeze(sym: str, ctx: dict) -> List[Signal]:
    """Short squeeze setup detection."""
    try:
        from squeeze_score import compute_squeeze_score
    except ImportError:
        return []

    df = load_price_df(sym)
    try:
        feats = compute_squeeze_score(sym, df)
    except Exception:
        return []

    if not feats:
        return []

    sq = feats.get("squeeze_score", 0)
    if sq < 0.5:
        return []

    score = norm_score(sq, 0, 1)
    return [Signal(
        symbol=sym, signal_type="SQUEEZE", score=score,
        action="BUY" if sq > 0.7 else "ALERT",
        rationale=f"SqueezeScore={sq:.2f}",
        timestamp=now_iso(), source="squeeze_score",
        confidence=sq, regime=ctx.get("regime", ""),
        meta=feats,
    )]


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL ENGINE — main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class SignalEngine:
    """
    Unified signal scanner. Registers all providers, runs them on schedule,
    deduplicates and ranks output.
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or list(config.WATCHLIST)
        self.registry = ProviderRegistry()
        self.context: dict = {}  # shared state across providers
        self.last_signals: List[Signal] = []
        self._register_all()

    def _register_all(self):
        """Register all signal providers with appropriate TTLs."""
        r = self.registry

        # ── Market-wide (run once per cycle) ──────────────────────────────
        r.register("regime",       _provide_regime,       ttl=300,  per_symbol=False)
        r.register("macro",        _provide_macro,        ttl=300,  per_symbol=False)
        r.register("news",         _provide_news,         ttl=1800, per_symbol=False)
        r.register("pcr",          _provide_pcr,          ttl=900,  per_symbol=False)

        # ── Per-symbol (60s default) ──────────────────────────────────────
        r.register("rule_score",   _provide_rule_score,   ttl=60)
        r.register("entry_filter", _provide_entry_filter, ttl=60)

        # ── Per-symbol (5 min — moderate cost) ────────────────────────────
        r.register("pead",         _provide_pead,         ttl=300)
        r.register("streak",       _provide_streak,       ttl=300)
        r.register("revision",     _provide_revision,     ttl=300)
        r.register("sector",       _provide_sector,       ttl=300)

        # ── Per-symbol (slow — API/cache heavy) ──────────────────────────
        r.register("insider",      _provide_insider,      ttl=3600)
        r.register("si",           _provide_si,           ttl=3600)
        r.register("squeeze",      _provide_squeeze,      ttl=3600)
        r.register("congress",     _provide_congress,     ttl=21600)

    def scan(self) -> List[Signal]:
        """Run all providers and return ranked signals."""
        log.info(f"Scanning {len(self.symbols)} symbols across {len(self.registry._providers)} providers...")
        t0 = time.time()

        raw_signals = self.registry.scan_all(self.symbols, self.context)

        # Sort: highest score first, then by confidence
        raw_signals.sort(key=lambda s: (s.score, s.confidence), reverse=True)

        elapsed = time.time() - t0
        log.info(f"Scan complete: {len(raw_signals)} signals in {elapsed:.1f}s")

        self.last_signals = raw_signals
        return raw_signals

    def top_signals(self, n: int = 20, min_score: int = 50) -> List[Signal]:
        """Return the top N actionable signals above min_score."""
        return [s for s in self.last_signals if s.score >= min_score][:n]

    def signals_for(self, symbol: str) -> List[Signal]:
        """Return all signals for a specific symbol."""
        return [s for s in self.last_signals if s.symbol == symbol]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert last scan to DataFrame."""
        if not self.last_signals:
            return pd.DataFrame()
        return pd.DataFrame([asdict(s) for s in self.last_signals])

    def to_json(self) -> str:
        """JSON output for downstream consumers."""
        return json.dumps([asdict(s) for s in self.last_signals], indent=2, default=str)

    def print_dashboard(self, top_n: int = 30, min_score: int = 40):
        """Pretty-print signal dashboard."""
        regime = self.context.get("regime", "UNKNOWN")
        print(f"\n{'=' * 90}")
        print(f"  SIGNAL ENGINE DASHBOARD    Regime: {regime}    {now_iso()}")
        print(f"{'=' * 90}")

        # Market-wide signals first
        market_sigs = [s for s in self.last_signals if s.symbol in ("SPY", "MACRO", "MARKET")]
        if market_sigs:
            print(f"\n  MARKET SIGNALS:")
            for s in market_sigs:
                print(f"    [{s.score:3d}] {s.source:20s}  {s.action:6s}  {s.rationale[:70]}")

        # Per-symbol signals
        sym_sigs = [s for s in self.last_signals
                    if s.symbol not in ("SPY", "MACRO", "MARKET") and s.score >= min_score]
        if sym_sigs:
            print(f"\n  TOP SIGNALS (score >= {min_score}):")
            print(f"  {'SYM':6s} {'SCORE':>5s} {'TYPE':14s} {'ACTION':7s} {'SOURCE':20s} {'RATIONALE'}")
            print(f"  {'-' * 85}")
            for s in sym_sigs[:top_n]:
                print(f"  {s.symbol:6s} {s.score:5d} {s.signal_type:14s} {s.action:7s} "
                      f"{s.source:20s} {s.rationale[:50]}")

        # Summary stats
        actions = {}
        for s in self.last_signals:
            actions[s.action] = actions.get(s.action, 0) + 1
        total = len(self.last_signals)
        print(f"\n  Total: {total} signals | "
              + " | ".join(f"{a}: {c}" for a, c in sorted(actions.items())))
        print(f"{'=' * 90}\n")

    def run_loop(self, interval: int = 60):
        """Run scan every `interval` seconds during market hours."""
        log.info(f"Starting signal loop (interval={interval}s)")
        while True:
            try:
                if is_market_hours():
                    self.scan()
                    self.print_dashboard()
                    # Save to file
                    output = ROOT / "cache_scanner" / "signal_engine_latest.json"
                    output.parent.mkdir(exist_ok=True)
                    output.write_text(self.to_json())
                else:
                    log.info("Market closed. Sleeping...")
            except KeyboardInterrupt:
                log.info("Shutting down signal engine.")
                break
            except Exception as e:
                log.error(f"Loop error: {e}\n{traceback.format_exc()}")

            time.sleep(interval)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]

    # Parse optional symbol subset
    symbols = None  # default = full watchlist
    json_mode = "--json" in args
    loop_mode = "--loop" in args
    interval = 60

    # Check for custom interval: --loop 30
    if loop_mode:
        idx = args.index("--loop")
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            interval = int(args[idx + 1])

    engine = SignalEngine(symbols=symbols)

    if loop_mode:
        engine.run_loop(interval=interval)
    else:
        signals = engine.scan()
        if json_mode:
            print(engine.to_json())
        else:
            engine.print_dashboard()
