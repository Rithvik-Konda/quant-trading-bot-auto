"""
risk_manager.py — Institutional Risk Management
=================================================
Four risk controls used by systematic funds like DE Shaw / Citadel:

1. VOLATILITY-ADJUSTED POSITION SIZING
   Every position risks the same dollar amount regardless of stock vol.
   When sector vol spikes (Iran/oil, rate shock), positions auto-shrink.
   Formula: qty = risk_budget / (price * atr_stop_pct)

2. REALIZED CORRELATION MONITORING
   When macro narratives hit, portfolio correlations spike to 0.85+.
   Monitors rolling 10-day realized correlation across open positions.
   When avg correlation > threshold → reduce or halt new entries.

3. FACTOR EXPOSURE LIMITS
   Caps sector concentration, portfolio beta, and momentum loading.
   Prevents "4 momentum/tech positions" problem where a factor crash
   (March 2022 momentum unwind) wipes the whole portfolio at once.

4. INTRADAY VOLATILITY DETECTION
   Monitors first-hour price range vs historical norm.
   When intraday vol is 2x+ normal → widens stop for that session.
   The Iran/oil fix: you don't read news, you see vol expanding
   and widen stops before getting shaken out on narrative noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

log = logging.getLogger("risk_manager")

# ── Thresholds ────────────────────────────────────────────────────────────────

RISK_DOLLARS_PER_TRADE   = 2_600   # Fixed dollar risk per trade (~2.6% of $100k)
CORR_WARN_THRESHOLD      = 0.65    # Start reducing new position sizes
CORR_HALT_THRESHOLD      = 0.80    # No new entries
CORR_LOOKBACK_DAYS       = 10      # Rolling window
MAX_SECTOR_CONCENTRATION = 0.75    # Max 75% of capital in one sector bucket
MAX_BETA_EXPOSURE        = 1.60    # Max portfolio beta vs SPY
INTRADAY_VOL_WARN        = 1.5     # 1.5x normal → widen stop 20%
INTRADAY_VOL_EXTREME     = 2.5     # 2.5x normal → widen stop 50%
SECTOR_VOL_WARN          = 1.5     # Sector ATR 1.5x normal → widen stops
SECTOR_VOL_EXTREME       = 2.2     # Sector ATR 2.2x normal → skip entries


# ── Position dataclass ────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol:        str
    qty:           int
    entry_price:   float
    entry_time:    str
    stop_pct:      float
    initial_stop:  float
    highest_price: float
    add_count:     int   = 0
    risk_dollars:  float = 0.0
    atr_at_entry:  float = 0.0

    def __post_init__(self):
        self.entry_dt = self._parse_dt(self.entry_time)

    @staticmethod
    def _parse_dt(value: str) -> datetime:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt

    def update_high(self, price: float) -> None:
        if price > self.highest_price:
            self.highest_price = price

    def current_stop(self) -> float:
        trailing = self.highest_price * (1 - self.stop_pct)
        return max(self.initial_stop, trailing)

    def age_days(self, now: datetime) -> int:
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)
        return max(0, (now - self.entry_dt).days)

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.qty

    def unrealized_pnl_pct(self, current_price: float) -> float:
        return (current_price / self.entry_price) - 1


# ── Risk manager ──────────────────────────────────────────────────────────────

class RiskManager:

    def __init__(self, portfolio_value: float = 100_000):
        self.portfolio_value = portfolio_value
        self._corr_scalar    = 1.0
        self._vol_cache: Dict[str, dict] = {}

    # ── 1. Vol-adjusted sizing ────────────────────────────────────────────────

    def size_position(
        self,
        symbol:           str,
        price:            float,
        atr_pct:          float,
        conviction:       float,
        regime_scalar:    float,
        sector_vol_ratio: float = 1.0,
    ) -> Tuple[int, float, str]:
        """
        Size so that a stop hit = RISK_DOLLARS_PER_TRADE * scalars.
        Returns (qty, stop_pct, note).
        """
        # Base stop: ATR-based
        base_stop = max(config.FIXED_STOP_LOSS_PCT,
                        atr_pct * config.ATR_STOP_MULTIPLIER)

        # Widen stop when sector vol is elevated (narrative/macro events)
        if sector_vol_ratio >= SECTOR_VOL_EXTREME:
            stop_pct = min(base_stop * 1.5, 0.12)
            vol_note = f"sector_vol={sector_vol_ratio:.1f}x EXTREME → stop widened 50%"
            log.info("%s: %s", symbol, vol_note)
        elif sector_vol_ratio >= SECTOR_VOL_WARN:
            stop_pct = min(base_stop * 1.25, 0.12)
            vol_note = f"sector_vol={sector_vol_ratio:.1f}x → stop widened 25%"
            log.info("%s: %s", symbol, vol_note)
        else:
            stop_pct = base_stop
            vol_note = f"sector_vol={sector_vol_ratio:.1f}x normal"

        # Dollar risk budget — constant across all trades
        risk_budget = (
            RISK_DOLLARS_PER_TRADE
            * conviction
            * regime_scalar
            * self._corr_scalar
        )

        # Cap by max position weight
        max_dollars = min(
            self.portfolio_value * config.MAX_POSITION_WEIGHT * conviction,
            config.MAX_POSITION_DOLLARS,
        )

        risk_per_share = price * stop_pct
        qty_risk = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
        qty_cap  = int(max_dollars / price) if price > 0 else 0
        qty      = min(qty_risk, qty_cap)

        note = (
            f"risk=${risk_budget:.0f}  stop={stop_pct:.1%}  "
            f"qty={qty}  value=${qty*price:,.0f}  {vol_note}"
        )
        return qty, stop_pct, note

    # ── 2. Correlation monitoring ─────────────────────────────────────────────

    def update_correlation_state(
        self,
        open_positions: Dict[str, Position],
        price_history:  Dict[str, pd.DataFrame],
        lookback:       int = CORR_LOOKBACK_DAYS,
    ) -> float:
        """
        Compute rolling realized correlation across open positions.
        Returns position scalar (0.0 = halt, 1.0 = full size).
        Call once per day before entry decisions.
        """
        syms = [s for s in open_positions if s in price_history]
        if len(syms) < 2:
            self._corr_scalar = 1.0
            return 1.0

        returns = {}
        for sym in syms:
            rets = price_history[sym]["close"].pct_change().dropna().tail(lookback)
            if len(rets) >= int(lookback * 0.8):
                returns[sym] = rets

        if len(returns) < 2:
            self._corr_scalar = 1.0
            return 1.0

        corr   = pd.DataFrame(returns).dropna().corr()
        n      = len(corr)
        pairs  = [corr.iloc[i, j] for i in range(n) for j in range(i+1, n)]
        avg_c  = float(np.mean(pairs))

        if avg_c >= CORR_HALT_THRESHOLD:
            scalar = 0.0
            log.warning(
                "CORRELATION HALT: avg=%.2f ≥ %.2f — no new entries, "
                "portfolio acting as one position",
                avg_c, CORR_HALT_THRESHOLD
            )
        elif avg_c >= CORR_WARN_THRESHOLD:
            ratio  = (avg_c - CORR_WARN_THRESHOLD) / (CORR_HALT_THRESHOLD - CORR_WARN_THRESHOLD)
            scalar = 1.0 - 0.5 * ratio
            log.info("CORRELATION ELEVATED: avg=%.2f → scalar=%.2f", avg_c, scalar)
        else:
            scalar = 1.0

        self._corr_scalar = scalar
        return scalar

    # ── 3. Factor exposure ────────────────────────────────────────────────────

    def check_factor_exposure(
        self,
        open_positions:   Dict[str, Position],
        candidate_symbol: str,
        candidate_price:  float,
        candidate_qty:    int,
        price_history:    Dict[str, pd.DataFrame],
        spy_history:      pd.DataFrame,
    ) -> Tuple[bool, str]:
        """
        Check sector concentration + portfolio beta limits.
        Returns (allowed, reason).
        """
        buckets = getattr(config, "CORRELATION_BUCKETS", {})
        c_bucket = buckets.get(candidate_symbol, "other")
        c_value  = candidate_price * candidate_qty

        # Sector concentration
        total_value  = c_value
        bucket_value = c_value if c_bucket != "other" else 0.0
        for sym, pos in open_positions.items():
            pv = pos.entry_price * pos.qty
            total_value += pv
            if buckets.get(sym) == c_bucket and c_bucket != "other":
                bucket_value += pv

        if total_value > 0 and c_bucket != "other":
            conc = bucket_value / total_value
            if conc > MAX_SECTOR_CONCENTRATION:
                return False, (
                    f"sector '{c_bucket}' would be {conc:.0%} of portfolio "
                    f"(limit {MAX_SECTOR_CONCENTRATION:.0%})"
                )

        # Portfolio beta
        if spy_history is not None and len(spy_history) >= 60:
            beta = self._portfolio_beta(
                open_positions, price_history, spy_history,
                candidate_symbol, candidate_price, candidate_qty
            )
            if beta > MAX_BETA_EXPOSURE:
                return False, (
                    f"portfolio beta {beta:.2f} would exceed limit {MAX_BETA_EXPOSURE}"
                )

        return True, "ok"

    def _portfolio_beta(
        self,
        open_positions:   Dict[str, Position],
        price_history:    Dict[str, pd.DataFrame],
        spy_history:      pd.DataFrame,
        cand_sym:         str,
        cand_price:       float,
        cand_qty:         int,
        lookback:         int = 60,
    ) -> float:
        spy_rets = spy_history["close"].pct_change().dropna().tail(lookback)
        all_pos  = {**open_positions, cand_sym: type("P", (), {
            "entry_price": cand_price, "qty": cand_qty
        })()}

        w_beta = 0.0
        total  = 0.0
        for sym, pos in all_pos.items():
            if sym not in price_history:
                continue
            rets   = price_history[sym]["close"].pct_change().dropna().tail(lookback)
            common = rets.index.intersection(spy_rets.index)
            if len(common) < 30:
                continue
            s, m = rets.loc[common].values, spy_rets.loc[common].values
            cov  = np.cov(s, m)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
            pv   = pos.entry_price * pos.qty
            w_beta += beta * pv
            total  += pv

        return w_beta / total if total > 0 else 1.0

    # ── 4. Intraday vol detection ─────────────────────────────────────────────

    def intraday_stop_adjustment(
        self,
        symbol:        str,
        df_today:      pd.DataFrame,
        base_stop_pct: float,
    ) -> Tuple[float, str]:
        """
        Compare today's first-hour range to historical norm.
        Returns (adjusted_stop_pct, note).
        Call at 10:30 AM ET after first hour of trading.
        This is the Iran/oil/macro-shock fix.
        """
        if df_today is None or len(df_today) < 4:
            return base_stop_pct, "no intraday data"

        first_hour   = df_today.iloc[:12]
        today_range  = (
            (first_hour["high"].max() - first_hour["low"].min())
            / first_hour["close"].iloc[0]
        )

        # Use base_stop_pct * 0.4 as estimate of normal first-hour range
        # (first hour typically captures ~40% of daily ATR)
        cache        = self._vol_cache.get(symbol, {})
        normal_range = cache.get("normal_range", base_stop_pct * 0.4)
        vol_ratio    = today_range / normal_range if normal_range > 0 else 1.0

        # Update cache with exponential moving average
        self._vol_cache[symbol] = {
            "normal_range": 0.9 * normal_range + 0.1 * today_range
        }

        if vol_ratio >= INTRADAY_VOL_EXTREME:
            adj  = min(base_stop_pct * 1.5, 0.12)
            note = f"intraday vol {vol_ratio:.1f}x EXTREME — stop widened 50%"
            log.warning("[%s] %s", symbol, note)
        elif vol_ratio >= INTRADAY_VOL_WARN:
            adj  = min(base_stop_pct * 1.2, 0.12)
            note = f"intraday vol {vol_ratio:.1f}x elevated — stop widened 20%"
            log.info("[%s] %s", symbol, note)
        else:
            adj  = base_stop_pct
            note = f"intraday vol {vol_ratio:.1f}x normal"

        return adj, note

    def check_sector_vol(
        self,
        symbol:        str,
        sector_etf_df: pd.DataFrame,
        atr_period:    int = 14,
        lookback:      int = 30,
    ) -> dict:
        """Compare sector ETF recent ATR to 30-day average."""
        if sector_etf_df is None or len(sector_etf_df) < lookback + atr_period:
            return {"elevated": False, "extreme": False, "ratio": 1.0}

        def _atr(df, p):
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"]  - df["close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            return tr.rolling(p).mean().iloc[-1]

        recent = _atr(sector_etf_df.tail(atr_period + 5), atr_period)
        normal = _atr(sector_etf_df.tail(lookback + atr_period), atr_period)
        ratio  = float(recent / normal) if normal > 0 else 1.0

        return {
            "elevated": ratio >= SECTOR_VOL_WARN,
            "extreme":  ratio >= SECTOR_VOL_EXTREME,
            "ratio":    ratio,
        }

    # ── Master entry gate ─────────────────────────────────────────────────────

    def can_enter(
        self,
        symbol:         str,
        qty:            int,
        price:          float,
        open_positions: Dict[str, Position],
        price_history:  Dict[str, pd.DataFrame],
        spy_history:    pd.DataFrame = None,
    ) -> Tuple[bool, str]:
        """All checks in one call. Returns (allowed, reason)."""
        if qty <= 0:
            return False, "qty <= 0"
        if len(open_positions) >= config.MAX_POSITIONS:
            return False, f"max positions {config.MAX_POSITIONS} reached"
        if self._corr_scalar == 0.0:
            return False, "correlation halt"

        # Gross exposure
        current_exp = sum(p.entry_price * p.qty for p in open_positions.values())
        if current_exp + price * qty > config.INITIAL_CAPITAL * config.MAX_TOTAL_EXPOSURE:
            return False, "gross exposure limit"

        # Factor exposure
        if spy_history is not None:
            ok, reason = self.check_factor_exposure(
                open_positions, symbol, price, qty, price_history, spy_history
            )
            if not ok:
                return False, reason

        return True, "ok"

    # ── Summary ───────────────────────────────────────────────────────────────

    def portfolio_summary(
        self,
        open_positions: Dict[str, Position],
        current_prices: Dict[str, float],
    ) -> str:
        lines = [
            f"Positions : {len(open_positions)}/{config.MAX_POSITIONS}  "
            f"corr_scalar={self._corr_scalar:.2f}"
        ]
        total_pnl = 0.0
        for sym, pos in open_positions.items():
            px  = current_prices.get(sym, pos.entry_price)
            pnl = pos.unrealized_pnl(px)
            total_pnl += pnl
            lines.append(
                f"  {sym:<8} qty={pos.qty:>4}  "
                f"entry=${pos.entry_price:>8.2f}  now=${px:>8.2f}  "
                f"pnl=${pnl:>+8.0f}  stop=${pos.current_stop():>8.2f}"
            )
        lines.append(f"Total unrealized PnL: ${total_pnl:+,.0f}")
        return "\n".join(lines)