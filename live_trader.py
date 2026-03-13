"""
live_trader.py — Production Live Trading Entry Point
=====================================================
Bridges the validated backtester strategy with the live execution layer.

Architecture:
  bot.py / broker.py / risk_manager.py  →  execution & position management
  strategy_core.py                       →  ML + rule signals (backtested)
  data_enrichment.py                     →  live enrichment signals (short interest,
                                            insider buys, analyst upgrades)
  signal_enhancer.py                     →  Granger causality, vol regime, Hurst
  circuit_breakers.py                    →  portfolio risk controls
  monitor.py                             →  email/SMS reporting
  state.py                               →  position persistence

Signal stack (live only — cannot be backtested historically):
  1. ML rank score          (backtested, primary driver)
  2. Rule score             (backtested, secondary)
  3. DataEnrichment boost   (live only — short interest, insiders, analyst upgrades)
  4. SignalEnhancer boost   (live only — Granger causality, vol regime filter)
  5. CircuitBreakers        (live only — portfolio DD, SPY crash)

Usage:
  # Paper trading (always start here)
  export ALPACA_API_KEY='your_key'
  export ALPACA_SECRET_KEY='your_secret'
  export ALPACA_BASE_URL='https://paper-api.alpaca.markets'
  python3 live_trader.py

  # To run signal scan without placing orders:
  python3 live_trader.py --scan-only

  # To run one cycle and exit (testing):
  python3 live_trader.py --once
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, date
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
from broker import build_broker
from risk_manager import Position
from circuit_breakers import CircuitBreakers
from monitor import Monitor
from state import StateManager, LivePosition

# Strategy core — same functions used in backtester
from strategy_core import (
    load_ranker_ensemble,
    market_regime,
    select_top_candidates,
    compute_atr_pct,
    trend_bullish,
)
from ml_model import compute_features
from backtester_clean import (
    build_rule_store_fast,
    batch_ml_scores_fast,
    build_fast_snapshots,
    conviction_multiplier,
    should_take_candidate,
    stop_pct_for_symbol,
)

# Live-only enrichment (not backtestable historically)
try:
    from data_enrichment import DataEnrichment
    _ENRICHMENT_AVAILABLE = True
except ImportError:
    _ENRICHMENT_AVAILABLE = False

try:
    from signal_enhancer import SignalEnhancer
    _ENHANCER_AVAILABLE = True
except ImportError:
    _ENHANCER_AVAILABLE = False

ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("live_trader.log"),
    ],
)
log = logging.getLogger("live_trader")


# ── Signal weighting ──────────────────────────────────────────────────────────
# Backtest-validated signal: 100% of score
# Live enrichment boost: additive adjustment to combined_score
# Kept conservative — enrichment data quality needs to be verified live first

ENRICHMENT_WEIGHT = 0.15   # 15% weight on enrichment composite signal
ENHANCER_WEIGHT   = 0.10   # 10% weight on Granger/regime boost

# Enrichment block threshold — if enrichment signal < this, skip entry
ENRICHMENT_BLOCK_THRESHOLD = -0.30  # Only block on strongly negative signals


class LiveTrader:
    """
    Full production live trader.
    Runs one scan cycle per day at market close, executes next morning.
    """

    def __init__(self, scan_only: bool = False):
        self.scan_only = scan_only
        self.broker    = build_broker()
        self.state     = StateManager()
        self.circuits  = CircuitBreakers(initial_capital=config.INITIAL_CAPITAL)
        self.monitor   = Monitor()
        self.rankers   = None
        self._pending_entries: List[dict] = []

        # Live enrichment modules — graceful degradation if unavailable
        self.enrichment = DataEnrichment() if _ENRICHMENT_AVAILABLE else None
        self.enhancer   = SignalEnhancer() if _ENHANCER_AVAILABLE else None

        if not _ENRICHMENT_AVAILABLE:
            log.warning("data_enrichment not available — running without enrichment signals")
        if not _ENHANCER_AVAILABLE:
            log.warning("signal_enhancer not available — running without Granger/regime boost")

    # ── Startup ───────────────────────────────────────────────────────────────

    def startup(self) -> None:
        log.info("=" * 55)
        log.info("LIVE TRADER STARTING")
        log.info("Paper mode: %s", os.environ.get("ALPACA_BASE_URL", "").count("paper") > 0)
        log.info("Universe:   %d symbols", len(config.WATCHLIST))
        log.info("Scan only:  %s", self.scan_only)
        log.info("=" * 55)

        self.state.load()
        self._sync_with_broker()

        log.info("Loading ML ensemble...")
        self.rankers = load_ranker_ensemble()
        log.info("ML ensemble loaded")

    def _sync_with_broker(self) -> None:
        """Broker is always the source of truth for position state."""
        broker_positions = self.broker.list_positions()
        local_symbols    = set(self.state.positions.keys())
        broker_symbols   = set(broker_positions.keys())

        # Ghost positions — in our state but closed at broker
        for sym in local_symbols - broker_symbols:
            log.warning("Removing ghost position (not at broker): %s", sym)
            self.state.remove_position(sym)

        # Orphaned positions — at broker but not in our state
        for sym in broker_symbols - local_symbols:
            log.warning("Orphaned broker position (no local state): %s", sym)
            p = broker_positions[sym]
            pos = LivePosition(
                symbol          = sym,
                qty             = int(p["qty"]),
                entry_price     = float(p["avg_entry_price"]),
                entry_date      = datetime.now(ET).strftime("%Y-%m-%d"),
                stop_price      = float(p["avg_entry_price"]) * (1 - config.FIXED_STOP_LOSS_PCT),
                stop_pct        = config.FIXED_STOP_LOSS_PCT,
                highest_price   = float(p["avg_entry_price"]),
                alpaca_order_id = "",
            )
            self.state.add_position(pos)

        equity = self.broker.account_equity()
        log.info("Account equity: $%,.0f  positions: %s",
                 equity, list(self.state.positions.keys()))

    # ── Market data ───────────────────────────────────────────────────────────

    def _fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch 450 days of OHLCV for all watchlist symbols + SPY."""
        data = {}
        symbols = list(config.WATCHLIST) + [config.BENCHMARK_SYMBOL]
        for sym in symbols:
            try:
                df = self.broker.get_daily_bars(sym, days=450)
                if len(df) >= 200:
                    data[sym] = df
            except Exception as e:
                log.warning("Price fetch failed %s: %s", sym, e)
        log.info("Fetched data for %d / %d symbols", len(data), len(symbols))
        return data

    # ── Signal computation ────────────────────────────────────────────────────

    def _compute_base_signals(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Run the same signal pipeline as the backtester.
        Returns snapshots dict keyed by symbol.
        """
        spy_df      = market_data.get(config.BENCHMARK_SYMBOL)
        symbol_data = {k: v for k, v in market_data.items()
                       if k != config.BENCHMARK_SYMBOL}

        # Current prices
        prices = {sym: float(df["close"].iloc[-1]) for sym, df in symbol_data.items()}

        # Regime check
        regime = market_regime(spy_df)
        log.info("Regime: bear=%s crash=%s vol_halt=%s scalar=%.2f",
                 regime["is_bear"], regime["spy_crash"],
                 regime["vol_halt"], regime["position_scalar"])

        # ML features
        feat_cols = sorted(set(
            list(self.rankers[3]["features"]) +
            list(self.rankers[5]["features"]) +
            list(self.rankers[7]["features"])
        ))

        panel_rows = {}
        for sym, df in symbol_data.items():
            try:
                feat = compute_features(df, symbol=sym)
                feat = feat.replace([np.inf, -np.inf], np.nan)
                last = feat.iloc[-1]
                panel_rows[sym] = {col: float(last.get(col, 0.0)) for col in feat_cols}
            except Exception:
                continue

        if not panel_rows:
            return {}, regime, prices

        X    = np.array([[row.get(c, 0.0) for c in feat_cols]
                          for row in panel_rows.values()], dtype=np.float32)
        syms = list(panel_rows.keys())
        ml_scores = batch_ml_scores_fast(X, syms, self.rankers, feat_cols)

        # Rule scores
        rule_store = build_rule_store_fast(list(symbol_data.keys()), symbol_data)

        # Snapshots — same as backtester
        last_date = max(df.index[-1] for df in symbol_data.values() if len(df) > 0)
        snapshots = build_fast_snapshots(
            date              = last_date,
            available_symbols = list(symbol_data.keys()),
            hist              = symbol_data,
            rule_store        = rule_store,
            ml_scores         = ml_scores,
        )

        return snapshots, regime, prices

    def _apply_live_enrichment(
        self,
        snapshots: Dict,
        symbol_data: Dict[str, pd.DataFrame],
    ) -> Dict:
        """
        Apply live-only enrichment signals on top of base snapshots.
        These signals are NOT in the backtest — they are additive live alpha.

        Conservative design: enrichment can BOOST or BLOCK but cannot create
        entries that base signals would not select. It adjusts scores only.
        """
        if not self.enrichment and not self.enhancer:
            return snapshots

        enriched = dict(snapshots)

        for sym, snap in snapshots.items():
            df = symbol_data.get(sym)
            if df is None or len(df) < 50:
                continue

            boost = 0.0

            # ── DataEnrichment signal ──────────────────────────────────────
            if self.enrichment:
                try:
                    enrich_score = self.enrichment.get_composite_signal(sym)

                    # Hard block: strongly negative enrichment (insider selling,
                    # analyst downgrades, rising short interest all at once)
                    if enrich_score < ENRICHMENT_BLOCK_THRESHOLD:
                        del enriched[sym]
                        log.info("ENRICHMENT BLOCK: %s score=%.3f", sym, enrich_score)
                        continue

                    boost += enrich_score * ENRICHMENT_WEIGHT

                except Exception as e:
                    log.debug("Enrichment failed %s: %s", sym, e)

            # ── SignalEnhancer boost ───────────────────────────────────────
            if self.enhancer and df is not None:
                try:
                    i = len(df) - 1
                    blocked, reason = self.enhancer.should_block(sym, df, i)
                    if blocked:
                        del enriched[sym]
                        log.info("ENHANCER BLOCK: %s reason=%s", sym, reason)
                        continue

                    # Peer data for Granger causality
                    from signal_enhancer import PEER_MAP
                    peers = {p: symbol_data[p] for p in PEER_MAP.get(sym, [])
                             if p in symbol_data}
                    enhancer_score = self.enhancer.get_enhanced_score(sym, df, i, peers)
                    boost += enhancer_score * ENHANCER_WEIGHT

                except Exception as e:
                    log.debug("Enhancer failed %s: %s", sym, e)

            # Apply boost to combined score
            if sym in enriched:
                old_score = enriched[sym].combined_score
                new_score = old_score + boost
                # Rebuild snapshot with boosted score
                import copy
                new_snap = copy.copy(enriched[sym])
                object.__setattr__(new_snap, "combined_score", new_score)
                enriched[sym] = new_snap
                if abs(boost) > 0.02:
                    log.debug("%s: combined %.3f → %.3f (boost=%.3f)",
                              sym, old_score, new_score, boost)

        return enriched

    # ── Exit logic ────────────────────────────────────────────────────────────

    def _execute_exits(self, current_prices: Dict[str, float]) -> None:
        """Check all open positions for exit conditions and execute."""
        for sym in list(self.state.positions.keys()):
            pos = self.state.positions.get(sym)
            if pos is None:
                continue

            px = current_prices.get(sym)
            if px is None:
                continue

            # Update trailing stop
            if px > pos.highest_price:
                pos.highest_price = px
                new_stop = pos.highest_price * (1 - pos.stop_pct)
                if new_stop > pos.stop_price:
                    pos.stop_price = new_stop
                    self.state.update_stop(sym, new_stop)

            exit_reason = None
            if px <= pos.stop_price:
                exit_reason = "stop"
            elif px >= pos.entry_price * (1 + config.TAKE_PROFIT_PCT):
                exit_reason = "take_profit"
            else:
                hold_days = (date.today() - date.fromisoformat(pos.entry_date)).days
                if hold_days >= config.MAX_HOLD_DAYS:
                    exit_reason = "max_hold"

            if exit_reason:
                pnl = (px - pos.entry_price) * pos.qty
                log.info("EXIT %s reason=%s entry=%.2f current=%.2f pnl=%.0f",
                         sym, exit_reason, pos.entry_price, px, pnl)

                if not self.scan_only:
                    self.broker.submit_sell(sym, pos.qty)
                    self.circuits.record_trade(pnl)
                    self.state.remove_position(sym)
                    self.monitor.send_fill_alert(sym, "sell", pos.qty, px, exit_reason)
                else:
                    log.info("[scan-only] Would exit %s", sym)

    # ── Entry logic ───────────────────────────────────────────────────────────

    def _compute_entries(
        self,
        snapshots: Dict,
        symbol_data: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
        regime: Dict,
        circuit_state,
    ) -> List[dict]:
        """Select top candidates and compute order parameters."""
        if circuit_state.halt_all_entries:
            log.warning("CIRCUIT HALTED — no entries: %s", circuit_state.halt_reason)
            return []

        open_symbols    = set(self.state.positions.keys())
        slots_available = config.MAX_POSITIONS - len(open_symbols)
        if slots_available <= 0:
            return []

        # Filter held symbols
        avail = {s: snap for s, snap in snapshots.items() if s not in open_symbols}

        selected = select_top_candidates(
            snapshots         = avail,
            symbol_to_df      = {s: symbol_data[s] for s in avail if s in symbol_data},
            current_positions = {},
            max_names         = slots_available,
            corr_matrix       = pd.DataFrame(),
        )

        entries   = []
        equity    = self.broker.account_equity()
        leader_score = selected[0].combined_score if selected else 0.0

        for idx, snap in enumerate(selected[:slots_available]):
            if not should_take_candidate(snap, idx, leader_score):
                continue

            sym   = snap.symbol
            price = current_prices.get(sym)
            if not price or price <= 0:
                continue

            conviction = conviction_multiplier(snap)
            scalar     = circuit_state.position_scalar * regime["position_scalar"]

            risk_budget    = config.INITIAL_CAPITAL * config.RISK_PER_TRADE * scalar * conviction
            risk_per_share = price * snap.stop_pct
            qty_risk       = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0

            max_dollars = min(
                config.INITIAL_CAPITAL * config.MAX_POSITION_WEIGHT * scalar * conviction,
                config.MAX_POSITION_DOLLARS * conviction,
                equity * 0.92,   # keep 8% cash buffer
            )
            qty = min(qty_risk, int(max_dollars / price))

            if qty <= 0:
                continue

            entries.append({
                "symbol":      sym,
                "qty":         qty,
                "price":       price,
                "stop_pct":    snap.stop_pct,
                "stop_price":  price * (1 - snap.stop_pct),
                "conviction":  conviction,
                "ml_rank_pct": snap.ml_rank_pct,
                "rule_score":  snap.rule_score,
                "combined":    snap.combined_score,
            })

        return entries

    def _execute_entries(self, entries: List[dict]) -> None:
        """Place buy orders for selected entries."""
        for e in entries:
            sym = e["symbol"]
            log.info("ENTRY %s qty=%d price=%.2f stop=%.2f combined=%.3f ml=%.3f",
                     sym, e["qty"], e["price"], e["stop_price"],
                     e["combined"], e["ml_rank_pct"])

            if self.scan_only:
                log.info("[scan-only] Would buy %s", sym)
                continue

            self.broker.submit_buy(sym, e["qty"])
            fill_price = self.broker.latest_price(sym)

            pos = LivePosition(
                symbol          = sym,
                qty             = e["qty"],
                entry_price     = fill_price,
                entry_date      = datetime.now(ET).strftime("%Y-%m-%d"),
                stop_price      = fill_price * (1 - e["stop_pct"]),
                stop_pct        = e["stop_pct"],
                highest_price   = fill_price,
                alpaca_order_id = "",
            )
            self.state.add_position(pos)
            self.monitor.send_fill_alert(sym, "buy", e["qty"], fill_price)

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_one_cycle(self) -> None:
        """Execute one complete scan → exit → entry cycle."""
        log.info("── Daily cycle start ──────────────────────────")

        # 1. Fetch data
        market_data  = self._fetch_market_data()
        spy_df       = market_data.get(config.BENCHMARK_SYMBOL)
        symbol_data  = {k: v for k, v in market_data.items()
                        if k != config.BENCHMARK_SYMBOL}
        current_prices = {sym: float(df["close"].iloc[-1])
                          for sym, df in symbol_data.items()}

        # 2. Circuit breakers
        equity      = self.broker.account_equity()
        spy_return  = (float(spy_df["close"].iloc[-1]) /
                       float(spy_df["close"].iloc[-2]) - 1) if spy_df is not None else 0.0
        self.circuits.update_portfolio_value(equity)

        open_pos_map = {
            sym: {"entry_price": pos.entry_price,
                  "current_open": current_prices.get(sym, pos.entry_price)}
            for sym, pos in self.state.positions.items()
        }
        circuit_state = self.circuits.evaluate(equity, spy_return, open_pos_map)

        # Force-exit gap-downs
        for sym in circuit_state.force_exit_symbols:
            if sym in self.state.positions:
                log.warning("FORCE EXIT gap-down: %s", sym)
                pos = self.state.positions[sym]
                if not self.scan_only:
                    self.broker.submit_sell(sym, pos.qty)
                    self.state.remove_position(sym)

        # 3. Exits
        self._execute_exits(current_prices)

        # 4. Base signals (backtester-equivalent)
        snapshots, regime, _ = self._compute_base_signals(market_data)
        log.info("Base snapshots: %d candidates", len(snapshots))

        # 5. Live enrichment (additive, not in backtest)
        if snapshots:
            snapshots = self._apply_live_enrichment(snapshots, symbol_data)
            log.info("After enrichment: %d candidates", len(snapshots))

        # 6. Entries
        entries = self._compute_entries(
            snapshots, symbol_data, current_prices, regime, circuit_state
        )

        # 7. Execute
        self._execute_entries(entries)

        # 8. Report
        self.state.record_equity(equity)
        self.monitor.send_daily_report(
            portfolio_value = equity,
            positions       = self.state.positions,
            planned_entries = entries,
            circuit_state   = circuit_state,
            spy_return      = spy_return,
        )

        log.info("── Cycle complete  equity=$%,.0f  positions=%d  entries=%d ──",
                 equity, len(self.state.positions), len(entries))

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(self, once: bool = False) -> None:
        self.startup()

        if once:
            self.run_one_cycle()
            return

        log.info("Scheduler running — Ctrl+C to stop")

        while True:
            try:
                now = datetime.now(ET)
                if now.weekday() >= 5:          # weekend
                    time.sleep(3600)
                    continue

                # Run at 3:50 PM ET
                if now.hour == 15 and now.minute == 50:
                    self.run_one_cycle()
                    time.sleep(120)             # avoid double-trigger
                else:
                    time.sleep(30)

            except KeyboardInterrupt:
                log.info("Stopped by user")
                break
            except Exception as e:
                log.exception("Cycle error: %s", e)
                time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Live trading scheduler")
    parser.add_argument("--scan-only", action="store_true",
                        help="Compute signals but do not place orders")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    args = parser.parse_args()

    trader = LiveTrader(scan_only=args.scan_only)
    trader.run(once=args.once)


if __name__ == "__main__":
    main()
