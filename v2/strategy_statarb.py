"""
strategy_statarb.py — Avellaneda-Lee ETF-Residual Statistical Arbitrage
========================================================================
Implementation of the ETF-based statistical arbitrage strategy from:
  Avellaneda & Lee, "Statistical Arbitrage in the U.S. Equities Market" (2008)

Core idea:
  Every stock in the universe is modeled as:
    R_stock(t) = beta_i * R_ETF(t) + dX_i(t)

  where X_i(t) is the idiosyncratic (residual) component, modeled as an
  Ornstein-Uhlenbeck process:
    dX_i = kappa_i * (m_i - X_i) dt + sigma_i dW

  We compute the s-score:
    s_i = (X_i - m_i) / sigma_eq_i
    sigma_eq_i = sigma_i / sqrt(2 * kappa_i)

  And trade when the s-score is far from equilibrium:
    s < -SBO → long stock, short beta_i units of sector ETF  (stock is cheap)
    s > +SSO → short stock, long beta_i units of sector ETF  (stock is dear)

  Exit:
    long  position: exit when s > -SBC (reverted toward zero)
    short position: exit when s < +SSC (reverted toward zero)

  Only trade stocks where kappa > KAPPA_MIN (fast enough mean reversion).

Parameters (from paper, validated 2003-2007):
  SBO = SSO = 1.25  (entry threshold)
  SBC = 0.50        (exit long)
  SSC = 0.75        (exit short)
  KAPPA_MIN = 8.4   (half-life < 30 days: kappa > 252/30)
  BETA_WINDOW = 60  (days for rolling beta regression)
  OU_WINDOW = 60    (days for OU parameter estimation)

Advantages over stock-vs-stock pairs:
  - ETF is a stable, liquid hedge — no asymmetric drift problem
  - 406 candidates every day instead of 15
  - OU model gives principled entry/exit based on mean reversion speed
  - Market-neutral by construction (long stock + short beta * ETF)
  - Negatively correlated with momentum — hedges momentum crashes

Integration with backtester_v2.py:
  See INTEGRATION_NOTES at bottom of file.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtester_clean import Trade, apply_fill_cost
import config


# ── Strategy parameters ───────────────────────────────────────────────────────

SBO          = 1.25   # entry: long stock when s < -SBO
SSO          = 1.25   # entry: short stock when s > +SSO
SBC          = 0.50   # exit long when s > -SBC
SSC          = 0.75   # exit short when s < +SSC
KAPPA_MIN    = 8.4    # min mean reversion speed (half-life < 30 days)
BETA_WINDOW  = 60     # rolling window for beta regression
OU_WINDOW    = 60     # rolling window for OU parameter estimation
CAPITAL_PCT  = 0.025  # % of capital per position (both legs combined)
MAX_POSITIONS = 20    # max simultaneous stat arb positions
COOLDOWN_DAYS = 3     # days before re-entering same stock after exit


# ── ETF cache path helper ─────────────────────────────────────────────────────

def _etf_cache_path(etf: str) -> str:
    """Return path to cached ETF price file."""
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cache_prices"
    )
    # Try _etf.csv first, then _3650d.csv
    for suffix in [f"{etf}_etf.csv", f"{etf}_3650d.csv"]:
        path = os.path.join(base, suffix)
        if os.path.exists(path):
            return path
    return os.path.join(base, f"{etf}_etf.csv")  # will raise if missing


# ── Position state ────────────────────────────────────────────────────────────

@dataclass
class StatArbPosition:
    """
    One open stat arb position: long/short the stock, hedge with sector ETF.

    side = "long":  bought stock, shorted beta * ETF
    side = "short": shorted stock, bought beta * ETF
    """
    symbol:       str
    etf:          str
    side:         str    # "long" or "short"
    stock_qty:    int
    etf_qty:      int    # ETF hedge quantity (always positive — direction from side)
    stock_entry:  float
    etf_entry:    float
    beta:         float  # hedge ratio at entry
    entry_date:   str
    entry_s:      float  # s-score at entry
    etf_margin:   float  # cash held as margin for short ETF leg (if side=long)
    stock_margin: float  # cash held as margin for short stock leg (if side=short)

    def age_days(self, current_date) -> int:
        from datetime import datetime
        try:
            entry = datetime.strptime(self.entry_date, "%Y-%m-%d")
            if hasattr(current_date, "to_pydatetime"):
                current_date = current_date.to_pydatetime()
            return max(0, (current_date - entry).days)
        except Exception:
            return 0


# ── OU parameter estimation ───────────────────────────────────────────────────

def estimate_ou_params(
    stock_prices: pd.Series,
    etf_prices:   pd.Series,
    beta_window:  int = BETA_WINDOW,
    ou_window:    int = OU_WINDOW,
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Estimate OU parameters for a stock vs its sector ETF.

    Steps:
    1. Compute log returns for stock and ETF over beta_window
    2. Regress stock returns on ETF returns → beta
    3. Compute cumulative idiosyncratic returns: X_t = cumsum(r_stock - beta*r_etf)
    4. Fit AR(1) to X_t → OU parameters
    5. Compute s-score

    Returns (s_score, kappa, sigma_eq, beta, X_current) or None if insufficient data.
    """
    common = stock_prices.index.intersection(etf_prices.index)
    if len(common) < beta_window + ou_window + 5:
        return None

    sp = stock_prices[common]
    ep = etf_prices[common]

    # Log returns
    r_stock = np.log(sp / sp.shift(1)).dropna()
    r_etf   = np.log(ep / ep.shift(1)).dropna()

    common_r = r_stock.index.intersection(r_etf.index)
    r_stock  = r_stock[common_r]
    r_etf    = r_etf[common_r]

    if len(r_stock) < beta_window + ou_window:
        return None

    # Rolling beta: regress stock returns on ETF returns over beta_window
    rs = r_stock.iloc[-beta_window:]
    re = r_etf.iloc[-beta_window:]

    X_mat = np.column_stack([re.values, np.ones(len(re))])
    coeffs, _, _, _ = np.linalg.lstsq(X_mat, rs.values, rcond=None)
    beta = float(coeffs[0])

    # Cumulative idiosyncratic returns over OU window
    rs_ou = r_stock.iloc[-(beta_window + ou_window): -beta_window + len(r_stock)]
    re_ou = r_etf.iloc[-(beta_window + ou_window): -beta_window + len(r_etf)]

    # Use all available data for OU window
    n = min(ou_window, len(r_stock))
    rs_ou = r_stock.iloc[-n:]
    re_ou = r_etf.iloc[-n:]

    idio = rs_ou - beta * re_ou
    X    = idio.cumsum().values

    if len(X) < 10:
        return None

    # Fit AR(1): X_t = a + b * X_{t-1} + eps
    X_lag  = X[:-1]
    X_curr = X[1:]
    A      = np.column_stack([X_lag, np.ones(len(X_lag))])
    try:
        fit, _, _, _ = np.linalg.lstsq(A, X_curr, rcond=None)
    except Exception:
        return None

    b, a = float(fit[0]), float(fit[1])

    # OU parameters from AR(1)
    # b = exp(-kappa * dt), dt = 1/252
    if b >= 1.0 or b <= 0.0:
        return None  # not mean-reverting

    kappa = -np.log(b) * 252  # annualized mean reversion speed
    if kappa < KAPPA_MIN:
        return None  # too slow to trade

    m     = float(a / (1 - b))           # long-run mean
    resid = X_curr - (a + b * X_lag)
    sigma = float(np.std(resid) * np.sqrt(252))  # annualized vol of residual

    if sigma < 1e-8:
        return None

    sigma_eq = sigma / np.sqrt(2 * kappa)  # equilibrium standard deviation

    if sigma_eq < 1e-8:
        return None

    X_now  = float(X[-1])
    s_score = (X_now - m) / sigma_eq

    return s_score, kappa, sigma_eq, beta, X_now


def get_s_score_series(
    stock_prices: pd.Series,
    etf_prices:   pd.Series,
    beta_window:  int = BETA_WINDOW,
    ou_window:    int = OU_WINDOW,
) -> pd.Series:
    """
    Compute full rolling s-score series for analysis/debugging.
    """
    common = stock_prices.index.intersection(etf_prices.index)
    sp = stock_prices[common]
    ep = etf_prices[common]

    r_stock = np.log(sp / sp.shift(1)).dropna()
    r_etf   = np.log(ep / ep.shift(1)).dropna()
    common_r = r_stock.index.intersection(r_etf.index)
    r_stock  = r_stock[common_r]
    r_etf    = r_etf[common_r]

    scores = {}
    total  = beta_window + ou_window

    for i in range(total, len(r_stock)):
        rs_beta = r_stock.iloc[i - beta_window: i]
        re_beta = r_etf.iloc[i - beta_window: i]
        X_mat   = np.column_stack([re_beta.values, np.ones(len(re_beta))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_mat, rs_beta.values, rcond=None)
            beta = float(coeffs[0])
        except Exception:
            continue

        rs_ou  = r_stock.iloc[i - ou_window: i]
        re_ou  = r_etf.iloc[i - ou_window: i]
        idio   = rs_ou - beta * re_ou
        X      = idio.cumsum().values

        X_lag  = X[:-1]
        X_curr = X[1:]
        A      = np.column_stack([X_lag, np.ones(len(X_lag))])
        try:
            fit, _, _, _ = np.linalg.lstsq(A, X_curr, rcond=None)
        except Exception:
            continue

        b, a  = float(fit[0]), float(fit[1])
        if b >= 1.0 or b <= 0.0:
            continue

        kappa = -np.log(b) * 252
        if kappa < KAPPA_MIN:
            continue

        m        = float(a / (1 - b))
        resid    = X_curr - (a + b * X_lag)
        sigma    = float(np.std(resid) * np.sqrt(252))
        sigma_eq = sigma / np.sqrt(2 * kappa)

        if sigma_eq < 1e-8:
            continue

        X_now   = float(X[-1])
        s       = (X_now - m) / sigma_eq
        date    = r_stock.index[i]
        scores[date] = s

    return pd.Series(scores)


# ── Main engine ───────────────────────────────────────────────────────────────

class StatArbEngine:
    """
    Avellaneda-Lee ETF-residual stat arb engine.
    Call update() once per trading day from the main backtest loop.
    """

    def __init__(self):
        self.positions: Dict[str, StatArbPosition] = {}
        self.cooldowns: Dict[str, pd.Timestamp]    = {}
        self._etf_prices: Dict[str, pd.DataFrame]  = {}  # lazy cache

        # Build reverse map: symbol → ETF
        self.symbol_to_etf: Dict[str, str] = {}
        for etf, syms in config.SECTOR_ETFS.items():
            for s in syms:
                self.symbol_to_etf[s] = etf

    def load_etf_prices(self) -> Dict[str, pd.DataFrame]:
        """Load all sector ETF price files. Called once at backtest start."""
        if self._etf_prices:
            return self._etf_prices
        for etf in config.SECTOR_ETFS:
            try:
                path = _etf_cache_path(etf)
                df   = pd.read_csv(path, index_col=0, parse_dates=True)
                df.columns = [c.lower() for c in df.columns]
                if "close" not in df.columns and "adj close" in df.columns:
                    df = df.rename(columns={"adj close": "close"})
                self._etf_prices[etf] = df
                print(f"  [statarb] loaded {etf}: {len(df)} bars")
            except Exception as e:
                print(f"  [statarb] WARNING: could not load {etf}: {e}")
        return self._etf_prices

    def update(
        self,
        date:          pd.Timestamp,
        next_date:     pd.Timestamp,
        prices_by_sym: Dict[str, pd.DataFrame],
        cash:          float,
        capital:       float,
        trades:        List[Trade],
        regime:        str = "CHOPPY",
    ) -> float:
        """Process exits then entries. Returns updated cash.
        
        Only enters new positions in CHOPPY or BEAR regimes.
        Mean reversion is suppressed in TRENDING_BULL — momentum
        runs over spread reversion, causing maxhold failures.
        Data validation: stat arb positive only in low-TREND years.
        """
        cash = self._handle_exits(date, prices_by_sym, cash, trades)
        if regime != "TRENDING_BULL":
            cash = self._handle_entries(date, next_date, prices_by_sym, cash, capital, trades)
        return cash

    def portfolio_value(
        self,
        prices_by_sym: Dict[str, pd.DataFrame],
        date:          pd.Timestamp,
    ) -> float:
        """
        Full mark-to-market value of all open positions.

        Must include full asset value, not just PnL deltas, because cash
        was reduced by the full purchase cost at entry.

        long side:  cash was reduced by stock_cost + etf_margin
                    → add back: sp * stock_qty + etf_margin + etf_short_pnl
        short side: cash was reduced by etf_cost + stock_margin
                    → add back: ep * etf_qty + stock_margin + stock_short_pnl
        """
        val = 0.0
        for pos in self.positions.values():
            try:
                sp = float(prices_by_sym[pos.symbol]["close"].loc[:date].iloc[-1])
                ep = float(self._etf_prices[pos.etf]["close"].loc[:date].iloc[-1])

                if pos.side == "long":
                    # Long stock (full current value) + margin held + ETF short PnL
                    val += sp * pos.stock_qty
                    val += pos.etf_margin
                    val += (pos.etf_entry - ep) * pos.etf_qty
                else:
                    # Short stock PnL + margin held + long ETF (full current value)
                    val += (pos.stock_entry - sp) * pos.stock_qty
                    val += pos.stock_margin
                    val += ep * pos.etf_qty
            except Exception:
                pass
        return val

    # ── Exits ──────────────────────────────────────────────────────────────────

    def _handle_exits(self, date, prices_by_sym, cash, trades):
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]

            try:
                sp_series  = prices_by_sym[pos.symbol]["close"].loc[:date]
                etf_series = self._etf_prices[pos.etf]["close"].loc[:date]
            except KeyError:
                continue

            result = estimate_ou_params(sp_series, etf_series)
            if result is None:
                continue

            s, kappa, sigma_eq, beta, X_now = result

            exit_reason = None

            if pos.side == "long":
                # Entered because s was LOW (stock cheap). Exit when s reverts up.
                if s >= -SBC:
                    exit_reason = "statarb_exit"
                elif s <= -(SBO + 2.0):
                    exit_reason = "statarb_stop"  # spread kept diverging
            else:  # short
                # Entered because s was HIGH (stock dear). Exit when s reverts down.
                if s <= SSC:
                    exit_reason = "statarb_exit"
                elif s >= (SSO + 2.0):
                    exit_reason = "statarb_stop"

            if pos.age_days(date) >= 20:
                exit_reason = "statarb_maxhold"

            if exit_reason:
                cash = self._close(pos, sym, date, prices_by_sym, exit_reason, cash, trades)
                self.cooldowns[sym] = date

        return cash

    def _close(self, pos, sym, date, prices_by_sym, exit_reason, cash, trades):
        try:
            sp = float(prices_by_sym[pos.symbol]["close"].loc[:date].iloc[-1])
            ep = float(self._etf_prices[pos.etf]["close"].loc[:date].iloc[-1])
        except Exception:
            return cash

        if pos.side == "long":
            # Close: sell stock, buy back ETF
            fill_s, comm_s = apply_fill_cost(sp, pos.stock_qty, "sell")
            pnl_s = (fill_s - pos.stock_entry) * pos.stock_qty - comm_s
            cash += fill_s * pos.stock_qty - comm_s

            fill_e, comm_e = apply_fill_cost(ep, pos.etf_qty, "buy")
            pnl_e = (pos.etf_entry - fill_e) * pos.etf_qty - comm_e
            cash += pnl_e
            cash += pos.etf_margin  # return ETF short margin
        else:
            # Close: buy back stock, sell ETF
            fill_s, comm_s = apply_fill_cost(sp, pos.stock_qty, "buy")
            pnl_s = (pos.stock_entry - fill_s) * pos.stock_qty - comm_s
            cash += pnl_s
            cash += pos.stock_margin  # return stock short margin

            fill_e, comm_e = apply_fill_cost(ep, pos.etf_qty, "sell")
            pnl_e = (fill_e - pos.etf_entry) * pos.etf_qty - comm_e
            cash += fill_e * pos.etf_qty - comm_e

        total_pnl = pnl_s + pnl_e

        trades.append(Trade(
            symbol        = f"{pos.symbol}/[{pos.etf}]",
            entry_date    = pos.entry_date,
            exit_date     = str(date.date()),
            entry_price   = pos.stock_entry,
            exit_price    = sp,
            qty           = pos.stock_qty,
            pnl           = total_pnl,
            reason        = exit_reason,
            ml_rank_pct   = 0.0,
            rule_score    = 0.0,
            combined_score= abs(pos.entry_s),
            side          = f"statarb_{pos.side}",
        ))

        del self.positions[sym]
        return cash

    # ── Entries ────────────────────────────────────────────────────────────────

    def _handle_entries(self, date, next_date, prices_by_sym, cash, capital, trades):
        if len(self.positions) >= MAX_POSITIONS:
            return cash

        candidates = []

        for sym, etf in self.symbol_to_etf.items():
            if sym in self.positions:
                continue

            last_exit = self.cooldowns.get(sym)
            if last_exit and (date - last_exit).days < COOLDOWN_DAYS:
                continue

            if etf not in self._etf_prices:
                continue

            try:
                sp_series  = prices_by_sym[sym]["close"].loc[:date]
                etf_series = self._etf_prices[etf]["close"].loc[:date]
            except KeyError:
                continue

            result = estimate_ou_params(sp_series, etf_series)
            if result is None:
                continue

            s, kappa, sigma_eq, beta, X_now = result

            # Only enter on strong signals
            if abs(s) < SBO:
                continue

            candidates.append((sym, etf, s, kappa, beta))

        # Sort by |s| descending — strongest signals first
        candidates.sort(key=lambda x: abs(x[2]), reverse=True)

        for sym, etf, s, kappa, beta in candidates:
            if len(self.positions) >= MAX_POSITIONS:
                break

            # Get next-day open prices
            try:
                future_s = prices_by_sym[sym].loc[prices_by_sym[sym].index > date]
                future_e = self._etf_prices[etf].loc[self._etf_prices[etf].index > date]
            except KeyError:
                continue

            if len(future_s) == 0 or len(future_e) == 0:
                continue

            stock_px = float(future_s.iloc[0]["open"])
            etf_px   = float(future_e.iloc[0]["open"])

            if stock_px <= 0 or etf_px <= 0:
                continue

            # Sizing: capital_pct split between stock and ETF legs
            leg_dollars = capital * CAPITAL_PCT
            stock_qty   = int(leg_dollars / stock_px)
            etf_qty     = max(1, int(stock_qty * abs(beta) * stock_px / etf_px))

            if stock_qty <= 0:
                continue

            if s < -SBO:
                # Long stock, short ETF
                side        = "long"
                stock_cost  = stock_qty * stock_px
                etf_margin  = etf_qty  * etf_px * 0.50
                total_cost  = stock_cost + etf_margin

                if total_cost > cash * 0.95:
                    continue

                fill_s, comm_s = apply_fill_cost(stock_px, stock_qty, "buy")
                fill_e, comm_e = apply_fill_cost(etf_px,   etf_qty,   "sell")

                cash -= fill_s * stock_qty + comm_s
                cash -= etf_margin

                self.positions[sym] = StatArbPosition(
                    symbol=sym, etf=etf, side=side,
                    stock_qty=stock_qty, etf_qty=etf_qty,
                    stock_entry=fill_s, etf_entry=fill_e,
                    beta=beta, entry_date=str(next_date.date()),
                    entry_s=s, etf_margin=etf_margin, stock_margin=0.0,
                )

            else:  # s > +SSO
                # Short stock, long ETF
                side         = "short"
                stock_margin = stock_qty * stock_px * 0.50
                etf_cost     = etf_qty  * etf_px
                total_cost   = stock_margin + etf_cost

                if total_cost > cash * 0.95:
                    continue

                fill_s, comm_s = apply_fill_cost(stock_px, stock_qty, "sell")
                fill_e, comm_e = apply_fill_cost(etf_px,   etf_qty,   "buy")

                cash -= stock_margin
                cash -= fill_e * etf_qty + comm_e

                self.positions[sym] = StatArbPosition(
                    symbol=sym, etf=etf, side=side,
                    stock_qty=stock_qty, etf_qty=etf_qty,
                    stock_entry=fill_s, etf_entry=fill_e,
                    beta=beta, entry_date=str(next_date.date()),
                    entry_s=s, etf_margin=0.0, stock_margin=stock_margin,
                )

        return cash


# ── Standalone backtest ───────────────────────────────────────────────────────

def backtest_statarb_standalone(days: int = 3650):
    """
    Run stat arb engine in isolation to validate before integrating.
    Only integrate if standalone Sharpe > 1.0.
    """
    from backtester_clean import fetch_history, calc_stats

    print("=" * 60)
    print("  STAT ARB (AVELLANEDA-LEE) STANDALONE BACKTEST")
    print("=" * 60)
    print(f"  Strategy: ETF-residual OU model")
    print(f"  Universe: {len(config.WATCHLIST)} stocks × {len(config.SECTOR_ETFS)} ETFs")
    print(f"  Entry:  |s| > {SBO}  |  Exit: long s>{-SBC:.1f}, short s<{SSC}")
    print(f"  Kappa min: {KAPPA_MIN} (half-life < 30d)")
    print()

    engine = StatArbEngine()
    etf_prices = engine.load_etf_prices()
    print()

    # Load stock prices
    hist: Dict[str, pd.DataFrame] = {}
    print(f"  Loading {len(config.WATCHLIST)} stock price files...")
    for s in config.WATCHLIST:
        try:
            hist[s] = fetch_history(s, days)
        except Exception:
            pass
    print(f"  Loaded {len(hist)} symbols\n")

    # Build common date index from SPY
    spy = fetch_history("SPY", days)
    all_dates = spy.index.sort_values()

    capital = float(config.INITIAL_CAPITAL)
    cash    = capital
    trades: List[Trade] = []
    equity  = []

    warmup = BETA_WINDOW + OU_WINDOW + 20

    print(f"  Running {len(all_dates) - warmup} trading days...\n")

    # Load real regime data for regime-conditional gating
    from regime_classifier import RegimeClassifier, compute_signals, load_macro_data
    macro_cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache_prices")
    spy_macro, hyg_macro, vix_macro = load_macro_data(cache_dir=macro_cache)
    clf = RegimeClassifier()
    regime_map = {}
    for d in all_dates:
        signals = compute_signals(spy_macro, hyg_macro, vix_macro, as_of_date=d)
        if signals:
            regime_map[d] = clf.update(d, signals)
    from collections import Counter
    print(f"  Regime distribution: {dict(Counter(regime_map.values()))}")
    print()

    for i in range(warmup, len(all_dates) - 1):
        date      = all_dates[i]
        next_date = all_dates[i + 1]
        regime    = regime_map.get(date, "CHOPPY")

        cash = engine.update(
            date=date, next_date=next_date,
            prices_by_sym=hist,
            cash=cash, capital=capital,
            trades=trades,
            regime=regime,
        )

        pair_val = engine.portfolio_value(hist, date)
        equity.append((date, cash + pair_val))

        if i % 250 == 0:
            n = len(engine.positions)
            print(f"  Step {i-warmup}/{len(all_dates)-warmup}"
                  f"  open={n}  trades={len(trades)}"
                  f"  equity=${cash+pair_val:,.0f}")

    equity_curve = pd.Series(
        data  = [v for _, v in equity],
        index = pd.to_datetime([d for d, _ in equity]),
        name  = "statarb_equity",
    )

    stats = calc_stats(equity_curve, trades)

    print("\n" + "=" * 60)
    print("  STAT ARB STANDALONE RESULTS")
    print("=" * 60)
    print(f"  CAGR         : {stats['cagr']:>10.2%}")
    print(f"  Sharpe       : {stats['sharpe']:>10.2f}")
    print(f"  Max Drawdown : {stats['max_drawdown']:>10.2%}")
    print(f"  Trades       : {stats['trades']:>10}")
    print(f"  Win Rate     : {stats['win_rate']:>10.2%}")
    print()

    if trades:
        df_t = pd.DataFrame([t.__dict__ for t in trades])

        # Exit reason breakdown
        print("  Exit breakdown:")
        for reason, grp in df_t.groupby("reason"):
            wr = (grp["pnl"] > 0).mean()
            print(f"    {reason:<22} {len(grp):4d}  WR={wr:.0%}  avg=${grp['pnl'].mean():.0f}")
        print()

        # ETF breakdown
        print("  By sector ETF:")
        df_t["etf"] = df_t["symbol"].str.extract(r'\[(\w+)\]')
        for etf, grp in df_t.groupby("etf"):
            wr = (grp["pnl"] > 0).mean()
            print(f"    {etf:<8} {len(grp):4d} trades  WR={wr:.0%}  avg=${grp['pnl'].mean():.0f}")

    if stats.get("annual"):
        print()
        for yr in sorted(stats["annual"]):
            a = stats["annual"][yr]
            flag = "✓" if a["cagr"] > 0 else "✗"
            print(f"  {yr}  {a['cagr']:>7.1%}  Sharpe {a['sharpe']:>5.2f}  {flag}")

    print("=" * 60)
    return equity_curve, trades, stats


# ── Current s-scores ──────────────────────────────────────────────────────────

def print_current_signals(top_n: int = 20):
    """Print current s-scores for all stocks. Shows active signals."""
    from backtester_clean import fetch_history

    engine = StatArbEngine()
    engine.load_etf_prices()

    results = []
    for sym, etf in engine.symbol_to_etf.items():
        if etf not in engine._etf_prices:
            continue
        try:
            sp = fetch_history(sym, 365)["close"]
            ep = engine._etf_prices[etf]["close"]
            result = estimate_ou_params(sp, ep)
            if result is None:
                continue
            s, kappa, sigma_eq, beta, _ = result
            hl = np.log(2) / kappa * 252 if kappa > 0 else 999
            results.append((sym, etf, s, kappa, hl))
        except Exception:
            pass

    results.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n{'Symbol':<8} {'ETF':<6} {'s-score':>8} {'kappa':>8} {'half-life':>10} {'Signal'}")
    print("-" * 55)
    for sym, etf, s, kappa, hl in results[:top_n]:
        sig = ""
        if s < -SBO:
            sig = "← LONG STOCK"
        elif s > SSO:
            sig = "→ SHORT STOCK"
        print(f"  {sym:<8} {etf:<6} {s:>8.3f} {kappa:>8.1f} {hl:>9.1f}d  {sig}")
    print()


# ── Integration instructions ──────────────────────────────────────────────────

INTEGRATION_NOTES = """
HOW TO INTEGRATE INTO backtester_v2.py
=======================================

1. Import at top of backtester_v2.py:
   from strategy_statarb import StatArbEngine

2. After loading macro data, load ETF prices:
   statarb_engine = StatArbEngine()
   statarb_engine.load_etf_prices()

3. Inside main loop, AFTER long exits, BEFORE long entries:
   cash = statarb_engine.update(
       date=date, next_date=next_date,
       prices_by_sym=prices_by_symbol,
       cash=cash, capital=config.INITIAL_CAPITAL,
       trades=trades,
   )

4. Update portfolio valuation:
   statarb_val = statarb_engine.portfolio_value(prices_by_symbol, date)
   equity.append((date, port_val + statarb_val))

IMPORTANT: Only integrate if standalone Sharpe > 1.0.
Run: python3.11 strategy_statarb.py --standalone
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", action="store_true",
                        help="Run standalone backtest")
    parser.add_argument("--signals",    action="store_true",
                        help="Print current s-scores")
    args = parser.parse_args()

    if args.standalone:
        backtest_statarb_standalone()
    elif args.signals:
        print_current_signals()
    else:
        print(INTEGRATION_NOTES)
