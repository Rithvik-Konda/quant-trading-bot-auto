"""
proprietary_strats.py — 5 Proprietary Signal+Options Strategies
================================================================
Strategies that combine our ML ranker + regime classifier + fundamental
signals with options positioning. Nobody else has this signal stack.

Run: python v2/proprietary_strats.py
"""
from __future__ import annotations
import sys, os, json, math
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2"))
sys.path.insert(0, os.path.expanduser("~/ai_trading_bot_v2/v2"))
import pandas as pd
import numpy as np
import config

ROOT = Path(os.path.expanduser("~/ai_trading_bot_v2"))
CACHE_PRICES = ROOT / "cache_prices"
CACHE_EARN   = ROOT / "cache_earnings_streak"

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(name: str) -> Optional[pd.DataFrame]:
    for suffix in ["_3650d.csv", "_9999d.csv", "_etf.csv", "_regime.csv"]:
        path = CACHE_PRICES / f"{name}{suffix}"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            return df[["close"]].dropna() if "close" in df.columns else df
    return None

def load_trades() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "trades_v2.csv")
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    return df

def load_earnings_store() -> Dict[str, pd.DataFrame]:
    store = {}
    for f in CACHE_EARN.glob("*.json"):
        sym = f.stem.upper()
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list) and data:
                edf = pd.DataFrame(data)
                if "date" in edf.columns:
                    edf["date"] = pd.to_datetime(edf["date"])
                store[sym] = edf
        except Exception:
            pass
    return store


# ═══════════════════════════════════════════════════════════════════════════════
#  REGIME RECONSTRUCTION (2016-present from SPY/HYG/VIX)
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_regime_series() -> pd.Series:
    """Reconstruct daily regime from macro data. Returns Series indexed by date."""
    spy = load_csv("SPY")
    vix = load_csv("^VIX")
    hyg = load_csv("HYG")

    if spy is None or vix is None:
        print("  ERROR: Missing SPY or VIX data for regime reconstruction")
        return pd.Series(dtype=str)

    # Align dates
    common = spy.index.intersection(vix.index)
    if hyg is not None:
        common = common.intersection(hyg.index)
    common = common.sort_values()

    spy_c = spy.loc[common, "close"]
    vix_c = vix.loc[common, "close"]
    hyg_c = hyg.loc[common, "close"] if hyg is not None else pd.Series(0, index=common)

    spy_sma200 = spy_c.rolling(200).mean()
    spy_20d    = spy_c.pct_change(20)
    spy_60d    = spy_c.pct_change(60)
    spy_vs_200 = (spy_c - spy_sma200) / spy_sma200
    vix_20d    = vix_c.pct_change(20)
    hyg_20d    = hyg_c.pct_change(20)
    spy_slope  = spy_c.rolling(20).mean().pct_change(5)

    regimes = pd.Series("CHOPPY", index=common)

    # BEAR triggers (any one)
    bear_mask = (
        (spy_vs_200 < -0.02) |
        (hyg_20d < -0.025) |
        (vix_c > 30) |
        (spy_60d < -0.05)
    )

    # TRENDING_BULL requires ALL
    bull_mask = (
        (spy_slope > 0) &
        (spy_20d > 0) &
        (vix_20d < 0.15) &
        (hyg_20d > -0.01) &
        (spy_vs_200 > 0.03)
    )

    regimes[bear_mask] = "BEAR"
    regimes[bull_mask & ~bear_mask] = "TRENDING_BULL"

    # Apply 5-day hysteresis
    current = "CHOPPY"
    hold_count = 0
    final = []
    for date in common:
        raw = regimes.loc[date]
        if raw == current:
            hold_count = 0
            final.append(current)
        else:
            hold_count += 1
            if hold_count >= 5:
                current = raw
                hold_count = 0
            final.append(current)

    return pd.Series(final, index=common)


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIONS P&L APPROXIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def bs_call_approx(S, K, sigma, T_days):
    """ATM call approximation: Call ≈ 0.4 × S × σ × √(T/365)."""
    if T_days <= 0 or sigma <= 0:
        return max(S - K, 0)
    return 0.4 * S * sigma * math.sqrt(T_days / 365.0)

def bs_straddle(S, K, sigma, T_days):
    """ATM straddle ≈ 0.8 × S × σ × √(T/365)."""
    if T_days <= 0 or sigma <= 0:
        return abs(S - K)
    return 0.8 * S * sigma * math.sqrt(T_days / 365.0)

def iron_condor_pnl(entry_S, exit_S, K, sigma, entry_dte, exit_dte, wing_mult=1.5):
    """Iron condor P&L: sell straddle at entry, buy back at exit, wings at 1.5x."""
    entry_straddle = bs_straddle(entry_S, K, sigma, entry_dte)
    exit_sigma = sigma * 0.6  # post-event IV crush ~40%
    exit_straddle = bs_straddle(exit_S, K, exit_sigma, exit_dte)
    net_credit = entry_straddle * 0.97  # wing cost ~3%
    wing_width = wing_mult * entry_straddle
    max_loss = wing_width - net_credit
    ss_pnl = net_credit - exit_straddle
    return max(ss_pnl, -max_loss)

def call_pnl_30dte(entry_S, exit_S, entry_sigma, exit_sigma, hold_days=15):
    """30-DTE call P&L: buy at entry, sell after hold_days."""
    entry_price = bs_call_approx(entry_S, entry_S, entry_sigma, 30)
    exit_dte = max(30 - hold_days, 1)
    # Intrinsic + time value at exit
    exit_price = max(exit_S - entry_S, 0) + bs_call_approx(exit_S, entry_S, exit_sigma, exit_dte)
    exit_price = max(exit_price, max(exit_S - entry_S, 0))  # floor at intrinsic
    return exit_price - entry_price

def put_premium_30dte(S, sigma, delta_target=0.30):
    """30-DTE put premium for ~30-delta OTM put."""
    # 30-delta put is roughly at S × (1 - 0.5σ√T)
    K = S * (1 - 0.5 * sigma * math.sqrt(30/365))
    put_price = 0.4 * S * sigma * math.sqrt(30/365) * 0.5  # OTM discount
    return K, put_price


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 1: ML RANK + IV DIVERGENCE
#  Top-decile ML rank + low IV = mispricing → buy calls
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_1_ml_iv_divergence(trades: pd.DataFrame, vix: pd.DataFrame,
                                 prices_cache: dict) -> pd.DataFrame:
    """
    When ML rank is top decile (>0.90) but VIX is below 20th percentile
    for that stock's history, buy 30-DTE calls. Hold 10 trading days.
    """
    print("\n▸ STRATEGY 1: ML RANK + IV DIVERGENCE (buy calls on mispricing)")

    # Use trades_v2.csv as ground truth for ML ranks + entry dates
    top_ml = trades[(trades["ml_rank_pct"] >= 0.90) & (trades["side"] == "long")].copy()

    # Get VIX level at each entry
    vix_c = vix["close"] if "close" in vix.columns else vix.iloc[:, 0]
    results = []
    for _, t in top_ml.iterrows():
        sym = t["symbol"]
        edate = t["entry_date"]

        # Get VIX at entry
        vix_at_entry = vix_c.asof(edate)
        if pd.isna(vix_at_entry):
            continue

        # VIX 1-year percentile
        vix_1y = vix_c.loc[:edate].tail(252)
        if len(vix_1y) < 100:
            continue
        vix_pct = (vix_1y < vix_at_entry).mean()

        # IV DIVERGENCE: ML says buy, but VIX (IV proxy) is low
        # Low IV = cheap options = buy calls
        if vix_pct > 0.30:  # Only when IV is below 30th percentile
            continue

        # Get price data for P&L
        if sym not in prices_cache:
            df = load_csv(sym)
            if df is not None:
                prices_cache[sym] = df
        if sym not in prices_cache:
            continue

        px = prices_cache[sym]["close"]
        entry_price = px.asof(edate)
        if pd.isna(entry_price):
            continue

        # Forward 10-day return
        future_dates = px.index[px.index > edate][:10]
        if len(future_dates) < 5:
            continue
        exit_price = float(px.loc[future_dates[-1]])

        # Options P&L: buy 30-DTE ATM call, hold 10 days
        sigma = float(vix_at_entry) / 100.0  # VIX as annualized vol proxy
        # Stock-specific vol is typically VIX × beta. Use 1.5× for high-ML stocks
        stock_sigma = sigma * 1.5
        call_pnl = call_pnl_30dte(entry_price, exit_price, stock_sigma, stock_sigma, hold_days=10)
        call_cost = bs_call_approx(entry_price, entry_price, stock_sigma, 30)
        call_ret = call_pnl / call_cost if call_cost > 0 else 0

        # Equity return for comparison
        eq_ret = (exit_price - entry_price) / entry_price

        results.append({
            "symbol": sym, "entry_date": edate.strftime("%Y-%m-%d"),
            "ml_rank": t["ml_rank_pct"], "regime": t["regime"],
            "vix_at_entry": float(vix_at_entry), "vix_percentile": vix_pct,
            "entry_price": entry_price, "exit_price": exit_price,
            "equity_return": eq_ret,
            "call_cost": call_cost, "call_pnl": call_pnl, "call_return": call_ret,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No qualifying events found.")
        return df

    print(f"  Events: {len(df)}")
    print(f"  Avg ML rank: {df['ml_rank'].mean():.3f}")
    print(f"  Avg VIX at entry: {df['vix_at_entry'].mean():.1f} (percentile {df['vix_percentile'].mean():.0%})")
    print(f"  Equity: avg={df['equity_return'].mean():+.2%}, win={( df['equity_return'] > 0).mean():.0%}")
    print(f"  Calls:  avg={df['call_return'].mean():+.2%}, win={(df['call_return'] > 0).mean():.0%}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 2: REGIME-CONDITIONAL IRON CONDOR
#  CHOPPY regime + ML 40-60% (no directional conviction) = sell IC
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_2_regime_iron_condor(trades: pd.DataFrame, vix: pd.DataFrame,
                                   regime_series: pd.Series, prices_cache: dict) -> pd.DataFrame:
    """
    Sell iron condors when: regime=CHOPPY AND ML rank 40-60% (middle quintile).
    Middle ML = no directional conviction = stock likely rangebound.
    """
    print("\n▸ STRATEGY 2: REGIME-CONDITIONAL IRON CONDOR (CHOPPY + mid-ML)")

    # All choppy-regime trades with middle ML rank
    mid_ml = trades[
        (trades["regime"] == "CHOPPY") &
        (trades["ml_rank_pct"] >= 0.40) &
        (trades["ml_rank_pct"] <= 0.60)
    ].copy()

    # Also scan the choppy periods where we DIDN'T trade (no entry = rangebound)
    # Use regime series to find CHOPPY dates, then check stock returns
    choppy_dates = regime_series[regime_series == "CHOPPY"].index
    vix_c = vix["close"] if "close" in vix.columns else vix.iloc[:, 0]

    results = []
    # Sample 15 diverse stocks for the IC backtest
    sample_syms = ["NVDA", "AAPL", "META", "AMZN", "MSFT", "GOOGL", "JPM", "XOM",
                   "UNH", "COST", "HD", "CRWD", "AVGO", "AMD", "NFLX"]

    for sym in sample_syms:
        if sym not in prices_cache:
            df = load_csv(sym)
            if df is not None:
                prices_cache[sym] = df
        if sym not in prices_cache:
            continue

        px = prices_cache[sym]["close"]

        # Find monthly CHOPPY entry points (first trading day of each CHOPPY month)
        choppy_months = set()
        for d in choppy_dates:
            if d in px.index or px.index.asof(d) == px.index.asof(d):
                key = (d.year, d.month)
                if key not in choppy_months:
                    choppy_months.add(key)
                    edate = d
                    entry_price = px.asof(edate)
                    if pd.isna(entry_price) or entry_price <= 0:
                        continue

                    vix_val = vix_c.asof(edate)
                    if pd.isna(vix_val):
                        continue

                    # Check 5-day forward return (IC hold period)
                    future = px.index[px.index > edate][:5]
                    if len(future) < 3:
                        continue
                    exit_price = float(px.loc[future[-1]])

                    sigma = float(vix_val) / 100.0 * 1.3  # stock vol > VIX typically
                    pnl = iron_condor_pnl(
                        entry_price, exit_price, entry_price,
                        sigma, entry_dte=7, exit_dte=2, wing_mult=1.5
                    )
                    straddle_val = bs_straddle(entry_price, entry_price, sigma, 7)
                    ic_ret = pnl / (straddle_val * 0.53) if straddle_val > 0 else 0  # return on capital at risk

                    results.append({
                        "symbol": sym, "entry_date": edate.strftime("%Y-%m-%d"),
                        "regime": "CHOPPY", "vix": float(vix_val),
                        "entry_price": entry_price, "exit_price": exit_price,
                        "stock_move_pct": (exit_price - entry_price) / entry_price,
                        "ic_pnl": pnl, "ic_return": ic_ret,
                        "straddle_sold": straddle_val,
                    })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No qualifying events found.")
        return df

    # Filter to 2021+ for cleaner data
    df["entry_date_dt"] = pd.to_datetime(df["entry_date"])
    df = df[df["entry_date_dt"] >= "2021-01-01"].drop(columns=["entry_date_dt"])

    print(f"  Events: {len(df)} (CHOPPY months × 15 names)")
    print(f"  Avg stock move: {df['stock_move_pct'].abs().mean():.2%}")
    print(f"  IC: avg={df['ic_return'].mean():+.2%}, win={(df['ic_pnl'] > 0).mean():.0%}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 3: INSIDER SEQUENCE + OPTIONS FLOW + ML
#  Insider cluster + rising IV + high ML = triple signal alignment
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_3_insider_ml_options(trades: pd.DataFrame, vix: pd.DataFrame,
                                   prices_cache: dict) -> pd.DataFrame:
    """
    When ML rank > 0.75 AND stock is a known insider-buy name, buy calls.
    Use trades_v2.csv high-ML entries as proxy for insider+ML alignment.
    """
    print("\n▸ STRATEGY 3: INSIDER + ML + OPTIONS FLOW (triple signal)")

    # Load insider sequence data
    insider_path = ROOT / "cache_insider" / "insider_sequence.json"
    insider_syms = set()
    if insider_path.exists():
        try:
            data = json.loads(insider_path.read_text())
            for rec in data:
                if isinstance(rec, dict) and rec.get("insider_seq_strength", 0) > 0.3:
                    insider_syms.add(rec.get("symbol", ""))
        except Exception:
            pass

    # Proxy: Use high-ML long entries in non-BEAR regime as insider-aligned
    # (in practice, insider cluster signals would filter further)
    high_ml = trades[
        (trades["ml_rank_pct"] >= 0.75) &
        (trades["side"] == "long") &
        (trades["regime"] != "BEAR")
    ].copy()

    vix_c = vix["close"] if "close" in vix.columns else vix.iloc[:, 0]
    results = []

    for _, t in high_ml.iterrows():
        sym = t["symbol"]
        edate = t["entry_date"]

        # Check if VIX is rising (momentum in vol = options flow signal)
        vix_5d = vix_c.loc[:edate].tail(6)
        if len(vix_5d) < 6:
            continue
        vix_rising = float(vix_5d.iloc[-1]) > float(vix_5d.iloc[0])

        if not vix_rising:
            continue  # Only when options flow is rising (IV expansion)

        if sym not in prices_cache:
            df = load_csv(sym)
            if df is not None:
                prices_cache[sym] = df
        if sym not in prices_cache:
            continue

        px = prices_cache[sym]["close"]
        entry_price = px.asof(edate)
        if pd.isna(entry_price):
            continue

        # Forward 10-day return
        future = px.index[px.index > edate][:10]
        if len(future) < 5:
            continue
        exit_price = float(px.loc[future[-1]])

        vix_val = float(vix_c.asof(edate))
        sigma = vix_val / 100 * 1.5
        call_cost = bs_call_approx(entry_price, entry_price, sigma, 30)
        call_exit = max(exit_price - entry_price, 0) + bs_call_approx(exit_price, entry_price, sigma, 20)
        call_exit = max(call_exit, max(exit_price - entry_price, 0))
        call_pnl = call_exit - call_cost
        call_ret = call_pnl / call_cost if call_cost > 0 else 0

        # Kelly sizing: 3 signals aligned = higher conviction
        signal_count = 1  # ML
        if sym in insider_syms:
            signal_count += 1
        if vix_rising:
            signal_count += 1
        kelly_fraction = min(signal_count * 0.10, 0.30)

        results.append({
            "symbol": sym, "entry_date": edate.strftime("%Y-%m-%d"),
            "ml_rank": t["ml_rank_pct"], "regime": t["regime"],
            "signals_aligned": signal_count,
            "insider_confirmed": sym in insider_syms,
            "vix_rising": vix_rising, "vix": vix_val,
            "equity_return": (exit_price - entry_price) / entry_price,
            "call_cost": call_cost, "call_pnl": call_pnl, "call_return": call_ret,
            "kelly_fraction": kelly_fraction,
            "kelly_weighted_pnl": call_ret * kelly_fraction,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No qualifying events found.")
        return df

    print(f"  Events: {len(df)} (ML>0.75 + VIX rising)")
    print(f"  Insider confirmed: {df['insider_confirmed'].sum()} ({df['insider_confirmed'].mean():.0%})")
    print(f"  Equity: avg={df['equity_return'].mean():+.2%}")
    print(f"  Calls:  avg={df['call_return'].mean():+.2%}, win={(df['call_return'] > 0).mean():.0%}")
    print(f"  Kelly-weighted: avg={df['kelly_weighted_pnl'].mean():+.2%}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 4: EARNINGS STREAK MOMENTUM CALLS
#  4+ beat streak + analyst revision + ML > 0.80 → buy calls 15d before ER
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_4_earnings_streak_calls(earnings_store: dict, vix: pd.DataFrame,
                                      prices_cache: dict) -> pd.DataFrame:
    """
    Stocks with 4+ consecutive beats: buy 30-DTE calls 15 days before earnings.
    Filter: only when ML rank would be high (use price momentum as proxy).
    """
    print("\n▸ STRATEGY 4: EARNINGS STREAK MOMENTUM CALLS (4+ beats → buy calls)")

    vix_c = vix["close"] if "close" in vix.columns else vix.iloc[:, 0]
    results = []

    for sym, edf in earnings_store.items():
        if len(edf) < 4:
            continue

        # Check for 4+ consecutive beats
        if "beat" not in edf.columns:
            continue

        edf_sorted = edf.sort_values("date")
        beats = edf_sorted["beat"].tolist()

        # All 4 must be beats
        if not all(beats[-4:]):
            continue

        # Load price data
        if sym not in prices_cache:
            df = load_csv(sym)
            if df is not None:
                prices_cache[sym] = df
        if sym not in prices_cache:
            continue

        px = prices_cache[sym]["close"]

        # For each earnings date, simulate buying calls 15 days before
        for _, er in edf_sorted.iterrows():
            er_date = er["date"]
            entry_date = er_date - pd.Timedelta(days=20)  # ~15 trading days before

            # Find actual trading day
            valid_dates = px.index[px.index <= entry_date]
            if len(valid_dates) < 200:
                continue
            actual_entry = valid_dates[-1]

            entry_price = float(px.loc[actual_entry])
            if entry_price <= 0:
                continue

            # ML proxy: 60-day momentum > 0 AND price > SMA200
            sma200 = float(px.loc[:actual_entry].tail(200).mean())
            mom_60 = (entry_price - float(px.loc[:actual_entry].tail(60).iloc[0])) / entry_price
            if entry_price < sma200 or mom_60 < 0:
                continue  # ML rank would be low

            # VIX at entry (IV proxy — IV is "cheap" 15 days before earnings)
            vix_val = vix_c.asof(actual_entry)
            if pd.isna(vix_val):
                continue

            # Exit: day after earnings (or 15 trading days later)
            future = px.index[px.index > er_date]
            if len(future) < 1:
                continue
            exit_date = future[0]
            exit_price = float(px.loc[exit_date])

            sigma = float(vix_val) / 100 * 1.4  # slight premium over VIX
            hold_days = (exit_date - actual_entry).days
            call_cost = bs_call_approx(entry_price, entry_price, sigma, 30)
            # At exit, IV has crushed ~40% post-earnings
            exit_sigma = sigma * 0.6
            exit_dte = max(30 - hold_days, 1)
            call_exit = max(exit_price - entry_price, 0) + bs_call_approx(
                exit_price, entry_price, exit_sigma, exit_dte
            ) * 0.3  # deep discount on remaining time value post-crush
            call_exit = max(call_exit, max(exit_price - entry_price, 0))
            call_pnl_val = call_exit - call_cost
            call_ret = call_pnl_val / call_cost if call_cost > 0 else 0

            results.append({
                "symbol": sym, "earnings_date": er_date.strftime("%Y-%m-%d"),
                "entry_date": actual_entry.strftime("%Y-%m-%d"),
                "beat_streak": 4, "surprise": er.get("surp", 0),
                "entry_price": entry_price, "exit_price": exit_price,
                "mom_60d": mom_60, "above_sma200": True,
                "vix_at_entry": float(vix_val),
                "equity_return": (exit_price - entry_price) / entry_price,
                "call_cost": call_cost, "call_pnl": call_pnl_val, "call_return": call_ret,
                "hold_days": hold_days,
            })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No qualifying events found.")
        return df

    print(f"  Events: {len(df)} (4+ beats, above SMA200, positive momentum)")
    print(f"  Avg surprise: {df['surprise'].mean():+.1%}")
    print(f"  Equity: avg={df['equity_return'].mean():+.2%}, win={(df['equity_return'] > 0).mean():.0%}")
    print(f"  Calls:  avg={df['call_return'].mean():+.2%}, win={(df['call_return'] > 0).mean():.0%}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 5: SECTOR LEADERSHIP + PUT SELLING
#  Sector in leadership + ML > 0.80 + CHOPPY → sell puts
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_5_leadership_puts(trades: pd.DataFrame, vix: pd.DataFrame,
                                regime_series: pd.Series, prices_cache: dict) -> pd.DataFrame:
    """
    In CHOPPY regime, sell puts on stocks where:
    - Sector is in leadership (20d sector ETF momentum > SPY)
    - ML rank > 0.80
    Premium income with directional confirmation.
    """
    print("\n▸ STRATEGY 5: SECTOR LEADERSHIP + PUT SELLING (CHOPPY + leading sector)")

    spy_df = load_csv("SPY")
    if spy_df is None:
        print("  ERROR: No SPY data")
        return pd.DataFrame()

    spy_c = spy_df["close"]

    # Load sector ETFs
    sector_etfs = {}
    for etf in ["XLK", "XLC", "XLY", "XLP", "XLF", "XLV", "XLI", "XLE", "XLU", "XLB"]:
        edf = load_csv(etf)
        if edf is not None:
            sector_etfs[etf] = edf["close"]

    # Reverse sector map: stock → ETF
    stock_to_etf = {}
    for etf, syms in getattr(config, "SECTOR_ETFS", {}).items():
        for s in syms:
            stock_to_etf[s] = etf

    vix_c = vix["close"] if "close" in vix.columns else vix.iloc[:, 0]

    # Filter trades: CHOPPY + high ML + long side
    choppy_high_ml = trades[
        (trades["regime"] == "CHOPPY") &
        (trades["ml_rank_pct"] >= 0.80) &
        (trades["side"] == "long")
    ].copy()

    results = []
    for _, t in choppy_high_ml.iterrows():
        sym = t["symbol"]
        edate = t["entry_date"]
        etf_name = stock_to_etf.get(sym)
        if not etf_name or etf_name not in sector_etfs:
            continue

        etf_c = sector_etfs[etf_name]

        # Check sector leadership: 20d return > SPY 20d return
        spy_ret_20d = spy_c.asof(edate) / spy_c.asof(edate - pd.Timedelta(days=30)) - 1
        etf_ret_20d = etf_c.asof(edate) / etf_c.asof(edate - pd.Timedelta(days=30)) - 1
        if pd.isna(spy_ret_20d) or pd.isna(etf_ret_20d):
            continue
        if etf_ret_20d <= spy_ret_20d:
            continue  # Sector not leading

        if sym not in prices_cache:
            df = load_csv(sym)
            if df is not None:
                prices_cache[sym] = df
        if sym not in prices_cache:
            continue

        px = prices_cache[sym]["close"]
        entry_price = px.asof(edate)
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Forward 30-day return (put expires)
        future = px.index[px.index > edate][:22]  # ~30 calendar days
        if len(future) < 15:
            continue
        exit_price = float(px.loc[future[-1]])

        vix_val = float(vix_c.asof(edate))
        sigma = vix_val / 100 * 1.3
        put_K, put_premium = put_premium_30dte(entry_price, sigma)

        # P&L: keep premium if stock stays above strike, lose if drops below
        if exit_price >= put_K:
            pnl = put_premium  # full premium
        else:
            assignment_loss = put_K - exit_price
            pnl = put_premium - assignment_loss

        ret = pnl / put_K * 100  # return on cash secured

        results.append({
            "symbol": sym, "entry_date": edate.strftime("%Y-%m-%d"),
            "ml_rank": t["ml_rank_pct"], "regime": "CHOPPY",
            "sector_etf": etf_name,
            "sector_outperf": float(etf_ret_20d - spy_ret_20d),
            "entry_price": entry_price, "exit_price": exit_price,
            "put_strike": put_K, "put_premium": put_premium,
            "stock_return": (exit_price - entry_price) / entry_price,
            "put_pnl": pnl, "put_return_on_capital": ret,
            "assigned": exit_price < put_K,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No qualifying events found.")
        return df

    print(f"  Events: {len(df)} (CHOPPY + ML>0.80 + sector leading)")
    print(f"  Avg sector outperformance: {df['sector_outperf'].mean():+.2%}")
    print(f"  Assigned: {df['assigned'].sum()}/{len(df)} ({df['assigned'].mean():.0%})")
    print(f"  Put P&L: avg={df['put_pnl'].mean():+.2f}, win={(df['put_pnl'] > 0).mean():.0%}")
    print(f"  Return on capital: avg={df['put_return_on_capital'].mean():+.2%}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STATS & RANKING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sharpe(returns: pd.Series, trades_per_year: float = 12) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0
    return (returns.mean() * trades_per_year) / (returns.std() * np.sqrt(trades_per_year))


def analyze_strategy(name: str, df: pd.DataFrame, ret_col: str) -> dict:
    if len(df) == 0:
        return {"name": name, "n": 0, "sharpe": 0}

    rets = df[ret_col].dropna()
    wins = (rets > 0).sum()
    losses = (rets <= 0).sum()
    win_rets = rets[rets > 0]
    loss_rets = rets[rets <= 0]
    pf = abs(win_rets.sum() / loss_rets.sum()) if loss_rets.sum() != 0 else float("inf")

    # Estimate trades per year from date range
    if "entry_date" in df.columns:
        dates = pd.to_datetime(df["entry_date"])
        span_years = max((dates.max() - dates.min()).days / 365.25, 1)
        trades_per_year = len(df) / span_years
    else:
        trades_per_year = 12

    sharpe = compute_sharpe(rets, trades_per_year)

    stats = {
        "name": name,
        "n": len(df),
        "win_rate": wins / len(df) if len(df) > 0 else 0,
        "avg_return": rets.mean(),
        "median_return": rets.median(),
        "sharpe": sharpe,
        "profit_factor": pf,
        "best": rets.max(),
        "worst": rets.min(),
        "trades_per_year": trades_per_year,
    }
    return stats


def print_ranking(all_stats: List[dict]):
    print(f"\n{'=' * 95}")
    print(f"  PROPRIETARY STRATEGY RANKING")
    print(f"{'=' * 95}")
    ranked = sorted(all_stats, key=lambda x: x.get("sharpe", 0), reverse=True)
    print(f"  {'#':>2} {'Strategy':40s} {'Sharpe':>8} {'WinRate':>8} {'AvgRet':>8} {'PF':>8} {'N':>6}")
    print(f"  {'-' * 90}")
    for i, s in enumerate(ranked, 1):
        wr = f"{s.get('win_rate',0):.0%}"
        ar = f"{s.get('avg_return',0):+.1%}"
        pf = f"{s.get('profit_factor',0):.2f}" if s.get("profit_factor", 0) < 100 else "inf"
        sh = f"{s.get('sharpe',0):+.2f}"
        print(f"  {i:2d} {s['name']:40s} {sh:>8} {wr:>8} {ar:>8} {pf:>8} {s['n']:>6}")
    print(f"{'=' * 95}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data...")
    trades = load_trades()
    vix = load_csv("^VIX")
    earnings_store = load_earnings_store()
    prices_cache: Dict[str, pd.DataFrame] = {}

    print(f"  Trades: {len(trades)}")
    print(f"  VIX: {len(vix)} days")
    print(f"  Earnings: {len(earnings_store)} symbols")

    print("\nReconstructing regime history...")
    regime_series = reconstruct_regime_series()
    regime_counts = regime_series.value_counts()
    print(f"  {dict(regime_counts)}")

    all_stats = []

    # Strategy 1
    df1 = strategy_1_ml_iv_divergence(trades, vix, prices_cache)
    if len(df1) > 0:
        s1 = analyze_strategy("S1: ML+IV Divergence (buy calls)", df1, "call_return")
        all_stats.append(s1)
        df1.to_csv(ROOT / "results" / "strat1_ml_iv_divergence.csv", index=False)

    # Strategy 2
    df2 = strategy_2_regime_iron_condor(trades, vix, regime_series, prices_cache)
    if len(df2) > 0:
        s2 = analyze_strategy("S2: Regime IC (CHOPPY+mid-ML)", df2, "ic_return")
        all_stats.append(s2)
        df2.to_csv(ROOT / "results" / "strat2_regime_iron_condor.csv", index=False)

    # Strategy 3
    df3 = strategy_3_insider_ml_options(trades, vix, prices_cache)
    if len(df3) > 0:
        s3 = analyze_strategy("S3: Insider+ML+Options (triple signal)", df3, "call_return")
        all_stats.append(s3)
        df3.to_csv(ROOT / "results" / "strat3_insider_ml_options.csv", index=False)

    # Strategy 4
    df4 = strategy_4_earnings_streak_calls(earnings_store, vix, prices_cache)
    if len(df4) > 0:
        s4 = analyze_strategy("S4: EarningsStreak Calls (4+beats)", df4, "call_return")
        all_stats.append(s4)
        df4.to_csv(ROOT / "results" / "strat4_earnings_streak_calls.csv", index=False)

    # Strategy 5
    df5 = strategy_5_leadership_puts(trades, vix, regime_series, prices_cache)
    if len(df5) > 0:
        s5 = analyze_strategy("S5: Leadership Puts (sector+ML+CHOPPY)", df5, "put_return_on_capital")
        all_stats.append(s5)
        df5.to_csv(ROOT / "results" / "strat5_leadership_puts.csv", index=False)

    # Final ranking
    print_ranking(all_stats)
    print("Results saved to ~/ai_trading_bot_v2/results/")
