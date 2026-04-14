"""
trending_bull_stop_profile.py — What distinguishes TRENDING_BULL stops from wins?

CONTEXT
  Stop forensics showed 144 of 195 stops happen in TRENDING_BULL regime, for
  -$74k damage. Concentration analysis showed concentration is NOT the
  distinguishing feature in TRENDING_BULL (Cohen's d < 0.12 for every metric).
  So what IS the distinguishing feature?

METHOD
  Two populations from trades_v2.csv, both restricted to regime=TRENDING_BULL,
  engine=momentum (we exclude meanrev which has its own dynamics):

    LOSERS  : reason='stop'
    WINNERS : reason='max_hold' AND pnl > 0
              (successful max-hold trades are the cleanest baseline)

  For every entry-time feature we have, compute Cohen's d. No thresholds are
  picked by inspection; we are just describing which features separate the
  populations and by how much.

ENTRY-TIME FEATURES
  From trades_v2.csv directly:
    ml_rank_pct      (the ranker's confidence)
    rule_score       (rule-based score)
    combined_score   (the blended score used for selection)
    ann_vol          (60d annualized volatility at entry)

  Enrichments (requires lifecycle + price cache):
    portfolio_size_at_entry   — how crowded was the book?
    portfolio_corr_at_entry   — how correlated?
    vix_at_entry              — macro fear gauge
    spy_20d_at_entry          — SPY trailing momentum
    spy_5d_at_entry           — SPY recent direction
    sector_5d_at_entry        — (not computed yet — no sector map)
    stock_ret_20d_at_entry    — price momentum at entry
    stock_vol_20d_at_entry    — realized vol at entry
    stock_above_sma50         — price > 50d SMA?
    stock_above_sma200        — price > 200d SMA?
    distance_from_52w_high    — how stretched?

OUTPUT
  For each feature:
    stop group mean | win group mean | difference | Cohen's d

  Features are sorted by |d|. Any with |d| > 0.3 is a candidate signal.

NO BACKFITTING
  We are not training any model here. We are not picking thresholds. We are
  describing population differences. Anything we decide to DO about what we
  find here will be: (a) train on 2015-2021 only, (b) test on 2022-2026 OOS,
  (c) measure vs the noise floor.

Usage:
  python3.11 tools/trending_bull_stop_profile.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRADES = REPO / "trades_v2.csv"
LIFECYCLE = REPO / "cache_backtester" / "position_lifecycle.csv"
PRICES_DIR = REPO / "cache_prices"


def _load_price(sym: str) -> pd.DataFrame | None:
    for pat in [f"{sym}_3650d.csv", f"{sym}_1825d.csv"]:
        p = PRICES_DIR / pat
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0)
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                df = df.loc[~df.index.isna()]
                if getattr(df.index, "tz", None):
                    df.index = df.index.tz_convert("UTC").tz_localize(None)
                df.columns = [str(c).lower() for c in df.columns]
                return df
            except Exception:
                continue
    return None


def _load_macro(name: str) -> pd.DataFrame | None:
    for pat in [f"{name}_regime.csv", f"{name}_3650d.csv"]:
        p = PRICES_DIR / pat
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0)
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                df = df.loc[~df.index.isna()]
                if getattr(df.index, "tz", None):
                    df.index = df.index.tz_convert("UTC").tz_localize(None)
                df.columns = [str(c).lower() for c in df.columns]
                return df
            except Exception:
                continue
    return None


def cohens_d(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int, int]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0, float(a.mean()) if len(a) else np.nan, float(b.mean()) if len(b) else np.nan, len(a), len(b)
    ma, mb = float(a.mean()), float(b.mean())
    sa, sb = float(a.std(ddof=1)), float(b.std(ddof=1))
    pooled = np.sqrt(((len(a)-1)*sa**2 + (len(b)-1)*sb**2) / (len(a)+len(b)-2))
    d = (ma - mb) / pooled if pooled > 0 else 0.0
    return float(d), ma, mb, len(a), len(b)


def enrich_trade(
    trade: pd.Series,
    spy: pd.DataFrame | None,
    vix: pd.DataFrame | None,
    lifecycle: pd.DataFrame | None,
) -> dict:
    out = {}
    sym = str(trade["symbol"]).upper()
    e_dt = pd.Timestamp(trade["entry_date"])

    # Macro at entry
    if spy is not None and pd.notna(e_dt):
        spy_to_entry = spy.loc[spy.index <= e_dt, "close"]
        if len(spy_to_entry) >= 21:
            out["spy_20d_at_entry"] = float(spy_to_entry.iloc[-1] / spy_to_entry.iloc[-21] - 1)
            out["spy_5d_at_entry"]  = float(spy_to_entry.iloc[-1] / spy_to_entry.iloc[-6]  - 1) if len(spy_to_entry) >= 6 else np.nan
        else:
            out["spy_20d_at_entry"] = out["spy_5d_at_entry"] = np.nan
    else:
        out["spy_20d_at_entry"] = out["spy_5d_at_entry"] = np.nan

    if vix is not None and pd.notna(e_dt):
        vix_to_entry = vix.loc[vix.index <= e_dt, "close"]
        out["vix_at_entry"] = float(vix_to_entry.iloc[-1]) if len(vix_to_entry) else np.nan
    else:
        out["vix_at_entry"] = np.nan

    # Stock state at entry (from price cache)
    px = _load_price(sym)
    if px is not None and pd.notna(e_dt) and "close" in px.columns:
        hist = px.loc[px.index <= e_dt, "close"]
        if len(hist) >= 210:
            last = float(hist.iloc[-1])
            out["stock_ret_20d_at_entry"] = float(hist.iloc[-1] / hist.iloc[-21] - 1) if len(hist) >= 21 else np.nan
            out["stock_ret_60d_at_entry"] = float(hist.iloc[-1] / hist.iloc[-61] - 1) if len(hist) >= 61 else np.nan
            out["stock_vol_20d_at_entry"] = float(hist.pct_change().tail(20).std() * np.sqrt(252))
            out["stock_vol_60d_at_entry"] = float(hist.pct_change().tail(60).std() * np.sqrt(252))
            sma50 = float(hist.tail(50).mean())
            sma200 = float(hist.tail(200).mean())
            out["stock_vs_sma50"] = last / sma50 - 1 if sma50 > 0 else np.nan
            out["stock_vs_sma200"] = last / sma200 - 1 if sma200 > 0 else np.nan
            high_252 = float(hist.tail(252).max())
            out["dist_from_52w_high"] = last / high_252 - 1 if high_252 > 0 else np.nan
            # Relative strength vs SPY last 20 days
            if spy is not None:
                spy_to_entry = spy.loc[spy.index <= e_dt, "close"]
                if len(spy_to_entry) >= 21:
                    stock_20d = float(hist.iloc[-1] / hist.iloc[-21] - 1)
                    spy_20d = float(spy_to_entry.iloc[-1] / spy_to_entry.iloc[-21] - 1)
                    out["rs_vs_spy_20d"] = stock_20d - spy_20d
                else:
                    out["rs_vs_spy_20d"] = np.nan
            # Volume context: last-day vs 20d avg
            if "volume" in px.columns:
                vol_hist = px.loc[px.index <= e_dt, "volume"]
                if len(vol_hist) >= 21:
                    vol_ma20 = float(vol_hist.tail(20).mean())
                    last_vol = float(vol_hist.iloc[-1])
                    out["vol_ratio_at_entry"] = last_vol / vol_ma20 if vol_ma20 > 0 else np.nan
                else:
                    out["vol_ratio_at_entry"] = np.nan
        else:
            for k in ("stock_ret_20d_at_entry", "stock_ret_60d_at_entry",
                      "stock_vol_20d_at_entry", "stock_vol_60d_at_entry",
                      "stock_vs_sma50", "stock_vs_sma200", "dist_from_52w_high",
                      "rs_vs_spy_20d", "vol_ratio_at_entry"):
                out[k] = np.nan
    else:
        for k in ("stock_ret_20d_at_entry", "stock_ret_60d_at_entry",
                  "stock_vol_20d_at_entry", "stock_vol_60d_at_entry",
                  "stock_vs_sma50", "stock_vs_sma200", "dist_from_52w_high",
                  "rs_vs_spy_20d", "vol_ratio_at_entry"):
            out[k] = np.nan

    # Portfolio state at entry from lifecycle
    if lifecycle is not None and pd.notna(e_dt):
        lc_on = lifecycle[lifecycle["date"] == e_dt]
        if len(lc_on) > 0:
            out["portfolio_size_at_entry"] = float(lc_on["portfolio_size"].max())
            out["portfolio_corr_at_entry"] = float(lc_on["portfolio_corr"].max())
        else:
            # Try the nearest prior date
            lc_prior = lifecycle[lifecycle["date"] < e_dt]
            if len(lc_prior) > 0:
                recent = lc_prior[lc_prior["date"] == lc_prior["date"].max()]
                out["portfolio_size_at_entry"] = float(recent["portfolio_size"].max()) if len(recent) else np.nan
                out["portfolio_corr_at_entry"] = float(recent["portfolio_corr"].max()) if len(recent) else np.nan
            else:
                out["portfolio_size_at_entry"] = np.nan
                out["portfolio_corr_at_entry"] = np.nan
    else:
        out["portfolio_size_at_entry"] = np.nan
        out["portfolio_corr_at_entry"] = np.nan

    return out


def main():
    os.chdir(REPO)
    print("═" * 78)
    print("  TRENDING_BULL STOP PROFILE")
    print("═" * 78)

    trades = pd.read_csv(TRADES)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")

    # Restrict to TRENDING_BULL + momentum (meanrev has different dynamics)
    tb = trades[(trades["regime"] == "TRENDING_BULL") & (trades.get("engine", "momentum") == "momentum")].copy()
    print(f"\n  Total TRENDING_BULL momentum trades: {len(tb)}")

    losers = tb[tb["reason"] == "stop"].copy()
    # Winners: max_hold exits with positive PnL (the cleanest "the thesis worked" baseline)
    winners = tb[(tb["reason"] == "max_hold") & (tb["pnl"] > 0)].copy()

    print(f"  LOSERS  (stop)           : {len(losers)}  avg ${losers['pnl'].mean():>8,.0f}  total ${losers['pnl'].sum():>10,.0f}")
    print(f"  WINNERS (max_hold, +PnL) : {len(winners)}  avg ${winners['pnl'].mean():>8,.0f}  total ${winners['pnl'].sum():>10,.0f}")

    if len(losers) < 10 or len(winners) < 10:
        print("  Insufficient sample. Bailing.")
        return

    # Load macro once
    print("\n  loading macro...")
    spy = _load_macro("SPY")
    vix = _load_macro("VIX")
    if vix is None:
        vix = _load_macro("VIXY")
    print(f"    SPY: {'OK' if spy is not None else 'MISSING'}")
    print(f"    VIX: {'OK' if vix is not None else 'MISSING'}")

    print("\n  loading lifecycle...")
    lifecycle = None
    if LIFECYCLE.exists():
        lifecycle = pd.read_csv(LIFECYCLE)
        lifecycle["date"] = pd.to_datetime(lifecycle["date"], errors="coerce")
        print(f"    {len(lifecycle)} rows")

    # Enrich each trade
    print(f"\n  enriching {len(losers)+len(winners)} trades (reading price caches)...")
    n_missing_px = 0
    loser_rows = []
    winner_rows = []

    for _, tr in losers.iterrows():
        enriched = enrich_trade(tr, spy, vix, lifecycle)
        # Add direct trade_csv features
        for col in ("ml_rank_pct", "rule_score", "combined_score", "ann_vol"):
            if col in tr:
                enriched[col] = tr[col]
        if np.isnan(enriched.get("stock_ret_20d_at_entry", np.nan)):
            n_missing_px += 1
        loser_rows.append(enriched)

    for _, tr in winners.iterrows():
        enriched = enrich_trade(tr, spy, vix, lifecycle)
        for col in ("ml_rank_pct", "rule_score", "combined_score", "ann_vol"):
            if col in tr:
                enriched[col] = tr[col]
        if np.isnan(enriched.get("stock_ret_20d_at_entry", np.nan)):
            n_missing_px += 1
        winner_rows.append(enriched)

    if n_missing_px:
        print(f"  ⚠ {n_missing_px} trades had no price cache (some features will be sparse)")

    L = pd.DataFrame(loser_rows)
    W = pd.DataFrame(winner_rows)

    # Compute Cohen's d for each feature
    features = [
        "ml_rank_pct", "rule_score", "combined_score", "ann_vol",
        "spy_20d_at_entry", "spy_5d_at_entry", "vix_at_entry",
        "stock_ret_20d_at_entry", "stock_ret_60d_at_entry",
        "stock_vol_20d_at_entry", "stock_vol_60d_at_entry",
        "stock_vs_sma50", "stock_vs_sma200", "dist_from_52w_high",
        "rs_vs_spy_20d", "vol_ratio_at_entry",
        "portfolio_size_at_entry", "portfolio_corr_at_entry",
    ]
    rows = []
    for f in features:
        if f not in L.columns or f not in W.columns:
            continue
        a = L[f].to_numpy(dtype=float)
        b = W[f].to_numpy(dtype=float)
        d, ma, mb, na, nb = cohens_d(a, b)
        rows.append({
            "feature": f,
            "losers_mean": ma,
            "winners_mean": mb,
            "diff": ma - mb,
            "cohens_d": d,
            "abs_d": abs(d),
            "n_losers": na,
            "n_winners": nb,
        })
    out = pd.DataFrame(rows).sort_values("abs_d", ascending=False)

    print("\n" + "─" * 78)
    print("  FEATURE SEPARATION — sorted by |Cohen's d|")
    print("─" * 78)
    print("  (d < 0.2 negligible · 0.2-0.5 small · 0.5-0.8 medium · > 0.8 large)")
    print()
    print(f"  {'feature':<28} {'losers':>10} {'winners':>10} {'diff':>10} {'d':>7}  {'n_L/n_W'}")
    print("  " + "─" * 76)
    for _, r in out.iterrows():
        if pd.isna(r["cohens_d"]) or not np.isfinite(r["losers_mean"]):
            continue
        flag = ""
        if abs(r["cohens_d"]) >= 0.5:
            flag = " ★★"
        elif abs(r["cohens_d"]) >= 0.3:
            flag = " ★"
        print(f"  {r['feature']:<28} {r['losers_mean']:>10.4f} {r['winners_mean']:>10.4f} "
              f"{r['diff']:>+10.4f} {r['cohens_d']:>+7.2f}  {int(r['n_losers'])}/{int(r['n_winners'])}{flag}")

    # Combined signal test: top-3 features by |d|, normalized, summed into one score
    top = out.head(3)["feature"].tolist()
    print("\n" + "─" * 78)
    print(f"  TOP 3 FEATURES: {top}")
    print("─" * 78)
    print("  If any show d > 0.3, that's a candidate for feature engineering")
    print("  OR a hard pre-trade gate. Do NOT tune a threshold from this output.")
    print("  Instead: train logistic regression on 2015-2021 only and test OOS.")

    print("\n" + "═" * 78)
    print("  Saved full comparison to /tmp/tb_stop_profile.csv")
    print("═" * 78 + "\n")
    out.to_csv("/tmp/tb_stop_profile.csv", index=False)


if __name__ == "__main__":
    main()
