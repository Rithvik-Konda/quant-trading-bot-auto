"""
factor_decomp.py — What is your ranker's edge actually made of?

Reverse-engineers your system's factor exposures by regressing the daily
PnL of your trades against the Fama-French 5-factor returns + momentum:

    ranker_return_t = α + β_MKT·MKT + β_SMB·SMB + β_HML·HML
                        + β_RMW·RMW + β_CMA·CMA + β_MOM·MOM + ε

Where the six factors mean:
  MKT = market excess return (the whole market going up)
  SMB = small-minus-big     (small cap premium)
  HML = high-minus-low B/M  (value premium)
  RMW = robust-minus-weak   (profitability / quality premium)
  CMA = conservative-minus-aggressive investment (investment premium)
  MOM = winners-minus-losers (momentum premium)

WHY THIS MATTERS
  Your system has 387 features and you have never asked what factor
  exposures it actually takes in aggregate. If β_MOM is 0.9 and every other
  β is near zero, you are a momentum machine and your ceiling is the
  momentum factor's ceiling. If you have meaningful RMW or CMA loadings,
  your system is secretly diversified and that's valuable. Either answer
  reshapes the roadmap.

  α (the intercept) is your TRUE alpha — the part of your return that
  can't be explained by exposure to any of these six factors. If α is
  statistically significant and positive, you have real skill beyond
  factor beta. If α is zero, your "edge" is just cheap factor exposure.

METHOD
  1. Reconstruct daily equity curve from trades_v2.csv
  2. Compute daily strategy returns (excess over risk-free)
  3. Download Fama-French 5-factor + momentum from Ken French's website
     (free, daily, goes back to 1926)
  4. Align dates, run OLS regression
  5. Report β for each factor, α, R², and t-stats

NO BACKFITTING
  This is a descriptive analysis. We are not choosing any threshold, not
  fitting anything to OOS, not optimizing. We are just describing what
  exposures the system has taken on average over its lifetime.

DEPENDENCIES
  Uses pandas, numpy, and urllib. No new packages required.

Usage:
  python3.11 tools/factor_decomp.py
"""

from __future__ import annotations
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRADES = REPO / "trades_v2.csv"
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
CACHE_DIR = REPO / "cache_prices"


def _download_and_parse_ff(url: str, cache_name: str) -> pd.DataFrame:
    """Download a Ken French factor zip, return a daily DataFrame."""
    cache = CACHE_DIR / cache_name
    if cache.exists():
        print(f"  [cache] {cache_name}")
        raw = cache.read_text()
    else:
        print(f"  [dl] {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
        except Exception as e:
            print(f"  [ERROR] download failed: {e}")
            return pd.DataFrame()
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                inner = zf.namelist()[0]
                raw = zf.read(inner).decode("latin-1")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(raw)
        except Exception as e:
            print(f"  [ERROR] unzip failed: {e}")
            return pd.DataFrame()

    # Ken French format: skip preamble lines until we find the header row
    lines = raw.split("\n")
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Header row has commas and contains "Mkt-RF" or similar
        if "," in stripped and any(tok in stripped for tok in ["Mkt-RF", "Mom", "WML", "Mkt"]):
            data_start = i
            break
    if data_start is None:
        print("  [ERROR] could not find header in Ken French file")
        return pd.DataFrame()

    # Collect lines until we hit blank line or end
    data_lines = [lines[data_start]]
    for line in lines[data_start + 1:]:
        stripped = line.strip()
        if not stripped:
            break
        # Data lines start with 8-digit date
        if not (stripped[:8].isdigit()):
            break
        data_lines.append(stripped)

    if len(data_lines) < 2:
        print("  [ERROR] no data rows parsed")
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df.columns = [c.strip() for c in df.columns]
    # First column is date as YYYYMMDD integer
    date_col = df.columns[0]
    df[date_col] = df[date_col].astype(str).str.strip()
    df.index = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
    df = df.drop(columns=[date_col])
    df = df.loc[df.index.notna()]
    # Values are in percent; convert to decimal
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df


def load_ff_factors() -> pd.DataFrame | None:
    ff5 = _download_and_parse_ff(FF5_URL, "ff5_daily.csv")
    mom = _download_and_parse_ff(MOM_URL, "mom_daily.csv")
    if ff5.empty or mom.empty:
        return None
    # Rename mom column for clarity
    mom_col = mom.columns[0]
    mom = mom.rename(columns={mom_col: "Mom"})
    out = ff5.join(mom, how="inner")
    # Clean column names
    out.columns = [c.strip().replace("-", "_") for c in out.columns]
    return out


def build_strategy_returns(trades: pd.DataFrame) -> pd.Series:
    """
    Build a daily strategy return series from trades.

    Correct method: aggregate per-day dollar PnL from trades, then
    compute daily return as daily_pnl / running_equity, where running
    equity starts at INITIAL_CAPITAL and accumulates all PnL realized
    up to that day.

    This avoids the per-share-return-averaging bug which distorts
    portfolios of differently-sized positions.
    """
    INITIAL_CAPITAL = 100_000.0  # matches config.INITIAL_CAPITAL default

    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["exit_date"]  = pd.to_datetime(trades["exit_date"], errors="coerce")
    trades = trades.dropna(subset=["entry_date", "exit_date", "pnl"])

    # Spread each trade's dollar PnL uniformly across its hold period.
    # This converts lumpy trade-level PnL into a smooth daily dollar PnL series
    # that approximates the mark-to-market path.
    all_dates = pd.date_range(trades["entry_date"].min(), trades["exit_date"].max(), freq="B")
    daily_pnl = pd.Series(0.0, index=all_dates)

    for _, row in trades.iterrows():
        span = pd.date_range(row["entry_date"], row["exit_date"], freq="B")
        if len(span) == 0:
            continue
        per_day_dollars = float(row["pnl"]) / len(span)
        # Only add to dates present in our grid
        for d in span:
            if d in daily_pnl.index:
                daily_pnl.loc[d] += per_day_dollars

    # Daily return = daily dollar PnL / running equity
    running_equity = INITIAL_CAPITAL + daily_pnl.cumsum().shift(1).fillna(0.0)
    daily_return = daily_pnl / running_equity
    # Drop initial zero-pnl days
    daily_return = daily_return.loc[daily_pnl != 0]
    return daily_return


def run_ols(y: pd.Series, X: pd.DataFrame) -> dict:
    """
    Simple OLS regression using numpy. Returns betas, t-stats, R².
    No scipy needed.
    """
    # Add intercept
    X = X.copy()
    X.insert(0, "alpha", 1.0)
    X_mat = X.values
    y_vec = y.values

    # β = (X'X)^-1 X'y
    XtX = X_mat.T @ X_mat
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return {"error": "singular design matrix"}
    betas = XtX_inv @ X_mat.T @ y_vec

    # Residuals and variance
    y_hat = X_mat @ betas
    residuals = y_vec - y_hat
    n, k = X_mat.shape
    dof = n - k
    if dof <= 0:
        return {"error": "insufficient degrees of freedom"}
    resid_var = (residuals @ residuals) / dof
    cov_betas = resid_var * XtX_inv
    se_betas = np.sqrt(np.diag(cov_betas))

    # R²
    ss_tot = ((y_vec - y_vec.mean()) ** 2).sum()
    ss_res = (residuals ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Adjusted R²
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else 0.0

    return {
        "betas": dict(zip(X.columns, betas)),
        "se": dict(zip(X.columns, se_betas)),
        "tstats": dict(zip(X.columns, betas / se_betas)),
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "dof": dof,
    }


def main():
    os.chdir(REPO)
    print("═" * 78)
    print("  FACTOR DECOMPOSITION — what is your edge made of?")
    print("═" * 78)

    if not TRADES.exists():
        print(f"FATAL: {TRADES} missing.")
        sys.exit(1)

    print("\n  [1/4] loading trades...")
    trades = pd.read_csv(TRADES)
    print(f"    {len(trades)} trades")

    print("  [2/4] building daily strategy returns...")
    strat = build_strategy_returns(trades)
    print(f"    {len(strat)} active trading days")
    print(f"    mean daily return: {strat.mean()*100:>+6.3f}%")
    print(f"    annualized return: {strat.mean()*252*100:>+6.1f}%")
    print(f"    daily std:         {strat.std()*100:>+6.3f}%")
    print(f"    annualized std:    {strat.std()*np.sqrt(252)*100:>+6.1f}%")
    sharpe = strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
    print(f"    Sharpe (approx):   {sharpe:>+6.2f}")

    print("\n  [3/4] loading Fama-French factors from Ken French's site...")
    ff = load_ff_factors()
    if ff is None or ff.empty:
        print("  ERROR: could not load factors. Aborting.")
        sys.exit(1)
    print(f"    {len(ff)} days of factor data, {len(ff.columns)} factors")
    print(f"    columns: {list(ff.columns)}")
    print(f"    date range: {ff.index.min().date()} → {ff.index.max().date()}")

    # Align
    df = pd.DataFrame({"strat": strat}).join(ff, how="inner")
    df = df.dropna()
    if "RF" in df.columns:
        df["strat_excess"] = df["strat"] - df["RF"]
    else:
        df["strat_excess"] = df["strat"]

    print(f"\n  [4/4] aligned data: {len(df)} days")
    if len(df) < 60:
        print("  WARN: very few aligned days — regression may be unstable")

    # Identify factor columns
    factor_cols = [c for c in df.columns if c not in ("strat", "strat_excess", "RF")]
    print(f"    regressing on: {factor_cols}")

    y = df["strat_excess"]
    X = df[factor_cols]

    results = run_ols(y, X)
    if "error" in results:
        print(f"\n  REGRESSION ERROR: {results['error']}")
        sys.exit(1)

    # Report
    print("\n" + "─" * 78)
    print("  FACTOR REGRESSION RESULTS")
    print("─" * 78)
    print(f"\n  N = {results['n']} days    R² = {results['r2']:.4f}    adj R² = {results['adj_r2']:.4f}")
    print()
    print(f"  {'term':<10} {'β (loading)':>14} {'SE':>10} {'t-stat':>10}  {'interpretation'}")
    print("  " + "─" * 74)

    order = ["alpha", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    labels = {
        "alpha":  "TRUE ALPHA (unexplained)",
        "Mkt_RF": "market beta",
        "SMB":    "small cap tilt",
        "HML":    "value tilt",
        "RMW":    "quality/profitability tilt",
        "CMA":    "conservative investment tilt",
        "Mom":    "momentum loading",
    }
    for term in order:
        if term not in results["betas"]:
            continue
        b = results["betas"][term]
        se = results["se"][term]
        t = results["tstats"][term]
        # Annualize alpha for readability
        if term == "alpha":
            b_display = b * 252
            se_display = se * 252
            print(f"  {term:<10} {b_display:>+13.3f}% {se_display:>9.3f}% {t:>+10.2f}  {labels.get(term, '')}  (annualized)")
        else:
            sig = ""
            if abs(t) > 2.58: sig = " ***"
            elif abs(t) > 1.96: sig = " **"
            elif abs(t) > 1.65: sig = " *"
            print(f"  {term:<10} {b:>+14.3f} {se:>10.3f} {t:>+10.2f}  {labels.get(term, '')}{sig}")

    print("\n  (*** p<0.01, ** p<0.05, * p<0.10)")
    print()

    # Interpretation
    print("─" * 78)
    print("  INTERPRETATION")
    print("─" * 78)
    betas = results["betas"]
    tstats = results["tstats"]
    alpha_daily = betas.get("alpha", 0)
    alpha_annual = alpha_daily * 252
    alpha_t = tstats.get("alpha", 0)

    print(f"\n  Annualized α:  {alpha_annual*100:+.2f}%   t-stat: {alpha_t:+.2f}")
    if abs(alpha_t) > 1.96:
        print("    → STATISTICALLY SIGNIFICANT true alpha.")
        print("      Your system has edge that is NOT explained by factor exposures.")
    else:
        print("    → Alpha NOT statistically significant.")
        print("      Most of your 'edge' is compensation for factor exposure.")
        print("      The good news: factor exposure is real return, just not 'skill'.")

    mom_beta = betas.get("Mom", 0)
    mom_t = tstats.get("Mom", 0)
    print(f"\n  Momentum loading:  β={mom_beta:+.2f}   t={mom_t:+.2f}")
    if abs(mom_t) > 1.96 and mom_beta > 0.3:
        print("    → Confirmed momentum strategy. Ceiling is the momentum factor itself.")
    elif abs(mom_t) > 1.96:
        print("    → Mild momentum tilt. You may have non-momentum alpha sources.")
    else:
        print("    → No significant momentum loading. Your system is NOT what you think it is.")

    rmw_t = tstats.get("RMW", 0)
    cma_t = tstats.get("CMA", 0)
    hml_t = tstats.get("HML", 0)
    smb_t = tstats.get("SMB", 0)
    diversified = sum(abs(t) > 1.96 for t in (rmw_t, cma_t, hml_t, smb_t))
    print(f"\n  Diversified factor exposures (|t|>1.96): {diversified}/4")
    if diversified >= 2:
        print("    → Genuinely diversified factor exposure. You have more than momentum.")
    else:
        print("    → Limited factor diversification. Opportunities in RMW/CMA/HML untapped.")

    print(f"\n  R² = {results['r2']:.1%}")
    if results["r2"] > 0.50:
        print("    → Factors explain most of your daily return variance.")
        print("      Your edge is factor-driven. Uncorrelated alpha sources are the unlock.")
    elif results["r2"] > 0.25:
        print("    → Factors explain about half your variance. Mixed factor + idio alpha.")
    else:
        print("    → Factors explain little of your variance. You have idiosyncratic selection skill.")

    print("\n" + "═" * 78 + "\n")


if __name__ == "__main__":
    main()
