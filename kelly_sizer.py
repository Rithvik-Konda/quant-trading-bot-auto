"""
kelly_sizer.py — Dynamic Kelly position sizing

Replaces hardcoded risk_pt=0.035 in backtester_v2.py

Kelly fraction: f* = W - (1-W)/R
  W = win rate on similar trades
  R = avg_win / avg_loss ratio

"Similar trades" defined by:
  1. Same regime (TRENDING_BULL / CHOPPY / BEAR)
  2. ML score bucket (0.80-0.90, 0.90-0.95, 0.95-1.0)
  3. Volatility bucket (low <0.30, mid 0.30-0.55, high >0.55)

Minimum 15 similar trades required — fall back to 3.5% if insufficient history.
Kelly fraction capped at 10% of capital to prevent ruin.
Half-Kelly used (standard practice) to account for estimation error.
"""
import numpy as np
import pandas as pd
from typing import List, Optional


def _vol_bucket(ann_vol: float) -> str:
    if ann_vol < 0.30:
        return "low"
    elif ann_vol < 0.55:
        return "mid"
    else:
        return "high"


def _ml_bucket(ml_score: float) -> str:
    if ml_score >= 0.95:
        return "top"
    elif ml_score >= 0.90:
        return "high"
    else:
        return "mid"


def compute_kelly(
    trades:     List[dict],
    regime:     str,
    ml_score:   float,
    ann_vol:    float,
    min_trades: int = 15,
    half_kelly: bool = True,
    max_kelly:  float = 0.10,
    fallback:   float = 0.035,
) -> float:
    """
    Compute Kelly fraction from recent trade history.

    Parameters
    ----------
    trades      : list of trade dicts with keys: pnl, regime, ml_rank_pct, ann_vol, side
    regime      : current regime string
    ml_score    : ML rank percentile of candidate (0-1)
    ann_vol     : annualized volatility of candidate
    min_trades  : minimum similar trades required (else fallback)
    half_kelly  : use half-Kelly (standard risk management practice)
    max_kelly   : hard cap on Kelly fraction (prevents over-betting)
    fallback    : default risk fraction if insufficient history

    Returns
    -------
    float : Kelly fraction of capital to risk (e.g. 0.042 = 4.2%)
    """
    if not trades:
        return fallback

    ml_bkt  = _ml_bucket(ml_score)
    vol_bkt = _vol_bucket(ann_vol)

    # Filter to similar trades — progressively relax if not enough
    filters = [
        # Exact match: regime + ML bucket + vol bucket
        lambda t: (t.get("regime") == regime and
                   _ml_bucket(t.get("ml_rank_pct", 0.85)) == ml_bkt and
                   _vol_bucket(t.get("ann_vol", 0.35)) == vol_bkt),
        # Relax vol bucket
        lambda t: (t.get("regime") == regime and
                   _ml_bucket(t.get("ml_rank_pct", 0.85)) == ml_bkt),
        # Relax ML bucket
        lambda t: t.get("regime") == regime,
        # All long trades
        lambda t: t.get("side", "long") == "long",
    ]

    similar = []
    for f in filters:
        similar = [t for t in trades if f(t) and t.get("side", "long") == "long"]
        if len(similar) >= min_trades:
            break

    if len(similar) < min_trades:
        return fallback

    pnls = [t["pnl"] for t in similar]
    wins  = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    if not wins or not losses:
        return fallback

    win_rate = len(wins) / len(similar)
    avg_win  = np.mean(wins)
    avg_loss = abs(np.mean(losses))

    if avg_loss == 0:
        return fallback

    R = avg_win / avg_loss  # reward/risk ratio

    kelly = win_rate - (1 - win_rate) / R

    if kelly <= 0:
        return fallback * 0.5  # negative edge — reduce size

    if half_kelly:
        kelly *= 0.5

    return float(np.clip(kelly, fallback * 0.3, max_kelly))


def compute_kelly_from_tradelog(
    trade_log:  pd.DataFrame,
    regime:     str,
    ml_score:   float,
    ann_vol:    float,
    lookback_n: int = 100,
    **kwargs,
) -> float:
    """
    Convenience wrapper that takes a trades DataFrame.
    Uses most recent N trades as the sample.
    """
    if trade_log is None or len(trade_log) == 0:
        return kwargs.get("fallback", 0.035)

    cols_needed = {"pnl", "regime", "ml_rank_pct", "side"}
    if not cols_needed.issubset(trade_log.columns):
        return kwargs.get("fallback", 0.035)

    # Add ann_vol column if not present (use vol bucket proxy)
    if "ann_vol" not in trade_log.columns:
        trade_log = trade_log.copy()
        trade_log["ann_vol"] = 0.35  # neutral default

    recent = trade_log.tail(lookback_n)
    trades = recent.to_dict("records")
    return compute_kelly(trades, regime, ml_score, ann_vol, **kwargs)


# ── Diagnostics ────────────────────────────────────────────────────────────────

def kelly_diagnostics(trade_log: pd.DataFrame) -> None:
    """Print Kelly fractions by regime × ML bucket for diagnostics."""
    if trade_log is None or len(trade_log) == 0:
        print("No trades yet")
        return

    print("\n=== Kelly Diagnostics ===")
    print(f"{'Regime':<16} {'ML Bucket':<10} {'Trades':>7} {'WR':>7} {'R':>6} {'Kelly':>8} {'HalfK':>8}")
    print("-" * 65)

    from regime_classifier import TRENDING_BULL, CHOPPY, BEAR
    trades = trade_log[trade_log["side"] == "long"].to_dict("records")

    for regime in [TRENDING_BULL, CHOPPY, BEAR]:
        for ml_bkt, ml_score in [("top", 0.97), ("high", 0.92), ("mid", 0.87)]:
            similar = [t for t in trades
                       if t.get("regime") == regime and
                       _ml_bucket(t.get("ml_rank_pct", 0.85)) == ml_bkt]
            if len(similar) < 5:
                continue

            pnls   = [t["pnl"] for t in similar]
            wins   = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            if not wins or not losses:
                continue

            wr  = len(wins) / len(similar)
            R   = np.mean(wins) / abs(np.mean(losses))
            raw = wr - (1 - wr) / R
            hk  = max(raw * 0.5, 0)
            print(f"  {regime:<14} {ml_bkt:<10} {len(similar):>7} "
                  f"{wr:>7.1%} {R:>6.2f} {raw:>8.3f} {hk:>8.3f}")

    print()


if __name__ == "__main__":
    # Test with synthetic trades
    import sys
    sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

    try:
        trades_df = pd.read_csv('/Users/rick/ai_trading_bot_v2/trades_v2.csv')
        print(f"Loaded {len(trades_df)} trades from trades_v2.csv")

        # Add ann_vol proxy if missing
        if "ann_vol" not in trades_df.columns:
            trades_df["ann_vol"] = 0.35

        # Print diagnostics
        import sys as _sys
        _sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')
        kelly_diagnostics(trades_df)

        # Test specific scenarios
        print("=== Kelly fractions for today's candidates ===")
        scenarios = [
            ("TRENDING_BULL", 0.92, 0.35, "NVDA-like mid-vol"),
            ("TRENDING_BULL", 0.97, 0.55, "APP-like high-vol"),
            ("CHOPPY",        0.94, 0.28, "LLY-like low-vol"),
            ("CHOPPY",        0.96, 0.48, "SOFI-like mid-vol"),
            ("BEAR",          0.88, 0.30, "Defensive low-vol"),
        ]
        trades_list = trades_df.to_dict("records")
        for regime, ml, vol, label in scenarios:
            k = compute_kelly(trades_list, regime, ml, vol)
            print(f"  {label:<25} regime={regime:<14} ML={ml:.2f} vol={vol:.2f} → Kelly={k:.3f} ({k*100:.1f}%)")

    except FileNotFoundError:
        print("trades_v2.csv not found — run backtest first")
        # Demo with synthetic data
        import random
        random.seed(42)
        synthetic = []
        for i in range(200):
            win = random.random() < 0.56
            synthetic.append({
                "pnl": random.uniform(200, 2000) if win else random.uniform(-1200, -100),
                "regime": random.choice(["TRENDING_BULL", "CHOPPY", "BEAR"]),
                "ml_rank_pct": random.uniform(0.80, 1.0),
                "ann_vol": random.uniform(0.15, 0.90),
                "side": "long",
            })
        k = compute_kelly(synthetic, "CHOPPY", 0.94, 0.35)
        print(f"Synthetic test — Kelly fraction: {k:.3f} ({k*100:.1f}%)")
