"""
v2/sizing.py — Shared position sizing logic between backtester and live trader.

This is the FIRST module in the Step 6 refactor. It extracts the momentum
position sizing math from v2/backtester_v2_phase2step1.py so that both the
backtester and live_trader_v2.py can call it.

DESIGN PRINCIPLES:
- Pure function: no I/O, no globals, no strategy module imports.
- Caller supplies all inputs (capital, trade history, params).
- Caller decides what "capital" means (backtester=INITIAL_CAPITAL, live=portfolio).
- Function returns qty (integer share count).

REGRESSION TEST: After backtester is updated to use this function, it MUST
reproduce BASELINE_2026_05_step1 metrics byte-identical. If not, this
extraction has a bug.
"""
from typing import List, Optional


def compute_position_size_momentum(
    capital: float,
    price: float,
    stop_pct: float,
    *,
    trades: Optional[List[dict]] = None,
    regime: str = "TRENDING_BULL",
    ml_score: float = 0.85,
    ann_vol: float = 0.35,
    vol_scalar: float = 1.0,
    conviction: float = 1.0,
    position_scalar: float = 1.0,
    max_position_weight: float = 0.35,
    max_total_exposure: float = 1.6,
    gross_exposure_now: float = 0.0,
    cash_available: Optional[float] = None,
    fallback_risk_pt: float = 0.035,
    use_kelly: bool = True,
) -> int:
    """
    Compute integer share quantity for a momentum entry.
    
    Mirrors the sizing logic from v2/backtester_v2_phase2step1.py lines 895-940.
    
    Parameters
    ----------
    capital : float
        Reference capital for percentage-based sizing. Backtester passes
        INITIAL_CAPITAL; live trader passes current portfolio value.
    price : float
        Entry price per share.
    stop_pct : float
        Stop loss as fraction of price (e.g., 0.08 = 8% stop).
    trades : list of dicts, optional
        Historical trade records for Kelly computation. Each dict needs:
        pnl, regime, ml_rank_pct, ann_vol, side. Pass None to skip Kelly
        and use fallback_risk_pt directly.
    regime : str
        Current market regime (TRENDING_BULL / CHOPPY / BEAR).
    ml_score : float
        ML rank percentile of this candidate (0-1).
    ann_vol : float
        Annualized vol of this candidate's underlying.
    vol_scalar : float
        Macro vol-based size multiplier (e.g., from VIX gating). 0.0 to 1.0.
    conviction : float
        Per-trade conviction multiplier (e.g., from rule_score blending).
    position_scalar : float
        Per-regime sizing multiplier from strategy params.
    max_position_weight : float
        Per-position cap as fraction of capital.
    max_total_exposure : float
        Portfolio gross exposure cap as fraction of capital (1.6 = 160%).
    gross_exposure_now : float
        Current gross dollar exposure of open positions.
    cash_available : float, optional
        Cash on hand. If None, uses capital * max_total_exposure as proxy.
    fallback_risk_pt : float
        Risk per trade if Kelly unavailable (default 3.5%).
    use_kelly : bool
        If True, compute Kelly-based risk_pt; if False, use fallback directly.
    
    Returns
    -------
    qty : int
        Number of shares to buy. Returns 0 if any constraint blocks the trade.
    """
    # Hard guards
    if price <= 0 or stop_pct <= 0 or capital <= 0:
        return 0
    if vol_scalar <= 0:
        return 0
    
    # Kelly computation (or fallback)
    if use_kelly and trades is not None and len(trades) > 0:
        try:
            from kelly_sizer import compute_kelly
            risk_pt = compute_kelly(
                trades=trades,
                regime=regime,
                ml_score=ml_score,
                ann_vol=ann_vol,
                min_trades=15,
                fallback=fallback_risk_pt,
            )
        except Exception:
            risk_pt = fallback_risk_pt
    else:
        risk_pt = fallback_risk_pt
    
    # Gross exposure cap
    max_exp = capital * max_total_exposure
    remaining = max(0.0, max_exp - gross_exposure_now)
    if remaining <= 0:
        return 0
    
    # Cash cap
    if cash_available is None:
        cash_available = remaining
    
    # Risk-based qty
    risk_budget = capital * risk_pt * position_scalar * conviction
    risk_per_share = price * stop_pct
    qty_risk = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
    
    # Weight-based dollar cap
    max_dollars = min(
        capital * max_position_weight * position_scalar * conviction,
        cash_available,
        remaining,
    )
    qty_cap = int(max_dollars / price) if price > 0 else 0
    
    qty = min(qty_risk, qty_cap)
    return max(0, qty)



def compute_position_size_short(
    capital: float,
    price: float,
    stop_pct: float,
    *,
    risk_per_trade: float = 0.025,
    position_scalar: float = 1.0,
    max_position_weight: float = 0.08,
    cash_available: float = None,
    margin_requirement: float = 0.60,
) -> int:
    """
    Compute integer share quantity for a short entry.

    Mirrors the sizing logic from v2/backtester_v2_phase2step1.py lines 1180-1196.

    Parameters
    ----------
    capital : float
        Reference capital for percentage-based sizing.
    price : float
        Entry price per share.
    stop_pct : float
        Stop loss as fraction of price (e.g., 0.08 = 8%).
    risk_per_trade : float
        Risk fraction (default 2.5%).
    position_scalar : float
        Per-regime sizing multiplier (e.g., bear_params.position_scalar_short).
    max_position_weight : float
        Per-position cap as fraction of capital (default 8%).
    cash_available : float, optional
        Cash on hand. If provided, blocks trade if margin > cash.
    margin_requirement : float
        Reg T short margin requirement (default 60% of notional).

    Returns
    -------
    qty : int
        Number of shares to short. Returns 0 if any constraint blocks the trade.
    """
    if price <= 0 or stop_pct <= 0 or capital <= 0:
        return 0

    risk_budget = capital * risk_per_trade * position_scalar
    risk_per_share = price * stop_pct
    qty_risk = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0

    cap_dollars = capital * max_position_weight
    qty_cap = int(cap_dollars / price) if price > 0 else 0

    qty = min(qty_risk, qty_cap)
    if qty <= 0:
        return 0

    # Margin check (live-relevant; backtester also enforces this)
    if cash_available is not None:
        margin = price * qty * margin_requirement
        if margin > cash_available:
            return 0

    return max(0, qty)
