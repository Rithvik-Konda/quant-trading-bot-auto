"""
macro_signal.py — Factor health and VIX regime signals
=======================================================
Computes two signals that the backtester and live trader both use:

1. FACTOR HEALTH: Is the momentum factor currently working?
   - MTUM vs VLUE rolling 20d return differential
   - When momentum is structurally dead for 15+ consecutive days:
     reduce long exposure by 40%, boost CHOPPY allocation
   - Data: 2021 had 43 dead days (-0.74% avg), 2025 had 77 dead days (-0.53% avg)
     Both were our worst years. 2019 had 29 dead days (+0.12% avg) — also weak.

2. VIX SPIKE GATE: Entering during macro shocks kills us.
   - When VIX rises >50% in 10 trading days: halt new longs for 10 days
   - 2025: 12 such events (max +205%). Those entries got stopped in 1-2 days.
   - This fires ~5-15 times per year in bad years, ~0-2 times in good years.

3. VIX TERM STRUCTURE (IVTS): VIX / VIX3M
   - Backwardation (IVTS > 1.0) = acute stress = reduce new longs
   - Contango-to-backwardation flip = re-entry signal after selloffs
   - Data: 2025 had 29 backwardation days (max IVTS 1.274)

All signals are computed from macro_factors.csv (updated daily by daily_scanner.py).
Returns a dict consumed by regime_classifier.py and alpaca_trader.py.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta

MACRO_CSV  = os.path.expanduser('~/ai_trading_bot_v2/macro_factors.csv')
CACHE: dict = {}

# ── Thresholds — all data-derived, no hardcoding ──────────────────────────────
# Factor health: 15-day streak from empirical distribution of dead periods
# VIX spike: 50% in 10 days = ~2σ of normal 10d VIX moves (data: max in good years ~40%)
# IVTS: 1.0 = textbook backwardation definition
FH_DEAD_STREAK   = 15    # consecutive days MTUM underperforms VLUE → momentum structurally dead
VIX_SPIKE_PCT    = 0.50  # 50% VIX rise in 10 trading days → halt new longs
VIX_SPIKE_PAUSE  = 10    # trading days to pause new longs after spike
IVTS_BACKWARDATION = 1.0 # VIX > VIX3M → acute stress
FH_EXPOSURE_CUT  = 0.40  # reduce long exposure by 40% when momentum dead


def load_macro(lookback_days: int = 60) -> pd.DataFrame:
    """Load macro_factors.csv, return last lookback_days rows."""
    global CACHE
    mtime = os.path.getmtime(MACRO_CSV) if os.path.exists(MACRO_CSV) else 0
    if CACHE.get('mtime') == mtime and CACHE.get('df') is not None:
        return CACHE['df']
    df = pd.read_csv(MACRO_CSV, index_col=0, parse_dates=True).sort_index()
    df = df.tail(max(lookback_days, 60))
    CACHE = {'mtime': mtime, 'df': df}
    return df


def compute_signals(as_of_date: str | None = None) -> dict:
    """
    Compute all macro signals as of a given date (or today).
    Returns dict with keys:
        factor_health_score    float  — rolling 20d MTUM-VLUE differential
        momentum_dead          bool   — True if streak >= FH_DEAD_STREAK days
        momentum_dead_streak   int    — consecutive days momentum underperforming
        vix_spike_active       bool   — True if VIX rose >50% in last 10 days
        vix_10d_chg            float  — VIX 10-day % change
        ivts                   float  — VIX / VIX3M ratio
        backwardation          bool   — True if IVTS >= 1.0
        entry_scalar           float  — 0.0 to 1.0 multiplier for new long entries
        signal_summary         str    — human-readable status
    """
    try:
        df = load_macro(lookback_days=80)

        if as_of_date:
            df = df[df.index <= pd.Timestamp(as_of_date)]

        if len(df) < 22:
            return _default_signals('insufficient data')

        # ── Factor health ──────────────────────────────────────────────────────
        df = df.copy()
        df['mtum_20d'] = df['mtum'].pct_change(20)
        df['vlue_20d'] = df['vlue'].pct_change(20)
        df['fh']       = df['mtum_20d'] - df['vlue_20d']
        df['fh_pos']   = df['fh'] > 0

        # Compute streak of consecutive days momentum is UNDERPERFORMING
        # (i.e., fh_pos == False)
        df['fh_neg'] = ~df['fh_pos']
        # Rolling streak: count consecutive True values ending at each row
        streak = 0
        streaks = []
        for val in df['fh_neg']:
            streak = streak + 1 if val else 0
            streaks.append(streak)
        df['dead_streak'] = streaks

        latest = df.iloc[-1]
        fh_score  = float(latest['fh'])
        dead_streak = int(latest['dead_streak'])
        momentum_dead = dead_streak >= FH_DEAD_STREAK

        # ── VIX spike gate ────────────────────────────────────────────────────
        df['vix_10d_chg'] = df['vix'].pct_change(10)
        latest_vix_chg = float(df['vix_10d_chg'].iloc[-1])
        # Check if ANY of the last VIX_SPIKE_PAUSE days had a spike
        recent_spike = bool((df['vix_10d_chg'].iloc[-VIX_SPIKE_PAUSE:] > VIX_SPIKE_PCT).any())

        # ── IVTS ──────────────────────────────────────────────────────────────
        ivts = float(latest['vix'] / latest['vix3m']) if latest['vix3m'] > 0 else 1.0
        backwardation = ivts >= IVTS_BACKWARDATION

        # ── Entry scalar — composite of all three signals ─────────────────────
        # Each signal independently reduces exposure. Multiplicative, not additive.
        # Momentum dead: reduce by FH_EXPOSURE_CUT (40%)
        # VIX spike active: block all new longs (scalar = 0.0)
        # Backwardation: reduce by 30%
        scalar = 1.0
        if recent_spike:
            scalar = 0.0   # hard block — data: day-1/day-2 stops cost $14,400 in 2025
        else:
            if momentum_dead:
                scalar *= (1.0 - FH_EXPOSURE_CUT)   # 0.60
            if backwardation:
                scalar *= 0.70                        # additional 30% cut

        # ── Summary ───────────────────────────────────────────────────────────
        flags = []
        if recent_spike:     flags.append(f'VIX_SPIKE_GATE (10d chg={latest_vix_chg:+.0%})')
        if momentum_dead:    flags.append(f'MOMENTUM_DEAD streak={dead_streak}d')
        if backwardation:    flags.append(f'BACKWARDATION IVTS={ivts:.3f}')
        summary = ' | '.join(flags) if flags else f'CLEAR (fh={fh_score:+.3f} ivts={ivts:.3f})'

        return {
            'factor_health_score':  round(fh_score, 4),
            'momentum_dead':        momentum_dead,
            'momentum_dead_streak': dead_streak,
            'vix_spike_active':     recent_spike,
            'vix_10d_chg':          round(latest_vix_chg, 4),
            'ivts':                 round(ivts, 4),
            'backwardation':        backwardation,
            'entry_scalar':         round(scalar, 3),
            'signal_summary':       summary,
        }

    except Exception as e:
        return _default_signals(f'error: {e}')


def _default_signals(reason: str) -> dict:
    return {
        'factor_health_score':  0.0,
        'momentum_dead':        False,
        'momentum_dead_streak': 0,
        'vix_spike_active':     False,
        'vix_10d_chg':          0.0,
        'ivts':                 0.9,
        'backwardation':        False,
        'entry_scalar':         1.0,
        'signal_summary':       reason,
    }


def simulate_historical(csv_path: str | None = None) -> pd.DataFrame:
    """
    Simulate what entry_scalar would have been each day 2017-2026.
    Used to validate signal before wiring into backtester.
    """
    df = load_macro(lookback_days=9999)
    results = []
    for i in range(22, len(df)):
        row_date = df.index[i]
        sig = compute_signals(as_of_date=str(row_date.date()))
        sig['date'] = row_date
        results.append(sig)
    return pd.DataFrame(results).set_index('date')


if __name__ == '__main__':
    # Test current signal
    sig = compute_signals()
    print('CURRENT MACRO SIGNALS:')
    for k, v in sig.items():
        print(f'  {k:<28} {v}')

    print()
    print('HISTORICAL SIMULATION:')
    hist = simulate_historical()
    hist['year'] = hist.index.year

    yr = hist.groupby('year').agg(
        days_blocked=('entry_scalar', lambda x: (x==0).sum()),
        days_reduced=('entry_scalar', lambda x: ((x>0)&(x<1)).sum()),
        days_clear=('entry_scalar', lambda x: (x==1.0).sum()),
        avg_scalar=('entry_scalar', 'mean'),
        momentum_dead_days=('momentum_dead', 'sum'),
        spike_days=('vix_spike_active', 'sum'),
    )
    print(yr.to_string())
