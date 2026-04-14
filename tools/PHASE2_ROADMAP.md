# Phase 2 Roadmap & Commitments

## Status as of 2026-04-14

### Completed
- [x] Step 1: Cleanup of 7 harmful overlays — committed as BASELINE_2026_05_step1
  - tcps, earnings_exclusion, biotech_gate, vix_vol_scalar
  - conviction_calls, friday_delay, vol_cooldown
  - OOS: 16.91% -> 19.83% CAGR, 1.68 -> 2.06 Sharpe

### Next Steps
- [ ] Step 2: VIX term structure overlay
  - Spec: tools/SPEC_vix_term_overlay.md
  - Validated signal: t=5.09 at 20d
  - Expected: +0.5 to +1.5% CAGR
- [ ] Step 3: CHOPPY concentration throttle
  - Logistic regression on (portfolio_size, correlation, vix)
  - Cohen's d=0.72 measured
  - Expected: +0.5 to +1.0% CAGR
- [ ] Step 4: Early-kill classifier for TRENDING_BULL stops
  - 68% of stops red within 3 days
  - Train on day 1-2 features
  - Expected: +1 to +2% CAGR
- [ ] Step 5: Fix mean reversion regime labeling bug
  - meanrev BEAR > CHOPPY anomaly
  - Likely regime-at-entry persistence bug
  - Expected: +0.5 to +1.0% CAGR
- [ ] Step 6: Refactor live trader to share strategy module with backtester
  - HARD GATE: Phase 5 cannot start until this is done
  - Currently live_trader_v2.py and backtester_v2.py are parallel implementations
  - Goal: extract shared module, both files import from it
  - Estimated: 2-4 days

### Then Phase 5 (Robustness Layer)
- Position sizing caps
- Portfolio drawdown freeze (>8% in 20d)
- Sector concentration cap (max 25%)
- Out-of-distribution detector
- Daily macro circuit breakers
- Stress test against historical disasters
- Conservative defaults audit

## Acknowledged Technical Debt
- Live trader currently runs OLD strategy (with 7 harmful overlays still active)
- Will be addressed in Step 6 batched update
- Paper trading only, so no financial cost
- DO NOT skip Step 6 to get to Phase 5 faster

## Target State at End of Phase 2
- BASELINE_2026_05 locked
- 22-25% OOS CAGR realistic
- Sharpe ~1.95-2.10
- MaxDD ~-5%
- Live trader running same strategy as backtester
