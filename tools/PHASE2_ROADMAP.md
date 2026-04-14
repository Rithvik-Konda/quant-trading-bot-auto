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

## Phase 2 Step 2: VIX Term Structure Overlay — DROPPED

**Date:** 2026-04-14
**Result:** Did not clear 2σ commit threshold. Reverted.

**Implementation:** Position-sizing scalar based on VIX/VIX3M ratio.
- Calm (ratio < 0.83): 1.10x boost
- Normal (0.83-0.99): 1.00x neutral
- Mild caution (0.99-1.02): 0.90x
- Backwardation (1.02-1.17): 0.70x
- Deep stress (>1.17): 0.50x

**OOS results vs BASELINE_2026_05_step1:**
- ΔCAGR: +0.44% (below 1σ noise floor of 1.0%)
- ΔSharpe: +0.03 (below 1σ of 0.06)
- ΔMaxDD: -0.12% (noise)
- Full-period MaxDD: -16.27% → -15.56% (small improvement)
- 2018 specifically: -16.3% → -15.6%

**Diagnosis:** The signal IS real (t=5.09 at 20d forward returns) and
test cases fired correctly on historical stress events (2018 vol crash,
2020 COVID, 2022, 2024 yen carry). But the position-sizing application
doesn't translate to enough CAGR improvement to clear noise floor.

**Lesson:** A real signal applied through the wrong mechanism can fail
to deliver meaningful improvement. VIX term structure is probably better
suited as:
1. An entry filter (skip new entries when ratio > 1.10) — Phase 5
2. A headline risk gate for the robustness layer — Phase 5
3. NOT as a continuous position-size scalar — does not work for us

**Action:** Reverted backtester_v2_phase2step1.py. Moving to Step 3
(CHOPPY concentration throttle, Cohen's d=0.72) which has a stronger
prior and clearer mechanism.

**Park for Phase 5:** VIX term structure overlay as a hard entry gate
when ratio > 1.10. Costs nothing to add as a filter, can't make CAGR
worse, may meaningfully reduce tail drawdown.

---

## Step 3 — SKIPPED (2026-04-14)

**Original plan:** Train logistic regression on (portfolio_size, portfolio_corr, vix_now) to predict P(stop) for CHOPPY entries. Use as entry filter.

**What we tried:**

Phase A — Build training data
- Initial join lost 56% of rows (162 of 365)
- Diagnosed off-by-one issue: lifecycle records decision_date, trades records execution_date (+1 business day)
- Fixed join: recovered 360 of 365 (98.6%)
- Final IS: 360 rows, 39 stops (10.8%); OOS: 210 rows, 28 stops (13.3%)

Phase B — Three model attempts:

1. Logreg, 3 features (portfolio_size, portfolio_corr, vix_now)
   IS AUC 0.746, OOS AUC 0.531, gap +0.214
   Best threshold: ratio 0.67, blocked PnL +$10,818 (anti-helpful)

2. Logreg, 7 features (added ml_rank_entry, portfolio_rank_mean, portfolio_rank_vel, ann_vol)
   IS AUC 0.756, OOS AUC 0.486, gap +0.271
   Best threshold: ratio 0.43, blocked PnL marginally negative

3. Gradient Boosting, 7 features (max_depth=2, n_est=50)
   IS AUC 0.816, OOS AUC 0.551, gap +0.265
   No profitable threshold found at any cutoff

**Diagnosis:** Three independent attempts with different model classes and feature sets produced consistent IS-OOS gaps of 0.21-0.27. This is a financial non-stationarity problem: the patterns that predict CHOPPY stops in 2017-2021 do not exist in 2022-2025. The 2017-2021 CHOPPY regime (containing 2018 vol crash, 2020 COVID) is structurally different from the 2022-2025 CHOPPY regime (post-rate-hike, AI-led concentration). No model trained on the former generalizes to the latter.

**Directional finding (IS only, does not generalize):**
- portfolio_size: Cohen's d = -0.854 (smaller portfolios stop MORE — opposite of original hypothesis)
- portfolio_corr: Cohen's d = +0.254
- vix_now: Cohen's d = +0.291

**Decision:** Step 3 closed as documented null result. Not abandoned forever — may revisit when:
- Lifecycle data pipeline is rebuilt with cleaner alignment
- More CHOPPY-specific stop data is available (2026+ live trading)
- Phase 6 second engine work reveals different ways to handle CHOPPY underperformance

**Moving to Step 4: early-kill classifier for TRENDING_BULL stops.**
Stronger prior (68% of stops red within 3 days), more training data
(~200+ stop events vs 39), simpler prediction problem (intra-trade
day 1-2 features → eventual stop), no lifecycle CSV dependency.
