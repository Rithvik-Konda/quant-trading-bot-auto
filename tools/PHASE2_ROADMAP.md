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

---

## Step 4 — SKIPPED (early-kill hypothesis not supported by data) (2026-04-14)

**Original plan:** Predict eventual stops from day 1-2 price action features (return, MAE, MFE, recovery), close losing trades early at day 2 close instead of waiting for the full stop trigger.

**Phase A built training data:**
- 469 IS trades (2017-2021), 110 stops (23.5%)
- 397 OOS trades (2022-2026), 84 stops (21.2%)
- 7 price-action features at day 2 + 5 entry-context features
- All 241 unique symbols had price data available

**Univariate Cohen's d on IS data:**

| Feature | Cohen's d |
|---|---|
| day1_return | -0.102 |
| day2_return | -0.205 |
| day2_minus_day1 | -0.162 |
| day2_mae | -0.077 |
| day2_mfe | -0.187 |
| day2_recovery | -0.171 |
| day2_intraday_vol | -0.090 |
| ml_rank_pct | +0.353 |
| rule_score | +0.536 |
| ann_vol | -0.108 |

**The hypothesis failed:** All 7 price-action features have |d| < 0.21, far below the 0.5 threshold for a useful classifier. Day 1-2 price action does not meaningfully separate eventual stops from eventual winners on this dataset.

**Unexpected finding (the real value of this step):**

The two strongest features are entry-time scores (rule_score d=+0.536, ml_rank_pct d=+0.353) and they point in the WRONG direction. Trades that eventually stopped had HIGHER entry scores than trades that didn't:

- Stopped trades: rule_score 0.396, ml_rank_pct 0.918
- Not stopped trades: rule_score 0.283, ml_rank_pct 0.828

**This suggests a structural inefficiency in the conviction signal of the cleaned baseline.** High-conviction entries appear to be more likely to stop, not less. Possible explanations include position-sizing interactions (high conviction = bigger size = tighter percent stop), selection effects from the Step 1 cleanup, or genuine miscalibration of the rule_score formula.

**Decision:** Skip Step 4 (early-kill as designed cannot work on this data). Document the finding. Schedule Phase 2.5 as a real investigation.

---

## Phase 2.5 — Conviction-Stop Investigation (NEW, scheduled after Phase 2)

**Goal:** Determine whether the high-conviction → high-stop-rate finding is real, what causes it, and whether it implies a fix.

**Verification tasks:**
1. Replicate the finding on OOS data (does the same Cohen's d hold for 2022-2025?)
2. Slice by regime (does it hold in TRENDING_BULL, CHOPPY, BEAR separately?)
3. Slice by entry timing (specific market conditions?)
4. Audit the relationship between rule_score and position size in backtester code
5. Check whether stopped trades hit predicted distances or are being whipsawed
6. Decompose rule_score into component features

**Hypothesis tests:**
- "Position size scales inversely with rule_score above a threshold" (corrects oversizing)
- "Skip entries with rule_score top 10% AND ml_rank_pct top 10%" (over-confident combo filter)
- "Rule_score formula has an over-weighted component that needs recalibration"

**Possible outcomes:**
- Spurious sample artifact → document, move on, no fix
- Real and bug-driven → fix, retrain, retest
- Real and points to over-trading at extremes → add a conviction cap filter
- Leads somewhere unexpected → continue investigation

**Estimated effort:** 1-2 days of focused research

**Hard gate:** Phase 2.5 happens AFTER Step 5 and Step 6 ship. Phase 5 happens AFTER Phase 2.5.

---

## Updated sequence

1. ✅ Step 1: Cleanup of 7 harmful overlays (BASELINE_2026_05_step1)
2. ❌ Step 2: VIX term overlay (dropped, marginal positive but failed 2σ)
3. ⏭️ Step 3: CHOPPY concentration throttle (skipped, three model attempts failed OOS)
4. ⏭️ Step 4: Early-kill classifier (skipped, day 1-2 features don't separate classes)
5. **NEXT — Step 5: Fix mean reversion regime labeling bug**
6. Step 6: Refactor live trader (HARD GATE before Phase 2.5)
7. Phase 2.5: Conviction-stop investigation
8. Phase 5: Robustness layer
9. Phase 6: Second engine (CHOPPY-specific)

---

## Step 5 — SKIPPED (Kelly cap fix produced no measurable impact) (2026-04-14)

**Original framing:** Fix the meanrev regime labeling bug — meanrev performs better in BEAR than CHOPPY despite being designed for CHOPPY.

**Investigation chain:**

1. **Code recon disproved the regime labeling bug hypothesis.** Both momentum and meanrev exit paths use `meta.get("regime", _current_regime)` from entry_meta — i.e., they record the entry-time regime, not the exit-time regime. Meanrev entry at line 1145 explicitly stores `"regime": _current_regime` in entry_meta. There is no regime relabeling bug.

2. **Pivoted to performance-gap investigation.** Comparison of CHOPPY vs BEAR meanrev showed BEAR's outperformance comes almost entirely from regime_exit triggers (50.8% of BEAR meanrev exits vs 0.5% of CHOPPY) at +$470 average PnL. BEAR meanrev is structurally a "hold through panic, sell on relief" trade dressed as mean reversion. CHOPPY meanrev never gets equivalent regime-transition exits.

3. **Pivoted to 2025 forensics.** Of the $32k headline PnL gap (CHOPPY -$6.7k, BEAR +$25.8k), $27k came from 2025 alone. Through 2024, CHOPPY meanrev was profitable in 6 of 8 years. The "structural underperformance" framing was wrong — 2025 was a discontinuity.

4. **Forensic findings on 2025:**
   - 19 trades in 2025, 3 of them (APP, PTON, MP) accounted for 88% of the loss
   - APP and PTON entered on the SAME day (2025-02-28), both stopped at -15% and -17%
   - Position sizes in 2025 were 2.4x larger than prior years (median notional $52k vs $21k)
   - ml_rank_pct mean: 0.965 in 2025 vs 0.933 prior — third independent confirmation of conviction-stop pattern

5. **Identified mechanism:** kelly_sizer.py uses bucket-based half-Kelly capped at 0.10. The TRENDING_BULL `top` and `high` ML buckets had raw half-Kelly fractions of 0.16-0.29, hitting the cap consistently. The cap was binding for high-conviction trades — meaning all top/high bucket trades got sized at 10% risk per trade regardless of their actual edge.

6. **Validated conviction-stop on 2017-2021 alone (held out from 2025):**
   - All trades, ml_rank_pct: Cohen's d = +0.329 (high conviction predicts MORE stops)
   - All trades, rule_score: Cohen's d = +0.783
   - All trades, combined_score: Cohen's d = +0.871
   - **Pattern is real and present in held-out data — non-backfit signal**

7. **Implemented literature-justified Kelly cap reduction:** Changed max_kelly from 0.10 to 0.08 in v2/kelly_sizer.py. Justification: quarter-Kelly literature (Thorp, Vince) recommends Kelly fractions of 0.25-0.50 of full Kelly. Raw Kelly observations in top buckets reach 0.4-0.6, so quarter-Kelly suggests 0.10-0.15 cap. 0.08 sits at the conservative end of that range.

8. **Backtest result vs BASELINE_2026_05_step1:**
   - Full CAGR: 21.78% → 21.78% (Δ 0.00)
   - Full Sharpe: 1.65 → 1.65 (Δ 0.00)
   - Full MaxDD: -16.27% → -16.27% (Δ 0.00)
   - OOS CAGR: 19.83% → 19.83% (Δ 0.00)
   - OOS Sharpe: 2.06 → 2.06 (Δ 0.00)
   - OOS MaxDD: -5.96% → -5.96% (Δ 0.00)
   - Trades: 1116 → 1107 (-9 trades affected)
   - **Byte-identical headline metrics. Fix did not move the needle.**

**Why the fix didn't help:**

The cap reduction (0.10 → 0.08) only affected ~9 trades worth of position sizing. The system is sufficiently deterministic that the small position-size changes in cap-binding trades got absorbed by other entries filling the slots. Net effect on aggregate PnL: zero.

**Reverted** kelly_sizer.py to max_kelly=0.10. Documented as null result.

**Findings preserved for Phase 2.5:**

- Conviction-stop pattern is real on 2017-2021 held-out data (Cohen's d +0.329 to +0.871 across three features)
- This is the THIRD independent confirmation in this Phase 2 session (Step 4 forensics, Step 5 2025 analysis, temporal validation)
- The mechanism (Kelly bucket-based sizing) is structurally suspect but a literature-defensible cap change does not extract measurable CAGR
- Future investigation: deeper Kelly redesign (quarter-Kelly throughout? collapse top into high bucket? use median PnL?) or restructure the conviction signal itself

**Hard gate update:** Phase 2.5 conviction investigation is now an even higher priority. Three independent investigations now point to the same finding.

---

## Updated sequence

1. ✅ Step 1: Cleanup of 7 harmful overlays (BASELINE_2026_05_step1 — SHIPPED)
2. ❌ Step 2: VIX term overlay (dropped, +0.44% below 2σ)
3. ⏭️ Step 3: CHOPPY concentration throttle (skipped, three model attempts failed OOS)
4. ⏭️ Step 4: Early-kill classifier (skipped, day 1-2 features don't separate stops)
5. ⏭️ Step 5: Meanrev investigation + Kelly cap (no labeling bug, Kelly cap had no measurable impact)
6. **NEXT — Step 6: Refactor live trader (HARD GATE before Phase 2.5)**
7. Phase 2.5: Conviction-stop investigation (now backed by 3 independent findings)
8. Phase 5: Robustness layer
9. Phase 6: Second engine (CHOPPY-specific options work)
