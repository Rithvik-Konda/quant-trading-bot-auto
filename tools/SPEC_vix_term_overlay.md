# VIX Term Structure Overlay — Implementation Spec

**Status:** SPEC ONLY. Do not implement until noise_floor finishes and σ is known.
**Author:** Phase 1 diagnostic session, 2026-04-12
**Pre-commit hash:** BASELINE_2026_04 (`2796671`)

---

## 1. The finding (from `tools/vix_term_structure.py`)

Tested on 2476 days of aligned SPY + VIX9D + VIX + VIX3M data (2016-2026).

| Test | Result | t-stat |
|---|---|---|
| ratio_30d_3m Q5-Q1 spread @ 5d | +0.44% | **+2.51** |
| ratio_30d_3m Q5-Q1 spread @ 10d | +0.79% | **+3.44** |
| ratio_30d_3m Q5-Q1 spread @ 20d | +1.56% | **+5.09** |
| ratio_9d_30d Q5-Q1 spread @ 10d | +0.62% | +2.61 |
| ratio_9d_30d Q5-Q1 spread @ 20d | +1.31% | +4.20 |
| Binary backwardation 20d fwd | +1.86% | — |
| Binary contango 20d fwd | +1.00% | — |

**Both ratios show monotonic Q1→Q5 progression**, both with t > 4 at 20d. This is a robust, replicated risk-premium signal.

## 2. The mechanism (why it works)

Standard volatility risk premium. When VIX > VIX3M (backwardation), short-term implied vol exceeds long-term. This is associated with:
- Recent vol spike (fear was expensive recently)
- Forward returns elevated as fear unwinds
- Vol sellers being compensated for taking the other side

When VIX < VIX3M (contango), it's the normal calm state. Forward returns are still positive (equity drift) but lower in magnitude.

Academic basis: this is "rolling down the VIX futures curve" in textbook form. Eraker, Wang, others. Has worked since VIX futures were introduced in 2004.

## 3. What the overlay does

A **gross-exposure multiplier** keyed off the 30d/3m VIX ratio. Multiplies position sizes (or skips entries) based on regime.

```
ratio = VIX_today / VIX3M_today

if ratio > 1.00:     position_scalar *= 1.20    (backwardation: scale up)
elif ratio > 0.95:   position_scalar *= 1.00    (neutral: no change)
elif ratio > 0.90:   position_scalar *= 0.90    (contango: slight reduction)
else:                position_scalar *= 0.75    (deep contango: meaningful reduction)
```

**Discipline check on the thresholds:**
- 1.00 is the natural Schelling point (above = backwardation by definition, below = contango)
- 0.95 is the median 30d/3m ratio in our sample (slightly contangoed is "normal")
- 0.90 is a 1-sigma move into contango from the median
- The multipliers (1.20, 1.00, 0.90, 0.75) are NOT fitted to any metric. They're chosen so that the average exposure across days roughly matches the no-overlay baseline (so we don't accidentally grow the book on average).

I want to flag honestly that the multipliers ARE choices, not derived from a model. To make them defensible, we should also test:
- A "binary" version (just 1.20 vs 0.80 split at ratio=1.0)
- A "no-overlay" version as control
- Both vs the BASELINE_2026_04

## 4. Where it plugs into the backtester

Anchor location: `v2/backtester_v2.py`, around line 760 where `_daily_vol_scalar` is computed.

**Current code (line ~750):**
```python
# ── Daily VIX scalar — shared by momentum AND meanrev ────────────
_vix_now = vix_macro["close"].reindex([date], method="ffill")
_vix_level = float(_vix_now.iloc[0]) if len(_vix_now) ...
if _vix_level >= 35.0:
    _daily_vol_scalar = 0.0
elif _vix_level >= 25.0:
    _daily_vol_scalar = 0.5
elif _vix_level >= 20.0:
    _daily_vol_scalar = 0.75
else:
    _daily_vol_scalar = 1.0
```

**Add immediately after (new code):**
```python
# ── VIX term structure multiplier ────────────────────────────────
# Validated on 2016-2026 data: t=5.09 on Q5-Q1 spread at 20d horizon.
# This MULTIPLIES the base vol scalar.
_vix3m_now = vix3m_macro["close"].reindex([date], method="ffill")
_vix3m_level = float(_vix3m_now.iloc[0]) if len(_vix3m_now) and not _vix3m_now.isna().all() else _vix_level
_vts_ratio = _vix_level / _vix3m_level if _vix3m_level > 0 else 1.0
if _vts_ratio > 1.00:    _vts_mult = 1.20
elif _vts_ratio > 0.95:  _vts_mult = 1.00
elif _vts_ratio > 0.90:  _vts_mult = 0.90
else:                    _vts_mult = 0.75
_daily_vol_scalar *= _vts_mult
```

**Also need:** load VIX3M into the backtester's macro data block. Around line 100:
```python
# Existing:
spy_macro, hyg_macro, vix_macro = load_macro_data(cache_dir=macro_cache)

# Add:
import pandas as pd
vix3m_path = os.path.join(macro_cache, "VIX3M_regime.csv")
if os.path.exists(vix3m_path):
    vix3m_macro = pd.read_csv(vix3m_path, index_col=0)
    vix3m_macro.index = pd.to_datetime(vix3m_macro.index, utc=True, errors="coerce")
    vix3m_macro = vix3m_macro.loc[~vix3m_macro.index.isna()]
    if getattr(vix3m_macro.index, "tz", None):
        vix3m_macro.index = vix3m_macro.index.tz_convert("UTC").tz_localize(None)
    vix3m_macro.columns = [c.lower() for c in vix3m_macro.columns]
else:
    # Fallback: use vix_macro itself, ratio always = 1.0, overlay no-ops
    vix3m_macro = vix_macro
```

Total LOC change: ~25 lines, all additive. No existing logic touched.

## 5. The sample window problem

VIX3M only goes back to 2016. Our backtest runs from 2017. **Coverage is fine for OOS** (2022-2026 is fully covered). The first ~9 months of 2017 may have missing VIX3M data depending on yfinance caching — the fallback sets the multiplier to 1.0, so the overlay is a no-op for those days. This is the right behavior.

## 6. Pre-committed expected result

**I am writing this BEFORE running the overlay so I can falsify it.**

Direction of effect: the overlay should slightly INCREASE OOS CAGR and SHARPE.
- Expected ΔCAGR: **+0.5% to +1.5%** annualized
- Expected ΔSharpe: **+0.10 to +0.25**
- Expected ΔMaxDD: **slightly worse to neutral** (because we sometimes scale up before shocks)
- Expected effect on 2022 (BEAR year): roughly neutral (already cuts exposure)
- Expected effect on 2025 (the miss year): *slightly negative* — 2025 was contangoed for most of the year, the overlay would have reduced exposure during a year that needed more
- Expected effect on 2024 (TRENDING_BULL year): positive, because 2024 had more backwardation episodes

**If the result deviates significantly from these directions, something is wrong** and we investigate before accepting.

## 7. Acceptance criteria (against noise floor)

Once we have the noise floor σ from `noise_floor_reparse.py`:

| Outcome | Decision |
|---|---|
| ΔCAGR > 2σ AND ΔSharpe > 2σ | **COMMIT.** Real improvement. |
| ΔCAGR > 1σ but < 2σ | **Run 5 more seeds with overlay on, re-measure.** Suggestive. |
| ΔCAGR within ±1σ | **REVERT.** Indistinguishable from noise. |
| ΔCAGR < -1σ | **REVERT and investigate.** Unexpected direction means bug or wrong hypothesis. |

Sharpe is the more important metric here because the overlay is designed to improve risk-adjusted return, not raw return. CAGR may not move much but Sharpe should.

## 8. How to test

```bash
# Step 1: Make sure VIX3M is cached
python3.11 -c "import yfinance as yf; df = yf.Ticker('^VIX3M').history(period='10y'); df.to_csv('cache_prices/VIX3M_regime.csv')"

# Step 2: Apply the patch (above) to v2/backtester_v2.py on a NEW BRANCH
git checkout -b vix_term_overlay
# ... edit v2/backtester_v2.py with the changes from section 4 ...
git add v2/backtester_v2.py
git commit -m "WIP: VIX term structure overlay"

# Step 3: Run the backtest
python3.11 v2/backtester_v2.py --days 3650 --oos 2>&1 | tee /tmp/vix_overlay.log

# Step 4: Compare against baseline
echo "=== BASELINE ===" && grep "OUT-OF-SAMPLE" -A 6 /tmp/baseline.log
echo "=== WITH OVERLAY ===" && grep "OUT-OF-SAMPLE" -A 6 /tmp/vix_overlay.log

# Step 5: Apply acceptance criteria from section 7
```

## 9. What we do next, regardless of result

- **If COMMIT:** merge to v2-multi-engine, tag as `BASELINE_2026_05a`. Move to next item in Phase 2.
- **If REVERT:** keep the spec doc and the test log for posterity. Move to next item in Phase 2.
- **If suggestive (1-2σ):** run more seeds against both baseline and overlay versions. Do NOT commit until clear signal.

Either way, this is one cleanly-measured experiment, not part of a stack of changes. The point of doing it this way is that we will *know* whether VIX term structure helps, instead of guessing.

## 10. What this is NOT

- Not a magic bullet. Expected impact is modest.
- Not a permanent solution to the 2025 problem. 2025 was a contango year and the overlay would have hurt slightly.
- Not a trade signal — it's a position-sizing modifier on an existing system.
- Not stackable with anything else until we test it alone.

## 11. Lessons we are committed to honoring

- One change at a time
- Pre-commit the expected direction so we can falsify it
- Measure against noise floor σ
- Revert if not significant
- Document the result regardless of outcome

---

**This spec is DONE. Not implemented. Implementation happens AFTER `noise_floor_reparse.py` gives us σ.**
