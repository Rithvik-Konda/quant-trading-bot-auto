# tools/ — Phase 1 Diagnostic Suite

Built to answer one question: **what in this codebase is actually working
and what is decoration?** Until you know your noise floor, every "win" is
suspect, and you have ~12 stacked overlays whose individual contributions
have never been measured.

## Run order

### Step 0 — clean the workspace (5 min)
```
python3.11 tools/archive_sweep.py --check
python3.11 tools/archive_sweep.py
```
Moves the 80 files in `archive/` out of the working tree to
`~/ai_trading_bot_v2_attic/archive_<timestamp>/`. Nothing is deleted.

### Step 1 — apply hygiene patch (1 min)
```
python3.11 tools/hygiene_patch.py --check    # see what would change
python3.11 tools/hygiene_patch.py            # apply
```
Fixes 4 lying comments and replaces 3 silent `except: pass` blocks with
logging. Backups written to `*.hygiene_bak`. Revert any time with
`--revert`.

### Step 2 — moment of truth (~1 hour)
```
python3.11 v2/backtester_v2.py --days 3650 --oos 2>&1 | tee /tmp/baseline.log
cat /tmp/v2_silent_errors.log
```
Run one backtest after the hygiene patch and check the silent error log.

If `/tmp/v2_silent_errors.log` is empty: your conviction-call overlay and
TCPS scaler are working. Good.

If it has thousands of lines: one or both have been silently failing on
every trade for months. Your "baseline" includes a non-functional component.
The errors will tell you what's broken.

### Step 3 — measure the noise floor (~5 hours, run overnight)
```
python3.11 tools/noise_floor.py --seeds 5
```
Patches `ml_model.py` to use 5 different LightGBM seeds, retrains the
ranker for each, runs the backtest against each, reports the σ of CAGR
across runs.

This is the number you've never had. Without it you cannot tell the
difference between a real improvement and a lucky seed. Anything smaller
than 2σ in the next step is statistical garbage.

### Step 4 — ablate every overlay (~6 hours, run overnight)
```
python3.11 tools/ablation.py --days 3650
```
Runs 14 backtests:
  - 1 baseline (everything on)
  - 13 with one overlay disabled at a time

Reports `ΔCAGR` for each. Read the table against your noise floor:

| ΔCAGR vs baseline | meaning | action |
|---|---|---|
| < −2σ (disabling hurts) | overlay HELPS | keep |
| within ±2σ | indistinguishable from noise | DELETE |
| > +2σ (disabling helps) | overlay HURTS | DELETE IMMEDIATELY |

The 13 overlays gated:
tcps, breakeven, ml_stop_hold, hold_extension, friday_delay, conviction_calls,
vix_vol_scalar, cascade_freeze, vol_cooldown, quality_gate, spy_5d_filter,
biotech_gate, earnings_exclusion.

You can also test a subset: `--only tcps,conviction_calls,friday_delay`.

The patched backtester is generated at `v2/backtester_v2_ablation.py`.
Don't edit it — regenerate with `--rebuild`.

### Step 5 — decide
After steps 3 and 4 you'll have:
1. A real noise floor (e.g. ±1.8% CAGR)
2. A ranked table of which overlays carry weight and which don't

Delete the dead weight. Re-baseline. Then we move to Phase 2 (solving
CHOPPY for real).

## Files

| file | purpose |
|---|---|
| `archive_sweep.py` | move 80 dead files out of working tree |
| `hygiene_patch.py` | fix comment/code mismatches + silent excepts |
| `noise_floor.py` | retrain ranker N seeds, measure CAGR variance |
| `ablation.py` | toggle 13 overlays, measure each contribution |
| `noise_floor_logs/` | per-seed retrain + backtest logs (created at runtime) |
| `ablation_logs/` | per-overlay backtest logs (created at runtime) |

## Things to know

- **Backtester is fully deterministic given trained models.** Running the
  same backtest twice gives the same result to the dollar. The only
  randomness anywhere is the LightGBM seed in `ml_model.py:1303`.
- **Hygiene patch is reversible.** All edits backup to `*.hygiene_bak`.
- **Ablation patch is non-destructive.** It generates a sibling file
  `v2/backtester_v2_ablation.py`. Your real backtester is untouched.
- **Noise floor patches `ml_model.py` in place but restores it in a
  `finally:` block.** If you Ctrl-C in the middle, it still restores.
  If your machine crashes, the original is in `ml_model.py.noise_floor_orig.bak`
  next to your `.joblib` files.
- **All scripts assume CWD = repo root.** They `os.chdir(REPO)` defensively.
