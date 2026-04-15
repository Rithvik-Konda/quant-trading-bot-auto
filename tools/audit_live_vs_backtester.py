#!/usr/bin/env python3.11
"""
Step 6 Phase A: Audit divergence between live_trader_v2.py and backtester_v2_phase2step1.py.

Produces a complete inventory of where the two files diverge so Step 6 refactor
can be planned with concrete numbers. NO code changes.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIVE = REPO / "live_trader_v2.py"
BACKTEST = REPO / "v2" / "backtester_v2_phase2step1.py"

print("="*78)
print("  STEP 6 PHASE A: live_trader_v2 vs backtester_v2_phase2step1 audit")
print("="*78)
print()

live_src = LIVE.read_text()
bt_src = BACKTEST.read_text()

print(f"live_trader_v2.py:                {len(live_src.splitlines()):>5d} lines, {len(live_src):>7d} chars")
print(f"v2/backtester_v2_phase2step1.py:  {len(bt_src.splitlines()):>5d} lines, {len(bt_src):>7d} chars")
print()

# ─── 1. SHARED IMPORTS — modules both files use ──────────────────────
print("─── 1. SHARED MODULE IMPORTS ─────────────────────────────────────")
def get_imports(src):
    imports = set()
    for line in src.split('\n'):
        line = line.strip()
        if line.startswith('from ') and ' import ' in line:
            mod = line.split('from ')[1].split(' import ')[0].strip()
            imports.add(mod)
        elif line.startswith('import '):
            mod = line.split('import ')[1].split(' as ')[0].split(',')[0].strip()
            imports.add(mod)
    return imports

live_imports = get_imports(live_src)
bt_imports = get_imports(bt_src)
shared = live_imports & bt_imports
live_only = live_imports - bt_imports
bt_only = bt_imports - live_imports

# Filter to project modules only (skip stdlib)
def is_project(mod):
    return not any(mod.startswith(s) for s in ['os', 'sys', 'json', 'time', 'argparse', 'requests',
                                                 'datetime', 'numpy', 'pandas', 'pickle', 'warnings',
                                                 'dataclasses', 'typing', 'pathlib', 'collections',
                                                 're', 'math', 'random'])

shared_proj = sorted([m for m in shared if is_project(m)])
live_only_proj = sorted([m for m in live_only if is_project(m)])
bt_only_proj = sorted([m for m in bt_only if is_project(m)])

print(f"\n  SHARED ({len(shared_proj)}):")
for m in shared_proj:
    print(f"    {m}")
print(f"\n  LIVE-ONLY ({len(live_only_proj)}):")
for m in live_only_proj:
    print(f"    {m}")
print(f"\n  BACKTESTER-ONLY ({len(bt_only_proj)}):")
for m in bt_only_proj:
    print(f"    {m}")
print()

# ─── 2. STRATEGY MODULE USAGE ────────────────────────────────────────
print("─── 2. STRATEGY MODULE USAGE ─────────────────────────────────────")
strat_modules = ['strat_bull', 'strat_chop', 'strat_bear', 'strat_mr',
                 'strategy_trending', 'strategy_choppy', 'strategy_bear', 'strategy_meanrev']
print(f"  {'module':22s}  {'live calls':>12s}  {'bt calls':>12s}")
print(f"  {'-'*22}  {'-'*12}  {'-'*12}")
for sm in strat_modules:
    live_count = len(re.findall(rf'\b{re.escape(sm)}\.', live_src))
    bt_count = len(re.findall(rf'\b{re.escape(sm)}\.', bt_src))
    if live_count + bt_count > 0:
        print(f"  {sm:22s}  {live_count:>12d}  {bt_count:>12d}")
print()

# ─── 3. KELLY SIZING USAGE ───────────────────────────────────────────
print("─── 3. KELLY SIZING USAGE ────────────────────────────────────────")
kelly_keywords = ['compute_kelly', 'kelly_sizer', 'risk_pt', 'half_kelly', 'max_kelly']
print(f"  {'keyword':18s}  {'live':>6s}  {'bt':>6s}")
print(f"  {'-'*18}  {'-'*6}  {'-'*6}")
for kw in kelly_keywords:
    l = len(re.findall(rf'\b{re.escape(kw)}\b', live_src))
    b = len(re.findall(rf'\b{re.escape(kw)}\b', bt_src))
    print(f"  {kw:18s}  {l:>6d}  {b:>6d}")
print()
print("  → If live shows 0 for compute_kelly: live trader does NOT use Kelly sizing.")
print()

# ─── 4. STEP 1 DELETED OVERLAYS — STILL IN LIVE TRADER? ─────────────
print("─── 4. STEP 1 DELETED OVERLAYS — STILL PRESENT IN LIVE? ──────────")
step1_deleted = [
    ('vix_vol_scalar',     ['vix_vol_scalar', 'vol_scalar = 0.0 if vix']),
    ('earnings_exclusion', ['earnings_exclusion', 'earnings_calendar', 'earnings_within']),
    ('biotech_gate',       ['biotech_gate', 'biotech', 'XBI', 'IBB']),
    ('tcps',               ['tcps', 'TCPS', 'trajectory_conditioned']),
    ('conviction_calls',   ['conviction_calls', 'conv_call']),
    ('friday_delay',       ['friday_delay', 'is_friday', 'weekday() == 4']),
    ('vol_cooldown',       ['vol_cooldown', 'cooldown_days']),
]
print(f"  {'overlay':22s}  {'live mentions':>15s}  {'bt mentions':>13s}  {'status':>20s}")
print(f"  {'-'*22}  {'-'*15}  {'-'*13}  {'-'*20}")
for name, patterns in step1_deleted:
    l_count = sum(len(re.findall(rf'{re.escape(p)}', live_src, re.IGNORECASE)) for p in patterns)
    b_count = sum(len(re.findall(rf'{re.escape(p)}', bt_src, re.IGNORECASE)) for p in patterns)
    if l_count > 0 and b_count > 0:
        status = "DIVERGENT (in both)"
    elif l_count > 0:
        status = "LIVE ONLY (DRIFT)"
    elif b_count > 0:
        status = "BT ONLY (gated)"
    else:
        status = "neither"
    print(f"  {name:22s}  {l_count:>15d}  {b_count:>13d}  {status:>20s}")
print()
print("  → 'LIVE ONLY (DRIFT)' means the live trader still uses an overlay we deleted from")
print("    the backtester in Step 1. These are accidental divergences that need to be fixed.")
print()

# ─── 5. KEPT OVERLAYS — IN LIVE TRADER? ─────────────────────────────
print("─── 5. KEPT OVERLAYS (from Step 1 ablation) — IN LIVE? ───────────")
kept = [
    ('quality_gate',  ['quality_gate', 'quality_min', 'q_min']),
    ('spy_5d_filter', ['spy_5d_filter', 'spy_5d', 'SPY_5D']),
    ('cascade_freeze',['cascade_freeze', 'cascade_bear', 'cascade']),
    ('ml_stop_hold',  ['ml_stop_hold', 'ml_stop']),
]
print(f"  {'overlay':22s}  {'live mentions':>15s}  {'bt mentions':>13s}")
print(f"  {'-'*22}  {'-'*15}  {'-'*13}")
for name, patterns in kept:
    l_count = sum(len(re.findall(rf'{re.escape(p)}', live_src, re.IGNORECASE)) for p in patterns)
    b_count = sum(len(re.findall(rf'{re.escape(p)}', bt_src, re.IGNORECASE)) for p in patterns)
    print(f"  {name:22s}  {l_count:>15d}  {b_count:>13d}")
print()
print("  → If live shows 0 for any KEPT overlay: live is missing a beneficial filter.")
print()

# ─── 6. POSITION SIZING DIVERGENCE ──────────────────────────────────
print("─── 6. POSITION SIZING ARITHMETIC ─────────────────────────────────")
# Find all lines containing risk_budget or position sizing math
print("\n  LIVE TRADER sizing lines:")
for i, line in enumerate(live_src.split('\n'), 1):
    if any(kw in line for kw in ['risk_budget', 'risk_pt', 'qty = ', 'INITIAL_CAPITAL * 0.0', 'portfolio * 0.0']):
        if 'def ' in line or '#' in line[:line.find('risk_budget') if 'risk_budget' in line else len(line)]:
            continue
        stripped = line.strip()
        if len(stripped) > 5 and not stripped.startswith('#'):
            print(f"    L{i}: {stripped[:90]}")

print("\n  BACKTESTER sizing lines:")
for i, line in enumerate(bt_src.split('\n'), 1):
    if 'risk_budget' in line and '=' in line:
        stripped = line.strip()
        if len(stripped) > 5 and not stripped.startswith('#'):
            print(f"    L{i}: {stripped[:90]}")
print()

# ─── 7. HARDCODED MAGIC NUMBERS ────────────────────────────────────
print("─── 7. HARDCODED MAGIC NUMBERS IN LIVE TRADER ─────────────────────")
magic_patterns = [
    (r'\b0\.035\b', '3.5% risk per trade'),
    (r'\b0\.10\b', '10% (could be max_kelly or stop)'),
    (r'\b0\.15\b', '15% portfolio cap'),
    (r'\bvix_level\s*>=\s*\d+', 'VIX threshold'),
    (r'\bspy_5d.*<.*-?\d', 'SPY 5d threshold'),
]
for pat, desc in magic_patterns:
    matches = []
    for i, line in enumerate(live_src.split('\n'), 1):
        if re.search(pat, line) and not line.strip().startswith('#'):
            matches.append(f"L{i}: {line.strip()[:80]}")
    if matches:
        print(f"\n  {desc}:")
        for m in matches[:5]:
            print(f"    {m}")
print()

# ─── 8. INTRADAY-SPECIFIC LOGIC (intentional divergence) ───────────
print("─── 8. INTRADAY LOGIC IN LIVE TRADER (intentional) ────────────────")
intraday_keywords = ['get_intraday_bars', 'compute_vwap', 'get_volume_ratio',
                     'get_intraday_confirmation', 'submit_order', 'cancel_all_orders']
print(f"  {'function':28s}  {'live':>6s}  {'bt':>6s}")
print(f"  {'-'*28}  {'-'*6}  {'-'*6}")
for kw in intraday_keywords:
    l = len(re.findall(rf'\b{re.escape(kw)}\b', live_src))
    b = len(re.findall(rf'\b{re.escape(kw)}\b', bt_src))
    print(f"  {kw:28s}  {l:>6d}  {b:>6d}")
print()
print("  → These are LIVE-ONLY and SHOULD stay live-only (intraday execution).")
print()

# ─── 9. SUMMARY ──────────────────────────────────────────────────────
print("="*78)
print("  AUDIT SUMMARY")
print("="*78)
print()
print("Step 6 refactor scope estimate based on this audit:")
print()
print("ACCIDENTAL DIVERGENCES (must fix):")
print("  - Each STEP 1 overlay still in live but deleted from backtester")
print("  - Sizing math (Kelly vs hardcoded 3.5%)")
print("  - Any KEPT overlay missing from live")
print()
print("INTENTIONAL DIVERGENCES (keep separate):")
print("  - Intraday data fetching (live needs it, backtester uses cached daily)")
print("  - Order submission (live calls Alpaca API, backtester just records)")
print("  - VWAP/volume ratio confirmation (live-specific intraday signals)")
print()
print("PROPOSED REFACTOR APPROACH:")
print("  1. Extract sizing into v2/sizing.py — both files import from it")
print("  2. Extract entry filters into v2/entry_filters.py — both files import")
print("  3. Live trader keeps its own intraday I/O wrapper")
print("  4. Backtester loop unchanged, just delegates sizing/filters to shared modules")
print("  5. Regression test: backtester must still produce BASELINE_2026_05_step1 metrics")
print()
print(f"Audit log saved to: /tmp/step6_audit.log")
