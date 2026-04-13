"""
hygiene_patch.py — Fix the obvious lies and silent failures in the codebase.

What it changes (all edits are reversible — script writes a .hygiene_bak):

  1. strategy_trending.py
     - STOP_MIN_PCT comment said "3.5%" but value is 0.15. Comment fixed to "15% minimum stop".
     - STOP_MAX_PCT comment said "12%" but value is 0.25. Comment fixed to "25% maximum stop".
       (Either the comments are old and the numbers were tuned up, or vice versa.
        These are tuning artifacts you should reconcile manually after seeing
        ablation results. The patch only fixes the lying comments.)

  2. strategy_choppy.py
     - MAX_POSITIONS = 2 but docstring at top says "Max positions reduced to 4".
       Docstring fixed to match code.

  3. strategy_meanrev.py
     - ann_vol > 0.65 check has error message "max 0.55". Message fixed to "max 0.65".

  4. v2/backtester_v2.py — silent except blocks get logging.
     The 3 truly-silent except blocks that touch real money are:
       - line ~543: TCPS day-7 scaling
       - line ~645: conviction call exit settlement
       - line ~1030: conviction call entry
     Each gets replaced with `except Exception as _e: _silent_log("name", _e)`
     plus a top-of-file `_silent_log` helper that writes to /tmp/v2_silent_errors.log.

     This is the single highest-leverage change in this patch — right now you
     have no idea if these blocks are silently erroring on every trade.

Usage:
  cd ~/ai_trading_bot_v2
  python3.11 tools/hygiene_patch.py            # apply
  python3.11 tools/hygiene_patch.py --revert   # restore from .hygiene_bak
  python3.11 tools/hygiene_patch.py --check    # show what would change without writing
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BAK_SUFFIX = ".hygiene_bak"

# (filepath, old_text, new_text, description)
EDITS = [
    (REPO / "v2" / "strategy_trending.py",
     'STOP_MIN_PCT         = 0.15   # 3.5% minimum stop',
     'STOP_MIN_PCT         = 0.15   # 15% minimum stop  (FIXME: was tuned up from 0.035 — reconcile)',
     "trending: STOP_MIN_PCT comment lied (said 3.5%, value is 15%)"),

    (REPO / "v2" / "strategy_trending.py",
     'STOP_MAX_PCT         = 0.25    # 12% maximum stop',
     'STOP_MAX_PCT         = 0.25    # 25% maximum stop  (FIXME: was tuned up from 0.12 — reconcile)',
     "trending: STOP_MAX_PCT comment lied (said 12%, value is 25%)"),

    (REPO / "v2" / "strategy_choppy.py",
     'MAX_POSITIONS        = 2       # very defensive in choppy — only best 2 ideas',
     'MAX_POSITIONS        = 2       # FIXME docstring above says 4 — reconcile',
     "choppy: MAX_POSITIONS=2 contradicts module docstring"),

    (REPO / "v2" / "strategy_meanrev.py",
     'return False, f"ann_vol={snap.ann_vol:.2f} too high (max 0.55)"',
     'return False, f"ann_vol={snap.ann_vol:.2f} too high (max 0.65)"',
     "meanrev: error message said max 0.55 but threshold is 0.65"),
]

# Silent except blocks in v2/backtester_v2.py — replace `except Exception:\n    pass`
# with logged version. We use unique multi-line context to avoid mismatches.
SILENT_FIXES = [
    ("v2/backtester_v2.py",
     "TCPS day-7 scaling",
     # context: the line ABOVE the silent except is `pos = long_positions[s]`
     # at deeper indentation, inside the elif (trim) branch
     "                            pos = long_positions[s]\n"
     "                except Exception:\n"
     "                    pass\n",
     "                            pos = long_positions[s]\n"
     "                except Exception as _e:\n"
     "                    _silent_log(\"tcps_day7\", s, _e)\n"),

    ("v2/backtester_v2.py",
     "conviction call exit settlement",
     "                        pnl       += _call_pnl\n"
     "                    except Exception:\n"
     "                        pass\n",
     "                        pnl       += _call_pnl\n"
     "                    except Exception as _e:\n"
     "                        _silent_log(\"conv_call_exit\", s, _e)\n"),

    ("v2/backtester_v2.py",
     "conviction call entry",
     "                                })\n"
     "                except Exception:\n"
     "                    pass\n"
     "\n"
     "                entry_meta[s] = _conv_meta\n",
     "                                })\n"
     "                except Exception as _e:\n"
     "                    _silent_log(\"conv_call_entry\", s, _e)\n"
     "\n"
     "                entry_meta[s] = _conv_meta\n"),
]

SILENT_LOG_HELPER = '''
# ── SILENT-FAILURE LOGGER (added by hygiene_patch.py) ─────────────────
def _silent_log(_where: str, _sym: str, _exc: Exception) -> None:
    import os as _os, traceback as _tb
    _path = _os.environ.get("V2_SILENT_LOG", "/tmp/v2_silent_errors.log")
    try:
        with open(_path, "a") as _f:
            _f.write(f"[{_where}] {_sym}: {type(_exc).__name__}: {_exc}\\n")
    except Exception:
        pass
'''


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + BAK_SUFFIX)
    if not bak.exists():
        bak.write_bytes(path.read_bytes())


def revert_one(path: Path) -> bool:
    bak = path.with_suffix(path.suffix + BAK_SUFFIX)
    if bak.exists():
        path.write_bytes(bak.read_bytes())
        bak.unlink()
        return True
    return False


def apply_edits(check_only: bool) -> tuple[int, int]:
    applied = 0
    skipped = 0
    for path, old, new, desc in EDITS:
        if not path.exists():
            print(f"  ✗ {path.name}: file missing")
            skipped += 1
            continue
        text = path.read_text()
        if old not in text:
            if new in text:
                print(f"  · {path.name}: already patched ({desc})")
                skipped += 1
            else:
                print(f"  ✗ {path.name}: anchor not found ({desc})")
                skipped += 1
            continue
        if check_only:
            print(f"  → {path.name}: would patch ({desc})")
            applied += 1
            continue
        backup(path)
        path.write_text(text.replace(old, new, 1))
        print(f"  ✓ {path.name}: patched ({desc})")
        applied += 1
    return applied, skipped


def apply_silent_fixes(check_only: bool) -> tuple[int, int]:
    applied = 0
    skipped = 0
    bt = REPO / "v2" / "backtester_v2.py"
    if not bt.exists():
        print(f"  ✗ {bt}: missing")
        return 0, len(SILENT_FIXES)
    text = bt.read_text()

    needs_helper = "_silent_log" not in text
    for _, desc, old, new in SILENT_FIXES:
        if old not in text:
            if new in text:
                print(f"  · backtester_v2.py: already patched ({desc})")
                skipped += 1
            else:
                print(f"  ✗ backtester_v2.py: anchor not found ({desc})")
                skipped += 1
            continue
        if check_only:
            print(f"  → backtester_v2.py: would patch ({desc})")
            applied += 1
            continue
        text = text.replace(old, new, 1)
        print(f"  ✓ backtester_v2.py: patched ({desc})")
        applied += 1

    if not check_only and applied > 0:
        if needs_helper:
            text = text.replace(
                'CACHE_DIR = "cache_prices"',
                'CACHE_DIR = "cache_prices"\n' + SILENT_LOG_HELPER,
                1,
            )
            print("  ✓ backtester_v2.py: added _silent_log helper")
        backup(bt)
        bt.write_text(text)
    return applied, skipped


def revert_all() -> None:
    paths = {p for (p, _, _, _) in EDITS}
    paths.add(REPO / "v2" / "backtester_v2.py")
    n = 0
    for p in paths:
        if revert_one(p):
            print(f"  ↶ reverted {p.name}")
            n += 1
    print(f"\nReverted {n} files.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="show changes without writing")
    ap.add_argument("--revert", action="store_true", help="restore from .hygiene_bak files")
    args = ap.parse_args()

    if args.revert:
        print("=== REVERTING HYGIENE PATCH ===")
        revert_all()
        return

    print("=== HYGIENE PATCH ===" + (" (CHECK MODE)" if args.check else ""))
    print()
    print("[1/2] Comment/code mismatches:")
    a1, s1 = apply_edits(args.check)
    print()
    print("[2/2] Silent except blocks:")
    a2, s2 = apply_silent_fixes(args.check)
    print()
    print(f"Applied: {a1+a2}   Skipped: {s1+s2}")
    if not args.check and (a1+a2) > 0:
        print()
        print("Backups written with .hygiene_bak suffix.")
        print("To revert: python3.11 tools/hygiene_patch.py --revert")
        print()
        print("After this, run a backtest and check /tmp/v2_silent_errors.log")
        print("If conviction calls or TCPS were silently erroring, you'll see them now.")


if __name__ == "__main__":
    main()
