"""
archive_sweep.py — Move the 80 archived experiments out of the working tree.

What it does:
  Moves ai_trading_bot_v2/archive/ → ../ai_trading_bot_v2_attic/archive_<timestamp>/
  
  These 80 files are duplicates of things in v2/, dead experiments, and old
  versions of strategies you've moved on from. Keeping them in-tree means:
    - Every grep returns 2-3x noise
    - IDE indexing wastes cycles on dead code
    - It's easy to accidentally edit the wrong file
    - You're carrying ~80 files of cognitive load you don't need

  This script does NOT delete anything. It moves the directory to a sibling
  location (your home dir). You can always copy stuff back if you need it.

Usage:
  cd ~/ai_trading_bot_v2
  python3.11 tools/archive_sweep.py            # move
  python3.11 tools/archive_sweep.py --check    # show what would move
  python3.11 tools/archive_sweep.py --restore  # symlink back (if you regret it)
"""

from __future__ import annotations
import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = REPO / "archive"
ATTIC_PARENT = REPO.parent / "ai_trading_bot_v2_attic"


def show() -> None:
    if not ARCHIVE.exists():
        print(f"  (no archive/ directory found at {ARCHIVE})")
        return
    files = sorted(ARCHIVE.glob("*.py"))
    print(f"  archive/ contains {len(files)} .py files")
    print(f"  total size: {sum(f.stat().st_size for f in files)/1024:.0f} KB")
    print()
    # Highlight duplicates with v2/
    v2 = REPO / "v2"
    v2_names = {f.name for f in v2.glob("*.py")} if v2.exists() else set()
    dupes = [f.name for f in files if f.name in v2_names]
    if dupes:
        print(f"  ⚠ {len(dupes)} files duplicate names in v2/:")
        for d in dupes:
            print(f"     - {d}")
    print()
    print(f"  would move to: {ATTIC_PARENT}/archive_<timestamp>/")


def move() -> None:
    if not ARCHIVE.exists():
        print(f"  nothing to do — {ARCHIVE} doesn't exist")
        return
    ATTIC_PARENT.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ATTIC_PARENT / f"archive_{stamp}"
    shutil.move(str(ARCHIVE), str(dest))
    print(f"  ✓ moved {ARCHIVE.name}/ → {dest}")
    print()
    print(f"  to restore: cp -r {dest} {ARCHIVE}")
    # Drop a marker so future-you knows what happened
    marker = REPO / ".archive_moved"
    marker.write_text(f"archive/ moved to {dest} at {stamp}\n")


def restore() -> None:
    """Find the most recent archive_* in attic and copy it back."""
    if not ATTIC_PARENT.exists():
        print(f"  no attic at {ATTIC_PARENT}")
        return
    candidates = sorted(ATTIC_PARENT.glob("archive_*"), reverse=True)
    if not candidates:
        print(f"  no archive snapshots in {ATTIC_PARENT}")
        return
    src = candidates[0]
    if ARCHIVE.exists():
        print(f"  {ARCHIVE} already exists — refusing to overwrite")
        return
    shutil.copytree(str(src), str(ARCHIVE))
    print(f"  ✓ restored {src} → {ARCHIVE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    print("=== ARCHIVE SWEEP ===")
    print()
    if args.restore:
        restore()
        return
    if args.check:
        show()
        return
    show()
    print()
    response = input("  proceed with move? [y/N] ").strip().lower()
    if response == "y":
        move()
    else:
        print("  cancelled")


if __name__ == "__main__":
    main()
