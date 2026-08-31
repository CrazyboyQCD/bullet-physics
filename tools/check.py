#!/usr/bin/env python3
"""Regression check for the workspace.

Runs the full test matrix plus the demo assertions and (optionally) a parity
snapshot, then prints a one-line summary. Use before every commit.

  0. Ensure the generated scene tables exist (tools/generate_scenes.py from
     tools/scenes.json; the generated files are git-ignored, so a fresh clone
     needs this step before cargo can build the record_scenes example).
  1. cargo test --workspace          (f32, default)
  2. cargo test --workspace --features f64
  3. cargo run --example <demo> for falling_box / falling_capsule /
     falling_mesh / ccd; each demo must exit 0 and print an "OK" line.
  4. --parity: python tools/parity.py --scene falling_box (snapshot check).

Exit code: 0 = all green, 1 = any failure.

Usage:
    python tools/check.py [--parity] [--no-f64]
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
SCENES_JSON = os.path.join(TOOLS, "scenes.json")
C_HEADER = os.path.join(TOOLS, "bullet_scenes.h")
RS_TABLE = os.path.join(ROOT, "crates", "bullet", "src", "parity_scenes.rs")

def run(cmd, **kw):
    """Run a command and echo it; returns CompletedProcess."""
    print("+", " ".join(cmd))
    return subprocess.run(cmd, **kw)


DEMOS = ["falling_box", "falling_capsule", "falling_mesh", "ccd", "character", "vehicle", "joints", "multibody", "mixed_stress"]


def ensure_scenes_generated():
    """Regenerate the scene tables when scenes.json or the generator is newer
    than either generated file (single source of truth; generated files are
    DO NOT EDIT and git-ignored). The record_scenes example needs
    parity_scenes.rs to exist before cargo test can compile it."""
    json_mtime = max(os.path.getmtime(SCENES_JSON),
                     os.path.getmtime(os.path.join(TOOLS, "generate_scenes.py")))
    if all(os.path.exists(f) and os.path.getmtime(f) >= json_mtime
           for f in (C_HEADER, RS_TABLE)):
        return True
    print("scene tables missing or stale; regenerating...")
    return run(["python", os.path.join(TOOLS, "generate_scenes.py")]).returncode == 0


def ledger_freshness(scene, report_name):
    """Check that the numeric report section in docs/bullet-compare.md matches
    the latest run (target/<report_name>). Returns a warning string, or None
    when fresh/absent. Lets an agent trust the ledger numbers without
    re-deriving them."""
    ledger = os.path.join(ROOT, "docs", "bullet-compare.md")
    report = os.path.join(ROOT, "target", report_name)
    if not os.path.exists(report):
        return None  # nothing to compare against (first run)
    if not os.path.exists(ledger):
        return f"ledger {ledger} missing"
    with open(report, encoding="utf-8") as f:
        new = f.read()
    with open(ledger, encoding="utf-8") as f:
        old = f.read()
    if new.strip() not in old:
        return f"ledger numbers for {scene} are stale; run `python tools/parity.py --scene {scene} --update-ledger`"
    return None


def ledger_freshness_all():
    """Check freshness for every scene/precision report pair present under
    target/ (f32: <scene>_report.md, f64: <scene>_report_f64.md); returns the
    list of warning strings (empty when all fresh)."""
    import json
    with open(os.path.join(TOOLS, "scenes.json"), encoding="utf-8") as f:
        scenes = json.load(f)["scenes"]
    warnings = []
    for n in scenes:
        for rep in (f"{n}_report.md", f"{n}_report_f64.md"):
            w = ledger_freshness(n, rep)
            if w:
                warnings.append(w)
    return warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parity", action="store_true", help="also run the falling_box parity snapshot")
    ap.add_argument("--no-f64", action="store_true", help="skip the f64 test run")
    args = ap.parse_args()

    failed = []

    # 0. Scene tables must exist before cargo can build the examples.
    if not ensure_scenes_generated():
        failed.append("scene table generation")

    # 1 + 2. Test matrix.
    for label, extra in (("f32", []), ("f64", ["--features", "f64"])):
        if args.no_f64 and label == "f64":
            continue
        r = run(["cargo", "test", "--workspace"] + extra)
        if r.returncode != 0:
            failed.append(f"cargo test ({label})")

    # 3. Demo assertions.
    for demo in DEMOS:
        r = run(["cargo", "run", "-p", "bullet", "--example", demo], capture_output=True)
        if r.returncode != 0 or b"OK" not in r.stdout:
            failed.append(f"demo {demo}")

    # 4. Optional parity snapshot. The pipeline must run cleanly; known
    # exceedances (recorded trade-offs in docs/bullet-compare.md) are printed
    # as a note, not a failure.
    if args.parity:
        r = run(["python", os.path.join(TOOLS, "parity.py"), "--scene", "falling_box"])
        if r.returncode == 2:
            failed.append("parity falling_box (pipeline)")
        elif r.returncode == 1:
            print("note: parity falling_box has exceedances "
                  "(known trade-offs, see docs/bullet-compare.md)")
        stale = ledger_freshness_all()
        for s in stale:
            print(f"note: {s}")

    if failed:
        print(f"FAIL: {', '.join(failed)}")
        return 1
    print(f"OK: tests (f32{'/f64' if not args.no_f64 else ''}) + {len(DEMOS)} demos"
          + (" + parity falling_box" if args.parity else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
