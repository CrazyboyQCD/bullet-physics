#!/usr/bin/env python3
"""Numerical comparison for the multibody probe scenes (tolerance-based).

The probe twins print one line per frame (columns after the frame label are
%.6f values). diff is too strict: a value crossing a %.6f rounding boundary
flips the last digit while the trajectories still agree. This script compares
column-wise with a tolerance and reports the first frame any column exceeds
it, plus the final-frame state of both sides.

Usage:
    python tools/mbdiff.py <bullet.txt> <rust.txt> [--tol 1e-4]
"""

import argparse
import sys


def parse(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.split()
        if not parts:
            continue
        label, nums = parts[0], [float(x) for x in parts[1:]]
        rows.append((label, nums))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bullet_csv")
    ap.add_argument("rust_csv")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    a, b = parse(args.bullet_csv), parse(args.rust_csv)
    if len(a) != len(b):
        print(f"length mismatch: {len(a)} vs {len(b)} lines")
        return 2
    if any(x[0] != y[0] for x, y in zip(a, b)):
        print("label mismatch (frame/phase order differs)")
        return 2
    width = max(len(x[1]) for x in a)
    if any(len(x[1]) != width for x in a) or any(len(y[1]) != width for y in b):
        print("column count mismatch")
        return 2

    first_bad = None
    max_diff = [0.0] * width
    for i, ((label, va), (_, vb)) in enumerate(zip(a, b)):
        diffs = [abs(x - y) for x, y in zip(va, vb)]
        for c in range(width):
            max_diff[c] = max(max_diff[c], diffs[c])
        if first_bad is None and any(d > args.tol for d in diffs):
            first_bad = (label, diffs)
    n = len(a)
    ident = sum(
        1 for (label, va), (_, vb) in zip(a, b) if all(x == y for x, y in zip(va, vb))
    )
    print(f"lines={n} bit-identical={ident}")
    print("max per-column diff: " + " ".join(f"{d:.6g}" for d in max_diff))
    if first_bad is None:
        print(f"all frames within tol {args.tol}")
        return 0
    label, diffs = first_bad
    print(f"first frame exceeding tol {args.tol}: {label} diffs=" +
          " ".join(f"{d:.6g}" for d in diffs))
    print(f"final: bullet={a[-1][1]}")
    print(f"final: rust  ={b[-1][1]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
