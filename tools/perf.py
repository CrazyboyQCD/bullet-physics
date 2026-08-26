#!/usr/bin/env python3
"""One-command performance comparison against Bullet (docs/bullet-perf.md).

Orchestrates the perf pipeline for one scene (or all):
  1. Regenerate the perf scene tables from tools/perf_scenes.json if stale
     (tools/generate_perf_scenes.py -> bullet_perf_scenes.h +
     perf_scenes_table.rs).
  2. Self-check the f32 Bullet static libs (imported from tools/parity.py;
     built via CMake if missing) and tools/bullet_perf.exe (rebuilt via
     tools/build_recorder.py's build_perf).
  3. Build the Rust runner in release (cargo build --release -p bullet
     --example perf_scenes).
  4. Run both runners `--repeats` times (fresh world per repeat), take the
     median per-frame/total/last300 wall-clock time per engine, and print a
     comparison table. --stages additionally re-runs the Rust side once with
     the per-stage step timings enabled (Rust-only instrumentation).
  5. --update-ledger replaces the numeric section of the scene's entry in
     docs/bullet-perf.md (same discipline as tools/parity.py's ledger).

Exit code: 0 = measurement completed, 2 = pipeline failure. There is no
tolerance verdict: perf numbers are recorded, not gated.

Usage:
    python tools/perf.py [--scene stack_1000|all] [--repeats 5] [--stages]
                         [--update-ledger] [--no-build]
"""

import argparse
import os
import statistics
import subprocess
import sys
from datetime import date

import build_recorder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TARGET = os.path.join(ROOT, "target")
LEDGER = os.path.join(ROOT, "docs", "bullet-perf.md")
SCENES_JSON = os.path.join(TOOLS, "perf_scenes.json")
C_HEADER = os.path.join(TOOLS, "bullet_perf_scenes.h")
RS_TABLE = os.path.join(ROOT, "crates", "bullet", "src", "perf_scenes_table.rs")

sys.path.insert(0, TOOLS)
from parity import ensure_bullet_libs  # noqa: E402  (top-level import has no side effects)


def run(cmd, **kw):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, **kw)


def ensure_tables():
    """Regenerate the perf scene tables when perf_scenes.json or the
    generator is newer than either generated file."""
    src_mtime = max(os.path.getmtime(SCENES_JSON),
                    os.path.getmtime(os.path.join(TOOLS, "generate_perf_scenes.py")))
    for f in (C_HEADER, RS_TABLE):
        if not os.path.exists(f) or os.path.getmtime(f) < src_mtime:
            print("perf scene tables stale; regenerating...")
            r = run(["python", os.path.join(TOOLS, "generate_perf_scenes.py")])
            return r.returncode == 0
    return True


def ensure_perf_exe():
    """Rebuild bullet_perf.exe if missing or older than its sources."""
    exe = build_recorder.PERF_EXE
    sources = [build_recorder.PERF_SRC, C_HEADER]
    if os.path.exists(exe) and all(os.path.getmtime(exe) >= os.path.getmtime(s) for s in sources):
        return True
    print(f"{os.path.basename(exe)} missing or stale; rebuilding...")
    return build_recorder.build_perf() == 0


def parse_perf_output(text):
    """Parse the shared runner protocol:

        perf scene <name> bodies <n> frames <f> warmup <w>
        repeat <i> total_ms <x> last300_ms <y>

    Returns (header, [{'total_ms': x, 'last300_ms': y}, ...]).
    """
    header = None
    repeats = []
    for line in text.splitlines():
        parts = line.split()
        if parts[:2] == ["perf", "scene"]:
            header = {"name": parts[2]}
            for i in range(3, len(parts) - 1, 2):
                header[parts[i]] = int(parts[i + 1])
        elif parts[:1] == ["repeat"]:
            row = {}
            # tokens: repeat <i> key value key value ... (skip the index)
            for i in range(2, len(parts) - 1, 2):
                row[parts[i]] = float(parts[i + 1])
            repeats.append(row)
    return header, repeats


def run_bullet(scene, repeats):
    r = subprocess.run([build_recorder.PERF_EXE, scene, str(repeats)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return None
    return parse_perf_output(r.stdout)


def run_rust(scene, repeats, extra=()):
    # --release: perf numbers are meaningless in a debug build. Pass -q so
    # the runner's protocol lines are the only stdout content.
    cmd = ["cargo", "run", "--release", "-q", "-p", "bullet",
           "--example", "perf_scenes", "--", scene, "--repeats", str(repeats), *extra]
    r = run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return None
    return parse_perf_output(r.stdout)


def parse_stages(text):
    """Parse `stage <name>_ms <x>` lines into an ordered dict."""
    stages = {}
    for line in text.splitlines():
        parts = line.split()
        if parts[:1] == ["stage"]:
            stages[parts[1]] = float(parts[2])
    return stages


def run_rust_stages(scene):
    cmd = ["cargo", "run", "--release", "-q", "-p", "bullet",
           "--example", "perf_scenes", "--", scene, "--stages"]
    r = run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return None
    return parse_stages(r.stdout)


def summarize(header, repeats):
    """Median total/last300/per-frame over the repeats of one engine (plus
    the header metadata the report needs)."""
    frames = max(header["frames"], 1)
    total = statistics.median(r["total_ms"] for r in repeats)
    last300 = statistics.median(r["last300_ms"] for r in repeats)
    window = min(300, frames)
    return {"total_ms": total, "last300_ms": last300,
            "per_frame_ms": total / frames,
            "steady_ms": last300 / window,
            "bodies": header.get("bodies"), "frames": header["frames"],
            "warmup": header.get("warmup")}


def build_report(scene, bullet, rust, stages):
    """Assemble the scene's markdown report section (auto-generated part of
    its ledger entry)."""
    today = date.today().isoformat()
    frames = bullet["frames"]
    lines = [
        f"### Generated report (auto-generated by tools/perf.py, {today})",
        "",
        f"Scene: {scene} (f32), bodies: {bullet['bodies']}, frames: {frames}, "
        f"warmup: {bullet['warmup']}, repeats median",
        "",
        "| Engine | per-frame median (ms) | total median (ms) | last300 median (ms) |",
        "|---|---|---|---|",
        f"| bullet | {bullet['per_frame_ms']:.3f} | {bullet['total_ms']:.1f} | "
        f"{bullet['last300_ms']:.1f} |",
        f"| rust | {rust['per_frame_ms']:.3f} | {rust['total_ms']:.1f} | "
        f"{rust['last300_ms']:.1f} |",
        f"| rust/bullet | {rust['per_frame_ms'] / bullet['per_frame_ms']:.2f}x | | |",
    ]
    if stages:
        total = stages.get("total_ms", 0.0) or 1.0
        lines += ["", f"Rust stage breakdown (cumulative over {frames} frames):", ""]
        for name, ms in stages.items():
            if name == "total_ms":
                continue
            key = name[:-3] if name.endswith("_ms") else name
            lines.append(f"- {key}: {ms:.1f} ms ({100.0 * ms / total:.1f}%)")
    return "\n".join(lines) + "\n"


def update_ledger(scene, report_md):
    """Replace the numeric '### Generated report ...' section of the scene's
    ledger entry; a missing '## Scene: X' section is created at the end (the
    classification text below the report stays untouched)."""
    if not os.path.exists(LEDGER):
        print(f"note: ledger {LEDGER} not found; skipping --update-ledger", file=sys.stderr)
        return
    with open(report_md, "r", encoding="utf-8") as f:
        new_section = f.read().rstrip("\n")
    with open(LEDGER, "r", encoding="utf-8") as f:
        text = f.read()

    scene_header = f"## Scene: {scene}"
    scene_pos = text.find(scene_header)
    if scene_pos < 0:
        section = f"\n{scene_header}\n\n{new_section}\n"
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(section)
        print(f"ledger updated: {LEDGER} ({scene}, new section)")
        return

    next_scene = text.find("\n## Scene:", scene_pos + len(scene_header))
    end = next_scene if next_scene >= 0 else len(text)
    block = text[scene_pos:end]
    rep_pos = block.find("### Generated report")
    if rep_pos < 0:
        print(f"note: no generated report under {scene_header}; skipping", file=sys.stderr)
        return
    sec_start = scene_pos + rep_pos
    next_heading = text.find("\n### ", sec_start)
    if next_heading < 0 or next_heading > end:
        next_heading = end
    updated = text[:sec_start] + new_section + "\n\n" + text[next_heading + 1:]
    with open(LEDGER, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"ledger updated: {LEDGER} ({scene})")


def scene_names():
    import json
    with open(SCENES_JSON, "r", encoding="utf-8") as f:
        return list(json.load(f)["scenes"].keys())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="all",
                    help="scene name from tools/perf_scenes.json, or 'all'")
    ap.add_argument("--repeats", type=int, default=3,
                    help="fresh-world runs per engine (median); 1 for a quick pass")
    ap.add_argument("--stages", action="store_true",
                    help="also record the Rust per-stage breakdown (one extra run)")
    ap.add_argument("--update-ledger", action="store_true",
                    help="replace the scene's numeric report section in docs/bullet-perf.md")
    ap.add_argument("--no-build", action="store_true",
                    help="skip table/lib/exe/build self-checks")
    args = ap.parse_args()

    names = scene_names()
    scenes = names if args.scene == "all" else [args.scene]
    for s in scenes:
        if s not in names:
            print(f"error: unknown scene {s!r}; available: {names}", file=sys.stderr)
            return 2

    if not args.no_build:
        if not ensure_tables():
            print("error: failed to regenerate the perf scene tables", file=sys.stderr)
            return 2
        if not ensure_bullet_libs("f32"):
            print("error: failed to build the Bullet static libs", file=sys.stderr)
            return 2
        if not ensure_perf_exe():
            print("error: failed to build bullet_perf.exe", file=sys.stderr)
            return 2
        r = run(["cargo", "build", "--release", "-p", "bullet",
                 "--example", "perf_scenes"], cwd=ROOT)
        if r.returncode != 0:
            print("error: failed to build the Rust perf runner", file=sys.stderr)
            return 2

    for scene in scenes:
        bullet = run_bullet(scene, args.repeats)
        rust = run_rust(scene, args.repeats)
        if bullet is None or rust is None or not bullet[1] or not rust[1]:
            print(f"error: runner failed for {scene}", file=sys.stderr)
            return 2
        b = summarize(*bullet)
        ru = summarize(*rust)
        print(f"scene {scene}: bodies {bullet[0]['bodies']} frames {bullet[0]['frames']}")
        print(f"  bullet  per-frame {b['per_frame_ms']:.3f} ms "
              f"(total {b['total_ms']:.1f} ms, last300 {b['last300_ms']:.1f} ms)")
        print(f"  rust    per-frame {ru['per_frame_ms']:.3f} ms "
              f"(total {ru['total_ms']:.1f} ms, last300 {ru['last300_ms']:.1f} ms)")
        print(f"  ratio   rust/bullet {ru['per_frame_ms'] / b['per_frame_ms']:.2f}x")

        stages = None
        if args.stages:
            stages = run_rust_stages(scene)
            if stages:
                total = stages.get("total_ms", 1.0) or 1.0
                for name, ms in stages.items():
                    if name != "total_ms":
                        print(f"    stage {name[:-3] if name.endswith('_ms') else name}: "
                              f"{ms:.1f} ms ({100.0 * ms / total:.1f}%)")

        if args.update_ledger:
            report_md = os.path.join(TARGET, f"perf_{scene}_report.md")
            with open(report_md, "w", encoding="utf-8") as f:
                f.write(build_report(scene, b, ru, stages))
            update_ledger(scene, report_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
