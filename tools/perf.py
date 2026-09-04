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
  5. --mem instead records the allocator-accounting memory trend per scene
     (one fresh-world run per engine, sampled every 50 frames; both runners
     report requested bytes, so the curves share one unit).
  6. --update-ledger replaces the numeric section of the scene's entry in
     docs/bullet-perf.md (same discipline as tools/parity.py's ledger).

Exit code: 0 = measurement completed, 2 = pipeline failure. There is no
tolerance verdict: perf numbers are recorded, not gated.

Usage:
    python tools/perf.py [--scene stack_1000|all] [--repeats 5] [--stages]
                         [--mem] [--update-ledger] [--no-build]
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

# 60 fps frame budget split the way realtime guidance suggests: the pure
# engine ceiling (physics owning the whole frame) and the common
# "physics gets half the frame" game budget (Unity/Unreal-style guidance).
FPS_BUDGET_MS = 1000.0 / 60.0       # 16.667 ms
GAME_BUDGET_MS = FPS_BUDGET_MS / 2  # 8.333 ms

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


def parse_mem(text):
    """Parse the shared --mem runner protocol into {frame: row}:

        mem frame <f> live_mb <x> peak_mb <y>
    """
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if parts[:2] == ["mem", "frame"]:
            row = {}
            for i in range(3, len(parts) - 1, 2):
                row[parts[i]] = float(parts[i + 1])
            rows[int(parts[2])] = row
    return rows


def run_bullet_mem(scene):
    r = subprocess.run([build_recorder.PERF_EXE, scene, "--mem"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return None
    return parse_mem(r.stdout)


def run_rust_mem(scene):
    cmd = ["cargo", "run", "--release", "-q", "-p", "bullet",
           "--example", "perf_scenes", "--", scene, "--mem"]
    r = run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return None
    return parse_mem(r.stdout)


def build_mem_report(scene, rust, bullet):
    """Assemble the scene's memory-trend section (auto-generated part of its
    ledger entry)."""
    today = date.today().isoformat()
    lines = [
        f"### Memory trend (auto-generated by tools/perf.py --mem, {today})",
        "",
        "Allocator accounting, requested bytes: Rust wraps the global allocator; "
        "C++ hooks global operator new + btAlignedAllocSetCustomAligned (Bullet's "
        "internal arrays; Windows _aligned_malloc route). CRT/runtime-internal "
        "allocations are outside either count, so the curves share one unit. "
        "live = bytes held at the sample; peak = high-water mark.",
        "",
        "| frame | rust live MB | rust peak MB | bullet live MB | bullet peak MB |",
        "|---|---|---|---|---|",
    ]
    for f in sorted(rust):
        r = rust[f]
        b = bullet.get(f, {})
        lines.append(
            f"| {f} | {r['live_mb']:.3f} | {r['peak_mb']:.3f} | "
            f"{b.get('live_mb', float('nan')):.3f} | "
            f"{b.get('peak_mb', float('nan')):.3f} |")
    return "\n".join(lines) + "\n"


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


def update_ledger_section(header, marker, report_md, append_on_missing=False):
    """Replace the '<marker> ...' section under '<header>' in the ledger; a
    missing header is appended at the end (the classification text below the
    numbers stays untouched). A missing marker inserts the section at the end
    of the scene's block when append_on_missing is set (first --mem run on a
    scene that already has a wall-clock section); otherwise it skips."""
    if not os.path.exists(LEDGER):
        print(f"note: ledger {LEDGER} not found; skipping --update-ledger", file=sys.stderr)
        return
    with open(report_md, "r", encoding="utf-8") as f:
        new_section = f.read().rstrip("\n")
    with open(LEDGER, "r", encoding="utf-8") as f:
        text = f.read()

    scene_pos = text.find(header)
    if scene_pos < 0:
        section = f"\n{header}\n\n{new_section}\n"
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(section)
        print(f"ledger updated: {LEDGER} ({header}, new section)")
        return

    next_scene = text.find("\n## ", scene_pos + len(header))
    end = next_scene if next_scene >= 0 else len(text)
    block = text[scene_pos:end]
    rep_pos = block.find(marker)
    if rep_pos < 0:
        if not append_on_missing:
            print(f"note: no '{marker}' under {header}; skipping", file=sys.stderr)
            return
        # First insertion of this section: park it at the end of the scene's
        # block (before the next scene heading), leaving every existing
        # paragraph (numbers and hand-written analysis) untouched.
        insert_pos = next_scene if next_scene >= 0 else len(text)
        rest = text[insert_pos + 1:] if insert_pos < len(text) else ""
        updated = (text[:insert_pos].rstrip("\n") + "\n\n" + new_section + "\n\n"
                   + rest)
        with open(LEDGER, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f"ledger updated: {LEDGER} ({header}, appended {marker})")
        return
    sec_start = scene_pos + rep_pos
    # The generated report ends at the next hand-written analysis
    # paragraph ("\n**"), the next heading, or the scene section's end —
    # whichever comes first. The old end-only fallback silently ate the
    # hand-written analysis paragraphs under the numbers (three past
    # recurrences of that bug).
    candidates = [pos for pos in (text.find("\n### ", sec_start),
                                  text.find("\n**", sec_start))
                   if 0 <= pos <= end]
    next_heading = min(candidates) if candidates else end
    updated = text[:sec_start] + new_section + "\n\n" + text[next_heading + 1:]
    with open(LEDGER, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"ledger updated: {LEDGER} ({header})")


def update_ledger(scene, report_md):
    update_ledger_section(f"## Scene: {scene}", "### Generated report", report_md)


def scene_names():
    import json
    with open(SCENES_JSON, "r", encoding="utf-8") as f:
        return list(json.load(f)["scenes"].keys())


def sweep_scenes(prefix):
    """Scene names matching '<prefix>_<number>', sorted by the number."""
    tiers = []
    for n in scene_names():
        suffix = n[len(prefix) + 1:]
        if n.startswith(prefix + "_") and suffix.isdigit():
            tiers.append((int(suffix), n))
    tiers.sort()
    return [n for _, n in tiers]


def pass_mark(ms, budget):
    return "pass" if ms <= budget else "FAIL"


def crit_tier(rows, engine, budget):
    """Highest tier whose steady ms/frame stays within budget (None = never;
    engine: 1 = bullet summary, 2 = rust summary)."""
    best = None
    for row in rows:
        if row[engine]["steady_ms"] <= budget:
            best = row[0]
    return best


def fmt_crit(tier):
    return f"N <= {tier}" if tier is not None else "no tier"


def build_sweep_report(prefix, rows, repeats):
    """Assemble the fps-sweep table (auto-generated part of the sweep's
    ledger section)."""
    today = date.today().isoformat()
    lines = [
        f"### Generated sweep table (auto-generated by tools/perf.py --sweep, {today})",
        "",
        f"Prefix `{prefix}`: steady-state ms/frame, last300 window, median of {repeats} repeats per engine.",
        f"Budgets: {FPS_BUDGET_MS:.2f} ms = pure-physics 60 fps ceiling; "
        f"{GAME_BUDGET_MS:.2f} ms = physics at half the frame.",
        "",
        "| N | bullet ms/f | rust ms/f | rust/bullet | bullet @16.7 | rust @16.7 | bullet @8.3 | rust @8.3 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, b, ru in rows:
        bs, rs = b["steady_ms"], ru["steady_ms"]
        ratio = rs / bs if bs > 0 else float("nan")
        lines.append(
            f"| {n} | {bs:.3f} | {rs:.3f} | {ratio:.2f}x | "
            f"{pass_mark(bs, FPS_BUDGET_MS)} | {pass_mark(rs, FPS_BUDGET_MS)} | "
            f"{pass_mark(bs, GAME_BUDGET_MS)} | {pass_mark(rs, GAME_BUDGET_MS)} |")
    lines += [
        "",
        f"Highest tier within budget, 16.7 ms: bullet {fmt_crit(crit_tier(rows, 1, FPS_BUDGET_MS))}, "
        f"rust {fmt_crit(crit_tier(rows, 2, FPS_BUDGET_MS))}; "
        f"8.3 ms: bullet {fmt_crit(crit_tier(rows, 1, GAME_BUDGET_MS))}, "
        f"rust {fmt_crit(crit_tier(rows, 2, GAME_BUDGET_MS))}.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="all",
                    help="scene name from tools/perf_scenes.json, or 'all'")
    ap.add_argument("--sweep", metavar="PREFIX",
                    help="run every '<PREFIX>_<N>' scene and aggregate the "
                         "ball-count vs frame-budget table (e.g. sphere_rain)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="fresh-world runs per engine (median); 1 for a quick pass")
    ap.add_argument("--stages", action="store_true",
                    help="also record the Rust per-stage breakdown (one extra run)")
    ap.add_argument("--mem", action="store_true",
                    help="record the allocator-accounting memory trend per scene "
                         "(one fresh-world run per engine, sampled every 50 frames; "
                         "ignores --repeats; mutually exclusive with --sweep)")
    ap.add_argument("--update-ledger", action="store_true",
                    help="replace the scene's numeric report section in docs/bullet-perf.md")
    ap.add_argument("--no-build", action="store_true",
                    help="skip table/lib/exe/build self-checks")
    args = ap.parse_args()

    names = scene_names()
    if args.mem and args.sweep:
        print("error: --mem and --sweep are mutually exclusive", file=sys.stderr)
        return 2
    if args.sweep:
        scenes = sweep_scenes(args.sweep)
        if not scenes:
            print(f"error: no scenes match prefix {args.sweep!r}; available: {names}",
                  file=sys.stderr)
            return 2
        if args.stages:
            print("note: --stages is ignored in --sweep mode", file=sys.stderr)
    else:
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

    if args.mem:
        for scene in scenes:
            bullet_rows = run_bullet_mem(scene)
            rust_rows = run_rust_mem(scene)
            if not bullet_rows or not rust_rows:
                print(f"error: --mem runner failed for {scene}", file=sys.stderr)
                return 2
            report = build_mem_report(scene, rust_rows, bullet_rows)
            print(f"scene {scene} memory trend (allocator-accounting MB):")
            print(report)
            if args.update_ledger:
                report_md = os.path.join(TARGET, f"mem_{scene}_report.md")
                with open(report_md, "w", encoding="utf-8") as f:
                    f.write(report)
                update_ledger_section(f"## Scene: {scene}", "### Memory trend",
                                      report_md, append_on_missing=True)
        return 0

    rows = []
    for scene in scenes:
        bullet = run_bullet(scene, args.repeats)
        rust = run_rust(scene, args.repeats)
        if bullet is None or rust is None or not bullet[1] or not rust[1]:
            print(f"error: runner failed for {scene}", file=sys.stderr)
            return 2
        b = summarize(*bullet)
        ru = summarize(*rust)
        if args.sweep:
            n = int(scene.rsplit("_", 1)[1])
            rows.append((n, b, ru))
            print(f"  tier {n:>5}: bullet steady {b['steady_ms']:.3f} ms/f | "
                  f"rust steady {ru['steady_ms']:.3f} ms/f")
            continue
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

    if args.sweep:
        print()
        report = build_sweep_report(args.sweep, rows, args.repeats)
        print(report)
        if args.update_ledger:
            report_md = os.path.join(TARGET, f"sweep_{args.sweep}_report.md")
            with open(report_md, "w", encoding="utf-8") as f:
                f.write(report)
            update_ledger_section(f"## Sweep: {args.sweep}", "### Generated sweep table",
                                  report_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
