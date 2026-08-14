#!/usr/bin/env python3
"""One-command frame-by-frame parity run against Bullet.

Orchestrates the whole pipeline for one scene (docs/parity-harness.md):
  1. Regenerate the scene tables from tools/scenes.json if stale
     (tools/generate_scenes.py -> bullet_scenes.h + scenes_generated.rs).
  2. Self-check the static Bullet libraries (build via CMake if missing;
     --precision f64 builds/uses a separate -DUSE_DOUBLE_PRECISION=ON tree).
  3. Self-check the recorder exe for that precision (rebuild via
     tools/build_recorder.py if missing or stale).
  4. Record the Bullet trajectory (tools/bullet_record[_f64].exe, cwd =
     target/; f64 recordings get a _f64 name suffix).
  5. Record the Rust trajectory (crates/bullet/examples/record_scenes.rs).
  6. Diff every recorded body with tools/compare.py's stats, print a per-body
     report, and optionally update the numeric section of the scene's ledger
     entry in docs/bullet-compare.md (--update-ledger). Each precision has its
     own ledger section ("## Scene: X (f32)" / "(f64)"); missing sections are
     created.

Exit code: 0 = all recorded bodies within tolerance, 1 = exceedances,
2 = pipeline failure.

Usage:
    python tools/parity.py [--scene falling_box] [--precision f32]
                           [--update-ledger] [--no-build]
"""

import argparse
import os
import subprocess
import sys

import build_recorder
import compare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TARGET = os.path.join(ROOT, "target")
LEDGER = os.path.join(ROOT, "docs", "bullet-compare.md")
SCENES_JSON = os.path.join(TOOLS, "scenes.json")
C_HEADER = os.path.join(TOOLS, "bullet_scenes.h")
RS_TABLE = os.path.join(ROOT, "crates", "bullet", "src", "parity_scenes.rs")

# f64 runs use a second Bullet build (target/bullet3-build-double) and a
# separately linked recorder, so the f32 artifacts are never clobbered.
BUILD_DIRS = build_recorder.BUILD_DIRS
BULLET_LIBS = {p: os.path.join(d, "lib", "Release") for p, d in BUILD_DIRS.items()}
EXES = build_recorder.EXES

TOL_POS = 1e-2
TOL_ANG = 1e-2


def run(cmd, **kw):
    """Run a command, echo it, and return the CompletedProcess."""
    print("+", " ".join(cmd))
    return subprocess.run(cmd, **kw)


def ensure_scenes_generated():
    """Regenerate the scene tables when scenes.json or the generator itself is
    newer than either generated file (single source of truth; generated files
    are DO NOT EDIT)."""
    json_mtime = max(os.path.getmtime(SCENES_JSON),
                     os.path.getmtime(os.path.join(TOOLS, "generate_scenes.py")))
    for f in (C_HEADER, RS_TABLE):
        if not os.path.exists(f) or os.path.getmtime(f) < json_mtime:
            print("scene tables stale; regenerating...")
            r = run(["python", os.path.join(TOOLS, "generate_scenes.py")])
            return r.returncode == 0
    return True


def ensure_bullet_libs(precision):
    """Build the static Bullet libs for the given precision if any is missing
    (CMake + MSVC). f64 uses a separate tree configured with
    -DUSE_DOUBLE_PRECISION=ON, so the f32 build is never touched."""
    lib_dir = BULLET_LIBS[precision]
    missing = [n for n in ("LinearMath.lib", "BulletCollision.lib", "BulletDynamics.lib")
               if not os.path.exists(os.path.join(lib_dir, n))]
    if not missing:
        return True
    print(f"missing Bullet static libs ({precision}): {', '.join(missing)}; configuring build...")
    build_dir = BUILD_DIRS[precision]
    extra = ["-DUSE_DOUBLE_PRECISION=ON"] if precision == "f64" else []
    r = run(["cmake", "-S", os.path.join(ROOT, "refs", "bullet3"),
             "-B", build_dir, "-A", "x64",
             "-DCMAKE_CONFIGURATION_TYPES=Release",
             "-DBUILD_CPU_DEMOS=OFF", "-DBUILD_EXTRAS=OFF", "-DBUILD_UNIT_TESTS=OFF",
             "-DBUILD_OPENGL3_DEMOS=OFF", "-DBUILD_PYBULLET=OFF",
             "-DBUILD_BULLET2_DEMOS=OFF", "-DBUILD_BULLET3=OFF",
             "-DBUILD_ENET=OFF", "-DBUILD_CLSOCKET=OFF", "-DINSTALL_LIBS=OFF"] + extra)
    if r.returncode != 0:
        return False
    r = run(["cmake", "--build", build_dir,
             "--config", "Release", "--target", "LinearMath", "BulletCollision", "BulletDynamics"])
    return r.returncode == 0


def ensure_recorder(precision):
    """Rebuild the precision's recorder exe if missing or older than its
    sources (the recorder plus the generated scene header)."""
    exe = EXES[precision]
    sources = [os.path.join(TOOLS, "bullet_record.cpp"), C_HEADER]
    if os.path.exists(exe) and all(os.path.getmtime(exe) >= os.path.getmtime(s) for s in sources):
        return True
    print(f"{os.path.basename(exe)} missing or stale; rebuilding...")
    return build_recorder.build(precision) == 0


def scene_bodies(scene_name):
    """Recorded body names for a scene, straight from tools/scenes.json (the
    single source of truth; the generated tables are derived from it)."""
    import json
    with open(SCENES_JSON, "r", encoding="utf-8") as f:
        scenes = json.load(f)["scenes"]
    spec = scenes.get(scene_name)
    return list(spec["record"]) if spec else None


def record_bullet(scene_name, precision, cwd):
    """Run the Bullet recorder; the output suffix ("bullet" / "bullet_f64")
    keeps the f32 and f64 recordings apart."""
    rec = EXES[precision]
    suffix = "bullet" if precision == "f32" else "bullet_f64"
    with open(os.devnull, "wb") as devnull:
        r = run([rec, scene_name, suffix], cwd=cwd, stdout=devnull, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print(r.stderr.decode(errors="replace"), file=sys.stderr)
        return False
    return True


def record_rust(scene_name, precision, cwd):
    suffix = "rust" if precision == "f32" else "rust_f64"
    cargo = ["cargo", "run", "-p", "bullet", "--example", "record_scenes",
             scene_name, suffix]
    if precision == "f64":
        cargo += ["--features", "f64"]
    with open(os.devnull, "wb") as devnull:
        r = run(cargo, cwd=cwd, stdout=devnull, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print(r.stderr.decode(errors="replace"), file=sys.stderr)
        return False
    return True


def build_report(scene_name, precision, bodies, stats, rows_bullet):
    """Assemble the scene-level markdown report section. `stats` maps each
    body name to the dict returned by compare.compare(); `rows_bullet` maps
    body name to the parsed bullet CSV (for frame timestamps)."""
    today = __import__("datetime").date.today().isoformat()
    lines = [
        f"### Generated report (auto-generated by tools/parity.py, {today})",
        "",
        f"Scene: {scene_name} ({precision}), frames: {stats[bodies[0]]['frames']}",
        "",
        "| Body | Position max dev | Frame (t) | Orientation max dev | Frame | Exceedances |",
        "|---|---|---|---|---|---|",
    ]
    total_exceed = 0
    for b in bodies:
        r = stats[b]
        mp = r["max_pos"]
        ma = r["max_ang"]
        n = len(r["exceed"])
        total_exceed += n
        rows = rows_bullet[b]
        mp_t = rows[mp[1]][0] if 0 <= mp[1] < len(rows) else float("nan")
        ma_t = rows[ma[1]][0] if 0 <= ma[1] < len(rows) else float("nan")
        lines.append(
            f"| {b} | {mp[0]:.6f} | {mp[1]} ({mp_t:.6f}) | {ma[0]:.6f} rad | "
            f"{ma[1]} ({ma_t:.6f}) | {n} |"
        )
    for b in bodies:
        r = stats[b]
        if not r["exceed"]:
            continue
        lines += ["", f"Exceedance frames ({b}):"]
        for i, t, pos, ang in r["exceed"][:20]:
            lines.append(f"- frame {i} (t={t:.6f}): pos={pos:.6f} ang={ang:.6f}")
        if len(r["exceed"]) > 20:
            lines.append(f"- ... and {len(r['exceed']) - 20} more")
    lines += ["", f"Total exceedances: {total_exceed} (pos tol {TOL_POS}, ang tol {TOL_ANG})"]
    return "\n".join(lines) + "\n"


def update_ledger(scene_name, precision, report_md):
    """Replace the numeric '### Generated report ...' section of the scene's
    ledger entry with the freshly generated one. Each precision keeps its own
    section ("## Scene: X (f32)" / "(f64)"); a missing section is created
    before the "## Next scenes" trailer. Classification text below the report
    is left untouched."""
    if not os.path.exists(LEDGER):
        print(f"note: ledger {LEDGER} not found; skipping --update-ledger", file=sys.stderr)
        return
    with open(report_md, "r", encoding="utf-8") as f:
        new_section = f.read().rstrip("\n")
    with open(LEDGER, "r", encoding="utf-8") as f:
        text = f.read()

    start_marker = "### Generated report"
    scene_header = f"## Scene: {scene_name} ({precision})"
    scene_pos = text.find(scene_header)
    if scene_pos < 0:
        # No section yet: create one before the "## Next scenes" trailer (or
        # at the end of the ledger) so the run is recorded even before the
        # manual description/classification is written.
        anchor = text.find("\n## Next scenes")
        trailer = "\n\n" if anchor < 0 else ""
        insert_at = anchor if anchor >= 0 else len(text)
        section = f"{scene_header}\n\n{new_section}\n"
        updated = text[:insert_at] + section + trailer + text[insert_at:]
        with open(LEDGER, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f"ledger updated: {LEDGER} ({scene_name}, {precision}, new section)")
        return

    next_scene = text.find("\n## Scene:", scene_pos + len(scene_header))
    end = next_scene if next_scene >= 0 else len(text)
    block = text[scene_pos:end]

    rep_pos = block.find(start_marker)
    if rep_pos < 0:
        print(f"note: no '{start_marker}' section under {scene_header}; skipping --update-ledger", file=sys.stderr)
        return
    sec_start = scene_pos + rep_pos
    next_heading = text.find("\n### ", sec_start + len(start_marker))
    if next_heading < 0:
        next_heading = end
    updated = text[:sec_start] + new_section + "\n\n" + text[next_heading + 1:]
    with open(LEDGER, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"ledger updated: {LEDGER} ({scene_name}, {precision})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="falling_box", help="scene name from tools/scenes.json")
    ap.add_argument("--precision", default="f32", choices=["f32", "f64"],
                    help="precision pair; f64 needs a BT_USE_DOUBLE_PRECISION Bullet build")
    ap.add_argument("--update-ledger", action="store_true",
                    help="replace the scene's numeric report section in docs/bullet-compare.md")
    ap.add_argument("--no-build", action="store_true", help="skip scene/lib/recorder self-build checks")
    args = ap.parse_args()

    bodies = scene_bodies(args.scene)
    if bodies is None:
        avail = []
        import json as _json
        with open(SCENES_JSON, "r", encoding="utf-8") as f:
            avail = list(_json.load(f)["scenes"].keys())
        print(f"error: unknown scene {args.scene!r}; available: {avail}", file=sys.stderr)
        return 2

    if not args.no_build:
        if not ensure_scenes_generated():
            print("error: failed to regenerate scene tables", file=sys.stderr)
            return 2
        if not ensure_bullet_libs(args.precision):
            print("error: failed to build Bullet static libs", file=sys.stderr)
            return 2
        if not ensure_recorder(args.precision):
            print("error: failed to build the recorder", file=sys.stderr)
            return 2

    os.makedirs(TARGET, exist_ok=True)
    if not record_bullet(args.scene, args.precision, cwd=TARGET):
        return 2
    if not record_rust(args.scene, args.precision, cwd=TARGET):
        return 2

    # Diff every recorded body (bullet vs rust CSV). f64 recordings carry a
    # _f64 suffix so they never clobber the f32 pair.
    bullet_suf = "bullet" if args.precision == "f32" else "bullet_f64"
    rust_suf = "rust" if args.precision == "f32" else "rust_f64"
    stats = {}
    rows_bullet = {}
    for b in bodies:
        bullet_csv = os.path.join(TARGET, f"{args.scene}_{b}_{bullet_suf}.csv")
        rust_csv = os.path.join(TARGET, f"{args.scene}_{b}_{rust_suf}.csv")
        rows_bullet[b] = compare.parse_csv(bullet_csv)
        rust_rows = compare.parse_csv(rust_csv)
        stats[b] = compare.compare(rows_bullet[b], rust_rows, TOL_POS, TOL_ANG)

    # Human-readable summary.
    print(f"scene: {args.scene} ({args.precision})")
    for b in bodies:
        r = stats[b]
        print(f"  {b:8s} pos max={r['max_pos'][0]:.6f} @ frame {r['max_pos'][1]} "
              f"| ang max={r['max_ang'][0]:.6f} rad @ frame {r['max_ang'][1]} "
              f"| exceed={len(r['exceed'])}")

    report_md = os.path.join(TARGET, f"{args.scene}_report.md" if args.precision == "f32"
                             else f"{args.scene}_report_f64.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(build_report(args.scene, args.precision, bodies, stats, rows_bullet))
    print(f"report section: {report_md}")

    if args.update_ledger:
        update_ledger(args.scene, args.precision, report_md)

    exceeded = any(stats[b]["exceed"] for b in bodies)
    if exceeded:
        print(f"EXCEEDANCES: {args.scene} outside tolerance - see report above")
        return 1
    print(f"OK: {args.scene} within tolerance (pos {TOL_POS}, ang {TOL_ANG})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
