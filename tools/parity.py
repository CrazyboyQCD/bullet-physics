#!/usr/bin/env python3
"""One-command frame-by-frame parity run against Bullet.

Orchestrates the whole pipeline for one scene (docs/parity-harness.md):
  1. Regenerate the scene tables from tools/scenes.json if stale
     (tools/generate_scenes.py -> bullet_scenes.h + scenes_generated.rs).
  2. Self-check the static Bullet libraries (build via CMake if missing).
  3. Self-check tools/bullet_record.exe (rebuild via tools/build_recorder.py
     if missing or stale).
  4. Record the Bullet trajectory (tools/bullet_record.cpp, cwd = target/).
  5. Record the Rust trajectory (crates/bullet/examples/record_scenes.rs).
  6. Diff every recorded body with tools/compare.py's stats, print a per-body
     report, and optionally update the numeric section of the scene's ledger
     entry in docs/bullet-compare.md (--update-ledger).

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
BULLET_LIBS = os.path.join(TARGET, "bullet3-build", "lib", "Release")
LEDGER = os.path.join(ROOT, "docs", "bullet-compare.md")
SCENES_JSON = os.path.join(TOOLS, "scenes.json")
C_HEADER = os.path.join(TOOLS, "bullet_scenes.h")
RS_TABLE = os.path.join(ROOT, "crates", "bullet", "src", "parity_scenes.rs")

TOL_POS = 1e-2
TOL_ANG = 1e-2


def run(cmd, **kw):
    """Run a command, echo it, and return the CompletedProcess."""
    print("+", " ".join(cmd))
    return subprocess.run(cmd, **kw)


def ensure_scenes_generated():
    """Regenerate the scene tables when scenes.json is newer than either
    generated file (single source of truth; generated files are DO NOT EDIT)."""
    json_mtime = os.path.getmtime(SCENES_JSON)
    for f in (C_HEADER, RS_TABLE):
        if not os.path.exists(f) or os.path.getmtime(f) < json_mtime:
            print("scene tables stale; regenerating...")
            r = run(["python", os.path.join(TOOLS, "generate_scenes.py")])
            return r.returncode == 0
    return True


def ensure_bullet_libs():
    """Build the static Bullet libs if any is missing (CMake + MSVC)."""
    missing = [n for n in ("LinearMath.lib", "BulletCollision.lib", "BulletDynamics.lib")
               if not os.path.exists(os.path.join(BULLET_LIBS, n))]
    if not missing:
        return True
    print(f"missing Bullet static libs: {', '.join(missing)}; configuring build...")
    r = run(["cmake", "-S", os.path.join(ROOT, "refs", "bullet3"),
             "-B", os.path.join(TARGET, "bullet3-build"), "-A", "x64",
             "-DCMAKE_CONFIGURATION_TYPES=Release",
             "-DBUILD_CPU_DEMOS=OFF", "-DBUILD_EXTRAS=OFF", "-DBUILD_UNIT_TESTS=OFF",
             "-DBUILD_OPENGL3_DEMOS=OFF", "-DBUILD_PYBULLET=OFF",
             "-DBUILD_BULLET2_DEMOS=OFF", "-DBUILD_BULLET3=OFF",
             "-DBUILD_ENET=OFF", "-DBUILD_CLSOCKET=OFF", "-DINSTALL_LIBS=OFF"])
    if r.returncode != 0:
        return False
    r = run(["cmake", "--build", os.path.join(TARGET, "bullet3-build"),
             "--config", "Release", "--target", "LinearMath", "BulletCollision", "BulletDynamics"])
    return r.returncode == 0


def ensure_recorder():
    """Rebuild tools/bullet_record.exe if missing or older than its sources
    (the recorder plus the generated scene header)."""
    exe = os.path.join(TOOLS, "bullet_record.exe")
    sources = [os.path.join(TOOLS, "bullet_record.cpp"), C_HEADER]
    if os.path.exists(exe) and all(os.path.getmtime(exe) >= os.path.getmtime(s) for s in sources):
        return True
    print("bullet_record.exe missing or stale; rebuilding...")
    return build_recorder.build() == 0


def scene_bodies(scene_name):
    """Recorded body names for a scene, straight from tools/scenes.json (the
    single source of truth; the generated tables are derived from it)."""
    import json
    with open(SCENES_JSON, "r", encoding="utf-8") as f:
        scenes = json.load(f)["scenes"]
    spec = scenes.get(scene_name)
    return list(spec["record"]) if spec else None


def record_bullet(scene_name, cwd):
    rec = os.path.join(TOOLS, "bullet_record.exe")
    with open(os.devnull, "wb") as devnull:
        r = run([rec, scene_name, "bullet"], cwd=cwd, stdout=devnull, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print(r.stderr.decode(errors="replace"), file=sys.stderr)
        return False
    return True


def record_rust(scene_name, precision, cwd):
    cargo = ["cargo", "run", "-p", "bullet", "--example", "record_scenes", scene_name, "rust"]
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


def update_ledger(scene_name, report_md):
    """Replace the numeric '### Generated report ...' section of the scene's
    ledger entry with the freshly generated one. Classification text below is
    left untouched."""
    if not os.path.exists(LEDGER):
        print(f"note: ledger {LEDGER} not found; skipping --update-ledger", file=sys.stderr)
        return
    with open(report_md, "r", encoding="utf-8") as f:
        new_section = f.read()
    with open(LEDGER, "r", encoding="utf-8") as f:
        text = f.read()

    start_marker = "### Generated report"
    scene_header = f"## Scene: {scene_name}"
    scene_pos = text.find(scene_header)
    if scene_pos < 0:
        print(f"note: no '{scene_header}' section in {LEDGER}; skipping --update-ledger", file=sys.stderr)
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
    updated = text[:sec_start] + new_section.rstrip("\n") + "\n\n" + text[next_heading + 1:]
    with open(LEDGER, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"ledger updated: {LEDGER} ({scene_name})")


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

    if args.precision == "f64":
        print("note: f64 parity needs a Bullet build with -DUSE_DOUBLE_PRECISION=ON; "
              "current target/bullet3-build is single precision", file=sys.stderr)

    if not args.no_build:
        if not ensure_scenes_generated():
            print("error: failed to regenerate scene tables", file=sys.stderr)
            return 2
        if not ensure_bullet_libs():
            print("error: failed to build Bullet static libs", file=sys.stderr)
            return 2
        if not ensure_recorder():
            print("error: failed to build tools/bullet_record.exe", file=sys.stderr)
            return 2

    os.makedirs(TARGET, exist_ok=True)
    if not record_bullet(args.scene, cwd=TARGET):
        return 2
    if not record_rust(args.scene, args.precision, cwd=TARGET):
        return 2

    # Diff every recorded body (bullet vs rust CSV).
    stats = {}
    rows_bullet = {}
    for b in bodies:
        bullet_csv = os.path.join(TARGET, f"{args.scene}_{b}_bullet.csv")
        rust_csv = os.path.join(TARGET, f"{args.scene}_{b}_rust.csv")
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

    report_md = os.path.join(TARGET, f"{args.scene}_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(build_report(args.scene, args.precision, bodies, stats, rows_bullet))
    print(f"report section: {report_md}")

    if args.update_ledger:
        update_ledger(args.scene, report_md)

    exceeded = any(stats[b]["exceed"] for b in bodies)
    if exceeded:
        print(f"EXCEEDANCES: {args.scene} outside tolerance - see report above")
        return 1
    print(f"OK: {args.scene} within tolerance (pos {TOL_POS}, ang {TOL_ANG})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
