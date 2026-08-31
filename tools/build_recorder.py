#!/usr/bin/env python3
"""Build tools/bullet_record.exe from bullet_record.cpp with MSVC.

Loads the MSVC 2022 developer environment (vcvars64.bat) into this process by
capturing its `set` output, then compiles the recorder against the static
Bullet libraries in target/bullet3-build. Replaces the old build_recorder.bat;
also callable as a function so tools/parity.py can import it directly.

Exit code: 0 = built, 1 = build failed.

Usage:
    python tools/build_recorder.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
VCVARS = r"D:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
SRC = os.path.join(TOOLS, "bullet_record.cpp")

# Two Bullet builds may coexist: target/bullet3-build (single precision,
# BT_USE_DOUBLE_PRECISION off) and target/bullet3-build-double (double
# precision). Each precision gets its own recorder exe so switching costs one
# rebuild of the (tiny) recorder, not of Bullet.
BUILD_DIRS = {
    "f32": os.path.join(ROOT, "target", "bullet3-build"),
    "f64": os.path.join(ROOT, "target", "bullet3-build-double"),
}
EXES = {"f32": os.path.join(TOOLS, "bullet_record.exe"),
        "f64": os.path.join(TOOLS, "bullet_record_f64.exe")}
OBJS = {"f32": os.path.join(TOOLS, "bullet_record.obj"),
        "f64": os.path.join(TOOLS, "bullet_record_f64.obj")}

# Perf runner (tools/bullet_perf.cpp vs tools/bullet_perf_scenes.h); f64-only
# by design — performance targets the default f32 build (docs/bullet-perf.md).
PERF_SRC = os.path.join(TOOLS, "bullet_perf.cpp")
PERF_EXE = os.path.join(TOOLS, "bullet_perf.exe")


def lib_dir(precision):
    return os.path.join(BUILD_DIRS[precision], "lib", "Release")


def libs(precision):
    return [os.path.join(lib_dir(precision), n)
            for n in ("LinearMath.lib", "BulletCollision.lib", "BulletDynamics.lib")]


def msvc_env():
    """Return the MSVC environment (PATH/INCLUDE/LIB/...) by running
    vcvars64.bat and capturing `set`. Keys are ASCII; values are decoded with
    the system ANSI codepage (mbcs) so non-ASCII paths survive.

    A tiny temporary batch file wraps the call: `cmd /c "..."` quote-nesting
    rules are brittle when the vcvars path contains spaces, while a .bat run
    by cmd has no such ambiguity."""
    tmp = os.path.join(TOOLS, "_vcvars_env.bat")
    with open(tmp, "w", encoding="mbcs") as f:
        f.write(f'@call "{VCVARS}" >nul\n@set\n')
    try:
        r = subprocess.run(["cmd", "/c", tmp], capture_output=True, encoding="mbcs", errors="replace")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    if r.returncode != 0:
        print(f"error: vcvars64.bat failed: {r.stderr}", file=sys.stderr)
        return None
    env = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v
    return env


def build_perf():
    """Compile bullet_perf.cpp -> bullet_perf.exe against the f32 Bullet libs
    (the perf harness measures the default precision only). Returns exit code.
    Also callable from tools/perf.py."""
    if not os.path.exists(VCVARS):
        print(f"error: vcvars64.bat not found at {VCVARS}", file=sys.stderr)
        return 1
    for lib in libs("f32"):
        if not os.path.exists(lib):
            print(f"error: {lib} missing; run tools/parity.py first (it builds "
                  f"the Bullet libs), or build them via CMake", file=sys.stderr)
            return 1

    env = msvc_env()
    if env is None:
        return 1
    full_env = dict(os.environ)
    full_env.update(env)

    obj = os.path.join(TOOLS, "bullet_perf.obj")
    # The Gimpact headers (btGImpactShape.h et al.) live in the Gimpact
    # subfolder, not next to the other collision headers.
    gimpact_inc = os.path.join(ROOT, "refs", "bullet3", "src", "BulletCollision", "Gimpact")
    cmd = ["cl", "/nologo", "/O2", "/EHsc", "/std:c++17",
           f"/Fo{obj}", f"/I{os.path.join(ROOT, 'refs', 'bullet3', 'src')}",
           f"/I{gimpact_inc}",
           PERF_SRC, f"/Fe:{PERF_EXE}", "/link", *libs("f32")]
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, env=full_env)
    if os.path.exists(obj):
        os.remove(obj)
    if r.returncode != 0:
        print("error: cl failed", file=sys.stderr)
    return r.returncode


def build(precision="f32"):
    """Compile bullet_record.cpp -> bullet_record[_f64].exe against the Bullet
    libs of the given precision. Returns exit code."""
    if precision not in BUILD_DIRS:
        print(f"error: unknown precision {precision!r}", file=sys.stderr)
        return 1
    exe, obj = EXES[precision], OBJS[precision]
    if not os.path.exists(VCVARS):
        print(f"error: vcvars64.bat not found at {VCVARS}", file=sys.stderr)
        return 1
    for lib in libs(precision):
        if not os.path.exists(lib):
            print(f"error: {lib} missing; run tools/parity.py first (it builds "
                  f"the Bullet libs), or build them via CMake", file=sys.stderr)
            return 1

    env = msvc_env()
    if env is None:
        return 1
    full_env = dict(os.environ)
    full_env.update(env)

    cmd = ["cl", "/nologo", "/O2", "/EHsc", "/std:c++17",
           f"/Fo{obj}", f"/I{os.path.join(ROOT, 'refs', 'bullet3', 'src')}",
           SRC, f"/Fe:{exe}", "/link", *libs(precision)]
    # The double-precision Bullet build defines BT_USE_DOUBLE_PRECISION
    # (btScalar = double); the recorder must compile with the same define so
    # its symbols match the double libs.
    if precision == "f64":
        cmd.insert(cmd.index(SRC), "/DBT_USE_DOUBLE_PRECISION")
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, env=full_env)
    # The .obj is a build artifact; remove it so the tools dir stays clean.
    if os.path.exists(obj):
        os.remove(obj)
    if r.returncode != 0:
        print("error: cl failed", file=sys.stderr)
    return r.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "perf":
        sys.exit(build_perf())
    sys.exit(build())
