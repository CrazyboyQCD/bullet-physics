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
EXE = os.path.join(TOOLS, "bullet_record.exe")
OBJ = os.path.join(TOOLS, "bullet_record.obj")
LIB_DIR = os.path.join(ROOT, "target", "bullet3-build", "lib", "Release")
LIBS = [os.path.join(LIB_DIR, n) for n in ("LinearMath.lib", "BulletCollision.lib", "BulletDynamics.lib")]


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


def build():
    """Compile bullet_record.cpp -> bullet_record.exe. Returns exit code."""
    if not os.path.exists(VCVARS):
        print(f"error: vcvars64.bat not found at {VCVARS}", file=sys.stderr)
        return 1
    for lib in LIBS:
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
           f"/Fo{OBJ}", f"/I{os.path.join(ROOT, 'refs', 'bullet3', 'src')}",
           SRC, f"/Fe:{EXE}", "/link", *LIBS]
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, env=full_env)
    # The .obj is a build artifact; remove it so the tools dir stays clean.
    if os.path.exists(OBJ):
        os.remove(OBJ)
    if r.returncode != 0:
        print("error: cl failed", file=sys.stderr)
    return r.returncode


if __name__ == "__main__":
    sys.exit(build())
