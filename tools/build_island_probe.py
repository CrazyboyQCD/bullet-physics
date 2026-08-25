#!/usr/bin/env python3
"""Build tools/island_probe.exe from island_probe.cpp with MSVC (f32 libs)."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_recorder import ROOT, VCVARS, libs, msvc_env  # noqa: E402


def main() -> int:
    lib_dir = os.path.join(ROOT, "target", "bullet3-build")
    if not all(os.path.exists(p) for p in libs("f32")):
        print("error: Bullet f32 libs missing", file=sys.stderr)
        return 1
    env = msvc_env()
    if env is None:
        return 1
    full_env = dict(os.environ)
    full_env.update(env)
    exe = os.path.join(ROOT, "tools", "island_probe.exe")
    src = os.path.join(ROOT, "tools", "island_probe.cpp")
    obj = os.path.join(ROOT, "tools", "island_probe.obj")
    cmd = ["cl", "/nologo", "/O2", "/EHsc", "/std:c++17",
           f"/Fo{obj}", f"/I{os.path.join(ROOT, 'refs', 'bullet3', 'src')}",
           src, f"/Fe:{exe}", "/link", *libs("f32")]
    r = subprocess.run(cmd, env=full_env)
    if os.path.exists(obj):
        os.remove(obj)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
