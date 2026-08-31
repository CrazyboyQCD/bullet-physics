#!/usr/bin/env python3
"""Build tools/softbody_probe[ _f64].exe from softbody_probe.cpp with MSVC.

The f32 build links target/bullet3-build; the f64 build links
target/bullet3-build-double and defines BT_USE_DOUBLE_PRECISION (btScalar =
double), like tools/build_recorder.py does for bullet_record. Links
BulletSoftBody.lib on top of the three core libs (build it first with:
cmake --build target/bullet3-build --target BulletSoftBody --config Release).

Usage:
    python tools/build_softbody_probe.py [f32|f64|both]
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_recorder import ROOT, lib_dir, msvc_env  # noqa: E402

TOOLS = os.path.join(ROOT, "tools")
SRC = os.path.join(TOOLS, "softbody_probe.cpp")


def libs(precision: str) -> list:
    return [os.path.join(lib_dir(precision), n)
            for n in ("LinearMath.lib", "BulletCollision.lib", "BulletDynamics.lib",
                      "BulletSoftBody.lib")]


def build(precision: str) -> int:
    suffix = "" if precision == "f32" else "_f64"
    exe = os.path.join(TOOLS, f"softbody_probe{suffix}.exe")
    obj = os.path.join(TOOLS, f"softbody_probe{suffix}.obj")
    if not all(os.path.exists(p) for p in libs(precision)):
        print(f"error: Bullet {precision} libs missing (BulletSoftBody included)", file=sys.stderr)
        return 1
    env = msvc_env()
    if env is None:
        return 1
    full_env = dict(os.environ)
    full_env.update(env)
    cmd = ["cl", "/nologo", "/O2", "/EHsc", "/std:c++17",
           f"/Fo{obj}", f"/I{os.path.join(ROOT, 'refs', 'bullet3', 'src')}"]
    if precision == "f64":
        cmd.append("/DBT_USE_DOUBLE_PRECISION")
    cmd += [SRC, f"/Fe:{exe}", "/link", *libs(precision)]
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, env=full_env)
    if os.path.exists(obj):
        os.remove(obj)
    if r.returncode != 0:
        print("error: cl failed", file=sys.stderr)
    return r.returncode


def main() -> int:
    precision = sys.argv[1] if len(sys.argv) > 1 else "both"
    if precision == "both":
        rc = build("f32")
        return rc if rc else build("f64")
    if precision not in ("f32", "f64"):
        print(f"error: unknown precision {precision!r}", file=sys.stderr)
        return 1
    return build(precision)


if __name__ == "__main__":
    sys.exit(main())
