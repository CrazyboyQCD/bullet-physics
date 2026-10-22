#!/usr/bin/env python3
"""Build tools/inverse_dynamics_probe[ _f64].exe from inverse_dynamics_probe.cpp
with MSVC, linking the real BulletInverseDynamics library.

The BID reference must be built IEEE-precise in the dedicated
target/bullet3-build-id[-double] directories:
    cmake -S refs/bullet3 -B target/bullet3-build-id -DUSE_MSVC_FAST_FLOATINGPOINT=OFF
    cmake --build target/bullet3-build-id --target LinearMath Bullet3Common BulletInverseDynamics --config Release
(and the -double variant with -DUSE_DOUBLE_PRECISION=ON). The main
bullet3-build[-double] rigs stay on the upstream default /fp:fast — bullet3's
CMakeLists turns it on by default, and under /fp:fast MSVC substitutes
approximate sin/cos implementations that diverge from ucrt (and from every
IEEE toolchain) by 1 ulp at some arguments. Bit parity requires the precise
build; the parity/perf ledger reference in the main rigs is untouched.

The f32 build links target/bullet3-build-id; the f64 build links
target/bullet3-build-id-double and defines BT_USE_DOUBLE_PRECISION (which
IDConfig.hpp turns into BT_ID_USE_DOUBLE_PRECISION, btScalar = double).

Usage:
    python tools/build_inverse_dynamics_probe.py [f32|f64|both]
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_recorder import ROOT, msvc_env  # noqa: E402

TOOLS = os.path.join(ROOT, "tools")
SRC = os.path.join(TOOLS, "inverse_dynamics_probe.cpp")


def lib_dir(precision: str) -> str:
    suffix = "-id-double" if precision == "f64" else "-id"
    return os.path.join(ROOT, "target", f"bullet3-build{suffix}", "lib", "Release")


def libs(precision: str) -> list:
    return [os.path.join(lib_dir(precision), n)
            for n in ("LinearMath.lib", "Bullet3Common.lib",
                      "BulletInverseDynamics.lib")]


def build(precision: str) -> int:
    suffix = "" if precision == "f32" else "_f64"
    exe = os.path.join(TOOLS, f"inverse_dynamics_probe{suffix}.exe")
    obj = os.path.join(TOOLS, f"inverse_dynamics_probe{suffix}.obj")
    if not all(os.path.exists(p) for p in libs(precision)):
        print(f"error: Bullet {precision} libs missing (BulletInverseDynamics included)",
              file=sys.stderr)
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
