#!/usr/bin/env python3
"""Generate the perf scene tables from tools/perf_scenes.json (single source
of truth) into:

  - tools/bullet_perf_scenes.h            (C++ runner, tools/bullet_perf.cpp)
  - crates/bullet/src/perf_scenes_table.rs (Rust runner, examples/perf_scenes.rs)

The generated files are DO NOT EDIT; tools/perf.py reruns this generator
before every measurement. Unlike the parity generator (generate_scenes.py),
bodies are produced by deterministic generators declared in the JSON (box
grids, mixed-shape rain lattices) and expanded here into flat static tables,
so a 1000-body scene stays a few lines of JSON. Multibody chain scenes carry
their chain parameters instead of a body table (a chain is not a rigid body).

Usage:
    python tools/generate_perf_scenes.py
"""

import json
import math
import os
import sys

from generate_scenes import fmt, mesh_indices, mesh_vertices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES_JSON = os.path.join(ROOT, "tools", "perf_scenes.json")
C_HEADER = os.path.join(ROOT, "tools", "bullet_perf_scenes.h")
# Lives under src/ (not examples/) so cargo does not treat it as a standalone
# example; perf_scenes.rs pulls it in with #[path] (same pattern as the
# parity tables).
RS_TABLE = os.path.join(ROOT, "crates", "bullet", "src", "perf_scenes_table.rs")

GENERATOR_TYPES = {"box_grid", "mixed_rain", "mesh_rain"}
SHAPE_TYPES = {"box", "sphere", "capsule", "mesh"}
SHAPE_ENUM = {"box": "PERF_SHAPE_BOX", "sphere": "PERF_SHAPE_SPHERE",
              "capsule": "PERF_SHAPE_CAPSULE", "mesh": "PERF_SHAPE_MESH"}

# Closed 12-triangle box shell (outward winding): the dynamic-mesh (GImpact)
# body template shared by the mesh_rain/mesh_grid generators. Closed so the
# far side of the shell can catch a tunneling contact (deviations: single
# panels have no recovery), and 12 triangles so the per-pair tree query and
# GJK band work match a real concave load.
SHELL_TRIANGLES = [
    0, 1, 2,  0, 2, 3,   # bottom (-Y)
    4, 6, 5,  4, 7, 6,   # top (+Y)
    0, 5, 1,  0, 4, 5,   # z = -hz (-Z)
    2, 7, 3,  2, 6, 7,   # z = +hz (+Z)
    1, 6, 2,  1, 5, 6,   # x = +hx (+X)
    0, 3, 7,  0, 7, 4,   # x = -hx (-X)
]


def shell_geometry(half):
    """Flat vertex/index buffers for the closed box shell template."""
    hx, hy, hz = half
    verts = [-hx, -hy, -hz, hx, -hy, -hz, hx, -hy, hz, -hx, -hy, hz,
             -hx, hy, -hz, hx, hy, -hz, hx, hy, hz, -hx, hy, hz]
    return verts, list(SHELL_TRIANGLES)


def expand_generator(g):
    """Expand one generator spec into a list of plain body dicts (same shape
    as the hand-written 'bodies' entries). Deterministic: no randomness, the
    body order is fixed by the loops."""
    kind = g["type"]
    out = []
    if kind == "box_grid":
        nx, ny, nz, sp = g["nx"], g["ny"], g["nz"], g["spacing"]
        bx, by, bz = g["base"]
        for k in range(ny):        # up
            for j in range(nz):    # z rows
                for i in range(nx):  # x cols
                    out.append({
                        "name": f"{kind}_{i}_{j}_{k}",
                        "shape": {"type": "box", "half_extents": g["half_extents"]},
                        "position": [bx + i * sp, by + k * sp, bz + j * sp],
                        "mass": g["mass"], "friction": g["friction"],
                        "restitution": g["restitution"],
                    })
    elif kind == "mixed_rain":
        nx, nz, layers = g["nx"], g["nz"], g["layers"]
        sx, sy, sz = g["spacing"]
        bx, by, bz = g["base"]
        m, fr, re = g["mass"], g["friction"], g["restitution"]
        for layer in range(layers):
            for j in range(nz):
                for i in range(nx):
                    which = (i + nz * j + layer) % 3
                    if which == 0:
                        shape = {"type": "box", "half_extents": [g["box_half"]] * 3}
                    elif which == 1:
                        shape = {"type": "sphere", "radius": g["sphere_radius"]}
                    else:
                        shape = {"type": "capsule", "radius": g["capsule_radius"],
                                 "half_height": g["capsule_half_height"]}
                    out.append({
                        "name": f"{kind}_{i}_{j}_{layer}",
                        "shape": shape,
                        "position": [bx + i * sx, by + layer * sy, bz + j * sz],
                        "mass": m, "friction": fr, "restitution": re,
                    })
    elif kind == "mesh_rain":
        # Dynamic triangle-mesh shells (GImpact) dropped in a grid: every
        # shell-vs-terrain / shell-vs-shell pair exercises the mesh-mesh path.
        nx, nz, layers = g["nx"], g["nz"], g["layers"]
        sx, sy, sz = g["spacing"]
        bx, by, bz = g["base"]
        m, fr, re = g["mass"], g["friction"], g["restitution"]
        shell = {"type": "mesh", "shell": g["shell_half"]}
        for layer in range(layers):
            for j in range(nz):
                for i in range(nx):
                    out.append({
                        "name": f"{kind}_{i}_{j}_{layer}",
                        "shape": dict(shell),
                        "position": [bx + i * sx, by + layer * sy, bz + j * sz],
                        "mass": m, "friction": fr, "restitution": re,
                    })
    else:
        raise ValueError(f"unknown generator type {kind!r}")
    return out


def expand_scene(sc):
    """Bodies = hand-written entries + generated ones, in that order."""
    bodies = [dict(b) for b in sc.get("bodies", [])]
    for g in sc.get("generators", []):
        bodies.extend(expand_generator(g))
    return bodies


def mesh_heights(shape):
    """Heights list: either explicit (parity-style) or the deterministic
    wave expression amp*sin(ax*x)*cos(az*z) evaluated on the grid."""
    if "heights" in shape:
        return shape["heights"]
    e = shape["heights_expr"]
    nx, nz = shape["grid"]
    return [e["amp"] * math.sin(x * e["ax"]) * math.cos(z * e["az"])
            for z in range(nz) for x in range(nx)]


def mesh_geometry(shape):
    """(flat vertices, flat indices) for one mesh shape: a height grid
    (static terrain) or the closed box shell template (dynamic shells)."""
    if "shell" in shape:
        return shell_geometry(shape["shell"])
    expanded = dict(shape)
    expanded["heights"] = mesh_heights(shape)
    nx, nz, verts = mesh_vertices(expanded)
    return verts, mesh_indices(nx, nz)


def shape_params(shape):
    """Flatten a shape dict into (type, [params...]) padded to 6 floats;
    mesh bodies carry geometry in a dedicated MeshSpec (params stay zero)."""
    t = shape["type"]
    if t == "box":
        return t, list(shape["half_extents"])
    if t == "sphere":
        return t, [shape["radius"]]
    if t == "capsule":
        return t, [shape["radius"], shape["half_height"]]
    if t == "mesh":
        return t, [0.0] * 6
    raise ValueError(f"unknown shape type {t!r}")


def gen_c(scenes):
    lines = [
        "// Auto-generated by tools/generate_perf_scenes.py from tools/perf_scenes.json - DO NOT EDIT.",
        "#pragma once",
        "",
        '#include "btBulletDynamicsCommon.h"',
        "",
        "enum PerfShapeType { PERF_SHAPE_BOX, PERF_SHAPE_SPHERE, PERF_SHAPE_CAPSULE, PERF_SHAPE_MESH };",
        "",
        "struct PerfMeshSpec {",
        "  int num_vertices;",
        "  const btScalar* vertices;  // num_vertices * 3 (x y z)",
        "  int num_triangles;",
        "  const int* indices;        // num_triangles * 3",
        "};",
        "",
        "struct PerfBodySpec {",
        "  PerfShapeType shape;",
        "  btScalar params[6];   // box: 3 half extents; sphere: r; capsule: r, half_height; mesh: unused",
        "  btScalar pos[3];",
        "  btScalar mass, friction, restitution;  // mass == 0 -> static",
        "  const PerfMeshSpec* mesh;              // non-null only for PERF_SHAPE_MESH",
        "};",
        "",
        "struct PerfChainSpec {  // grid of fixed-base revolute pendulums (0 chains = rigid-only scene)",
        "  int count, cols, links;",
        "  btScalar spacing, base_height, base_mass, link_mass, link_length;",
        "  btScalar initial_angle, collider_radius;  // radians / sphere radius",
        "};",
        "",
        "struct PerfSceneSpec {",
        "  const char* name;",
        "  int dt_num, dt_den, frames, warmup, iterations;",
        "  btScalar gravity[3];",
        "  btScalar bp_min[3], bp_max[3];  // btAxisSweep3 world extent hint",
        "  int bp_kind;                   // 0: btAxisSweep3, 1: btDbvtBroadphase",
        "  int script;                    // 0: plain stepping, 1: arena tick (launcher + dozer)",
        "  int num_bodies;",
        "  const PerfBodySpec* bodies;",
        "  PerfChainSpec chains;",
        "};",
        "",
    ]
    for name, sc in scenes.items():
        bodies = expand_scene(sc)
        # Emit mesh geometry first so the body specs can reference it.
        for b in bodies:
            if b["shape"]["type"] != "mesh":
                continue
            verts, idx = mesh_geometry(b["shape"])
            vs = ", ".join(f"btScalar({fmt(x)})" for x in verts)
            is_ = ", ".join(str(i) for i in idx)
            lines.append(f"static const btScalar k{name}_{b['name']}_verts[] = {{ {vs} }};")
            lines.append(f"static const int k{name}_{b['name']}_idx[] = {{ {is_} }};")
            lines.append(f"static const PerfMeshSpec k{name}_{b['name']}_mesh = "
                         f"{{ {len(verts) // 3}, k{name}_{b['name']}_verts, "
                         f"{len(idx) // 3}, k{name}_{b['name']}_idx }};")
        lines.append(f"static const PerfBodySpec k{name}_bodies[] = {{")
        for b in bodies:
            t, params = shape_params(b["shape"])
            params = (params + [0.0] * 6)[:6]
            p = ", ".join(f"btScalar({fmt(x)})" for x in params)
            pos = ", ".join(f"btScalar({fmt(x)})" for x in b["position"])
            mass = fmt(b.get("mass", 0.0))
            if b.get("mass", 0.0) == 0:
                fr, re = "0.5", "0.0"
            else:
                fr, re = fmt(b.get("friction", 0.5)), fmt(b.get("restitution", 0.0))
            mesh_ref = f"&k{name}_{b['name']}_mesh" if t == "mesh" else "nullptr"
            lines.append(f"  {{{SHAPE_ENUM[t]}, {{{p}}}, {{{pos}}}, "
                         f"btScalar({mass}), btScalar({fr}), btScalar({re}), {mesh_ref}}},")
        if not bodies:
            # MSVC rejects zero-size arrays; chain scenes never read the table
            # (num_bodies == 0), so park one unused placeholder entry.
            lines.append("  {PERF_SHAPE_BOX, {0, 0, 0, 0, 0, 0}, {0, 0, 0}, 0, 0, 0, nullptr},")
        lines.append("};")
        lines.append("")
        ch = sc.get("chains")
        if ch:
            angle = math.radians(ch["initial_angle_deg"])
            chains = (f"{{ {ch['count']}, {ch['cols']}, {ch['links']}, btScalar({fmt(ch['spacing'])}), "
                      f"btScalar({fmt(ch['base_height'])}), btScalar({fmt(ch['base_mass'])}), "
                      f"btScalar({fmt(ch['link_mass'])}), btScalar({fmt(ch['link_length'])}), "
                      f"btScalar({fmt(angle)}), btScalar({fmt(ch['collider_radius'])}) }}")
        else:
            chains = "{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }"
        dt_num, dt_den = sc["dt"]
        g = ", ".join(fmt(x) for x in sc["gravity"])
        lo = ", ".join(fmt(x) for x in sc["broadphase_aabb"][0])
        hi = ", ".join(fmt(x) for x in sc["broadphase_aabb"][1])
        lines.append(f"static const PerfSceneSpec k{name} = {{\"{name}\", {dt_num}, {dt_den}, "
                     f"{sc['frames']}, {sc['warmup']}, {sc['iterations']}, {{{g}}}, "
                     f"{{{lo}}}, {{{hi}}}, {1 if sc['broadphase'] == 'dbvt' else 0}, "
                     f"{1 if sc['script'] == 'arena' else 0}, {len(bodies)}, k{name}_bodies, {chains}}};")
        lines.append("")
    lines.append("static const PerfSceneSpec* kPerfScenes[] = {")
    for name in scenes:
        lines.append(f"  &k{name},")
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def gen_rs(scenes):
    lines = [
        "// Auto-generated by tools/generate_perf_scenes.py from tools/perf_scenes.json - DO NOT EDIT.",
        "// Included by perf_scenes.rs via #[path]; not part of the library.",
        "",
        "#![allow(dead_code)]",
        "#![allow(clippy::approx_constant, clippy::excessive_precision)]",  # wave heights / 60-degree angles legitimately near constants,
        "",
        "#[derive(Clone, Copy)]",
        "pub enum PerfShapeSpec {",
        "    Box([f32; 3]),",
        "    Sphere(f32),",
        "    Capsule(f32, f32),",
        "    Mesh(&'static [f32], &'static [u32]),",
        "}",
        "",
        "pub struct PerfBodySpec {",
        "    pub shape: PerfShapeSpec,",
        "    pub position: [f32; 3],",
        "    pub mass: f32,",
        "    pub friction: f32,",
        "    pub restitution: f32,",
        "}",
        "",
        "pub struct PerfChainSpec {",
        "    pub count: usize,",
        "    pub cols: usize,",
        "    pub links: usize,",
        "    pub spacing: f32,",
        "    pub base_height: f32,",
        "    pub base_mass: f32,",
        "    pub link_mass: f32,",
        "    pub link_length: f32,",
        "    pub initial_angle: f32,",
        "    pub collider_radius: f32,",
        "}",
        "",
        "pub struct PerfSceneSpec {",
        "    pub dt_num: u32,",
        "    pub dt_den: u32,",
        "    pub frames: u32,",
        "    pub warmup: u32,",
        "    pub iterations: u32,",
        "    pub gravity: [f32; 3],",
        "    pub broadphase_dbvt: bool,",
        "    pub arena_script: bool,",
        "    pub bodies: &'static [PerfBodySpec],",
        "    pub chains: Option<PerfChainSpec>,",
        "}",
        "",
        "pub const PERF_SCENES: &[(&str, PerfSceneSpec)] = &[",
    ]
    for name, sc in scenes.items():
        bodies = expand_scene(sc)
        lines.append(f"    (\"{name}\", PerfSceneSpec {{")
        lines.append("        bodies: &[")
        for b in bodies:
            t, params = shape_params(b["shape"])
            if t == "box":
                shape = f"PerfShapeSpec::Box([{', '.join(fmt(x) for x in params)}])"
            elif t == "sphere":
                shape = f"PerfShapeSpec::Sphere({fmt(params[0])})"
            elif t == "capsule":
                shape = f"PerfShapeSpec::Capsule({fmt(params[0])}, {fmt(params[1])})"
            else:  # mesh
                verts, idx = mesh_geometry(b["shape"])
                vs = ", ".join(fmt(x) for x in verts)
                is_ = ", ".join(str(i) for i in idx)
                shape = f"PerfShapeSpec::Mesh(&[{vs}], &[{is_}])"
            pos = f"[{', '.join(fmt(x) for x in b['position'])}]"
            mass = fmt(b.get("mass", 0.0))
            if b.get("mass", 0.0) == 0:
                fr, re = "0.5", "0.0"
            else:
                fr, re = fmt(b.get("friction", 0.5)), fmt(b.get("restitution", 0.0))
            lines.append(f"            PerfBodySpec {{ shape: {shape}, position: {pos}, "
                         f"mass: {mass}, friction: {fr}, restitution: {re} }},")
        lines.append("        ],")
        ch = sc.get("chains")
        if ch:
            angle = math.radians(ch["initial_angle_deg"])
            lines.append(f"        chains: Some(PerfChainSpec {{ count: {ch['count']}, "
                         f"cols: {ch['cols']}, links: {ch['links']}, "
                         f"spacing: {fmt(ch['spacing'])}, base_height: {fmt(ch['base_height'])}, "
                         f"base_mass: {fmt(ch['base_mass'])}, link_mass: {fmt(ch['link_mass'])}, "
                         f"link_length: {fmt(ch['link_length'])}, initial_angle: {fmt(angle)}, "
                         f"collider_radius: {fmt(ch['collider_radius'])} }}),")
        else:
            lines.append("        chains: None,")
        dt_num, dt_den = sc["dt"]
        g = ", ".join(fmt(x) for x in sc["gravity"])
        lines.append(f"        dt_num: {dt_num}, dt_den: {dt_den},")
        lines.append(f"        frames: {sc['frames']}, warmup: {sc['warmup']}, "
                     f"iterations: {sc['iterations']}, gravity: [{g}],")
        lines.append(f"        broadphase_dbvt: {'true' if sc['broadphase'] == 'dbvt' else 'false'},")
        lines.append(f"        arena_script: {'true' if sc['script'] == 'arena' else 'false'},")
        lines.append("    }),")
    lines.append("];")
    lines.append("")
    return "\n".join(lines)


def validate(scenes):
    for name, sc in scenes.items():
        if "dt" not in sc or "frames" not in sc or "gravity" not in sc:
            print(f"error: {name}: missing dt/frames/gravity", file=sys.stderr)
            return False
        sc.setdefault("warmup", 0)
        sc.setdefault("iterations", 10)
        sc.setdefault("bodies", [])
        sc.setdefault("generators", [])
        sc.setdefault("broadphase", "sweep")
        sc.setdefault("broadphase_aabb", [[-50, -50, -50], [50, 50, 50]])
        sc.setdefault("script", "none")
        if sc["broadphase"] not in ("sweep", "dbvt"):
            print(f"error: {name}: bad broadphase {sc['broadphase']!r} (want 'sweep' or 'dbvt')",
                  file=sys.stderr)
            return False
        if sc["script"] not in ("none", "arena"):
            print(f"error: {name}: bad script {sc['script']!r} (want 'none' or 'arena')",
                  file=sys.stderr)
            return False
        if sc["script"] == "arena":
            bodies = sc.get("bodies", [])
            dozer = bodies[1] if len(bodies) > 1 else None
            if (not dozer or dozer.get("mass") != 50
                    or dozer.get("shape", {}).get("type") != "sphere"):
                print(f"error: {name}: arena scripts expect bodies[1] to be the "
                      "mass-50 dozer sphere", file=sys.stderr)
                return False
        for b in sc["bodies"]:
            if b["shape"]["type"] not in SHAPE_TYPES:
                print(f"error: {name}/{b['name']}: bad shape type {b['shape']['type']!r}",
                      file=sys.stderr)
                return False
        for g in sc["generators"]:
            if g["type"] not in GENERATOR_TYPES:
                print(f"error: {name}: bad generator type {g['type']!r}", file=sys.stderr)
                return False
        if "chains" in sc and sc.get("bodies"):
            print(f"error: {name}: chain scenes must not mix with rigid bodies", file=sys.stderr)
            return False
        try:
            bodies = expand_scene(sc)
        except (ValueError, KeyError) as e:
            print(f"error: {name}: {e}", file=sys.stderr)
            return False
        if not bodies and "chains" not in sc:
            print(f"error: {name}: empty scene", file=sys.stderr)
            return False
    return True


def main():
    with open(SCENES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenes = data["scenes"]
    if not validate(scenes):
        return 1
    with open(C_HEADER, "w", encoding="utf-8", newline="\n") as f:
        f.write(gen_c(scenes))
    with open(RS_TABLE, "w", encoding="utf-8", newline="\n") as f:
        f.write(gen_rs(scenes))
    for name, sc in scenes.items():
        n = len(expand_scene(sc)) + (sc["chains"]["count"] if "chains" in sc else 0)
        print(f"scene {name}: {n} dynamic objects")
    print(f"generated {C_HEADER}")
    print(f"generated {RS_TABLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
