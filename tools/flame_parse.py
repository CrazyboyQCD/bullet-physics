"""Parse inferno flamegraph SVG into a stack tree; report self/inclusive time."""
import re
import sys
from collections import defaultdict

svg = open(sys.argv[1], encoding="utf-8").read()

# Frames: <g class="fN"><title>name (count samples, pct%)</title><rect x=.. width=../>
# Inferno nests <g> elements hierarchically.
frame_re = re.compile(
    r'<g><title>(.*?) \(([0-9,]+) samples, ([0-9.]+)%\)</title>'
    r'<rect x="([0-9.]+)%" y="(?P<y>[0-9.]+)" width="([0-9.]+)%"[^>]*fg:x="[0-9]+" fg:w="([0-9]+)"',
)

frames = []
for m in frame_re.finditer(svg):
    name, count = m.group(1), int(m.group(2).replace(',', ''))
    x, w, y = float(m.group(4)), float(m.group(6)), float(m.group(5))
    frames.append((m.start(), m.end(), name, count, x, w, y))

# Reconstruct tree geometrically with upward search:
# parent of f = nearest frame at some row above whose x-range contains f (tolerance absorbs
# inferno's pixel merge rounding, which can push children slightly outside parents).
frames.sort(key=lambda f: (f[6], f[4]))  # by depth then x
TOL = 0.15  # percent
rows = sorted({f[6] for f in frames})
row_index = {y: i for i, y in enumerate(rows)}

parent_of = {}
roots = []
for f in frames:
    p = None
    ri = row_index[f[6]]
    x0, x1 = f[4] - TOL, f[4] + f[5] + TOL
    for ry in rows[ri - 1::-1]:
        cand = [g for g in frames if g[6] == ry and g[4] - TOL <= x0 and g[4] + g[5] + TOL >= x1]
        if cand:
            p = cand[0]
            break
    if p is None:
        roots.append(f)
    else:
        parent_of[id(f)] = p

tree_by_parent = defaultdict(list)
id_to_frame = {id(f): f for f in frames}
for fid, p in parent_of.items():
    tree_by_parent[p].append(id_to_frame[fid])

# Normalize by the true profile total = the widest root ("all"). Geometric
# mismatches create extra roots whose counts re-count samples across stack
# depths; they must not inflate the denominator.
total = max(f[3] for f in roots)

def self_time(f):
    return f[3] - sum(k[3] for k in tree_by_parent[f])

# Top by self time
self_agg = defaultdict(int)
incl_agg = defaultdict(int)
def walk(f):
    incl_agg[f[2]] += f[3]
    self_agg[f[2]] += self_time(f)
    for c in tree_by_parent[f]:
        walk(c)
for f in roots:
    walk(f)

print(f"roots: {len(roots)}, total samples {total}")
print()
print("=== TOP SELF (exclusive) ===")
for name, c in sorted(self_agg.items(), key=lambda kv: -kv[1])[:35]:
    if c > 0:
        print(f"{100*c/total:6.2f}%  {c:7d}  {name[:130]}")

print()
print("=== TOP INCLUSIVE (leaf-merged) ===")
for name, c in sorted(incl_agg.items(), key=lambda kv: -kv[1])[:25]:
    print(f"{100*c/total:6.2f}%  {c:7d}  {name[:130]}")

# Show stack paths to a focus function (argv[2], default epa::penetration)
focus = sys.argv[2] if len(sys.argv) > 2 else "epa::penetration"
print()
print(f"=== STACK PATHS to {focus} ===")
def path(f, acc):
    acc.append(f[2])
    if focus in f[2]:
        print(" <- ".join(reversed([a.split('`')[-1][:70] for a in acc[-9:]])))
        print()
    for c in tree_by_parent[f]:
        path(c, acc)
    acc.pop()
for f in roots:
    path(f, [])
