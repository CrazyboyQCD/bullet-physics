"""Build the self-contained wasm demo page.

Compiles the `bullet-wasm-demo` cdylib for wasm32-unknown-unknown, then
inlines the wasm binary (base64) into `template.html` to produce
`tools/wasm-demo/dist/index.html` — a single file whose only external
reference is the Babylon.js CDN script tag.

Usage:
    python tools/wasm-demo/build.py          # build + emit dist/index.html
    python tools/wasm-demo/build.py --serve  # also serve dist/ on :8134
"""

import argparse
import base64
import http.server
import os
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WASM = os.path.join(
    ROOT, "target", "wasm32-unknown-unknown", "release", "bullet_wasm_demo.wasm"
)
TEMPLATE = os.path.join(HERE, "template.html")
OUT_DIR = os.path.join(HERE, "dist")
OUT = os.path.join(OUT_DIR, "index.html")
PORT = 8134


def build_wasm() -> None:
    cmd = [
        "cargo",
        "build",
        "--release",
        "--target",
        "wasm32-unknown-unknown",
        "-p",
        "bullet-wasm-demo",
        # The workspace release profile keeps debug info for host profiling;
        # the demo page does not need it (6.1 MB -> 0.5 MB).
        "--config",
        "profile.release.debug=false",
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def emit_html() -> None:
    with open(WASM, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        html = f.read()
    if "@@WASM_B64@@" not in html:
        raise SystemExit("template.html is missing the @@WASM_B64@@ placeholder")
    html = html.replace("@@WASM_B64@@", b64)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


def serve() -> None:
    os.chdir(OUT_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"serving http://127.0.0.1:{PORT}/index.html (Ctrl+C to stop)")
        httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="serve dist/ after building")
    args = ap.parse_args()
    build_wasm()
    emit_html()
    if args.serve:
        threading.Timer(0.5, lambda: None).start()
        serve()


if __name__ == "__main__":
    sys.exit(main())
