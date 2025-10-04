#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, os, sys
from pathlib import Path
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--base"); b.add_argument("--head")
    args = p.parse_args()
    if args.cmd != "build": return 0
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out/"pack.json").write_text(json.dumps({"ok": True, "base":args.base, "head":args.head}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out/"pack.md").write_text("# Mock pack\n", encoding="utf-8")
    (out/"prompt.json").write_text("{}", encoding="utf-8")
    return 0
if __name__ == "__main__":
    sys.exit(main())
