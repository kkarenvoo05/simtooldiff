#!/usr/bin/env python3
"""Verify the phase-gated-noise wiring is complete and consistent.

Run once before launching collection:
    conda run -n dp python scripts/verify_closure_gate.py

Compiles the touched files, AST-checks the gate is wired into both rollout
loops, and confirms the new CLI flags exist on the collector + driver.
No Isaac Gym / GPU needed.
"""

from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLLECTOR = HERE / "collect_dataset_pick_place_release.py"
DRIVER = HERE / "stage5_multi_object_driver.py"
GATE = HERE / "closure_gate.py"
NOISE_CFG = HERE / "stage5_noise_config.py"

NEW_FLAGS = [
    "--noise-phase-gating", "--closure-proximity-threshold", "--closure-palm-z-threshold",
    "--closure-noise-scale", "--closure-groups", "--closure-window-padding",
    "--finger-noise-multiplier", "--wrist-noise-multiplier",
]


def _check(label, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def main():
    all_ok = True
    print("1. Compile:")
    for f in (GATE, NOISE_CFG, COLLECTOR, DRIVER):
        try:
            py_compile.compile(str(f), doraise=True)
            all_ok &= _check(f.name, True)
        except py_compile.PyCompileError as e:
            all_ok &= _check(f"{f.name}: {e}", False)

    src = COLLECTOR.read_text()
    drv = DRIVER.read_text()

    print("2. Collector wiring:")
    all_ok &= _check("imports ClosureGate", "from closure_gate import" in src)
    all_ok &= _check("gate constructed (>=2 sites)", src.count("ClosureGate(") >= 2)
    all_ok &= _check("gate applied (>=2 sites)", src.count("closure_gate.apply(env") >= 2)
    all_ok &= _check("records gate config", "to_attrs()" in src)
    all_ok &= _check("multiplier wrapper present", "_orig_resolve_noise_args" in src)
    for flag in NEW_FLAGS:
        all_ok &= _check(f"collector exposes {flag}", flag in src)

    print("3. Driver forwarding:")
    for flag in NEW_FLAGS:
        all_ok &= _check(f"driver exposes+forwards {flag}", drv.count(flag) >= 2)

    print("4. Gate ordering (scales accumulated OU state, not just sample):")
    ok_order = True
    tree = ast.parse(src)
    for node in ast.walk(tree):
        seg = ast.get_source_segment(src, node) or ""
        if isinstance(node, ast.FunctionDef) and "closure_gate.apply" in seg:
            i_ou = seg.find("* args.ou_dt")
            i_apply = seg.find("closure_gate.apply")
            if not (i_ou != -1 and i_apply > i_ou):
                ok_order = False
    all_ok &= _check("apply() after OU update in every loop", ok_order)

    print()
    print("RESULT:", "ALL PASS - safe to collect." if all_ok else "FAILURES - fix before collecting.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
