#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the short, non-training SIGMOD evidence suite in the required order."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(
    "/path/to/artifact_workspace/sigmod2027_20241129/two_day_fast_suite"
)
STEPS = (
    (
        "budgeted_planner_replay",
        PROJECT_ROOT
        / "sigmod_experiments"
        / "server_no_parser"
        / "14_two_day_planner_replay_no_parser.py",
    ),
    ("forward_gradient_exactness", PROJECT_ROOT / "run_exactness_matrix_1to4gpu_no_parser.py"),
    ("pipeline_recurrence", PROJECT_ROOT / "run_pipeline_recurrence_2to4gpu_no_parser.py"),
    ("collective_alpha_beta", PROJECT_ROOT / "run_collective_alpha_beta_1to4gpu_no_parser.py"),
)


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
records = []
for index, (name, script) in enumerate(STEPS, 1):
    if not script.exists():
        raise FileNotFoundError(script)
    log = OUTPUT_ROOT / f"{index:02d}_{name}.log"
    print(f"STEP {index}/{len(STEPS)} {name}: {script}", flush=True)
    started = time.time()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record = {
        "step": index,
        "name": name,
        "script": str(script),
        "log": str(log),
        "elapsed_sec": time.time() - started,
        "returncode": completed.returncode,
        "status": "complete" if completed.returncode == 0 else "failed",
    }
    records.append(record)
    (OUTPUT_ROOT / "suite_status.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if completed.returncode != 0:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-160:])
        raise RuntimeError(f"Step {name} failed:\n{tail}")
    print(f"DONE {name}: {record['elapsed_sec']:.1f}s", flush=True)

print(f"TWO-DAY FAST SUITE COMPLETED: {OUTPUT_ROOT}")

