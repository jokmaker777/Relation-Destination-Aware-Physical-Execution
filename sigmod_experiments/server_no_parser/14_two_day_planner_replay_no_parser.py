#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect existing GPU runs and replay the budgeted joint planner.

This script has no command-line arguments.  It does not launch training jobs and
does not modify completed results.  It scans all existing industrial
``train_result_rank0.json`` files, builds a unified P/B/C/G plan table, and
produces budget-regret, warm/cold/random, memory-model, fallback, and scheduling
bound results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from sigmod_experiments.budgeted_joint_planner import run_analysis, write_csv
from sigmod_experiments.collect_industrial_results import collect


SOURCE_ROOT = Path("/path/to/artifact_workspace/sigmod2027_20241129")
OUTPUT_ROOT = SOURCE_ROOT / "two_day_initial_results"
MEASUREMENTS_CSV = OUTPUT_ROOT / "all_existing_industrial_measurements.csv"
REUSE_EPOCHS = 50
TRIAL_BUDGETS = (1, 2, 4, 8, 16, 32)
RANDOM_SEEDS = tuple(20261004 + value for value in range(20))
MEMORY_SAFETY_FACTOR = 1.10


def detect_memory_capacity_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            values = [
                float(torch.cuda.get_device_properties(index).total_memory) / (1024 ** 2)
                for index in range(torch.cuda.device_count())
            ]
            return min(values)
        if hasattr(torch, "npu") and torch.npu.is_available():
            # Some NPU-compatible runtimes expose total memory through mem_get_info.
            free_bytes, total_bytes = torch.npu.mem_get_info(0)
            del free_bytes
            return float(total_bytes) / (1024 ** 2)
    except Exception as exc:
        print(f"MEMORY CAPACITY DETECTION UNAVAILABLE: {type(exc).__name__}: {exc}")
    return 0.0


def collect_existing_results():
    rows = []
    failures = []
    paths = sorted(SOURCE_ROOT.rglob("train_result_rank0.json"))
    for index, path in enumerate(paths, 1):
        try:
            row = collect(path, REUSE_EPOCHS)
            if float(row.get("steady_epoch_sec_mean", 0.0)) <= 0:
                raise RuntimeError("missing positive steady_epoch_sec_mean")
            rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if index % 50 == 0:
            print(f"COLLECT PROGRESS {index}/{len(paths)}")
    if not rows:
        raise RuntimeError(f"No usable industrial results found under {SOURCE_ROOT}")
    rows.sort(
        key=lambda row: (
            str(row.get("graph")),
            int(row.get("world_size", 1)),
            str(row.get("strategy")),
            int(row.get("block_edges", 0)),
            int(row.get("chunks", 1)),
            str(row.get("mode")),
            int(row.get("repeat", 0)),
        )
    )
    write_csv(MEASUREMENTS_CSV, rows)
    (OUTPUT_ROOT / "collection_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"COLLECTED {len(rows)} RUNS; rejected={len(failures)}; "
        f"output={MEASUREMENTS_CSV}"
    )
    return rows


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
collect_existing_results()
capacity_mb = detect_memory_capacity_mb()
print(f"GPU MEMORY CAPACITY FOR FEASIBILITY CHECK: {capacity_mb:.1f} MB")
run_analysis(
    input_csv=MEASUREMENTS_CSV,
    output_dir=OUTPUT_ROOT,
    source_root=SOURCE_ROOT,
    horizon_epochs=REUSE_EPOCHS,
    budgets=TRIAL_BUDGETS,
    seeds=RANDOM_SEEDS,
    memory_capacity_mb=capacity_mb,
    memory_safety_factor=MEMORY_SAFETY_FACTOR,
)
print(f"TWO-DAY PLANNER REPLAY COMPLETED: {OUTPUT_ROOT}")

