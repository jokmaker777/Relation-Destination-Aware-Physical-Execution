#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finish the remaining SIGMOD planner, baseline and constraint evidence.

Run this file directly; no command-line arguments are required.  It covers:

* aligned Native-PyG versus custom exact R-GCN semantics on HGB-DBLP;
* a measured P x B x C x G table on the representative industrial snapshot;
* W/G, w_max, greedy-bound and actual-makespan diagnostics;
* conservative memory-capacity sweeps and observed allocator-OOM evaluation;
* quality-filtered strong-scaling, budget-regret and horizon-total-cost reports.

The companion ``artifact/run_sigmod_gpuA_supplements_no_parser.py`` may be
started as a second process.  Both launchers share a Linux file lock so timing
experiments cannot overlap on the same physical four-GPU server.  This script
waits for the companion completion marker before producing the final report.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import run_sigmod_core_suite_no_parser as core
from sigmod_experiments.server_no_parser.common_no_parser import (
    run_multidate_industrial_job,
)


GRAPH = "20241129"
GPU_COUNTS = (4, 2)
PARTITION_PLANS = (
    "superblock_edge_count",
    "superblock_vector_balance",
)
BLOCK_EDGES = (10_000, 25_000, 50_000, 100_000)
COMMUNICATION_PLANS = (
    core.DENSE_BLOCKING,
    core.ASYNC_C2,
)
REPEATS = (1,)

RUN_PUBLIC_ALIGNMENT = True
RUN_GPU_MATRIX = True
WAIT_FOR_GPU_A_SECONDS = 3 * 60 * 60
WAIT_POLL_SECONDS = 30

LOCK_PATH = core.INDUSTRIAL_ROOT / ".sigmod_publication_gpu.lock"
GPU_A_MARKER = (
    core.CORE_ROOT
    / "stage_status"
    / "gpuA_scaling_and_memory_supplements.json"
)
COMPLETION_PATH = (
    core.CORE_ROOT
    / "stage_status"
    / "remaining_joint_constraints_and_final_report.json"
)


@contextmanager
def publication_gpu_lock(label: str):
    import fcntl

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as stream:
        print(f"WAIT GPU LOCK: {LOCK_PATH} ({label})", flush=True)
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {"pid": os.getpid(), "label": label, "started": time.time()},
                ensure_ascii=False,
            )
        )
        stream.flush()
        print(f"ACQUIRED GPU LOCK: {label}", flush=True)
        try:
            yield
        finally:
            print(f"RELEASE GPU LOCK: {label}", flush=True)
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_completed(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if payload.get("completed") else None
    except Exception:
        return None


def preflight_current_code() -> None:
    checks = {
        Path(core.__file__): (
            "native_pyg_aligned",
            "collect_quality_filtered_industrial",
            "memory_capacity_sweep_rows",
        ),
        core.EXPERIMENT_ROOT / "budgeted_joint_planner.py": (
            "best_plan_id",
            "def memory_capacity_sweep",
            "def assignment_bounds",
        ),
        core.PROJECT_ROOT
        / "end_to_end_prunning"
        / "models"
        / "communication_optimized_training_manager_forward_agg_overlap_no_parser.py": (
            "total_work_W",
            "w_max",
            "greedy_upper_bound",
        ),
    }
    stale: List[str] = []
    for path, tokens in checks.items():
        if not path.exists():
            stale.append(f"missing {path}")
            continue
        source = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in source:
                stale.append(f"{path.name} lacks {token!r}")
    if stale:
        raise RuntimeError(
            "STALE REMOTE PROJECT: upload the complete updated artifact "
            "directory before running. " + "; ".join(stale)
        )
    core.run_preflight()


def run_joint_matrix() -> None:
    total = (
        len(GPU_COUNTS)
        * len(PARTITION_PLANS)
        * len(BLOCK_EDGES)
        * len(COMMUNICATION_PLANS)
        * len(REPEATS)
    )
    print(
        f"PBCG MATRIX: up to {total} cells; completed cells are skipped.",
        flush=True,
    )
    for world_size in GPU_COUNTS:
        for strategy in PARTITION_PLANS:
            for block_edges in BLOCK_EDGES:
                for mode in COMMUNICATION_PLANS:
                    for repeat in REPEATS:
                        run_multidate_industrial_job(
                            stage="core_joint_pbcg",
                            graph_tag=GRAPH,
                            world_size=world_size,
                            strategy=strategy,
                            block_edges=block_edges,
                            mode=mode,
                            repeat=repeat,
                        )


def wait_for_gpu_a() -> Dict[str, Any] | None:
    payload = read_completed(GPU_A_MARKER)
    if payload is not None:
        return payload
    deadline = time.time() + WAIT_FOR_GPU_A_SECONDS
    print(
        f"WAITING FOR COMPANION: {GPU_A_MARKER} "
        f"(timeout={WAIT_FOR_GPU_A_SECONDS}s)",
        flush=True,
    )
    while time.time() < deadline:
        payload = read_completed(GPU_A_MARKER)
        if payload is not None:
            print("COMPANION COMPLETE", flush=True)
            return payload
        time.sleep(WAIT_POLL_SECONDS)
    print(
        "WARNING: companion marker was not observed before timeout; "
        "the final report will use all currently available valid measurements.",
        flush=True,
    )
    return None


def evaluate_controlled_memory_boundary() -> Dict[str, Any]:
    boundary_path = (
        core.CORE_ROOT / "memory_feasibility" / "controlled_boundary.json"
    )
    output_root = core.CORE_ROOT / "planner_total_cost"
    evaluation_path = output_root / "controlled_memory_boundary_evaluation.csv"
    summary_path = output_root / "controlled_memory_boundary_summary.json"
    boundary = read_completed(boundary_path)
    model_rows = core.read_csv(output_root / "memory_model_evaluation.csv")
    if boundary is None or not model_rows:
        summary = {
            "completed": False,
            "available": False,
            "reason": "controlled boundary or memory-model evaluation is missing",
        }
        core.write_json_atomic(summary_path, summary)
        return summary

    by_plan = {str(row.get("plan_id")): row for row in model_rows}
    rows: List[Dict[str, Any]] = []
    for cell in boundary.get("cells", []):
        plan_id = (
            f"P={cell['strategy']}|B={int(cell['block_edges'])}|C=1|"
            f"G={int(cell['world_size'])}|M=blocking_c1|A=0"
        )
        prediction = by_plan.get(plan_id)
        if prediction is None:
            rows.append(
                {
                    **cell,
                    "plan_id": plan_id,
                    "prediction_available": False,
                    "predicted_feasible": None,
                    "unsafe_accept": None,
                    "false_reject": None,
                }
            )
            continue
        safe_peak = core._float(prediction.get("safe_predicted_peak_memory_mb"))
        capacity = core._float(cell.get("effective_capacity_mb"))
        predicted_feasible = safe_peak <= capacity
        actual_feasible = bool(cell.get("actual_feasible"))
        rows.append(
            {
                **cell,
                "plan_id": plan_id,
                "prediction_available": True,
                "predicted_peak_memory_mb": core._float(
                    prediction.get("predicted_peak_memory_mb")
                ),
                "safe_predicted_peak_memory_mb": safe_peak,
                "predicted_feasible": predicted_feasible,
                "unsafe_accept": predicted_feasible and not actual_feasible,
                "false_reject": (not predicted_feasible) and actual_feasible,
            }
        )
    core.write_csv(evaluation_path, rows)
    classified = [row for row in rows if row.get("prediction_available")]
    infeasible = [row for row in classified if not bool(row.get("actual_feasible"))]
    feasible = [row for row in classified if bool(row.get("actual_feasible"))]
    summary = {
        "completed": True,
        "available": True,
        "cells": len(rows),
        "classified_cells": len(classified),
        "observed_feasible_cells": len(feasible),
        "observed_oom_cells": len(infeasible),
        "unsafe_accept_count": sum(bool(row.get("unsafe_accept")) for row in classified),
        "unsafe_accept_rate": (
            sum(bool(row.get("unsafe_accept")) for row in infeasible)
            / len(infeasible)
            if infeasible
            else None
        ),
        "false_reject_count": sum(bool(row.get("false_reject")) for row in classified),
        "false_reject_rate": (
            sum(bool(row.get("false_reject")) for row in feasible) / len(feasible)
            if feasible
            else None
        ),
        "note": (
            "actual feasibility comes from allocator-limited executions; "
            "predictions come from the leave-one-snapshot-out memory model"
        ),
    }
    core.write_json_atomic(summary_path, summary)
    return summary


def verify_outputs(
    semantic: Dict[str, Any], controlled: Dict[str, Any]
) -> Dict[str, Any]:
    planner_path = (
        core.CORE_ROOT / "planner_total_cost" / "two_day_analysis_summary.json"
    )
    if not planner_path.exists():
        raise RuntimeError(f"planner summary was not created: {planner_path}")
    planner = json.loads(planner_path.read_text(encoding="utf-8"))
    coverage = planner.get("coverage", {})
    block_edges = {int(value) for value in coverage.get("block_edges", [])}
    required_blocks = set(BLOCK_EDGES)
    problems: List[str] = []
    if not coverage.get("is_full_joint_PBCG"):
        problems.append("is_full_joint_PBCG is false")
    if not required_blocks.issubset(block_edges):
        problems.append(
            f"missing block sizes: {sorted(required_blocks - block_edges)}"
        )
    if int(planner.get("assignment_bound_rows") or 0) <= 0:
        problems.append("assignment_bound_rows is zero")
    if int(planner.get("memory_capacity_sweep_rows") or 0) < 5:
        problems.append("memory_capacity_sweep_rows is incomplete")
    if int(semantic.get("pairs") or 0) != 3:
        problems.append("three aligned PyG/custom repeat pairs were not collected")

    horizon_methods = {
        str(row.get("method"))
        for row in core.read_csv(
            core.CORE_ROOT / "planner_total_cost" / "horizon_total_cost.csv"
        )
    }
    required_methods = {
        "history_warm",
        "cold_start",
        "random",
        "joint_search",
        "fixed_default",
        "two_stage_history",
        "measured_oracle",
    }
    if not required_methods.issubset(horizon_methods):
        problems.append(
            f"horizon table lacks methods: {sorted(required_methods - horizon_methods)}"
        )
    if problems:
        raise RuntimeError("Final SIGMOD output verification failed: " + "; ".join(problems))

    return {
        "completed": True,
        "planner_output": str(planner_path.parent),
        "is_full_joint_PBCG": coverage.get("is_full_joint_PBCG"),
        "block_edges": sorted(block_edges),
        "world_sizes": coverage.get("world_sizes"),
        "chunks": coverage.get("chunks"),
        "assignment_bound_rows": planner.get("assignment_bound_rows"),
        "memory_capacity_sweep_rows": planner.get("memory_capacity_sweep_rows"),
        "semantic_alignment": semantic,
        "controlled_memory_boundary": controlled,
        "horizon_methods": sorted(horizon_methods),
    }


def main() -> None:
    preflight_current_code()
    started = time.time()
    with publication_gpu_lock("artifact joint planner and public alignment"):
        if RUN_PUBLIC_ALIGNMENT:
            core.launch_public_semantic_alignment()
        if RUN_GPU_MATRIX:
            run_joint_matrix()

    companion = wait_for_gpu_a()
    # Everything below is read-only with respect to GPU timing measurements.
    core.run_planner_total_cost()
    semantic = core.collect_public_semantic_alignment()
    controlled = evaluate_controlled_memory_boundary()
    core.collect_report()
    final = verify_outputs(semantic, controlled)
    final["gpu_a_companion"] = companion
    final["elapsed_sec"] = time.time() - started
    core.write_json_atomic(COMPLETION_PATH, final)
    print("\n=== FINAL SIGMOD SUPPLEMENT SUMMARY ===", flush=True)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    print(f"COMPLETE: {COMPLETION_PATH}", flush=True)


if __name__ == "__main__":
    main()

