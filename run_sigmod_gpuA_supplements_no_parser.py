#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finish the short industrial scaling and controlled-memory supplements.

Run this file directly; there are no command-line arguments.  It performs two
publication-facing tasks on the 2024-11-29 industrial snapshot:

1. clean one-GPU repeats R04/R05 for Direct and SuperBlock, with mandatory
   peak-memory/epoch/runtime quality checks;
2. a small two-GPU allocator-capacity boundary for Direct and SuperBlock.

The raw capacity-limited jobs are intentionally written outside the main
industrial result tree so that they cannot contaminate the normal planner cost
table.  A compact feasibility summary is copied into ``sigmod_core_suite``.

This launcher and ``artifact/run_remaining_joint_constraints_no_parser.py``
use the same Linux file lock.  Both may be started, but publication timing jobs
will never overlap on the same physical four-GPU server.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_ROOT = PROJECT_ROOT / "sigmod_experiments" / "server_no_parser"
sys.path.insert(0, str(SERVER_ROOT))

from common_no_parser import (  # noqa: E402
    industrial_experiment_name,
    load_multidate_metadata,
    run_multidate_industrial_job,
)
from settings_no_parser import (  # noqa: E402
    BASE_OUTPUT_ROOT,
    BASE_SEED,
    EPOCHS,
    GPU_IDS,
    INDUSTRIAL_RUNNER,
)


GRAPH_TAG = "20241129"
BLOCK_EDGES = 25_000
CLEAN_REPEATS = (4, 5)
CLEAN_STAGE = "core_industrial_scaling_clean_1gpu"
CLEAN_CONFIGURATIONS = (
    ("Direct edge-balanced", "balanced_edge"),
    ("SuperBlock edge-count", "superblock_edge_count"),
)
DENSE_BLOCKING = {"name": "blocking_c1", "chunks": 1, "overlap": False}

MEMORY_WORLD_SIZE = 2
MEMORY_EPOCHS = 2
# Include a deliberately tight allocator cap so the sweep is likely to contain
# at least one observed OOM label instead of only feasible extrapolation points.
MEMORY_FRACTIONS = (0.35, 0.55, 0.75, 0.95)
MEMORY_STRATEGIES = ("balanced_edge", "superblock_edge_count")

CORE_ROOT = BASE_OUTPUT_ROOT / "sigmod_core_suite"
LOCK_PATH = BASE_OUTPUT_ROOT / ".sigmod_publication_gpu.lock"
BOUNDARY_ROOT = BASE_OUTPUT_ROOT.parent / "sigmod_memory_boundary_20241129"
BOUNDARY_SUMMARY_ROOT = CORE_ROOT / "memory_feasibility"
MEMORY_FRACTION_WORKER = BOUNDARY_ROOT / "memory_fraction_worker.py"
COMPLETION_PATH = (
    CORE_ROOT / "stage_status" / "gpuA_scaling_and_memory_supplements.json"
)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    columns: List[str] = []
    for row in values:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


@contextmanager
def publication_gpu_lock(label: str):
    """Serialize timing experiments launched from the two project copies."""

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


def ensure_memory_fraction_worker() -> Path:
    """Create a torchrun worker that also supports an older training runner.

    PyCharm deployments can update this launcher without updating the large
    industrial runner next to it.  The compatibility worker applies the CUDA
    allocator limit before executing that runner, so the controlled-OOM sweep
    remains a real measured capacity experiment in either deployment state.
    """

    source = '''#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import torch

target = Path(os.environ["COMMOPT_INDUSTRIAL_RUNNER"]).resolve()
fraction = float(os.environ["COMMOPT_CUDA_MEMORY_FRACTION"])
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required by the controlled-memory worker")
torch.cuda.set_device(local_rank)
torch.cuda.set_per_process_memory_fraction(fraction, device=local_rank)
print(
    f"CONTROLLED CUDA MEMORY: local_rank={local_rank} fraction={fraction:.4f}",
    flush=True,
)
sys.path.insert(0, str(target.parent))
runpy.run_path(str(target), run_name="__main__")
'''
    MEMORY_FRACTION_WORKER.parent.mkdir(parents=True, exist_ok=True)
    temporary = MEMORY_FRACTION_WORKER.with_suffix(".py.tmp")
    temporary.write_text(source, encoding="utf-8")
    temporary.replace(MEMORY_FRACTION_WORKER)
    return MEMORY_FRACTION_WORKER


def preflight() -> Dict[str, Dict[str, Any]]:
    if not INDUSTRIAL_RUNNER.exists():
        raise FileNotFoundError(INDUSTRIAL_RUNNER)
    source = INDUSTRIAL_RUNNER.read_text(encoding="utf-8")
    worker = ensure_memory_fraction_worker()
    if "COMMOPT_CUDA_MEMORY_FRACTION" in source:
        print("memory_fraction_support=native+compatibility_worker", flush=True)
    else:
        print(
            "memory_fraction_support=compatibility_worker "
            f"(older industrial runner detected: {INDUSTRIAL_RUNNER})",
            flush=True,
        )
    print(f"memory_fraction_worker={worker}", flush=True)
    metadata = load_multidate_metadata()
    if GRAPH_TAG not in metadata:
        raise KeyError(f"Snapshot {GRAPH_TAG} is absent from multidate metadata")
    graph_path = Path(metadata[GRAPH_TAG]["graph_path"])
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    if len(GPU_IDS) < MEMORY_WORLD_SIZE:
        raise RuntimeError(
            f"Need {MEMORY_WORLD_SIZE} configured GPUs, found {len(GPU_IDS)}"
        )
    print(f"graph={graph_path}", flush=True)
    print(
        f"clean_scaling_jobs={len(CLEAN_REPEATS) * len(CLEAN_CONFIGURATIONS)}; "
        f"memory_boundary_jobs={len(MEMORY_FRACTIONS) * len(MEMORY_STRATEGIES)}",
        flush=True,
    )
    return metadata


def read_runtime_epochs(result_path: Path) -> List[float]:
    runtime = result_path.with_name("runtime_rank0.csv")
    if not runtime.exists():
        return []
    values: List[float] = []
    with runtime.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("event") != "train_epoch":
                continue
            try:
                value = float(row.get("elapsed_sec", ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                values.append(value)
    return values


def find_industrial_result(
    *, graph_tag: str, strategy: str, repeat: int, world_size: int
) -> Path:
    metadata = load_multidate_metadata()[graph_tag]
    expected = industrial_experiment_name(
        graph_tag=graph_tag,
        mode_name=DENSE_BLOCKING["name"],
        strategy=strategy,
        block_edges=BLOCK_EDGES,
        repeat=repeat,
        seed=BASE_SEED + repeat - 1,
        total_labeled_edges=int(metadata["observed_labeled_edges"]),
    )
    matches = [
        path
        for path in (BASE_OUTPUT_ROOT / "industrial").rglob(
            "train_result_rank0.json"
        )
        if path.parent.name == expected
        and path.parent.parent.name == f"gpus-{world_size}"
    ]
    if not matches:
        raise FileNotFoundError(f"No result found for {expected}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def existing_clean_reference_median() -> float | None:
    values: List[float] = []
    for path in (BASE_OUTPUT_ROOT / "industrial").rglob(
        "train_result_rank0.json"
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            str(payload.get("graph_tag")) != GRAPH_TAG
            or int(payload.get("world_size", 0)) != 1
            or str(payload.get("strategy"))
            not in {value[1] for value in CLEAN_CONFIGURATIONS}
            or str(payload.get("mode")) != "blocking_c1"
            or int(payload.get("target_super_block_size", 0)) != BLOCK_EDGES
            or float(payload.get("peak_memory_mb") or 0.0) <= 0
        ):
            continue
        epochs = read_runtime_epochs(path)
        if len(epochs) >= EPOCHS:
            values.append(statistics.median(epochs[1:] or epochs))
    return statistics.median(values) if values else None


def validate_clean_result(
    result_path: Path, reference_median: float | None
) -> Dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    epochs = read_runtime_epochs(result_path)
    steady = epochs[1:] if len(epochs) > 1 else epochs
    steady_median = statistics.median(steady) if steady else 0.0
    reasons: List[str] = []
    peak = float(payload.get("peak_memory_mb") or 0.0)
    if peak <= 0:
        reasons.append("peak_memory_mb<=0")
    if len(epochs) < EPOCHS:
        reasons.append(f"observed_epochs={len(epochs)}<{EPOCHS}")
    if steady_median <= 0:
        reasons.append("steady_epoch_median<=0")
    if reference_median and not (
        0.5 * reference_median <= steady_median <= 2.0 * reference_median
    ):
        reasons.append(
            f"steady_epoch_median={steady_median:.4f}s outside "
            f"[0.5x,2x] reference={reference_median:.4f}s"
        )
    row = {
        "result_path": str(result_path),
        "graph": str(payload.get("graph_tag")),
        "world_size": int(payload.get("world_size", 0)),
        "strategy": str(payload.get("strategy")),
        "repeat": int(payload.get("repeat_id", 0)),
        "peak_memory_mb": peak,
        "observed_epochs": len(epochs),
        "steady_epoch_median_sec": steady_median,
        "reference_median_sec": reference_median,
        "valid": not reasons,
        "rejection_reason": "; ".join(reasons),
    }
    if reasons:
        raise RuntimeError(
            "Clean scaling quality gate failed; the job is not retried "
            f"automatically: {json.dumps(row, ensure_ascii=False)}"
        )
    return row


def run_clean_scaling_repeats() -> List[Dict[str, Any]]:
    quality_rows: List[Dict[str, Any]] = []
    reference = existing_clean_reference_median()
    print(f"ONE-GPU REFERENCE MEDIAN: {reference}", flush=True)
    for repeat in CLEAN_REPEATS:
        for label, strategy in CLEAN_CONFIGURATIONS:
            print(
                f"CLEAN SCALING graph={GRAPH_TAG} GPU=1 repeat={repeat} "
                f"config={label}",
                flush=True,
            )
            run_multidate_industrial_job(
                stage=CLEAN_STAGE,
                graph_tag=GRAPH_TAG,
                world_size=1,
                strategy=strategy,
                block_edges=BLOCK_EDGES,
                mode=DENSE_BLOCKING,
                repeat=repeat,
            )
            result = find_industrial_result(
                graph_tag=GRAPH_TAG,
                strategy=strategy,
                repeat=repeat,
                world_size=1,
            )
            quality_rows.append(validate_clean_result(result, reference))
    write_csv(CORE_ROOT / "industrial_clean_1gpu_quality.csv", quality_rows)
    return quality_rows


def gpu_total_memory_mb() -> float:
    # Some managed compute images expose CUDA through PyTorch but do not place
    # the nvidia-smi executable on PATH.  Query the same physical property from
    # CUDA directly and use logical device indices, which also respects any
    # scheduler-provided CUDA_VISIBLE_DEVICES remapping.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable while reading GPU capacity")
    visible = torch.cuda.device_count()
    if visible < MEMORY_WORLD_SIZE:
        raise RuntimeError(
            f"Expected at least {MEMORY_WORLD_SIZE} visible GPUs, found {visible}"
        )
    values = [
        torch.cuda.get_device_properties(index).total_memory / (1024.0 * 1024.0)
        for index in range(MEMORY_WORLD_SIZE)
    ]
    return min(values)


def boundary_experiment_name(
    *, strategy: str, mode_name: str, total_edges: int
) -> str:
    blocks = max(1, math.ceil(total_edges / BLOCK_EDGES))
    return (
        f"forwardagg_{mode_name}_{strategy}_{GRAPH_TAG}_K{blocks}_"
        f"Q{BLOCK_EDGES}_R01_ep{MEMORY_EPOCHS}_seed{BASE_SEED}"
    )


def run_memory_boundary_cell(
    metadata: Dict[str, Dict[str, Any]], strategy: str, fraction: float
) -> Dict[str, Any]:
    total_edges = int(metadata[GRAPH_TAG]["observed_labeled_edges"])
    graph_path = Path(metadata[GRAPH_TAG]["graph_path"])
    mode_name = f"blocking_c1_memcap{int(round(100 * fraction)):02d}"
    name = boundary_experiment_name(
        strategy=strategy, mode_name=mode_name, total_edges=total_edges
    )
    cell_path = (
        BOUNDARY_SUMMARY_ROOT
        / "cells"
        / f"{strategy}__fraction_{fraction:.2f}.json"
    )
    if cell_path.exists():
        try:
            existing = json.loads(cell_path.read_text(encoding="utf-8"))
            if existing.get("completed"):
                print(f"SKIP complete memory boundary cell: {cell_path}", flush=True)
                return existing
        except Exception:
            pass

    job_root = BOUNDARY_ROOT / "runs" / f"gpus-{MEMORY_WORLD_SIZE}"
    result_path = job_root / name / "train_result_rank0.json"
    log_path = BOUNDARY_ROOT / "logs" / f"g{MEMORY_WORLD_SIZE}__{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = max(1, math.ceil(total_edges / BLOCK_EDGES))
    selected_gpu_ids = tuple(GPU_IDS[:MEMORY_WORLD_SIZE])
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in selected_gpu_ids),
            "COMMOPT_FORCE_CUDA": "1",
            "COMMOPT_STRICT_DEVICE": "1",
            "COMMOPT_CUDA_DEVICE_MAP": ",".join(
                str(value) for value in range(MEMORY_WORLD_SIZE)
            ),
            "COMMOPT_PHYSICAL_GPU_IDS": ",".join(
                str(value) for value in selected_gpu_ids
            ),
            "COMMOPT_CUDA_MEMORY_FRACTION": str(fraction),
            "COMMOPT_INDUSTRIAL_RUNNER": str(INDUSTRIAL_RUNNER),
            "GRAPH_TAG": GRAPH_TAG,
            "CACHE_PATH": str(graph_path),
            "TOTAL_LABELED_EDGES": str(total_edges),
            "OUTPUT_ROOT": str(job_root),
            "PARTITION_STRATEGY": strategy,
            "TARGET_SUPER_BLOCK_SIZE": str(BLOCK_EDGES),
            "TARGET_NUM_BLOCKS": str(blocks),
            "NUM_EPOCHS": str(MEMORY_EPOCHS),
            "REPEAT_ID": "1",
            "BASE_SEED": str(BASE_SEED),
            "VARY_SEED_BY_REPEAT": "0",
            "FORWARD_AGG_EXPERIMENT_MODE": mode_name,
            "ENABLE_FORWARD_AGGREGATION": "1",
            "FORWARD_AGGREGATION_OPERATOR": "dense",
            "ENABLE_FORWARD_AGG_OVERLAP": "0",
            "AGGREGATION_NUM_CHUNKS": "1",
            "AGGREGATION_ASYNC_OP": "0",
            "AGGREGATION_DOUBLE_BUFFER": "0",
            "AGGREGATION_CHUNK_POLICY": "destination_range",
            "ENABLE_FORWARD_AGG_TIMING": "1",
            "RUN_PARENT_EXPERIMENTS": "0",
            "COMMOPT_DDP_LAUNCHED": "1",
            "FORCE_SINGLE_PROCESS": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={MEMORY_WORLD_SIZE}",
        str(MEMORY_FRACTION_WORKER),
    ]
    print(
        f"MEMORY BOUNDARY strategy={strategy} fraction={fraction:.2f}",
        flush=True,
    )
    started = time.time()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    lowered = log_text.lower()
    oom_tokens = (
        "out of memory",
        "torch.cuda.outofmemoryerror",
        "cublas_status_alloc_failed",
        "cuda_error_out_of_memory",
    )
    observed_oom = completed.returncode != 0 and any(
        token in lowered for token in oom_tokens
    )
    if completed.returncode != 0 and not observed_oom:
        tail = "\n".join(log_text.splitlines()[-120:])
        print(tail, flush=True)
        raise RuntimeError(
            f"Memory-boundary job failed for a non-OOM reason, "
            f"returncode={completed.returncode}, log={log_path}"
        )

    total_memory = gpu_total_memory_mb()
    row: Dict[str, Any] = {
        "completed": True,
        "graph": GRAPH_TAG,
        "world_size": MEMORY_WORLD_SIZE,
        "strategy": strategy,
        "block_edges": BLOCK_EDGES,
        "chunks": 1,
        "base_mode": "blocking_c1",
        "memory_fraction": fraction,
        "physical_capacity_mb": total_memory,
        "effective_capacity_mb": fraction * total_memory,
        "actual_feasible": completed.returncode == 0,
        "observed_oom": observed_oom,
        "returncode": completed.returncode,
        "elapsed_sec": elapsed,
        "result_path": str(result_path),
        "log_path": str(log_path),
        "observed_peak_memory_mb": None,
    }
    if completed.returncode == 0:
        if not result_path.exists():
            raise RuntimeError(
                f"Successful boundary job did not create {result_path}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        row["observed_peak_memory_mb"] = float(
            payload.get("peak_memory_mb") or 0.0
        )
        if row["observed_peak_memory_mb"] <= 0:
            raise RuntimeError(
                f"Boundary job recorded invalid peak memory: {result_path}"
            )
    write_json_atomic(cell_path, row)
    print(
        f"MEMORY BOUNDARY DONE feasible={row['actual_feasible']} "
        f"oom={row['observed_oom']} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return row


def run_memory_boundary(
    metadata: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows = []
    for fraction in MEMORY_FRACTIONS:
        for strategy in MEMORY_STRATEGIES:
            rows.append(run_memory_boundary_cell(metadata, strategy, fraction))
    write_csv(BOUNDARY_SUMMARY_ROOT / "controlled_boundary.csv", rows)
    summary = {
        "completed": True,
        "evaluation_kind": "observed_allocator_capacity_boundary",
        "cells": rows,
        "feasible_cells": sum(bool(row["actual_feasible"]) for row in rows),
        "oom_cells": sum(bool(row["observed_oom"]) for row in rows),
        "note": (
            "CUDA allocator capacity is controlled with "
            "torch.cuda.set_per_process_memory_fraction; OOM labels are observed, "
            "not inferred from measured peak memory."
        ),
    }
    write_json_atomic(BOUNDARY_SUMMARY_ROOT / "controlled_boundary.json", summary)
    return rows


def main() -> None:
    metadata = preflight()
    started = time.time()
    with publication_gpu_lock("artifact GPU-A supplements"):
        quality_rows = run_clean_scaling_repeats()
        boundary_rows = run_memory_boundary(metadata)
    completion = {
        "completed": True,
        "graph": GRAPH_TAG,
        "clean_scaling_rows": len(quality_rows),
        "all_clean_scaling_rows_valid": all(row["valid"] for row in quality_rows),
        "memory_boundary_rows": len(boundary_rows),
        "memory_boundary_feasible": sum(
            bool(row["actual_feasible"]) for row in boundary_rows
        ),
        "memory_boundary_oom": sum(bool(row["observed_oom"]) for row in boundary_rows),
        "elapsed_sec": time.time() - started,
    }
    write_json_atomic(COMPLETION_PATH, completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
    print(f"COMPLETE: {COMPLETION_PATH}", flush=True)


if __name__ == "__main__":
    main()

