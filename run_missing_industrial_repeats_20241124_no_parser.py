#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete the missing 20241124 Direct/SuperBlock scaling repeats.

This is a no-argument, restartable launcher for the first four-GPU server.
It runs exactly 14 missing cells and writes them under a new publication
subdirectory.  Existing source results are never deleted or overwritten.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Fixed configuration: no command-line parser is required.
# ---------------------------------------------------------------------------
TARGET_GRAPH = "20241124"
PHYSICAL_GPU_IDS = (0, 1, 2, 3)
BLOCK_EDGES = 25_000
EPOCHS = 10
BASE_SEED = 20261004
MODE_NAME = "blocking_c1"
SOURCE_ROOT = Path("/path/to/artifact_workspace/sigmod2027_20241129")
PUBLICATION_ROOT = SOURCE_ROOT / "publication_repeats_20261006"
STAGE = f"representative_scaling_{TARGET_GRAPH}"
LOCK_PATH = SOURCE_ROOT / ".sigmod_publication_gpu.lock"
RUNNER_FILE_NAME = (
    "run_forward_aggregation_overlap_5day_seq_chunk10_repeat2_ep10_no_parser.py"
)

# Set an absolute server path here only if automatic discovery fails.
INDUSTRIAL_RUNNER_OVERRIDE: Path | None = None

# Exact 14-run deficit from industrial_repeat_gap.csv.
JOBS: Tuple[Tuple[int, str, int], ...] = (
    (1, "balanced_edge", 1),
    (1, "balanced_edge", 2),
    (1, "balanced_edge", 3),
    (1, "superblock_edge_count", 1),
    (1, "superblock_edge_count", 2),
    (1, "superblock_edge_count", 3),
    (2, "balanced_edge", 2),
    (2, "balanced_edge", 3),
    (3, "balanced_edge", 2),
    (3, "balanced_edge", 3),
    (3, "superblock_edge_count", 2),
    (3, "superblock_edge_count", 3),
    (4, "balanced_edge", 2),
    (4, "balanced_edge", 3),
)


PROJECT_ROOT = Path(__file__).resolve().parent
STATUS_ROOT = PUBLICATION_ROOT / "stage_status"
STATUS_JSON = STATUS_ROOT / f"{STAGE}.json"
STATUS_CSV = STATUS_ROOT / f"{STAGE}.csv"


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
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


def find_industrial_runner() -> Path:
    if INDUSTRIAL_RUNNER_OVERRIDE is not None:
        return Path(INDUSTRIAL_RUNNER_OVERRIDE)
    candidates = (
        PROJECT_ROOT / RUNNER_FILE_NAME,
        PROJECT_ROOT / "sigmod_experiments" / RUNNER_FILE_NAME,
        PROJECT_ROOT.parent / RUNNER_FILE_NAME,
        Path.cwd() / RUNNER_FILE_NAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    for candidate in PROJECT_ROOT.rglob(RUNNER_FILE_NAME):
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0]


def load_graph_metadata() -> Dict[str, Any]:
    path = SOURCE_ROOT / "multidate_graph_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing multidate metadata: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        if str(row.get("graph_tag")) == TARGET_GRAPH:
            return dict(row)
    raise KeyError(f"Graph {TARGET_GRAPH} is absent from {path}")


def experiment_name(
    total_edges: int, world_size: int, strategy: str, repeat: int
) -> str:
    del world_size  # world size is represented by the parent gpus-N directory.
    blocks = max(1, math.ceil(total_edges / BLOCK_EDGES))
    seed = BASE_SEED + repeat - 1
    return (
        f"forwardagg_{MODE_NAME}_{strategy}_{TARGET_GRAPH}_"
        f"K{blocks}_Q{BLOCK_EDGES}_R{repeat:02d}_ep{EPOCHS}_seed{seed}"
    )


def job_directory(
    total_edges: int, world_size: int, strategy: str, repeat: int
) -> Path:
    return (
        PUBLICATION_ROOT
        / "industrial"
        / STAGE
        / f"gpus-{world_size}"
        / experiment_name(total_edges, world_size, strategy, repeat)
    )


def count_epochs(runtime_path: Path) -> int:
    if not runtime_path.exists():
        return 0
    with runtime_path.open("r", encoding="utf-8-sig", newline="") as stream:
        return sum(1 for row in csv.DictReader(stream) if row.get("event") == "train_epoch")


def inspect_job(
    total_edges: int, world_size: int, strategy: str, repeat: int
) -> Dict[str, Any]:
    directory = job_directory(total_edges, world_size, strategy, repeat)
    result_path = directory / "train_result_rank0.json"
    row: Dict[str, Any] = {
        "graph": TARGET_GRAPH,
        "world_size": world_size,
        "strategy": strategy,
        "repeat": repeat,
        "seed": BASE_SEED + repeat - 1,
        "result_path": str(result_path),
        "accepted": False,
        "epochs_observed": 0,
        "peak_memory_mb": 0.0,
        "reason": "result_missing",
    }
    if not result_path.exists():
        return row
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        row["reason"] = f"invalid_result_json:{type(exc).__name__}"
        return row

    epoch_counts = [
        count_epochs(directory / f"runtime_rank{rank}.csv")
        for rank in range(world_size)
    ]
    row["epochs_observed"] = min(epoch_counts) if epoch_counts else 0
    peak_values: List[float] = []
    for rank in range(world_size):
        rank_result = directory / f"train_result_rank{rank}.json"
        try:
            rank_payload = json.loads(rank_result.read_text(encoding="utf-8"))
            peak_values.append(float(rank_payload.get("peak_memory_mb", 0.0)))
        except Exception:
            peak_values.append(0.0)
    row["peak_memory_mb"] = max(peak_values) if peak_values else 0.0

    reasons: List[str] = []
    if str(payload.get("graph_tag", payload.get("graph_date", ""))) != TARGET_GRAPH:
        reasons.append("graph_mismatch")
    if int(payload.get("world_size", -1)) != world_size:
        reasons.append("world_size_mismatch")
    if str(payload.get("strategy", "")) != strategy:
        reasons.append("strategy_mismatch")
    if int(payload.get("repeat_id", -1)) != repeat:
        reasons.append("repeat_mismatch")
    if str(payload.get("mode", "")) != MODE_NAME:
        reasons.append("mode_mismatch")
    if row["epochs_observed"] < EPOCHS:
        reasons.append(f"epochs_observed<{EPOCHS}")
    if row["peak_memory_mb"] <= 0:
        reasons.append("peak_memory_mb<=0")
    row["accepted"] = not reasons
    row["reason"] = ";".join(reasons)
    return row


def archive_incomplete_job(directory: Path) -> None:
    if not directory.exists():
        return
    resolved_root = PUBLICATION_ROOT.resolve()
    resolved_directory = directory.resolve()
    try:
        resolved_directory.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to move a directory outside publication root: {directory}"
        ) from exc
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = directory.with_name(
        f"{directory.name}__rejected_{stamp}_pid{os.getpid()}"
    )
    print(f"ARCHIVE INCOMPLETE: {directory} -> {destination}", flush=True)
    directory.rename(destination)


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


def run_logged(command: List[str], environment: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(command), flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=str(find_industrial_runner().parent),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    if completed.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-120:]), flush=True)
        raise RuntimeError(
            f"Job failed with returncode={completed.returncode}; log={log_path}"
        )
    print(f"DONE: {elapsed:.1f}s, log={log_path}", flush=True)


def run_job(
    runner: Path,
    graph_path: Path,
    total_edges: int,
    world_size: int,
    strategy: str,
    repeat: int,
) -> Dict[str, Any]:
    directory = job_directory(total_edges, world_size, strategy, repeat)
    before = inspect_job(total_edges, world_size, strategy, repeat)
    if before["accepted"]:
        print(f"SKIP accepted: {before['result_path']}", flush=True)
        return before
    if directory.exists():
        archive_incomplete_job(directory)

    job_root = directory.parent
    blocks = max(1, math.ceil(total_edges / BLOCK_EDGES))
    seed = BASE_SEED + repeat - 1
    selected = PHYSICAL_GPU_IDS[:world_size]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in selected),
            "COMMOPT_FORCE_CUDA": "1",
            "COMMOPT_STRICT_DEVICE": "1",
            "COMMOPT_CUDA_DEVICE_MAP": ",".join(str(value) for value in range(world_size)),
            "COMMOPT_PHYSICAL_GPU_IDS": ",".join(str(value) for value in selected),
            "GRAPH_TAG": TARGET_GRAPH,
            "CACHE_PATH": str(graph_path),
            "TOTAL_LABELED_EDGES": str(total_edges),
            "OUTPUT_ROOT": str(job_root),
            "PARTITION_STRATEGY": strategy,
            "TARGET_SUPER_BLOCK_SIZE": str(BLOCK_EDGES),
            "TARGET_NUM_BLOCKS": str(blocks),
            "NUM_EPOCHS": str(EPOCHS),
            "REPEAT_ID": str(repeat),
            "BASE_SEED": str(seed),
            "VARY_SEED_BY_REPEAT": "0",
            "FORWARD_AGG_EXPERIMENT_MODE": MODE_NAME,
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
        f"--nproc_per_node={world_size}",
        str(runner),
    ]
    log_path = (
        PUBLICATION_ROOT
        / "launcher_logs"
        / STAGE
        / f"g{world_size}__{directory.name}.log"
    )
    run_logged(command, environment, log_path)
    after = inspect_job(total_edges, world_size, strategy, repeat)
    if not after["accepted"]:
        raise RuntimeError(
            f"Job returned zero but failed quality checks: {after['reason']}; "
            f"result={after['result_path']}"
        )
    return after


def snapshot_status(total_edges: int, completed: bool = False) -> Dict[str, Any]:
    rows = [
        inspect_job(total_edges, world_size, strategy, repeat)
        for world_size, strategy, repeat in JOBS
    ]
    accepted = sum(bool(row["accepted"]) for row in rows)
    payload = {
        "completed": bool(completed and accepted == len(JOBS)),
        "graph": TARGET_GRAPH,
        "publication_root": str(PUBLICATION_ROOT),
        "jobs_planned": len(JOBS),
        "jobs_accepted": accepted,
        "jobs_remaining": len(JOBS) - accepted,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    write_json_atomic(STATUS_JSON, payload)
    write_csv(STATUS_CSV, rows)
    return payload


def preflight() -> Tuple[Path, Path, int]:
    if len(JOBS) != 14:
        raise RuntimeError(f"Expected 14 jobs, found {len(JOBS)}")
    runner = find_industrial_runner()
    if not runner.exists():
        raise FileNotFoundError(
            f"Industrial runner not found: {runner}. Set INDUSTRIAL_RUNNER_OVERRIDE."
        )
    metadata = load_graph_metadata()
    graph_path = Path(str(metadata["graph_path"]))
    total_edges = int(metadata["observed_labeled_edges"])
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError(
            f"Four visible GPUs are required; found {torch.cuda.device_count()}"
        )
    checks = {
        "graph": TARGET_GRAPH,
        "graph_path": str(graph_path),
        "total_edges": total_edges,
        "industrial_runner": str(runner),
        "publication_root": str(PUBLICATION_ROOT),
        "jobs": len(JOBS),
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    write_json_atomic(STATUS_ROOT / f"{STAGE}_preflight.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    return runner, graph_path, total_edges


def main() -> None:
    runner, graph_path, total_edges = preflight()
    snapshot_status(total_edges)
    label = f"publication repeats {TARGET_GRAPH} ({len(JOBS)} jobs)"
    with publication_gpu_lock(label):
        for index, (world_size, strategy, repeat) in enumerate(JOBS, start=1):
            print(
                f"[{index}/{len(JOBS)}] graph={TARGET_GRAPH} G={world_size} "
                f"strategy={strategy} repeat={repeat}",
                flush=True,
            )
            run_job(
                runner,
                graph_path,
                total_edges,
                world_size,
                strategy,
                repeat,
            )
            status = snapshot_status(total_edges)
            print(
                f"PROGRESS: {status['jobs_accepted']}/{status['jobs_planned']} "
                f"accepted, {status['jobs_remaining']} remaining",
                flush=True,
            )
    final = snapshot_status(total_edges, completed=True)
    if not final["completed"]:
        raise RuntimeError(
            f"Completion audit failed: {final['jobs_remaining']} jobs remain"
        )
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

