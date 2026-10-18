#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-argument search for faster exact forward-aggregation schedules.

Stage A screens chunk size, global edge-balanced chunking, chunk execution
orders, and deeper in-flight collective queues on two disjoint 2-GPU groups.
Stage B automatically confirms the two best new methods, the blocking baseline,
and the current async-C4 method on all four GPUs without concurrent contention.

Do not run this script while another process is using GPUs 0--3.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from sigmod_experiments.collect_industrial_results import (
    collect as collect_industrial_result,
)


SCHEDULER_PATCH_VERSION = "20261004-relation-group-v2"
PROJECT_ROOT = Path(__file__).resolve().parent
RUNNER = PROJECT_ROOT / (
    "run_forward_aggregation_overlap_5day_seq_chunk10_repeat2_ep10_no_parser.py"
)
METADATA_PATH = Path(
    "/path/to/artifact_workspace/sigmod2027_20241129/"
    "multidate_graph_metadata.json"
)
OUTPUT_ROOT = Path(
    "/path/to/artifact_workspace/sigmod2027_20241129/async_scheduler_search"
)
GRAPH_TAGS = ("20241124", "20241129", "20241213", "20241218")
GPU_POOLS: Tuple[Tuple[int, ...], ...] = ((0, 1), (2, 3))
FOUR_GPU_POOL = (0, 1, 2, 3)
PARTITION_STRATEGY = "superblock_edge_count"
BLOCK_EDGES = 25_000
SCREEN_EPOCHS = 6
CONFIRM_EPOCHS = 10
SEED = 20261004
REUSE_HORIZON_EPOCHS = 50
PRINT_LOCK = threading.Lock()


GPU_POOL_PROBE = r"""
import json
import os
import torch

device_ids = json.loads(os.environ["COMMOPT_CUDA_DEVICE_IDS_JSON"])
payload = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()),
    "devices": [],
}
if not payload["cuda_available"]:
    raise RuntimeError("torch.cuda.is_available() is False")
if not device_ids:
    raise RuntimeError("The CUDA device pool is empty")
if min(device_ids) < 0 or max(device_ids) >= payload["device_count"]:
    raise RuntimeError(
        f"Requested physical CUDA devices {device_ids}, but PyTorch sees "
        f"logical device IDs 0..{payload['device_count'] - 1}"
    )
for index in device_ids:
    torch.cuda.set_device(index)
    probe = torch.ones(1, device=f"cuda:{index}")
    torch.cuda.synchronize(index)
    payload["devices"].append(
        {
            "logical_id": index,
            "name": torch.cuda.get_device_name(index),
            "probe": float(probe.item()),
        }
    )
print(json.dumps(payload, ensure_ascii=False))
"""


def method(
    name: str,
    chunks: int,
    overlap: bool,
    chunk_policy: str = "destination_range",
    chunk_order: str = "natural",
    max_inflight: int = 1,
    fuse_degree: bool = False,
) -> Dict[str, Any]:
    return {
        "name": name,
        "chunks": int(chunks),
        "overlap": bool(overlap),
        "chunk_policy": chunk_policy,
        "chunk_order": chunk_order,
        "max_inflight": int(max_inflight),
        "fuse_degree": bool(fuse_degree),
    }


METHODS: Tuple[Dict[str, Any], ...] = (
    method("sched_blocking_c1", 1, False),
    method("sched_async_c2_range_natural_d1", 2, True),
    method("sched_async_c4_range_natural_d1", 4, True),
    method("sched_async_c8_range_natural_d1", 8, True),
    method(
        "sched_async_c4_edge_balanced_natural_d1",
        4,
        True,
        chunk_policy="global_edge_balanced",
    ),
    method(
        "sched_async_c4_range_largest_d1",
        4,
        True,
        chunk_order="largest_first",
    ),
    method(
        "sched_async_c4_range_smallest_d1",
        4,
        True,
        chunk_order="smallest_first",
    ),
    method(
        "sched_async_c4_range_greedy_d1",
        4,
        True,
        chunk_order="overlap_greedy",
    ),
    method(
        "sched_async_c4_range_greedy_d2",
        4,
        True,
        chunk_order="overlap_greedy",
        max_inflight=2,
    ),
    method(
        "sched_async_c8_edge_greedy_d2",
        8,
        True,
        chunk_policy="global_edge_balanced",
        chunk_order="overlap_greedy",
        max_inflight=2,
    ),
    method(
        "sched_async_c4_edge_greedy_d2_fused",
        4,
        True,
        chunk_policy="global_edge_balanced",
        chunk_order="overlap_greedy",
        max_inflight=2,
        fuse_degree=True,
    ),
)

BASELINE_NAME = "sched_blocking_c1"
CURRENT_NAME = "sched_async_c4_range_natural_d1"


def safe_print(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def load_metadata() -> Dict[str, Dict[str, Any]]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {METADATA_PATH}; run "
            "sigmod_experiments/server_no_parser/08_multidate_preflight_no_parser.py first"
        )
    rows = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    by_tag = {str(row["graph_tag"]): row for row in rows}
    missing = [tag for tag in GRAPH_TAGS if tag not in by_tag]
    if missing:
        raise RuntimeError(f"Metadata is missing graph dates: {missing}")
    return by_tag


def assert_no_existing_gpu_jobs() -> None:
    """Protect both the old experiment and this benchmark from GPU contention."""
    if os.environ.get("ALLOW_BUSY_GPUS", "0") == "1":
        return
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return
    active = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode == 0 and active:
        details = "\n".join(f"  {line}" for line in active)
        raise RuntimeError(
            "GPUs already have active compute processes. Wait for the current "
            "72-job experiment to finish before running this search; otherwise "
            f"both measurements are invalid. Active processes:\n{details}"
        )


def validate_gpu_pools() -> None:
    """Fail before graph loading if either isolated CUDA pool is unusable."""
    pools = list(GPU_POOLS) + [FOUR_GPU_POOL]
    for gpu_ids in pools:
        environment = os.environ.copy()
        # This server's PyTorch build does not honor per-child
        # CUDA_VISIBLE_DEVICES consistently.  Keep all assigned devices visible
        # and select the requested physical IDs explicitly in every DDP rank.
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "COMMOPT_CUDA_DEVICE_IDS_JSON": json.dumps(list(gpu_ids)),
            }
        )
        completed = subprocess.run(
            [sys.executable, "-c", GPU_POOL_PROBE],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = completed.stdout.strip()
        if completed.returncode != 0:
            raise RuntimeError(
                f"CUDA pool preflight failed for physical GPUs {tuple(gpu_ids)}. "
                f"The scheduler search was not started.\n{output}"
            )
        safe_print(f"CUDA POOL OK physical_GPUs={tuple(gpu_ids)}: {output}")


def experiment_name(
    graph_tag: str,
    method_config: Mapping[str, Any],
    epochs: int,
    repeat: int,
    total_edges: int,
) -> str:
    blocks = max(1, math.ceil(int(total_edges) / BLOCK_EDGES))
    return (
        f"forwardagg_{method_config['name']}_{PARTITION_STRATEGY}_{graph_tag}_"
        f"K{blocks}_Q{BLOCK_EDGES}_R{repeat:02d}_ep{epochs}_seed{SEED}"
    )


def result_is_complete(
    path: Path,
    expected_epochs: int | None = None,
    expected_world_size: int | None = None,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("mode"):
            return False
        declared_epochs = int(payload.get("num_epochs", 0))
        required_epochs = int(expected_epochs or declared_epochs)
        world_size = int(payload.get("world_size", 0))
        if required_epochs <= 0 or declared_epochs < required_epochs or world_size <= 0:
            return False
        if expected_world_size is not None and world_size != int(expected_world_size):
            return False
        final_loss = float(payload.get("final_loss_extracted"))
        if not math.isfinite(final_loss):
            return False

        result_dir = path.parent
        expected_mode = str(payload["mode"])
        for rank in range(world_size):
            rank_result = result_dir / f"train_result_rank{rank}.json"
            rank_payload = json.loads(rank_result.read_text(encoding="utf-8"))
            if str(rank_payload.get("mode")) != expected_mode:
                return False
            rank_loss = float(rank_payload.get("final_loss_extracted"))
            if not math.isfinite(rank_loss):
                return False
            runtime_path = result_dir / f"runtime_rank{rank}.csv"
            with runtime_path.open("r", encoding="utf-8-sig", newline="") as stream:
                observed_epochs = sum(
                    1 for row in csv.DictReader(stream)
                    if row.get("event") == "train_epoch"
                )
            if observed_epochs < required_epochs:
                return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def run_job(
    stage: str,
    gpu_ids: Sequence[int],
    graph_row: Mapping[str, Any],
    method_config: Mapping[str, Any],
    epochs: int,
    repeat: int = 1,
) -> Path:
    world_size = len(gpu_ids)
    graph_tag = str(graph_row["graph_tag"])
    total_edges = int(graph_row["observed_labeled_edges"])
    name = experiment_name(
        graph_tag, method_config, epochs, repeat, total_edges
    )
    job_root = OUTPUT_ROOT / stage / f"gpus-{world_size}"
    result_path = job_root / name / "train_result_rank0.json"
    if result_is_complete(result_path, epochs, world_size):
        safe_print(f"SKIP complete: {result_path}")
        return result_path

    log_path = OUTPUT_ROOT / "logs" / stage / f"g{world_size}__{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    job_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "COMMOPT_CUDA_DEVICE_MAP": ",".join(str(value) for value in gpu_ids),
            "COMMOPT_FORCE_CUDA": "1",
            "COMMOPT_STRICT_DEVICE": "1",
            "GRAPH_TAG": graph_tag,
            "CACHE_PATH": str(graph_row["graph_path"]),
            "TOTAL_LABELED_EDGES": str(total_edges),
            "OUTPUT_ROOT": str(job_root),
            "PARTITION_STRATEGY": PARTITION_STRATEGY,
            "TARGET_SUPER_BLOCK_SIZE": str(BLOCK_EDGES),
            "TARGET_NUM_BLOCKS": str(max(1, math.ceil(total_edges / BLOCK_EDGES))),
            "NUM_EPOCHS": str(epochs),
            "REPEAT_ID": str(repeat),
            "BASE_SEED": str(SEED),
            "VARY_SEED_BY_REPEAT": "0",
            "FORWARD_AGG_EXPERIMENT_MODE": str(method_config["name"]),
            "ENABLE_FORWARD_AGGREGATION": "1",
            "ENABLE_FORWARD_AGG_OVERLAP": (
                "1" if method_config["overlap"] else "0"
            ),
            "AGGREGATION_NUM_CHUNKS": str(method_config["chunks"]),
            "AGGREGATION_ASYNC_OP": "1" if method_config["overlap"] else "0",
            "AGGREGATION_DOUBLE_BUFFER": (
                "1" if method_config["overlap"] else "0"
            ),
            "AGGREGATION_CHUNK_POLICY": str(method_config["chunk_policy"]),
            "AGGREGATION_CHUNK_ORDER": str(method_config["chunk_order"]),
            "AGGREGATION_MAX_INFLIGHT": str(method_config["max_inflight"]),
            "AGGREGATION_FUSE_DEGREE": (
                "1" if method_config["fuse_degree"] else "0"
            ),
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
        str(RUNNER),
    ]
    safe_print(
        f"RUN stage={stage} graph={graph_tag} method={method_config['name']} "
        f"GPUs={list(gpu_ids)} log={log_path}"
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
    if completed.returncode != 0 or not result_is_complete(
        result_path, epochs, world_size
    ):
        try:
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                    -100:
                ]
            )
        except OSError as exc:
            tail = f"Cannot read log: {exc}"
        safe_print(f"FAILED {name}\n{tail}")
        raise RuntimeError(
            f"Job failed: stage={stage}, graph={graph_tag}, "
            f"method={method_config['name']}, log={log_path}"
        )
    safe_print(f"DONE {name}: {elapsed:.1f}s")
    return result_path


def run_parallel_screen(metadata: Mapping[str, Mapping[str, Any]]) -> None:
    failures: List[Dict[str, Any]] = []
    total = len(GRAPH_TAGS) * len(METHODS)
    completed = 0
    safe_print(
        f"START 2-GPU PARALLEL SCREEN: {total} jobs, "
        f"GPU pools={GPU_POOLS}, epochs={SCREEN_EPOCHS}"
    )
    for date_index, graph_tag in enumerate(GRAPH_TAGS):
        safe_print("=" * 100)
        safe_print(f"SCREEN DATE {graph_tag}")
        for pair_start in range(0, len(METHODS), len(GPU_POOLS)):
            batch = METHODS[pair_start : pair_start + len(GPU_POOLS)]
            pools = list(GPU_POOLS)
            if (date_index + pair_start // len(GPU_POOLS)) % 2:
                pools.reverse()
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    executor.submit(
                        run_job,
                        "screen_2gpu",
                        pools[index],
                        metadata[graph_tag],
                        method_config,
                        SCREEN_EPOCHS,
                    ): method_config
                    for index, method_config in enumerate(batch)
                }
                for future in as_completed(futures):
                    method_config = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        failure = {
                            "graph_tag": graph_tag,
                            "method": method_config["name"],
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        safe_print(
                            f"RECORDED FAILURE graph={graph_tag}, "
                            f"method={method_config['name']}"
                        )
                    completed += 1
                    safe_print(f"SCREEN PROGRESS {completed}/{total}")
            if len(failures) >= 4 and completed <= len(METHODS):
                raise RuntimeError(
                    "At least four jobs failed during the first date; stopping "
                    "to avoid repeating a systematic error."
                )
    failure_path = OUTPUT_ROOT / "screen_failures.json"
    if failures:
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        safe_print(f"SCREEN FINISHED WITH {len(failures)} FAILURES: {failure_path}")
    elif failure_path.exists():
        failure_path.unlink()


def collect_result(result_path: Path) -> Dict[str, Any]:
    row = collect_industrial_result(result_path, REUSE_HORIZON_EPOCHS)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    row.update(
        {
            "method": str(payload.get("mode", "unknown")),
            "chunk_policy": str(
                payload.get("aggregation_chunk_policy", "destination_range")
            ),
            "chunk_order": str(payload.get("aggregation_chunk_order", "natural")),
            "max_inflight": int(payload.get("aggregation_max_inflight", 1)),
            "fuse_degree": bool(payload.get("aggregation_fuse_degree", False)),
        }
    )
    return row


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rank_stage(stage: str, expected_graph_count: int) -> List[Dict[str, Any]]:
    stage_root = OUTPUT_ROOT / stage
    result_paths = sorted(stage_root.rglob("train_result_rank0.json"))
    rows = [collect_result(path) for path in result_paths if result_is_complete(path)]
    if not rows:
        raise RuntimeError(f"No completed results under {stage_root}")
    write_csv(OUTPUT_ROOT / f"{stage}_measurements.csv", rows)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    ranking: List[Dict[str, Any]] = []
    for method_name, method_rows in grouped.items():
        sample = method_rows[0]
        steady_values = [float(row["steady_epoch_sec_mean"]) for row in method_rows]
        total_values = [float(row["estimated_total_sec_at_horizon"]) for row in method_rows]
        ranking.append(
            {
                "method": method_name,
                "rank": 0,
                "graph_count": len({str(row["graph"]) for row in method_rows}),
                "result_count": len(method_rows),
                "complete_for_selection": (
                    len({str(row["graph"]) for row in method_rows})
                    == expected_graph_count
                ),
                "chunks": int(sample["chunks"]),
                "chunk_policy": sample["chunk_policy"],
                "chunk_order": sample["chunk_order"],
                "max_inflight": int(sample["max_inflight"]),
                "fuse_degree": bool(sample["fuse_degree"]),
                "steady_epoch_sec_mean": mean(steady_values),
                "steady_epoch_sec_median": median(steady_values),
                "prep_sec_mean": mean(float(row["prep_sec_max"]) for row in method_rows),
                "comm_exposed_sec_mean": mean(
                    float(row["comm_exposed_sec_mean"]) for row in method_rows
                ),
                "overlap_ratio_mean": mean(
                    float(row["overlap_ratio_mean"]) for row in method_rows
                ),
                "total_horizon_sec_mean": mean(total_values),
            }
        )

    eligible = [row for row in ranking if row["complete_for_selection"]]
    baseline = next(
        (row for row in eligible if row["method"] == BASELINE_NAME), None
    )
    baseline_time = (
        float(baseline["steady_epoch_sec_mean"]) if baseline is not None else 0.0
    )
    ranking.sort(
        key=lambda row: (
            not bool(row["complete_for_selection"]),
            float(row["steady_epoch_sec_mean"]),
        )
    )
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
        value = float(row["steady_epoch_sec_mean"])
        row["speedup_vs_blocking"] = (
            baseline_time / value
            if bool(row["complete_for_selection"]) and baseline_time > 0 and value > 0
            else None
        )
    write_csv(OUTPUT_ROOT / f"{stage}_ranking.csv", ranking)
    safe_print(f"RANKING {stage}:")
    for row in ranking:
        speedup = row["speedup_vs_blocking"]
        speedup_text = f"{float(speedup):.3f}x" if speedup is not None else "N/A"
        safe_print(
            f"  #{row['rank']} {row['method']}: "
            f"steady={row['steady_epoch_sec_mean']:.4f}s, "
            f"speedup={speedup_text}, "
            f"graphs={row['graph_count']}"
        )
    return ranking


def choose_confirmation_methods(
    screen_ranking: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_name = {str(value["name"]): value for value in METHODS}
    selected_names = [BASELINE_NAME, CURRENT_NAME]
    for row in screen_ranking:
        name = str(row["method"])
        if not bool(row["complete_for_selection"]):
            continue
        if name in selected_names:
            continue
        selected_names.append(name)
        if len(selected_names) >= 4:
            break
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise RuntimeError(f"Unknown selected methods: {missing}")
    return [by_name[name] for name in selected_names]


def run_four_gpu_confirmation(
    metadata: Mapping[str, Mapping[str, Any]],
    selected_methods: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    total = len(GRAPH_TAGS) * len(selected_methods)
    index = 0
    safe_print(
        f"START 4-GPU CONFIRMATION: {total} jobs, epochs={CONFIRM_EPOCHS}, "
        f"methods={[value['name'] for value in selected_methods]}"
    )
    failures: List[Dict[str, Any]] = []
    for graph_tag in GRAPH_TAGS:
        for method_config in selected_methods:
            index += 1
            safe_print(
                f"CONFIRM {index}/{total}: graph={graph_tag}, "
                f"method={method_config['name']}"
            )
            try:
                run_job(
                    "confirm_4gpu",
                    FOUR_GPU_POOL,
                    metadata[graph_tag],
                    method_config,
                    CONFIRM_EPOCHS,
                )
            except Exception as exc:
                failures.append(
                    {
                        "graph_tag": graph_tag,
                        "method": method_config["name"],
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                safe_print(f"CONFIRM FAILURE RECORDED: {failures[-1]}")
    if failures:
        path = OUTPUT_ROOT / "confirm_failures.json"
        path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        safe_print(f"CONFIRMATION FINISHED WITH {len(failures)} FAILURES: {path}")
    return failures


def main() -> None:
    safe_print(f"ASYNC SCHEDULER PATCH: {SCHEDULER_PATCH_VERSION}")
    if not RUNNER.exists():
        raise FileNotFoundError(f"Missing runner: {RUNNER}")
    assert_no_existing_gpu_jobs()
    validate_gpu_pools()
    metadata = load_metadata()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_parallel_screen(metadata)
    screen_ranking = rank_stage("screen_2gpu", len(GRAPH_TAGS))
    selected = choose_confirmation_methods(screen_ranking)
    selection_path = OUTPUT_ROOT / "selected_for_4gpu.json"
    selection_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    safe_print(f"WROTE 4-GPU selection: {selection_path}")
    failures = run_four_gpu_confirmation(metadata, selected)
    rank_stage("confirm_4gpu", len(GRAPH_TAGS))
    if failures:
        raise RuntimeError(
            f"4-GPU confirmation had {len(failures)} failed job(s); "
            "incomplete methods were excluded from speedup claims."
        )
    safe_print("ASYNC SCHEDULER SEARCH AND CONFIRMATION COMPLETED")
    safe_print(f"RESULTS: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

