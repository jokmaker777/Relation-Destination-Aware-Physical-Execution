#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure and validate the two-stream range recurrence on 2/3/4 GPUs."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist


OUTPUT_ROOT = Path(
    os.environ.get(
        "SIGMOD_PIPELINE_RECURRENCE_OUTPUT",
        "/path/to/artifact_workspace/sigmod2027_20241129/pipeline_recurrence_2to4gpu",
    )
)
GPU_COUNTS = (2, 3, 4)
CHUNK_CHOICES = (1, 2, 4, 8)
NUM_SRC = 100_000
NUM_DST = 100_000
NUM_EDGES = 1_000_000
HIDDEN_DIM = 32
WARMUPS = 1
REPEATS = 4
SEED = 20261004
WORKER_FLAG = "SIGMOD_PIPELINE_RECURRENCE_WORKER"


def choose_device(local_rank: int):
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)
        return "hccl", torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return "nccl", torch.device(f"cuda:{local_rank}")
    raise RuntimeError("A CUDA/NPU runtime is required")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)


def critical(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def bounds(chunk: int, chunks: int) -> Tuple[int, int]:
    start = (chunk * NUM_DST + chunks - 1) // chunks
    end = ((chunk + 1) * NUM_DST + chunks - 1) // chunks
    return start, min(end, NUM_DST)


def local_chunk(
    local_edges: torch.Tensor,
    features: torch.Tensor,
    chunk: int,
    chunks: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    start, end = bounds(chunk, chunks)
    mask = (local_edges[1] >= start) & (local_edges[1] < end)
    selected = local_edges[:, mask]
    output = torch.zeros((end - start, HIDDEN_DIM), dtype=features.dtype, device=features.device)
    degree = torch.zeros((end - start, 1), dtype=features.dtype, device=features.device)
    if selected.shape[1] > 0:
        dst = selected[1] - start
        output.index_add_(0, dst, features.index_select(0, selected[0]))
        degree.index_add_(
            0,
            dst,
            torch.ones((selected.shape[1], 1), dtype=features.dtype, device=features.device),
        )
    return output, degree, int(selected.shape[1])


def measure_compute(
    local_edges: torch.Tensor,
    features: torch.Tensor,
    chunk: int,
    chunks: int,
    device: torch.device,
) -> Tuple[float, int]:
    values = []
    edge_count = 0
    for repeat in range(WARMUPS + REPEATS):
        dist.barrier()
        sync(device)
        started = time.perf_counter()
        output, degree, edge_count = local_chunk(local_edges, features, chunk, chunks)
        sync(device)
        elapsed = critical(time.perf_counter() - started, device)
        if repeat >= WARMUPS:
            values.append(elapsed)
        del output, degree
    edge_tensor = torch.tensor([edge_count], dtype=torch.int64, device=device)
    dist.all_reduce(edge_tensor, op=dist.ReduceOp.SUM)
    return statistics.median(values), int(edge_tensor.item())


def measure_collective(
    chunk: int, chunks: int, device: torch.device
) -> float:
    start, end = bounds(chunk, chunks)
    values = []
    for repeat in range(WARMUPS + REPEATS):
        output = torch.ones((end - start, HIDDEN_DIM), dtype=torch.float32, device=device)
        degree = torch.ones((end - start, 1), dtype=torch.float32, device=device)
        dist.barrier()
        sync(device)
        started = time.perf_counter()
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        dist.all_reduce(degree, op=dist.ReduceOp.SUM)
        sync(device)
        elapsed = critical(time.perf_counter() - started, device)
        if repeat >= WARMUPS:
            values.append(elapsed)
        del output, degree
    return statistics.median(values)


def run_pipeline(
    local_edges: torch.Tensor,
    features: torch.Tensor,
    chunks: int,
    async_mode: bool,
    device: torch.device,
) -> float:
    values = []
    for repeat in range(WARMUPS + REPEATS):
        dist.barrier()
        sync(device)
        started = time.perf_counter()
        pending = None
        retained = []
        for chunk in range(chunks):
            output, degree, _ = local_chunk(local_edges, features, chunk, chunks)
            if pending is not None:
                pending[0].wait()
                pending[1].wait()
                pending = None
            if async_mode:
                pending = (
                    dist.all_reduce(output, op=dist.ReduceOp.SUM, async_op=True),
                    dist.all_reduce(degree, op=dist.ReduceOp.SUM, async_op=True),
                )
            else:
                dist.all_reduce(output, op=dist.ReduceOp.SUM)
                dist.all_reduce(degree, op=dist.ReduceOp.SUM)
            retained.append((output, degree))
        if pending is not None:
            pending[0].wait()
            pending[1].wait()
        sync(device)
        elapsed = critical(time.perf_counter() - started, device)
        if repeat >= WARMUPS:
            values.append(elapsed)
        del retained
    return statistics.median(values)


def recurrence(a_values: List[float], r_values: List[float]) -> Tuple[float, float]:
    compute_end = 0.0
    communication_end = 0.0
    for compute, communication in zip(a_values, r_values):
        compute_end += compute
        communication_end = max(compute_end, communication_end) + communication
    return communication_end, max(0.0, communication_end - compute_end)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def worker() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    backend, device = choose_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    cpu_generator = torch.Generator(device="cpu").manual_seed(SEED)
    src = torch.randint(NUM_SRC, (NUM_EDGES,), generator=cpu_generator, dtype=torch.int64)
    uniform = torch.rand(NUM_EDGES, generator=cpu_generator)
    dst = torch.floor(uniform.square() * NUM_DST).long().clamp_max(NUM_DST - 1)
    full_edges = torch.stack([src, dst]).to(device)
    local_ids = torch.arange(rank, NUM_EDGES, world, dtype=torch.int64, device=device)
    local_edges = full_edges.index_select(1, local_ids)
    feature_generator = torch.Generator(device=device).manual_seed(SEED + 1)
    features = torch.randn((NUM_SRC, HIDDEN_DIM), generator=feature_generator, device=device)

    range_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for chunks in CHUNK_CHOICES:
        a_values = []
        r_values = []
        range_metadata = []
        for chunk in range(chunks):
            compute_sec, global_edges = measure_compute(
                local_edges, features, chunk, chunks, device
            )
            collective_sec = measure_collective(chunk, chunks, device)
            start, end = bounds(chunk, chunks)
            bytes_per_collective = (end - start) * (HIDDEN_DIM + 1) * 4
            a_values.append(compute_sec)
            r_values.append(collective_sec)
            range_metadata.append((start, end, global_edges, bytes_per_collective))
        predicted, terminal_tail = recurrence(a_values, r_values)
        blocking = run_pipeline(local_edges, features, chunks, False, device)
        async_time = run_pipeline(local_edges, features, chunks, True, device)
        signed_overhead = async_time - predicted
        for chunk, (compute_sec, collective_sec, metadata) in enumerate(
            zip(a_values, r_values, range_metadata)
        ):
            start, end, global_edges, payload_bytes = metadata
            range_rows.append(
                {
                    "world_size": world,
                    "chunks": chunks,
                    "range_id": chunk,
                    "dst_start": start,
                    "dst_end": end,
                    "global_edges": global_edges,
                    "payload_bytes_two_collectives": payload_bytes,
                    "a_c_local_compute_sec": compute_sec,
                    "r_c_collective_sec": collective_sec,
                    "terminal_range": chunk == chunks - 1,
                }
            )
        summaries.append(
            {
                "world_size": world,
                "chunks": chunks,
                "sum_a_c_sec": sum(a_values),
                "sum_r_c_sec": sum(r_values),
                "predicted_pipeline_sec": predicted,
                "measured_blocking_pipeline_sec": blocking,
                "measured_async_pipeline_sec": async_time,
                "terminal_tail_sec": terminal_tail,
                "signed_launch_event_index_overhead_sec": signed_overhead,
                "clipped_overhead_sec": max(0.0, signed_overhead),
                "prediction_absolute_percentage_error": abs(predicted - async_time) / max(async_time, 1e-12),
            }
        )
        if rank == 0:
            print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)

    baseline = next(row["measured_blocking_pipeline_sec"] for row in summaries if row["chunks"] == 1)
    for row in summaries:
        row["predicted_net_gain_vs_blocking_c1_sec"] = baseline - row["predicted_pipeline_sec"]
        row["measured_net_gain_vs_blocking_c1_sec"] = baseline - row["measured_async_pipeline_sec"]
        row["fallback_predicted_mode"] = (
            "async" if row["predicted_net_gain_vs_blocking_c1_sec"] > 0 else "blocking"
        )
        row["fallback_oracle_mode"] = (
            "async" if row["measured_net_gain_vs_blocking_c1_sec"] > 0 else "blocking"
        )
        chosen = (
            row["measured_async_pipeline_sec"]
            if row["fallback_predicted_mode"] == "async"
            else baseline
        )
        oracle = min(baseline, row["measured_async_pipeline_sec"])
        row["fallback_regret"] = (chosen - oracle) / max(oracle, 1e-12)
        row["fallback_correct"] = row["fallback_predicted_mode"] == row["fallback_oracle_mode"]

    if rank == 0:
        directory = OUTPUT_ROOT / f"gpus-{world}"
        write_csv(directory / "range_a_c_r_c.csv", range_rows)
        write_csv(directory / "pipeline_recurrence_summary.csv", summaries)
    dist.barrier()
    dist.destroy_process_group()


def parent() -> None:
    if not torch.cuda.is_available() and not (
        hasattr(torch, "npu") and torch.npu.is_available()
    ):
        raise RuntimeError("No GPU/NPU runtime is available")
    available = torch.cuda.device_count() if torch.cuda.is_available() else torch.npu.device_count()
    if available < max(GPU_COUNTS):
        raise RuntimeError(f"Expected at least four accelerators, found {available}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for world in GPU_COUNTS:
        environment = os.environ.copy()
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment[WORKER_FLAG] = "1"
        log = OUTPUT_ROOT / f"gpus-{world}.log"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            str(Path(__file__).resolve()),
        ]
        print(f"RUN PIPELINE RECURRENCE G={world}: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parent),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
            raise RuntimeError(f"Pipeline recurrence G={world} failed:\n{tail}")

    ranges = []
    summaries = []
    for world in GPU_COUNTS:
        with (OUTPUT_ROOT / f"gpus-{world}" / "range_a_c_r_c.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            ranges.extend(csv.DictReader(stream))
        with (OUTPUT_ROOT / f"gpus-{world}" / "pipeline_recurrence_summary.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            summaries.extend(csv.DictReader(stream))
    write_csv(OUTPUT_ROOT / "range_a_c_r_c_2to4gpu.csv", ranges)
    write_csv(OUTPUT_ROOT / "pipeline_recurrence_2to4gpu_summary.csv", summaries)
    print(f"PIPELINE RECURRENCE 2/3/4-GPU COMPLETED: {OUTPUT_ROOT}")


if __name__ == "__main__":
    if os.environ.get(WORKER_FLAG) == "1":
        worker()
    else:
        parent()

