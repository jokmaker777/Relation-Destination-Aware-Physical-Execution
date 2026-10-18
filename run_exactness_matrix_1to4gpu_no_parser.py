#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-argument forward/gradient exactness matrix for 1/2/3/4 GPUs.

The matrix covers blocking/async execution, C=1/2/4/8, natural/largest-first
range order, and sum/mean aggregation.  It also verifies global edge coverage
and a non-overlapping exhaustive destination-range assignment.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from end_to_end_prunning.models.communication_optimized_training_manager_forward_agg_overlap_no_parser import (
    COMMOPT_PATCH_VERSION,
    _DetachedChunkedForwardAllReduceWithBackward,
)


OUTPUT_ROOT = Path(
    os.environ.get(
        "SIGMOD_EXACTNESS_OUTPUT",
        "/path/to/artifact_workspace/sigmod2027_20241129/exactness_matrix_1to4gpu",
    )
)
GPU_COUNTS = (1, 2, 3, 4)
WORKER_FLAG = "SIGMOD_EXACTNESS_WORKER"
NUM_SRC = 4_096
NUM_DST = 3_000
NUM_EDGES = 60_000
HIDDEN_DIM = 32
SEED = 20261004
CONFIGS = (
    {"name": "blocking_c1_mean", "chunks": 1, "async": False, "order": "natural", "aggregation": "mean"},
    {"name": "blocking_c4_mean", "chunks": 4, "async": False, "order": "natural", "aggregation": "mean"},
    {"name": "async_c2_mean", "chunks": 2, "async": True, "order": "natural", "aggregation": "mean"},
    {"name": "async_c4_mean", "chunks": 4, "async": True, "order": "natural", "aggregation": "mean"},
    {"name": "async_c8_mean", "chunks": 8, "async": True, "order": "natural", "aggregation": "mean"},
    {"name": "async_c4_largest_mean", "chunks": 4, "async": True, "order": "largest_first", "aggregation": "mean"},
    {"name": "async_c4_sum", "chunks": 4, "async": True, "order": "natural", "aggregation": "sum"},
)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)


def choose_device(local_rank: int):
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)
        return "hccl", torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return "nccl", torch.device(f"cuda:{local_rank}")
    raise RuntimeError("A CUDA/NPU runtime is required")


def bounds(index: int, chunks: int) -> Tuple[int, int]:
    start = (index * NUM_DST + chunks - 1) // chunks
    end = ((index + 1) * NUM_DST + chunks - 1) // chunks
    return start, min(end, NUM_DST)


def validate_ranges(chunks: int) -> None:
    ranges = [bounds(index, chunks) for index in range(chunks)]
    if ranges[0][0] != 0 or ranges[-1][1] != NUM_DST:
        raise RuntimeError(f"destination ranges are not exhaustive: {ranges}")
    for left, right in zip(ranges, ranges[1:]):
        if left[1] != right[0]:
            raise RuntimeError(f"destination ranges overlap or have a gap: {ranges}")


def global_edges(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    src = torch.randint(NUM_SRC, (NUM_EDGES,), generator=generator, dtype=torch.int64)
    # Zipf-like destination skew without depending on optional libraries.
    uniform = torch.rand(NUM_EDGES, generator=generator)
    dst = torch.floor((uniform.square()) * NUM_DST).long().clamp_max(NUM_DST - 1)
    return torch.stack([src, dst]).to(device)


def reference(
    edges: torch.Tensor,
    features: torch.Tensor,
    probe: torch.Tensor,
    aggregation: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    value = features.detach().clone().requires_grad_(True)
    output = torch.zeros((NUM_DST, HIDDEN_DIM), dtype=value.dtype, device=value.device)
    degree = torch.zeros((NUM_DST, 1), dtype=value.dtype, device=value.device)
    output = output.index_add(0, edges[1], value.index_select(0, edges[0]))
    degree = degree.index_add(
        0,
        edges[1],
        torch.ones((edges.shape[1], 1), dtype=value.dtype, device=value.device),
    )
    if aggregation == "mean":
        output = output / degree.clamp_min(1.0)
    (output * probe).sum().backward()
    return output.detach(), value.grad.detach()


def distributed_operator(
    edges: torch.Tensor,
    features: torch.Tensor,
    probe: torch.Tensor,
    chunks: int,
    async_mode: bool,
    execution_order: str,
    aggregation: str,
    rank: int,
    world: int,
    group,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    local_ids = torch.arange(rank, NUM_EDGES, world, dtype=torch.int64, device=edges.device)
    local_edges = edges.index_select(1, local_ids)
    x = features.detach().clone().requires_grad_(True)
    chunk_edge_counts = []
    for chunk in range(chunks):
        start, end = bounds(chunk, chunks)
        chunk_edge_counts.append(int(((local_edges[1] >= start) & (local_edges[1] < end)).sum().item()))
    order = list(range(chunks))
    if execution_order == "largest_first":
        global_counts = torch.tensor(chunk_edge_counts, dtype=torch.int64, device=edges.device)
        if world > 1:
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=group)
        order.sort(key=lambda index: (-int(global_counts[index].item()), index))

    completed: List[Dict[str, Any] | None] = [None] * chunks
    pending: List[Dict[str, Any]] = []
    started = time.perf_counter()

    def finish(item: Dict[str, Any]) -> Dict[str, Any]:
        for work in item["works"]:
            work.wait()
        return item

    for chunk in order:
        start, end = bounds(chunk, chunks)
        mask = (local_edges[1] >= start) & (local_edges[1] < end)
        selected = local_edges[:, mask]
        local_sum = torch.zeros((end - start, HIDDEN_DIM), dtype=x.dtype, device=x.device)
        local_degree = torch.zeros((end - start, 1), dtype=x.dtype, device=x.device)
        if selected.shape[1] > 0:
            dst = selected[1] - start
            local_sum = local_sum.index_add(0, dst, x.index_select(0, selected[0]))
            local_degree = local_degree.index_add(
                0,
                dst,
                torch.ones((selected.shape[1], 1), dtype=x.dtype, device=x.device),
            )
        if async_mode and pending:
            previous = finish(pending.pop(0))
            completed[int(previous["chunk"])] = previous
        sum_buffer = local_sum.detach().clone()
        degree_buffer = local_degree.detach().clone()
        works = []
        if world > 1:
            if async_mode:
                works = [
                    dist.all_reduce(sum_buffer, op=dist.ReduceOp.SUM, group=group, async_op=True),
                    dist.all_reduce(degree_buffer, op=dist.ReduceOp.SUM, group=group, async_op=True),
                ]
            else:
                dist.all_reduce(sum_buffer, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(degree_buffer, op=dist.ReduceOp.SUM, group=group)
        item = {
            "chunk": chunk,
            "local_sum": local_sum,
            "sum_buffer": sum_buffer,
            "degree_buffer": degree_buffer,
            "works": works,
        }
        if async_mode:
            pending.append(item)
        else:
            completed[chunk] = item

    while pending:
        item = finish(pending.pop(0))
        completed[int(item["chunk"])] = item
    sync(edges.device)
    forward_sec = time.perf_counter() - started
    if any(item is None for item in completed):
        raise RuntimeError("a destination chunk was not produced")
    items = [item for item in completed if item is not None]
    local_sums = [item["local_sum"] for item in items]
    reduced_sums = [item["sum_buffer"] for item in items]
    if world > 1:
        global_sums = _DetachedChunkedForwardAllReduceWithBackward.apply(
            group, *local_sums, *reduced_sums
        )
        if isinstance(global_sums, torch.Tensor):
            global_sums = (global_sums,)
    else:
        global_sums = tuple(local_sums)
    outputs = []
    for value, item in zip(global_sums, items):
        outputs.append(
            value / item["degree_buffer"].clamp_min(1.0)
            if aggregation == "mean"
            else value
        )
    output = torch.cat(outputs, dim=0)
    ((output * probe).sum() / float(world)).backward()
    gradient = x.grad.detach().clone()
    if world > 1:
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=group)
    return output.detach(), gradient, forward_sec


def max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def worker() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    backend, device = choose_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    group = dist.new_group(ranks=list(range(world)), backend=backend)
    edges = global_edges(device)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    features = torch.randn((NUM_SRC, HIDDEN_DIM), generator=generator, device=device)
    probe = torch.randn((NUM_DST, HIDDEN_DIM), generator=generator, device=device)

    local_ids = torch.arange(rank, NUM_EDGES, world, dtype=torch.int64, device=device)
    coverage = torch.tensor(
        [
            local_ids.numel(),
            int(local_ids.sum().item()),
            int((local_ids * local_ids).sum().item()),
        ],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(coverage, op=dist.ReduceOp.SUM)
    expected = torch.tensor(
        [
            NUM_EDGES,
            NUM_EDGES * (NUM_EDGES - 1) // 2,
            NUM_EDGES * (NUM_EDGES - 1) * (2 * NUM_EDGES - 1) // 6,
        ],
        dtype=torch.int64,
        device=device,
    )
    edge_coverage_pass = bool(torch.equal(coverage, expected))
    rows: List[Dict[str, Any]] = []

    for config in CONFIGS:
        validate_ranges(int(config["chunks"]))
        reference_output, reference_gradient = reference(
            edges, features, probe, str(config["aggregation"])
        )
        output, gradient, forward_sec = distributed_operator(
            edges,
            features,
            probe,
            int(config["chunks"]),
            bool(config["async"]),
            str(config["order"]),
            str(config["aggregation"]),
            rank,
            world,
            group,
        )
        output_diff = (output - reference_output).abs()
        gradient_diff = (gradient - reference_gradient).abs()
        output_rel = output_diff / reference_output.abs().clamp_min(1e-7)
        gradient_rel = gradient_diff / reference_gradient.abs().clamp_min(1e-7)
        metrics = {
            "forward_max_abs_error": max_across_ranks(float(output_diff.max().item()), device),
            "forward_mean_abs_error": max_across_ranks(float(output_diff.mean().item()), device),
            "forward_max_rel_error": max_across_ranks(float(output_rel.max().item()), device),
            "gradient_max_abs_error": max_across_ranks(float(gradient_diff.max().item()), device),
            "gradient_mean_abs_error": max_across_ranks(float(gradient_diff.mean().item()), device),
            "gradient_max_rel_error": max_across_ranks(float(gradient_rel.max().item()), device),
            "forward_sec": max_across_ranks(forward_sec, device),
        }
        passed = (
            edge_coverage_pass
            and torch.allclose(output, reference_output, rtol=1e-4, atol=1e-5)
            and torch.allclose(gradient, reference_gradient, rtol=1e-4, atol=1e-5)
        )
        passed_tensor = torch.tensor([1 if passed else 0], dtype=torch.int64, device=device)
        dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
        if rank == 0:
            row = {
                "commopt_patch_version": COMMOPT_PATCH_VERSION,
                "world_size": world,
                "config": config["name"],
                "chunks": config["chunks"],
                "async": config["async"],
                "execution_order": config["order"],
                "aggregation": config["aggregation"],
                "edge_coverage_pass": edge_coverage_pass,
                "destination_range_coverage_pass": True,
                **metrics,
                "passed": bool(passed_tensor.item()),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        dist.barrier()

    if rank == 0:
        directory = OUTPUT_ROOT / f"gpus-{world}"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "exactness_results.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if not all(bool(row["passed"]) for row in rows):
            raise RuntimeError(f"Exactness failed for G={world}")
    dist.barrier()
    dist.destroy_process_group(group)
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
        print(f"RUN EXACTNESS G={world}: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
            raise RuntimeError(f"Exactness G={world} failed:\n{tail}")

    combined = []
    for world in GPU_COUNTS:
        with (OUTPUT_ROOT / f"gpus-{world}" / "exactness_results.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            combined.extend(csv.DictReader(stream))
    with (OUTPUT_ROOT / "exactness_1to4gpu_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(f"ALL 1/2/3/4-GPU FORWARD AND GRADIENT EXACTNESS PASSED: {OUTPUT_ROOT}")


if __name__ == "__main__":
    if os.environ.get(WORKER_FLAG) == "1":
        worker()
    else:
        parent()

