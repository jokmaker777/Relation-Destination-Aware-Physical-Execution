#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measured four-GPU phase boundary for active-destination communication.

Direct-run, no argparse.  The parent launches this same file with torchrun.
The worker compares four exact aggregation plans:

  direct_dense             edge-balanced layout + N_dst x d all-reduce
  direct_compact           edge-balanced layout + A_dst x d all-reduce
  superblock_dense         destination-grouped SuperBlocks + dense all-reduce
  superblock_compact       destination-grouped SuperBlocks + compact all-reduce

The matrix changes active-destination ratio, hidden dimension, relation count,
graph scale, and degree skew.  Timings include local gather/index_add, NCCL
all-reduce, exact mean normalization, and autograd all-reduce backward.  Each
case also checks forward and globally accumulated feature-gradient equivalence
against direct_dense.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(
    "/path/to/artifact_workspace/sigmod2027_20241129/active_destination_phase_boundary_4gpu"
)
GPU_IDS = (0, 1, 2, 3)
WORLD_SIZE = 4
WARMUPS = 2
REPEATS = 5
BLOCK_EDGES = 25_000
BYTES_PER_VALUE = 4
BASE_SEED = 20261004
METHODS = (
    "direct_dense",
    "direct_compact",
    "superblock_dense",
    "superblock_compact",
)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def completed(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("completed"))
    except Exception:
        return False


def safe_name(value: Any) -> str:
    return str(value).replace(".", "p")


def cases() -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    def add(axis, value, edges, nodes, relations, active, skew, hidden):
        output.append(
            {
                "axis": axis,
                "axis_value": value,
                "total_edges": int(edges),
                "num_dst": int(nodes),
                "num_relations": int(relations),
                "active_ratio": float(active),
                "skew": float(skew),
                "hidden_dim": int(hidden),
                "case_id": (
                    f"{axis}-{safe_name(value)}__e{int(edges)}__n{int(nodes)}"
                    f"__r{int(relations)}__a{safe_name(active)}"
                    f"__s{safe_name(skew)}__h{int(hidden)}"
                ),
            }
        )

    for ratio in (0.01, 0.05, 0.10, 0.25, 0.50, 1.00):
        add("active_ratio", ratio, 1_000_000, 250_000, 4, ratio, 1.0, 64)
    for hidden in (16, 32, 64, 128, 256):
        add("hidden_dim", hidden, 1_000_000, 250_000, 4, 0.25, 1.0, hidden)
    for relations in (1, 2, 4, 8, 16, 32):
        add("relation_count", relations, 1_000_000, 50_000, relations, 0.25, 1.0, 64)
    for edges in (100_000, 500_000, 1_000_000, 2_000_000):
        add("scale_edges", edges, edges, max(10_000, edges // 8), 4, 0.25, 1.0, 64)
    for skew in (0.0, 0.5, 1.0, 1.5, 2.0):
        add("degree_skew", skew, 1_000_000, 250_000, 4, 0.25, skew, 64)
    return output


def generate_relation(
    num_edges: int,
    num_src: int,
    num_dst: int,
    active_ratio: float,
    skew: float,
    seed: int,
) -> Dict[str, Any]:
    import torch

    generator = torch.Generator().manual_seed(seed)
    active_count = max(1, min(num_dst, num_edges, int(round(num_dst * active_ratio))))
    coverage = torch.arange(active_count, dtype=torch.long)
    remaining = num_edges - active_count
    if remaining > 0:
        uniform = torch.rand((remaining,), generator=generator, dtype=torch.float64)
        sampled = torch.clamp(
            (uniform.pow(1.0 + max(0.0, skew)) * active_count).long(),
            max=active_count - 1,
        )
        dst = torch.cat([coverage, sampled])
    else:
        dst = coverage
    dst = dst.index_select(0, torch.randperm(num_edges, generator=generator))
    src = torch.randint(num_src, (num_edges,), generator=generator)
    degree = torch.bincount(dst, minlength=num_dst)
    active = torch.nonzero(degree > 0, as_tuple=False).view(-1)
    return {
        "src": src,
        "dst": dst,
        "degree": degree,
        "active": active,
        "num_src": num_src,
        "num_dst": num_dst,
        "num_edges": num_edges,
    }


def make_relations(case: Mapping[str, Any]) -> List[Dict[str, Any]]:
    relation_count = int(case["num_relations"])
    total_edges = int(case["total_edges"])
    base = total_edges // relation_count
    remainder = total_edges % relation_count
    result = []
    for relation_id in range(relation_count):
        count = base + (1 if relation_id < remainder else 0)
        result.append(
            generate_relation(
                num_edges=count,
                num_src=int(case["num_dst"]),
                num_dst=int(case["num_dst"]),
                active_ratio=float(case["active_ratio"]),
                skew=float(case["skew"]),
                seed=BASE_SEED + 7919 * relation_id + sum(ord(x) for x in str(case["case_id"])),
            )
        )
    return result


def superblock_owners(relation: Mapping[str, Any], world_size: int):
    import torch

    active = relation["active"]
    counts = relation["degree"].index_select(0, active)
    cumulative_before = torch.cumsum(counts, dim=0) - counts
    raw_block_ids = torch.div(cumulative_before, BLOCK_EDGES, rounding_mode="floor")
    _, block_ids = torch.unique(raw_block_ids, sorted=True, return_inverse=True)
    block_weights = torch.zeros(
        (int(block_ids.max().item()) + 1,), dtype=torch.long
    )
    block_weights.index_add_(0, block_ids, counts)
    block_owner = torch.empty_like(block_weights)
    loads = [0 for _ in range(world_size)]
    for block_id in torch.argsort(block_weights, descending=True).tolist():
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        block_owner[block_id] = rank
        loads[rank] += int(block_weights[block_id].item())
    active_owner = block_owner.index_select(0, block_ids)
    owner_by_dst = torch.zeros((int(relation["num_dst"]),), dtype=torch.long)
    owner_by_dst.index_copy_(0, active, active_owner)
    return owner_by_dst, loads, int(block_weights.numel()), int(block_weights.max().item())


def global_superblock_owners(
    relations: Sequence[Mapping[str, Any]], world_size: int
):
    """Assign blocks from all relations in one global LPT pass.

    Resetting rank loads for every small relation can place every relation's
    first block on rank zero.  The production layout uses one global workload
    view, so the benchmark mirrors that behavior.
    """
    import torch

    per_relation = []
    descriptors = []
    max_block = 0
    for relation_id, relation in enumerate(relations):
        active = relation["active"]
        counts = relation["degree"].index_select(0, active)
        cumulative_before = torch.cumsum(counts, dim=0) - counts
        raw_ids = torch.div(cumulative_before, BLOCK_EDGES, rounding_mode="floor")
        _, block_ids = torch.unique(raw_ids, sorted=True, return_inverse=True)
        weights = torch.zeros((int(block_ids.max().item()) + 1,), dtype=torch.long)
        weights.index_add_(0, block_ids, counts)
        owners = torch.empty_like(weights)
        per_relation.append((active, block_ids, owners, relation["num_dst"]))
        for block_id, weight in enumerate(weights.tolist()):
            descriptors.append((int(weight), relation_id, block_id))
            max_block = max(max_block, int(weight))

    loads = [0 for _ in range(world_size)]
    for weight, relation_id, block_id in sorted(
        descriptors, key=lambda value: (-value[0], value[1], value[2])
    ):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        per_relation[relation_id][2][block_id] = rank
        loads[rank] += weight

    owner_maps = []
    for active, block_ids, block_owners, num_dst in per_relation:
        active_owner = block_owners.index_select(0, block_ids)
        owner_by_dst = torch.zeros((int(num_dst),), dtype=torch.long)
        owner_by_dst.index_copy_(0, active, active_owner)
        owner_maps.append(owner_by_dst)
    return owner_maps, loads, len(descriptors), max_block


def direct_span(relation: Mapping[str, Any], world_size: int) -> Dict[str, float]:
    import torch

    dst = relation["dst"]
    count = int(dst.numel())
    bitmask = torch.zeros((int(relation["num_dst"]),), dtype=torch.uint8)
    for rank in range(world_size):
        start = count * rank // world_size
        end = count * (rank + 1) // world_size
        ids = torch.unique(dst[start:end])
        bitmask[ids] |= int(1 << rank)
    lookup = torch.tensor(
        [value.bit_count() for value in range(1 << world_size)], dtype=torch.uint8
    )
    spans = lookup[bitmask[bitmask > 0].long()].double()
    return {
        "active_destinations": int(spans.numel()),
        "mean_destination_span": float(spans.mean().item()),
        "p95_destination_span": float(torch.quantile(spans, 0.95).item()),
        "cross_gpu_destination_ratio": float((spans > 1).double().mean().item()),
    }


def build_bundle(
    relations: Sequence[Mapping[str, Any]],
    method: str,
    rank: int,
    world_size: int,
    device,
    relation_process_groups,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import torch
    import torch.distributed as dist

    grouped = method.startswith("superblock")
    compact = method.endswith("compact")
    started = time.perf_counter()
    bundle = []
    all_rank_loads = [0 for _ in range(world_size)]
    span_weighted = 0.0
    active_total = 0
    cross_weighted = 0.0
    blocks_total = 0
    max_block = 0
    if grouped:
        grouped_owner_maps, grouped_global_loads, grouped_blocks, grouped_max_block = (
            global_superblock_owners(relations, world_size)
        )
    else:
        grouped_owner_maps = []
        grouped_global_loads = []
        grouped_blocks = 0
        grouped_max_block = 0
    for relation_id, relation in enumerate(relations):
        edge_count = int(relation["num_edges"])
        if grouped:
            owner_by_dst = grouped_owner_maps[relation_id]
            owners = owner_by_dst.index_select(0, relation["dst"])
            positions = torch.nonzero(owners == rank, as_tuple=False).view(-1)
            span = {
                "active_destinations": int(relation["active"].numel()),
                "mean_destination_span": 1.0,
                "p95_destination_span": 1.0,
                "cross_gpu_destination_ratio": 0.0,
            }
            loads = [0 for _ in range(world_size)]  # overwritten by global LPT loads below
        else:
            start = edge_count * rank // world_size
            end = edge_count * (rank + 1) // world_size
            positions = torch.arange(start, end, dtype=torch.long)
            loads = [
                edge_count * (value + 1) // world_size
                - edge_count * value // world_size
                for value in range(world_size)
            ]
            span = direct_span(relation, world_size)
        for value in range(world_size):
            all_rank_loads[value] += int(loads[value])
        active_count = int(relation["active"].numel())
        active_total += active_count
        span_weighted += span["mean_destination_span"] * active_count
        cross_weighted += span["cross_gpu_destination_ratio"] * active_count

        local_src = relation["src"].index_select(0, positions)
        local_dst = relation["dst"].index_select(0, positions)
        if compact:
            buffer_dst = torch.searchsorted(relation["active"], local_dst)
            buffer_rows = active_count
        else:
            buffer_dst = local_dst
            buffer_rows = int(relation["num_dst"])
        bundle.append(
            {
                "src": local_src.to(device, non_blocking=False),
                "dst": buffer_dst.to(device, non_blocking=False),
                "active": relation["active"].to(device),
                "degree": relation["degree"].index_select(
                    0, relation["active"]
                ).to(device=device, dtype=torch.float32),
                "buffer_rows": buffer_rows,
                "num_dst": int(relation["num_dst"]),
                "local_edges": int(positions.numel()),
                "process_group": relation_process_groups[relation_id],
            }
        )
    if grouped:
        all_rank_loads = list(grouped_global_loads)
        blocks_total = int(grouped_blocks)
        max_block = int(grouped_max_block)
    torch.cuda.synchronize(device)
    prep_sec = time.perf_counter() - started
    prep_tensor = torch.tensor([prep_sec], device=device, dtype=torch.float64)
    dist.all_reduce(prep_tensor, op=dist.ReduceOp.MAX)
    mean_load = statistics.mean(all_rank_loads)
    metadata = {
        "prep_sec_max": float(prep_tensor.item()),
        "rank_edge_loads": all_rank_loads,
        "max_rank_edges": max(all_rank_loads),
        "edge_imbalance_max_over_mean": max(all_rank_loads) / max(mean_load, 1e-12),
        "active_destinations_total": active_total,
        "mean_destination_span": span_weighted / max(1, active_total),
        "cross_gpu_destination_ratio": cross_weighted / max(1, active_total),
        "superblocks": blocks_total,
        "max_superblock_edges": max_block,
        "compact": compact,
        "layout": "superblock" if grouped else "direct",
    }
    return bundle, metadata


def exact_allreduce(value, process_group, autograd_anchor):
    import torch
    import torch.distributed as dist

    operation = getattr(exact_allreduce, "operation", None)
    if operation is None:
        class ExactAllReduce(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor, anchor, group):
                ctx.process_group = group
                ctx.tensor_requires_grad = bool(tensor.requires_grad)
                result = tensor.contiguous().clone()
                dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
                return result

            @staticmethod
            def backward(ctx, gradient):
                result = gradient.contiguous().clone()
                dist.all_reduce(
                    result,
                    op=dist.ReduceOp.SUM,
                    group=ctx.process_group,
                )
                # ``anchor`` exists only to keep this Function in the graph
                # when a rank owns no edges for the relation.  Returning None
                # is the exact zero derivative and avoids allocating a full
                # gradient buffer for an empty local tensor.
                tensor_gradient = result if ctx.tensor_requires_grad else None
                return tensor_gradient, None, None

        operation = ExactAllReduce
        exact_allreduce.operation = operation
    return operation.apply(value, autograd_anchor, process_group)


def operator_forward(features, bundle, compact: bool, return_outputs: bool = False):
    import torch

    loss = features.new_zeros(())
    outputs = []
    autograd_anchor = features.reshape(-1)[0]
    for relation in bundle:
        local = features.new_zeros((relation["buffer_rows"], features.shape[1]))
        if relation["src"].numel() > 0:
            local.index_add_(
                0,
                relation["dst"],
                features.index_select(0, relation["src"]),
            )
        reduced = exact_allreduce(
            local,
            relation["process_group"],
            autograd_anchor,
        )
        if compact:
            active_sum = reduced
        else:
            active_sum = reduced.index_select(0, relation["active"])
        mean = active_sum / relation["degree"].view(-1, 1).clamp_min(1.0)
        loss = loss + mean.square().mean()
        if return_outputs:
            outputs.append(mean)
    return loss, outputs


def global_feature_gradient(features):
    import torch.distributed as dist

    gradient = features.grad.detach().clone()
    dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
    return gradient


def correctness_reference(features, bundle):
    import torch

    features.grad = None
    loss, outputs = operator_forward(features, bundle, compact=False, return_outputs=True)
    loss.backward()
    gradient = global_feature_gradient(features)
    stored_outputs = [value.detach().cpu() for value in outputs]
    stored_gradient = gradient.detach().cpu()
    features.grad = None
    torch.cuda.synchronize(features.device)
    return stored_outputs, stored_gradient


def correctness_against_reference(features, bundle, compact, reference_outputs, reference_gradient):
    import torch

    features.grad = None
    loss, outputs = operator_forward(features, bundle, compact=compact, return_outputs=True)
    loss.backward()
    gradient = global_feature_gradient(features).detach().cpu()
    output_abs = 0.0
    output_delta_sq = 0.0
    output_reference_sq = 0.0
    for value, reference in zip(outputs, reference_outputs):
        delta = value.detach().cpu() - reference
        output_abs = max(output_abs, float(delta.abs().max().item()))
        output_delta_sq += float(delta.double().square().sum().item())
        output_reference_sq += float(reference.double().square().sum().item())
    gradient_delta = gradient - reference_gradient
    result = {
        "forward_max_abs_error": output_abs,
        "forward_relative_l2_error": math.sqrt(output_delta_sq / max(output_reference_sq, 1e-24)),
        "gradient_max_abs_error": float(gradient_delta.abs().max().item()),
        "gradient_relative_l2_error": float(
            gradient_delta.double().norm().item()
            / max(reference_gradient.double().norm().item(), 1e-12)
        ),
    }
    features.grad = None
    torch.cuda.synchronize(features.device)
    return result


def benchmark_method(features, bundle, compact: bool, device) -> Dict[str, Any]:
    import torch
    import torch.distributed as dist

    forward_values = []
    backward_values = []
    step_values = []
    torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(WARMUPS + REPEATS):
        features.grad = None
        dist.barrier()
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        forward_started = time.perf_counter()
        loss, _ = operator_forward(features, bundle, compact=compact)
        torch.cuda.synchronize(device)
        forward_sec = time.perf_counter() - forward_started
        backward_started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_sec = time.perf_counter() - backward_started
        step_sec = time.perf_counter() - step_started
        values = torch.tensor(
            [forward_sec, backward_sec, step_sec], device=device, dtype=torch.float64
        )
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        if iteration >= WARMUPS:
            forward_values.append(float(values[0].item()))
            backward_values.append(float(values[1].item()))
            step_values.append(float(values[2].item()))
    peak = torch.tensor(
        [torch.cuda.max_memory_allocated(device) / (1024 ** 2)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    features.grad = None
    return {
        "warmups": WARMUPS,
        "repeats": REPEATS,
        "forward_sec_mean": statistics.mean(forward_values),
        "forward_sec_stdev": statistics.stdev(forward_values) if len(forward_values) > 1 else 0.0,
        "backward_sec_mean": statistics.mean(backward_values),
        "backward_sec_stdev": statistics.stdev(backward_values) if len(backward_values) > 1 else 0.0,
        "step_sec_mean": statistics.mean(step_values),
        "step_sec_median": statistics.median(step_values),
        "step_sec_stdev": statistics.stdev(step_values) if len(step_values) > 1 else 0.0,
        "step_sec_min": min(step_values),
        "step_sec_max": max(step_values),
        "peak_gpu_memory_mb": float(peak.item()),
        "step_samples_sec": step_values,
    }


def result_path(case: Mapping[str, Any], method: str) -> Path:
    return OUTPUT_ROOT / "cases" / str(case["case_id"]) / f"{method}.json"


def should_skip(path: Path, rank: int, device) -> bool:
    import torch
    import torch.distributed as dist

    value = 1 if rank == 0 and completed(path) else 0
    tensor = torch.tensor([value], device=device, dtype=torch.int32)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def run_worker() -> None:
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires four CUDA GPUs")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"Expected {WORLD_SIZE} ranks, found {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    random.seed(BASE_SEED)
    # Avoid torch.manual_seed(): on the server's CUDA-compatible runtime it
    # implicitly indexes an empty torch.cuda.default_generators tuple.  Data
    # and features are generated deterministically on CPU and then copied.
    torch.random.default_generator.manual_seed(BASE_SEED)

    group_ranks = list(range(world_size))
    # One dedicated communicator is sufficient now that the scalar autograd
    # anchor guarantees identical relation-collective order on every rank.
    # Keeping custom aggregation off the default group prevents timing/scalar
    # collectives from colliding without creating one communicator per relation.
    relation_process_group = dist.new_group(ranks=group_ranks)
    maximum_relations = max(int(case["num_relations"]) for case in cases())
    relation_process_groups = [relation_process_group] * maximum_relations

    for case_index, case in enumerate(cases()):
        if rank == 0:
            print(
                f"CASE {case_index + 1}/{len(cases())}: {case['case_id']}",
                flush=True,
            )
        relations = make_relations(case)
        feature_generator = torch.Generator().manual_seed(BASE_SEED + case_index)
        features = torch.randn(
            (int(case["num_dst"]), int(case["hidden_dim"])),
            generator=feature_generator,
            dtype=torch.float32,
        ).to(device).requires_grad_(True)

        reference_bundle, _ = build_bundle(
            relations,
            "direct_dense",
            rank,
            world_size,
            device,
            relation_process_groups,
        )
        reference_outputs, reference_gradient = correctness_reference(
            features, reference_bundle
        )
        del reference_bundle
        torch.cuda.empty_cache()

        for method in METHODS:
            path = result_path(case, method)
            if should_skip(path, rank, device):
                if rank == 0:
                    print(f"  SKIP {method}", flush=True)
                continue
            bundle, metadata = build_bundle(
                relations,
                method,
                rank,
                world_size,
                device,
                relation_process_groups,
            )
            compact = method.endswith("compact")
            correctness = correctness_against_reference(
                features,
                bundle,
                compact,
                reference_outputs,
                reference_gradient,
            )
            timing = benchmark_method(features, bundle, compact, device)
            payload_rows = sum(int(value["buffer_rows"]) for value in bundle)
            payload_bytes = payload_rows * int(case["hidden_dim"]) * BYTES_PER_VALUE
            dense_rows = len(bundle) * int(case["num_dst"])
            dense_bytes = dense_rows * int(case["hidden_dim"]) * BYTES_PER_VALUE
            ring_factor = 2.0 * (world_size - 1) / world_size
            result = {
                "completed": True,
                **case,
                "method": method,
                "world_size": world_size,
                **metadata,
                **correctness,
                **timing,
                "collective_payload_rows": payload_rows,
                "collective_payload_bytes_forward": payload_bytes,
                "dense_reference_payload_bytes_forward": dense_bytes,
                "payload_reduction_percent": 100.0 * (1.0 - payload_bytes / max(1, dense_bytes)),
                "forward_backward_payload_bytes": 2 * payload_bytes,
                "estimated_ring_wire_bytes_per_rank_step": 2 * payload_bytes * ring_factor,
                "timing_scope": "local feature gather + index_add + NCCL all-reduce + exact mean + autograd all-reduce backward",
            }
            if rank == 0:
                write_json_atomic(path, result)
                print(
                    f"  DONE {method}: median_step={result['step_sec_median']:.6f}s "
                    f"payload={payload_bytes / (1024 ** 2):.2f}MiB "
                    f"fwd_err={result['forward_relative_l2_error']:.3e}",
                    flush=True,
                )
            dist.barrier()
            del bundle
            torch.cuda.empty_cache()

        del reference_outputs, reference_gradient, features, relations
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        collect_results()
    dist.barrier()
    dist.destroy_process_group()


def collect_results() -> None:
    rows = []
    for path in sorted((OUTPUT_ROOT / "cases").rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("completed") and value.get("method"):
                value.pop("step_samples_sec", None)
                rows.append(value)
        except Exception:
            continue
    if not rows:
        raise RuntimeError("No completed phase-boundary results")
    by_case = defaultdict(dict)
    for row in rows:
        by_case[str(row["case_id"])][str(row["method"])] = row
    for methods in by_case.values():
        direct = methods.get("direct_dense")
        if direct is None:
            continue
        base = float(direct["step_sec_median"])
        for row in methods.values():
            row["speedup_vs_direct_dense"] = base / max(
                float(row["step_sec_median"]), 1e-12
            )
            row["speedup_metric"] = "median_of_step_samples"
        grouped_dense = methods.get("superblock_dense")
        grouped_compact = methods.get("superblock_compact")
        if grouped_dense and grouped_compact:
            grouped_compact["compact_speedup_with_same_layout"] = float(
                grouped_dense["step_sec_median"]
            ) / max(float(grouped_compact["step_sec_median"]), 1e-12)
        direct_compact = methods.get("direct_compact")
        if direct_compact:
            direct_compact["compact_speedup_with_same_layout"] = base / max(
                float(direct_compact["step_sec_median"]), 1e-12
            )
    rows = [row for methods in by_case.values() for row in methods.values()]
    rows.sort(
        key=lambda row: (
            str(row["axis"]), float(row["axis_value"]), METHODS.index(row["method"])
        )
    )
    write_csv(OUTPUT_ROOT / "phase_boundary_4gpu_summary.csv", rows)
    create_plots(rows)
    create_report(rows)
    write_json_atomic(
        OUTPUT_ROOT / "complete.json",
        {
            "completed": True,
            "cases": len(by_case),
            "result_rows": len(rows),
            "methods": METHODS,
            "world_size": WORLD_SIZE,
        },
    )


def create_plots(rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"PLOTS SKIPPED: {type(exc).__name__}: {exc}", flush=True)
        return
    active = [row for row in rows if row["axis"] == "active_ratio"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for method in METHODS:
        values = sorted(
            [row for row in active if row["method"] == method],
            key=lambda row: float(row["axis_value"]),
        )
        axes[0].plot(
            [float(row["axis_value"]) for row in values],
            [float(row["speedup_vs_direct_dense"]) for row in values],
            marker="o",
            label=method.replace("_", " "),
        )
        axes[1].plot(
            [float(row["axis_value"]) for row in values],
            [float(row["peak_gpu_memory_mb"]) for row in values],
            marker="o",
            label=method.replace("_", " "),
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Active-destination ratio")
    axes[0].set_ylabel("Measured speedup vs direct+dense")
    axes[1].set_xlabel("Active-destination ratio")
    axes[1].set_ylabel("Peak allocated GPU memory (MB)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "active_ratio_phase_boundary_4gpu.png", dpi=220)
    plt.close(fig)

    hidden = [row for row in rows if row["axis"] == "hidden_dim"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method in METHODS:
        values = sorted(
            [row for row in hidden if row["method"] == method],
            key=lambda row: float(row["axis_value"]),
        )
        ax.plot(
            [float(row["axis_value"]) for row in values],
            [float(row["speedup_vs_direct_dense"]) for row in values],
            marker="o",
            label=method.replace("_", " "),
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Hidden dimension")
    ax.set_ylabel("Measured speedup vs direct+dense")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "hidden_dimension_phase_boundary_4gpu.png", dpi=220)
    plt.close(fig)


def create_report(rows: Sequence[Mapping[str, Any]]) -> None:
    active = sorted(
        [row for row in rows if row["axis"] == "active_ratio" and row["method"] == "superblock_compact"],
        key=lambda row: float(row["axis_value"]),
    )
    max_forward_error = max(float(row["forward_relative_l2_error"]) for row in rows)
    max_gradient_error = max(float(row["gradient_relative_l2_error"]) for row in rows)
    lines = [
        "# Four-GPU active-destination phase boundary",
        "",
        "These are measured NCCL forward+backward operator results, not alpha+beta projections.",
        "",
        f"- Cases: {len({str(row['case_id']) for row in rows})}",
        f"- Result rows: {len(rows)}",
        f"- Maximum forward relative L2 error: {max_forward_error:.3e}",
        f"- Maximum globally accumulated gradient relative L2 error: {max_gradient_error:.3e}",
        "",
        "## SuperBlock + compact by active-destination ratio",
        "",
        "| Active ratio | Median step time | Speedup vs Direct+dense | Payload reduction | Peak memory |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in active:
        lines.append(
            f"| {float(row['axis_value']):.2f} | {float(row['step_sec_median']):.6f}s "
            f"| {float(row['speedup_vs_direct_dense']):.3f}x "
            f"| {float(row['payload_reduction_percent']):.2f}% "
            f"| {float(row['peak_gpu_memory_mb']):.1f} MB |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this experiment to identify the measured compact/dense phase boundary. End-to-end DBLP and industrial runs remain the evidence for whole-training speedup.",
            "",
            "## Files",
            "",
            "- `phase_boundary_4gpu_summary.csv`",
            "- `active_ratio_phase_boundary_4gpu.png`",
            "- `hidden_dimension_phase_boundary_4gpu.png`",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch_parent() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in GPU_IDS),
            "SIGMOD_ACTIVE_DEST_WORKER": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
    ]
    log_path = OUTPUT_ROOT / "launcher.log"
    print("RUN:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        returncode = process.wait()
    if returncode != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-160:]
        )
        print(tail, flush=True)
        raise RuntimeError(f"four-GPU phase-boundary experiment failed: {log_path}")
    print(f"COMPLETE: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    if os.environ.get("SIGMOD_ACTIVE_DEST_WORKER") == "1":
        run_worker()
    else:
        launch_parent()

