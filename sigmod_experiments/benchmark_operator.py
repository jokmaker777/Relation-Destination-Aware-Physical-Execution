from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist

from sigmod_exp.datasets import (
    DatasetSpec,
    edge_type_to_str,
    load_graph,
    select_relations,
    summarize_graph,
)
from sigmod_exp.partition import PhysicalPlan, build_local_partition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact distributed relation aggregation microbenchmark"
    )
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", default="superblock_greedy")
    parser.add_argument("--block-edges", type=int, default=25_000)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--collective-mode", choices=["blocking", "async"], default="async")
    parser.add_argument("--aggregation", choices=["sum", "mean"], default="mean")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--validate-correctness", action="store_true")
    parser.add_argument(
        "--correctness-max-edges",
        type=int,
        default=2_000_000,
        help="Skip full-reference validation above this edge count",
    )
    return parser.parse_args()


def init_runtime() -> Tuple[int, int, int, torch.device, bool]:
    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if launched:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    initialized_here = False
    if launched and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True
    return rank, world_size, local_rank, device, initialized_here


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()


def reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _range_bounds(index: int, ranges: int, num_dst: int) -> Tuple[int, int]:
    start = (index * num_dst + ranges - 1) // ranges
    end = ((index + 1) * num_dst + ranges - 1) // ranges
    return start, min(end, num_dst)


@torch.inference_mode()
def aggregate_once(
    local_edges: torch.Tensor,
    offsets: List[int],
    num_dst: int,
    features: torch.Tensor,
    plan: PhysicalPlan,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    ranges = len(offsets) - 1
    outputs: List[torch.Tensor] = []
    degrees: List[torch.Tensor] = []
    pending: Tuple[Any, Any] | None = None
    exposed_wait_sec = 0.0

    for chunk in range(ranges):
        range_start, range_end = _range_bounds(chunk, ranges, num_dst)
        width = max(0, range_end - range_start)
        left, right = offsets[chunk], offsets[chunk + 1]
        chunk_edges = local_edges[:, left:right]
        output = torch.zeros((width, features.shape[1]), dtype=torch.float32, device=device)
        degree = torch.zeros((width,), dtype=torch.float32, device=device)
        if chunk_edges.numel() > 0:
            src = chunk_edges[0]
            dst = chunk_edges[1] - range_start
            output.index_add_(0, dst, features.index_select(0, src))
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))

        if pending is not None:
            # Previous communication overlaps the current chunk's local compute.
            wait_started = time.perf_counter()
            pending[0].wait()
            pending[1].wait()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            exposed_wait_sec += time.perf_counter() - wait_started
            pending = None

        if dist.is_initialized() and dist.get_world_size() > 1:
            if plan.collective_mode == "blocking":
                wait_started = time.perf_counter()
                dist.all_reduce(output, op=dist.ReduceOp.SUM)
                dist.all_reduce(degree, op=dist.ReduceOp.SUM)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                exposed_wait_sec += time.perf_counter() - wait_started
            else:
                pending = (
                    dist.all_reduce(output, op=dist.ReduceOp.SUM, async_op=True),
                    dist.all_reduce(degree, op=dist.ReduceOp.SUM, async_op=True),
                )
        outputs.append(output)
        degrees.append(degree)

    if pending:
        wait_started = time.perf_counter()
        pending[0].wait()
        pending[1].wait()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        exposed_wait_sec += time.perf_counter() - wait_started

    if plan.aggregation == "mean":
        for output, degree in zip(outputs, degrees):
            output.div_(degree.clamp_min_(1.0).unsqueeze(1))
    return torch.cat(outputs, dim=0), exposed_wait_sec


@torch.inference_mode()
def validate_output(
    edge_index: torch.Tensor,
    num_dst: int,
    features: torch.Tensor,
    actual: torch.Tensor,
    aggregation: str,
    device: torch.device,
) -> Dict[str, Any]:
    full_edges = edge_index.to(device=device, non_blocking=False)
    reference = torch.zeros(
        (num_dst, features.shape[1]), dtype=torch.float32, device=device
    )
    degree = torch.zeros((num_dst,), dtype=torch.float32, device=device)
    if full_edges.numel() > 0:
        reference.index_add_(0, full_edges[1], features.index_select(0, full_edges[0]))
        degree.index_add_(
            0, full_edges[1], torch.ones_like(full_edges[1], dtype=torch.float32)
        )
    if aggregation == "mean":
        reference.div_(degree.clamp_min_(1.0).unsqueeze(1))
    difference = (actual - reference).abs()
    denominator = reference.abs().clamp_min(1e-7)
    return {
        "validated": True,
        "max_abs_error": float(difference.max().item()) if difference.numel() else 0.0,
        "max_rel_error": float((difference / denominator).max().item()) if difference.numel() else 0.0,
        "allclose_rtol_1e-4_atol_1e-5": bool(
            torch.allclose(actual, reference, rtol=1e-4, atol=1e-5)
        ),
    }


def hardware_record(device: torch.device, local_rank: int) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(local_rank)
        nccl_version = torch.cuda.nccl.version()
        if isinstance(nccl_version, tuple):
            nccl_text = ".".join(str(x) for x in nccl_version)
        else:
            nccl_text = str(nccl_version)
        value.update(
            {
                "gpu_name": properties.name,
                "gpu_memory_bytes": int(properties.total_memory),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "nccl": nccl_text,
            }
        )
    return value


def run_benchmark(args: argparse.Namespace) -> None:
    rank, world_size, local_rank, device, initialized_here = init_runtime()
    if args.hidden_dim <= 0 or args.warmups < 0 or args.repeats <= 0:
        raise ValueError("hidden-dim/repeats must be positive and warmups non-negative")

    spec = DatasetSpec.from_dict(json.loads(args.dataset_json))
    data = load_graph(spec)
    graph_summary = summarize_graph(data)
    relations = select_relations(data, spec.relation_selector)
    plan = PhysicalPlan(
        policy=args.policy,
        block_edges=args.block_edges,
        destination_ranges=args.chunks,
        collective_mode=args.collective_mode,
        aggregation=args.aggregation,
    )
    plan.validate()

    records: List[Dict[str, Any]] = []
    for relation_id, edge_type in enumerate(relations):
        store = data[edge_type]
        edge_index = store.edge_index.long().cpu().contiguous()
        num_src = int(graph_summary["node_counts"][edge_type[0]])
        num_dst = int(graph_summary["node_counts"][edge_type[2]])
        local_edges, partition = build_local_partition(
            edge_index=edge_index,
            num_dst=num_dst,
            rank=rank,
            world_size=world_size,
            plan=plan,
        )
        prep_sec_max = reduce_max(float(partition["prep_sec"]), device)
        local_edges = local_edges.to(device=device, non_blocking=False)
        offsets = [int(value) for value in partition["range_offsets"]]
        generator = torch.Generator(device=device).manual_seed(args.seed + relation_id)
        features = torch.randn(
            (num_src, args.hidden_dim),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        last_output = None
        for _ in range(args.warmups):
            last_output, _ = aggregate_once(
                local_edges, offsets, num_dst, features, plan, device
            )
        synchronize(device)

        elapsed_values: List[float] = []
        wait_values: List[float] = []
        for _ in range(args.repeats):
            synchronize(device)
            started = time.perf_counter()
            last_output, wait_sec = aggregate_once(
                local_edges, offsets, num_dst, features, plan, device
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            local_elapsed = time.perf_counter() - started
            elapsed_values.append(reduce_max(local_elapsed, device))
            wait_values.append(reduce_max(wait_sec, device))

        correctness: Dict[str, Any] = {"validated": False}
        should_validate = args.validate_correctness and edge_index.shape[1] <= args.correctness_max_edges
        if should_validate and rank == 0 and last_output is not None:
            correctness = validate_output(
                edge_index,
                num_dst,
                features,
                last_output,
                args.aggregation,
                device,
            )
        elif args.validate_correctness:
            correctness = {
                "validated": False,
                "reason": f"num_edges exceeds correctness-max-edges={args.correctness_max_edges}",
            }

        if rank == 0:
            median_sec = float(statistics.median(elapsed_values))
            num_edges = int(edge_index.shape[1])
            payload_elements = num_dst * (args.hidden_dim + 1)
            actual_ranges = int(partition["destination_ranges"])
            collective_calls = actual_ranges * 2 if world_size > 1 else 0
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            )
            records.append(
                {
                    "schema_version": 1,
                    "dataset": spec.name,
                    "dataset_kind": spec.kind,
                    "relation": edge_type_to_str(edge_type),
                    "num_src": num_src,
                    "num_dst": num_dst,
                    "num_edges": num_edges,
                    "world_size": world_size,
                    "plan": plan.to_dict(),
                    "hidden_dim": args.hidden_dim,
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                    "prep_sec_max": prep_sec_max,
                    "operator_sec_values": elapsed_values,
                    "operator_sec_median": median_sec,
                    "operator_sec_mean": float(statistics.mean(elapsed_values)),
                    "operator_sec_stdev": (
                        float(statistics.stdev(elapsed_values))
                        if len(elapsed_values) > 1
                        else 0.0
                    ),
                    "exposed_wait_sec_values": wait_values,
                    "throughput_edges_per_sec": num_edges / max(median_sec, 1e-12),
                    "collective_calls_per_repeat": collective_calls,
                    "collective_payload_bytes_per_repeat": (
                        payload_elements * 4 if world_size > 1 else 0
                    ),
                    "peak_memory_bytes_rank0": peak_memory,
                    "partition": partition,
                    "correctness": correctness,
                    "hardware": hardware_record(device, local_rank),
                }
            )

        del features, local_edges, last_output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "created_at_unix": time.time(),
            "dataset_spec": json.loads(args.dataset_json),
            "graph_summary": graph_summary,
            "records": records,
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
        print(f"wrote {output}")

    if initialized_here:
        dist.destroy_process_group()


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()

