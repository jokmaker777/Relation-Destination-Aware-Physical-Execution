#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-argument 2/4-GPU safety gate for the exact aggregation collectives.

Run this file with ordinary Python.  The parent process launches torchrun for
2 GPUs and then 4 GPUs.  The worker checks two failure modes that previously
caused the C=8 hang:

1. one rank does not use the reduced tensor in its real loss;
2. chunk-plan cache keys must not depend on rank-local tensor pointers.

No industrial graph is loaded, so the gate normally finishes in a few minutes.
"""

from __future__ import annotations

import json
import inspect
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(
    "/path/to/artifact_workspace/sigmod2027_20241129/collective_safety_gate"
)
GPU_COUNTS = (2, 4)
WORKER_FLAG = "COMMOPT_COLLECTIVE_SAFETY_WORKER"

sys.path.insert(0, str(PROJECT_ROOT))

from end_to_end_prunning.models.communication_optimized_training_manager_forward_agg_overlap_no_parser import (  # noqa: E402
    COMMOPT_PATCH_VERSION,
    CommunicationOptimizedTrainingManager,
    LABELED_EDGE_TYPES,
    _DetachedChunkedForwardAllReduceWithBackward,
)


def mapped_device(local_rank: int) -> int:
    mapping = [
        int(value.strip())
        for value in os.environ.get("COMMOPT_CUDA_DEVICE_MAP", "").split(",")
        if value.strip()
    ]
    return mapping[local_rank] if mapping else local_rank


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = float((actual - expected).abs().max().item())
    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"{name} failed: max_abs_error={error}")
    return error


def test_unused_rank_backward(group, device: torch.device, rank: int, world: int) -> float:
    manager = CommunicationOptimizedTrainingManager.__new__(
        CommunicationOptimizedTrainingManager
    )
    manager.world_size = world
    manager._forward_agg_backward_anchors = []
    manager._reset_forward_agg_backward_anchors()

    local_chunks = []
    reduced_chunks = []
    for chunk_id in range(8):
        local = (
            torch.linspace(0.0, 1.0, 32_768, device=device)
            + float(rank + 1)
            + float(chunk_id) / 10.0
        ).requires_grad_(True)
        reduced = local.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
        local_chunks.append(local)
        reduced_chunks.append(reduced)

    global_chunks = _DetachedChunkedForwardAllReduceWithBackward.apply(
        group, *local_chunks, *reduced_chunks
    )
    if isinstance(global_chunks, torch.Tensor):
        global_chunks = (global_chunks,)
    for value in global_chunks:
        manager._register_forward_agg_backward_anchor(value)

    # Each rank consumes a different subset of real chunk outputs.  The single
    # chunked autograd node must still issue all eight gradient collectives in
    # the same order, substituting zero gradients where needed.
    owned_losses = [
        value.square().mean()
        for chunk_id, value in enumerate(global_chunks)
        if chunk_id % world == rank
    ]
    base_loss = torch.stack(owned_losses).sum()
    loss = manager._attach_forward_agg_backward_anchors(base_loss)
    loss.backward()

    max_error = 0.0
    for chunk_id, (local, reduced) in enumerate(
        zip(local_chunks, reduced_chunks)
    ):
        expected = 2.0 * reduced / float(reduced.numel())
        max_error = max(
            max_error,
            assert_close(
                f"unused-rank backward chunk={chunk_id}", local.grad, expected
            ),
        )
    return max_error


def test_pointer_independent_plan_cache(
    group, device: torch.device, rank: int, world: int
) -> None:
    manager = CommunicationOptimizedTrainingManager.__new__(
        CommunicationOptimizedTrainingManager
    )
    manager.config = SimpleNamespace(
        training_config=SimpleNamespace(
            aggregation_chunk_policy="global_edge_balanced",
            aggregation_chunk_order="natural",
        )
    )
    manager.world_size = world
    manager._forward_agg_group = group
    manager._forward_agg_groups = {("src", "rel", "dst"): group}
    manager._forward_agg_graph_generation = 1
    manager._forward_agg_chunk_plan_cache = {}

    num_dst = 256
    edge_count = 4096 + rank * 257
    generator = torch.Generator(device=device).manual_seed(20261004 + rank)
    src = torch.randint(1024, (edge_count,), generator=generator, device=device)
    dst = torch.randint(num_dst, (edge_count,), generator=generator, device=device)
    edge_index = torch.stack([src, dst])

    first = manager._get_destination_chunk_plan(
        ("src", "rel", "dst"), edge_index, num_dst, 4
    )
    # A clone has a new object id and data pointer.  It must reuse the same
    # deterministic logical-plan cache entry on every rank.
    second = manager._get_destination_chunk_plan(
        ("src", "rel", "dst"), edge_index.clone(), num_dst, 4
    )
    if first["ranges"] != second["ranges"]:
        raise RuntimeError("pointer-independent plan cache returned different ranges")
    if len(manager._forward_agg_chunk_plan_cache) != 1:
        raise RuntimeError(
            "plan cache still depends on tensor identity: "
            f"entries={len(manager._forward_agg_chunk_plan_cache)}"
        )

    boundaries = [first["ranges"][0][0]] + [end for _, end in first["ranges"]]
    values = torch.tensor(boundaries, dtype=torch.int64, device=device)
    minima, maxima = values.clone(), values.clone()
    dist.all_reduce(minima, op=dist.ReduceOp.MIN)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    if not torch.equal(minima, maxima):
        raise RuntimeError("chunk boundaries differ across ranks")


def test_relation_group_routing() -> None:
    """Catch stale builds that create relation groups but never select them."""
    if len(LABELED_EDGE_TYPES) < 2:
        raise RuntimeError("relation-group safety check needs at least two relations")
    manager = CommunicationOptimizedTrainingManager.__new__(
        CommunicationOptimizedTrainingManager
    )
    first_group, second_group = object(), object()
    manager._forward_agg_group = first_group
    manager._forward_agg_groups = {
        LABELED_EDGE_TYPES[0]: first_group,
        LABELED_EDGE_TYPES[1]: second_group,
    }
    manager._ensure_forward_aggregation_group = lambda: first_group
    if manager._get_forward_aggregation_group(LABELED_EDGE_TYPES[1]) is not second_group:
        raise RuntimeError("edge type was routed to the default process group")

    source = inspect.getsource(
        CommunicationOptimizedTrainingManager._distributed_sage_relation_forward
    )
    expected = "self._get_forward_aggregation_group(edge_type)"
    if expected not in source:
        raise RuntimeError(
            "distributed relation forward still uses the default process group"
        )


def worker() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device_id = mapped_device(local_rank)
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    group = dist.new_group(
        ranks=list(range(world)), backend="nccl", timeout=timedelta(minutes=3)
    )

    started = time.perf_counter()
    test_relation_group_routing()
    max_error = test_unused_rank_backward(group, device, rank, world)
    test_pointer_independent_plan_cache(group, device, rank, world)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    dist.barrier()

    if rank == 0:
        output = OUTPUT_ROOT / f"gpus-{world}" / "result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "world_size": world,
                    "gpu_name": torch.cuda.get_device_name(device_id),
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "commopt_patch_version": COMMOPT_PATCH_VERSION,
                    "relation_group_routing": True,
                    "unused_rank_backward_max_abs_error": max_error,
                    "pointer_independent_plan_cache": True,
                    "elapsed_sec": elapsed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"COLLECTIVE SAFETY PASSED: {output}", flush=True)

    dist.destroy_process_group(group)
    dist.destroy_process_group()


def parent() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This safety gate must be run on the GPU server")
    available = int(torch.cuda.device_count())
    if available < max(GPU_COUNTS):
        raise RuntimeError(
            f"Expected at least {max(GPU_COUNTS)} GPUs, found {available}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for world in GPU_COUNTS:
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env[WORKER_FLAG] = "1"
        env["COMMOPT_CUDA_DEVICE_MAP"] = ",".join(str(i) for i in range(world))
        log = OUTPUT_ROOT / f"gpus-{world}.log"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            str(Path(__file__).resolve()),
        ]
        print(f"RUN COLLECTIVE SAFETY: GPUs=0..{world - 1}, log={log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            tail = "\n".join(
                log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
            )
            raise RuntimeError(
                f"{world}-GPU collective safety gate failed.\n{tail}"
            )
    print("ALL 2/4-GPU COLLECTIVE SAFETY GATES PASSED", flush=True)


if __name__ == "__main__":
    if os.environ.get(WORKER_FLAG) == "1":
        worker()
    else:
        parent()

