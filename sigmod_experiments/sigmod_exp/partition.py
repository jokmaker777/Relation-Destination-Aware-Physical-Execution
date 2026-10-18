from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PhysicalPlan:
    policy: str
    block_edges: int
    destination_ranges: int
    collective_mode: str
    aggregation: str = "mean"

    def validate(self) -> None:
        if self.policy not in {
            "contiguous",
            "round_robin",
            "destination_hash",
            "destination_greedy",
            "superblock_greedy",
        }:
            raise ValueError(f"Unsupported partition policy: {self.policy}")
        if self.block_edges <= 0:
            raise ValueError("block_edges must be positive")
        if self.destination_ranges <= 0:
            raise ValueError("destination_ranges must be positive")
        if self.collective_mode not in {"blocking", "async"}:
            raise ValueError("collective_mode must be blocking or async")
        if self.aggregation not in {"sum", "mean"}:
            raise ValueError("aggregation must be sum or mean")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _rank_loads(assignments: torch.Tensor, world_size: int) -> List[int]:
    if assignments.numel() == 0:
        return [0 for _ in range(world_size)]
    counts = torch.bincount(assignments.long(), minlength=world_size)
    return [int(value) for value in counts.tolist()]


def _target_span_stats(dst: torch.Tensor, assignments: torch.Tensor, num_dst: int, world_size: int):
    if dst.numel() == 0:
        return {"target_groups": 0, "cross_gpu_target_groups": 0, "mean_target_span": 0.0}
    bitmask = torch.zeros((num_dst,), dtype=torch.int16)
    for rank in range(world_size):
        rank_dst = torch.unique(dst[assignments == rank])
        if rank_dst.numel() > 0:
            bitmask[rank_dst] = bitmask[rank_dst] | int(1 << rank)
    active = bitmask[bitmask > 0]
    lookup = torch.tensor([int(i).bit_count() for i in range(1 << world_size)], dtype=torch.int16)
    spans = lookup[active.long()]
    return {
        "target_groups": int(active.numel()),
        "cross_gpu_target_groups": int((spans > 1).sum().item()),
        "mean_target_span": float(spans.float().mean().item()) if spans.numel() else 0.0,
    }


def _destination_greedy_assignment(dst: torch.Tensor, world_size: int) -> Tuple[torch.Tensor, int, int]:
    order = torch.argsort(dst, stable=True)
    sorted_dst = dst.index_select(0, order)
    _, counts = torch.unique_consecutive(sorted_dst, return_counts=True)
    loads = [0 for _ in range(world_size)]
    assignment = torch.empty((dst.numel(),), dtype=torch.long)
    offset = 0
    for count in counts.tolist():
        rank = min(range(world_size), key=lambda p: (loads[p], p))
        positions = order[offset : offset + count]
        assignment[positions] = rank
        loads[rank] += int(count)
        offset += int(count)
    return assignment, int(counts.numel()), int(counts.max().item()) if counts.numel() else 0


def _superblock_assignment(
    dst: torch.Tensor,
    world_size: int,
    block_edges: int,
) -> Tuple[torch.Tensor, int, int, int]:
    order = torch.argsort(dst, stable=True)
    sorted_dst = dst.index_select(0, order)
    _, counts = torch.unique_consecutive(sorted_dst, return_counts=True)
    group_counts = [int(value) for value in counts.tolist()]

    blocks: List[Tuple[int, int, int]] = []
    block_start = 0
    block_total = 0
    group_offset = 0
    max_block = 0
    for count in group_counts:
        if block_total > 0 and block_total + count > block_edges:
            blocks.append((block_start, group_offset, block_total))
            max_block = max(max_block, block_total)
            block_start = group_offset
            block_total = 0
        block_total += count
        group_offset += count
        if count > block_edges:
            blocks.append((block_start, group_offset, block_total))
            max_block = max(max_block, block_total)
            block_start = group_offset
            block_total = 0
    if block_total > 0:
        blocks.append((block_start, group_offset, block_total))
        max_block = max(max_block, block_total)

    loads = [0 for _ in range(world_size)]
    assignment = torch.empty((dst.numel(),), dtype=torch.long)
    for start, end, count in sorted(blocks, key=lambda value: (-value[2], value[0])):
        rank = min(range(world_size), key=lambda p: (loads[p], p))
        assignment[order[start:end]] = rank
        loads[rank] += count
    return assignment, int(counts.numel()), len(blocks), int(max_block)


def build_local_partition(
    edge_index: torch.Tensor,
    num_dst: int,
    rank: int,
    world_size: int,
    plan: PhysicalPlan,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build one rank's relation partition and reusable destination-range order."""

    plan.validate()
    if edge_index.device.type != "cpu":
        edge_index = edge_index.cpu()
    edge_index = edge_index.long().contiguous()
    num_edges = int(edge_index.shape[1])
    dst = edge_index[1]
    started = time.perf_counter()

    if world_size <= 1:
        assignments = torch.zeros((num_edges,), dtype=torch.long)
        num_groups = int(torch.unique(dst).numel()) if num_edges else 0
        num_blocks = 1 if num_edges else 0
        max_block = num_edges
    elif plan.policy == "contiguous":
        assignments = torch.clamp(
            (torch.arange(num_edges, dtype=torch.long) * world_size) // max(1, num_edges),
            max=world_size - 1,
        )
        num_groups = int(torch.unique(dst).numel()) if num_edges else 0
        num_blocks = world_size
        max_block = int(math.ceil(num_edges / world_size))
    elif plan.policy == "round_robin":
        assignments = torch.arange(num_edges, dtype=torch.long) % world_size
        num_groups = int(torch.unique(dst).numel()) if num_edges else 0
        num_blocks = world_size
        max_block = int(math.ceil(num_edges / world_size))
    elif plan.policy == "destination_hash":
        assignments = dst % world_size
        num_groups = int(torch.unique(dst).numel()) if num_edges else 0
        num_blocks = num_groups
        max_block = int(torch.bincount(dst, minlength=num_dst).max().item()) if num_edges else 0
    elif plan.policy == "destination_greedy":
        assignments, num_groups, max_block = _destination_greedy_assignment(dst, world_size)
        num_blocks = num_groups
    else:
        assignments, num_groups, num_blocks, max_block = _superblock_assignment(
            dst, world_size, plan.block_edges
        )

    local_positions = torch.nonzero(assignments == rank, as_tuple=False).view(-1)
    local_edges = edge_index.index_select(1, local_positions)

    ranges = min(max(1, plan.destination_ranges), max(1, num_dst))
    if local_edges.shape[1] > 0:
        chunk_ids = torch.clamp(
            (local_edges[1] * ranges) // max(1, num_dst), max=ranges - 1
        )
        order = torch.argsort(chunk_ids, stable=True)
        local_edges = local_edges.index_select(1, order)
        chunk_counts = torch.bincount(chunk_ids, minlength=ranges).tolist()
    else:
        chunk_counts = [0 for _ in range(ranges)]
    offsets = [0]
    for count in chunk_counts:
        offsets.append(offsets[-1] + int(count))

    rank_edges = _rank_loads(assignments, world_size)
    mean_edges = sum(rank_edges) / max(1, world_size)
    span = _target_span_stats(dst, assignments, num_dst, world_size)
    prep_sec = float(time.perf_counter() - started)
    metadata = {
        "prep_sec": prep_sec,
        "num_edges": num_edges,
        "num_target_groups": num_groups,
        "num_blocks": num_blocks,
        "max_block_edges": max_block,
        "rank_edges": rank_edges,
        "max_rank_edges": max(rank_edges) if rank_edges else 0,
        "edge_imbalance_to_mean": (
            max(rank_edges) / mean_edges if mean_edges > 0 else 0.0
        ),
        "range_offsets": offsets,
        "destination_ranges": ranges,
        **span,
    }
    return local_edges.contiguous(), metadata

