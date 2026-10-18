from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import torch

from .edge_block import EdgeBlock
from .partition_metrics import compute_partition_metrics


@dataclass
class PartitionState:
    num_gpus: int
    assignment: Dict[int, int]
    compute_load: List[float]
    memory_load: List[float]
    type_load: List[Dict[Tuple[str, str, str], float]]
    target_to_gpus: Dict[Tuple[str, int], Set[int]]
    memory_budget: float


def _coerce_assignment(assignment: Union[Dict[int, int], torch.Tensor], num_blocks: int, num_gpus: int) -> Dict[int, int]:
    if isinstance(assignment, torch.Tensor):
        if assignment.ndim != 1 or assignment.shape[0] != num_blocks:
            raise ValueError("assignment tensor 长度与 blocks 不一致")
        return {i: int(assignment[i].clamp(0, num_gpus - 1).item()) for i in range(num_blocks)}
    return {i: int(assignment.get(i, 0)) for i in range(num_blocks)}


def _make_state(blocks: List[EdgeBlock], assignment: Dict[int, int], num_gpus: int, memory_budget: float) -> PartitionState:
    type_load = [{} for _ in range(num_gpus)]
    target_to_gpus: Dict[Tuple[str, int], Set[int]] = {}
    compute_load = [0.0 for _ in range(num_gpus)]
    memory_load = [0.0 for _ in range(num_gpus)]

    for block in blocks:
        rank = int(assignment.get(block.block_id, 0))
        rank = max(0, min(num_gpus - 1, rank))
        cost = float(block.compute_cost) if isinstance(block.compute_cost, (int, float)) else float(block.num_edges)
        mem = float(block.memory_cost) if isinstance(block.memory_cost, (int, float)) else 0.0
        compute_load[rank] += cost
        memory_load[rank] += mem
        type_load[rank][block.edge_type] = type_load[rank].get(block.edge_type, 0.0) + cost
        key = (block.dst_node_type, int(block.dst_node_id))
        target_to_gpus.setdefault(key, set()).add(rank)

    return PartitionState(
        num_gpus=num_gpus,
        assignment=dict(assignment),
        compute_load=compute_load,
        memory_load=memory_load,
        type_load=type_load,
        target_to_gpus=target_to_gpus,
        memory_budget=float(memory_budget),
    )


def _evaluate_state(blocks: List[EdgeBlock], state: PartitionState) -> Dict[str, float]:
    num_gpus = state.num_gpus
    max_compute = max(state.compute_load) if state.compute_load else 0.0
    avg_compute = sum(state.compute_load) / num_gpus if num_gpus > 0 else 0.0
    load_imbalance = max_compute / avg_compute if avg_compute > 0 else 0.0

    conflict_nodes = 0
    aggregation_contention = 0.0
    for gpu_set in state.target_to_gpus.values():
        rho = float(len(gpu_set))
        if rho > 1:
            conflict_nodes += 1
            aggregation_contention += rho - 1

    memory_over_budget = sum(max(0.0, mem - state.memory_budget) for mem in state.memory_load)

    return {
        "max_compute_load": max_compute,
        "load_imbalance": load_imbalance,
        "aggregation_contention": aggregation_contention,
        "memory_over_budget": memory_over_budget,
        "conflict_nodes": float(conflict_nodes),
    }


def _incremental_score(
    block: EdgeBlock,
    p: int,
    state: PartitionState,
    lambda_c: float = 1.0,
    lambda_m: float = 0.5,
    lambda_r: float = 0.2,
    lambda_a: float = 0.5,
) -> float:
    cost = float(block.compute_cost) if isinstance(block.compute_cost, (int, float)) else float(block.num_edges)
    mem = float(block.memory_cost) if isinstance(block.memory_cost, (int, float)) else 0.0
    weight = float(block.aggregation_weight) if isinstance(block.aggregation_weight, (int, float)) else cost

    new_compute = state.compute_load[p] + cost
    max_after = max(max(state.compute_load), new_compute)
    avg_after = (sum(state.compute_load) + cost) / state.num_gpus
    load_score = max_after - avg_after if avg_after > 0 else 0.0

    key = (block.dst_node_type, int(block.dst_node_id))
    old_gpus = set(state.target_to_gpus.get(key, set()))
    agg_score = 0.0
    if p not in old_gpus and old_gpus:
        agg_score = weight

    new_mem = state.memory_load[p] + mem
    mem_score = max(0.0, new_mem - state.memory_budget)

    type_score = 0.0
    edge_type = block.edge_type
    current_type = state.type_load[p].get(edge_type, 0.0)
    max_type = max((row.get(edge_type, 0.0) for row in state.type_load), default=0.0)
    if current_type + cost > max_type:
        type_score = current_type + cost - max_type

    return lambda_c * load_score + lambda_a * agg_score + lambda_m * mem_score + lambda_r * type_score


def assign_blocks_by_cost_aware_greedy(
    blocks: List[EdgeBlock],
    num_gpus: int,
    *,
    memory_budget: float = 0.0,
    lambda_c: float = 1.0,
    lambda_m: float = 0.5,
    lambda_r: float = 0.2,
    lambda_a: float = 0.5,
) -> Dict[int, int]:
    blocks = sorted(blocks, key=lambda b: float(b.compute_cost) if isinstance(b.compute_cost, (int, float)) else float(b.num_edges), reverse=True)
    assignment: Dict[int, int] = {}
    state = _make_state(blocks, assignment, num_gpus, memory_budget)

    for block in blocks:
        best_p = min(range(num_gpus), key=lambda p: _incremental_score(block, p, state, lambda_c, lambda_m, lambda_r, lambda_a))
        assignment[block.block_id] = best_p
        state = _make_state(blocks, assignment, num_gpus, memory_budget)

    return assignment


def refine_partition_assignment(
    blocks: List[EdgeBlock],
    num_gpus: int,
    base_assignment: Union[Dict[int, int], torch.Tensor],
    *,
    memory_budget: float = 0.0,
    max_iters: int = 1,
    lambda_c: float = 1.0,
    lambda_m: float = 0.5,
    lambda_r: float = 0.2,
    lambda_a: float = 0.5,
    tol: float = 0.0,
) -> Dict[int, int]:
    assignment = _coerce_assignment(base_assignment, len(blocks), num_gpus)
    state = _make_state(blocks, assignment, num_gpus, memory_budget)
    current_score = _evaluate_state(blocks, state)

    for _ in range(max(1, max_iters)):
        improved = False
        for block in blocks:
            cur_p = int(assignment.get(block.block_id, 0))
            best_q = cur_p
            best_delta = float("inf")

            for q in range(num_gpus):
                if q == cur_p:
                    continue

                candidate = dict(assignment)
                candidate[block.block_id] = q
                cand_state = _make_state(blocks, candidate, num_gpus, memory_budget)
                cand_score = _evaluate_state(blocks, cand_state)

                delta = (
                    cand_score["max_compute_load"] - current_score["max_compute_load"]
                    + lambda_a * (cand_score["aggregation_contention"] - current_score["aggregation_contention"])
                    + lambda_m * (cand_score["memory_over_budget"] - current_score["memory_over_budget"])
                )

                if delta < best_delta - tol:
                    best_delta = delta
                    best_q = q

            if best_q != cur_p:
                assignment[block.block_id] = best_q
                state = _make_state(blocks, assignment, num_gpus, memory_budget)
                current_score = _evaluate_state(blocks, state)
                improved = True

        if not improved:
            break

    return assignment


def partition_edges_communication_aware(
    data: torch.Tensor,
    num_gpus: int,
    *,
    max_block_size: int = 50000,
    low_degree_threshold: int = 5000,
    memory_budget: float = 0.0,
    lambda_c: float = 1.0,
    lambda_m: float = 0.5,
    lambda_r: float = 0.2,
    lambda_a: float = 0.5,
) -> Tuple[Dict[int, int], Dict[str, object]]:
    from end_to_end_prunning.partitioning.edge_block import build_edge_blocks
    from end_to_end_prunning.partitioning.partition_metrics import compute_partition_metrics

    blocks = build_edge_blocks(data, max_block_size=max_block_size, low_degree_threshold=low_degree_threshold)
    if not blocks:
        return {}, {"error": "no edge blocks"}

    assignment = assign_blocks_by_cost_aware_greedy(
        blocks,
        num_gpus,
        memory_budget=memory_budget,
        lambda_c=lambda_c,
        lambda_m=lambda_m,
        lambda_r=lambda_r,
        lambda_a=lambda_a,
    )
    assignment = refine_partition_assignment(
        blocks,
        num_gpus,
        assignment,
        memory_budget=memory_budget,
        lambda_c=lambda_c,
        lambda_m=lambda_m,
        lambda_r=lambda_r,
        lambda_a=lambda_a,
    )
    metrics = compute_partition_metrics(blocks, assignment, num_gpus=num_gpus)
    return assignment, metrics

