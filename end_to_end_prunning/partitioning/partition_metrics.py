from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import torch

from .edge_block import EdgeBlock


def _to_scalar_block_cost(block: EdgeBlock) -> float:
    if isinstance(block.compute_cost, (int, float)):
        return float(block.compute_cost)
    return float(block.num_edges)


def compute_partition_metrics(
    blocks: List[EdgeBlock],
    assignment: Union[Dict[int, int], torch.Tensor],
    num_gpus: int,
    bytes_per_value: int = 4,
    hidden_dim: int = 64,
    edge_index_bytes_per_edge: int = 16,
    *,
    aggregation_weights: Optional[Dict[Tuple[str, int], float]] = None,
) -> Dict[str, object]:
    """统计给定划分下的负载、边类型分布、rho、冲突和显存估计。"""
    if isinstance(assignment, torch.Tensor):
        if assignment.ndim != 1 or assignment.shape[0] != len(blocks):
            raise ValueError(
                f"assignment tensor shape {assignment.shape} 与 blocks 数量 {len(blocks)} 不匹配"
            )
        assignment = {block.block_id: int(assignment[i].item()) for i, block in enumerate(blocks)}

    if num_gpus <= 0:
        raise ValueError("num_gpus 必须为正整数")

    compute_load = [0.0 for _ in range(num_gpus)]
    memory_load = [0.0 for _ in range(num_gpus)]
    type_load: Dict[int, Dict[Tuple[str, str, str], float]] = defaultdict(lambda: defaultdict(float))
    target_to_gpus: Dict[Tuple[str, int], Set[int]] = defaultdict(set)
    memory_estimates = [0.0 for _ in range(num_gpus)]

    edge_index_bytes = edge_index_bytes_per_edge
    bytes_per_value = max(1, int(bytes_per_value))
    hidden_dim = max(0, int(hidden_dim))

    for block in blocks:
        if not isinstance(block, EdgeBlock):
            continue

        rank = int(assignment.get(block.block_id, 0))
        if rank < 0 or rank >= num_gpus:
            raise ValueError(
                f"block {block.block_id} 被分配到 rank {rank}，超出 [0, {num_gpus}) 范围"
            )

        cost = _to_scalar_block_cost(block)
        compute_load[rank] += cost

        type_key = block.edge_type
        type_load[rank][type_key] += cost

        key = (block.dst_node_type, block.dst_node_id)
        target_to_gpus[key].add(rank)

        mem = 0.0
        mem += edge_index_bytes * block.num_edges
        mem += bytes_per_value * block.num_edges * block.edge_feat_dim
        mem += bytes_per_value * 2 * block.num_edges * hidden_dim
        mem += bytes_per_value * block.num_edges * 64
        block.memory_cost = mem
        memory_load[rank] += mem
        memory_estimates[rank] += mem

    rhos: List[float] = []
    conflict_nodes = 0
    aggregation_contention = 0
    max_rho = 0.0
    avg_rho = 0.0
    total_weighted_contention = 0.0

    for key, gpu_set in target_to_gpus.items():
        rho = float(len(gpu_set))
        rhos.append(rho)
        if rho > 1:
            conflict_nodes += 1
            contention = rho - 1
            aggregation_contention += contention
            weight = 1.0
            if aggregation_weights is not None and key in aggregation_weights:
                weight = float(aggregation_weights[key])
            total_weighted_contention += weight * contention
        if rho > max_rho:
            max_rho = rho

    if rhos:
        avg_rho = sum(rhos) / len(rhos)

    max_compute_load = max(compute_load) if compute_load else 0.0
    avg_compute_load = sum(compute_load) / len(compute_load) if compute_load else 0.0
    max_memory = max(memory_estimates) if memory_estimates else 0.0

    return {
        "compute_load": compute_load,
        "memory_load": memory_load,
        "type_load": {rank: dict(type_load[rank]) for rank in range(num_gpus)},
        "rho": {
            "max": max_rho,
            "avg": avg_rho,
            "values": rhos,
            "conflict_nodes": conflict_nodes,
            "aggregation_contention": aggregation_contention,
            "weighted_aggregation_contention": total_weighted_contention,
        },
        "memory": {
            "max_estimate": max_memory,
            "per_gpu": memory_estimates,
        },
        "summary": {
            "num_blocks": len(blocks),
            "num_gpus": num_gpus,
            "max_compute_load": max_compute_load,
            "avg_compute_load": avg_compute_load,
            "load_imbalance": max_compute_load / avg_compute_load if avg_compute_load > 0 else 0.0,
            "max_memory_estimate": max_memory,
        },
    }

