from __future__ import annotations

from typing import List

from torch_geometric.data import HeteroData

from end_to_end_prunning.partitioning.edge_block import EdgeBlock, build_edge_blocks
from end_to_end_prunning.partitioning.partition_metrics import compute_partition_metrics


def demo_partition_metrics(data: HeteroData) -> None:
    blocks: List[EdgeBlock] = build_edge_blocks(data)

    if not blocks:
        print("未构造出任何 EdgeBlock，请检查输入的 HeteroData 是否包含边。")
        return

    num_gpus = 4
    assignment = {block.block_id: block.block_id % num_gpus for block in blocks}
    metrics = compute_partition_metrics(blocks, assignment, num_gpus=num_gpus)

    summary = metrics["summary"]
    rho = metrics["rho"]
    memory = metrics["memory"]

    print(f"EdgeBlock 数量: {summary['num_blocks']}")
    print(f"GPU 数量: {summary['num_gpus']}")
    print(f"每 GPU 平均计算负载: {summary['avg_compute_load']:.2f}")
    print(f"最大计算负载: {summary['max_compute_load']:.2f}")
    print(f"负载不均衡比例: {summary['load_imbalance']:.4f}")
    print(f"最大聚合跨度 rho: {rho['max']:.2f}")
    print(f"平均聚合跨度 rho: {rho['avg']:.2f}")
    print(f"冲突节点数: {rho['conflict_nodes']}")
    print(f"聚合冲突总量: {rho['aggregation_contention']}")
    print(f"加权聚合冲突总量: {rho['weighted_aggregation_contention']:.2f}")
    print(f"最大显存估计: {memory['max_estimate']:.2f}")

