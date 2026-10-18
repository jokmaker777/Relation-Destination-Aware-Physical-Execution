from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData


@dataclass
class EdgeBlock:
    block_id: int

    # 异构边类型，例如 ('inventory', 'to_transport_labeled', 'inventory')
    edge_type: Tuple[str, str, str]

    # 目标节点类型和目标节点 local id
    dst_node_type: str
    dst_node_id: int

    # 这个边块包含的边在原 edge_index 中的位置
    edge_ids: torch.Tensor

    # 边块基本统计
    num_edges: int
    edge_feat_dim: int
    num_labeled_edges: int

    # 可选：时间属性、period_code、边类型组等
    period_code: Optional[int] = None

    # profiling 后得到
    compute_cost: Optional[float] = None
    memory_cost: Optional[float] = None

    # 聚合权重：可后续替换为业务重要性或度数相关权重
    aggregation_weight: Optional[float] = None


def _get_edge_feat_dim(data: HeteroData, edge_type: Tuple[str, str, str]) -> int:
    if edge_type not in data.edge_types:
        return 0

    store = data[edge_type]
    if hasattr(store, "edge_attr") and store.edge_attr is not None:
        feat = store.edge_attr
        if feat.ndim == 2:
            return int(feat.shape[1])
    return 0


def _get_num_labeled_edges(
    data: HeteroData,
    edge_type: Tuple[str, str, str],
    edge_ids: torch.Tensor,
) -> int:
    if edge_type not in data.edge_types:
        return 0

    store = data[edge_type]
    if hasattr(store, "y") and store.y is not None:
        try:
            labels = store.y[edge_ids]
            if labels.ndim > 1:
                labels = labels.squeeze()
            return int((labels > 0).sum().item())
        except Exception:
            return 0
    return 0


def _split_edge_ids(edge_ids: torch.Tensor, max_block_size: int) -> List[torch.Tensor]:
    if max_block_size <= 0 or edge_ids.numel() <= max_block_size:
        return [edge_ids]
    return list(edge_ids.split(max_block_size))


def build_edge_blocks(
    data: HeteroData,
    max_block_size: int = 50000,
    low_degree_threshold: int = 5000,
    edge_types: Optional[List[Tuple[str, str, str]]] = None,
) -> List[EdgeBlock]:
    """将 HeteroData 的边组织为 EdgeBlock 列表。

    第一版边块按 ``(edge_type, target_node)`` 构造：
    - 低度数目标节点：入边整体成块；
    - 高度数目标节点：入边按 ``max_block_size`` 受控拆分。
    """
    if edge_types is None:
        edge_types = list(data.edge_types)

    blocks: List[EdgeBlock] = []
    block_id = 0

    for edge_type in edge_types:
        if edge_type not in data.edge_types:
            continue

        src_type, rel_type, dst_type = edge_type
        edge_store = data[edge_type]
        if not hasattr(edge_store, "edge_index") or edge_store.edge_index is None:
            continue

        edge_index = edge_store.edge_index
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            continue

        num_edges = int(edge_index.shape[1])
        if num_edges == 0:
            continue

        dst_nodes = edge_index[1]
        unique_dst_nodes = torch.unique(dst_nodes)

        edge_feat_dim = _get_edge_feat_dim(data, edge_type)

        for dst in unique_dst_nodes:
            edge_ids = torch.nonzero(dst_nodes == dst, as_tuple=False).view(-1)
            degree = int(edge_ids.numel())

            if degree <= low_degree_threshold:
                blocks.append(
                    EdgeBlock(
                        block_id=block_id,
                        edge_type=edge_type,
                        dst_node_type=dst_type,
                        dst_node_id=int(dst.item()),
                        edge_ids=edge_ids,
                        num_edges=degree,
                        edge_feat_dim=edge_feat_dim,
                        num_labeled_edges=_get_num_labeled_edges(data, edge_type, edge_ids),
                    )
                )
                block_id += 1
            else:
                for chunk in _split_edge_ids(edge_ids, max_block_size):
                    blocks.append(
                        EdgeBlock(
                            block_id=block_id,
                            edge_type=edge_type,
                            dst_node_type=dst_type,
                            dst_node_id=int(dst.item()),
                            edge_ids=chunk,
                            num_edges=int(chunk.numel()),
                            edge_feat_dim=edge_feat_dim,
                            num_labeled_edges=_get_num_labeled_edges(data, edge_type, chunk),
                        )
                    )
                    block_id += 1

    return blocks

