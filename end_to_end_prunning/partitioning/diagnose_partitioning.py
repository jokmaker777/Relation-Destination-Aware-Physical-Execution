"""训练前诊断脚本：输入图结构 pkl，输出 partition metrics.csv。

支持输入格式：
- 单个 HeteroData 的 pkl
- HeteroData 列表的 pkl
- 缓存目录下多个 graph_*.pkl

配置方式：
- 直接修改 ``PARTITION_DIAGNOSE_CONFIG`` 中的参数
- 或在代码中调用 ``run_diagnostics(graphs, graph_files, ...)``
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch_geometric.data import HeteroData

try:
    from end_to_end_prunning.config.partition_diagnose_config import (
        PARTITION_DIAGNOSE_CONFIG,
        PartitionDiagnoseConfig,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from end_to_end_prunning.config.partition_diagnose_config import (
        PARTITION_DIAGNOSE_CONFIG,
        PartitionDiagnoseConfig,
    )

try:
    from .edge_block import EdgeBlock, build_edge_blocks
    from .partition_metrics import compute_partition_metrics
    from .hypergraph_partitioner import (
        assign_blocks_by_cost_aware_greedy,
        refine_partition_assignment,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from end_to_end_prunning.partitioning.edge_block import EdgeBlock, build_edge_blocks
    from end_to_end_prunning.partitioning.partition_metrics import compute_partition_metrics
    from end_to_end_prunning.partitioning.hypergraph_partitioner import (
        assign_blocks_by_cost_aware_greedy,
        refine_partition_assignment,
    )


def _coerce_hetero_data_list(obj: object) -> List[HeteroData]:
    if isinstance(obj, HeteroData):
        return [obj]
    if isinstance(obj, list):
        graphs = []
        for item in obj:
            if isinstance(item, HeteroData):
                graphs.append(item)
            else:
                raise TypeError(f"列表中包含非 HeteroData 元素: {type(item).__name__}")
        return graphs
    raise TypeError(f"不支持的 pkl 顶层类型: {type(obj).__name__}")


def load_hetero_graphs_from_pkl(path: Union[str, Path]) -> List[HeteroData]:
    path = Path(path)

    def _load(target: Path) -> List[HeteroData]:
        import pickle

        with target.open("rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "graph" in data:
            data = data["graph"]

        graphs = _coerce_hetero_data_list(data)
        if not graphs:
            raise ValueError(f"pkl 中未找到 HeteroData: {target}")
        return graphs

    if path.suffix == ".tmp":
        if path.exists():
            return _load(path)
        fallback = path.with_suffix("")
        if fallback.exists():
            return _load(fallback)
        raise FileNotFoundError(path)

    if path.exists():
        return _load(path)

    temp_path = Path(str(path) + ".tmp")
    if temp_path.exists():
        return _load(temp_path)

    raise FileNotFoundError(path)


def iter_graph_files(
    directory: Union[str, Path],
    pattern: str = "graph_*.pkl",
) -> Iterable[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        tmp_pattern = pattern.replace(".pkl", ".pkl.tmp")
        if tmp_pattern != pattern:
            files = sorted(directory.glob(tmp_pattern))
    if not files:
        raise FileNotFoundError(f"目录下未匹配到文件: {directory / pattern}")
    return files


def _edge_count_balanced_assignment(
    blocks: List[EdgeBlock],
    num_gpus: int,
) -> Dict[int, int]:
    blocks = sorted(blocks, key=lambda b: b.num_edges, reverse=True)
    load = [0.0 for _ in range(num_gpus)]
    assignment: Dict[int, int] = {}
    for block in blocks:
        cost = float(block.num_edges)
        p = min(range(num_gpus), key=lambda rank: load[rank])
        assignment[block.block_id] = p
        load[p] += cost
    return assignment


def _target_grouped_assignment(
    blocks: List[EdgeBlock],
    num_gpus: int,
) -> Dict[int, int]:
    groups: Dict[Tuple[str, int], List[EdgeBlock]] = defaultdict(list)
    for block in blocks:
        groups[(block.dst_node_type, block.dst_node_id)].append(block)

    load = [0.0 for _ in range(num_gpus)]
    assignment: Dict[int, int] = {}
    for group in groups.values():
        p = min(range(num_gpus), key=lambda rank: load[rank])
        cost = sum(float(b.num_edges) for b in group)
        for block in group:
            assignment[block.block_id] = p
        load[p] += cost
    return assignment


def _cost_aware_assignment(
    blocks: List[EdgeBlock],
    num_gpus: int,
    memory_budget: float = 0.0,
) -> Dict[int, int]:
    assignment = assign_blocks_by_cost_aware_greedy(
        blocks,
        num_gpus,
        memory_budget=memory_budget,
    )
    assignment = refine_partition_assignment(
        blocks,
        num_gpus,
        assignment,
        memory_budget=memory_budget,
        max_iters=1,
    )
    return assignment


def evaluate_strategy(
    graph: HeteroData,
    strategy: str,
    num_gpus: int,
    max_block_size: int = 50000,
    low_degree_threshold: int = 5000,
    memory_budget: float = 0.0,
    *,
    hidden_dim: int = 64,
    bytes_per_value: int = 4,
    edge_index_bytes_per_edge: int = 16,
    aggregation_weights: Optional[Dict[Tuple[str, int], float]] = None,
) -> Dict[str, object]:
    blocks = build_edge_blocks(
        graph,
        max_block_size=max_block_size,
        low_degree_threshold=low_degree_threshold,
    )
    if not blocks:
        raise ValueError("未构造出任何 EdgeBlock，无法计算 metrics")

    if strategy == "edge_count_balanced":
        assignment = _edge_count_balanced_assignment(blocks, num_gpus)
    elif strategy == "target_grouped":
        assignment = _target_grouped_assignment(blocks, num_gpus)
    elif strategy == "cost_aware":
        assignment = _cost_aware_assignment(blocks, num_gpus, memory_budget=memory_budget)
    else:
        raise ValueError(f"未知策略: {strategy}")

    metrics = compute_partition_metrics(
        blocks,
        assignment,
        num_gpus=num_gpus,
        bytes_per_value=bytes_per_value,
        hidden_dim=hidden_dim,
        edge_index_bytes_per_edge=edge_index_bytes_per_edge,
        aggregation_weights=aggregation_weights,
    )
    summary = metrics["summary"]
    rho = metrics["rho"]
    memory = metrics["memory"]
    return {
        "graph_file": "",
        "strategy": strategy,
        "num_gpus": num_gpus,
        "num_blocks": int(summary["num_blocks"]),
        "max_compute_load": float(summary["max_compute_load"]),
        "avg_compute_load": float(summary["avg_compute_load"]),
        "load_imbalance": float(summary["load_imbalance"]),
        "max_rho": float(rho["max"]),
        "avg_rho": float(rho["avg"]),
        "conflict_nodes": int(rho["conflict_nodes"]),
        "aggregation_contention": int(rho["aggregation_contention"]),
        "weighted_aggregation_contention": float(rho["weighted_aggregation_contention"]),
        "max_memory_estimate": float(memory["max_estimate"]),
    }


_FIELDNAMES = [
    "graph_file",
    "strategy",
    "num_gpus",
    "num_blocks",
    "max_compute_load",
    "avg_compute_load",
    "load_imbalance",
    "max_rho",
    "avg_rho",
    "conflict_nodes",
    "aggregation_contention",
    "weighted_aggregation_contention",
    "max_memory_estimate",
]


def run_diagnostics(
    graphs: List[HeteroData],
    graph_files: List[str],
    num_gpus: int = 4,
    strategies: Optional[List[str]] = None,
    max_block_size: int = 50000,
    low_degree_threshold: int = 5000,
    memory_budget: float = 0.0,
    output_csv: Optional[Union[str, Path]] = None,
    *,
    hidden_dim: int = 64,
    bytes_per_value: int = 4,
    edge_index_bytes_per_edge: int = 16,
    aggregation_weights: Optional[Dict[Tuple[str, int], float]] = None,
) -> List[Dict[str, object]]:
    if strategies is None:
        strategies = ["edge_count_balanced", "target_grouped", "cost_aware"]
    if len(graphs) != len(graph_files):
        raise ValueError("graphs 与 graph_files 长度不一致")

    rows: List[Dict[str, object]] = []
    for graph, graph_file in zip(graphs, graph_files):
        for strategy in strategies:
            try:
                row = evaluate_strategy(
                    graph,
                    strategy,
                    num_gpus=num_gpus,
                    max_block_size=max_block_size,
                    low_degree_threshold=low_degree_threshold,
                    memory_budget=memory_budget,
                    hidden_dim=hidden_dim,
                    bytes_per_value=bytes_per_value,
                    edge_index_bytes_per_edge=edge_index_bytes_per_edge,
                    aggregation_weights=aggregation_weights,
                )
                row["graph_file"] = graph_file
                rows.append(row)
            except Exception as exc:  # pragma: no cover - diagnostic path
                rows.append({
                    "graph_file": graph_file,
                    "strategy": strategy,
                    "num_gpus": num_gpus,
                    "num_blocks": -1,
                    "max_compute_load": float("nan"),
                    "avg_compute_load": float("nan"),
                    "load_imbalance": float("nan"),
                    "max_rho": float("nan"),
                    "avg_rho": float("nan"),
                    "conflict_nodes": -1,
                    "aggregation_contention": -1,
                    "weighted_aggregation_contention": float("nan"),
                    "max_memory_estimate": float("nan"),
                    "error": str(exc),
                })

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(_FIELDNAMES)
        if any("error" in r for r in rows):
            fieldnames.append("error")
        with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    return rows


def _load_config() -> PartitionDiagnoseConfig:
    try:
        from end_to_end_prunning.config.partition_diagnose_config import PARTITION_DIAGNOSE_CONFIG
        return PARTITION_DIAGNOSE_CONFIG
    except ImportError:
        return PartitionDiagnoseConfig()


def main() -> None:
    cfg = _load_config()

    if cfg.pkl and cfg.pkl_dir:
        raise ValueError("pkl 与 pkl_dir 仅能使用一个")
    if not cfg.pkl and not cfg.pkl_dir:
        raise ValueError("必须提供 pkl 或 pkl_dir")

    if cfg.pkl:
        graph_files = [str(cfg.pkl)]
        graphs = load_hetero_graphs_from_pkl(cfg.pkl)
    else:
        graph_files = [str(p) for p in iter_graph_files(cfg.pkl_dir, cfg.pattern)]
        graphs = []
        for path in graph_files:
            graphs.extend(load_hetero_graphs_from_pkl(path))

    output_csv = cfg.output_csv
    if output_csv is None:
        base = Path(cfg.pkl or cfg.pkl_dir)
        if base.is_dir():
            output_csv = base / "partition_metrics.csv"
        else:
            output_csv = base.with_suffix(".partition_metrics.csv")

    rows = run_diagnostics(
        graphs,
        graph_files,
        num_gpus=cfg.num_gpus,
        strategies=cfg.strategies,
        max_block_size=cfg.max_block_size,
        low_degree_threshold=cfg.low_degree_threshold,
        memory_budget=cfg.memory_budget,
        output_csv=output_csv,
        hidden_dim=cfg.hidden_dim,
        bytes_per_value=cfg.bytes_per_value,
        edge_index_bytes_per_edge=cfg.edge_index_bytes_per_edge,
    )
    print(f"已输出 metrics: {output_csv}")
    for row in rows[: min(20, len(rows))]:
        if "error" in row:
            print(f"[ERROR] {row['graph_file']} | {row['strategy']} | {row['error']}")
        else:
            print(
                f"{Path(row['graph_file']).name} | {row['strategy']} | "
                f"blocks={row['num_blocks']} | load_imb={row['load_imbalance']:.3f} | "
                f"max_rho={row['max_rho']:.2f} | conflict_nodes={row['conflict_nodes']} | "
                f"agg_contention={row['aggregation_contention']} | max_mem={row['max_memory_estimate']:.2f}"
            )


if __name__ == "__main__":
    main()

