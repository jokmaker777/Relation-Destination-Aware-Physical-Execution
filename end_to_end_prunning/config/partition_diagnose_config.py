"""
训练前划分诊断参数配置
直接修改 PARTITION_DIAGNOSE_CONFIG 中的参数即可
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PartitionDiagnoseConfig:
    """训练前划分诊断参数配置"""

    # ============= 输入数据配置 =============
    pkl: Optional[str] = None
    pkl_dir: Optional[str] = None
    pattern: str = "graph_*.pkl"

    # ============= 划分诊断配置 =============
    num_gpus: int = 4
    strategies: List[str] = None
    max_block_size: int = 50000
    low_degree_threshold: int = 5000
    memory_budget: float = 0.0

    # ============= 显存估计配置 =============
    hidden_dim: int = 64
    bytes_per_value: int = 4
    edge_index_bytes_per_edge: int = 16

    # ============= 输出配置 =============
    output_csv: Optional[str] = None

    def __post_init__(self):
        if self.strategies is None:
            self.strategies = ["edge_count_balanced", "target_grouped", "cost_aware"]

    def print_config(self):
        print("\n" + "=" * 60)
        print("训练前划分诊断参数配置")
        print("=" * 60)

        print("\n输入数据配置:")
        print(f"  单文件模式: {self.pkl}")
        print(f"  目录模式: {self.pkl_dir}")
        print(f"  目录匹配模式: {self.pattern}")

        print("\n划分诊断配置:")
        print(f"  模拟GPU数量: {self.num_gpus}")
        print(f"  诊断策略: {self.strategies}")
        print(f"  高度数节点单块最大边数: {self.max_block_size}")
        print(f"  低度数节点阈值: {self.low_degree_threshold}")
        print(f"  单卡显存预算: {self.memory_budget}")

        print("\n显存估计配置:")
        print(f"  节点隐层维度: {self.hidden_dim}")
        print(f"  每个数值占用字节数: {self.bytes_per_value}")
        print(f"  每条边 edge_index 占用字节数: {self.edge_index_bytes_per_edge}")

        print("\n输出配置:")
        print(f"  metrics.csv 输出路径: {self.output_csv}")
        print("=" * 60)


# ============= 唯一配置 - 直接修改这里的参数 =============

PARTITION_DIAGNOSE_CONFIG = PartitionDiagnoseConfig(
    pkl="/path/to/artifact_workspace/cache/graph_20241126.pkl",
    pkl_dir=None,
    pattern="graph_*.pkl",
    num_gpus=2,
    strategies=["edge_count_balanced", "target_grouped", "cost_aware"],
    max_block_size=50000,
    low_degree_threshold=5000,
    memory_budget=0.0,
    hidden_dim=64,
    bytes_per_value=4,
    edge_index_bytes_per_edge=16,
    output_csv=None,
)

