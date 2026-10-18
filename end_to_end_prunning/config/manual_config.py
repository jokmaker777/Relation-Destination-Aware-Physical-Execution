"""
手动参数配置文件 - 简化版本
直接修改 MANUAL_CONFIG 中的参数即可
"""

from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime
import torch
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



@dataclass
class ManualTrainingConfig:
    """手动训练参数配置 - 简化版本"""
    
    # ============= 数据集配置 =============
    train_start_date: datetime
    train_end_date: datetime
    val_start_date: datetime
    val_end_date: datetime
    test_start_date: datetime
    test_end_date: datetime
    
    # ============= 核心训练参数 =============
    num_epochs: int
    learning_rate: float
    batch_size: int
    weight_decay: float
    patience: int
    l2_lambda: float
    loss_mode: str  # 损失函数计算模式: 'both', 'nodes_only', 'edges_only'

    # ============= 模型架构参数 =============
    global_hidden_dim: int
    num_conv_layers: int
    dropout_rate: float
    
    # ============= 数据处理参数 =============
    previous_days: int
    cache_version: str
    
    # ============= 设备配置 =============
    device: str
    
    # ============= 多卡训练配置 =============
    # 边划分策略: 
    #   'all_edges' - 对所有边类型均匀划分（显存优化最大，推荐显存不足时使用）
    #   'balanced_edge' - 只划分标签边（默认，兼容性最好）
    #   'edge' - 原始边划分
    #   'node' - 节点划分（会导致边被过滤）
    partition_strategy: str = 'balanced_edge'

    
    def print_config(self):
        """打印配置信息"""
        print("\n" + "="*60)
        print("手动训练参数配置")
        print("="*60)
        
        print("\n数据集配置:")
        print(f"  训练集: {self.train_start_date.strftime('%Y-%m-%d')} 到 {self.train_end_date.strftime('%Y-%m-%d')}")
        print(f"  验证集: {self.val_start_date.strftime('%Y-%m-%d')} 到 {self.val_end_date.strftime('%Y-%m-%d')}")
        print(f"  测试集: {self.test_start_date.strftime('%Y-%m-%d')} 到 {self.test_end_date.strftime('%Y-%m-%d')}")
        
        # 计算数据集天数
        train_days = (self.train_end_date - self.train_start_date).days + 1
        val_days = (self.val_end_date - self.val_start_date).days + 1
        test_days = (self.test_end_date - self.test_start_date).days + 1
        total_days = train_days + val_days + test_days
        print(f"  数据集天数: 训练{train_days}天, 验证{val_days}天, 测试{test_days}天 (总计{total_days}天)")
        
        print("\n核心训练参数:")
        print(f"  训练轮数 (num_epochs): {self.num_epochs}")
        print(f"  学习率 (learning_rate): {self.learning_rate}")
        print(f"  批大小 (batch_size): {self.batch_size}")
        print(f"  权重衰减 (weight_decay): {self.weight_decay}")
        print(f"  早停耐心值 (patience): {self.patience}")
        print(f"  L2正则化系数 (l2_lambda): {self.l2_lambda}")
        print(f"  损失函数模式 (loss_mode): {self.loss_mode}")

        print("\n模型架构参数:")
        print(f"  全局隐藏维度 (global_hidden_dim): {self.global_hidden_dim}")
        print(f"  卷积层数 (num_conv_layers): {self.num_conv_layers}")
        print(f"  Dropout率 (dropout_rate): {self.dropout_rate}")
        
        print("\n数据处理参数:")
        print(f"  历史数据天数 (previous_days): {self.previous_days}")
        print(f"  缓存版本 (cache_version): {self.cache_version}")
        
        print("\n设备配置:")
        print(f"  设备 (device): {self.device}")
        
        print("\n多卡训练配置:")
        print(f"  边划分策略 (partition_strategy): {self.partition_strategy}")
        partition_desc = {
            'all_edges': '对所有边类型均匀划分（显存优化最大）',
            'balanced_edge': '只划分标签边（默认）',
            'edge': '原始边划分',
            'node': '节点划分'
        }
        print(f"    说明: {partition_desc.get(self.partition_strategy, '未知策略')}")
        
        print("="*60)


# ============= 唯一配置 - 直接修改这里的参数 =============

MANUAL_CONFIG = ManualTrainingConfig(
    # 数据集配置
    train_start_date=datetime(2024, 11, 26),
    train_end_date=datetime(2024, 11, 26),
    val_start_date=datetime(2024, 11, 26),
    val_end_date=datetime(2024, 11, 26),
    test_start_date=datetime(2024, 11, 26),
    test_end_date=datetime(2024, 11, 26),

    # 核心训练参数
    num_epochs=2,                  # 训练轮数
    learning_rate=0.0005,          # 学习率
    batch_size=1,                 # 批大小
    weight_decay=5e-5,             # 权重衰减
    patience=50,                   # 早停耐心值
    l2_lambda=5e-2,                 # L2正则化系数
    loss_mode='both',              # 损失函数模式: 'both', 'nodes_only', 'edges_only'

    # 模型架构参数
    global_hidden_dim=32,          # 全局隐藏层维度
    num_conv_layers=1,             # 卷积层数
    dropout_rate=0.08,              # Dropout率

    # 数据处理参数
    previous_days=3,               # 历史数据天数（严格=7）
    cache_version="v1",
    # 设备配置
    device="auto",                 # 设备选择: "auto", "cpu", "cuda"
) 
