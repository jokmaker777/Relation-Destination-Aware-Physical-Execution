import torch
import torch.nn as nn
from typing import Dict, List, Any
from dataclasses import dataclass
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@dataclass
class DeviceConfig:
    """设备配置 - 支持 CUDA 和 CANN"""
    preferred_device: str = "auto"  # "auto", "cuda", "npu", "cpu"

    def get_device(self, local_rank: int = None) -> torch.device:
        """获取计算设备 - 支持 CUDA 和 CANN
        
        Args:
            local_rank: 本地设备编号。如果为None，则使用环境变量LOCAL_RANK，
                       如果环境变量也没有，则默认为0
        """
        # 🔧 分布式支持：从环境变量获取 local_rank
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))


@dataclass
class LayerConfig:
    """层级配置 - 所有参数都从manual_config获取"""
    hidden_dim: int
    dropout_rate: float
    batch_norm: bool = True
    activation: str = "LeakyReLU"


@dataclass
class EdgeConfig:
    """边类型配置 - 所有参数都从manual_config获取"""
    hidden_dim: int
    dropout_rate: float
    batch_norm: bool = True
    activation: str = "LeakyReLU"


@dataclass
class TrainingConfig:
    """训练配置 - 所有参数都从manual_config获取"""
    num_epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    batch_size: int
    l2_lambda: float = 0.001
    # 损失函数计算模式: 'both' (节点+边), 'nodes_only' (仅节点), 'edges_only' (仅边)
    loss_mode: str = 'both'
    # 多卡训练边划分策略: 'all_edges', 'balanced_edge', 'edge', 'node'
    partition_strategy: str = 'balanced_edge'
    # 是否启用图划分
    enable_graph_partition: bool = True


@dataclass
class ModelArchitectureConfig:
    """模型架构配置 - 所有参数都从manual_config获取"""
    num_conv_layers: int
    global_hidden_dim: int


class ModelConfig:
    """简化的模型配置类 - 所有参数都从manual_config获取"""

    def __init__(self, max_layer: int = None):
        # 基本配置 - 不设置任何默认值，全部由manual_config提供
        self.device_config = DeviceConfig()
        self.training_config = None  # 将由manual_config设置
        self.architecture_config = None  # 将由manual_config设置

        # 层级和边配置
        self.layer_configs: Dict[int, LayerConfig] = {}
        self.edge_configs: Dict[tuple, EdgeConfig] = {}
        self.max_layer = max_layer if max_layer is not None else 6

        # 边类型定义 (基于2.py中的实际图结构)
        self.edge_types = [
            # 交付边：库存节点 <-> 交付节点（双向）
            ('inventory', 'to_forecast', 'forecast_delivery'),
            ('forecast_delivery', 'to_forecast', 'inventory'),
            ('inventory', 'to_order', 'order_delivery'),
            ('order_delivery', 'to_order', 'inventory'),
            # 补充库存边：库存节点 <-> 生产节点（双向）
            ('inventory', 'to_replenish', 'production'),
            ('production', 'to_replenish', 'inventory'),
            # 库存剩余边：库存节点 <-> 库存节点（双向）
            ('inventory', 'to_remain', 'inventory'),
            # # 库存运输边：库存节点 <-> 库存节点（双向）
            # ('inventory', 'to_transport', 'inventory'),
            # 虚拟节点边（拆分）：库存节点 <-> 虚拟节点（component 与 alternate 分开，双向）
            ('inventory', 'to_alt', 'virtual'),
            ('virtual', 'to_alt', 'inventory'),
            ('inventory', 'to_comp', 'virtual'),
            ('virtual', 'to_comp', 'inventory'),
            # 虚拟节点边：生产节点 <-> 虚拟节点（双向）
            ('production', 'to', 'virtual'),
            ('virtual', 'to', 'production'),
            # 生产原子项边：生产节点 <-> 库存节点（双向）
            ('production', 'to_2', 'inventory'),
            ('inventory', 'to_2', 'production'),
            # 产能组边：产能组节点 <-> 生产节点（双向）
            ('production', 'to', 'capacitygroup'),
            ('capacitygroup', 'to', 'production'),
            # 产能边：产能组节点 <-> 产能节点（双向）
            ('capacitygroup', 'to', 'capacity'),
            ('capacity', 'to', 'capacitygroup'),
            # 生产对边：生产节点 <-> 生产节点
            ('production', 'pair', 'production'),
            # SR边：SR节点 <-> SR节点
            ('sr', 'to', 'sr'),
            # SR生产边：SR节点 <-> 生产节点（双向）
            ('sr', 'to', 'production'),
            ('production', 'to', 'sr'),
            # 提顶生产边：生产节点 <-> 提顶生产节点（双向）
            ('production', 'to', 'topproduction'),
            ('topproduction', 'to', 'production')
        ]

    def update_max_layer(self, max_layer: int):
        """更新最大层级"""
        self.max_layer = max_layer
        self._init_configs()

    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 如果属性不存在，记录警告但不抛出异常
                print(f"Warning: ModelConfig does not have attribute '{key}', ignoring.")

        # 如果更新了max_layer，重新初始化配置
        if 'max_layer' in kwargs:
            self._init_configs()

    def _init_configs(self):
        """初始化层级和边配置 - 使用architecture_config的值"""
        if self.architecture_config is None:
            return  # 如果architecture_config还没设置，跳过初始化

        # 为所有层级创建配置
        for layer_id in range(-1, self.max_layer + 1):
            if layer_id not in self.layer_configs:
                self.layer_configs[layer_id] = LayerConfig(
                    hidden_dim=self.architecture_config.global_hidden_dim,
                    dropout_rate=0.0  # 将由manual_config设置
                )

        # 为所有边类型创建配置
        for edge_type in self.edge_types:
            if edge_type not in self.edge_configs:
                self.edge_configs[edge_type] = EdgeConfig(
                    hidden_dim=self.architecture_config.global_hidden_dim,
                    dropout_rate=0.0  # 将由manual_config设置
                )

    def get_layer_config(self, layer_id: int) -> LayerConfig:
        """获取层级配置"""
        if layer_id not in self.layer_configs:
            if self.architecture_config is None:
                raise ValueError("architecture_config未设置，无法创建layer_config")
            self.layer_configs[layer_id] = LayerConfig(
                hidden_dim=self.architecture_config.global_hidden_dim,
                dropout_rate=0.0  # 将由manual_config设置
            )
        return self.layer_configs[layer_id]

    def get_edge_config(self, edge_type: tuple) -> EdgeConfig:
        """获取边配置"""
        if edge_type not in self.edge_configs:
            if self.architecture_config is None:
                raise ValueError("architecture_config未设置，无法创建edge_config")
            self.edge_configs[edge_type] = EdgeConfig(
                hidden_dim=self.architecture_config.global_hidden_dim,
                dropout_rate=0.0  # 将由manual_config设置
            )
        return self.edge_configs[edge_type]

    def get_activation_function(self, activation_name: str) -> nn.Module:
        """获取激活函数（NPU兼容）
        
        GNN科研标准：
        - LeakyReLU: GNN消息传递的标准选择（GAT/PNA等），避免死亡神经元
        - ELU: 平滑且负值区域有梯度，适合深层GNN
        - GELU(approximate='tanh'): NPU兼容版本，避免erf算子
        - SiLU/Swish: 平滑非单调激活，现代架构常用
        - PReLU: 可学习斜率的参数化ReLU
        """
        activations = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(negative_slope=0.01),
            'ELU': nn.ELU(alpha=1.0),
            'GELU': nn.GELU(approximate='tanh'),
            'SiLU': nn.SiLU(),
            'PReLU': nn.PReLU(),
            'Mish': nn.Mish(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid()
        }
        return activations.get(activation_name, nn.LeakyReLU(negative_slope=0.01))

    def print_config_summary(self):
        """打印配置摘要"""
        print("\n=== Model Configuration Summary ===")
        print(f"Device: {self.device_config.get_device()}")

        if self.architecture_config:
            print(f"Global Hidden Dim: {self.architecture_config.global_hidden_dim}")
            print(f"Num Conv Layers: {self.architecture_config.num_conv_layers}")
        else:
            print("Architecture Config: 未设置")

        if self.training_config:
            print(f"Learning Rate: {self.training_config.learning_rate}")
            print(f"Num Epochs: {self.training_config.num_epochs}")
            print(f"Batch Size: {self.training_config.batch_size}")
        else:
            print("Training Config: 未设置")

        print(f"Max Layer: {self.max_layer}")
        print("=" * 40)

    def to_dict(self) -> dict:
        """安全的配置序列化方法，避免递归"""
        from datetime import datetime

        config_dict = {
            'config_type': 'ModelConfig',
            'serialization_method': 'native_to_dict',
            'timestamp': datetime.now().isoformat(),
            'max_layer': self.max_layer
        }

        # 设备配置
        if self.device_config:
            config_dict['device_config'] = {
                'preferred_device': getattr(self.device_config, 'preferred_device', 'auto')
            }

        # 训练配置
        if self.training_config:
            config_dict['training_config'] = {
                'num_epochs': getattr(self.training_config, 'num_epochs', 0),
                'learning_rate': getattr(self.training_config, 'learning_rate', 0.0),
                'batch_size': getattr(self.training_config, 'batch_size', 0),
                'weight_decay': getattr(self.training_config, 'weight_decay', 0.0),
                'patience': getattr(self.training_config, 'patience', 0)
            }

        # 架构配置
        if self.architecture_config:
            config_dict['architecture_config'] = {
                'global_hidden_dim': getattr(self.architecture_config, 'global_hidden_dim', 0),
                'num_conv_layers': getattr(self.architecture_config, 'num_conv_layers', 0)
            }

        # 边类型列表
        config_dict['edge_types'] = [str(edge_type) for edge_type in self.edge_types]

        # 简化的层级和边配置统计
        config_dict['layer_configs_count'] = len(self.layer_configs)
        config_dict['edge_configs_count'] = len(self.edge_configs)

        return config_dict


def create_preset_config(preset_name: str, max_layer: int = None) -> ModelConfig:
    """创建预设配置 - 所有参数最终会被manual_config覆盖"""
    return ModelConfig(max_layer=max_layer)
