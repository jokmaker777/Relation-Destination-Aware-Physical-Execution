import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

from ..config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class LayerBasedMessagePassing(HeteroConv):

    def __init__(self, conv_dict, aggr='mean'):
        super().__init__(conv_dict, aggr)

    def forward(self, x_dict, edge_index_dict):
        """传统GCN消息传递 - 使用标准PyG HeteroConv机制"""
        try:
            # 直接调用父类forward
            return super().forward(x_dict, edge_index_dict)
        except Exception as e:
            logger.error(f"Critical error in traditional GCN message passing: {e}")
            # 返回输入作为回退
            return x_dict

    def forward_for_initialization(self, x_dict, edge_index_dict):
        """初始化时的前向传播，与标准forward完全一致"""
        return self.forward(x_dict, edge_index_dict)


class SupplyChainGCN(nn.Module):
    def __init__(self, node_features_dim_dict: dict, config: ModelConfig, previous_days: int = 7):
        super().__init__()
        self.config = config

        # AMP attributes must be set early, before any .to() or lazy-init that might trigger forward()
        self._amp_enabled = False
        self._amp_device_type = None
        self._amp_dtype = torch.float32

        # 初始设备（可能会在DDP中被外部移动）
        try:
            self._init_device = config.device_config.get_device()
        except Exception as e:
            logger.warning(f"Device initialization failed: {e}, falling back to CPU")
            self._init_device = torch.device("cpu")

        # 从配置获取参数（移除所有硬编码）
        self.hidden_dim = config.architecture_config.global_hidden_dim
        self.num_layers = config.architecture_config.num_conv_layers
        # 移除 num_iterations，使用传统GCN消息传递
        # self.num_iterations = config.training_config.num_iterations

        # 正则化参数
        self.l2_lambda = getattr(config.training_config, 'l2_lambda', 0.001)
        
        # 损失模式：控制是否进行边预测
        # 'both' - 节点和边都预测
        # 'nodes_only' - 只预测节点，跳过边预测
        # 'edges_only' - 只预测边
        self.loss_mode = getattr(config.training_config, 'loss_mode', 'both')
        logger.info(f"SupplyChainGCN initialized with loss_mode={self.loss_mode}")

        # 训练历史记录
        self.training_history = {
            'train_loss': [],
            'train_metrics': defaultdict(list)
        }

        # 为每种节点类型创建特征投影层
        self.feature_projection = nn.ModuleDict({
            node_type: nn.Linear(in_dim, self.hidden_dim)
            for node_type, in_dim in node_features_dim_dict.items()
        })

        # Edge types（从配置获取）
        self.edge_types = config.edge_types

        # Create message passing layers - 统一使用global_hidden_dim
        self.conv_layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                # 统一使用global_hidden_dim，确保维度一致性
                conv_dict[edge_type] = SAGEConv((-1, -1), self.hidden_dim)

            self.conv_layers.append(LayerBasedMessagePassing(conv_dict))

        # GNN层间后处理：LayerNorm + 激活 + Dropout + 残差
        layer_config_0 = config.get_layer_config(0)
        self.conv_norms = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            norm_dict = {}
            for node_type in node_features_dim_dict:
                norm_dict[node_type] = nn.LayerNorm(self.hidden_dim)
            self.conv_norms.append(nn.ModuleDict(norm_dict))
        self.gnn_activation = config.get_activation_function(layer_config_0.activation)
        self.gnn_dropout_rate = layer_config_0.dropout_rate

        # 定义需要预测的节点类型
        self.target_node_types = ['production', 'inventory']

        # Dropout layers（只为需要预测的节点类型创建）
        self.dropout = nn.ModuleDict()
        for node_type in self.target_node_types:
            if node_type in node_features_dim_dict:
                # 使用第0层配置作为默认
                layer_config = config.get_layer_config(0)
                self.dropout[node_type] = nn.Dropout(layer_config.dropout_rate)

        # Output prediction layers（只为需要预测的节点类型创建）
        self.predictor = nn.ModuleDict()
        for node_type in self.target_node_types:
            if node_type in node_features_dim_dict:
                # 根据层级配置获取参数
                layer_config = config.get_layer_config(0)  # 使用第0层配置作为默认

                # 构建预测器
                layers = []
                layers.append(nn.Linear(self.hidden_dim, layer_config.hidden_dim))

                # 添加激活函数
                activation = config.get_activation_function(layer_config.activation)
                layers.append(activation)

                # 添加dropout
                layers.append(nn.Dropout(layer_config.dropout_rate))

                # 输出层
                layers.append(nn.Linear(layer_config.hidden_dim, 1))

                self.predictor[node_type] = nn.Sequential(*layers)

        # 边预测器 (支持动态维度的边预测器)
        self.edge_predictor = nn.ModuleDict()
        self.layer_config = config.get_layer_config(0)  # 保存配置用于动态创建

        # 节点特征维度 = hidden_dim (经过GCN更新后的节点特征)
        self.node_feature_dim = self.layer_config.hidden_dim

        # 支持的边类型列表
        self.supported_edge_types = [
            'inventory_to_virtual',
            'inventory_to_transport_labeled',
            'inventory_to_transport_reverse'
        ]

        # 存储动态创建的预测器信息
        self.dynamic_predictors = {}

        logger.info(f"Edge predictor initialization:")
        logger.info(f"  Node feature dim (post-GCN): {self.node_feature_dim}")
        logger.info(f"  Supported edge types: {self.supported_edge_types}")

        # 将模型先移动到初始设备
        try:
            self.to(self._init_device)
            logger.info(f"Initialized SupplyChainGCN on device: {self._init_device}")
            for predictor in self.edge_predictor.values():
                predictor.to(self._init_device)
        except Exception as e:
            logger.warning(f"Failed to move model to {self._init_device}: {e}")
            self.to(torch.device("cpu"))
            for predictor in self.edge_predictor.values():
                predictor.to(torch.device("cpu"))
            logger.info("Fallback: Initialized SupplyChainGCN on CPU")

        logger.info(f"Model: hidden_dim={self.hidden_dim}, num_layers={self.num_layers}")
        logger.info(f"All edge convolutions use unified hidden_dim: {self.hidden_dim}")
        logger.info(f"GNN inter-layer: LayerNorm -> {self.gnn_activation.__class__.__name__} -> Dropout({self.gnn_dropout_rate}) -> Residual")
        logger.info(f"Using traditional GCN message passing mechanism (PyG HeteroConv)")

    def configure_amp(self, device_type: str, dtype: torch.dtype):
        """
        配置细粒度混合精度：仅对 SAGEConv 消息传递层使用低精度
        
        特征投影 (float32) → SAGEConv×N (bfloat16/float16) → Predictor+Sigmoid (float32)
        
        Args:
            device_type: 'npu' 或 'cuda'
            dtype: torch.bfloat16 或 torch.float16
        """
        self._amp_enabled = True
        self._amp_device_type = device_type
        self._amp_dtype = dtype
        logger.info(f"Model AMP configured: SAGEConv layers use {dtype}, others use float32")

    def precreate_edge_predictors_from_metadata(self, edge_type_dim_map: dict):
        """预创建边预测器"""
        for edge_key, edge_feat_dim in edge_type_dim_map.items():
            if edge_key in self.supported_edge_types and edge_key not in self.edge_predictor:
                self._create_edge_predictor(edge_key, edge_feat_dim)

    def precreate_all_edge_predictors_from_data(self, data):
        """从数据中自动检测并预创建所有支持的边预测器"""
        created_count = 0
        for edge_type in data.edge_types:
            edge_key = self._get_edge_predictor_key(edge_type)
            if edge_key and edge_key in self.supported_edge_types:
                if hasattr(data[edge_type], 'x') and hasattr(data[edge_type], 'y'):
                    if edge_key not in self.edge_predictor:
                        edge_feature_dim = data[edge_type].x.shape[1]
                        self._create_edge_predictor(edge_key, edge_feature_dim)
                        created_count += 1
        return created_count

    def _get_edge_predictor_key(self, edge_type):
        """将边类型tuple转换为字符串键名"""
        if edge_type == ('inventory', 'to', 'virtual'):
            return 'inventory_to_virtual'
        elif edge_type == ('inventory', 'to_transport_labeled', 'inventory'):
            return 'inventory_to_transport_labeled'
        elif edge_type == ('inventory', 'to_transport_reverse', 'inventory'):
            return 'inventory_to_transport_reverse'
        else:
            return None

    def _create_edge_predictor(self, edge_type_key: str, edge_feature_dim: int):
        """动态创建边预测器"""
        try:
            device = self._current_device()
            combined_input_dim = edge_feature_dim + 2 * self.node_feature_dim

            edge_layers = [
                nn.Linear(combined_input_dim, self.layer_config.hidden_dim),
                self.config.get_activation_function(self.layer_config.activation),
                nn.Dropout(self.layer_config.dropout_rate),
                nn.Linear(self.layer_config.hidden_dim, self.layer_config.hidden_dim // 2),
                self.config.get_activation_function(self.layer_config.activation),
                nn.Dropout(self.layer_config.dropout_rate),
                nn.Linear(self.layer_config.hidden_dim // 2, self.layer_config.hidden_dim // 4),
                self.config.get_activation_function(self.layer_config.activation),
                nn.Dropout(self.layer_config.dropout_rate),
                nn.Linear(self.layer_config.hidden_dim // 4, 1)
            ]
            enhanced_predictor = nn.Sequential(*edge_layers).to(device)

            simple_layers = [
                nn.Linear(edge_feature_dim, self.layer_config.hidden_dim // 2),
                self.config.get_activation_function(self.layer_config.activation),
                nn.Dropout(self.layer_config.dropout_rate),
                nn.Linear(self.layer_config.hidden_dim // 2, 1)
            ]
            simple_predictor = nn.Sequential(*simple_layers).to(device)

            self.edge_predictor[edge_type_key] = enhanced_predictor
            self.edge_predictor[f'{edge_type_key}_simple'] = simple_predictor

            self.dynamic_predictors[edge_type_key] = {
                'edge_feature_dim': edge_feature_dim,
                'combined_input_dim': combined_input_dim,
                'created': True
            }

        except Exception as e:
            logger.error(f"Failed to create edge predictor for {edge_type_key}: {e}")
            raise

    def forward_for_initialization(self, data):
        """Forward pass for LazyModule initialization"""
        try:
            device = self._current_device()
            if hasattr(data, 'to'):
                data = data.to(device)

            # 特征投影
            x_dict = {}
            for node_type in data.x_dict:
                if isinstance(node_type, tuple):
                    continue
                if not isinstance(node_type, str):
                    continue
                if node_type not in self.feature_projection:
                    continue
                x = data.x_dict[node_type]
                if x.device != device:
                    x = x.to(device)
                x_dict[node_type] = self.feature_projection[node_type](x)

            # 边索引设备同步
            edge_index_dict = {}
            for k, v in data.edge_index_dict.items():
                if v.device != device:
                    v = v.to(device)
                if not v.is_contiguous():
                    v = v.contiguous()
                edge_index_dict[k] = v

            # GNN消息传递
            for layer_idx, conv_layer in enumerate(self.conv_layers):
                x_residual = x_dict
                x_dict = conv_layer.forward_for_initialization(x_dict, edge_index_dict)
                for node_type in list(x_dict.keys()):
                    h = x_dict[node_type]
                    if node_type in self.conv_norms[layer_idx]:
                        h = self.conv_norms[layer_idx][node_type](h)
                    h = self.gnn_activation(h)
                    h = F.dropout(h, p=self.gnn_dropout_rate, training=self.training)
                    if node_type in x_residual and x_residual[node_type].shape == h.shape:
                        h = h + x_residual[node_type]
                    x_dict[node_type] = h
                for node_type in x_residual:
                    if node_type not in x_dict:
                        x_dict[node_type] = x_residual[node_type]

            # 节点预测器初始化
            out_dict = {}
            for node_type in self.target_node_types:
                if (node_type in data.node_types and
                        hasattr(data[node_type], 'y') and
                        node_type in x_dict and
                        node_type in self.dropout and
                        node_type in self.predictor):
                    try:
                        x_processed = self.dropout[node_type](x_dict[node_type])
                        out = self.predictor[node_type](x_processed)
                        out = out.squeeze(-1)
                        out_dict[node_type] = out
                    except Exception as e:
                        logger.warning(f"Error in initialization prediction for {node_type}: {e}")
                        if node_type in x_dict:
                            num_nodes = len(x_dict[node_type])
                            out_dict[node_type] = torch.zeros(num_nodes, device=device)

            # 边预测器初始化
            for edge_type in data.edge_types:
                edge_key = self._get_edge_predictor_key(edge_type)
                if edge_key and edge_key in self.supported_edge_types:
                    try:
                        edge_data = data[edge_type]
                        has_features = hasattr(edge_data, 'x') and edge_data.x is not None
                        has_labels = hasattr(edge_data, 'y') and edge_data.y is not None

                        if has_features and has_labels:
                            if edge_key not in self.edge_predictor:
                                edge_feature_dim = edge_data.x.shape[1]
                                self._create_edge_predictor(edge_key, edge_feature_dim)

                            edge_features = edge_data.x.to(device)
                            predictor_key = edge_key + '_simple'
                            if predictor_key in self.edge_predictor:
                                edge_out = self.edge_predictor[predictor_key](edge_features)
                                edge_out = edge_out.squeeze(-1)
                                edge_out = torch.sigmoid(edge_out)
                                out_dict[edge_type] = edge_out
                            elif edge_key in self.edge_predictor:
                                num_edges = edge_features.shape[0]
                                out_dict[edge_type] = torch.zeros(num_edges, device=device)
                    except Exception as e:
                        logger.warning(f"Error initializing edge predictor for {edge_type}: {e}")

            return out_dict

        except Exception as e:
            logger.error(f"Critical error in initialization forward pass: {e}")
            return {}

    def forward(self, data):
        """Forward pass with traditional GCN message passing"""
        try:
            device = self._current_device()
            if hasattr(data, 'to'):
                data = data.to(device)

            # 特征投影
            x_dict = {}
            for node_type in data.x_dict:
                if isinstance(node_type, tuple):
                    continue
                if not isinstance(node_type, str):
                    continue
                if node_type not in self.feature_projection:
                    continue
                x = data.x_dict[node_type]
                if x.device != device:
                    x = x.to(device)
                if not x.is_contiguous():
                    x = x.contiguous()
                x_dict[node_type] = self.feature_projection[node_type](x)

            # 边索引设备同步
            edge_index_dict = {}
            for k, v in data.edge_index_dict.items():
                if v.device != device:
                    v = v.to(device)
                if not v.is_contiguous():
                    v = v.contiguous()
                edge_index_dict[k] = v

            # GNN消息传递
            for layer_idx, conv_layer in enumerate(self.conv_layers):
                x_residual = x_dict
                try:
                    amp_enabled = getattr(self, '_amp_enabled', False)
                    amp_device = getattr(self, '_amp_device_type', None)
                    amp_dtype = getattr(self, '_amp_dtype', None)
                    
                    if amp_enabled and amp_device is not None and amp_dtype is not None:
                        # 安全检查：只在设备真正支持该dtype时才使用
                        try:
                            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
                                x_dict_new = conv_layer.forward(x_dict, edge_index_dict)
                            # 转换回 float32
                            x_dict_new = {k: v.float() if v.dtype != torch.float32 else v for k, v in x_dict_new.items()}
                        except Exception as amp_error:
                            # 如果 AMP 失败，回退到非 AMP 模式
                            logger.warning(f"AMP failed: {amp_error}, falling back to FP32")
                            x_dict_new = conv_layer.forward(x_dict, edge_index_dict)
                    else:
                        x_dict_new = conv_layer.forward(x_dict, edge_index_dict)

                    for node_type in list(x_dict_new.keys()):
                        h = x_dict_new[node_type]
                        if node_type in self.conv_norms[layer_idx]:
                            h = self.conv_norms[layer_idx][node_type](h)
                        h = self.gnn_activation(h)
                        h = F.dropout(h, p=self.gnn_dropout_rate, training=self.training)
                        if node_type in x_residual and x_residual[node_type].shape == h.shape:
                            h = h + x_residual[node_type]
                        x_dict_new[node_type] = h

                    for node_type in x_residual:
                        if node_type not in x_dict_new:
                            x_dict_new[node_type] = x_residual[node_type]

                    x_dict = x_dict_new
                except Exception as e:
                    logger.warning(f"Error in conv layer {layer_idx}: {e}")

            # 节点预测
            out_dict = {}
            for node_type in self.target_node_types:
                if (node_type in data.node_types and
                        hasattr(data[node_type], 'y') and
                        node_type in x_dict):
                    labels = data[node_type].y
                    valid_labels = labels[labels != -1.0]
                    if len(valid_labels) == 0:
                        continue

                    try:
                        x_processed = self.dropout[node_type](x_dict[node_type])
                        out = self.predictor[node_type](x_processed)
                        out = out.squeeze(-1)
                        out = torch.sigmoid(out)
                        out_dict[node_type] = out
                    except Exception as e:
                        logger.warning(f"Error in prediction for {node_type}: {e}")
                        try:
                            num_nodes = len(x_dict[node_type])
                            out_dict[node_type] = torch.zeros(num_nodes, device=device)
                        except Exception:
                            continue

            # 根据 loss_mode 决定是否进行边预测
            if self.loss_mode == 'nodes_only':
                return out_dict

            # 边预测
            for edge_type in data.edge_types:
                edge_key = self._get_edge_predictor_key(edge_type)
                if edge_key and edge_key in self.supported_edge_types:
                    edge_out = self._predict_edge(edge_type, edge_key, data, x_dict, device)
                    if edge_out is not None:
                        out_dict[edge_type] = edge_out

            return out_dict

        except Exception as e:
            logger.error(f"Critical error in forward pass: {e}")
            return {}

    def _predict_edge(self, edge_type, edge_key, data, x_dict, device):
        """边预测辅助方法"""
        try:
            edge_data = data[edge_type]
            if not hasattr(edge_data, 'y') or not hasattr(edge_data, 'x'):
                return None

            if edge_key not in self.edge_predictor:
                return None

            if not hasattr(edge_data, 'edge_index') or edge_data.edge_index.shape[1] == 0:
                return None

            edge_features = edge_data.x.to(device)
            if not edge_features.is_contiguous():
                edge_features = edge_features.contiguous()

            num_edges = edge_data.edge_index.shape[1]
            if edge_features.shape[0] != num_edges:
                logger.error(f"Edge features count mismatch for {edge_type}")
                return None

            # 尝试使用组合特征
            predictor_to_use = edge_key
            features_to_use = edge_features

            if hasattr(edge_data, 'edge_index') and len(edge_type) == 3:
                src_type, _, dst_type = edge_type
                if src_type in x_dict and dst_type in x_dict:
                    edge_index = edge_data.edge_index
                    src_max_idx = x_dict[src_type].shape[0] - 1
                    dst_max_idx = x_dict[dst_type].shape[0] - 1

                    valid_src = edge_index[0] <= src_max_idx
                    valid_dst = edge_index[1] <= dst_max_idx
                    if valid_src.all() and valid_dst.all():
                        src_features = x_dict[src_type][edge_index[0]]
                        dst_features = x_dict[dst_type][edge_index[1]]
                        combined = torch.cat([edge_features, src_features, dst_features], dim=1)

                        if combined.shape[1] == self.edge_predictor[edge_key][0].in_features:
                            features_to_use = combined
                        elif edge_features.shape[1] == self.edge_predictor[edge_key + '_simple'][0].in_features:
                            predictor_to_use = edge_key + '_simple'

            # 确保预测器存在
            if predictor_to_use not in self.edge_predictor:
                predictor_to_use = edge_key

            edge_out = self.edge_predictor[predictor_to_use](features_to_use)
            edge_out = edge_out.squeeze(-1)
            edge_out = torch.sigmoid(edge_out)

            return edge_out

        except Exception as e:
            logger.warning(f"Error in edge prediction for {edge_type}: {e}")
            return None

    def compute_loss(self, predictions, data):
        """计算损失函数"""
        loss_mode = getattr(self.config.training_config, 'loss_mode', 'both')

        total_loss = 0
        count = 0
        node_loss_count = 0
        edge_loss_count = 0

        if loss_mode in ['both', 'nodes_only']:
            binary_node_types = self.target_node_types

            for node_type in predictions:
                if node_type in binary_node_types and hasattr(data[node_type], 'y'):
                    pred = predictions[node_type]
                    true = data[node_type].y.to(self._current_device())

                    mask = ~torch.isnan(true) & (true != -1.0)

                    if mask.any():
                        pred = pred[mask]
                        true = true[mask]

                        true = (true != 0).float()
                        pred = pred.float()
                        true = true.float()

                        pred_clamped = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
                        loss = F.binary_cross_entropy(pred_clamped, true, reduction='mean')

                        total_loss += loss
                        count += 1
                        node_loss_count += 1

        if loss_mode in ['both', 'edges_only']:
            edge_types_with_labels = [
                ('inventory', 'to', 'virtual'),
                ('inventory', 'to_transport_labeled', 'inventory')
            ]
            for edge_type in edge_types_with_labels:
                if edge_type in predictions and hasattr(data[edge_type], 'y'):
                    edge_pred = predictions[edge_type]
                    edge_true = data[edge_type].y.to(self._current_device())

                    if edge_pred.shape[0] != edge_true.shape[0]:
                        min_len = min(edge_pred.shape[0], edge_true.shape[0])
                        edge_pred = edge_pred[:min_len]
                        edge_true = edge_true[:min_len]

                    edge_mask = ~torch.isnan(edge_true)

                    if edge_mask.any():
                        edge_pred = edge_pred[edge_mask]
                        edge_true = edge_true[edge_mask]

                        edge_true = edge_true.float()
                        edge_pred = edge_pred.float()
                        edge_true = edge_true.float()

                        edge_pred_clamped = torch.clamp(edge_pred, min=1e-7, max=1 - 1e-7)
                        edge_loss = F.binary_cross_entropy(edge_pred_clamped, edge_true, reduction='mean')

                        total_loss += edge_loss
                        count += 1
                        edge_loss_count += 1

        if count > 0:
            main_loss = total_loss / count

            l2_regularization = 0.0
            param_count = 0
            for param in self.parameters():
                if param.requires_grad:
                    l2_regularization += torch.norm(param, 2) ** 2
                    param_count += param.numel()

            if param_count > 0:
                normalized_l2_reg = l2_regularization / param_count
                reg_term = self.l2_lambda * normalized_l2_reg
            else:
                reg_term = 0.0

            regularized_loss = main_loss + reg_term

            if not hasattr(self, '_loss_log_counter'):
                self._loss_log_counter = 0
            self._loss_log_counter += 1

            if self._loss_log_counter % 100 == 0:
                logger.info(
                    f"Loss breakdown [Mode: {loss_mode}] - Main: {main_loss:.6f}, L2 Reg: {reg_term:.6f}, Total: {regularized_loss:.6f}")
                logger.info(
                    f"Node losses: {node_loss_count}, Edge losses: {edge_loss_count}, Total params: {param_count}")

            return regularized_loss
        else:
            logger.warning(f"No valid losses found for loss computation in mode: {loss_mode}")
            return torch.tensor(0.001, device=self._current_device())

    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 6))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        # 绘制指标曲线
        plt.subplot(1, 2, 2)
        for metric in ['mse', 'mae', 'rmse']:
            if metric in self.training_history['train_metrics']:
                plt.plot(self.training_history['train_metrics'][metric],
                         label=f'Train {metric.upper()}')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Training Metrics')
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def save_model(self, path, optimizer=None, epoch=None, loss=None):
        """保存模型状态"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'training_history': self.training_history
        }
        if optimizer:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            save_dict['epoch'] = epoch
        if loss is not None:
            save_dict['loss'] = loss
        torch.save(save_dict, path)

    def load_model(self, path, optimizer=None):
        """加载模型状态"""
        device = self._current_device()
        map_location_str = str(device)
        checkpoint = torch.load(path, map_location=map_location_str)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'train_metrics': defaultdict(list)
        })
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', None), checkpoint.get('loss', None)

    def get_device(self):
        """获取模型当前所在的设备（动态）"""
        return self._current_device()

    def _current_device(self):
        """动态获取当前参数所在设备，避免DDP下缓存设备错误"""
        try:
            return next(self.parameters()).device
        except StopIteration:
            # 模型还没有参数，回退到初始化设备
            return getattr(self, '_init_device', torch.device('cpu'))

    @property
    def device(self):
        """兼容旧代码的设备属性访问，动态返回当前设备"""
        return self._current_device()
