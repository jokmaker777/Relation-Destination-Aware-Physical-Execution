import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging
from collections import defaultdict
import os
import math
from datetime import datetime
from pathlib import Path

from ..config.path_config import path_config

logger = logging.getLogger(__name__)

# 🔧 版本标识：确认修复代码已部署
VERSION_TAG = "NODE_DIMENSION_FIX_V2.0_20250929"


class PredictionHandler:
    """处理GCN预测结果的可视化和输出，支持基于日期的命名"""

    def __init__(self, save_dir: str = None, rank: int = 0, world_size: int = 1):
        # 🔧 版本确认日志
        logger.info(f"🔧 PredictionHandler 初始化，版本: {VERSION_TAG}")
        logger.info(f"🔧 分布式参数: rank={rank}, world_size={world_size}")

        if save_dir is None:
            # 不再默认用全局result_dir，直接raise或警告
            raise ValueError("PredictionHandler必须指定experiment的save_dir")
        self.save_dir = Path(save_dir)

        # 分布式训练参数
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)

        # 🔧 方案Q修复1: 添加日志计数器
        self.transport_edge_format_errors = 0
        self.max_transport_error_logs = 10  # 最多显示10条详细日志

        # 🔧 多卡汇总修复：只有主进程创建目录和文件，其他进程不创建
        if self.is_main_process:
            self.save_dir.mkdir(parents=True, exist_ok=True)

            # 在experiment目录下创建visualizations和csv_outputs文件夹
            self.visualization_dir = self.save_dir / "visualizations"
            self.csv_output_dir = self.save_dir / "csv_outputs"

            # 创建必要的目录
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
            self.csv_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"✅ 主进程(rank 0)创建输出目录: {self.save_dir}")
        else:
            # 非主进程也需要路径信息，但不创建目录
            self.visualization_dir = self.save_dir / "visualizations"
            self.csv_output_dir = self.save_dir / "csv_outputs"
            logger.info(f"⚠️ 非主进程(rank {rank})不创建输出目录，仅记录路径")

        # 存储所有预测结果
        self.all_predictions = {}  # {dataset_type_date: {node_type: data}}
        self.date_mapping = {}  # 图ID到日期的映射

    def collect_predictions(self, predictions: Dict[str, torch.Tensor],
                            data, graph_date: str, dataset_type: str, node_manager=None, edge_manager=None,
                            graph_builder=None, max_items_per_node_type: Optional[int] = None,
                            max_edges_per_type: Optional[int] = None):
        """收集单个图的预测结果，支持基于日期的键名，包含item_code映射，使用日期特定映射
        
        🔧 多卡汇总修复：只有主进程收集预测结果，其他进程直接返回
        """
        # 🔧 多卡汇总修复：非主进程直接返回，不收集预测结果
        if not self.is_main_process:
            logger.debug(f"⚠️ 非主进程(rank {self.rank})跳过预测结果收集")
            return
        
        # 创建基于日期和数据集类型的键
        graph_key = f"{dataset_type}_{graph_date}"

        logger.info(f"✅ 主进程(rank 0)开始收集预测结果: {graph_key}")

        self.all_predictions[graph_key] = {}
        self.date_mapping[graph_key] = graph_date

        # 优化的映射加载
        date_specific_mappings = self._load_date_specific_mappings(graph_builder, graph_date)

        # 增强日志记录
        if not date_specific_mappings:
            logger.warning(f"❌ 未加载到日期映射 {graph_key}")
        else:
            logger.info(f"✅ 映射加载完成 {graph_key}: {list(date_specific_mappings.keys())}")
            # 添加每种节点类型的映射条目数量信息
            for node_type_key, mapping in date_specific_mappings.items():
                logger.info(f"节点类型 {node_type_key} 的映射条目数: {len(mapping)}")
                # 记录前几个映射项作为样本
                sample_items = list(mapping.items())[:3]
                logger.info(f"映射样本: {sample_items}")

        for node_type in predictions:
            # 跳过边类型，边类型在后面单独处理
            if isinstance(node_type, tuple):
                continue
            if hasattr(data[node_type], 'y'):
                # 移动到CPU并转换为numpy
                pred = predictions[node_type].detach().cpu().numpy()
                true = data[node_type].y.detach().cpu().numpy()

                # 获取层级信息
                layers = None
                if hasattr(data[node_type], 'layer'):
                    layers = data[node_type].layer.detach().cpu().numpy()

                # 快速映射获取
                num_nodes = len(pred)
                logger.info(f"处理节点类型 {node_type}, 节点数量: {num_nodes}")

                # 🔧 增强映射验证和诊断
                diagnosis = self._diagnose_mapping_quality(node_type, date_specific_mappings)

                # 🔧 修复后的处理策略
                if diagnosis['has_existing_multidim']:
                    logger.info(f"✅ {node_type} 映射中已包含多维度ID，直接使用")
                    # 🔧 方案F: 直接使用现有的多维度映射
                    item_codes = self._get_multidim_item_codes_from_mapping(node_type, num_nodes,
                                                                            date_specific_mappings)
                    if item_codes:
                        logger.info(f"✅ 从现有多维度映射中获取了 {len(item_codes)} 个{node_type}节点ID")
                        continue_with_original_ids = True
                    else:
                        logger.warning(f"⚠️ 多维度映射获取失败，回退到优化获取逻辑")
                        continue_with_original_ids = False
                elif diagnosis['needs_original_id_lookup']:
                    logger.warning(f"⚠️ {node_type} 映射中主要是单维度键，需要查找原始多维度ID")
                    # 🔧 方案D: 实现从图数据中查找原始多维度ID的逻辑
                    item_codes = self._get_item_codes_with_original_lookup(node_type, num_nodes, date_specific_mappings,
                                                                           data, node_manager)
                    if item_codes:
                        logger.info(f"✅ 成功从图数据中获取 {len(item_codes)} 个原始{node_type}节点ID")
                        continue_with_original_ids = True
                    else:
                        logger.warning(f"⚠️ 原始ID查找失败，回退到优化获取逻辑")
                        continue_with_original_ids = False
                else:
                    continue_with_original_ids = False

                if not continue_with_original_ids:
                    if not diagnosis['has_mapping']:
                        logger.warning(f"⚠️ {node_type} 映射不存在，将使用回退逻辑")
                    elif diagnosis['mapping_size'] == 0:
                        logger.warning(f"⚠️ {node_type} 映射为空，可能导致回退ID")

                    item_codes = self._get_optimized_item_codes(node_type, num_nodes, date_specific_mappings, data,
                                                                node_manager)

                # 🔧 验证item_codes质量
                if item_codes:
                    missing_count = sum(1 for code in item_codes if code and 'MISSING_DATE_MAP' in str(code))
                    fallback_count = sum(
                        1 for code in item_codes if code and ('fallback_' in str(code) or 'EMPTY_' in str(code)))
                    valid_count = len(item_codes) - missing_count - fallback_count

                    logger.info(f"🔍 {node_type} item_codes质量分析:")
                    logger.info(
                        f"   - 有效映射: {valid_count}/{len(item_codes)} ({valid_count / len(item_codes) * 100:.1f}%)")
                    logger.info(
                        f"   - 缺失映射: {missing_count}/{len(item_codes)} ({missing_count / len(item_codes) * 100:.1f}%)")
                    logger.info(
                        f"   - 回退ID: {fallback_count}/{len(item_codes)} ({fallback_count / len(item_codes) * 100:.1f}%)")

                    if missing_count > len(item_codes) * 0.5:
                        logger.warning(f"⚠️ {node_type} 超过50%的节点使用缺失映射，数据收集可能有问题")
                else:
                    logger.error(f"❌ {node_type} item_codes为None，数据收集完全失败")

                # 采样以限制输出规模（可选）
                if max_items_per_node_type is not None and max_items_per_node_type > 0:
                    limit_n = min(max_items_per_node_type, len(pred))
                    if limit_n < len(pred):
                        logger.info(f"Sampling node predictions for {graph_key} {node_type}: {limit_n}/{len(pred)}")
                        pred = pred[:limit_n]
                        true = true[:limit_n]
                        layers = layers[:limit_n] if layers is not None else None
                        item_codes = item_codes[:limit_n] if item_codes is not None else None

                # 存储预测结果
                self.all_predictions[graph_key][node_type] = {
                    'pred': pred,
                    'true': true,
                    'item_codes': item_codes
                }

        edge_types_with_labels = [
            ('inventory', 'to', 'virtual'),
            ('inventory', 'to_transport_labeled', 'inventory')
        ]
        
        for edge_type in edge_types_with_labels:
            if edge_type in predictions and hasattr(data[edge_type], 'y'):
                self._process_edge_predictions(data, predictions, graph_key, edge_type, 
                                              date_specific_mappings, max_edges_per_type)

        logger.info("跳过reverse边的预测收集（反向边仅用于GNN消息传递）")

    def _process_edge_predictions(self, data, predictions, graph_key, edge_type,
                                date_specific_mappings, max_edges_per_type):
        """处理边的预测结果"""
        edge_pred = predictions[edge_type].detach().cpu().numpy()
        edge_true = data[edge_type].y.detach().cpu().numpy()

        if len(edge_pred) != len(edge_true):
            logger.warning(f"边预测与真实值数量不一致: {edge_type}")
            if len(edge_pred) < len(edge_true):
                edge_pred = np.concatenate([edge_pred, np.zeros(len(edge_true) - len(edge_pred), dtype=edge_pred.dtype)])
            else:
                edge_pred = edge_pred[:len(edge_true)]

        if max_edges_per_type and max_edges_per_type > 0:
            limit_e = min(max_edges_per_type, len(edge_pred))
            if limit_e < len(edge_pred):
                edge_pred = edge_pred[:limit_e]
                edge_true = edge_true[:limit_e]

        edge_identifiers = self._build_edge_identifiers(
            data, edge_type, len(edge_pred), date_specific_mappings)

        self.all_predictions[graph_key][edge_type] = {
            'pred': edge_pred.flatten(),
            'true': edge_true.flatten(),
            'edge_identifiers': edge_identifiers
        }

    def _build_edge_identifiers(self, data, edge_type, num_edges, date_specific_mappings):
        """构建边标识符"""
        if hasattr(self, 'graph_builder') and hasattr(self.graph_builder, 'edge_manager'):
            stored = self.graph_builder.edge_manager.edges.get(edge_type, {}).get('edge_identifiers', [])
            if stored and len(stored) >= num_edges:
                return stored[:num_edges]

        if edge_type == ('inventory', 'to', 'virtual'):
            return self._build_virtual_edge_identifiers(data, num_edges, date_specific_mappings)
        elif edge_type == ('inventory', 'to_transport_labeled', 'inventory'):
            return self._build_transport_edge_identifiers(data, num_edges, date_specific_mappings)
        return [f"edge_{i}" for i in range(num_edges)]

    def _build_virtual_edge_identifiers(self, data, num_edges, date_specific_mappings):
        """构建虚拟边标识符"""
        if 'inventory' not in date_specific_mappings or 'virtual' not in date_specific_mappings:
            return [f"MISSING_MAPPING_{i}" for i in range(num_edges)]

        inv_map = {idx: code for code, idx in date_specific_mappings['inventory'].items()}
        vir_map = {idx: code for code, idx in date_specific_mappings['virtual'].items()}
        edge_index = data[('inventory', 'to', 'virtual')].edge_index.detach().cpu().numpy()

        edge_ids = []
        for i in range(num_edges):
            if i < edge_index.shape[1]:
                inv_idx, vir_idx = edge_index[0, i], edge_index[1, i]
                inv_code = inv_map.get(inv_idx, f"inv_{inv_idx}")
                vir_code = vir_map.get(vir_idx, f"vir_{vir_idx}")

                if isinstance(vir_code, tuple) and len(vir_code) >= 2:
                    edge_id = f"{vir_code[0]}/{vir_code[1]}/{inv_code}"
                elif isinstance(vir_code, str) and '|' in vir_code:
                    parts = vir_code.split('|')
                    edge_id = f"{parts[0]}/{parts[1]}/{inv_code}" if len(parts) >= 2 else f"unknown/unknown/{inv_code}"
                else:
                    edge_id = f"unknown/unknown/{inv_code}"
                edge_ids.append(edge_id)
            else:
                edge_ids.append(f"edge_error_{i}")
        return edge_ids

    def _build_transport_edge_identifiers(self, data, num_edges, date_specific_mappings):
        """构建运输边标识符"""
        if 'inventory' not in date_specific_mappings or not hasattr(data[('inventory', 'to_transport_labeled', 'inventory')], 'edge_index'):
            return [f"transport_edge_{i}" for i in range(num_edges)]

        inv_map = {idx: code for code, idx in date_specific_mappings['inventory'].items()}
        edge_index = data[('inventory', 'to_transport_labeled', 'inventory')].edge_index.detach().cpu().numpy()

        edge_ids = []
        for i in range(num_edges):
            if i < edge_index.shape[1]:
                src_idx, dst_idx = edge_index[0, i], edge_index[1, i]
                src_code = inv_map.get(src_idx, f"inv_{src_idx}")
                dst_code = inv_map.get(dst_idx, f"inv_{dst_idx}")

                src_parts = src_code.split('|') if '|' in src_code else []
                dst_parts = dst_code.split('|') if '|' in dst_code else []

                if len(src_parts) >= 3 and len(dst_parts) >= 3:
                    try:
                        lead_time = int(dst_parts[1]) - int(src_parts[1])
                    except (ValueError, TypeError):
                        lead_time = 'unknown'
                    edge_id = f"{src_parts[2]}/{dst_parts[2]}/{src_parts[0]}/{src_parts[1]}/{dst_parts[1]}/{lead_time}"
                else:
                    edge_id = f"parse_failed_{i}"
                edge_ids.append(edge_id)
            else:
                edge_ids.append(f"transport_overflow_{i}")
        return edge_ids

    def _diagnose_node_id_format(self, graph_key: str, predictions: Dict):
        """诊断节点ID的实际格式"""
        logger.info(f"🔍 诊断 {graph_key} 的节点ID格式...")

        for node_type, data in predictions.items():
            if isinstance(node_type, str):  # 只处理节点类型
                # 🔧 检查多个可能的ID来源
                node_indices = data.get('node_indices', [])
                item_codes = data.get('item_codes', [])

                logger.info(f"🔍 {node_type} 数据源:")
                logger.info(
                    f"   - node_indices: {len(node_indices)} 个, 样例: {node_indices[:5] if node_indices else '无'}")
                logger.info(f"   - item_codes: {len(item_codes)} 个, 样例: {item_codes[:5] if item_codes else '无'}")

                # 选择主要的ID源进行分析
                primary_ids = node_indices if node_indices else item_codes

                if len(primary_ids) > 0:
                    # 分析前10个节点ID的格式
                    samples = primary_ids[:10]
                    logger.info(f"🔍 {node_type} 主要ID样例: {samples}")

                    # 检测格式模式
                    patterns = {
                        'pipe_separated': sum(1 for x in samples if '|' in str(x)),
                        'underscore_separated': sum(1 for x in samples if '_' in str(x)),
                        'dash_separated': sum(1 for x in samples if '-' in str(x)),
                        'single_number': sum(1 for x in samples if str(x).replace('.', '').isdigit()),
                        'float_number': sum(1 for x in samples if self._is_float(str(x))),
                        'complex_string': sum(1 for x in samples if len(str(x)) > 10),
                        'total_samples': len(samples)
                    }
                    logger.info(f"🔍 {node_type} 格式分析: {patterns}")

                    # 🔧 推荐处理策略
                    if patterns['single_number'] == patterns['total_samples']:
                        logger.info(f"💡 {node_type} 推荐策略: 数字ID，使用智能构造")
                    elif patterns['pipe_separated'] > 0:
                        logger.info(f"💡 {node_type} 推荐策略: 管道分隔符解析")
                    elif patterns['complex_string'] > 0:
                        logger.info(f"💡 {node_type} 推荐策略: 复杂字符串，尝试多种分隔符")
                    else:
                        logger.info(f"💡 {node_type} 推荐策略: 使用默认维度")
                else:
                    logger.warning(f"⚠️ {node_type} 没有可用的ID数据")

    def _is_float(self, s: str) -> bool:
        """检查字符串是否为浮点数"""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _parse_node_dimensions(self, node_id: str, node_type: str) -> dict:
        """
        智能解析节点ID获取各个维度信息

        Args:
            node_id: 节点ID字符串
            node_type: 节点类型

        Returns:
            dict: 包含各个维度的字典
        """
        # 🔧 添加调试信息
        logger.debug(f"🔍 解析节点维度: node_id={node_id}, node_type={node_type}")

        dimensions = {}
        node_id_str = str(node_id)

        # 🔧 特殊处理：检查是否为回退ID格式
        if self._is_fallback_id(node_id_str):
            logger.debug(f"🔍 检测到回退ID格式: {node_id_str}")
            return self._parse_fallback_id(node_id_str, node_type)

        # 🔧 特殊处理：检查是否为MISSING_DATE_MAP格式
        if node_id_str.startswith('MISSING_DATE_MAP_'):
            logger.debug(f"🔍 检测到MISSING_DATE_MAP格式: {node_id_str}")
            return self._parse_missing_date_map_id(node_id_str, node_type)

        # 🔧 智能检测分隔符类型
        if '|' in node_id_str:
            parts = node_id_str.split('|')
            logger.debug(f"🔍 使用管道分隔符解析: {parts}")
        elif '_' in node_id_str:
            parts = node_id_str.split('_')
            logger.debug(f"🔍 使用下划线分隔符解析: {parts}")
        elif '-' in node_id_str:
            parts = node_id_str.split('-')
            logger.debug(f"🔍 使用连字符分隔符解析: {parts}")
        else:
            # 单一ID格式，尝试从图数据查找
            logger.debug(f"🔍 检测到单一ID格式: {node_id_str}")
            return self._lookup_node_dimensions_from_graph(node_type, node_id_str)

        parts = [part.strip() for part in parts]  # 清理空白字符

        try:
            if node_type == 'production':
                if len(parts) >= 3:
                    dimensions = {
                        'item_code': parts[0],
                        'period_code': parts[1],
                        'plant_code': parts[2]
                    }
            elif node_type == 'inventory':
                if len(parts) >= 3:
                    dimensions = {
                        'item_code': parts[0],
                        'period_code': parts[1],
                        'plant_code': parts[2]
                    }
            elif node_type == 'virtual':
                if len(parts) >= 4:
                    dimensions = {
                        'assembly': parts[0],
                        'component': parts[1],
                        'period_code': parts[2],
                        'plant_code': parts[3]
                    }
            elif node_type == 'capacity':
                if len(parts) >= 4:
                    dimensions = {
                        'set_code': parts[0],
                        'period_code': parts[1],
                        'plant_code': parts[2],
                        'shift_code': parts[3]
                    }
            elif node_type == 'capacitygroup':
                if len(parts) >= 3:
                    dimensions = {
                        'set_group': parts[0],
                        'period_code': parts[1],
                        'plant_code': parts[2]
                    }
            elif node_type == 'sr':
                if len(parts) >= 4:
                    dimensions = {
                        'item_code': parts[0],
                        'plant_code': parts[1],
                        'start_period_code': parts[2],
                        'end_period_code': parts[3]
                    }
            elif node_type == 'topproduction':
                if len(parts) >= 2:
                    dimensions = {
                        'item_code': parts[0],
                        'period_code': parts[1]
                    }
        except (ValueError, IndexError) as e:
            logger.warning(f"解析节点维度失败: node_id={node_id}, node_type={node_type}, error={e}")
            return self._lookup_node_dimensions_from_graph(node_type, node_id_str)

        # 如果解析失败，尝试从图数据查找
        if not dimensions:
            logger.debug(f"⚠️ 标准解析失败，尝试图数据查找: node_id={node_id}, node_type={node_type}")
            return self._lookup_node_dimensions_from_graph(node_type, node_id_str)
        else:
            logger.debug(f"✅ 节点维度解析成功: node_id={node_id} -> {dimensions}")

        return dimensions

    def _lookup_node_dimensions_from_graph(self, node_type: str, node_id: str) -> Dict[str, str]:
        """从图数据中查找节点的维度信息"""
        try:
            # 🔧 方法1: 尝试从当前图数据查找节点映射
            if hasattr(self, 'current_graph_data') and self.current_graph_data:
                node_mapping = self.current_graph_data.get('node_mapping', {})
                if node_type in node_mapping:
                    # 查找节点索引对应的原始数据
                    for original_data, node_idx in node_mapping[node_type].items():
                        if str(node_idx) == str(node_id):
                            logger.debug(f"🔍 从图数据找到节点映射: {original_data}")
                            return self._parse_original_node_data(node_type, original_data)

            # 🔧 方法2: 尝试从全局节点映射查找（如果存在）
            if hasattr(self, 'global_node_mapping') and self.global_node_mapping:
                if node_type in self.global_node_mapping:
                    mapping = self.global_node_mapping[node_type]
                    if str(node_id) in mapping:
                        original_id = mapping[str(node_id)]
                        logger.debug(f"🔍 从全局映射找到原始ID: {original_id}")
                        return self._parse_original_node_data(node_type, original_id)

            # 🔧 方法3: 尝试从节点管理器查找
            if hasattr(self, 'node_manager') and self.node_manager:
                dimensions = self.node_manager.get_node_dimensions(node_type, node_id)
                if dimensions:
                    logger.debug(f"🔍 从节点管理器找到维度: {dimensions}")
                    return dimensions

            # 🔧 方法4: 如果是数字ID，尝试构造合理的维度信息
            if str(node_id).replace('.', '').isdigit():
                logger.debug(f"🔍 检测到数字ID，使用智能构造")
                return self._construct_dimensions_for_numeric_id(node_type, node_id)

            logger.debug(f"⚠️ 所有查找方法失败，使用默认维度")
            return self._get_default_dimensions(node_type)

        except Exception as e:
            logger.warning(f"图数据维度查找失败 {node_type}:{node_id}: {e}")
            return self._get_default_dimensions(node_type)

    def _parse_original_node_data(self, node_type: str, original_data: str) -> Dict[str, str]:
        """解析原始节点数据获取维度信息"""
        try:
            # 尝试不同的分隔符
            for separator in ['|', '_', '-']:
                if separator in original_data:
                    parts = original_data.split(separator)
                    return self._extract_dimensions_from_parts(node_type, parts)

            # 如果没有分隔符，返回默认维度
            return self._get_default_dimensions(node_type)

        except Exception as e:
            logger.debug(f"原始数据解析失败: {e}")
            return self._get_default_dimensions(node_type)

    def _construct_dimensions_for_numeric_id(self, node_type: str, node_id: str) -> Dict[str, str]:
        """为数字ID构造合理的维度信息"""
        # 🔧 改进：对于数字ID，我们尝试提供更有意义的维度信息
        base_dimensions = self._get_default_dimensions(node_type)

        # 🔧 策略：将数字ID作为节点索引，尝试构造合理的维度值
        node_idx = str(node_id)

        if node_type in ['production', 'inventory']:
            # 对于生产和库存节点，使用节点索引作为item_code
            base_dimensions['item_code'] = f"item_{node_idx}"
            base_dimensions['period_code'] = "period_unknown"
            base_dimensions['plant_code'] = "plant_unknown"
        # 移除 forecast_delivery 和 order_delivery 节点的处理
        elif node_type == 'virtual':
            # 对于虚拟节点
            base_dimensions['assembly'] = f"assembly_{node_idx}"
            base_dimensions['component'] = "component_unknown"
            base_dimensions['period_code'] = "period_unknown"
            base_dimensions['plant_code'] = "plant_unknown"

        logger.debug(f"🔧 为数字ID构造维度: {node_id} -> {base_dimensions}")
        return base_dimensions

    def _extract_dimensions_from_parts(self, node_type: str, parts: List[str]) -> Dict[str, str]:
        """从分割后的部分提取维度信息"""
        dimensions = {}
        parts = [part.strip() for part in parts]

        try:
            if node_type == 'production':
                if len(parts) >= 3:
                    dimensions = {
                        'item_code': parts[0],
                        'period_code': parts[1],
                        'plant_code': parts[2]
                    }
            # 移除 forecast_delivery 和 order_delivery 节点的处理
            elif node_type == 'inventory':
                if len(parts) >= 3:
                    dimensions = {
                        'item_code': parts[0],
                        'period_code': parts[1],
                        'plant_code': parts[2]
                    }
            elif node_type == 'virtual':
                if len(parts) >= 4:
                    dimensions = {
                        'assembly': parts[0],
                        'component': parts[1],
                        'period_code': parts[2],
                        'plant_code': parts[3]
                    }
        except Exception as e:
            logger.debug(f"维度提取失败: {e}")

        return dimensions if dimensions else self._get_default_dimensions(node_type)

    def _get_default_dimensions(self, node_type: str) -> Dict[str, str]:
        """获取节点类型的默认维度结构"""
        defaults = {
            'production': {'item_code': 'unknown', 'period_code': 'unknown', 'plant_code': 'unknown'},
            'inventory': {'item_code': 'unknown', 'period_code': 'unknown', 'plant_code': 'unknown'},
            # 移除 forecast_delivery 和 order_delivery 节点的默认维度
            'virtual': {'assembly': 'unknown', 'component': 'unknown', 'period_code': 'unknown',
                        'plant_code': 'unknown'},
            'capacity': {'set_code': 'unknown', 'period_code': 'unknown', 'plant_code': 'unknown',
                         'shift_code': 'unknown'},
            'capacitygroup': {'set_group': 'unknown', 'period_code': 'unknown', 'plant_code': 'unknown'},
            'sr': {'item_code': 'unknown', 'plant_code': 'unknown', 'start_period_code': 'unknown',
                   'end_period_code': 'unknown'},
            'topproduction': {'item_code': 'unknown', 'period_code': 'unknown'}
        }
        return defaults.get(node_type, {'id': 'unknown'})

    def _diagnose_mapping_quality(self, node_type: str, date_specific_mappings: dict) -> dict:
        """诊断映射质量，分析键的格式和内容"""
        diagnosis = {
            'has_mapping': False,
            'mapping_size': 0,
            'multi_dim_count': 0,
            'single_dim_count': 0,
            'sample_keys': [],
            'needs_original_id_lookup': False,
            'has_existing_multidim': False  # 🔧 修复：确保此键始终存在
        }

        if node_type not in date_specific_mappings:
            logger.warning(f"⚠️ {node_type} 映射不存在")
            return diagnosis

        mapping = date_specific_mappings[node_type]
        diagnosis['has_mapping'] = True
        diagnosis['mapping_size'] = len(mapping)

        if diagnosis['mapping_size'] == 0:
            logger.warning(f"⚠️ {node_type} 映射为空")
            return diagnosis

        # 分析前10个键的格式
        sample_keys = list(mapping.keys())[:10]
        diagnosis['sample_keys'] = sample_keys

        # 🔧 方案G修复2: 增强多维度键检测，支持tuple格式
        for key in sample_keys:
            key_str = str(key)

            # 检查是否为tuple格式的多维度键
            if isinstance(key, tuple) and len(key) >= 2:
                diagnosis['multi_dim_count'] += 1
                logger.debug(f"🔍 检测到tuple多维度键: {key}")
            elif '|' in key_str:
                diagnosis['multi_dim_count'] += 1
                logger.debug(f"🔍 检测到字符串多维度键: {key_str}")
            else:
                diagnosis['single_dim_count'] += 1
                logger.debug(f"🔍 检测到单维度键: {key_str}")

        # 🔧 修复判断逻辑：只有当映射中全是单维度键时才需要查找原始ID
        diagnosis['needs_original_id_lookup'] = diagnosis['single_dim_count'] > 0 and diagnosis['multi_dim_count'] == 0
        diagnosis['has_existing_multidim'] = diagnosis['multi_dim_count'] > 0

        # 🔧 方案G修复5: 增强诊断阶段详细日志
        logger.info(f"🔍 {node_type} 映射格式分析:")
        logger.info(f"   - 映射大小: {diagnosis['mapping_size']}")
        logger.info(f"   - 多维度键: {diagnosis['multi_dim_count']}/{len(sample_keys)}")
        logger.info(f"   - 单维度键: {diagnosis['single_dim_count']}/{len(sample_keys)}")
        logger.info(f"   - 样例键原始格式: {sample_keys}")
        logger.info(f"   - 样例键字符串化: {[str(k) for k in sample_keys]}")
        logger.info(
            f"   - 多维度检测结果: {['|' in str(k) or (isinstance(k, tuple) and len(k) >= 2) for k in sample_keys]}")
        logger.info(f"   - 需要查找原始ID: {diagnosis['needs_original_id_lookup']}")
        logger.info(f"   - 🔧 有现有多维度ID: {diagnosis['has_existing_multidim']}")
        logger.info(
            f"   - 最终判断逻辑: multi_dim_count({diagnosis['multi_dim_count']}) > 0 = {diagnosis['has_existing_multidim']}")

        return diagnosis

    def _extract_clean_item_code(self, item_code_raw: str) -> str:
        """提取干净的数字item_code，用户希望只要纯数字"""
        if not item_code_raw:
            return 'unknown'

        item_code_str = str(item_code_raw)

        # 如果是多维度格式，取第一个维度
        if '|' in item_code_str:
            first_part = item_code_str.split('|')[0]
            logger.debug(f"🔍 从多维度ID提取item_code: {item_code_str} → {first_part}")
            return first_part

        # 如果是回退格式，提取数字部分
        if 'fallback_item_' in item_code_str:
            clean_code = item_code_str.replace('fallback_item_', '')
            logger.debug(f"🔍 从回退ID提取item_code: {item_code_str} → {clean_code}")
            return clean_code

        # 如果是missing_map格式，提取数字部分
        if 'missing_map_item_' in item_code_str:
            clean_code = item_code_str.replace('missing_map_item_', '')
            logger.debug(f"🔍 从missing_map ID提取item_code: {item_code_str} → {clean_code}")
            return clean_code

        # 🔧 修复1: 如果是新的fallback_multidim格式，提取数字部分
        if 'fallback_multidim_' in item_code_str:
            # 提取最后的数字部分 fallback_multidim_production_123 → 123
            parts = item_code_str.split('_')
            if len(parts) >= 3 and parts[-1].isdigit():
                clean_code = parts[-1]
                logger.debug(f"🔍 从fallback_multidim ID提取item_code: {item_code_str} → {clean_code}")
                return clean_code

        # 如果包含其他前缀，尝试提取数字部分
        import re
        numbers = re.findall(r'\d+', item_code_str)
        if numbers:
            clean_code = numbers[0]  # 取第一个数字
            logger.debug(f"🔍 从复杂ID提取数字: {item_code_str} → {clean_code}")
            return clean_code

        # 如果已经是纯数字或简单字符串，直接返回
        return item_code_str

    def _clean_unknown_value(self, value: str) -> str:
        """🔧 阶段1修复: 清理unknown值，统一处理格式"""
        if not value:
            return 'unknown'

        value_str = str(value).strip()

        # 处理各种unknown格式
        if (value_str == 'unknown_assembly' or
                value_str == 'unknown_component' or
                value_str.startswith('unknown_')):
            return 'unknown'

        return value_str

    def _parse_inventory_id_dimensions_enhanced(self, inventory_id: str) -> Dict[str, str]:
        """🔧 阶段1修复: 增强inventory_id解析的容错性，支持多种格式和边缘情况"""
        # 处理空值和unknown情况
        if not inventory_id or inventory_id == 'unknown' or inventory_id.strip() == '':
            return self._get_default_inventory_dimensions()

        inventory_str = str(inventory_id).strip()

        # 移除可能的前缀
        if inventory_str.startswith('inventory_'):
            inventory_str = inventory_str[10:]
            logger.debug(f"🔍 移除inventory_前缀: {inventory_id} → {inventory_str}")

        # 如果是多维度格式 "145884|71|153"
        if '|' in inventory_str:
            parts = inventory_str.split('|')
            result = {
                'inventory_item_code': parts[0].strip() if len(parts) > 0 and parts[0].strip() else 'unknown',
                'inventory_period_code': parts[1].strip() if len(parts) > 1 and parts[1].strip() else 'unknown',
                'inventory_plant_code': parts[2].strip() if len(parts) > 2 and parts[2].strip() else 'unknown',
                'inventory_id': parts[0].strip() if len(parts) > 0 and parts[0].strip() else inventory_str
                # 只保留第一维作为主ID
            }
            logger.debug(f"🔍 拆分inventory_id: {inventory_str} → {result}")
            return result
        else:
            # 🔧 阶段1修复: 增强单一维度的处理逻辑
            clean_inventory_str = inventory_str.strip()

            # 如果是纯数字，直接使用
            if clean_inventory_str.isdigit():
                return {
                    'inventory_item_code': clean_inventory_str,
                    'inventory_period_code': 'unknown',
                    'inventory_plant_code': 'unknown',
                    'inventory_id': clean_inventory_str
                }

            # 如果包含特殊字符或格式，尝试提取有用信息
            import re
            # 尝试提取数字部分
            numbers = re.findall(r'\d+', clean_inventory_str)
            if numbers:
                main_id = numbers[0]  # 使用第一个数字作为主ID
                logger.debug(f"🔍 从复杂inventory_id提取数字: {clean_inventory_str} → {main_id}")
                return {
                    'inventory_item_code': main_id,
                    'inventory_period_code': 'unknown',
                    'inventory_plant_code': 'unknown',
                    'inventory_id': main_id
                }

            # 默认情况：保持原样
            return {
                'inventory_item_code': clean_inventory_str,
                'inventory_period_code': 'unknown',
                'inventory_plant_code': 'unknown',
                'inventory_id': clean_inventory_str
            }

    def _get_default_inventory_dimensions(self) -> Dict[str, str]:
        """🔧 阶段1修复: 获取默认的inventory维度信息"""
        return {
            'inventory_item_code': 'unknown',
            'inventory_period_code': 'unknown',
            'inventory_plant_code': 'unknown',
            'inventory_id': 'unknown'
        }

    def _parse_inventory_id_dimensions(self, inventory_id: str) -> Dict[str, str]:
        """保持向后兼容的原始方法，内部调用增强版本"""
        return self._parse_inventory_id_dimensions_enhanced(inventory_id)

    def _is_fallback_id(self, node_id: str) -> bool:
        """检查是否为回退ID格式"""
        return (node_id.startswith('fallback_') or
                node_id.startswith('EMPTY_ITEM_CODES_') or
                node_id.startswith('error_node_') or
                '_overflow_' in node_id)

    def _parse_fallback_id(self, node_id: str, node_type: str) -> Dict[str, str]:
        """解析回退ID格式"""
        try:
            # 🔧 如果是我们构造的可解析格式
            if '|' in node_id:
                parts = node_id.split('|')
                return self._extract_dimensions_from_parts(node_type, parts)

            # 🔧 如果是其他回退格式，提取索引信息
            index = "0"
            if '_' in node_id:
                parts = node_id.split('_')
                # 尝试找到数字部分作为索引
                for part in reversed(parts):
                    if part.isdigit():
                        index = part
                        break

            # 🔧 根据节点类型构造有意义的维度
            if node_type == 'production':
                return {
                    'item_code': f"fallback_item_{index}",
                    'period_code': 'period_fallback',
                    'plant_code': 'plant_fallback'
                }
            # 移除 forecast_delivery 和 order_delivery 节点的处理
            elif node_type == 'inventory':
                return {
                    'item_code': f"fallback_item_{index}",
                    'period_code': 'period_fallback',
                    'plant_code': 'plant_fallback'
                }
            elif node_type == 'virtual':
                return {
                    'assembly': f"fallback_assembly_{index}",
                    'component': 'component_fallback',
                    'period_code': 'period_fallback',
                    'plant_code': 'plant_fallback'
                }
            else:
                return self._get_default_dimensions(node_type)

        except Exception as e:
            logger.debug(f"回退ID解析失败: {e}")
            return self._get_default_dimensions(node_type)

    def _parse_missing_date_map_id(self, node_id: str, node_type: str) -> Dict[str, str]:
        """解析MISSING_DATE_MAP格式的ID"""
        try:
            # 格式: MISSING_DATE_MAP_{node_type}_{index}
            parts = node_id.split('_')
            index = parts[-1] if len(parts) > 3 else "0"

            # 🔧 构造有意义的维度信息
            if node_type == 'production':
                return {
                    'item_code': f"missing_map_item_{index}",
                    'period_code': 'period_missing_map',
                    'plant_code': 'plant_missing_map'
                }
            # 移除 forecast_delivery 和 order_delivery 节点的处理
            elif node_type == 'inventory':
                return {
                    'item_code': f"missing_map_item_{index}",
                    'period_code': 'period_missing_map',
                    'plant_code': 'plant_missing_map'
                }
            elif node_type == 'virtual':
                return {
                    'assembly': f"missing_map_assembly_{index}",
                    'component': 'component_missing_map',
                    'period_code': 'period_missing_map',
                    'plant_code': 'plant_missing_map'
                }
            else:
                return self._get_default_dimensions(node_type)

        except Exception as e:
            logger.debug(f"MISSING_DATE_MAP ID解析失败: {e}")
            return self._get_default_dimensions(node_type)

    def _get_safe_node_id(self, item_codes: List, index: int, node_type: str) -> str:
        """安全地获取节点ID，增强智能回退逻辑"""
        try:
            # 🔧 方法1: 直接从item_codes获取
            if item_codes and index < len(item_codes):
                node_id = str(item_codes[index])
                logger.debug(f"🔍 从item_codes获取节点ID[{index}]: {node_id}")
                return node_id

            # 🔧 方法2: 处理索引溢出情况
            elif item_codes and len(item_codes) > 0:
                # 使用最后一个有效的item_code作为模板
                last_valid = str(item_codes[-1])
                fallback_id = f"{last_valid}_overflow_{index}"
                logger.debug(f"🔍 索引溢出，使用回退ID[{index}]: {fallback_id}")
                return fallback_id

            # 🔧 方法3: 完全没有item_codes时的智能构造
            else:
                if isinstance(node_type, tuple):
                    fallback_id = f"edge_unknown_{index}"
                else:
                    # ✅ 智能构造可解析的节点ID
                    fallback_id = self._construct_parseable_fallback_id(node_type, index)
                logger.debug(f"🔍 无item_codes，使用智能生成ID[{index}]: {fallback_id}")
                return fallback_id

        except Exception as e:
            logger.warning(f"获取节点ID失败[{index}]: {e}")
            # 🔧 错误情况下也生成可解析的ID
            return self._construct_parseable_fallback_id(node_type, index)

    def _construct_parseable_fallback_id(self, node_type: str, index: int) -> str:
        """构造可被正确解析的回退节点ID"""
        # 🔧 根据节点类型构造符合解析格式的ID
        if node_type == 'production':
            # 格式: item_code|period_code|plant_code
            return f"fallback_item_{index}|period_unknown|plant_unknown"
        # 移除 forecast_delivery 和 order_delivery 节点的处理
        elif node_type == 'inventory':
            # 格式: item_code|period_code|plant_code
            return f"fallback_item_{index}|period_unknown|plant_unknown"
        elif node_type == 'virtual':
            # 格式: assembly|component|period_code|plant_code
            return f"fallback_assembly_{index}|fallback_component|period_unknown|plant_unknown"
        elif node_type == 'capacity':
            # 格式: set_code|period_code|plant_code|shift_code
            return f"fallback_set_{index}|period_unknown|plant_unknown|shift_unknown"
        elif node_type == 'capacitygroup':
            # 格式: set_group|period_code|plant_code
            return f"fallback_group_{index}|period_unknown|plant_unknown"
        elif node_type == 'sr':
            # 格式: item_code|plant_code|start_period_code|end_period_code
            return f"fallback_item_{index}|plant_unknown|start_unknown|end_unknown"
        elif node_type == 'topproduction':
            # 格式: item_code|period_code
            return f"fallback_item_{index}|period_unknown"
        else:
            # 未知节点类型，使用通用格式
            return f"fallback_{node_type}_{index}|unknown_dim1|unknown_dim2"

    def _parse_and_expand_node_info(self, node_id: str, node_type: str) -> dict:
        """解析节点ID并展开为维度列"""
        # 如果node_type是边类型（tuple），直接返回原始信息
        if isinstance(node_type, tuple):
            return {'item_code': node_id}

        # 调用维度解析方法
        dimensions = self._parse_node_dimensions(str(node_id), node_type)

        # 🔧 方案B: 修复item_code格式，确保是纯数字
        if 'item_code' in dimensions:
            original_item_code = dimensions['item_code']
            clean_item_code = self._extract_clean_item_code(original_item_code)
            if clean_item_code != original_item_code:
                logger.debug(f"🔧 item_code格式修复: {original_item_code} → {clean_item_code}")
                dimensions['item_code'] = clean_item_code

        if node_type == 'production':
            return {
                'item_code': dimensions.get('item_code', 'unknown'),
                'period_code': dimensions.get('period_code', 'unknown'),
                'plant_code': dimensions.get('plant_code', 'unknown')
            }
        # 移除 forecast_delivery 和 order_delivery 节点的处理
        elif node_type == 'inventory':
            return {
                'item_code': dimensions.get('item_code', 'unknown'),
                'period_code': dimensions.get('period_code', 'unknown'),
                'plant_code': dimensions.get('plant_code', 'unknown')
            }
        elif node_type == 'virtual':
            return {
                'assembly': dimensions.get('assembly', 'unknown'),
                'component': dimensions.get('component', 'unknown'),
                'period_code': dimensions.get('period_code', 'unknown'),
                'plant_code': dimensions.get('plant_code', 'unknown')
            }
        elif node_type == 'capacity':
            return {
                'set_code': dimensions.get('set_code', 'unknown'),
                'period_code': dimensions.get('period_code', 'unknown'),
                'plant_code': dimensions.get('plant_code', 'unknown'),
                'shift_code': dimensions.get('shift_code', 'unknown')
            }
        elif node_type == 'capacitygroup':
            return {
                'set_group': dimensions.get('set_group', 'unknown'),
                'period_code': dimensions.get('period_code', 'unknown'),
                'plant_code': dimensions.get('plant_code', 'unknown')
            }
        elif node_type == 'sr':
            return {
                'item_code': dimensions.get('item_code', 'unknown'),
                'plant_code': dimensions.get('plant_code', 'unknown'),
                'start_period_code': dimensions.get('start_period_code', 'unknown'),
                'end_period_code': dimensions.get('end_period_code', 'unknown')
            }
        elif node_type == 'topproduction':
            return {
                'item_code': dimensions.get('item_code', 'unknown'),
                'period_code': dimensions.get('period_code', 'unknown')
            }
        else:
            # 对于未知节点类型，返回原始node_id作为item_code
            return {'item_code': str(node_id)}

    def _clean_unknown_value(self, value: str) -> str:
        """🔧 阶段1修复: 清理unknown值，统一处理格式"""
        if not value:
            return 'unknown'

        value_str = str(value).strip()

        # 处理各种unknown格式
        if (value_str == 'unknown_assembly' or
                value_str == 'unknown_component' or
                value_str.startswith('unknown_')):
            return 'unknown'

        return value_str

    def _get_default_transport_edge_info(self) -> Dict[str, str]:
        """默认运输边信息，返回核心维度字段"""
        return {
            'src_plant_code': 'unknown',
            'dst_plant_code': 'unknown',
            'item_code': 'unknown',
            'transport_edge_type': 'inventory_transport',
            'node_type': 'transport_edge'
        }

    def _validate_edge_identifier_format(self, edge_id: str, edge_type: tuple) -> bool:
        """验证边标识符格式"""
        parts = edge_id.split('/')
        
        if edge_type == ('inventory', 'to', 'virtual'):
            return len(parts) >= 2
        elif edge_type == ('inventory', 'to_transport_labeled', 'inventory'):
            return len(parts) >= 3
        return len(parts) >= 1

    def _extract_edge_info_safely(self, edge_type: tuple, edge_index: int, edge_id: str) -> dict:
        """增强边信息解析逻辑，包含格式验证和错误处理"""
        try:
            if not self._validate_edge_identifier_format(edge_id, edge_type):
                logger.warning(f"边标识符格式验证失败: {edge_id}")
                if edge_type == ('inventory', 'to', 'virtual'):
                    default_dims = self._get_default_inventory_dimensions()
                    result = {'assembly': 'format_invalid', 'component': 'format_invalid'}
                    result.update(default_dims)
                    return result
                elif edge_type == ('inventory', 'to_transport_labeled', 'inventory'):
                    return self._get_default_transport_edge_info()
                return {'edge_id': edge_id, 'edge_index': edge_index}

            parts = edge_id.split('/')

            if edge_type == ('inventory', 'to', 'virtual'):
                result = {
                    'assembly': parts[0] if len(parts) > 0 else 'unknown',
                    'component': parts[1] if len(parts) > 1 else 'unknown',
                    'inventory_id': parts[2] if len(parts) > 2 else 'unknown',
                    'edge_index': edge_index,
                    'edge_id': edge_id
                }
            elif edge_type == ('inventory', 'to_transport_labeled', 'inventory'):
                result = {
                    'src_plant_code': parts[0] if len(parts) > 0 else 'unknown',
                    'dst_plant_code': parts[1] if len(parts) > 1 else 'unknown',
                    'item_code': parts[2] if len(parts) > 2 else 'unknown',
                    'src_period': parts[3] if len(parts) > 3 else 'unknown',
                    'dst_period': parts[4] if len(parts) > 4 else 'unknown',
                    'lead_time': parts[5] if len(parts) > 5 else 'unknown',
                    'edge_index': edge_index,
                    'edge_id': edge_id
                }
            else:
                result = {'edge_id': edge_id, 'edge_index': edge_index}

            return result
        except Exception as e:
            logger.warning(f"边信息解析失败: {edge_id}, error: {e}")
            return {'edge_id': edge_id, 'edge_index': edge_index, 'error': str(e)}

    def _validate_output_data(self, df_results, graph_key: str):
        """验证输出数据的完整性"""
        logger.info(f"验证 {graph_key} 输出数据完整性...")

        # 检查unknown值的比例
        unknown_columns = ['item_code', 'period_code', 'plant_code', 'assembly', 'component']
        for col in unknown_columns:
            if col in df_results.columns:
                unknown_count = (
                    df_results[col].astype(str).str.contains('unknown|parse_failed|extraction_failed', na=False)).sum()
                total_count = len(df_results)
                if unknown_count > 0:
                    percentage = unknown_count / total_count * 100
                    if percentage > 10:  # 如果超过10%的数据有问题，记录错误
                        logger.error(f"❌ 列 {col} 中有 {unknown_count}/{total_count} ({percentage:.1f}%) 个异常值")
                    else:
                        logger.warning(f"⚠️ 列 {col} 中有 {unknown_count}/{total_count} ({percentage:.1f}%) 个异常值")
                else:
                    logger.info(f"✅ 列 {col} 无异常值")

        # 检查各节点类型的维度列是否存在
        expected_columns = {
            'production': ['item_code', 'period_code', 'plant_code'],
            # 移除 forecast_delivery 和 order_delivery 节点的列定义
            'inventory': ['item_code', 'period_code', 'plant_code'],
            'virtual': ['assembly', 'component', 'period_code', 'plant_code'],
            'capacity': ['set_code', 'period_code', 'plant_code', 'shift_code'],
            'capacitygroup': ['set_group', 'period_code', 'plant_code'],
            'sr': ['item_code', 'plant_code', 'start_period_code', 'end_period_code'],
            'topproduction': ['item_code', 'period_code']
        }

        # 检查每种节点类型的数据
        node_types_in_data = df_results['node_type'].unique() if 'node_type' in df_results.columns else []

        for node_type in node_types_in_data:
            if isinstance(node_type, str) and node_type in expected_columns:
                expected_cols = expected_columns[node_type]
                missing_cols = [col for col in expected_cols if col not in df_results.columns]
                if missing_cols:
                    logger.error(f"❌ {node_type}节点缺少维度列: {missing_cols}")
                else:
                    # 检查该节点类型数据的完整性
                    node_data = df_results[df_results['node_type'] == node_type]
                    node_count = len(node_data)
                    logger.info(f"✅ {node_type}节点 ({node_count}条) 维度列完整: {expected_cols}")

        # 检查边类型的数据
        edge_types_in_data = []
        if 'edge_type' in df_results.columns:
            edge_types_in_data = df_results['edge_type'].unique()

        for edge_type in edge_types_in_data:
            edge_data = df_results[df_results['edge_type'] == edge_type]
            edge_count = len(edge_data)

            # 检查边的关键列
            required_edge_cols = ['edge_identifier', 'assembly', 'component']
            missing_edge_cols = [col for col in required_edge_cols if col not in df_results.columns]
            if missing_edge_cols:
                logger.error(f"❌ 边类型 {edge_type} 缺少关键列: {missing_edge_cols}")
            else:
                logger.info(f"✅ 边类型 {edge_type} ({edge_count}条) 关键列完整")

        # 统计总体情况
        total_rows = len(df_results)
        total_cols = len(df_results.columns)
        logger.info(f"📊 数据总览: {total_rows}行 x {total_cols}列")

        # 显示列名
        logger.info(f"📋 输出列: {list(df_results.columns)}")

    def _display_csv_samples(self, df_results, graph_key: str):
        """显示CSV样例数据，便于调试"""
        logger.info(f"📋 {graph_key} CSV样例数据预览:")

        # 显示基本信息
        total_rows = len(df_results)
        total_cols = len(df_results.columns)
        logger.info(f"   总计: {total_rows}行 x {total_cols}列")

        # 按数据类型分组显示样例
        # 🔧 修复：正确使用DataFrame列筛选语法，并添加列存在性检查
        if 'node_type' in df_results.columns:
            node_data = df_results[df_results['node_type'].isin(['production', 'inventory', 'virtual'])]
        else:
            node_data = pd.DataFrame()
        
        if 'edge_type' in df_results.columns:
            edge_data = df_results[df_results['edge_type'].notna()]
        else:
            edge_data = pd.DataFrame()

        # 显示节点数据样例
        if not node_data.empty:
            logger.info(f"📊 节点数据样例 (共{len(node_data)}行):")
            for i, (idx, row) in enumerate(node_data.head(3).iterrows()):
                logger.info(f"   样例{i + 1}: {dict(row.head(8))}")  # 只显示前8列避免过长

        # 显示边数据样例
        if not edge_data.empty:
            logger.info(f"📊 边数据样例 (共{len(edge_data)}行):")
            for i, (idx, row) in enumerate(edge_data.head(3).iterrows()):
                logger.info(f"   样例{i + 1}: {dict(row.head(8))}")  # 只显示前8列避免过长

        # 统计各种数据类型的数量
        if 'node_type' in df_results.columns:
            node_type_counts = df_results['node_type'].value_counts()
            logger.info(f"📈 节点类型分布: {dict(node_type_counts)}")

        if 'edge_type' in df_results.columns:
            edge_type_counts = df_results['edge_type'].value_counts()
            logger.info(f"📈 边类型分布: {dict(edge_type_counts)}")

        # 检查关键列的数据质量
        key_columns = ['item_code', 'period_code', 'plant_code', 'assembly', 'component']
        for col in key_columns:
            if col in df_results.columns:
                non_null_count = df_results[col].notna().sum()
                unique_count = df_results[col].nunique()
                sample_values = df_results[col].dropna().head(5).tolist()
                logger.info(f"🔍 列'{col}': {non_null_count}个非空值, {unique_count}个唯一值, 样例: {sample_values}")

    def generate_daily_comparison_csv(self):
        """生成每日预测对比CSV表格"""
        if not self.is_main_process:
            return

        if not self.all_predictions:
            logger.warning("没有预测结果需要处理")
            return

        logger.info(f"生成CSV文件... 共 {len(self.all_predictions)} 个图")

        thresholds = [0.5, 0.2, 0.1, 0.05, 0.001]
        processed_keys = set()

        for graph_key, predictions in self.all_predictions.items():
            if graph_key in processed_keys:
                continue
            processed_keys.add(graph_key)

            self._diagnose_node_id_format(graph_key, predictions)
            csv_data = self._process_node_predictions(predictions, thresholds)
            
            edge_types = [('inventory', 'to', 'virtual'), ('inventory', 'to_transport_labeled', 'inventory')]
            for edge_type in edge_types:
                if edge_type in predictions:
                    edge_data = predictions[edge_type]
                    edge_pred = edge_data['pred']
                    edge_true = edge_data['true']
                    edge_identifiers = edge_data.get('edge_identifiers', [])
                    
                    min_len = min(len(edge_pred), len(edge_true))
                    edge_pred = edge_pred[:min_len]
                    edge_true = edge_true[:min_len]
                    if len(edge_identifiers) > min_len:
                        edge_identifiers = edge_identifiers[:min_len]
                    
                    edge_valid_mask = ~(np.isnan(edge_true) | np.isnan(edge_pred))
                    
                    for i in range(min_len):
                        if edge_valid_mask[i]:
                            edge_id = edge_identifiers[i] if i < len(edge_identifiers) else f"edge_{i}"
                            edge_info = self._extract_edge_info_safely(edge_type, i, edge_id)
                            
                            edge_type_str = "_".join(edge_type)
                            if edge_type == ('inventory', 'to_transport_labeled', 'inventory'):
                                edge_type_str = "inventory_to_transport_labeled_inventory"
                            
                            row = {
                                'edge_type': edge_type_str, 'node_type': '',
                                'edge_index': i, 'true_value': edge_true[i], 'predicted_value_float': edge_pred[i]
                            }
                            
                            if edge_type == ('inventory', 'to', 'virtual'):
                                row.update({
                                    'edge_identifier': edge_id,
                                    'assembly': edge_info.get('assembly', 'unknown'),
                                    'component': edge_info.get('component', 'unknown'),
                                    'inventory_id': edge_info.get('inventory_id', edge_id),
                                    'inventory_item_code': edge_info.get('inventory_item_code', 'unknown'),
                                    'inventory_period_code': edge_info.get('inventory_period_code', 'unknown'),
                                    'inventory_plant_code': edge_info.get('inventory_plant_code', 'unknown')
                                })
                            else:
                                row.update({
                                    'edge_identifier': edge_info.get('edge_id', edge_id),
                                    'src_plant_code': edge_info.get('src_plant_code', 'unknown'),
                                    'dst_plant_code': edge_info.get('dst_plant_code', 'unknown'),
                                    'item_code': edge_info.get('item_code', 'unknown'),
                                    'src_period': edge_info.get('src_period', 'unknown'),
                                    'dst_period': edge_info.get('dst_period', 'unknown'),
                                    'lead_time': edge_info.get('lead_time', 'unknown'),
                                })
                            csv_data.append(row)
            
            self._save_and_validate_csv(graph_key, csv_data)
            self._generate_threshold_accuracy_csv(graph_key, predictions, thresholds)
    def _process_node_predictions(self, predictions: dict, thresholds: List[float]) -> List[dict]:
        """处理节点预测数据，生成CSV行"""
        csv_data = []
        all_edge_types = [
            ('inventory', 'to', 'virtual'), ('virtual', 'to', 'inventory'),
            ('inventory', 'to_transport_labeled', 'inventory'),
            ('inventory', 'to_transport_reverse', 'inventory')
        ]
        
        for node_type, data in predictions.items():
            if node_type in all_edge_types:
                continue
            
            pred = data['pred']
            true = data['true']
            item_codes = data.get('item_codes', [])
            valid_mask = ~(np.isnan(true) | np.isnan(pred))
            
            if not valid_mask.any():
                continue
            
            for i in range(len(pred)):
                if valid_mask[i]:
                    item_code = self._get_safe_node_id(item_codes, i, node_type)
                    node_dimensions = self._parse_and_expand_node_info(item_code, node_type)
                    
                    row = {
                        'node_type': node_type,
                        'true_value': true[i],
                        'predicted_value_float': pred[i],
                    }
                    row.update(node_dimensions)
                    
                    if node_type in ['production']:
                        for threshold in thresholds:
                            pred_binary = 1 if pred[i] > threshold else 0
                            row[f'predicted_binary_th_{threshold}'] = pred_binary
                            row[f'correct_prediction_th_{threshold}'] = int(true[i] == pred_binary)
                        row['predicted_binary'] = 1 if pred[i] > 0.5 else 0
                        row['correct_prediction'] = int(true[i] == (1 if pred[i] > 0.5 else 0))
                    
                    csv_data.append(row)
        return csv_data


    def _save_and_validate_csv(self, graph_key: str, csv_data: List[dict]):
        """保存并验证CSV文件"""
        if not csv_data:
            logger.warning(f"No valid CSV data generated for {graph_key}")
            return
        
        df = pd.DataFrame(csv_data)
        
        if 'item_code' in df.columns:
            item_code_column = df['item_code'].dropna()
            unique_codes = item_code_column.nunique()
            unknown_count = sum(1 for code in item_code_column if str(code).startswith(
                ('unknown_', 'no_item_code', 'error_', 'empty_mapping_', 'no_mapping_', 'MISSING_', 'NO_DATE_MAPPING_')))
            logger.info(f"{graph_key}: {len(item_code_column)} records, unique codes: {unique_codes}, valid: {unique_codes - unknown_count}")
        
        csv_filename = f"graph_{graph_key}_predictions_comparison.csv"
        csv_path = self.csv_output_dir / csv_filename
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions comparison to {csv_path}")
        
        self._display_csv_samples(df, graph_key)
        self._validate_output_data(df, graph_key)

    def _generate_threshold_accuracy_csv(self, graph_key: str, predictions: dict, thresholds: List[float]):
        """生成多阈值准确率统计CSV表"""
        logger.info(f"Generating threshold accuracy statistics for {graph_key}...")

        accuracy_data = []

        for node_type in ['production']:
            if node_type not in predictions:
                continue

            data = predictions[node_type]
            pred = data['pred']
            true = data['true']

            # 不使用有效掩码，使用完整数据集（与predictions_comparison一致）
            # 这样确保总数统计正确
            # 计算每个阈值下的准确率 - 修复版本：不过滤数据，正确处理0 vs 非0
            for threshold in thresholds:
                # 使用完整数据集，确保total_nodes正确
                total_predictions = len(pred)

                # 将NaN视为0处理（保持数据完整性）
                true_clean = np.nan_to_num(true, nan=0.0)
                pred_clean = np.nan_to_num(pred, nan=0.0)

                # 二分类：0 vs 非0（这里的1指的是非0值）
                true_binary = (true_clean != 0).astype(int)  # 0=零值，1=非零值
                pred_binary = (pred_clean > threshold).astype(int)  # 预测的二分类结果

                # 标签统计：基于完整数据集
                label_0_count = np.sum(true_binary == 0)  # 真实零值的数量
                label_1_count = np.sum(true_binary == 1)  # 真实非零值的数量

                # 验证总数
                assert label_0_count + label_1_count == total_predictions, f"标签统计错误: {label_0_count} + {label_1_count} != {total_predictions}"

                # 计算准确率（基于完整数据）
                correct_predictions = np.sum(true_binary == pred_binary)
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

                # 计算混淆矩阵（基于完整数据）
                tn = np.sum((true_binary == 0) & (pred_binary == 0))  # 正确预测为0
                fp = np.sum((true_binary == 0) & (pred_binary == 1))  # 错误预测为非0
                fn = np.sum((true_binary == 1) & (pred_binary == 0))  # 错误预测为0
                tp = np.sum((true_binary == 1) & (pred_binary == 1))  # 正确预测为非0

                # 验证混淆矩阵
                assert tn + fp + fn + tp == total_predictions, f"混淆矩阵错误: {tn + fp + fn + tp} != {total_predictions}"

                # 计算剪枝相关的实用指标
                total_pruned = fn + tn  # 剪枝总数（预测为0的总数）
                pruned_false_negative_rate = fn / (fn + tn) if (fn + tn) > 0 else 0  # 剪枝中错误剪枝的比例
                label_0_retention_rate = tn / label_0_count if label_0_count > 0 else 0  # 零值保留率（相对于总零值）
                label_1_loss_rate = fn / label_1_count if label_1_count > 0 else 0  # 非零值损失率（相对于总非零值）

                # 计算各标签的准确率
                correct_0_predictions = tn  # 正确预测为0的数量
                correct_1_predictions = tp  # 正确预测为非0的数量

                accuracy_label_0 = correct_0_predictions / label_0_count if label_0_count > 0 else 0
                accuracy_label_1 = correct_1_predictions / label_1_count if label_1_count > 0 else 0

                row = {
                    'graph_key': graph_key,
                    'node_type': node_type,
                    'threshold': threshold,
                    'total_nodes': total_predictions,  # 完整数据集大小
                    'label_0_count': label_0_count,  # 零值数量
                    'label_1_count': label_1_count,  # 非零值数量
                    'correct_predictions': correct_predictions,
                    'correct_0_predictions': correct_0_predictions,  # 正确预测为0的数量
                    'correct_1_predictions': correct_1_predictions,  # 正确预测为非0的数量
                    'accuracy_label_0': accuracy_label_0,  # 零值预测准确率
                    'accuracy_label_1': accuracy_label_1,  # 非零值预测准确率
                    'true_positive': tp,
                    'true_negative': tn,
                    'false_positive': fp,
                    'false_negative': fn,
                    'total_pruned': total_pruned,  # 剪枝总数
                    'pruned_false_negative_rate': pruned_false_negative_rate,  # 剪枝中错误剪枝比例
                    'label_0_retention_rate': label_0_retention_rate,  # 零值保留率
                    'label_1_loss_rate': label_1_loss_rate  # 非零值损失率
                }
                # 移除详细阈值日志以提升性能
                accuracy_data.append(row)

        # 🔧 处理边预测的阈值分析：仅支持虚拟边和新的正向运输边
        edge_types_with_labels = [
            ('inventory', 'to', 'virtual'), 
            ('inventory', 'to_transport_labeled', 'inventory')  # 新的正向边（有标签）
        ]
        for edge_type in edge_types_with_labels:
            if edge_type not in predictions:
                continue

            edge_data = predictions[edge_type]
            edge_pred = edge_data['pred']
            edge_true = edge_data['true']
            edge_type_str = "_".join(edge_type)

            # 计算每个阈值下的准确率 - 修复版本：不过滤数据，正确处理0 vs 非0
            for threshold in thresholds:
                # 使用完整数据集，确保total_nodes正确
                total_predictions = len(edge_pred)

                # 将NaN视为0处理（保持数据完整性）
                edge_true_clean = np.nan_to_num(edge_true, nan=0.0)
                edge_pred_clean = np.nan_to_num(edge_pred, nan=0.0)

                # 二分类：0 vs 非0（这里的1指的是非0值）
                edge_true_binary = (edge_true_clean != 0).astype(int)  # 0=零值，1=非零值
                edge_pred_binary = (edge_pred_clean > threshold).astype(int)  # 预测的二分类结果

                # 标签统计：基于完整数据集
                label_0_count = np.sum(edge_true_binary == 0)  # 真实零值的数量
                label_1_count = np.sum(edge_true_binary == 1)  # 真实非零值的数量

                # 验证总数
                assert label_0_count + label_1_count == total_predictions, f"边标签统计错误: {label_0_count} + {label_1_count} != {total_predictions}"

                # 计算混淆矩阵（基于完整数据）
                tn = np.sum((edge_true_binary == 0) & (edge_pred_binary == 0))  # 正确预测为0
                fp = np.sum((edge_true_binary == 0) & (edge_pred_binary == 1))  # 错误预测为非0
                fn = np.sum((edge_true_binary == 1) & (edge_pred_binary == 0))  # 错误预测为0
                tp = np.sum((edge_true_binary == 1) & (edge_pred_binary == 1))  # 正确预测为非0

                # 验证混淆矩阵
                assert tn + fp + fn + tp == total_predictions, f"边混淆矩阵错误: {tn + fp + fn + tp} != {total_predictions}"

                # 计算剪枝相关的实用指标
                total_pruned = fn + tn  # 剪枝总数（预测为0的总数）
                pruned_false_negative_rate = fn / (fn + tn) if (fn + tn) > 0 else 0  # 剪枝中错误剪枝的比例
                label_0_retention_rate = tn / label_0_count if label_0_count > 0 else 0  # 零值保留率（相对于总零值）
                label_1_loss_rate = fn / label_1_count if label_1_count > 0 else 0  # 非零值损失率（相对于总非零值）

                # 计算各标签的准确率
                correct_0_predictions = tn  # 正确预测为0的数量
                correct_1_predictions = tp  # 正确预测为非0的数量

                accuracy_label_0 = correct_0_predictions / label_0_count if label_0_count > 0 else 0
                accuracy_label_1 = correct_1_predictions / label_1_count if label_1_count > 0 else 0

                row = {
                    'graph_key': graph_key,
                    'edge_type': edge_type_str,  # 🔧 方案1: 正确！边类型应该填入edge_type
                    'node_type': 'edge',  # 🔧 方案1: 边数据的node_type应该是'edge'
                    'threshold': threshold,
                    'total_nodes': total_predictions,  # 完整数据集大小
                    'label_0_count': label_0_count,  # 零值数量
                    'label_1_count': label_1_count,  # 非零值数量
                    'correct_predictions': correct_predictions,
                    'correct_0_predictions': correct_0_predictions,  # 正确预测为0的数量
                    'correct_1_predictions': correct_1_predictions,  # 正确预测为非0的数量
                    'accuracy_label_0': accuracy_label_0,  # 零值预测准确率
                    'accuracy_label_1': accuracy_label_1,  # 非零值预测准确率
                    'true_positive': tp,
                    'true_negative': tn,
                    'false_positive': fp,
                    'false_negative': fn,
                    'total_pruned': total_pruned,  # 剪枝总数
                    'pruned_false_negative_rate': pruned_false_negative_rate,  # 剪枝中错误剪枝比例
                    'label_0_retention_rate': label_0_retention_rate,  # 零值保留率
                    'label_1_loss_rate': label_1_loss_rate  # 非零值损失率
                }
                # 移除详细边阈值日志以提升性能
                accuracy_data.append(row)

        if accuracy_data:
            df = pd.DataFrame(accuracy_data)
            csv_filename = f"graph_{graph_key}_threshold_accuracy_analysis.csv"
            csv_path = self.csv_output_dir / csv_filename
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved threshold accuracy analysis to {csv_path}")

    def generate_binary_node_statistics_csv(self):
        """生成二分类节点统计CSV，包含标签分布和直方图统计，增强版本"""
        # 只有主进程生成文件
        if not self.is_main_process:
            return

        # 检查是否有数据需要处理
        if not self.all_predictions:
            logger.warning("没有预测结果需要处理，跳过统计CSV生成")
            return

        logger.info(f"Generating enhanced binary node statistics CSV... 共 {len(self.all_predictions)} 个图")

        # 定义多个阈值
        thresholds = [0.5, 0.2, 0.1, 0.05, 0.001]
        statistics_data = []

        for graph_key, predictions in self.all_predictions.items():
            # 只处理二分类节点，不包括virtual节点
            for node_type in ['production', 'inventory']:
                if node_type not in predictions:
                    continue

                data = predictions[node_type]
                pred = data['pred']
                true = data['true']

                # 创建有效掩码
                valid_mask = ~(np.isnan(true) | np.isnan(pred))

                if not valid_mask.any():
                    continue

                pred_valid = pred[valid_mask]
                true_valid = true[valid_mask]

                # 统计真实标签分布
                label_0_count = np.sum(true_valid == 0)
                label_1_count = np.sum(true_valid == 1)

                # 生成直方图统计
                hist_stats = self._generate_histogram_statistics(
                    pred_valid, true_valid, node_type
                )

                # 获取item_code统计信息（如果有的话）
                item_code_info = {}
                if 'item_codes' in data and data['item_codes']:
                    item_codes_valid = np.array(data['item_codes'])[valid_mask]
                    unique_item_codes = len(set(item_codes_valid))
                    # 过滤掉unknown/error类型的item_codes来获取真实的有效codes
                    valid_codes = [code for code in item_codes_valid
                                   if not str(code).startswith(
                            ('unknown_', 'no_item_code', 'error_', 'empty_mapping_', 'no_mapping_'))]
                    actual_valid_codes = len(set(valid_codes))

                    item_code_info = {
                        'unique_item_codes': unique_item_codes,
                        'actual_valid_item_codes': actual_valid_codes,
                        'example_item_codes': ','.join(list(set(item_codes_valid))[:5])  # 显示前5个示例
                    }

                    logger.info(f"{graph_key} {node_type}: {unique_item_codes} unique codes, "
                                f"{actual_valid_codes} actual valid codes in statistics")
                else:
                    logger.warning(f"{graph_key} {node_type}: No item_codes available for statistics")
                    item_code_info = {
                        'unique_item_codes': 0,
                        'actual_valid_item_codes': 0,
                        'example_item_codes': 'no_item_codes_available'
                    }

                # 基础统计信息
                base_row = {
                    'graph_key': graph_key,
                    'node_type': node_type,
                    'total_nodes': len(true_valid),
                    'label_0_count': label_0_count,
                    'label_1_count': label_1_count,
                    'mean_prediction': np.mean(pred_valid),
                    'std_prediction': np.std(pred_valid),
                    'min_prediction': np.min(pred_valid),
                    'max_prediction': np.max(pred_valid),
                    **hist_stats,
                    **item_code_info  # 新增：item_code统计信息
                }

                # 为每个阈值添加统计信息
                for threshold in thresholds:
                    pred_binary = (pred_valid > threshold).astype(int)

                    # 计算混淆矩阵
                    tn = np.sum((true_valid == 0) & (pred_binary == 0))
                    fp = np.sum((true_valid == 0) & (pred_binary == 1))
                    fn = np.sum((true_valid == 1) & (pred_binary == 0))
                    tp = np.sum((true_valid == 1) & (pred_binary == 1))

                    accuracy = (tp + tn) / len(true_valid) if len(true_valid) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                    # 添加阈值特定的统计
                    threshold_stats = {
                        f'th_{threshold}_true_negative': tn,
                        f'th_{threshold}_false_positive': fp,
                        f'th_{threshold}_false_negative': fn,
                        f'th_{threshold}_true_positive': tp,
                        f'th_{threshold}_accuracy': accuracy,
                        f'th_{threshold}_precision': precision,
                        f'th_{threshold}_recall': recall
                    }

                    base_row.update(threshold_stats)

                statistics_data.append(base_row)

        # 🔧 处理边预测的统计：仅支持虚拟边和新的正向运输边
        for graph_key, predictions in self.all_predictions.items():
            edge_types_with_labels = [
                ('inventory', 'to', 'virtual'), 
                ('inventory', 'to_transport_labeled', 'inventory')  # 新的正向边（有标签）
            ]
            for edge_type in edge_types_with_labels:
                if edge_type not in predictions:
                    continue

                edge_data = predictions[edge_type]
                edge_pred = edge_data['pred']
                edge_true = edge_data['true']

                # 创建有效掩码
                edge_valid_mask = ~(np.isnan(edge_true) | np.isnan(edge_pred))

                if not edge_valid_mask.any():
                    continue

                edge_pred_valid = edge_pred[edge_valid_mask]
                edge_true_valid = edge_true[edge_valid_mask]

                edge_type_str = "_".join(edge_type)

                # 统计真实标签分布
                label_0_count = np.sum(edge_true_valid == 0)
                label_1_count = np.sum(edge_true_valid == 1)

                # 生成直方图统计
                hist_stats = self._generate_histogram_statistics(
                    edge_pred_valid, edge_true_valid, edge_type_str
                )

                # 获取边标识符统计信息
                edge_identifier_info = {}
                if 'edge_identifiers' in edge_data and edge_data['edge_identifiers']:
                    edge_identifiers_valid = np.array(edge_data['edge_identifiers'])[edge_valid_mask]
                    unique_edge_identifiers = len(set(edge_identifiers_valid))
                    # 过滤掉unknown/error类型的edge_identifiers
                    valid_identifiers = [eid for eid in edge_identifiers_valid
                                         if not str(eid).startswith(
                            ('edge_unknown_', 'edge_error_', 'edge_no_mapping_'))]
                    actual_valid_identifiers = len(set(valid_identifiers))

                    edge_identifier_info = {
                        'unique_item_codes': unique_edge_identifiers,  # 保持列名一致性
                        'actual_valid_item_codes': actual_valid_identifiers,
                        'example_item_codes': ','.join(list(set(edge_identifiers_valid))[:5])  # 显示前5个示例
                    }

                    logger.info(f"{graph_key} {edge_type_str}: {unique_edge_identifiers} unique edge identifiers, "
                                f"{actual_valid_identifiers} actual valid identifiers in statistics")
                else:
                    logger.warning(f"{graph_key} {edge_type_str}: No edge_identifiers available for statistics")
                    edge_identifier_info = {
                        'unique_item_codes': 0,
                        'actual_valid_item_codes': 0,
                        'example_item_codes': 'no_edge_identifiers_available'
                    }

                # 基础统计信息
                base_row = {
                    'graph_key': graph_key,
                    'edge_type': edge_type_str,  # 🔧 方案1: 正确！边类型应该填入edge_type
                    'node_type': 'edge',  # 🔧 方案1: 边数据的node_type应该是'edge'
                    'total_nodes': len(edge_true_valid),
                    'label_0_count': label_0_count,
                    'label_1_count': label_1_count,
                    'mean_prediction': np.mean(edge_pred_valid),
                    'std_prediction': np.std(edge_pred_valid),
                    'min_prediction': np.min(edge_pred_valid),
                    'max_prediction': np.max(edge_pred_valid),
                    **hist_stats,
                    **edge_identifier_info  # 使用edge_identifier统计信息
                }

                # 为每个阈值添加统计信息
                for threshold in thresholds:
                    edge_pred_binary = (edge_pred_valid > threshold).astype(int)

                    # 计算混淆矩阵
                    tn = np.sum((edge_true_valid == 0) & (edge_pred_binary == 0))
                    fp = np.sum((edge_true_valid == 0) & (edge_pred_binary == 1))
                    fn = np.sum((edge_true_valid == 1) & (edge_pred_binary == 0))
                    tp = np.sum((edge_true_valid == 1) & (edge_pred_binary == 1))

                    accuracy = (tp + tn) / len(edge_true_valid) if len(edge_true_valid) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                    # 添加阈值特定的统计
                    threshold_stats = {
                        f'th_{threshold}_true_negative': tn,
                        f'th_{threshold}_false_positive': fp,
                        f'th_{threshold}_false_negative': fn,
                        f'th_{threshold}_true_positive': tp,
                        f'th_{threshold}_accuracy': accuracy,
                        f'th_{threshold}_precision': precision,
                        f'th_{threshold}_recall': recall
                    }

                    base_row.update(threshold_stats)

                statistics_data.append(base_row)

        if statistics_data:
            df = pd.DataFrame(statistics_data)
            csv_path = self.csv_output_dir / "enhanced_binary_node_statistics.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved enhanced binary node statistics to {csv_path}")

            # 保持原有的简化版本以向后兼容
            self._generate_legacy_binary_statistics(statistics_data)

    def _generate_legacy_binary_statistics(self, statistics_data: List[Dict]):
        """生成传统的二分类统计表以保持向后兼容"""
        legacy_data = []

        for row in statistics_data:
            # 提取0.5阈值的统计信息作为传统格式
            legacy_row = {
                'graph_key': row['graph_key'],
                'node_type': row['node_type'],
                'total_nodes': row['total_nodes'],
                'label_0_count': row['label_0_count'],
                'label_1_count': row['label_1_count'],
                'true_negative': row['th_0.5_true_negative'],
                'false_positive': row['th_0.5_false_positive'],
                'false_negative': row['th_0.5_false_negative'],
                'true_positive': row['th_0.5_true_positive'],
                'accuracy': row['th_0.5_accuracy'],
                'precision': row['th_0.5_precision'],
                'recall': row['th_0.5_recall']
            }

            # 添加直方图统计
            for key, value in row.items():
                if key.startswith('label_0_hist_bin_') or key.startswith('label_1_hist_bin_') or \
                        key.startswith('label_0_mean_') or key.startswith('label_0_std_') or \
                        key.startswith('label_1_mean_') or key.startswith('label_1_std_'):
                    legacy_row[key] = value

            legacy_data.append(legacy_row)

        if legacy_data:
            df = pd.DataFrame(legacy_data)
            csv_path = self.csv_output_dir / "binary_node_statistics.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved legacy binary node statistics to {csv_path}")

    def _generate_histogram_statistics(self, pred: np.ndarray, true: np.ndarray,
                                       node_type: str) -> Dict[str, Any]:
        """生成直方图统计信息"""
        # 分别为label 0和1的节点生成直方图统计
        label_0_mask = true == 0
        label_1_mask = true == 1

        stats = {}

        # Label 0 的预测值分布
        if np.any(label_0_mask):
            pred_label_0 = pred[label_0_mask]
            hist_0, bins_0 = np.histogram(pred_label_0, bins=20, range=(0, 1))
            stats.update({
                f'label_0_hist_bin_{i}': count for i, count in enumerate(hist_0)
            })
            stats['label_0_mean_pred'] = np.mean(pred_label_0)
            stats['label_0_std_pred'] = np.std(pred_label_0)

        # Label 1 的预测值分布
        if np.any(label_1_mask):
            pred_label_1 = pred[label_1_mask]
            hist_1, bins_1 = np.histogram(pred_label_1, bins=20, range=(0, 1))
            stats.update({
                f'label_1_hist_bin_{i}': count for i, count in enumerate(hist_1)
            })
            stats['label_1_mean_pred'] = np.mean(pred_label_1)
            stats['label_1_std_pred'] = np.std(pred_label_1)

        return stats

    def visualize_binary_predictions(self, graph_key: str, node_type: str):
        """可视化二分类预测结果（0 vs 非0）- 修复版本，显示完整数据集"""
        if graph_key not in self.all_predictions or node_type not in self.all_predictions[graph_key]:
            logger.warning(f"No predictions found for {graph_key} {node_type}")
            return

        data = self.all_predictions[graph_key][node_type]
        pred = data['pred']
        true = data['true']

        # 不过滤数据，使用全部数据来显示完整分布
        logger.info(f"Visualizing {node_type} with complete dataset: total={len(pred)} predictions")
        # 数据预处理：将NaN转为0，确保数据完整性
        true_clean = np.nan_to_num(true, nan=0.0)
        pred_clean = np.nan_to_num(pred, nan=0.0)
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Label 0的预测分布 - 零值
        label_0_mask = true_clean == 0  # 零值
        if np.any(label_0_mask):
            pred_label_0 = pred_clean[label_0_mask]  # 获取零值对应的预测值
            n, bins, patches = ax1.hist(pred_label_0, bins=20, alpha=0.7,
                                        color='blue', edgecolor='black', range=(0, 1))

            # 在每个条形上添加数字
            for i, (count, bin_edge) in enumerate(zip(n, bins[:-1])):
                if count > 0:
                    ax1.text(bin_edge + (bins[1] - bins[0]) / 2, count + max(n) * 0.01,
                             f'{int(count)}', ha='center', va='bottom')
            # 更新标题显示完整数据集信息
            title = f'{graph_key} {node_type} - Predictions for Label 0 \n'
            title += f'Total Label 0: {len(pred_label_0)}'
            ax1.set_title(title)
            ax1.set_xlabel('Predicted Value')
            ax1.set_ylabel('Count')
            ax1.set_xlim(0, 1)  # 限制x轴范围为0-1
            ax1.grid(True, alpha=0.3)

        # Label 1的预测分布
        label_1_mask = true_clean != 0  # 非零值
        if np.any(label_1_mask):
            pred_label_1 = pred_clean[label_1_mask]
            n, bins, patches = ax2.hist(pred_label_1, bins=20, alpha=0.7,
                                        color='red', edgecolor='black', range=(0, 1))

            # 在每个条形上添加数字
            for i, (count, bin_edge) in enumerate(zip(n, bins[:-1])):
                if count > 0:
                    ax2.text(bin_edge + (bins[1] - bins[0]) / 2, count + max(n) * 0.01,
                             f'{int(count)}', ha='center', va='bottom')

            # 更新标题显示完整数据集信息
            title = f'{graph_key} {node_type} - Predictions for Label 1 \n'
            title += f'Total Label 1: {len(pred_label_1)}'
            ax2.set_title(title)
            ax2.set_xlabel('Predicted Value')
            ax2.set_ylabel('Count')
            ax2.set_xlim(0, 1)  # 限制x轴范围为0-1
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片（使用基于日期的命名）
        plot_filename = f"graph_{graph_key}_{node_type}_binary_histogram.png"
        plot_path = self.visualization_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved binary histogram to {plot_path}")

    def visualize_continuous_predictions(self, graph_key: str, node_type: str):
        """可视化连续值预测结果"""
        if graph_key not in self.all_predictions or node_type not in self.all_predictions[graph_key]:
            logger.warning(f"No predictions found for {graph_key} {node_type}")
            return

        data = self.all_predictions[graph_key][node_type]
        pred = data['pred']
        true = data['true']

        # 边类型没有layers信息，使用默认值
        if isinstance(node_type, tuple):  # 边类型
            layers = np.zeros_like(pred)  # 边没有layer概念，使用0
        else:  # 节点类型
            layers = data['layers']

        # 创建有效掩码
        valid_mask = ~(np.isnan(true) | np.isnan(pred))
        if not valid_mask.any():
            return

        pred_valid = pred[valid_mask]
        true_valid = true[valid_mask]
        layers_valid = layers[valid_mask] if layers is not None else None

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 整体散点图
        axes[0, 0].scatter(true_valid, pred_valid, alpha=0.6, s=10)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2)  # 使用0-1范围的对角线
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title(f'{graph_key} {node_type} - Overall Predictions')
        axes[0, 0].set_xlim(0, 1)  # 限制x轴范围为0-1
        axes[0, 0].set_ylim(0, 1)  # 限制y轴范围为0-1
        axes[0, 0].grid(True, alpha=0.3)

        # 残差图
        residuals = pred_valid - true_valid
        axes[0, 1].scatter(true_valid, residuals, alpha=0.6, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('True Values')
        axes[0, 1].set_ylabel('Residuals (Pred - True)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].set_xlim(0, 1)  # 限制x轴范围为0-1
        axes[0, 1].grid(True, alpha=0.3)

        # 误差分布直方图
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # 按层级的散点图（如果有层级信息）
        if layers_valid is not None:
            unique_layers = np.unique(layers_valid)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_layers)))

            for layer, color in zip(unique_layers, colors):
                layer_mask = layers_valid == layer
                axes[1, 1].scatter(true_valid[layer_mask], pred_valid[layer_mask],
                                   alpha=0.6, label=f'Layer {layer}', color=color, s=10)

            axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2)  # 使用0-1范围的对角线
            axes[1, 1].set_xlabel('True Values')
            axes[1, 1].set_ylabel('Predictions')
            axes[1, 1].set_title('Layer-wise Predictions')
            axes[1, 1].set_xlim(0, 1)  # 限制x轴范围为0-1
            axes[1, 1].set_ylim(0, 1)  # 限制y轴范围为0-1
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # 如果没有层级信息，显示预测值分布
            axes[1, 1].hist(pred_valid, bins=30, alpha=0.7, label='Predictions', color='blue', range=(0, 1))
            axes[1, 1].hist(true_valid, bins=30, alpha=0.7, label='True Values', color='orange', range=(0, 1))
            axes[1, 1].set_xlabel('Values')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Value Distribution')
            axes[1, 1].set_xlim(0, 1)  # 限制x轴范围为0-1
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片（使用基于日期的命名）
        plot_filename = f"graph_{graph_key}_{node_type}_continuous_analysis.png"
        plot_path = self.visualization_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved continuous analysis plot to {plot_path}")

    def visualize_edge_predictions(self, graph_key: str, edge_type: tuple):
        """通用边预测结果可视化（支持多种边类型）"""
        if graph_key not in self.all_predictions or edge_type not in self.all_predictions[graph_key]:
            logger.warning(f"No edge predictions found for {graph_key} {edge_type}")
            return

        data = self.all_predictions[graph_key][edge_type]
        pred = data['pred']
        true = data['true']

        # 创建有效掩码
        valid_mask = ~(np.isnan(true) | np.isnan(pred))
        if not valid_mask.any():
            return

        pred_valid = pred[valid_mask]
        true_valid = true[valid_mask]

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 生成标题前缀
        edge_type_str = "_".join(edge_type)
        if edge_type == ('inventory', 'to', 'virtual'):
            title_prefix = "Inventory-Virtual Edge"
        elif edge_type == ('inventory', 'to_transport_labeled', 'inventory'):
            title_prefix = "Inventory-Transport Labeled Edge"
        elif edge_type == ('inventory', 'to_transport_reverse', 'inventory'):
            title_prefix = "Inventory-Transport Reverse Edge"
        else:
            title_prefix = f"{edge_type_str} Edge"

        # Label 0的预测分布
        label_0_mask = true_valid == 0
        if np.any(label_0_mask):
            pred_label_0 = pred_valid[label_0_mask]
            n, bins, patches = ax1.hist(pred_label_0, bins=20, alpha=0.7,
                                        color='blue', edgecolor='black', range=(0, 1))

            # 在每个条形上添加数字
            for i, (count, bin_edge) in enumerate(zip(n, bins[:-1])):
                if count > 0:
                    ax1.text(bin_edge + (bins[1] - bins[0]) / 2, count + max(n) * 0.01,
                             f'{int(count)}', ha='center', va='bottom')

            ax1.set_title(f'{graph_key} {title_prefix} - Predictions for Label 0 (Count: {len(pred_label_0)})')
            ax1.set_xlabel('Predicted Value')
            ax1.set_ylabel('Count')
            ax1.set_xlim(0, 1)
            ax1.grid(True, alpha=0.3)

        # Label 1的预测分布
        label_1_mask = true_valid == 1
        if np.any(label_1_mask):
            pred_label_1 = pred_valid[label_1_mask]
            n, bins, patches = ax2.hist(pred_label_1, bins=20, alpha=0.7,
                                        color='red', edgecolor='black', range=(0, 1))

            # 在每个条形上添加数字
            for i, (count, bin_edge) in enumerate(zip(n, bins[:-1])):
                if count > 0:
                    ax2.text(bin_edge + (bins[1] - bins[0]) / 2, count + max(n) * 0.01,
                             f'{int(count)}', ha='center', va='bottom')

            ax2.set_title(f'{graph_key} {title_prefix} - Predictions for Label 1 (Count: {len(pred_label_1)})')
            ax2.set_xlabel('Predicted Value')
            ax2.set_ylabel('Count')
            ax2.set_xlim(0, 1)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像到可视化目录
        plot_filename = f"graph_{graph_key}_{edge_type_str}_binary_histogram.png"
        plot_path = self.visualization_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved {edge_type_str} edge histogram to {plot_path}")

    def visualize_inventory_virtual_edge_predictions(self, graph_key: str, edge_type: tuple):
        """保持向后兼容的inventory-to-virtual边可视化方法"""
        self.visualize_edge_predictions(graph_key, edge_type)

    def generate_all_visualizations(self):
        """生成所有可视化"""
        # 只有主进程生成文件
        if not self.is_main_process:
            return

        # 检查是否有数据需要处理
        if not self.all_predictions:
            logger.warning("没有预测结果需要处理，跳过可视化生成")
            return

        logger.info(f"Generating all visualizations... 共 {len(self.all_predictions)} 个图")

        # 跟踪已处理的图和节点组合，避免重复处理
        processed_items = set()

        for graph_key in self.all_predictions:
            for node_type in self.all_predictions[graph_key]:
                # 创建唯一标识符，避免重复处理
                item_id = f"{graph_key}_{node_type}"
                if item_id in processed_items:
                    logger.info(f"图 {graph_key} 的 {node_type} 节点已处理，跳过重复处理")
                    continue

                processed_items.add(item_id)

                # 🔧 处理边可视化：仅支持有标签边
                if isinstance(node_type, tuple):
                    # 仅对有标签边进行可视化：虚拟边 + 新的正向运输边
                    if node_type in [('inventory', 'to', 'virtual'),
                                   ('inventory', 'to_transport_labeled', 'inventory')]:
                        self.visualize_edge_predictions(graph_key, node_type)
                    # 其他边类型跳过可视化
                # 处理节点可视化
                elif node_type in ['production', 'inventory']:
                    self.visualize_binary_predictions(graph_key, node_type)
                elif node_type not in ['capacity']:  # 跳过capacity节点，其他连续值节点进行可视化.现在是空的
                    self.visualize_continuous_predictions(graph_key, node_type)

    def process_all_results(self):
        """处理所有结果：生成CSV和可视化
        
        🔧 多卡汇总修复：只有主进程处理结果，确保只输出一份汇总结果
        """
        # 🔧 诊断日志（使用 print 确保可见）
        print(f"\n[Rank {self.rank}] 📊 process_all_results() 入口", flush=True)
        print(f"[Rank {self.rank}]   is_main_process={self.is_main_process}", flush=True)
        print(f"[Rank {self.rank}]   all_predictions 数量={len(self.all_predictions) if self.all_predictions else 0}", flush=True)
        
        # 只有主进程处理结果
        if not self.is_main_process:
            print(f"[Rank {self.rank}] ⚠️ 非主进程，跳过结果处理", flush=True)
            logger.info(f"⚠️ 非主进程(rank {self.rank})跳过结果处理")
            return

        # 检查是否有预测结果可处理
        if not self.all_predictions:
            print(f"[Rank {self.rank}] ❌ 没有预测结果！all_predictions 为空", flush=True)
            logger.warning("❌ 没有预测结果可处理，请确保在调用前已收集预测结果")
            return

        print(f"[Rank {self.rank}] ✅ 有 {len(self.all_predictions)} 个图的预测结果", flush=True)
        print(f"[Rank {self.rank}] 📁 输出目录: {self.save_dir}", flush=True)
        print(f"[Rank {self.rank}] 📊 CSV输出目录: {self.csv_output_dir}", flush=True)
        print(f"[Rank {self.rank}] 📈 可视化输出目录: {self.visualization_dir}", flush=True)
        
        logger.info(f"✅ 主进程(rank 0)开始处理所有预测结果... 共 {len(self.all_predictions)} 个图")
        logger.info(f"📁 输出目录: {self.save_dir}")
        logger.info(f"📊 CSV输出目录: {self.csv_output_dir}")
        logger.info(f"📈 可视化输出目录: {self.visualization_dir}")

        # 生成各种输出
        print(f"[Rank {self.rank}] 🔧 生成 daily comparison CSV...", flush=True)
        self.generate_daily_comparison_csv()
        print(f"[Rank {self.rank}] 🔧 生成 binary node statistics CSV...", flush=True)
        self.generate_binary_node_statistics_csv()
        print(f"[Rank {self.rank}] 🔧 生成可视化...", flush=True)
        self.generate_all_visualizations()

        print(f"[Rank {self.rank}] ✅ 所有结果处理完成!", flush=True)
        logger.info("All prediction results processed successfully")

    def clear_predictions(self):
        """清空存储的预测结果"""
        self.all_predictions.clear()
        self.date_mapping.clear()
        logger.info("Cleared all stored predictions")

    def _load_date_specific_mappings(self, graph_builder, graph_date: str) -> dict:
        """优化的日期特定映射加载，增强版本带有详细日志"""
        date_specific_mappings = {}
        if graph_builder and hasattr(graph_builder, 'date_specific_mappings'):
            from datetime import datetime
            date_obj = datetime.strptime(graph_date, "%Y%m%d")
            logger.info(f"从图构建器中加载日期 {graph_date} 的特定映射")

            # 使用列表推导优化性能
            node_types = ['production', 'virtual', 'inventory', 'capacity']
            for node_type in node_types:
                mapping = graph_builder.get_date_specific_mapping(date_obj, node_type)
                if mapping:
                    date_specific_mappings[node_type] = mapping
                    logger.info(f"成功加载节点类型 {node_type} 的映射，包含 {len(mapping)} 个条目")

                    # 🔧 方案G修复1: 详细映射诊断
                    sample_keys = list(mapping.keys())[:5]
                    logger.info(f"🔍 {node_type} 映射详细诊断:")
                    logger.info(f"   - 映射大小: {len(mapping)}")
                    logger.info(f"   - 前5个键原始: {sample_keys}")
                    logger.info(f"   - 键类型: {[type(k).__name__ for k in sample_keys]}")
                    logger.info(f"   - 键字符串化: {[str(k) for k in sample_keys]}")
                    logger.info(f"   - 多维度检测: {['|' in str(k) for k in sample_keys]}")

                    # 特别检查tuple类型的键
                    tuple_keys = [k for k in sample_keys if isinstance(k, tuple)]
                    if tuple_keys:
                        logger.info(f"   - 🔍 发现tuple键: {tuple_keys}")
                        logger.info(f"   - tuple转字符串: {[str(k) for k in tuple_keys]}")
                        logger.info(
                            f"   - tuple转多维度格式: {['|'.join(str(x) for x in k) if isinstance(k, tuple) else str(k) for k in tuple_keys]}")

                    # 检查inventory前缀
                    if node_type == 'inventory':
                        prefix_count = sum(
                            1 for k in mapping.keys() if isinstance(k, str) and k.startswith('inventory_'))
                        if prefix_count > 0:
                            logger.info(f"'inventory_'前缀统计: {prefix_count}/{len(mapping)}")
                else:
                    logger.warning(f"无法加载节点类型 {node_type} 的映射")
        else:
            logger.error(f"图构建器不包含date_specific_mappings属性或图构建器为空")

        return date_specific_mappings

    def _get_multidim_item_codes_from_mapping(self, node_type: str, num_nodes: int,
                                              date_specific_mappings: dict) -> list:
        """🔧 方案F: 从映射中直接获取现有的多维度item_codes"""
        logger.info(f"🔍 从现有多维度映射中获取 {node_type} 的item_codes...")

        try:
            if node_type not in date_specific_mappings:
                logger.error(f"❌ {node_type} 映射不存在")
                return None

            mapping = date_specific_mappings[node_type]
            if not mapping:
                logger.error(f"❌ {node_type} 映射为空")
                return None

            # 按值排序，确保索引对应正确
            sorted_items = sorted(mapping.items(), key=lambda x: x[1])

            # 🔧 方案G修复3: 详细执行过程日志
            logger.info(f"🔧 方案F详细执行过程:")
            logger.info(f"   - 映射类型: {type(mapping)}")
            logger.info(f"   - 映射前3项原始: {dict(list(mapping.items())[:3])}")
            logger.info(f"   - 排序后前3项: {sorted_items[:3] if sorted_items else 'None'}")

            # 🔧 方案G修复3: 处理tuple格式的键，转换为多维度字符串
            item_codes = []
            target_count = min(len(sorted_items), num_nodes)

            for i, item in enumerate(sorted_items[:target_count]):
                key = item[0]
                if isinstance(key, tuple):
                    # 将tuple转换为 "|" 分隔的字符串
                    key_str = '|'.join(str(x) for x in key)
                    logger.debug(f"🔍 tuple键转换: {key} → {key_str}")
                    item_codes.append(key_str)
                else:
                    item_codes.append(str(key))

            # 如果映射不足，补齐回退ID
            if len(sorted_items) < num_nodes:
                logger.warning(f"⚠️ 映射大小({len(sorted_items)}) < 需要节点数({num_nodes})")
                for i in range(len(item_codes), num_nodes):
                    fallback_id = f"fallback_multidim_{node_type}_{i}"
                    item_codes.append(fallback_id)

            logger.info(f"   - 转换后前3项: {item_codes[:3] if item_codes else 'None'}")

            # 验证多维度格式
            multidim_count = sum(1 for code in item_codes[:10] if '|' in str(code))
            logger.info(f"✅ 获取了 {len(item_codes)} 个item_codes，其中 {multidim_count}/10 为多维度格式")

            # 显示样例
            sample_codes = item_codes[:5] if len(item_codes) >= 5 else item_codes
            logger.info(f"🔍 多维度item_codes样例: {sample_codes}")

            return item_codes

        except Exception as e:
            logger.error(f"❌ 从多维度映射获取item_codes失败: {e}")
            return None

    def _get_item_codes_with_original_lookup(self, node_type: str, num_nodes: int,
                                             date_specific_mappings: dict, data, node_manager) -> list:
        """🔧 方案D: 从图数据中查找原始多维度ID的逻辑"""
        logger.info(f"🔍 开始从图数据中查找 {node_type} 的原始多维度ID...")

        try:
            item_codes = []

            # 方法1: 尝试从图构建器的原始映射中查找
            if hasattr(self, 'graph_builder') and self.graph_builder:
                original_mappings = getattr(self.graph_builder, 'node_mappings', {})
                if node_type in original_mappings:
                    logger.info(f"🔍 在图构建器中找到 {node_type} 原始映射，大小: {len(original_mappings[node_type])}")

                    # 创建反向映射：从索引到原始ID
                    reverse_mapping = {}
                    for original_id, mapped_idx in original_mappings[node_type].items():
                        reverse_mapping[mapped_idx] = original_id

                    # 根据节点索引查找原始ID
                    for i in range(num_nodes):
                        if i in reverse_mapping:
                            original_id = reverse_mapping[i]
                            item_codes.append(str(original_id))
                        else:
                            item_codes.append(f"MISSING_ORIGINAL_{node_type}_{i}")

                    logger.info(f"✅ 从图构建器映射中获取了 {len(item_codes)} 个原始ID")
                    return item_codes

            # 方法2: 尝试从数据管理器中查找
            if data and hasattr(data, node_type):
                logger.info(f"🔍 尝试从数据管理器中查找 {node_type} 原始ID...")
                node_data = getattr(data, node_type)

                # 如果节点数据有原始ID信息
                if hasattr(node_data, 'node_ids') or hasattr(node_data, 'original_ids'):
                    original_ids = getattr(node_data, 'node_ids', None) or getattr(node_data, 'original_ids', None)
                    if original_ids and len(original_ids) >= num_nodes:
                        item_codes = [str(original_ids[i]) for i in range(num_nodes)]
                        logger.info(f"✅ 从数据管理器中获取了 {len(item_codes)} 个原始ID")
                        return item_codes

            # 方法3: 尝试从节点管理器中查找
            if node_manager and hasattr(node_manager, 'get_original_node_ids'):
                logger.info(f"🔍 尝试从节点管理器中查找 {node_type} 原始ID...")
                original_ids = node_manager.get_original_node_ids(node_type, num_nodes)
                if original_ids:
                    item_codes = [str(id_) for id_ in original_ids]
                    logger.info(f"✅ 从节点管理器中获取了 {len(item_codes)} 个原始ID")
                    return item_codes

            # 方法4: 尝试从date_specific_mappings的键中推断
            if date_specific_mappings and node_type in date_specific_mappings:
                mapping = date_specific_mappings[node_type]
                if mapping:
                    # 获取所有键，按值排序来重建原始顺序
                    sorted_items = sorted(mapping.items(), key=lambda x: x[1])
                    if len(sorted_items) >= num_nodes:
                        item_codes = [str(item[0]) for item in sorted_items[:num_nodes]]
                        logger.info(f"✅ 从映射键中推断了 {len(item_codes)} 个原始ID")
                        return item_codes

            logger.warning(f"⚠️ 所有原始ID查找方法都失败了，返回None")
            return None

        except Exception as e:
            logger.error(f"❌ 原始ID查找过程中出现错误: {e}")
            return None

    def _get_optimized_item_codes(self, node_type: str, num_nodes: int, date_specific_mappings: dict,
                                  data, node_manager) -> list:
        """极速优化的item_code获取方法 - 增强多维度信息保留和错误处理"""

        # 只使用日期特定映射，确保性能和日期隔离
        if node_type in date_specific_mappings:
            mapping_dict = date_specific_mappings[node_type]

            # 记录映射信息
            logger.info(f"为节点类型 {node_type} 准备生成 {num_nodes} 个节点的item_codes")

            # 使用numpy直接构建反向映射数组，避免pandas开销
            item_codes = [None] * num_nodes

            # 记录多维度信息和前缀处理的统计数据
            tuple_codes_count = 0
            multi_dim_str_count = 0
            inventory_prefix_count = 0

            # 直接遍历映射字典，避免中间数据结构
            for code, idx in mapping_dict.items():
                if 0 <= idx < num_nodes:
                    # 快速字符串转换，但保留多维度信息
                    if isinstance(code, tuple):
                        # 保留原始多维度结构，使用更有意义的分隔符
                        item_codes[idx] = '/'.join(map(str, code))
                        tuple_codes_count += 1
                    else:
                        # 检查字符串形式的多维度编码 (使用 | 分隔符)
                        str_code = str(code)
                        if '|' in str_code:
                            # 处理多维度字符串
                            parts = str_code.split('|')

                            # 根据节点类型处理多维度信息
                            if node_type == 'virtual' and len(parts) >= 4:
                                # virtual: assembly|component|period_code|plant_code
                                assembly = parts[0]
                                component = parts[1]
                                # 只使用原始的assembly和component作为item_code
                                item_codes[idx] = f"{assembly}/{component}"
                                multi_dim_str_count += 1
                            elif (node_type == 'production' or node_type == 'inventory') and len(parts) >= 1:
                                # production/inventory: item_code|period_code|plant_code
                                # 只使用原始的item_code
                                item_code = parts[0]
                                item_codes[idx] = item_code
                                multi_dim_str_count += 1
                            else:
                                # 其他多维度格式
                                # 为安全起见，使用第一个维度作为item_code
                                item_codes[idx] = parts[0] if parts else '/'.join(parts)
                                multi_dim_str_count += 1
                        else:
                            # 移除inventory_前缀（如果存在）
                            if node_type == 'inventory' and str_code.startswith('inventory_'):
                                str_code = str_code[10:]  # 移除'inventory_'前缀
                                inventory_prefix_count += 1
                            item_codes[idx] = str_code

            # 记录处理结果
            total_multi_dim = tuple_codes_count + multi_dim_str_count
            logger.info(
                f"节点类型 {node_type}: 处理了 {total_multi_dim} 个多维度代码 (元组格式: {tuple_codes_count}, 字符串格式: {multi_dim_str_count})")
            if node_type == 'inventory':
                logger.info(f"移除了 {inventory_prefix_count} 个'inventory_'前缀")

            # 一次性填充所有缺失值并记录
            missing_count = 0
            for i in range(num_nodes):
                if item_codes[i] is None:
                    item_codes[i] = f"MISSING_DATE_MAP_{node_type}_{i}"
                    missing_count += 1

            if missing_count > 0:
                logger.warning(f"节点类型 {node_type}: {missing_count}/{num_nodes} 节点缺失映射")

            # 记录样本
            sample_size = min(5, num_nodes)
            if sample_size > 0:
                samples = item_codes[:sample_size]
                logger.info(f"节点类型 {node_type} item_codes 样本: {samples}")

            return item_codes

        # 如果没有日期映射，说明配置有问题，应该立即报错而不是回退
        logger.error(f"❌ 日期映射缺失 {node_type}，检查graph_builder配置")
        return [f"NO_DATE_MAPPING_{node_type}_{i}" for i in range(num_nodes)]
