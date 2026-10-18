import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import HeteroData
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
import logging
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import inspect
import gc
import time

from .gcn_model import SupplyChainGCN
from .prediction_handler import PredictionHandler
from ..config.model_config import ModelConfig
from ..config.path_config import path_config

logger = logging.getLogger(__name__)


class ConfigSerializer:
    """配置序列化器"""

    @staticmethod
    def serialize_config(config) -> dict:
        """序列化配置"""
        if hasattr(config, 'to_dict') and callable(getattr(config, 'to_dict')):
            try:
                result = config.to_dict()
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        config_dict = {
            'config_type': str(type(config).__name__),
            'config_module': str(type(config).__module__),
            'timestamp': datetime.now().isoformat()
        }

        ConfigSerializer._serialize_basic_attrs(config, config_dict)
        ConfigSerializer._serialize_sub_config(config, config_dict, 'device_config', ['preferred_device'])
        ConfigSerializer._serialize_sub_config(config, config_dict, 'training_config', 
            ['num_epochs', 'learning_rate', 'batch_size', 'weight_decay', 'patience'])
        ConfigSerializer._serialize_sub_config(config, config_dict, 'architecture_config', 
            ['global_hidden_dim', 'num_conv_layers'])
        ConfigSerializer._serialize_layer_configs(config, config_dict)
        ConfigSerializer._serialize_edge_configs(config, config_dict)
        
        if hasattr(config, 'edge_types'):
            try:
                config_dict['edge_types'] = [str(et) for et in getattr(config, 'edge_types', [])]
            except Exception:
                config_dict['edge_types'] = []

        for attr in ['experiment_name', 'save_dir']:
            if hasattr(config, attr):
                try:
                    value = getattr(config, attr)
                    config_dict[attr] = str(value) if hasattr(value, '__fspath__') else value
                except Exception:
                    config_dict[attr] = None

        return config_dict

    @staticmethod
    def _serialize_basic_attrs(config, config_dict):
        """序列化基本属性"""
        for attr in ['max_layer']:
            if hasattr(config, attr):
                try:
                    config_dict[attr] = getattr(config, attr)
                except Exception:
                    config_dict[attr] = None

    @staticmethod
    def _serialize_sub_config(config, config_dict, config_name: str, attrs: List[str]):
        """序列化子配置"""
        if hasattr(config, config_name) and getattr(config, config_name):
            try:
                sub_config = getattr(config, config_name)
                config_dict[config_name] = {attr: getattr(sub_config, attr, 0) for attr in attrs}
            except Exception:
                config_dict[config_name] = {attr: 0 for attr in attrs}

    @staticmethod
    def _serialize_layer_configs(config, config_dict):
        """序列化层级配置"""
        if hasattr(config, 'layer_configs'):
            try:
                config_dict['layer_configs'] = {}
                for layer_id, layer_config in getattr(config, 'layer_configs', {}).items():
                    if layer_config:
                        config_dict['layer_configs'][str(layer_id)] = {
                            'hidden_dim': getattr(layer_config, 'hidden_dim', 0),
                            'dropout_rate': getattr(layer_config, 'dropout_rate', 0.0),
                            'batch_norm': getattr(layer_config, 'batch_norm', True),
                            'activation': getattr(layer_config, 'activation', 'ReLU')
                        }
            except Exception:
                config_dict['layer_configs'] = {}

    @staticmethod
    def _serialize_edge_configs(config, config_dict):
        """序列化边类型配置"""
        if hasattr(config, 'edge_configs'):
            try:
                config_dict['edge_configs'] = {}
                for edge_type, edge_config in getattr(config, 'edge_configs', {}).items():
                    if edge_config:
                        edge_key = str(edge_type) if not isinstance(edge_type, str) else edge_type
                        config_dict['edge_configs'][edge_key] = {
                            'hidden_dim': getattr(edge_config, 'hidden_dim', 0),
                            'dropout_rate': getattr(edge_config, 'dropout_rate', 0.0),
                            'batch_norm': getattr(edge_config, 'batch_norm', True),
                            'activation': getattr(edge_config, 'activation', 'ReLU')
                        }
            except Exception:
                config_dict['edge_configs'] = {}

    @staticmethod
    def validate_and_fix_config(config) -> object:
        """验证并修复配置对象"""
        logger.info("Validating config object")

        from ..config.model_config import DeviceConfig, TrainingConfig, ModelArchitectureConfig
        
        config_map = {
            'device_config': DeviceConfig,
            'training_config': TrainingConfig,
            'architecture_config': ModelArchitectureConfig
        }
        
        for attr, config_class in config_map.items():
            if not hasattr(config, attr):
                logger.warning(f"Config missing {attr}, creating default")
                setattr(config, attr, config_class())

        return config


class TrainingManager:
    """管理模型训练、验证和测试的统一接口"""

    def __init__(self, model: SupplyChainGCN, config: ModelConfig,
                 experiment_name: str = None, node_manager=None, graph_builder=None,
                 use_distributed: bool = None, local_rank: Optional[int] = None, world_size: Optional[int] = None):
        self.model = model
        self.node_manager = node_manager
        self.graph_builder = graph_builder
        self.config = ConfigSerializer.validate_and_fix_config(config)
        
        self._init_distributed_config(env_world=os.environ.get('WORLD_SIZE', '1'),
                                      env_local_rank=os.environ.get('LOCAL_RANK'),
                                      env_rank=os.environ.get('RANK'),
                                      use_distributed=use_distributed,
                                      world_size=world_size,
                                      local_rank=local_rank)
        
        self._init_device()
        self._init_experiment_name(experiment_name)
        self._init_training_state()
        self._init_amp()
        
        if self.use_distributed:
            self._init_distributed()
        
        self.prediction_handler = PredictionHandler(self.save_dir, self.rank, self.world_size)

    def _init_distributed_config(self, env_world: str, env_local_rank: str, env_rank: str,
                                 use_distributed: bool, world_size: Optional[int], local_rank: Optional[int]):
        """初始化分布式配置"""
        env_world_val = int(env_world)
        if use_distributed is None:
            use_distributed = env_world_val > 1
        self.use_distributed = use_distributed
        self.world_size = world_size if world_size is not None else (env_world_val if use_distributed else 1)
        self.local_rank = local_rank if local_rank is not None else (
            int(env_local_rank) if env_local_rank is not None else 0)

    def _init_device(self):
        """初始化设备"""
        force_cuda = os.environ.get('COMMOPT_FORCE_CUDA', '0') == '1'
        strict_device = os.environ.get('COMMOPT_STRICT_DEVICE', '0') == '1'
        try:
            # 优先使用配置中指定的设备类型
            preferred = getattr(self.config.device_config, 'preferred_device', 'auto')
            has_npu = False
            if not force_cuda and hasattr(torch, 'npu'):
                try:
                    has_npu = bool(torch.npu.is_available())
                except Exception as npu_error:
                    logger.warning(f"NPU availability check failed; continuing with CUDA check: {npu_error}")
            has_cuda = torch.cuda.is_available()
            cuda_device_id = self.local_rank
            cuda_device_map = os.environ.get('COMMOPT_CUDA_DEVICE_MAP', '').strip()
            if cuda_device_map:
                try:
                    mapped_ids = [
                        int(value.strip())
                        for value in cuda_device_map.split(',')
                        if value.strip()
                    ]
                except ValueError as map_error:
                    raise RuntimeError(
                        f"Invalid COMMOPT_CUDA_DEVICE_MAP={cuda_device_map!r}"
                    ) from map_error
                if self.local_rank < 0 or self.local_rank >= len(mapped_ids):
                    raise RuntimeError(
                        f"LOCAL_RANK={self.local_rank} has no entry in "
                        f"COMMOPT_CUDA_DEVICE_MAP={cuda_device_map!r}"
                    )
                cuda_device_id = mapped_ids[self.local_rank]
            self.cuda_device_id = cuda_device_id

            if force_cuda:
                visible_count = int(torch.cuda.device_count())
                if cuda_device_id < 0 or cuda_device_id >= visible_count:
                    raise RuntimeError(
                        f"Mapped CUDA device {cuda_device_id} is outside "
                        f"{visible_count} devices; LOCAL_RANK={self.local_rank}, "
                        f"COMMOPT_CUDA_DEVICE_MAP={cuda_device_map!r}"
                    )
                torch.cuda.set_device(cuda_device_id)
                self.device = torch.device(f"cuda:{cuda_device_id}")
                # Force eager initialization here.  A failure must be reported
                # before NCCL is created; silently retaining CPU parameters
                # would only fail much later inside DDP with a misleading
                # "No backend type associated with device type cpu" error.
                probe = torch.ones(1, device=self.device)
                torch.cuda.synchronize(cuda_device_id)
                del probe
                logger.info(f"TrainingManager forced CUDA: {self.device}")
                return
            
            if self.use_distributed:
                if preferred == 'cuda' and has_cuda:
                    torch.cuda.set_device(cuda_device_id)
                    self.device = torch.device(f"cuda:{cuda_device_id}")
                    logger.info(f"TrainingManager using CUDA: cuda:{cuda_device_id}")
                elif preferred in ('auto', 'npu') and has_npu:
                    torch.npu.set_device(self.local_rank)
                    self.device = torch.device(f"npu:{self.local_rank}")
                    logger.info(f"TrainingManager using NPU: npu:{self.local_rank}")
                elif has_cuda:
                    torch.cuda.set_device(cuda_device_id)
                    self.device = torch.device(f"cuda:{cuda_device_id}")
                    logger.info(f"TrainingManager using CUDA: cuda:{cuda_device_id}")
                else:
                    self.device = torch.device("cpu")
                    logger.info(f"TrainingManager using CPU")
            else:
                # 非分布式模式也遵循 preferred_device 设置
                if preferred == 'cuda' and has_cuda:
                    self.device = self.config.device_config.get_device()
                    logger.info(f"TrainingManager target device: {self.device} (forced CUDA)")
                elif preferred in ('auto', 'npu') and has_npu:
                    self.device = self.config.device_config.get_device()
                    logger.info(f"TrainingManager target device: {self.device}")
                elif has_cuda:
                    self.device = self.config.device_config.get_device()
                    logger.info(f"TrainingManager target device: {self.device}")
                else:
                    self.device = torch.device("cpu")
                    logger.info(f"TrainingManager using CPU (fallback)")
        except Exception as e:
            if force_cuda or strict_device:
                raise RuntimeError(
                    f"Device initialization failed in strict mode: {e}"
                ) from e
            logger.warning(f"Device initialization failed: {e}, using CPU")
            self.device = torch.device("cpu")

    def _init_experiment_name(self, experiment_name: str = None):
        """初始化实验名称"""
        if experiment_name is None:
            if self.use_distributed and dist.is_initialized():
                current_rank = dist.get_rank()
                experiment_name = self._sync_experiment_name_across_ranks(current_rank)
            else:
                experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.save_dir = path_config.get_result_path(experiment_name)
        logger.info(f"[Rank {dist.get_rank() if (self.use_distributed and dist.is_initialized()) else 0}] save_dir: {self.save_dir}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        time.sleep(0.5)

    def _sync_experiment_name_across_ranks(self, current_rank: int) -> str:
        """跨rank同步实验名称"""
        sync_file = path_config.get_dist_temp_dir() / '_experiment_name_sync.txt'
        
        if current_rank == 0:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            sync_file.parent.mkdir(parents=True, exist_ok=True)
            with open(sync_file, 'w') as f:
                f.write(experiment_name)
                f.flush()
            os.fsync(f.fileno())
            logger.info(f"[Rank 0] experiment_name: {experiment_name}")
        else:
            max_retries = 60
            experiment_name = None
            for retry in range(max_retries):
                if sync_file.exists():
                    try:
                        with open(sync_file, 'r') as f:
                            experiment_name = f.read().strip()
                        if experiment_name:
                            logger.info(f"[Rank {current_rank}] 读取 experiment_name: {experiment_name}")
                            break
                    except Exception as e:
                        logger.warning(f"[Rank {current_rank}] 读取同步文件失败: {e}")
                time.sleep(0.5)
            else:
                logger.error(f"[Rank {current_rank}] 无法读取 experiment_name")
                experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        time.sleep(2)
        if current_rank == 0:
            time.sleep(3)
            if sync_file.exists():
                try:
                    sync_file.unlink()
                except:
                    pass
        
        return experiment_name

    def _init_training_state(self):
        """初始化训练状态"""
        self.optimizer = None
        self.scheduler = None
        self.best_model_path = self.save_dir / 'best_model.pth'
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'train_metrics': defaultdict(list),
            'val_metrics': defaultdict(list),
            'validation_metrics': defaultdict(list),
            'test_metrics': defaultdict(list)
        }

    def _init_amp(self):
        """初始化混合精度训练"""
        self.use_amp = False
        self.amp_scaler = None
        self.amp_dtype = torch.float32
        self._amp_device_type = None
        
        try:
            if self.device.type == 'npu':
                self._amp_device_type = 'npu'
                # NPU: 优先尝试 bfloat16（安全、不需要 GradScaler）
                try:
                    test_t = torch.randn(2, 2, device=self.device, dtype=torch.bfloat16)
                    _ = test_t @ test_t
                    del test_t
                    self.use_amp = True
                    self.amp_dtype = torch.bfloat16
                    self.amp_scaler = None  # bfloat16 不需要 GradScaler
                    logger.info(f"混合精度训练(AMP)已启用: device_type=npu, dtype=bfloat16 (无需GradScaler)")
                except Exception as e_bf16:
                    # bfloat16 不支持，禁用 AMP（不使用 float16，因为 GCN 在 float16 下会 NaN）
                    self.use_amp = False
                    self._amp_device_type = None
                    logger.warning(f"NPU不支持bfloat16: {e_bf16}")
                    logger.warning(f"禁用AMP（float16在GCN模型中会因溢出产生NaN），使用FP32训练")
            elif self.device.type == 'cuda':
                self._amp_device_type = 'cuda'
                self.use_amp = True
                self.amp_dtype = torch.float16
                self.amp_scaler = torch.amp.GradScaler('cuda')
                logger.info(f"混合精度训练(AMP)已启用: device_type=cuda, dtype=float16 + GradScaler")
            else:
                self.use_amp = False
                logger.info(f"CPU设备，不启用混合精度训练")
        except Exception as e:
            self.use_amp = False
            self.amp_scaler = None
            self._amp_device_type = None
            logger.warning(f"混合精度训练初始化失败: {e}，将使用 FP32 训练")

    def _init_distributed(self):
        """初始化分布式训练"""
        strict_device = os.environ.get('COMMOPT_STRICT_DEVICE', '0') == '1'
        try:
            logger.info(f"Preparing model on device: {self.device}")
            self.model.to(self.device)

            if dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                logger.info(f"分布式已由外部初始化. rank={self.rank}, world={self.world_size}")
            else:
                env_rank = os.environ.get('RANK')
                has_npu = hasattr(torch, 'npu') and torch.npu.is_available()
                if has_npu:
                    backend = 'hccl'
                    logger.info("Using HCCL backend for NPU distributed training")
                elif torch.cuda.is_available():
                    backend = 'nccl'
                    logger.info("Using NCCL backend for CUDA distributed training")
                else:
                    backend = 'gloo'
                    logger.info("Using GLOO backend for CPU distributed training")
                
                dist.init_process_group(backend=backend, init_method='env://',
                                       world_size=self.world_size,
                                       rank=int(env_rank) if env_rank is not None else self.local_rank)
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                logger.info(f"DDP initialized. rank={self.rank}, world={self.world_size}")

            logger.info("DDP包装将在设置样本数据后进行")
        except Exception as e:
            if strict_device:
                logger.exception(
                    f"Strict distributed device initialization failed on {self.device}"
                )
                raise
            logger.warning(f"Failed to prepare model on {self.device}: {e}")
            self.device = torch.device("cpu")
            self.model.to(self.device)
            self.rank = 0

    def _prepare_model_for_ddp(self):
        """在DDP封装前准备模型：预创建边预测器，完成懒初始化"""
        logger.info("开始准备模型用于DDP封装...")

        if hasattr(self, '_sample_data_for_initialization') and self._sample_data_for_initialization is not None:
            sample_data = self._sample_data_for_initialization
        else:
            logger.warning("没有样本数据，跳过边预测器预创建")
            return

        try:
            sample_data = sample_data.to(self.device)

            if hasattr(self.model, 'precreate_all_edge_predictors_from_data'):
                created_count = self.model.precreate_all_edge_predictors_from_data(sample_data)
                logger.info(f"预创建了 {created_count} 个边预测器")

            logger.info("执行初始化前向传播...")
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, 'forward_for_initialization'):
                    init_outputs = self.model.forward_for_initialization(sample_data)
                else:
                    init_outputs = self.model(sample_data)

            if hasattr(self.model, 'edge_predictor'):
                logger.info(f"边预测器已创建: {list(self.model.edge_predictor.keys())}")

            logger.info("模型DDP准备完成")

        except Exception as e:
            logger.error(f"模型DDP准备失败: {e}")
            raise e

    def set_sample_data_for_initialization(self, sample_data):
        """
        设置用于模型初始化的样本数据
        应在TrainingManager初始化后、训练开始前调用
        """
        self._sample_data_for_initialization = sample_data
        logger.info("已设置样本数据用于模型初始化")

        if (
            self.use_distributed
            and dist.is_initialized()
            and "nccl" in str(dist.get_backend()).lower()
            and self.device.type != "cuda"
        ):
            raise RuntimeError(
                "NCCL process group cannot wrap a CPU model: "
                f"device={self.device}, local_rank={self.local_rank}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}. "
                "The rank-local CUDA binding failed before DDP construction."
            )

        # 立即进行模型DDP准备（如果模型还没有被DDP包装）
        if not isinstance(self.model, DDP):
            logger.info("检测到模型尚未DDP包装，开始进行DDP准备...")
            self._prepare_model_for_ddp()

            # 如果使用分布式，现在进行DDP包装
            if self.use_distributed:
                logger.info("开始DDP包装...")
                # NPU适配：根据设备类型设置 device_ids 和 output_device
                # NPU 和 CUDA 都需要指定 device_ids，CPU 则设为 None
                if self.device.type in ('cuda', 'npu'):
                    device_id = self.device.index
                    device_ids = [device_id]
                    output_device = device_id
                else:
                    device_ids = None
                    output_device = None
                
                self.model = DDP(
                    self.model,
                    device_ids=device_ids,
                    output_device=output_device,
                    find_unused_parameters=True,
                    gradient_as_bucket_view=True
                )
                # 启用静态图优化
                try:
                    if hasattr(self.model, '_set_static_graph'):
                        self.model._set_static_graph()
                        logger.info("DDP static graph 已启用")
                except Exception as _:
                    pass
                logger.info("DDP包装完成")
            else:
                logger.info("非分布式模式，模型准备完成")
        else:
            logger.warning("模型已经被DDP包装，跳过重复包装")

        # 确保分布式初始化完成后创建预测处理器
        try:
            self.prediction_handler = PredictionHandler(self.save_dir, self.rank, self.world_size)
            logger.info(f"PredictionHandler initialized for rank {self.rank}")
        except Exception as e:
            logger.error(f"Failed to initialize PredictionHandler: {e}")
            # 创建一个简单的占位实例，防止属性缺失
            self.prediction_handler = None

    def setup_training(self):
        """设置训练环境"""
        logger.info("Setting up training environment...")

        self._validate_config()
        self._log_training_parameters()

        self.model = self.model.to(self.device) if not isinstance(self.model, DDP) else self.model
        model_device = next(self.model.parameters()).device

        if str(model_device) != str(self.device):
            logger.warning(f"Device mismatch! Expected {self.device}, but model is on {model_device}")
            try:
                self.model = self.model.to(self.device)
            except Exception as e:
                logger.error(f"Failed to fix device mismatch: {e}")

        final_device = next(self.model.parameters()).device
        if final_device.type == 'cuda':
            try:
                _ = sum(p.norm() for p in self.model.parameters())
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"GPU training readiness test failed: {e}")

        self._setup_optimizer()
        self._setup_scheduler()

        logger.info("Training environment setup completed")

    def _validate_config(self):
        """验证配置的完整性和一致性"""
        logger.info("Validating configuration...")

        # 检查必要的配置属性
        required_attrs = [
            ('training_config', 'num_epochs'),
            ('training_config', 'learning_rate'),
            ('training_config', 'patience'),
            ('training_config', 'weight_decay'),
            ('architecture_config', 'global_hidden_dim'),
            ('architecture_config', 'num_conv_layers')
        ]

        for config_section, attr in required_attrs:
            if not hasattr(self.config, config_section):
                raise ValueError(f"Missing configuration section: {config_section}")

            config_obj = getattr(self.config, config_section)
            if not hasattr(config_obj, attr):
                raise ValueError(f"Missing configuration attribute: {config_section}.{attr}")

        logger.info("Configuration validation passed")

    def _log_training_parameters(self):
        """记录实际使用的训练参数"""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING PARAMETERS (实际使用的参数)")
        logger.info("=" * 60)

        # 训练配置
        training_config = self.config.training_config
        logger.info("训练配置:")
        logger.info(f"  训练轮数 (num_epochs): {training_config.num_epochs}")
        logger.info(f"  学习率 (learning_rate): {training_config.learning_rate}")
        logger.info(f"  早停耐心值 (patience): {training_config.patience}")
        logger.info(f"  权重衰减 (weight_decay): {training_config.weight_decay}")
        logger.info(f"  使用传统GCN消息传递机制 (Traditional GCN Message Passing)")

        # 架构配置
        arch_config = self.config.architecture_config
        logger.info("\n模型架构配置:")
        logger.info(f"  全局隐藏维度 (global_hidden_dim): {arch_config.global_hidden_dim}")
        logger.info(f"  卷积层数 (num_conv_layers): {arch_config.num_conv_layers}")

        # 设备配置
        logger.info(f"\n设备配置:")
        logger.info(f"  使用设备: {self.device}")

        # 模型信息（不统计参数，避免LazyModule问题）
        logger.info(f"\n模型信息:")
        logger.info(f"  模型类型: {self.model.__class__.__name__}")
        logger.info(f"  模型已初始化: {'是' if hasattr(self.model, '_parameters_initialized') else '否'}")

        logger.info("=" * 60)

    def _setup_optimizer(self):
        """设置优化器"""
        training_config = self.config.training_config

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )

        logger.info(f"Optimizer setup: Adam(lr={training_config.learning_rate}, "
                    f"weight_decay={training_config.weight_decay})")

    def _setup_scheduler(self):
        """设置学习率调度器"""
        training_config = self.config.training_config

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=training_config.patience // 2,
            # verbose=True,
            min_lr=1e-6
        )

        logger.info(f"Scheduler setup: ReduceLROnPlateau(patience={training_config.patience // 2})")

    def _move_data_to_device(self, data: HeteroData) -> HeteroData:
        """安全地将数据移动到指定设备"""
        try:
            return data.to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"CUDA out of memory, clearing cache and retrying: {e}")
                self._clear_memory()
                try:
                    return data.to(self.device)
                except RuntimeError as e2:
                    logger.error(f"Still out of memory after clearing cache, switching to CPU: {e2}")
                    self._switch_to_cpu()
                    return data.to(self.device)
            else:
                logger.warning(f"Failed to move data to {self.device}: {e}")
                # 如果移动失败，尝试CPU
                if self.device != torch.device("cpu"):
                    logger.info("Trying to move data to CPU")
                    self._switch_to_cpu()
                    return data.to(self.device)
                else:
                    raise e
        except Exception as e:
            logger.warning(f"Unexpected error moving data to {self.device}: {e}")
            if self.device != torch.device("cpu"):
                logger.info("Trying to move data to CPU")
                self._switch_to_cpu()
                return data.to(self.device)
            else:
                raise e

    def _clear_memory(self):
        """清理内存"""
        if self.device.type == 'npu':
            try:
                torch.npu.synchronize(self.device)
                torch.npu.empty_cache()
            except Exception:
                pass
        elif self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        gc.collect()
        
    def _handle_oom_error(self, error_msg: str):
        """处理OOM错误"""
        logger.error(f"OOM Error: {error_msg}")
        self._clear_memory()
        
        if self.use_distributed and dist.is_initialized():
            try:
                oom_signal = torch.tensor([1], dtype=torch.int32, device=self.device)
                dist.all_reduce(oom_signal, op=dist.ReduceOp.SUM)
            except Exception:
                pass
        
        logger.info("Consider reducing batch size or enabling gradient accumulation")

    def _switch_to_cpu(self):
        """切换到CPU"""
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self._clear_memory()
        logger.warning("Switched to CPU due to memory constraints")

    def _get_memory_info(self) -> Dict[str, float]:
        """获取内存使用信息"""
        info = {}
        if self.device.type == 'npu':
            try:
                info['npu_allocated_gb'] = torch.npu.memory_allocated(self.device) / (1024 ** 3)
                info['npu_reserved_gb'] = torch.npu.memory_reserved(self.device) / (1024 ** 3)
            except Exception:
                pass
        elif self.device.type == 'cuda':
            info['gpu_allocated_gb'] = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            info['gpu_reserved_gb'] = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            info['gpu_free_gb'] = (torch.cuda.get_device_properties(self.device).total_memory -
                                  torch.cuda.memory_reserved(self.device)) / (1024 ** 3)
        return info

    def train_epoch(self, graphs: List[HeteroData]) -> float:
        """训练一个epoch"""
        self.model.train()
        model_device = next(self.model.parameters()).device

        # 分布式数据分配
        if self.use_distributed and self.world_size > 1:
            local_graphs = self._distribute_graphs_balanced(graphs)
        else:
            local_graphs = graphs

        total_loss = 0
        successful_batches = 0

        has_data = len(local_graphs) > 0
        
        # DDP同步
        if self.use_distributed and dist.is_initialized():
            try:
                data_count = torch.tensor([1.0 if has_data else 0.0], dtype=torch.float32, device=self.device)
                dist.all_reduce(data_count, op=dist.ReduceOp.SUM)
                ranks_with_data = int(data_count.item())
                
                if ranks_with_data == 0 and has_data:
                    ranks_with_data = 1
                    
                if ranks_with_data == 0 and not has_data:
                    self._execute_dummy_step()
                    return float('inf')
            except Exception as e:
                logger.warning(f"Data state collection failed: {e}")
        
        if self.use_distributed and not has_data:
            try:
                self._execute_dummy_step()
            except Exception:
                pass

        for i, graph in enumerate(local_graphs):
            try:
                graph = self._move_data_to_device(graph)
                self.optimizer.zero_grad(set_to_none=True)
                
                # 混合精度训练：autocast 只包裹前向传播（模型推理）
                if self.use_amp and self._amp_device_type is not None:
                    with torch.amp.autocast(device_type=self._amp_device_type, dtype=self.amp_dtype):
                        predictions = self.model(graph)
                    
                    # NaN 自动检测：如果 AMP 导致预测结果出现 NaN，自动禁用 AMP 并重试
                    has_nan = False
                    if predictions:
                        for k, v in predictions.items():
                            if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                                has_nan = True
                                break
                    
                    if has_nan or not predictions:
                        nan_msg = ", NaN" if has_nan else ", 空"
                        logger.warning(f"AMP 导致预测结果异常{nan_msg}, 自动禁用 AMP 并用 FP32 重试")
                        self.use_amp = False
                        self.amp_scaler = None
                        self._amp_device_type = None
                        predictions = self.model(graph)
                    
                    if not predictions:
                        logger.warning(f"Empty predictions for graph {i}, skipping")
                        continue
                    
                    # 损失计算在 autocast 外部以 FP32 执行
                    predictions_fp32 = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in predictions.items()}
                    if hasattr(self.model, 'module'):
                        loss = self.model.module.compute_loss(predictions_fp32, graph)
                    else:
                        loss = self.model.compute_loss(predictions_fp32, graph)
                else:
                    predictions = self.model(graph)
                    
                    if not predictions:
                        logger.warning(f"Empty predictions for graph {i}, skipping")
                        continue
                    
                    if hasattr(self.model, 'module'):
                        loss = self.model.module.compute_loss(predictions, graph)
                    else:
                        loss = self.model.compute_loss(predictions, graph)

                # 检查损失是否有效
                loss_value = loss.item()
                if math.isnan(loss_value) or math.isinf(loss_value):
                    logger.warning(f"Invalid loss for graph {i}: {loss_value}, skipping")
                    continue

                # 反向传播 + 优化器更新
                if self.use_amp and self.amp_scaler is not None:
                    # float16 + GradScaler 路径（仅 CUDA）
                    self.amp_scaler.scale(loss).backward()
                    self.amp_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    # bfloat16（无需 GradScaler）或 FP32 路径
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                total_loss += loss.item()
                successful_batches += 1

                # 每8个batch记录一次设备状态
                amp_scale_info = ""
                if self.use_amp and self.amp_scaler is not None:
                    amp_scale_info = f", AMP scale: {self.amp_scaler.get_scale():.0f}"
                elif self.use_amp:
                    amp_scale_info = f", AMP: {self.amp_dtype}"
                if model_device.type == 'npu' and i % 8 == 0:
                    try:
                        logger.info(
                            f"[Rank {self.rank}] Batch {i}: Loss {loss.item():.6f}, 显存: {torch.npu.memory_allocated(model_device) / 1024 ** 3:.2f}GB{amp_scale_info}")
                    except Exception:
                        pass
                elif model_device.type == 'cuda' and i % 8 == 0:
                    try:
                        logger.info(
                            f"[Rank {self.rank}] Batch {i}: Loss {loss.item():.6f}, 显存: {torch.cuda.memory_allocated(model_device) / 1024 ** 3:.2f}GB{amp_scale_info}")
                    except Exception:
                        pass

            except Exception as e:
                if 'out of memory' in str(e).lower() or 'hip out of memory' in str(e).lower():
                    self._handle_oom_error(str(e))
                else:
                    logger.warning(f"Error processing graph {i}: {e}")
                
                if self.use_distributed and dist.is_initialized():
                    try:
                        fail_flag = torch.tensor([1], dtype=torch.int32, device=self.device)
                        dist.all_reduce(fail_flag, op=dist.ReduceOp.SUM)
                    except Exception:
                        pass
                continue

        if successful_batches == 0:
            logger.error("No successful batches in this epoch!")
            return float('inf')

        # 本地平均
        denom = max(1, successful_batches)
        avg_loss = total_loss / denom

        # DDP同步
        if self.use_distributed and dist.is_initialized():
            try:
                dist.barrier()
                
                if torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss)):
                    avg_loss = 0.0
                
                loss_tensor = torch.tensor(avg_loss, device=self.device, dtype=torch.float32)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = (loss_tensor / self.world_size).item()
                
            except Exception as e:
                logger.warning(f"DDP synchronization failed: {e}")

        return avg_loss

    def _execute_dummy_step(self):
        """执行dummy训练步骤以保持DDP同步"""
        try:
            self.optimizer.zero_grad(set_to_none=True)
            
            # 创建一个小的dummy损失
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # 确保所有参数都参与梯度计算（方案B：dummy step 无需 autocast）
            for param in self.model.parameters():
                if param.requires_grad:
                    dummy_loss = dummy_loss + (param.sum() * 0.0)
            
            # 反向传播 + 优化器步骤（全程 float32）
            dummy_loss.backward()
            self.optimizer.step()
            
            logger.debug(f"[Rank {self.rank}] Dummy step completed")
            
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Dummy step failed: {e}")
            raise

    def _distribute_graphs_balanced(self, graphs: List[HeteroData]) -> List[HeteroData]:
        """改进的分布式图分配策略，确保所有GPU都有合理的工作负载"""
        if not graphs:
            return []
        
        if len(graphs) >= self.world_size:
            return graphs[self.rank::self.world_size]
        
        if len(graphs) == 1:
            enable_partition = bool(getattr(self.config.training_config, 'enable_graph_partition', True))
            
            if enable_partition:
                try:
                    partition_strategy = getattr(self.config.training_config, 'partition_strategy', 'balanced_edge')
                    
                    if partition_strategy == 'all_edges':
                        subgraph, mapping_info = self._partition_all_edges_balanced(
                            graphs[0], self.world_size, self.rank
                        )
                    elif partition_strategy == 'balanced_edge':
                        subgraph, mapping_info = self._partition_graph_by_edges_balanced(
                            graphs[0], self.world_size, self.rank
                        )
                    elif partition_strategy == 'edge':
                        subgraph, mapping_info = self._partition_graph_by_edges_for_rank(
                            graphs[0], self.world_size, self.rank
                        )
                    else:
                        subgraph, mapping_info = self._partition_graph_for_rank(
                            graphs[0], self.world_size, self.rank
                        )
                    
                    if self._validate_subgraph(subgraph):
                        self._current_mapping_info = mapping_info
                        self._original_graph = graphs[0]
                        self.use_graph_partition = True
                        return [subgraph]
                        
                except Exception as e:
                    logger.warning(f"Graph partition failed: {e}, falling back to replication")
            
            return graphs
        
        local_graphs = []
        for i in range(self.rank, len(graphs), self.world_size):
            local_graphs.append(graphs[i])
        
        if not local_graphs and graphs:
            local_graphs = [graphs[-1]]
        
        return local_graphs
    
    def _validate_subgraph(self, subgraph: HeteroData) -> bool:
        """验证子图是否有效（非空且有边）"""
        try:
            has_nodes = False
            for node_type in subgraph.node_types:
                if hasattr(subgraph[node_type], 'x') and subgraph[node_type].x.size(0) > 0:
                    has_nodes = True
                    break
                elif hasattr(subgraph[node_type], 'num_nodes') and subgraph[node_type].num_nodes > 0:
                    has_nodes = True
                    break
            
            has_edges = False
            for edge_type in subgraph.edge_types:
                if hasattr(subgraph[edge_type], 'edge_index') and subgraph[edge_type].edge_index.size(1) > 0:
                    has_edges = True
                    break
            
            return has_nodes and has_edges
        except Exception as e:
            logger.warning(f"子图验证失败: {e}")
            return False
    
    def _partition_graph_by_edges_balanced(self, graph: HeteroData, world_size: int, rank: int) -> Tuple[HeteroData, Dict]:
        """改进的边均衡划分策略，只对标签边进行均匀分配，保留所有节点"""
        try:
            sub = graph.clone()
            self._init_empty_edge_stores(sub, graph)
            
            edge_types_with_labels = [
                ('inventory', 'to', 'virtual'), 
                ('inventory', 'to_transport_labeled', 'inventory')
            ]
            node_info = self._collect_node_info(graph)
            edge_counts = self._count_labeled_edges(graph, edge_types_with_labels)
            total_edges_all_types = sum(edge_counts.values())
            
            if total_edges_all_types == 0:
                return self._partition_graph_by_edges_for_rank(graph, world_size, rank)
            
            edge_eids_mapping, edge_type_allocation = self._allocate_labeled_edges(
                sub, graph, node_info, edge_types_with_labels, edge_counts, world_size, rank)
            
            if sum(edge_type_allocation.values()) == 0 and total_edges_all_types > 0:
                self._force_allocate_minimal_edges(
                    sub, graph, edge_types_with_labels, edge_counts, world_size, rank,
                    edge_eids_mapping, edge_type_allocation)
            
            mapping_info = {
                'rank': rank,
                'world_size': world_size,
                'edge_eids': edge_eids_mapping,
                'node_info': node_info,
                'partition_strategy': 'balanced_edge_only'
            }
            return sub, mapping_info
            
        except Exception as e:
            logger.error(f"[Rank {rank}] 均衡边划分失败: {e}")
            return self._partition_graph_by_edges_for_rank(graph, world_size, rank)

    def _init_empty_edge_stores(self, sub: HeteroData, graph: HeteroData):
        """初始化空边存储"""
        for edge_type in sub.edge_types:
            if not hasattr(sub[edge_type], 'edge_index') or sub[edge_type].edge_index is None:
                device = torch.device('cpu')
                for et in sub.edge_types:
                    if hasattr(sub[et], 'edge_index') and sub[et].edge_index is not None:
                        device = sub[et].edge_index.device
                        break
                sub[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    def _collect_node_info(self, graph: HeteroData) -> Dict:
        """收集节点数量信息"""
        node_info = {}
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'num_nodes') and graph[node_type].num_nodes is not None:
                num_nodes = int(graph[node_type].num_nodes)
            elif hasattr(graph[node_type], 'x'):
                num_nodes = int(graph[node_type].x.shape[0])
            else:
                num_nodes = 0
            node_info[node_type] = num_nodes
        return node_info

    def _count_labeled_edges(self, graph: HeteroData, edge_types: List) -> Dict:
        """统计标签边数量"""
        edge_counts = {}
        for edge_type in edge_types:
            if edge_type in graph.edge_types and hasattr(graph[edge_type], 'edge_index'):
                edge_counts[edge_type] = graph[edge_type].edge_index.shape[1]
            else:
                edge_counts[edge_type] = 0
        return edge_counts

    def _allocate_labeled_edges(self, sub: HeteroData, graph: HeteroData, node_info: Dict,
                              edge_types: List, edge_counts: Dict, world_size: int, rank: int) -> Tuple[Dict, Dict]:
        """分配标签边到各rank"""
        edge_eids_mapping = {}
        edge_type_allocation = {}
        allocated_edges = 0
        
        for edge_type in edge_types:
            if edge_type not in graph.edge_types or edge_counts[edge_type] == 0:
                self._set_empty_edge_type(sub, graph, edge_type)
                edge_eids_mapping[edge_type] = []
                edge_type_allocation[edge_type] = 0
                continue
            
            edge_store = graph[edge_type]
            total_edges_this_type = edge_counts[edge_type]
            
            if len(edge_type) != 3:
                continue
            
            src_type, _, dst_type = edge_type
            src_num_nodes = node_info.get(src_type, 0)
            dst_num_nodes = node_info.get(dst_type, 0)
            
            start_edge, end_edge = self._compute_edge_range(
                total_edges_this_type, world_size, rank)
            
            if start_edge >= end_edge or start_edge >= total_edges_this_type:
                self._set_empty_edge_type(sub, graph, edge_type)
                edge_eids_mapping[edge_type] = []
                edge_type_allocation[edge_type] = 0
            else:
                selected_indices, eids = self._select_and_validate_edges(
                    edge_store, start_edge, end_edge, src_num_nodes, dst_num_nodes)
                self._assign_edge_data(sub, edge_type, edge_store, selected_indices, eids)
                allocated_edges += len(selected_indices)
                edge_type_allocation[edge_type] = len(selected_indices)
                edge_eids_mapping[edge_type] = eids
        
        return edge_eids_mapping, edge_type_allocation

    def _compute_edge_range(self, total_edges: int, world_size: int, rank: int) -> Tuple[int, int]:
        """计算边的分配范围"""
        edges_per_rank = max(1, total_edges // world_size)
        start_edge = rank * edges_per_rank
        end_edge = min(total_edges, (rank + 1) * edges_per_rank)
        if rank == world_size - 1:
            end_edge = total_edges
        return start_edge, end_edge

    def _select_and_validate_edges(self, edge_store, start_idx: int, end_idx: int,
                                   src_num_nodes: int, dst_num_nodes: int) -> Tuple[torch.Tensor, List]:
        """选择并验证边索引"""
        selected_indices = torch.arange(start_idx, end_idx, dtype=torch.long, device=edge_store.edge_index.device)
        edge_index_selected = edge_store.edge_index[:, selected_indices]
        
        src_indices = edge_index_selected[0]
        dst_indices = edge_index_selected[1]
        
        valid_mask = (
            (src_indices >= 0) & (src_indices < src_num_nodes) &
            (dst_indices >= 0) & (dst_indices < dst_num_nodes)
        )
        
        invalid_count = (~valid_mask).sum().item()
        if invalid_count > 0:
            edge_index_selected = edge_index_selected[:, valid_mask]
            selected_indices = selected_indices[valid_mask]
        
        if hasattr(edge_store, 'eid'):
            eids = edge_store.eid[selected_indices].cpu().tolist()
        else:
            original_eids = list(range(start_idx, end_idx))
            if invalid_count > 0:
                valid_indices_np = valid_mask.cpu().numpy()
                eids = [eid for eid, is_valid in zip(original_eids, valid_indices_np) if is_valid]
            else:
                eids = original_eids
        
        return selected_indices, eids

    def _assign_edge_data(self, sub: HeteroData, edge_type, edge_store, selected_indices, eids: List):
        """分配边数据到子图"""
        sub[edge_type].edge_index = edge_store.edge_index[:, selected_indices]
        
        if hasattr(edge_store, 'y') and edge_store.y is not None:
            sub[edge_type].y = edge_store.y[selected_indices]
        if hasattr(edge_store, 'x') and edge_store.x is not None:
            try:
                sub[edge_type].x = edge_store.x[selected_indices]
            except Exception:
                pass
        if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
            try:
                sub[edge_type].edge_attr = edge_store.edge_attr[selected_indices]
            except Exception:
                pass
        if hasattr(edge_store, 'eid'):
            sub[edge_type].eid = edge_store.eid[selected_indices]

    def _set_empty_edge_type(self, sub: HeteroData, graph: HeteroData, edge_type):
        """设置空边类型"""
        if edge_type in graph.edge_types:
            sub[edge_type].edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=graph[edge_type].edge_index.device)
            if hasattr(graph[edge_type], 'y'):
                sub[edge_type].y = torch.empty(
                    (0,), dtype=graph[edge_type].y.dtype, device=graph[edge_type].y.device)

    def _force_allocate_minimal_edges(self, sub: HeteroData, graph: HeteroData,
                                     edge_types: List, edge_counts: Dict,
                                     world_size: int, rank: int,
                                     edge_eids_mapping: Dict, edge_type_allocation: Dict):
        """强制分配最小数量边"""
        for edge_type in edge_types:
            if edge_type in graph.edge_types and edge_counts[edge_type] > 0:
                edge_store = graph[edge_type]
                num_edges = min(10, max(1, edge_counts[edge_type] // world_size))
                start_idx = (rank * num_edges) % edge_counts[edge_type]
                end_idx = min(edge_counts[edge_type], start_idx + num_edges)
                
                selected_indices = torch.arange(
                    start_idx, end_idx, dtype=torch.long, device=edge_store.edge_index.device)
                sub[edge_type].edge_index = edge_store.edge_index[:, selected_indices]
                
                if hasattr(edge_store, 'y'):
                    sub[edge_type].y = edge_store.y[selected_indices]
                if hasattr(edge_store, 'x') and edge_store.x is not None:
                    sub[edge_type].x = edge_store.x[selected_indices]
                
                edge_eids_mapping[edge_type] = list(range(start_idx, end_idx))
                edge_type_allocation[edge_type] = end_idx - start_idx
                break

    def _partition_all_edges_balanced(self, graph: HeteroData, world_size: int, rank: int) -> Tuple[HeteroData, Dict]:
        """全边类型均匀划分策略"""
        try:
            sub = graph.clone()
            device = self._get_graph_device(graph)
            node_info = self._collect_node_info(graph)
            
            edge_eids_mapping = {}
            edge_allocation_stats = {}
            total_allocated_edges = 0
            
            for edge_type in graph.edge_types:
                edge_store = graph[edge_type]
                
                if not hasattr(edge_store, 'edge_index') or edge_store.edge_index is None:
                    sub[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    edge_eids_mapping[edge_type] = []
                    edge_allocation_stats[edge_type] = {'original': 0, 'allocated': 0}
                    continue
                
                total_edges = edge_store.edge_index.shape[1]
                
                if total_edges == 0:
                    edge_eids_mapping[edge_type] = []
                    edge_allocation_stats[edge_type] = {'original': 0, 'allocated': 0}
                    continue
                
                if len(edge_type) != 3:
                    continue
                
                src_type, _, dst_type = edge_type
                src_num = node_info.get(src_type, 0)
                dst_num = node_info.get(dst_type, 0)
                
                start_edge, end_edge = self._compute_edge_range(total_edges, world_size, rank)
                
                if start_edge >= total_edges:
                    sub[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    if hasattr(edge_store, 'y') and edge_store.y is not None:
                        sub[edge_type].y = torch.empty((0,), dtype=edge_store.y.dtype, device=device)
                    edge_eids_mapping[edge_type] = []
                    edge_allocation_stats[edge_type] = {'original': total_edges, 'allocated': 0}
                    continue
                
                selected_indices, eids, valid_mask = self._select_and_validate_all_edges(
                    edge_store, start_edge, end_edge, src_num, dst_num, device)
                
                sub[edge_type].edge_index = edge_store.edge_index[:, selected_indices]
                
                if hasattr(edge_store, 'y') and edge_store.y is not None:
                    sub[edge_type].y = edge_store.y[start_edge:end_edge][valid_mask]
                
                if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                    try:
                        sub[edge_type].edge_attr = edge_store.edge_attr[start_edge:end_edge][valid_mask]
                    except Exception:
                        pass
                
                if hasattr(edge_store, 'x') and edge_store.x is not None:
                    try:
                        sub[edge_type].x = edge_store.x[start_edge:end_edge][valid_mask]
                    except Exception:
                        if hasattr(sub[edge_type], 'x'):
                            delattr(sub[edge_type], 'x')
                
                if hasattr(edge_store, 'eid'):
                    sub[edge_type].eid = edge_store.eid[start_edge:end_edge][valid_mask]
                
                edge_eids_mapping[edge_type] = eids
                total_allocated_edges += len(selected_indices)
                edge_allocation_stats[edge_type] = {
                    'original': total_edges, 
                    'allocated': len(selected_indices),
                    'range': f'[{start_edge}:{end_edge}]'
                }
            
            mapping_info = {
                'rank': rank,
                'world_size': world_size,
                'edge_eids': edge_eids_mapping,
                'node_info': node_info,
                'partition_strategy': 'all_edges_balanced',
                'edge_allocation_stats': edge_allocation_stats
            }
            return sub, mapping_info
            
        except Exception as e:
            logger.error(f"全边类型划分失败: {e}")
            return self._partition_graph_by_edges_balanced(graph, world_size, rank)

    def _get_graph_device(self, graph: HeteroData) -> torch.device:
        """获取图的设备"""
        for et in graph.edge_types:
            if hasattr(graph[et], 'edge_index') and graph[et].edge_index is not None:
                return graph[et].edge_index.device
        return torch.device('cpu')

    def _select_and_validate_all_edges(self, edge_store, start_idx: int, end_idx: int,
                                      src_num: int, dst_num: int, device) -> Tuple[torch.Tensor, List, torch.Tensor]:
        """选择并验证所有边"""
        selected_indices = torch.arange(start_idx, end_idx, dtype=torch.long, device=device)
        edge_index_selected = edge_store.edge_index[:, selected_indices]
        
        src_indices = edge_index_selected[0]
        dst_indices = edge_index_selected[1]
        
        valid_mask = (
            (src_indices >= 0) & (src_indices < src_num) &
            (dst_indices >= 0) & (dst_indices < dst_num)
        )
        
        invalid_count = (~valid_mask).sum().item()
        if invalid_count > 0:
            edge_index_selected = edge_index_selected[:, valid_mask]
            selected_indices = selected_indices[valid_mask]
        
        if hasattr(edge_store, 'eid'):
            eids = edge_store.eid[start_idx:end_idx][valid_mask].cpu().tolist()
        else:
            valid_indices_np = valid_mask.cpu().numpy()
            original_eids = list(range(start_idx, end_idx))
            eids = [eid for eid, is_valid in zip(original_eids, valid_indices_np) if is_valid]
        
        return selected_indices, eids, valid_mask

    def _partition_graph_for_rank(self, graph: HeteroData, world_size: int, rank: int) -> Tuple[HeteroData, Dict]:
        """将单个HeteroData按各节点类型切成world_size份，返回本rank对应的子图和索引映射"""
        try:
            node_idx_map = {}
            original_to_sub_mapping = {}
            sub_to_original_mapping = {}

            target_node_types = ['production']
            available_targets = [nt for nt in target_node_types if nt in graph.node_types]

            primary_type = None
            if available_targets:
                max_n = -1
                for nt in available_targets:
                    if hasattr(graph[nt], 'num_nodes') and graph[nt].num_nodes is not None:
                        n = int(graph[nt].num_nodes)
                    elif hasattr(graph[nt], 'x'):
                        n = int(graph[nt].x.shape[0])
                    else:
                        n = 0
                    if n > max_n:
                        max_n = n
                        primary_type = nt
            
            if primary_type is None:
                max_n = -1
                for nt in graph.node_types:
                    if hasattr(graph[nt], 'num_nodes') and graph[nt].num_nodes is not None:
                        n = int(graph[nt].num_nodes)
                    elif hasattr(graph[nt], 'x'):
                        n = int(graph[nt].x.shape[0])
                    else:
                        n = 0
                    if n > max_n:
                        max_n = n
                        primary_type = nt

            def split_range(n: int, parts: int, idx: int):
                if n <= 0:
                    return (0, 0)
                start = (n * idx) // parts
                end = (n * (idx + 1)) // parts
                return (start, end)

            for nt in graph.node_types:
                if hasattr(graph[nt], 'num_nodes') and graph[nt].num_nodes is not None:
                    n = int(graph[nt].num_nodes)
                elif hasattr(graph[nt], 'x'):
                    n = int(graph[nt].x.shape[0])
                else:
                    n = 0

                if n == 0:
                    continue

                s, e = split_range(n, world_size, rank)
                import torch as _torch
                original_indices = _torch.arange(s, e, dtype=_torch.long)
                node_idx_map[nt] = original_indices

                original_to_sub_mapping[nt] = {}
                sub_to_original_mapping[nt] = {}
                for sub_idx, orig_idx in enumerate(original_indices):
                    original_to_sub_mapping[nt][orig_idx.item()] = sub_idx
                    sub_to_original_mapping[nt][sub_idx] = orig_idx.item()

            try:
                sub = graph.subgraph(node_idx_map)
            except Exception as _:
                import torch as _torch
                complete_map = {}
                for nt in graph.node_types:
                    if nt in node_idx_map:
                        complete_map[nt] = node_idx_map[nt]
                    else:
                        complete_map[nt] = _torch.arange(0, 0, dtype=_torch.long)
                        original_to_sub_mapping[nt] = {}
                        sub_to_original_mapping[nt] = {}
                sub = graph.subgraph(complete_map)

            mapping_info = {
                'rank': rank,
                'world_size': world_size,
                'original_to_sub': original_to_sub_mapping,
                'sub_to_original': sub_to_original_mapping,
                'node_ranges': {}
            }

            for nt in graph.node_types:
                if hasattr(graph[nt], 'num_nodes') and graph[nt].num_nodes is not None:
                    n = int(graph[nt].num_nodes)
                elif hasattr(graph[nt], 'x'):
                    n = int(graph[nt].x.shape[0])
                else:
                    n = 0
                s, e = split_range(n, world_size, rank)
                mapping_info['node_ranges'][nt] = {'start': s, 'end': e, 'total': n}

            return sub, mapping_info
        except Exception as e:
            raise RuntimeError(f"partition_graph_for_rank failed: {e}")

    def _partition_graph_by_edges_for_rank(self, graph: HeteroData, world_size: int, rank: int) -> Tuple[
        HeteroData, Dict]:
        """基于边进行均匀划分：对关键预测边类型，按eid等分到各rank；节点保持不变"""
        try:
            sub = graph.clone()
            
            for edge_type in sub.edge_types:
                if not hasattr(sub[edge_type], 'edge_index') or sub[edge_type].edge_index is None:
                    device = torch.device('cpu')
                    for et in sub.edge_types:
                        if hasattr(sub[et], 'edge_index') and sub[et].edge_index is not None:
                            device = sub[et].edge_index.device
                            break
                    sub[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            
            edge_types_with_labels = [('inventory', 'to', 'virtual'), 
                                      ('inventory', 'to_transport_labeled', 'inventory')]
            edge_eids_mapping = {}
            node_info = {}
            
            for node_type in graph.node_types:
                if hasattr(graph[node_type], 'num_nodes') and graph[node_type].num_nodes is not None:
                    num_nodes = int(graph[node_type].num_nodes)
                elif hasattr(graph[node_type], 'x'):
                    num_nodes = int(graph[node_type].x.shape[0])
                else:
                    num_nodes = 0
                node_info[node_type] = num_nodes

            for edge_type in edge_types_with_labels:
                if edge_type not in graph.edge_types:
                    continue
                    
                edge_store = graph[edge_type]
                if not hasattr(edge_store, 'edge_index'):
                    continue
                    
                total_edges = int(edge_store.edge_index.shape[1])
                if total_edges == 0:
                    continue
                
                if len(edge_type) == 3:
                    src_type, _, dst_type = edge_type
                else:
                    continue
                
                src_num_nodes = node_info.get(src_type, 0)
                dst_num_nodes = node_info.get(dst_type, 0)
                
                if src_num_nodes == 0 or dst_num_nodes == 0:
                    continue
                
                if hasattr(edge_store, 'eid'):
                    eids = edge_store.eid.detach().cpu().tolist()
                else:
                    eids = list(range(total_edges))

                sorted_pairs = sorted([(eid, idx) for idx, eid in enumerate(eids)])
                sorted_eids = [p[0] for p in sorted_pairs]
                sorted_local_indices = [p[1] for p in sorted_pairs]

                per_rank = total_edges // world_size
                start = rank * per_rank
                end = (rank + 1) * per_rank if rank < world_size - 1 else total_edges
                selected_sorted_idx = list(range(start, end))

                selected_local_cols = [sorted_local_indices[i] for i in selected_sorted_idx]
                selected_local_cols_tensor = torch.tensor(
                    selected_local_cols, 
                    dtype=torch.long,
                    device=edge_store.edge_index.device
                )

                edge_index_selected = edge_store.edge_index[:, selected_local_cols_tensor]
                
                src_indices = edge_index_selected[0]
                dst_indices = edge_index_selected[1]
                
                valid_mask = (
                    (src_indices >= 0) & (src_indices < src_num_nodes) &
                    (dst_indices >= 0) & (dst_indices < dst_num_nodes)
                )
                
                invalid_count = (~valid_mask).sum().item()
                if invalid_count > 0:
                    edge_index_selected = edge_index_selected[:, valid_mask]
                    selected_local_cols_tensor = selected_local_cols_tensor[valid_mask]
                    valid_indices_np = valid_mask.cpu().numpy()
                    selected_sorted_idx = [idx for idx, is_valid in zip(selected_sorted_idx, valid_indices_np) if is_valid]
                
                sub[edge_type].edge_index = edge_index_selected
                
                if hasattr(edge_store, 'y') and edge_store.y is not None:
                    sub[edge_type].y = edge_store.y[selected_local_cols_tensor]
                    
                if hasattr(edge_store, 'x') and edge_store.x is not None:
                    try:
                        sub[edge_type].x = edge_store.x[selected_local_cols_tensor]
                    except Exception:
                        pass
                        
                if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                    try:
                        sub[edge_type].edge_attr = edge_store.edge_attr[selected_local_cols_tensor]
                    except Exception:
                        pass
                        
                if hasattr(edge_store, 'eid'):
                    sub[edge_type].eid = edge_store.eid[selected_local_cols_tensor]
                else:
                    sub[edge_type].eid = torch.tensor(
                        [sorted_eids[i] for i in selected_sorted_idx], 
                        dtype=torch.long,
                        device=edge_store.edge_index.device
                    )

                edge_eids_mapping[edge_type] = [sorted_eids[i] for i in selected_sorted_idx]

            mapping_info = {
                'rank': rank,
                'world_size': world_size,
                'edge_eids': edge_eids_mapping,
                'node_info': node_info,
                'partition_strategy': 'edge_only'
            }
            
            return sub, mapping_info
            
        except Exception as e:
            logger.error(f"partition_graph_by_edges_for_rank 失败: {e}")
            raise RuntimeError(f"partition_graph_by_edges_for_rank failed: {e}")

    def _validate_single_graph(self, graph_idx: int, graphs: List[HeteroData], 
                              dataset_type: str) -> Tuple[float, Dict]:
        """验证单个图"""
        if self.use_distributed and self.world_size > 1:
            dist.barrier()
        
        if graph_idx < len(graphs):
            original_graph = graphs[graph_idx]
        else:
            logger.warning(f"[Rank {self.rank}] 图索引{graph_idx}超出范围，跳过")
            return 0.0, {}
        
        self._original_graphs[graph_idx] = original_graph
        
        enable_partition = bool(getattr(self.config.training_config, 'enable_graph_partition', True))
        if enable_partition and self.use_distributed and self.world_size > 1:
            graph_to_process, mapping_info = self._try_partition_graph(
                original_graph, graph_idx)
            if graph_to_process != original_graph:
                self._graph_mappings[graph_idx] = mapping_info
                self.use_graph_partition = True
        else:
            graph_to_process = original_graph
        
        graph = self._move_data_to_device(graph_to_process)
        
        # 混合精度推理：autocast 只包裹前向传播
        if self.use_amp and self._amp_device_type is not None:
            with torch.amp.autocast(device_type=self._amp_device_type, dtype=self.amp_dtype):
                predictions = self.model(graph)
            
            # 将预测结果转为 float32 确保后续计算数值安全
            predictions = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in predictions.items()}
        else:
            predictions = self.model(graph)
        
        if not predictions:
            logger.warning(f"[Rank {self.rank}] 图{graph_idx} 模型预测结果为空")
        
        if hasattr(self.model, 'module'):
            loss = self.model.module.compute_loss(predictions, graph)
        else:
            loss = self.model.compute_loss(predictions, graph)

        metrics = self._compute_metrics(predictions, graph)
        
        graph_date = self._extract_date_from_graph(original_graph, graph_idx)
        self._collect_predictions_for_validation(
            predictions, graph, original_graph, graph_idx, graph_date, dataset_type)
        
        self._cleanup_after_validation(graph, predictions)
        
        return loss.item(), metrics

    def _try_partition_graph(self, original_graph: HeteroData, graph_idx: int) -> Tuple[HeteroData, Dict]:
        """尝试对图进行划分"""
        try:
            partition_strategy = getattr(self.config.training_config, 'partition_strategy', 'balanced_edge')
            if partition_strategy == 'all_edges':
                subgraph, mapping_info = self._partition_all_edges_balanced(
                    original_graph, self.world_size, self.rank)
            elif partition_strategy == 'balanced_edge':
                subgraph, mapping_info = self._partition_graph_by_edges_balanced(
                    original_graph, self.world_size, self.rank)
            elif partition_strategy == 'edge':
                subgraph, mapping_info = self._partition_graph_by_edges_for_rank(
                    original_graph, self.world_size, self.rank)
            else:
                subgraph, mapping_info = self._partition_graph_for_rank(
                    original_graph, self.world_size, self.rank)
            
            if self._validate_subgraph(subgraph):
                return subgraph, mapping_info
            else:
                logger.warning(f"[Rank {self.rank}] 图{graph_idx}划分验证失败")
                return original_graph, {}
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] 图{graph_idx}划分异常: {e}")
            return original_graph, {}

    def _collect_predictions_for_validation(self, predictions: Dict, graph: HeteroData,
                                            original_graph: HeteroData, graph_idx: int,
                                            graph_date: str, dataset_type: str):
        """收集验证阶段的预测结果"""
        mapping_info = self._graph_mappings.get(graph_idx, {})
        use_partition = getattr(self, 'use_graph_partition', False)

        if self.use_distributed and self.world_size > 1:
            if use_partition:
                self._collect_predictions_distributed_with_partition(
                    predictions, original_graph, graph_idx, graph_date, dataset_type)
            else:
                self._collect_predictions_distributed_without_partition(
                    predictions, graph_idx, graph_date, dataset_type)
        else:
            self._collect_predictions_single(predictions, graph, graph_date, dataset_type)

    def _collect_predictions_distributed_with_partition(self, predictions: Dict, original_graph: HeteroData,
                                                       graph_idx: int, graph_date: str, dataset_type: str):
        """分布式模式（有图划分）收集预测"""
        try:
            full_predictions = self._gather_and_reconstruct_predictions(
                predictions, None, original_graph, {},
                dataset_type, graph_date)
            
            if self.rank == 0 and full_predictions:
                self._ensure_prediction_handler()
                if graph_idx in self._original_graphs:
                    self.prediction_handler.collect_predictions(
                        full_predictions, self._original_graphs[graph_idx], graph_date, 
                        dataset_type, self.node_manager, None, self.graph_builder)
            
            self._clear_device_cache()
        except Exception as e:
            logger.error(f"分布式预测结果汇总失败(图划分): {e}")

    def _collect_predictions_distributed_without_partition(self, predictions: Dict,
                                                          graph_idx: int, graph_date: str, dataset_type: str):
        """分布式模式（无图划分）收集预测"""
        try:
            gathered_packages = self._gather_predictions_via_filesystem(
                predictions, graph_date, graph_idx, dataset_type)
            
            if self.rank == 0:
                self._ensure_prediction_handler()
                collected_dates = set()
                for package in gathered_packages:
                    if package is None:
                        continue
                    rank_predictions = package.get('full_predictions') or package.get('predictions')
                    rank_graph_date = package.get('graph_date')
                    rank_graph_idx = package.get('graph_idx')
                    
                    if not isinstance(rank_predictions, dict) or not rank_predictions:
                        continue
                    if rank_graph_date in collected_dates:
                        continue
                    if rank_graph_idx is not None and rank_graph_idx in self._original_graphs:
                        self.prediction_handler.collect_predictions(
                            rank_predictions, self._original_graphs[rank_graph_idx], 
                            rank_graph_date, dataset_type, self.node_manager, None, self.graph_builder)
                        collected_dates.add(rank_graph_date)
            
            self._clear_device_cache()
        except Exception as e:
            logger.error(f"预测结果汇总失败: {e}")

    def _collect_predictions_single(self, predictions: Dict, graph: HeteroData,
                                   graph_date: str, dataset_type: str):
        """单卡模式收集预测"""
        try:
            self._ensure_prediction_handler()
            self.prediction_handler.collect_predictions(
                predictions, graph, graph_date, dataset_type, 
                self.node_manager, None, self.graph_builder)
        except Exception as e:
            logger.error(f"collect_predictions 失败: {e}")

    def _ensure_prediction_handler(self):
        """确保prediction_handler已初始化"""
        if not hasattr(self, 'prediction_handler') or self.prediction_handler is None:
            self.prediction_handler = PredictionHandler(self.save_dir, self.rank, self.world_size)

    def _cleanup_after_validation(self, graph: HeteroData, predictions: Dict):
        """验证后清理资源"""
        try:
            del graph
            del predictions
            self._clear_device_cache()
        except:
            pass

    def _clear_device_cache(self):
        """清理设备缓存"""
        if self.device.type == 'npu':
            torch.npu.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def validate(self, graphs: List[HeteroData],
                 dataset_type: str = 'validation') -> Tuple[float, Dict]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_metrics = defaultdict(list)
        
        logger.info(f"[Rank {self.rank}] 开始验证: {dataset_type}, 图数量={len(graphs)}")
        
        self._graph_mappings = getattr(self, '_graph_mappings', {})
        self._original_graphs = getattr(self, '_original_graphs', {})
        self._graph_mappings.clear()
        self._original_graphs.clear()
        self.use_graph_partition = False

        with torch.no_grad():
            num_graphs = self._sync_num_graphs(graphs)
            
            for graph_idx in range(num_graphs):
                loss, metrics = self._validate_single_graph(graph_idx, graphs, dataset_type)
                total_loss += loss
                for metric_name, value in metrics.items():
                    all_metrics[metric_name].append(value)

        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        avg_loss = self._compute_avg_loss(total_loss, num_graphs)

        return avg_loss, avg_metrics

    def _sync_num_graphs(self, graphs: List[HeteroData]) -> int:
        """同步图数量"""
        num_graphs = len(graphs)
        if self.use_distributed and self.world_size > 1:
            num_graphs_tensor = torch.tensor([num_graphs], dtype=torch.int32, device=self.device)
            dist.all_reduce(num_graphs_tensor, op=dist.ReduceOp.MAX)
            num_graphs = num_graphs_tensor.item()
            if len(graphs) != num_graphs:
                graphs[:] = graphs[:num_graphs] if len(graphs) > num_graphs else graphs
        return num_graphs

    def _compute_avg_loss(self, total_loss: float, num_graphs: int) -> float:
        """计算平均损失"""
        use_eval_collectives = self.use_distributed and dist.is_initialized() and \
                               bool(getattr(self.config.training_config, 'eval_all_ranks', True))
        if use_eval_collectives:
            try:
                total_loss_tensor = torch.tensor(total_loss, device=self.device, dtype=torch.float32)
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                global_total_loss = total_loss_tensor.item()
                
                num_graphs_tensor = torch.tensor(num_graphs, device=self.device, dtype=torch.float32)
                dist.all_reduce(num_graphs_tensor, op=dist.ReduceOp.MAX)
                global_num_graphs = int(num_graphs_tensor.item())
                
                avg_loss = global_total_loss / global_num_graphs if global_num_graphs > 0 else 0.0
                dist.barrier()
                return avg_loss
            except Exception as e:
                logger.warning(f"验证同步失败: {e}")
                return total_loss / max(1, num_graphs)
        return total_loss / max(1, num_graphs)

    def _gather_predictions_via_filesystem(self, predictions: Dict, graph_date: str, 
                                           graph_idx: int, dataset_type: str) -> List[Dict]:
        """使用文件系统通信替代all_gather_object，避免EOFError"""
        import tempfile
        import pickle
        
        temp_dir = path_config.get_dist_temp_dir()
        temp_dir.mkdir(exist_ok=True, parents=True)
        temp_file = temp_dir / f'rank_{self.rank}_graph_{graph_idx}_{dataset_type}.pkl'
        
        try:
            serializable_predictions = {}
            if predictions:
                for k, v in predictions.items():
                    if isinstance(v, torch.Tensor):
                        try:
                            if v.device.type in ('cuda', 'npu'):
                                serializable_predictions[k] = v.cpu().detach().numpy()
                            else:
                                serializable_predictions[k] = v.detach().numpy()
                        except Exception as e:
                            logger.warning(f"转换tensor {k} 失败: {e}")
                            continue
                    else:
                        serializable_predictions[k] = v
            
            data_package = {
                'full_predictions': serializable_predictions,
                'graph_date': graph_date,
                'graph_idx': graph_idx,
                'rank': self.rank
            }
            
            with open(temp_file, 'wb') as f:
                pickle.dump(data_package, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            del serializable_predictions
            if self.device.type == 'npu':
                torch.npu.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if self.use_distributed and dist.is_initialized():
                dist.barrier()
            
            all_packages = [None for _ in range(self.world_size)]
            if self.rank == 0:
                for r in range(self.world_size):
                    r_temp_file = temp_dir / f'rank_{r}_graph_{graph_idx}_{dataset_type}.pkl'
                    try:
                        if r_temp_file.exists():
                            with open(r_temp_file, 'rb') as f:
                                package = pickle.load(f)
                            
                            if package and 'full_predictions' in package:
                                converted_preds = {}
                                for k, v in package['full_predictions'].items():
                                    if isinstance(v, np.ndarray):
                                        try:
                                            converted_preds[k] = torch.from_numpy(v)
                                        except Exception:
                                            continue
                                    elif isinstance(v, torch.Tensor):
                                        converted_preds[k] = v
                                    else:
                                        converted_preds[k] = v
                                package['full_predictions'] = converted_preds
                            
                            all_packages[r] = package
                    except Exception:
                        pass
            
            if self.use_distributed and dist.is_initialized():
                dist.barrier()
            
            try:
                if temp_file.exists():
                    temp_file.unlink()
                if self.rank == 0:
                    for r in range(self.world_size):
                        r_temp_file = temp_dir / f'rank_{r}_graph_{graph_idx}_{dataset_type}.pkl'
                        if r_temp_file.exists():
                            r_temp_file.unlink()
            except Exception:
                pass
            
            return all_packages
            
        except Exception as e:
            logger.error(f"文件系统通信失败: {e}")
            result = [None for _ in range(self.world_size)]
            result[self.rank] = {
                'full_predictions': predictions,
                'graph_date': graph_date,
                'graph_idx': graph_idx,
                'rank': self.rank
            }
            return result

    def _gather_dict_via_filesystem(self, data_dict: Dict, prefix: str) -> List[Dict]:
        """使用文件系统通信收集字典数据"""
        import pickle
        
        temp_dir = path_config.get_dist_temp_dir()
        temp_dir.mkdir(exist_ok=True, parents=True)
        temp_file = temp_dir / f'{prefix}_rank_{self.rank}.pkl'
        
        try:
            write_success = False
            
            try:
                with open(temp_file, 'wb') as f:
                    pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.flush()
                    os.fsync(f.fileno())
                
                if temp_file.exists() and temp_file.stat().st_size > 0:
                    write_success = True
            except Exception as write_e:
                logger.warning(f"写入临时文件失败: {write_e}")
            
            if self.use_distributed and dist.is_initialized():
                success_tensor = torch.tensor([1.0 if write_success else 0.0], 
                                             dtype=torch.float32, device=self.device)
                dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)
            
            if self.use_distributed and dist.is_initialized():
                dist.barrier()
                time.sleep(3.0)
                dist.barrier()
            
            all_dicts = [None for _ in range(self.world_size)]
            if self.rank == 0:
                for r in range(self.world_size):
                    r_temp_file = temp_dir / f'{prefix}_rank_{r}.pkl'
                    try:
                        max_retries = 10
                        for retry in range(max_retries):
                            if r_temp_file.exists() and r_temp_file.stat().st_size > 0:
                                with open(r_temp_file, 'rb') as f:
                                    data = pickle.load(f)
                                
                                if isinstance(data, dict):
                                    converted_data = {}
                                    for k, v in data.items():
                                        if isinstance(v, np.ndarray):
                                            try:
                                                converted_data[k] = torch.from_numpy(v)
                                            except Exception:
                                                converted_data[k] = v
                                        elif isinstance(v, torch.Tensor):
                                            converted_data[k] = v
                                        else:
                                            converted_data[k] = v
                                    all_dicts[r] = converted_data
                                else:
                                    all_dicts[r] = data
                                break
                            else:
                                if retry < max_retries - 1:
                                    time.sleep(2.0)
                    except Exception:
                        pass
            
            if self.use_distributed and dist.is_initialized():
                dist.barrier()
            
            return all_dicts
            
        except Exception as e:
            logger.error(f"文件系统通信(字典)失败: {e}")
            result = [None for _ in range(self.world_size)]
            result[self.rank] = data_dict
            return result

    def _gather_and_reconstruct_predictions(self, local_predictions: Dict, local_graph: HeteroData,
                                            original_graph: HeteroData, mapping_info: Dict,
                                            dataset_type: str, graph_date: str) -> Dict:
        """汇总各GPU的子图预测结果，重构成完整图的预测结果"""
        if not self.use_distributed or not dist.is_initialized():
            return local_predictions

        try:
            if self.use_distributed and dist.is_initialized():
                try:
                    cpu_local_predictions = {}
                    if local_predictions:
                        for k, v in local_predictions.items():
                            try:
                                if isinstance(v, torch.Tensor):
                                    if v.device.type in ('cuda', 'npu'):
                                        cpu_local_predictions[k] = v.cpu().detach().numpy()
                                    else:
                                        cpu_local_predictions[k] = v.detach().numpy()
                                else:
                                    cpu_local_predictions[k] = v
                            except Exception:
                                pass
                    
                    try:
                        del local_graph
                    except:
                        pass
                    
                    if self.use_distributed and dist.is_initialized():
                        dist.barrier()
                    
                    gathered_predictions = self._gather_dict_via_filesystem(
                        cpu_local_predictions, f"pred_merge_{graph_date}"
                    )
                    
                    if self.rank == 0:
                        merged_predictions = {}
                        
                        for rank_idx, rank_predictions in enumerate(gathered_predictions):
                            if not isinstance(rank_predictions, dict):
                                continue
                            
                            for pred_key, pred_data in rank_predictions.items():
                                if pred_key not in merged_predictions:
                                    merged_predictions[pred_key] = []
                                merged_predictions[pred_key].append(pred_data.cpu() if isinstance(pred_data, torch.Tensor) else pred_data)
                        
                        cpu_full_predictions = self._reconstruct_predictions_from_merged(merged_predictions, original_graph, dataset_type, graph_date)
                        
                        if self.use_distributed and dist.is_initialized():
                            try:
                                dist.barrier()
                            except:
                                pass
                        
                        return cpu_full_predictions
                    else:
                        try:
                            dist.barrier()
                        except:
                            pass
                        return {}
                        
                except Exception as e:
                    logger.error(f"分布式预测合并失败: {e}")
                    try:
                        if self.use_distributed and dist.is_initialized():
                            dist.barrier()
                    except:
                        pass
                    return {} if self.rank != 0 else local_predictions

            return local_predictions

        except Exception as e:
            logger.error(f"预测结果汇总失败: {e}")
            if self.rank == 0:
                return local_predictions
            else:
                return {}

    def _reconstruct_predictions_from_merged(self, merged_predictions: Dict, original_graph: HeteroData, 
                                           dataset_type: str, graph_date: str) -> Dict:
        """从合并后的预测结果重构完整预测"""
        full_predictions = {}
        
        node_types = ['production', 'inventory']
        for node_type in node_types:
            if node_type in merged_predictions and node_type in original_graph.node_types:
                if hasattr(original_graph[node_type], 'num_nodes') and original_graph[node_type].num_nodes is not None:
                    total_nodes = int(original_graph[node_type].num_nodes)
                elif hasattr(original_graph[node_type], 'x'):
                    total_nodes = int(original_graph[node_type].x.shape[0])
                else:
                    continue
                
                all_predictions = merged_predictions[node_type]
                if all_predictions:
                    converted_predictions = []
                    for pred in all_predictions:
                        if isinstance(pred, np.ndarray):
                            converted_predictions.append(torch.from_numpy(pred))
                        elif isinstance(pred, torch.Tensor):
                            converted_predictions.append(pred)
                        else:
                            converted_predictions.append(pred)
                    all_predictions = converted_predictions
                    
                    has_node_mapping = hasattr(self, '_current_mapping_info') and \
                                      self._current_mapping_info is not None and \
                                      'node_indices' in self._current_mapping_info and \
                                      node_type in self._current_mapping_info.get('node_indices', {})
                    
                    if has_node_mapping:
                        full_pred = torch.zeros(total_nodes, dtype=all_predictions[0].dtype)
                        
                        for rank_idx, pred in enumerate(all_predictions):
                            if not isinstance(pred, torch.Tensor):
                                continue
                            
                            nodes_per_rank = total_nodes // len(all_predictions)
                            start_idx = rank_idx * nodes_per_rank
                            end_idx = total_nodes if rank_idx == len(all_predictions) - 1 else (rank_idx + 1) * nodes_per_rank
                            
                            actual_len = min(len(pred), end_idx - start_idx)
                            if actual_len > 0:
                                full_pred[start_idx:start_idx + actual_len] = pred[:actual_len].cpu()
                        
                        full_predictions[node_type] = full_pred
                    else:
                        first_pred = all_predictions[0]
                        if isinstance(first_pred, torch.Tensor) and len(first_pred) == total_nodes:
                            full_predictions[node_type] = first_pred.cpu()
                        else:
                            combined_pred = torch.cat(all_predictions, dim=0)
                            if len(combined_pred) >= total_nodes:
                                full_predictions[node_type] = combined_pred[:total_nodes]
        
        edge_types_with_labels = [('inventory', 'to', 'virtual'), ('inventory', 'to_transport_labeled', 'inventory')]
        for edge_type in edge_types_with_labels:
            if edge_type in merged_predictions and edge_type in original_graph.edge_types:
                if hasattr(original_graph[edge_type], 'edge_index'):
                    total_edges = original_graph[edge_type].edge_index.shape[1]
                else:
                    continue
                
                all_edge_predictions = merged_predictions[edge_type]
                if all_edge_predictions:
                    cpu_edge_predictions = []
                    for pred in all_edge_predictions:
                        if isinstance(pred, np.ndarray):
                            cpu_edge_predictions.append(torch.from_numpy(pred))
                        elif isinstance(pred, torch.Tensor):
                            cpu_edge_predictions.append(pred.cpu() if pred.is_cuda else pred)
                        else:
                            cpu_edge_predictions.append(pred)
                    
                    combined_edge_pred = torch.cat(cpu_edge_predictions, dim=0)
                    if len(combined_edge_pred) >= total_edges:
                        full_predictions[edge_type] = combined_edge_pred[:total_edges].cpu() if combined_edge_pred.is_cuda else combined_edge_pred[:total_edges]
                    else:
                        missing_count = total_edges - len(combined_edge_pred)
                        error_msg = (f"边类型 {edge_type} 预测数量不足: {len(combined_edge_pred)} < {total_edges}, "
                                    f"缺失 {missing_count} 个边预测")
                        logger.error(error_msg)
                        raise ValueError(error_msg)
        
        cpu_full_predictions = {}
        for k, v in full_predictions.items():
            if isinstance(v, torch.Tensor):
                cpu_full_predictions[k] = v.cpu() if v.is_cuda else v
            else:
                cpu_full_predictions[k] = v
        
        return cpu_full_predictions

    def _extract_date_from_graph(self, graph: HeteroData, graph_idx: int) -> str:
        """从图中提取日期信息"""
        # 尝试从图的属性中获取日期
        if hasattr(graph, 'date'):
            return graph.date
        elif hasattr(graph, 'graph_date'):
            return graph.graph_date
        else:
            # 如果没有日期信息，使用索引
            return f"graph_{graph_idx:04d}"

    def _compute_metrics(self, predictions: Dict[str, torch.Tensor],
                         graph: HeteroData) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        target_node_types = ['production', 'inventory']
        edge_types_with_labels = [('inventory', 'to', 'virtual'), ('inventory', 'to_transport_labeled', 'inventory')]

        for node_type in predictions:
            if node_type in target_node_types and hasattr(graph[node_type], 'y'):
                self._compute_node_metrics(predictions, graph, node_type, metrics)

        for edge_type in edge_types_with_labels:
            if edge_type in predictions and hasattr(graph[edge_type], 'y'):
                self._compute_edge_metrics(predictions, graph, edge_type, metrics)

        self._compute_overall_metrics(metrics)
        return metrics

    def _compute_node_metrics(self, predictions: Dict, graph: HeteroData, 
                            node_type: str, metrics: Dict):
        """计算节点预测指标"""
        pred = predictions[node_type].detach().cpu().numpy()
        true = graph[node_type].y.detach().cpu().numpy()
        
        valid_mask = ~(np.isnan(true) | np.isnan(pred) | (true == -1.0))
        if not valid_mask.any():
            return

        pred_valid = pred[valid_mask]
        true_valid = (true[valid_mask] != 0).astype(float)
        
        self._add_regression_metrics(metrics, f'{node_type}', pred_valid, true_valid)
        self._add_classification_metrics(metrics, f'{node_type}', pred_valid, true_valid)

    def _compute_edge_metrics(self, predictions: Dict, graph: HeteroData,
                             edge_type: Tuple, metrics: Dict):
        """计算边预测指标"""
        edge_pred = predictions[edge_type].detach().cpu().numpy()
        edge_true = graph[edge_type].y.detach().cpu().numpy()
        
        valid_mask = ~(np.isnan(edge_true) | np.isnan(edge_pred))
        if not valid_mask.any():
            return

        edge_pred_valid = edge_pred[valid_mask]
        edge_true_valid = edge_true[valid_mask]
        edge_type_str = "_".join(edge_type)
        
        self._add_regression_metrics(metrics, edge_type_str, edge_pred_valid, edge_true_valid)
        self._add_classification_metrics(metrics, edge_type_str, edge_pred_valid, edge_true_valid)

    def _add_regression_metrics(self, metrics: Dict, prefix: str, pred: np.ndarray, true: np.ndarray):
        """添加回归指标"""
        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(mse)
        
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics[f'{prefix}_mse'] = mse
        metrics[f'{prefix}_mae'] = mae
        metrics[f'{prefix}_rmse'] = rmse
        metrics[f'{prefix}_r2'] = r2

    def _add_classification_metrics(self, metrics: Dict, prefix: str, pred: np.ndarray, true: np.ndarray):
        """添加分类指标"""
        pred_binary = (pred > 0.5).astype(int)
        true_binary = true.astype(int)
        
        if len(np.unique(true_binary)) <= 1:
            return
        
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            metrics[f'{prefix}_accuracy'] = accuracy_score(true_binary, pred_binary)
            metrics[f'{prefix}_precision'] = precision_score(true_binary, pred_binary, zero_division=0)
            metrics[f'{prefix}_recall'] = recall_score(true_binary, pred_binary, zero_division=0)
            metrics[f'{prefix}_f1'] = f1_score(true_binary, pred_binary, zero_division=0)
        except Exception:
            pass

    def _compute_overall_metrics(self, metrics: Dict):
        """计算总体指标"""
        mse_values = [v for k, v in metrics.items() if k.endswith('_mse')]
        if mse_values:
            metrics['overall_mse'] = np.mean(mse_values)
            metrics['overall_rmse'] = np.sqrt(metrics['overall_mse'])
        
        mae_values = [v for k, v in metrics.items() if k.endswith('_mae')]
        if mae_values:
            metrics['overall_mae'] = np.mean(mae_values)
        
        r2_values = [v for k, v in metrics.items() if k.endswith('_r2')]
        if r2_values:
            metrics['overall_r2'] = np.mean(r2_values)

    def train(self, train_graphs: List[HeteroData],
              val_graphs: List[HeteroData] = None) -> Dict:
        """训练模型"""
        logger.info("Starting model training...")

        self.setup_training()
        self._log_actual_training_start()

        training_config = self.config.training_config
        num_epochs = training_config.num_epochs
        patience = training_config.patience

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_graphs)
            self.training_history['train_loss'].append(train_loss)

            # 验证
            val_loss = 0.0
            val_metrics = {}
            if val_graphs:
                val_loss, val_metrics = self.validate(val_graphs, 'validation')
                self.training_history['val_loss'].append(val_loss)

                # 更新学习率调度器
                if self.scheduler:
                    self.scheduler.step(val_loss)

            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['learning_rate'].append(current_lr)

            # 日志输出 - 确保参数一致性
            if (not self.use_distributed) or (self.rank == 0):
                logger.info(f"Epoch {epoch}/{num_epochs}")
                logger.info(f"Train Loss: {train_loss:.6f}")
                logger.info(f"Val Loss: {val_loss:.6f}")
                logger.info(f"Learning Rate: {current_lr:.8f}")

            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0

                # 保存最佳模型
                state_dict = self.model.module.state_dict() if hasattr(self.model,
                                                                       'module') else self.model.state_dict()
                # 🔧 NPU适配：将state_dict移到CPU
                state_dict_cpu = self._move_state_dict_to_cpu(state_dict)
                optimizer_state_cpu = self._move_state_dict_to_cpu(self.optimizer.state_dict())
                
                self.best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': state_dict_cpu,
                    'optimizer_state_dict': optimizer_state_cpu,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'config': ConfigSerializer.serialize_config(self.config),
                    'amp_mode': '粗粒度_autocast' if self.use_amp else 'fp32'
                }

                if (not self.use_distributed) or (self.rank == 0):
                    torch.save(self.best_model_state, self.best_model_path)
                    logger.info(f"Saved best model at epoch {epoch} with val_loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break

        # 训练完成后的总结
        # 确保所有rank先完成训练、再由rank0统一打印总结与绘图，避免未同步导致的NaN展示
        if self.use_distributed and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
        if (not self.use_distributed) or (self.rank == 0):
            self._log_training_summary(epoch)

        # 分布式训练结束同步
        if self.use_distributed:
            dist.barrier()

        return {
            'best_val_loss': self.best_val_loss,
            'final_epoch': epoch,
            'training_history': self.training_history
        }

    def _log_actual_training_start(self):
        """记录训练开始时的实际参数"""
        training_config = self.config.training_config

        logger.info("\n" + "=" * 50)
        logger.info("TRAINING START - 实际使用参数确认")
        logger.info("=" * 50)
        logger.info(f"训练轮数: {training_config.num_epochs}")

        if self.optimizer is not None:
            logger.info(f"学习率: {self.optimizer.param_groups[0]['lr']}")
        else:
            logger.info(f"学习率: {training_config.learning_rate}")

        logger.info(f"权重衰减: {training_config.weight_decay}")
        logger.info(f"早停耐心值: {training_config.patience}")
        logger.info(f"全局隐藏维度: {self.config.architecture_config.global_hidden_dim}")
        logger.info(f"混合精度: {'启用' if self.use_amp else '未启用'}")
        logger.info("=" * 50)

    def _log_training_summary(self, final_epoch: int):
        """记录训练总结"""
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING SUMMARY - 训练总结")
        logger.info("=" * 50)
        logger.info(f"完成轮数: {final_epoch}")
        logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")

        if self.optimizer is not None:
            logger.info(f"最终学习率: {self.optimizer.param_groups[0]['lr']:.8f}")

        logger.info(f"训练历史长度: {len(self.training_history['train_loss'])}")
        logger.info("=" * 50)

    def clear_predictions(self):
        """清除预测结果 - 新增方法"""
        if hasattr(self, 'prediction_handler') and self.prediction_handler is not None:
            self.prediction_handler.clear_predictions()
            logger.info("Cleared all predictions from prediction_handler")
        else:
            logger.warning("prediction_handler not initialized, cannot clear predictions")

    def evaluate(self, graphs: List[HeteroData],
                 dataset_type: str = 'test') -> Tuple[float, Dict]:
        """评估模型（旧版本兼容方法）"""
        # 加载最佳模型
        if self.best_model_state is not None:
            self._load_state_dict_safely(self.best_model_state['model_state_dict'], strict=False)

        logger.info(f"Evaluating on {dataset_type} set with {len(graphs)} graphs")
        # 确保所有rank进入评估前同步，避免只有rank0进入导致另一侧未参与
        if self.use_distributed and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass

        # 防御性检查，确保prediction_handler已初始化
        if not hasattr(self, 'prediction_handler') or self.prediction_handler is None:
            logger.warning("在evaluate方法中prediction_handler未初始化，现在创建")
            self.prediction_handler = PredictionHandler(self.save_dir, self.rank, self.world_size)

        # 清除之前的预测结果
        self.prediction_handler.clear_predictions()

        # 为避免训练完成后立即打印“最终结果”导致顺序混乱：统一由rank0在全部评估完成后打印。
        # 此处仅计算并返回数值。
        avg_loss, avg_metrics = self.validate(graphs, dataset_type)
        if self.use_distributed and dist.is_initialized() and bool(
                getattr(self.config.training_config, 'eval_all_ranks', True)):
            try:
                loss_tensor = torch.tensor(avg_loss, device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = (loss_tensor / self.world_size).item()
            except Exception:
                pass

        # 记录测试指标（动态创建字典键）
        metrics_key = f'{dataset_type}_metrics'
        if metrics_key not in self.training_history:
            self.training_history[metrics_key] = defaultdict(list)

        for metric_name, value in avg_metrics.items():
            self.training_history[metrics_key][metric_name].append(value)

        # 生成所有预测结果和可视化：仅rank0执行，避免DDP不一致和重复输出
        if (not self.use_distributed) or (self.rank == 0):
            self.prediction_handler.process_all_results()

        # 打印评估结果：仅rank0打印
        if (not self.use_distributed) or (self.rank == 0):
            self._print_evaluation_results(dataset_type, avg_loss, avg_metrics)

        # 保存评估结果：仅rank0保存
        if (not self.use_distributed) or (self.rank == 0):
            self._save_evaluation_results(dataset_type, avg_loss, avg_metrics)

        return avg_loss, avg_metrics

    def evaluate_without_clearing(self, graphs: List[HeteroData],
                                  dataset_type: str = 'test') -> Tuple[float, Dict]:
        """评估模型但不清除预测结果，用于收集多个数据集的预测"""
        if self.use_distributed and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass

        if not hasattr(self, 'prediction_handler') or self.prediction_handler is None:
            self.prediction_handler = PredictionHandler(self.save_dir, self.rank, self.world_size)

        if self.best_model_state is not None:
            self._load_state_dict_safely(self.best_model_state['model_state_dict'], strict=False)

        if (not self.use_distributed) or (self.rank == 0):
            logger.info(f"Evaluating on {dataset_type} set with {len(graphs)} graphs")

        avg_loss, avg_metrics = self.validate(graphs, dataset_type)

        metrics_key = f'{dataset_type}_metrics'
        if metrics_key not in self.training_history:
            self.training_history[metrics_key] = defaultdict(list)

        for metric_name, value in avg_metrics.items():
            self.training_history[metrics_key][metric_name].append(value)

        if (not self.use_distributed) or (self.rank == 0):
            self._print_evaluation_results(dataset_type, avg_loss, avg_metrics)
            self._save_evaluation_results(dataset_type, avg_loss, avg_metrics)

        if self.use_distributed and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass

        return avg_loss, avg_metrics

    def _print_evaluation_results(self, dataset_type: str, loss: float, metrics: Dict):
        """打印评估结果"""
        logger.info(f"\n=== {dataset_type.title()} Set Evaluation Results ===")
        logger.info(f"Average Loss: {loss:.6f}")

        # 打印总体指标
        logger.info("\nOverall Metrics:")
        for metric in ['mse', 'mae', 'rmse', 'r2']:
            key = f'overall_{metric}'
            if key in metrics:
                logger.info(f"  {metric.upper()}: {metrics[key]:.6f}")

        # 打印各节点类型指标
        node_types = ['production', 'delivery', 'inventory', 'virtual', 'capacity']
        for node_type in node_types:
            node_metrics = {k: v for k, v in metrics.items() if k.startswith(f'{node_type}_')}
            if node_metrics:
                logger.info(f"\n{node_type.title()} Metrics:")
                for metric_name, value in node_metrics.items():
                    short_name = metric_name.replace(f'{node_type}_', '')
                    logger.info(f"  {short_name.upper()}: {value:.6f}")

    def _save_evaluation_results(self, dataset_type: str, loss: float, metrics: Dict):
        """保存评估结果"""

        def to_builtin(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: to_builtin(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_builtin(v) for v in obj]
            else:
                return obj

        # 使用智能配置序列化器
        config_dict = ConfigSerializer.serialize_config(self.config)

        results = {
            'dataset_type': to_builtin(dataset_type),
            'loss': to_builtin(loss),
            'metrics': to_builtin(metrics),
            'timestamp': to_builtin(datetime.now().isoformat()),
            'config': to_builtin(config_dict)
        }

        results_path = self.save_dir / f'{dataset_type}_evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Saved {dataset_type} evaluation results to {results_path}")

    def _move_state_dict_to_cpu(self, state_dict):
        """🔧 NPU适配：将state_dict中的所有tensor移到CPU"""
        cpu_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                # 🔧 修复NPU检测：使用device.type而不是is_npu()
                if v.device.type in ('cuda', 'npu'):
                    cpu_state_dict[k] = v.cpu()
                else:
                    cpu_state_dict[k] = v
            elif isinstance(v, dict):
                cpu_state_dict[k] = self._move_state_dict_to_cpu(v)
            elif isinstance(v, list):
                # 🔧 修复NPU检测：使用device.type而不是is_npu()
                cpu_state_dict[k] = [self._move_state_dict_to_cpu(item) if isinstance(item, dict) else 
                                     item.cpu() if isinstance(item, torch.Tensor) and item.device.type in ('cuda', 'npu') else item
                                     for item in v]
            else:
                cpu_state_dict[k] = v
        return cpu_state_dict

    def save_model(self):
        """保存模型状态 - NPU/CUDA适配"""
        # 使用智能配置序列化器
        config_dict = ConfigSerializer.serialize_config(self.config)

        # 始终以去DDP后的权重保存，避免后续加载时key不匹配
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 🔧 NPU适配：将所有tensor移到CPU再保存
        model_state_cpu = self._move_state_dict_to_cpu(model_to_save.state_dict())
        optimizer_state_cpu = self._move_state_dict_to_cpu(self.optimizer.state_dict()) if self.optimizer else None
        
        save_dict = {
            'model_state_dict': model_state_cpu,
            'optimizer_state_dict': optimizer_state_cpu,
            'config': config_dict,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'experiment_name': self.experiment_name
        }
        torch.save(save_dict, self.best_model_path)
        logger.debug(f"Saved model to {self.best_model_path}")

    def load_model(self, model_path: str = None):
        """加载模型状态"""
        if model_path is None:
            model_path = self.best_model_path

        if not Path(model_path).exists():
            logger.warning(f"Model file not found: {model_path}")
            return False

        checkpoint = torch.load(model_path, map_location=self.device)
        self._load_state_dict_safely(checkpoint['model_state_dict'])

        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Loaded model from {model_path}")
        return True

    def _load_state_dict_safely(self, state_dict: dict, strict: bool = True) -> bool:
        """根据当前模型是否为DDP，自动适配带/不带'module.'前缀的权重键，安全加载state_dict。"""
        try:
            is_ddp_model = hasattr(self.model, 'module')
            has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

            if is_ddp_model and not has_module_prefix:
                # 模型是DDP，但权重没有module前缀 → 直接加载到module
                self.model.module.load_state_dict(state_dict, strict=strict)
                return True
            elif (not is_ddp_model) and has_module_prefix:
                # 模型非DDP，但权重带module前缀 → 去前缀
                stripped = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
                self.model.load_state_dict(stripped, strict=strict)
                return True
            else:
                # 其他情况：直接按当前模型形态加载
                self.model.load_state_dict(state_dict, strict=strict)
                return True
        except Exception as e:
            logger.error(f"Safe load state_dict failed: {e}")
            return False

    def save_training_history(self):
        """保存训练历史"""
        history_path = self.save_dir / 'training_history.json'

        # 转换numpy类型为python原生类型
        serializable_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, dict):
                serializable_history[key] = {
                    k: [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
                    for k, v in value.items()
                }
            else:
                serializable_history[key] = [
                    float(x) if isinstance(x, (np.float32, np.float64)) else x
                    for x in value
                ]

        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)

        logger.info(f"Saved training history to {history_path}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 损失曲线
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss', color='blue')
        if self.training_history['val_loss']:
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss', color='orange')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 学习率曲线
        axes[0, 1].plot(self.training_history['learning_rate'], color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)

        # MSE指标
        if 'overall_mse' in self.training_history['val_metrics']:
            axes[1, 0].plot(self.training_history['val_metrics']['overall_mse'],
                            label='Val MSE', color='red')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MSE')
            axes[1, 0].set_title('Validation MSE')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # R²指标
        if 'overall_r2' in self.training_history['val_metrics']:
            axes[1, 1].plot(self.training_history['val_metrics']['overall_r2'],
                            label='Val R²', color='purple')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('R²')
            axes[1, 1].set_title('Validation R²')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved training curves to {plot_path}")

    def get_experiment_summary(self) -> Dict:
        """获取实验摘要"""
        # 使用智能配置序列化器
        config_dict = ConfigSerializer.serialize_config(self.config)

        return {
            'experiment_name': self.experiment_name,
            'config': config_dict,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'device': str(self.device),
            'save_dir': str(self.save_dir)
        }

