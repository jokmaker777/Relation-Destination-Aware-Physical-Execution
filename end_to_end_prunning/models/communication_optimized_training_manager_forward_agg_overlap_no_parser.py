#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
communication_optimized_training_manager.py

一个不污染原始 training_manager.py 的通信/划分优化测试版 TrainingManager。

使用方式：
1. 将本文件复制到：
   end_to_end_prunning/models/communication_optimized_training_manager.py

2. 在新的训练测试脚本中使用：
   from end_to_end_prunning.models.communication_optimized_training_manager import CommunicationOptimizedTrainingManager

3. 支持的 partition_strategy：
   - balanced_edge
       原始 TrainingManager 中的连续边均分 baseline。
   - superblock_edge_count
       Target-group preserving SuperBlock + 边数负载贪婪分配。
   - superblock_calibrated_cost
       Target-group preserving SuperBlock + profiling calibrated cost 贪婪分配。
       若未提供 profiling 文件，则自动回退为 edge_count cost。
   - superblock_vector_balance
       Target-group preserving SuperBlock + vector-balanced GPU assignment。
       在总边数负载之外，同时近似平衡不同 edge type 在各 GPU 上的分布。

注意：
- 本文件不改 enhanced_supply_chain_graph.py，不改 gcn_model.py，不改 prediction_handler.py。
- 训练时仍然使用原始 SupplyChainGCN 与原始 PredictionHandler。
- 当前版本用于测试“边划分/通信冲突优化策略”对真实训练时间、显存、loss 的影响。
- 当 enable_forward_aggregation=True 时，本版本为两类被划分的 labeled relations
  实现基于邻居 SUM + degree SUM 的精确默认 SAGEConv(mean) 跨 rank 聚合，并支持
  destination-chunk 异步 AllReduce 与本地消息计算重叠。
- 该精确路径仅适用于当前项目使用的默认 SAGEConv(mean)。若模型改为其他聚合器，
  代码会显式报错，而不会静默回退为不等价语义。
"""

from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from .training_manager import TrainingManager

try:
    from stage_timing_utils_no_parser import StageTimer
except Exception:
    try:
        from end_to_end_prunning.models.stage_timing_utils_no_parser import StageTimer
    except Exception:
        StageTimer = None

logger = logging.getLogger(__name__)


EdgeType = Tuple[str, str, str]

LABELED_EDGE_TYPES: List[EdgeType] = [
    ("inventory", "to", "virtual"),
    ("inventory", "to_transport_labeled", "inventory"),
]

# Printed on the GPU server so stale PyCharm deployments are immediately
# distinguishable from the collective-ordering fix in this file.
COMMOPT_PATCH_VERSION = "20261004-active-destination-compact-v1"


def edge_type_to_str(edge_type: EdgeType) -> str:
    return "__".join(edge_type)


def parse_edge_type(edge_type_str: str) -> EdgeType:
    parts = edge_type_str.split("__")
    if len(parts) != 3:
        raise ValueError(f"Invalid edge type string: {edge_type_str}")
    return (parts[0], parts[1], parts[2])


@dataclass
class TargetGroup:
    group_id: int
    edge_type: EdgeType
    dst_node_type: str
    dst_node_id: int
    edge_ids: torch.Tensor
    num_edges: int


@dataclass
class SuperBlock:
    block_id: int
    edge_type: EdgeType
    dst_node_type: str
    target_node_ids: List[int]
    edge_ids: torch.Tensor
    num_edges: int
    num_target_groups: int
    calibrated_cost: float




class _DetachedForwardAllReduceWithBackward(torch.autograd.Function):
    """
    Attach an asynchronously reduced detached buffer to the autograd graph.

    Forward communication is launched outside this Function so that NCCL can run
    concurrently with the next destination chunk's local message computation.
    Backward performs a SUM all-reduce on grad_output using a dedicated process
    group. With DDP's usual gradient averaging, this gives the correct gradient
    for the global mean of rank-local losses in the edge-partitioned forward.
    """

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, reduced_buffer: torch.Tensor, process_group):
        ctx.process_group = process_group
        # reduced_buffer is detached and already contains SUM_p local_tensor_p.
        return reduced_buffer

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_local = grad_output.contiguous()
        if dist.is_available() and dist.is_initialized() and dist.get_world_size(group=ctx.process_group) > 1:
            dist.all_reduce(grad_local, op=dist.ReduceOp.SUM, group=ctx.process_group)
        return grad_local, None, None


class _DetachedChunkedForwardAllReduceWithBackward(torch.autograd.Function):
    """One autograd node that reduces all chunk gradients in a fixed order.

    Inputs after ``process_group`` are ``local_0..local_n`` followed by the
    already reduced detached buffers ``reduced_0..reduced_n``.  Returning all
    chunks from one Function prevents rank-local autograd readiness from
    reordering the per-chunk backward collectives.
    """

    @staticmethod
    def forward(ctx, process_group, *tensors: torch.Tensor):
        if len(tensors) == 0 or len(tensors) % 2 != 0:
            raise ValueError("chunked all-reduce requires local/reduced tensor pairs")
        num_chunks = len(tensors) // 2
        local_tensors = tensors[:num_chunks]
        reduced_tensors = tensors[num_chunks:]
        ctx.process_group = process_group
        ctx.num_chunks = num_chunks
        ctx.save_for_backward(*local_tensors)
        return tuple(reduced_tensors)

    @staticmethod
    def backward(ctx, *grad_outputs: Optional[torch.Tensor]):
        local_tensors = ctx.saved_tensors
        grad_locals: List[torch.Tensor] = []
        for grad_output, local_tensor in zip(grad_outputs, local_tensors):
            grad_local = (
                torch.zeros_like(local_tensor)
                if grad_output is None
                else grad_output.contiguous()
            )
            if (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size(group=ctx.process_group) > 1
            ):
                dist.all_reduce(
                    grad_local, op=dist.ReduceOp.SUM, group=ctx.process_group
                )
            grad_locals.append(grad_local)
        return (None, *grad_locals, *([None] * int(ctx.num_chunks)))


class CommunicationOptimizedTrainingManager(TrainingManager):
    """
    在原始 TrainingManager 基础上增加 SuperBlock 划分策略。

    新增配置项均从 config.training_config 中读取，如果没有就使用默认值：
      - partition_strategy:
          balanced_edge / superblock_edge_count / superblock_calibrated_cost
      - target_super_block_size: int
      - superblock_profile_path: str
      - superblock_assignment_output_dir: str
      - save_partition_plan: bool
    """

    def __init__(self, *args, **kwargs):
        """
        修复原始 TrainingManager 在非分布式模式下的 rank 初始化问题。

        原始 TrainingManager.__init__ 末尾会直接执行：
            PredictionHandler(self.save_dir, self.rank, self.world_size)

        但当 use_distributed=False 时，原始逻辑不会调用 _init_distributed()，
        因此 self.rank 没有被创建。这里在调用 super().__init__ 前先给 rank /
        world_size / local_rank 设置安全默认值，避免单卡调试时报 AttributeError。
        """
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", str(kwargs.get("world_size", 1) or 1)))

        # 分区缓存：同一个 graph / strategy / rank 在多个 epoch 中只划分一次。
        # 这是关键优化，否则 SuperBlock 构造会在每个 epoch 前重复扫描 438 万 target groups。
        self._superblock_partition_cache = {}

        super().__init__(*args, **kwargs)

        # ------------------------------------------------------------
        # Steady-epoch data-stage optimization diagnostics.
        #
        # In the original flow, each epoch calls _move_data_to_device(graph).
        # The measured StageTimer result showed that data_sec is a large part
        # of steady epoch time.  This optional cache keeps the rank-local graph
        # shard on device after the first move and reuses it in later epochs.
        #
        # This is intentionally controlled by config flags so that the runner
        # can compare baseline vs. persistent device cache without changing
        # model semantics.
        # ------------------------------------------------------------
        self._persistent_device_graph_cache: Dict[Tuple[int, str], HeteroData] = {}
        self._persistent_device_cache_hits = 0
        self._persistent_device_cache_misses = 0

        # Exact forward aggregation overlap state. Only LABELED_EDGE_TYPES are
        # edge-partitioned; all other relations remain replicated and are not
        # communicated. A dedicated process group isolates these collectives
        # from DDP gradient bucket collectives.
        self._forward_agg_group = None
        self._forward_agg_groups: Dict[EdgeType, Any] = {}
        self._forward_agg_chunk_plan_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._forward_agg_compact_plan_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._forward_agg_global_degree_cache: Dict[Tuple[Any, ...], torch.Tensor] = {}
        # The generation is a deterministic, rank-independent cache namespace.
        # Pointer-based cache keys are unsafe here: a cache hit controls whether
        # collectives are issued, and Python object ids / device pointers are not
        # comparable across ranks.
        self._forward_agg_graph_generation = 0
        self._forward_agg_active_graph_key: Optional[Tuple[int, str]] = None

        # Every forward aggregation result is attached to the loss through a
        # zero-valued scalar.  This makes autograd visit every custom collective
        # on every rank even when a rank has no labelled edge using that output.
        self._forward_agg_backward_anchors: List[torch.Tensor] = []

        # 再保险：如果父类初始化过程中没有设置这些属性，则补齐。
        if not hasattr(self, "rank"):
            self.rank = int(os.environ.get("RANK", "0"))
        if not hasattr(self, "local_rank"):
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not hasattr(self, "world_size"):
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # Lightweight stage timing for compute-communication exposure analysis.
        # This does not change training semantics. It only records stage-level wall time.
        self.stage_timer = None
        try:
            enable_stage_timing = bool(getattr(self.config.training_config, "enable_stage_timing", True))
            if enable_stage_timing and StageTimer is not None:
                try:
                    timing_output_dir = str(self._get_assignment_output_dir())
                except Exception:
                    timing_output_dir = str(getattr(self, "save_dir", "."))
                self.stage_timer = StageTimer(
                    rank=int(getattr(self, "rank", 0)),
                    world_size=int(getattr(self, "world_size", 1)),
                    device=getattr(self, "device", None),
                    output_dir=timing_output_dir,
                    enabled=True,
                    sync_cuda=True,
                    eta_comm=float(getattr(self.config.training_config, "overlap_eta_comm", 0.5)),
                    eta_data=float(getattr(self.config.training_config, "overlap_eta_data", 0.3)),
                )
                logger.info(f"[Rank {self.rank}] StageTimer enabled, output_dir={timing_output_dir}")
        except Exception as e:
            logger.warning(f"[Rank {getattr(self, 'rank', 0)}] failed to initialize StageTimer: {e}")
            self.stage_timer = None

    def _make_superblock_cache_key(self, graph: HeteroData, strategy: str) -> Tuple:
        """
        同一张图、同一种策略、同一个 rank/world_size 下，SuperBlock 分区结果是确定的。
        因此可以跨 epoch 复用，不需要每一轮重新 build target groups / pack blocks / assign。
        """
        graph_date = getattr(graph, "graph_date", None) or getattr(graph, "date", None) or "unknown_graph"
        profile_path = self._get_profile_path() if strategy == "superblock_calibrated_cost" else ""
        return (
            str(graph_date),
            int(id(graph)),
            strategy,
            int(getattr(self, "world_size", 1)),
            int(getattr(self, "rank", 0)),
            int(self._get_target_super_block_size()),
            str(profile_path or ""),
        )

    def _distribute_graphs_balanced(self, graphs: List[HeteroData]) -> List[HeteroData]:
        """
        重写原始图分配逻辑，只新增 superblock_* 策略。
        其他策略仍然走父类逻辑，不影响原始功能。
        """
        if not graphs:
            return []

        if len(graphs) >= self.world_size:
            return graphs[self.rank::self.world_size]

        if len(graphs) == 1:
            enable_partition = bool(getattr(self.config.training_config, "enable_graph_partition", True))

            if enable_partition:
                partition_strategy = getattr(self.config.training_config, "partition_strategy", "balanced_edge")

                try:
                    if partition_strategy in (
                            "superblock_edge_count",
                            "superblock_calibrated_cost",
                            "superblock_vector_balance",
                            "superblock_cost_balance",
                    ):
                        cache_key = self._make_superblock_cache_key(graphs[0], partition_strategy)

                        if cache_key in self._superblock_partition_cache:
                            cached = self._superblock_partition_cache[cache_key]
                            subgraph = cached["subgraph"]
                            mapping_info = cached["mapping_info"]
                            logger.info(
                                f"[Rank {self.rank}] reuse cached {partition_strategy} partition: "
                                f"edge_alloc={mapping_info.get('edge_type_allocation', {})}"
                            )
                        else:
                            subgraph, mapping_info = self._partition_graph_by_superblocks(
                                graph=graphs[0],
                                world_size=self.world_size,
                                rank=self.rank,
                                strategy=partition_strategy,
                            )
                            self._superblock_partition_cache[cache_key] = {
                                "subgraph": subgraph,
                                "mapping_info": mapping_info,
                            }

                    else:
                        # 原始策略直接交给父类，避免复制原有复杂逻辑。
                        return super()._distribute_graphs_balanced(graphs)

                    if self._validate_subgraph(subgraph):
                        self._current_mapping_info = mapping_info
                        self._original_graph = graphs[0]
                        self.use_graph_partition = True
                        return [subgraph]
                    else:
                        logger.warning(f"[Rank {self.rank}] SuperBlock subgraph invalid, fallback to replication")

                except Exception as e:
                    logger.error(f"[Rank {self.rank}] SuperBlock partition failed: {e}", exc_info=True)
                    logger.warning(f"[Rank {self.rank}] fallback to parent partition logic")
                    return super()._distribute_graphs_balanced(graphs)

            return graphs

        # 多图场景保持原始轮转分配
        local_graphs = []
        for i in range(self.rank, len(graphs), self.world_size):
            local_graphs.append(graphs[i])

        if not local_graphs and graphs:
            local_graphs = [graphs[-1]]

        return local_graphs


    # ============================================================
    # Steady-epoch data movement diagnostics / persistent GPU cache
    # ============================================================

    def _is_persistent_device_cache_enabled(self) -> bool:
        """Whether to reuse the rank-local graph shard on device across epochs."""
        try:
            return bool(getattr(self.config.training_config, "enable_persistent_device_cache", False))
        except Exception:
            return False

    def _append_data_move_record(self, record: Dict[str, Any]):
        """Append data movement/cache diagnostics for later steady-epoch analysis."""
        try:
            out_dir = self._get_assignment_output_dir()
            path = out_dir / f"data_move_timing_rank{getattr(self, 'rank', 0)}.csv"
            df = pd.DataFrame([record])
            if path.exists():
                df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.debug(f"failed to append data move record: {e}")

    def _move_data_to_device(self, graph: HeteroData) -> HeteroData:
        """
        Optional persistent device cache for steady epochs.

        Baseline mode:
            call parent _move_data_to_device every epoch.

        Cache mode:
            first call moves rank-local graph shard to device and stores it;
            later epochs return the cached device graph directly.

        The method records cache hit/miss and elapsed time in:
            data_move_timing_rank*.csv
        """
        t0 = time.perf_counter()
        enable_cache = self._is_persistent_device_cache_enabled()
        device_str = str(getattr(self, "device", "unknown_device"))
        key = (int(id(graph)), device_str)

        # All ranks process the same logical graph sequence.  Incrementing a
        # small generation counter when the local graph object changes produces
        # identical cache namespaces without embedding rank-local pointers in
        # keys that govern collective execution.
        if self._forward_agg_active_graph_key != key:
            self._forward_agg_active_graph_key = key
            self._forward_agg_graph_generation += 1
            self._forward_agg_chunk_plan_cache.clear()
            self._forward_agg_compact_plan_cache.clear()
            self._forward_agg_global_degree_cache.clear()

        cache_hit = False
        if enable_cache and key in self._persistent_device_graph_cache:
            moved_graph = self._persistent_device_graph_cache[key]
            cache_hit = True
            self._persistent_device_cache_hits += 1
        else:
            moved_graph = super()._move_data_to_device(graph)
            if enable_cache:
                self._persistent_device_graph_cache[key] = moved_graph
                self._persistent_device_cache_misses += 1
            else:
                self._persistent_device_cache_misses += 1

        elapsed = float(time.perf_counter() - t0)

        try:
            epoch_idx = int(getattr(self, "_stage_epoch_counter", 0))
        except Exception:
            epoch_idx = -1

        self._append_data_move_record({
            "epoch": int(epoch_idx),
            "rank": int(getattr(self, "rank", 0)),
            "world_size": int(getattr(self, "world_size", 1)),
            "device": device_str,
            "enable_persistent_device_cache": bool(enable_cache),
            "cache_hit": bool(cache_hit),
            "cache_size": int(len(getattr(self, "_persistent_device_graph_cache", {}))),
            "cache_hits_total": int(getattr(self, "_persistent_device_cache_hits", 0)),
            "cache_misses_total": int(getattr(self, "_persistent_device_cache_misses", 0)),
            "data_move_elapsed_sec": elapsed,
            "timestamp": time.time(),
        })

        return moved_graph

    # ============================================================
    # Forward-loss chunk / edge-block diagnostics
    # ============================================================

    def _is_forward_detail_enabled(self) -> bool:
        try:
            return bool(getattr(self.config.training_config, "enable_forward_loss_detail_timing", True))
        except Exception:
            return True

    def _is_chunked_edge_prediction_enabled(self) -> bool:
        try:
            return bool(getattr(self.config.training_config, "enable_chunked_edge_prediction", False))
        except Exception:
            return False

    def _is_chunked_loss_enabled(self) -> bool:
        try:
            return bool(getattr(self.config.training_config, "enable_chunked_loss", False))
        except Exception:
            return False

    def _get_edge_prediction_chunk_size(self) -> int:
        return int(getattr(self.config.training_config, "edge_prediction_chunk_size", 0) or 0)

    def _get_loss_chunk_size(self) -> int:
        return int(getattr(self.config.training_config, "loss_chunk_size", 0) or 0)


    # ============================================================
    # Relation-wise / edge-type-wise message-passing scheduling
    # ============================================================

    def _is_relation_scheduled_conv_enabled(self) -> bool:
        try:
            return bool(getattr(self.config.training_config, "enable_relation_scheduled_conv", False))
        except Exception:
            return False

    def _is_relation_timing_enabled(self) -> bool:
        try:
            return bool(getattr(self.config.training_config, "enable_relation_forward_timing", False))
        except Exception:
            return False

    def _get_relation_schedule_order(self) -> str:
        try:
            return str(getattr(self.config.training_config, "relation_schedule_order", "edge_count_desc"))
        except Exception:
            return "edge_count_desc"

    def _get_relation_filter_mode(self) -> str:
        try:
            return str(getattr(self.config.training_config, "relation_filter_mode", "all"))
        except Exception:
            return "all"

    def _get_relation_group_size(self) -> int:
        try:
            return int(getattr(self.config.training_config, "relation_group_size", 0) or 0)
        except Exception:
            return 0

    def _append_relation_forward_record(self, record: Dict[str, Any]):
        try:
            out_dir = self._get_assignment_output_dir()
            path = out_dir / f"relation_forward_timing_rank{getattr(self, 'rank', 0)}.csv"
            df = pd.DataFrame([record])
            if path.exists():
                df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.debug(f"failed to append relation forward record: {e}")

    def _append_relation_chunk_record(self, record: Dict[str, Any]):
        try:
            out_dir = self._get_assignment_output_dir()
            path = out_dir / f"relation_chunk_timing_rank{getattr(self, 'rank', 0)}.csv"
            df = pd.DataFrame([record])
            if path.exists():
                df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.debug(f"failed to append relation chunk record: {e}")

    def _relation_edge_type_allowed(self, edge_type: EdgeType) -> bool:
        """
        Relation filtering is an experimental scheduling knob.

        mode='all' preserves the current model semantics.
        Other modes intentionally change the message-passing graph and should be
        interpreted as upper-bound / ablation experiments, not as exact full-graph
        training unless downstream accuracy is revalidated.
        """
        mode = self._get_relation_filter_mode()
        if mode in ("", "all", "none"):
            return True
        if mode == "labeled_only":
            return edge_type in LABELED_EDGE_TYPES
        if mode == "labeled_and_reverse":
            return edge_type in LABELED_EDGE_TYPES or edge_type == ("inventory", "to_transport_reverse", "inventory")
        if mode == "inventory_dst_only":
            return len(edge_type) == 3 and edge_type[2] == "inventory"
        if mode == "virtual_or_inventory_dst":
            return len(edge_type) == 3 and edge_type[2] in ("virtual", "inventory")
        return True

    def _relation_order_key(self, edge_type: EdgeType, edge_count: int, original_pos: int):
        order = self._get_relation_schedule_order()
        et_str = edge_type_to_str(edge_type)
        is_labeled = 0 if edge_type in LABELED_EDGE_TYPES else 1
        if order == "original":
            return (original_pos,)
        if order == "edge_count_asc":
            return (edge_count, original_pos)
        if order == "edge_count_desc":
            return (-edge_count, original_pos)
        if order == "labeled_first":
            return (is_labeled, -edge_count, original_pos)
        if order == "transport_first":
            is_transport = 0 if "transport" in et_str else 1
            return (is_transport, -edge_count, original_pos)
        if order == "virtual_first":
            is_virtual = 0 if edge_type == ("inventory", "to", "virtual") else 1
            return (is_virtual, -edge_count, original_pos)
        return (-edge_count, original_pos)

    def _get_conv_for_edge_type(self, conv_layer, edge_type: EdgeType):
        """Robustly retrieve the PyG relation-specific conv module."""
        if not hasattr(conv_layer, "convs"):
            return None
        convs = conv_layer.convs
        candidates = [
            edge_type,
            edge_type_to_str(edge_type),
            "__".join(edge_type),
            str(edge_type),
        ]
        for key in candidates:
            try:
                if key in convs:
                    return convs[key]
            except Exception:
                pass
        try:
            et_str = edge_type_to_str(edge_type)
            for key, module in convs.items():
                if str(key) == et_str or str(key) == "__".join(edge_type) or str(key) == str(edge_type):
                    return module
        except Exception:
            pass
        return None

    def _aggregate_relation_outputs(self, outs: List[torch.Tensor], aggr: str):
        if not outs:
            return None
        if len(outs) == 1:
            return outs[0]
        aggr = str(aggr or "mean")
        if aggr == "sum":
            out = outs[0]
            for x in outs[1:]:
                out = out + x
            return out
        if aggr == "max":
            return torch.stack(outs, dim=0).max(dim=0).values
        if aggr == "min":
            return torch.stack(outs, dim=0).min(dim=0).values
        # Default PyG HeteroConv aggr in the project is mean.
        return torch.stack(outs, dim=0).mean(dim=0)


    # ============================================================
    # Exact cross-rank forward aggregation + compute/communication overlap
    # ============================================================

    def _reset_forward_agg_backward_anchors(self) -> None:
        self._forward_agg_backward_anchors = []

    def _register_forward_agg_backward_anchor(self, tensor: torch.Tensor) -> None:
        """Force every rank to traverse every custom all-reduce in backward."""
        if int(getattr(self, "world_size", 1)) <= 1:
            return
        # The scalar is exactly zero but retains a differentiable dependency on
        # the complete relation/chunk output.
        self._forward_agg_backward_anchors.append(tensor.sum() * 0.0)

    def _attach_forward_agg_backward_anchors(self, loss: torch.Tensor) -> torch.Tensor:
        anchors = list(getattr(self, "_forward_agg_backward_anchors", []))
        if not anchors:
            return loss
        anchor = torch.stack(
            [value.to(device=loss.device, dtype=loss.dtype) for value in anchors]
        ).sum()
        return loss + anchor

    def _is_forward_aggregation_enabled(self) -> bool:
        try:
            return bool(getattr(self.config.training_config, "enable_forward_aggregation", False))
        except Exception:
            return False

    def _is_forward_aggregation_overlap_enabled(self) -> bool:
        try:
            return bool(getattr(self.config.training_config, "enable_forward_agg_overlap", False))
        except Exception:
            return False

    def _get_forward_aggregation_operator(self) -> str:
        """Return the exact physical communication operator.

        ``dense`` preserves the existing full-destination collective. ``compact``
        builds one stable active-destination map per relation and communicates
        only rows touched by at least one edge.  Both operators implement the
        same SAGEConv(mean) semantics.
        """
        try:
            value = str(
                getattr(
                    self.config.training_config,
                    "forward_aggregation_operator",
                    "dense",
                )
            ).strip().lower()
        except Exception:
            value = "dense"
        aliases = {
            "full": "dense",
            "dense_allreduce": "dense",
            "active": "compact",
            "active_destination": "compact",
            "compact_allreduce": "compact",
        }
        value = aliases.get(value, value)
        if value not in {"dense", "compact"}:
            raise ValueError(f"Unsupported forward_aggregation_operator={value!r}")
        return value

    def _get_forward_aggregation_num_chunks(self) -> int:
        try:
            return max(1, int(getattr(self.config.training_config, "aggregation_num_chunks", 1) or 1))
        except Exception:
            return 1

    def _get_forward_aggregation_chunk_policy(self) -> str:
        try:
            value = str(
                getattr(
                    self.config.training_config,
                    "aggregation_chunk_policy",
                    "destination_range",
                )
            ).strip().lower()
        except Exception:
            value = "destination_range"
        aliases = {
            "destination": "destination_range",
            "range": "destination_range",
            "equal_range": "destination_range",
            "edge": "global_edge_balanced",
            "edge_balanced": "global_edge_balanced",
            "global_edge": "global_edge_balanced",
        }
        value = aliases.get(value, value)
        if value not in {"destination_range", "global_edge_balanced"}:
            raise ValueError(f"Unsupported aggregation_chunk_policy={value!r}")
        return value

    def _get_forward_aggregation_chunk_order(self) -> str:
        try:
            value = str(
                getattr(
                    self.config.training_config,
                    "aggregation_chunk_order",
                    "natural",
                )
            ).strip().lower()
        except Exception:
            value = "natural"
        aliases = {
            "range": "natural",
            "heavy_first": "largest_first",
            "light_first": "smallest_first",
            "greedy": "overlap_greedy",
        }
        value = aliases.get(value, value)
        if value not in {
            "natural",
            "largest_first",
            "smallest_first",
            "zigzag",
            "overlap_greedy",
        }:
            raise ValueError(f"Unsupported aggregation_chunk_order={value!r}")
        return value

    def _get_forward_aggregation_max_inflight(self) -> int:
        try:
            return max(
                1,
                int(
                    getattr(
                        self.config.training_config,
                        "aggregation_max_inflight",
                        1,
                    )
                    or 1
                ),
            )
        except Exception:
            return 1

    def _is_forward_aggregation_fuse_degree_enabled(self) -> bool:
        try:
            return bool(
                getattr(
                    self.config.training_config,
                    "aggregation_fuse_degree",
                    False,
                )
            )
        except Exception:
            return False

    def _get_forward_aggregation_mode(self) -> str:
        try:
            return str(getattr(self.config.training_config, "steady_experiment_mode", "sequential_aggregation"))
        except Exception:
            return "sequential_aggregation"

    def _is_forward_aggregation_timing_enabled(self) -> bool:
        try:
            return bool(getattr(self.config.training_config, "enable_forward_agg_timing", True))
        except Exception:
            return True

    def _ensure_forward_aggregation_group(self):
        if not self._is_forward_aggregation_enabled():
            return None
        if not (dist.is_available() and dist.is_initialized()):
            return None
        if int(getattr(self, "world_size", 1)) <= 1:
            return None
        if not getattr(self, "_forward_agg_groups", {}):
            # Create one communicator per partitioned relation.  Relation-local
            # autograd paths can become ready in a different order across ranks;
            # separate groups make that ordering irrelevant.  Every rank creates
            # the groups here in the same metadata order before the first forward.
            ranks = list(range(int(getattr(self, "world_size", dist.get_world_size()))))
            self._forward_agg_groups = {}
            for edge_type in LABELED_EDGE_TYPES:
                self._forward_agg_groups[edge_type] = dist.new_group(ranks=ranks)
            self._forward_agg_group = self._forward_agg_groups[LABELED_EDGE_TYPES[0]]
            logger.info(
                f"[Rank {getattr(self, 'rank', 0)}] "
                f"commopt_patch={COMMOPT_PATCH_VERSION}, "
                "created dedicated forward aggregation groups, "
                f"relations={[edge_type_to_str(et) for et in LABELED_EDGE_TYPES]}, ranks={ranks}"
            )
        return self._forward_agg_group

    def _get_forward_aggregation_group(self, edge_type: EdgeType):
        self._ensure_forward_aggregation_group()
        groups = getattr(self, "_forward_agg_groups", {})
        return groups.get(edge_type, self._forward_agg_group)

    def _append_forward_agg_overlap_record(self, record: Dict[str, Any]):
        try:
            out_dir = self._get_assignment_output_dir()
            path = out_dir / f"forward_agg_overlap_timing_rank{getattr(self, 'rank', 0)}.csv"
            df = pd.DataFrame([record])
            if path.exists():
                df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.debug(f"failed to append forward aggregation overlap record: {e}")

    def _validate_exact_sage_conv(self, conv, edge_type: EdgeType):
        """Fail loudly rather than silently changing GNN semantics."""
        if conv.__class__.__name__ != "SAGEConv":
            raise TypeError(
                f"Exact distributed aggregation currently supports SAGEConv only, got {conv.__class__.__name__} for {edge_type}"
            )
        if not hasattr(conv, "lin_l"):
            raise AttributeError(f"SAGEConv for {edge_type} has no lin_l; unsupported PyG version")
        aggr_name = str(getattr(conv, "aggr", "mean")).lower()
        if "mean" not in aggr_name:
            # The project's SupplyChainGCN uses default SAGEConv(mean). Supporting
            # max/min exactly would require different distributed reductions.
            raise ValueError(f"Exact overlap path requires SAGEConv mean aggregation, got {aggr_name} for {edge_type}")

    def _get_compact_destination_plan(
        self,
        edge_type: EdgeType,
        edge_index: torch.Tensor,
        num_dst_nodes: int,
    ) -> Dict[str, Any]:
        """Build and cache the global active-destination map for one relation.

        The one-time dense degree reduction is preparation metadata, not an
        epoch collective.  Every later epoch communicates an ``A_r x d`` tensor
        instead of ``N_r x d``, where ``A_r`` is the number of active
        destinations.  ``edge_compact_ids`` is rank-local while
        ``active_destination_ids`` and ``global_degree`` are identical on every
        rank in the relation-specific process group.
        """
        num_dst_nodes = int(num_dst_nodes)
        key = (
            int(getattr(self, "_forward_agg_graph_generation", 0)),
            edge_type_to_str(edge_type),
            num_dst_nodes,
            str(edge_index.device),
        )
        cached = self._forward_agg_compact_plan_cache.get(key)
        if cached is not None:
            return cached

        started = time.perf_counter()
        num_edges = int(edge_index.shape[1]) if edge_index.dim() == 2 else 0
        dst = (
            edge_index[1].long().contiguous()
            if num_edges > 0
            else torch.empty((0,), dtype=torch.long, device=edge_index.device)
        )
        local_degree = torch.bincount(dst, minlength=max(0, num_dst_nodes)).to(
            dtype=torch.int64
        )
        global_degree = local_degree.clone()
        group = self._get_forward_aggregation_group(edge_type)
        world = int(getattr(self, "world_size", 1))
        if world > 1 and group is not None and global_degree.numel() > 0:
            dist.all_reduce(global_degree, op=dist.ReduceOp.SUM, group=group)

        active_ids = torch.nonzero(global_degree > 0, as_tuple=False).view(-1).long()
        if dst.numel() > 0:
            compact_ids = torch.searchsorted(active_ids, dst)
            if active_ids.numel() == 0 or not bool(
                torch.equal(active_ids.index_select(0, compact_ids), dst)
            ):
                raise RuntimeError(
                    f"Invalid compact destination map for relation {edge_type_to_str(edge_type)}"
                )
        else:
            compact_ids = torch.empty((0,), dtype=torch.long, device=edge_index.device)

        active_degree = global_degree.index_select(0, active_ids)
        plan = {
            "active_destination_ids": active_ids,
            "edge_compact_ids": compact_ids,
            "global_degree": active_degree,
            "num_active_destinations": int(active_ids.numel()),
            "num_total_destinations": int(num_dst_nodes),
            "active_ratio": (
                float(active_ids.numel()) / float(num_dst_nodes)
                if num_dst_nodes > 0
                else 0.0
            ),
            "metadata_build_sec": float(time.perf_counter() - started),
            "cache_key": key,
        }
        self._forward_agg_compact_plan_cache[key] = plan
        return plan

    def _local_sage_neighbor_sum_compact(
        self,
        conv,
        x_src: torch.Tensor,
        edge_index: torch.Tensor,
        edge_compact_ids: torch.Tensor,
        num_active_destinations: int,
    ) -> torch.Tensor:
        self._validate_exact_sage_conv(conv, ("unknown", "unknown", "unknown"))
        source_features = x_src
        if bool(getattr(conv, "project", False)):
            if not hasattr(conv, "lin"):
                raise AttributeError("SAGEConv(project=True) has no lin projection")
            source_features = conv.lin(source_features).relu()
        local_sum = torch.zeros(
            (int(num_active_destinations), int(source_features.shape[-1])),
            dtype=source_features.dtype,
            device=source_features.device,
        )
        if edge_index.shape[1] > 0 and num_active_destinations > 0:
            src_ids = edge_index[0].long()
            messages = source_features.index_select(0, src_ids)
            local_sum = local_sum.index_add(0, edge_compact_ids.long(), messages)
        return local_sum

    def _apply_sage_post_aggregation_compact(
        self,
        conv,
        global_sum: torch.Tensor,
        global_degree: torch.Tensor,
        active_ids: torch.Tensor,
        x_dst: torch.Tensor,
    ) -> torch.Tensor:
        out_channels = int(getattr(conv, "out_channels", x_dst.shape[-1]))
        out = torch.zeros(
            (int(x_dst.shape[0]), out_channels),
            dtype=x_dst.dtype,
            device=x_dst.device,
        )
        bias = getattr(getattr(conv, "lin_l", None), "bias", None)
        if bias is not None:
            out = out + bias.view(1, -1)
        if active_ids.numel() > 0:
            global_mean = global_sum / global_degree.to(
                dtype=global_sum.dtype, device=global_sum.device
            ).view(-1, 1).clamp_min(1.0)
            active_out = conv.lin_l(global_mean)
            out = out.index_copy(0, active_ids.long(), active_out)
        if bool(getattr(conv, "root_weight", True)) and hasattr(conv, "lin_r"):
            out = out + conv.lin_r(x_dst)
        if bool(getattr(conv, "normalize", False)):
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def _distributed_sage_relation_forward_compact(
        self,
        conv,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: EdgeType,
        layer_idx: int,
    ) -> torch.Tensor:
        """Exact blocking active-destination compact all-reduce."""
        self._validate_exact_sage_conv(conv, edge_type)
        group = self._get_forward_aggregation_group(edge_type)
        world = int(getattr(self, "world_size", 1))
        plan = self._get_compact_destination_plan(
            edge_type=edge_type,
            edge_index=edge_index,
            num_dst_nodes=int(x_dst.shape[0]),
        )

        timing = self._is_forward_aggregation_timing_enabled()
        if timing:
            self._sync_device_for_timing()
        pipeline_t0 = time.perf_counter()
        compute_t0 = time.perf_counter()
        local_sum = self._local_sage_neighbor_sum_compact(
            conv=conv,
            x_src=x_src,
            edge_index=edge_index,
            edge_compact_ids=plan["edge_compact_ids"],
            num_active_destinations=int(plan["num_active_destinations"]),
        )
        if timing:
            self._sync_device_for_timing()
        local_compute_sec = float(time.perf_counter() - compute_t0)

        comm_t0 = time.perf_counter()
        if world > 1 and group is not None:
            reduced = local_sum.detach().clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
            global_sum = _DetachedForwardAllReduceWithBackward.apply(
                local_sum, reduced, group
            )
            self._register_forward_agg_backward_anchor(global_sum)
        else:
            global_sum = local_sum
        if timing:
            self._sync_device_for_timing()
        comm_sec = float(time.perf_counter() - comm_t0)

        out = self._apply_sage_post_aggregation_compact(
            conv=conv,
            global_sum=global_sum,
            global_degree=plan["global_degree"],
            active_ids=plan["active_destination_ids"],
            x_dst=x_dst,
        )
        if timing:
            self._sync_device_for_timing()
        pipeline_sec = float(time.perf_counter() - pipeline_t0)

        element_bytes = int(x_src.element_size())
        feature_dim = int(x_src.shape[-1])
        dense_payload = int(x_dst.shape[0]) * feature_dim * element_bytes
        compact_payload = (
            int(plan["num_active_destinations"]) * feature_dim * element_bytes
        )
        if timing:
            self._append_forward_agg_overlap_record({
                "epoch": int(getattr(self, "_stage_epoch_counter", -1)),
                "rank": int(getattr(self, "rank", 0)),
                "world_size": world,
                "mode": self._get_forward_aggregation_mode(),
                "layer_idx": int(layer_idx),
                "edge_type": edge_type_to_str(edge_type),
                "communication_operator": "compact",
                "aggregation_num_chunks": 1,
                "aggregation_async_op": False,
                "aggregation_double_buffer": False,
                "aggregation_chunk_policy": "active_destination",
                "aggregation_chunk_order": "natural",
                "aggregation_execution_order": "0",
                "aggregation_max_inflight": 1,
                "aggregation_fuse_degree": False,
                "aggregation_global_edge_counts": str(int(edge_index.shape[1])),
                "num_local_edges": int(edge_index.shape[1]),
                "num_dst_nodes": int(x_dst.shape[0]),
                "num_active_destinations": int(plan["num_active_destinations"]),
                "active_destination_ratio": float(plan["active_ratio"]),
                "dense_payload_bytes": dense_payload,
                "compact_payload_bytes": compact_payload,
                "payload_reduction_ratio": (
                    float(compact_payload) / float(dense_payload)
                    if dense_payload > 0
                    else 0.0
                ),
                "compact_metadata_build_sec": float(plan["metadata_build_sec"]),
                "local_message_compute_sec": local_compute_sec,
                "aggregation_comm_total_sec": comm_sec,
                "aggregation_comm_exposed_sec": comm_sec,
                "aggregation_comm_hidden_sec": 0.0,
                "aggregation_wait_sec": comm_sec,
                "forward_pipeline_total_sec": pipeline_sec,
                "pipeline_overhead_sec": max(
                    0.0, pipeline_sec - local_compute_sec - comm_sec
                ),
                "overlap_ratio": 0.0,
            })
        return out

    def _get_destination_chunk_plan(
        self,
        edge_type: EdgeType,
        edge_index: torch.Tensor,
        num_dst_nodes: int,
        requested_chunks: int,
    ) -> Dict[str, Any]:
        num_dst_nodes = int(num_dst_nodes)
        if num_dst_nodes <= 0:
            return {"num_chunks": 1, "order": None, "offsets": [0, 0], "ranges": [(0, 0)]}
        num_chunks = max(1, min(int(requested_chunks), num_dst_nodes))
        chunk_policy = self._get_forward_aggregation_chunk_policy()
        key = (
            int(getattr(self, "_forward_agg_graph_generation", 0)),
            edge_type_to_str(edge_type),
            num_dst_nodes,
            num_chunks,
            chunk_policy,
            str(edge_index.device),
        )
        cached = self._forward_agg_chunk_plan_cache.get(key)
        if cached is not None:
            return cached

        num_edges = int(edge_index.shape[1]) if edge_index.dim() == 2 else 0
        dst = edge_index[1].long() if num_edges > 0 else torch.empty(
            (0,), dtype=torch.long, device=edge_index.device
        )

        if chunk_policy == "global_edge_balanced" and num_chunks > 1:
            # Build the boundaries once from the global destination-degree
            # histogram.  Every rank receives identical boundaries, so the
            # collective sequence and tensor shapes remain identical.
            with torch.no_grad():
                global_degree = torch.bincount(dst, minlength=num_dst_nodes).to(
                    dtype=torch.int64
                )
                group = self._get_forward_aggregation_group(edge_type)
                world = int(getattr(self, "world_size", 1))
                if world > 1 and group is not None:
                    dist.all_reduce(global_degree, op=dist.ReduceOp.SUM, group=group)
                total_edges = int(global_degree.sum().item())
                if total_edges > 0:
                    prefix = torch.cumsum(global_degree, dim=0)
                    targets = torch.tensor(
                        [
                            (total_edges * chunk_id + num_chunks - 1) // num_chunks
                            for chunk_id in range(1, num_chunks)
                        ],
                        dtype=prefix.dtype,
                        device=prefix.device,
                    )
                    candidates = (
                        torch.searchsorted(prefix, targets, right=False) + 1
                    ).detach().cpu().tolist()
                    boundaries = [0]
                    for index, candidate in enumerate(candidates, 1):
                        remaining = num_chunks - index
                        lower = boundaries[-1] + 1
                        upper = num_dst_nodes - remaining
                        boundaries.append(max(lower, min(int(candidate), upper)))
                    boundaries.append(num_dst_nodes)
                else:
                    boundaries = [
                        (chunk_id * num_dst_nodes + num_chunks - 1) // num_chunks
                        for chunk_id in range(num_chunks + 1)
                    ]
                del global_degree
        else:
            boundaries = [
                (chunk_id * num_dst_nodes + num_chunks - 1) // num_chunks
                for chunk_id in range(num_chunks + 1)
            ]

        ranges = [
            (int(boundaries[chunk_id]), int(boundaries[chunk_id + 1]))
            for chunk_id in range(num_chunks)
        ]
        if num_edges > 0:
            if num_chunks > 1:
                internal = torch.tensor(
                    boundaries[1:-1], dtype=dst.dtype, device=dst.device
                )
                chunk_ids = torch.bucketize(dst.contiguous(), internal, right=True)
            else:
                chunk_ids = torch.zeros_like(dst)
            order = torch.argsort(chunk_ids, stable=True)
            local_counts_tensor = torch.bincount(
                chunk_ids, minlength=num_chunks
            ).to(dtype=torch.int64)
            counts = local_counts_tensor.detach().cpu().tolist()
        else:
            order = torch.empty((0,), dtype=torch.long, device=edge_index.device)
            local_counts_tensor = torch.zeros(
                (num_chunks,), dtype=torch.int64, device=edge_index.device
            )
            counts = [0] * num_chunks

        global_counts_tensor = local_counts_tensor.clone()
        group = self._get_forward_aggregation_group(edge_type)
        world = int(getattr(self, "world_size", 1))
        if world > 1 and group is not None:
            dist.all_reduce(global_counts_tensor, op=dist.ReduceOp.SUM, group=group)
        global_counts = global_counts_tensor.detach().cpu().tolist()

        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + int(count))

        plan = {
            "num_chunks": num_chunks,
            "order": order,
            "offsets": offsets,
            "ranges": ranges,
            "chunk_policy": chunk_policy,
            "global_edge_counts": [int(value) for value in global_counts],
            "cache_key": key,
        }
        self._forward_agg_chunk_plan_cache[key] = plan
        return plan

    def _get_chunk_execution_order(self, plan: Dict[str, Any]) -> List[int]:
        num_chunks = int(plan["num_chunks"])
        chunk_ids = list(range(num_chunks))
        policy = self._get_forward_aggregation_chunk_order()
        edge_counts = [int(value) for value in plan.get("global_edge_counts", [])]
        if len(edge_counts) != num_chunks:
            edge_counts = [0] * num_chunks
        destination_counts = [
            max(0, int(end) - int(start)) for start, end in plan["ranges"]
        ]

        if policy == "natural" or num_chunks <= 1:
            return chunk_ids
        if policy == "largest_first":
            return sorted(chunk_ids, key=lambda item: (-edge_counts[item], item))
        if policy == "smallest_first":
            return sorted(chunk_ids, key=lambda item: (edge_counts[item], item))
        if policy == "zigzag":
            ranked = sorted(chunk_ids, key=lambda item: (-edge_counts[item], item))
            result: List[int] = []
            left, right = 0, len(ranked) - 1
            while left <= right:
                result.append(ranked[left])
                left += 1
                if left <= right:
                    result.append(ranked[right])
                    right -= 1
            return result

        # Pipeline-aware static heuristic:
        #   * start with a cheap compute chunk so the first collective launches early;
        #   * place compute-heavy chunks in the middle to hide preceding communication;
        #   * finish with the smallest communication tensor to reduce the exposed tail.
        first = min(chunk_ids, key=lambda item: (edge_counts[item], item))
        remaining = [item for item in chunk_ids if item != first]
        if not remaining:
            return [first]
        last = min(
            remaining,
            key=lambda item: (destination_counts[item], edge_counts[item], item),
        )
        middle = [item for item in remaining if item != last]
        middle.sort(key=lambda item: (-edge_counts[item], item))
        return [first, *middle, last]

    def _make_cuda_event(self):
        try:
            device = getattr(self, "device", None)
            if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
                return torch.cuda.Event(enable_timing=True)
        except Exception:
            pass
        return None

    @staticmethod
    def _event_elapsed_sec(start_event, end_event) -> float:
        if start_event is None or end_event is None:
            return 0.0
        try:
            return float(start_event.elapsed_time(end_event)) / 1000.0
        except Exception:
            return 0.0

    def _local_sage_neighbor_sum_chunk(
        self,
        conv,
        x_src: torch.Tensor,
        edge_index: torch.Tensor,
        edge_positions: torch.Tensor,
        dst_start: int,
        dst_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_exact_sage_conv(conv, ("unknown", "unknown", "unknown"))
        chunk_dst_count = max(0, int(dst_end) - int(dst_start))

        source_features = x_src
        if bool(getattr(conv, "project", False)):
            if not hasattr(conv, "lin"):
                raise AttributeError("SAGEConv(project=True) has no lin projection")
            source_features = conv.lin(source_features).relu()

        feature_dim = int(source_features.shape[-1])
        local_sum = torch.zeros(
            (chunk_dst_count, feature_dim),
            dtype=source_features.dtype,
            device=source_features.device,
        )
        local_degree = torch.zeros(
            (chunk_dst_count, 1),
            dtype=source_features.dtype,
            device=source_features.device,
        )

        if edge_positions is not None and int(edge_positions.numel()) > 0 and chunk_dst_count > 0:
            src_ids = edge_index[0].index_select(0, edge_positions).long()
            dst_ids = edge_index[1].index_select(0, edge_positions).long() - int(dst_start)
            messages = source_features.index_select(0, src_ids)
            # Out-of-place index_add keeps a clean autograd path from messages.
            local_sum = local_sum.index_add(0, dst_ids, messages)
            ones = torch.ones((int(dst_ids.numel()), 1), dtype=source_features.dtype, device=source_features.device)
            local_degree = local_degree.index_add(0, dst_ids, ones)

        return local_sum, local_degree

    def _apply_sage_post_aggregation(
        self,
        conv,
        global_sum: torch.Tensor,
        global_degree: torch.Tensor,
        x_dst_chunk: torch.Tensor,
    ) -> torch.Tensor:
        global_mean = global_sum / global_degree.clamp_min(1.0)
        out = conv.lin_l(global_mean)
        if bool(getattr(conv, "root_weight", True)) and x_dst_chunk is not None and hasattr(conv, "lin_r"):
            out = out + conv.lin_r(x_dst_chunk)
        if bool(getattr(conv, "normalize", False)):
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def _distributed_sage_relation_forward(
        self,
        conv,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: EdgeType,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Exact default-SAGEConv forward for one edge-partitioned relation.

        For every destination chunk:
          1) compute rank-local neighbor SUM and degree;
          2) SUM-reduce both across ranks;
          3) divide by global degree;
          4) apply SAGEConv lin_l and one copy of lin_r(root).

        The async path keeps one NCCL operation in flight while computing the next
        destination chunk. Detached communication buffers are connected back to
        autograd by _DetachedChunkedForwardAllReduceWithBackward.
        """
        if self._get_forward_aggregation_operator() == "compact":
            return self._distributed_sage_relation_forward_compact(
                conv=conv,
                x_src=x_src,
                x_dst=x_dst,
                edge_index=edge_index,
                edge_type=edge_type,
                layer_idx=layer_idx,
            )
        self._validate_exact_sage_conv(conv, edge_type)
        # Each communicating relation owns a dedicated process group.  Autograd
        # is free to make different relations ready in a different order on
        # different ranks.  Reusing the default/first relation's group here
        # therefore lets two ranks submit different relation tensors at the
        # same collective sequence number, which can deadlock (most visible at
        # C=8).  Select the group by relation, as the metadata collectives do.
        group = self._get_forward_aggregation_group(edge_type)
        world = int(getattr(self, "world_size", 1))
        overlap = bool(self._is_forward_aggregation_overlap_enabled() and world > 1 and group is not None)
        requested_chunks = self._get_forward_aggregation_num_chunks()
        plan = self._get_destination_chunk_plan(edge_type, edge_index, int(x_dst.shape[0]), requested_chunks)
        num_chunks = int(plan["num_chunks"])
        chunk_policy = self._get_forward_aggregation_chunk_policy()
        chunk_order_policy = self._get_forward_aggregation_chunk_order()
        execution_order = self._get_chunk_execution_order(plan)
        max_inflight = (
            min(num_chunks, self._get_forward_aggregation_max_inflight())
            if overlap
            else 1
        )
        fuse_degree = self._is_forward_aggregation_fuse_degree_enabled()

        timing = self._is_forward_aggregation_timing_enabled()
        if timing:
            self._sync_device_for_timing()
        pipeline_wall_t0 = time.perf_counter()

        compute_event_pairs = []
        comm_lifetime_event_pairs = []
        wait_event_pairs = []
        cpu_local_compute_sec = 0.0
        cpu_wait_sec = 0.0
        completed_chunks: List[Optional[Dict[str, Any]]] = [None] * num_chunks
        pending_queue: List[Dict[str, Any]] = []

        degree_cache_base = (
            int(getattr(self, "_forward_agg_graph_generation", 0)),
            edge_type_to_str(edge_type),
            int(x_dst.shape[0]),
            num_chunks,
            chunk_policy,
            str(edge_index.device),
        )

        def finalize(item):
            nonlocal cpu_wait_sec
            wait_start_event = self._make_cuda_event()
            wait_end_event = self._make_cuda_event()
            if wait_start_event is not None:
                wait_start_event.record()
            wait_t0 = time.perf_counter()
            if item["work_sum"] is not None:
                item["work_sum"].wait()
            if item["work_degree"] is not None:
                item["work_degree"].wait()
            cpu_wait_sec += float(time.perf_counter() - wait_t0)
            if wait_end_event is not None:
                wait_end_event.record()
            wait_event_pairs.append((wait_start_event, wait_end_event))
            comm_lifetime_event_pairs.append((item["launch_event"], wait_end_event))

            global_degree = item["degree_buffer"]
            self._forward_agg_global_degree_cache[item["degree_cache_key"]] = global_degree.detach()
            return item

        for chunk_id in execution_order:
            dst_start, dst_end = plan["ranges"][chunk_id]
            p0, p1 = int(plan["offsets"][chunk_id]), int(plan["offsets"][chunk_id + 1])
            order = plan["order"]
            edge_positions = order[p0:p1] if order is not None else None

            compute_start_event = self._make_cuda_event()
            compute_end_event = self._make_cuda_event()
            if compute_start_event is not None:
                compute_start_event.record()
            local_t0 = time.perf_counter()
            local_sum, local_degree = self._local_sage_neighbor_sum_chunk(
                conv=conv,
                x_src=x_src,
                edge_index=edge_index,
                edge_positions=edge_positions,
                dst_start=dst_start,
                dst_end=dst_end,
            )
            cpu_local_compute_sec += float(time.perf_counter() - local_t0)
            if compute_end_event is not None:
                compute_end_event.record()
            compute_event_pairs.append((compute_start_event, compute_end_event))

            # Keep at most max_inflight collectives queued.  With depth=1 this is
            # the original double-buffer schedule: communication for chunk i runs
            # while chunk i+1 is computed.  A depth larger than one tolerates
            # short/imbalanced compute chunks without immediately stalling on the
            # previous collective.
            if overlap:
                while len(pending_queue) >= max_inflight:
                    completed_item = pending_queue.pop(0)
                    completed_chunks[int(completed_item["chunk_id"])] = finalize(
                        completed_item
                    )

            degree_cache_key = degree_cache_base + (int(chunk_id),)
            cached_degree = self._forward_agg_global_degree_cache.get(degree_cache_key)
            fused_buffer = None
            if cached_degree is None and fuse_degree:
                fused_buffer = torch.cat(
                    [local_sum.detach(), local_degree.detach()], dim=1
                ).contiguous()
                feature_dim = int(local_sum.shape[1])
                sum_buffer = fused_buffer[:, :feature_dim]
                degree_buffer = fused_buffer[:, feature_dim:]
            else:
                sum_buffer = local_sum.detach().clone()
                degree_buffer = (
                    cached_degree.clone()
                    if cached_degree is not None
                    else local_degree.detach().clone()
                )
            launch_event = self._make_cuda_event()
            if launch_event is not None:
                launch_event.record()

            work_sum = None
            work_degree = None
            if world > 1 and group is not None:
                if fused_buffer is not None:
                    work_sum = dist.all_reduce(
                        fused_buffer,
                        op=dist.ReduceOp.SUM,
                        group=group,
                        async_op=overlap,
                    )
                else:
                    work_sum = dist.all_reduce(
                        sum_buffer,
                        op=dist.ReduceOp.SUM,
                        group=group,
                        async_op=overlap,
                    )
                if cached_degree is None and fused_buffer is None:
                    work_degree = dist.all_reduce(
                        degree_buffer,
                        op=dist.ReduceOp.SUM,
                        group=group,
                        async_op=overlap,
                    )

            item = {
                "chunk_id": int(chunk_id),
                "dst_start": int(dst_start),
                "dst_end": int(dst_end),
                "local_sum": local_sum,
                "sum_buffer": sum_buffer,
                "degree_buffer": degree_buffer,
                "degree_cache_key": degree_cache_key,
                "work_sum": work_sum,
                "work_degree": work_degree,
                "launch_event": launch_event,
                "fused_buffer": fused_buffer,
            }

            if overlap:
                pending_queue.append(item)
            else:
                completed_chunks[int(chunk_id)] = finalize(item)

        while pending_queue:
            completed_item = pending_queue.pop(0)
            completed_chunks[int(completed_item["chunk_id"])] = finalize(completed_item)

        if any(value is None for value in completed_chunks):
            missing = [index for index, value in enumerate(completed_chunks) if value is None]
            raise RuntimeError(f"Forward aggregation did not produce chunks: {missing}")
        ordered_items = [value for value in completed_chunks if value is not None]
        local_sums = [value["local_sum"] for value in ordered_items]
        reduced_sums = [value["sum_buffer"] for value in ordered_items]
        if world > 1 and group is not None:
            global_sums = _DetachedChunkedForwardAllReduceWithBackward.apply(
                group, *local_sums, *reduced_sums
            )
            if isinstance(global_sums, torch.Tensor):
                global_sums = (global_sums,)
            for global_sum in global_sums:
                self._register_forward_agg_backward_anchor(global_sum)
        else:
            global_sums = tuple(local_sums)

        ordered_output_chunks: List[torch.Tensor] = []
        for global_sum, value in zip(global_sums, ordered_items):
            dst_chunk = x_dst[value["dst_start"]:value["dst_end"]]
            ordered_output_chunks.append(
                self._apply_sage_post_aggregation(
                    conv, global_sum, value["degree_buffer"], dst_chunk
                )
            )
        out = (
            torch.cat(ordered_output_chunks, dim=0)
            if ordered_output_chunks
            else torch.empty(
                (0, int(getattr(conv, "out_channels", x_src.shape[-1]))),
                dtype=x_src.dtype,
                device=x_src.device,
            )
        )

        if timing:
            self._sync_device_for_timing()
        pipeline_total_sec = float(time.perf_counter() - pipeline_wall_t0)

        local_message_compute_sec = sum(self._event_elapsed_sec(a, b) for a, b in compute_event_pairs)
        aggregation_wait_sec = sum(self._event_elapsed_sec(a, b) for a, b in wait_event_pairs)
        aggregation_comm_total_sec = sum(self._event_elapsed_sec(a, b) for a, b in comm_lifetime_event_pairs)
        if local_message_compute_sec <= 0.0:
            local_message_compute_sec = float(cpu_local_compute_sec)
        if aggregation_wait_sec <= 0.0:
            aggregation_wait_sec = float(cpu_wait_sec)
        if aggregation_comm_total_sec <= 0.0:
            aggregation_comm_total_sec = float(aggregation_wait_sec)
        aggregation_comm_exposed_sec = float(aggregation_wait_sec)
        aggregation_comm_hidden_sec = max(0.0, aggregation_comm_total_sec - aggregation_comm_exposed_sec)
        overlap_ratio = (
            aggregation_comm_hidden_sec / aggregation_comm_total_sec
            if aggregation_comm_total_sec > 0.0 else 0.0
        )
        pipeline_overhead_sec = max(
            0.0,
            pipeline_total_sec - local_message_compute_sec - aggregation_comm_exposed_sec,
        )

        if timing:
            element_bytes = int(x_src.element_size())
            feature_dim = int(x_src.shape[-1])
            dense_payload = int(x_dst.shape[0]) * feature_dim * element_bytes
            self._append_forward_agg_overlap_record({
                "epoch": int(getattr(self, "_stage_epoch_counter", -1)),
                "rank": int(getattr(self, "rank", 0)),
                "world_size": world,
                "mode": self._get_forward_aggregation_mode(),
                "layer_idx": int(layer_idx),
                "edge_type": edge_type_to_str(edge_type),
                "communication_operator": "dense",
                "aggregation_num_chunks": int(num_chunks),
                "aggregation_async_op": bool(overlap),
                "aggregation_double_buffer": bool(overlap and max_inflight == 1),
                "aggregation_chunk_policy": chunk_policy,
                "aggregation_chunk_order": chunk_order_policy,
                "aggregation_execution_order": ",".join(str(value) for value in execution_order),
                "aggregation_max_inflight": int(max_inflight),
                "aggregation_fuse_degree": bool(fuse_degree),
                "aggregation_global_edge_counts": ",".join(
                    str(value) for value in plan.get("global_edge_counts", [])
                ),
                "num_local_edges": int(edge_index.shape[1]),
                "num_dst_nodes": int(x_dst.shape[0]),
                "num_active_destinations": int(x_dst.shape[0]),
                "active_destination_ratio": 1.0,
                "dense_payload_bytes": dense_payload,
                "compact_payload_bytes": dense_payload,
                "payload_reduction_ratio": 1.0,
                "compact_metadata_build_sec": 0.0,
                "local_message_compute_sec": float(local_message_compute_sec),
                "aggregation_comm_total_sec": float(aggregation_comm_total_sec),
                "aggregation_comm_exposed_sec": float(aggregation_comm_exposed_sec),
                "aggregation_comm_hidden_sec": float(aggregation_comm_hidden_sec),
                "aggregation_wait_sec": float(aggregation_wait_sec),
                "forward_pipeline_total_sec": float(pipeline_total_sec),
                "pipeline_overhead_sec": float(pipeline_overhead_sec),
                "aggregation_overlap_ratio": float(overlap_ratio),
                "timestamp": time.time(),
            })

        return out

    def _install_forward_aggregation_overlap_if_needed(self):
        """Patch HeteroConv layers with exact distributed SAGE aggregation."""
        if not self._is_forward_aggregation_enabled():
            return
        if not (self.use_distributed and dist.is_available() and dist.is_initialized() and self.world_size > 1):
            logger.warning("forward aggregation requested without multi-rank distributed execution; using local path")
        self._ensure_forward_aggregation_group()

        model_core = self._get_model_core()
        if not hasattr(model_core, "conv_layers"):
            raise AttributeError("model has no conv_layers; cannot install forward aggregation")
        if getattr(model_core, "_commopt_forward_agg_overlap_installed", False):
            return

        manager_self = self
        for layer_idx, conv_layer in enumerate(model_core.conv_layers):
            if getattr(conv_layer, "_commopt_forward_agg_overlap_patched", False):
                continue
            original_forward = conv_layer.forward
            layer_aggr = getattr(conv_layer, "aggr", "mean")

            def make_forward(layer, layer_index: int, original_fwd, hetero_aggr):
                def distributed_forward(x_dict, edge_index_dict):
                    out_by_dst: Dict[str, List[torch.Tensor]] = {}
                    try:
                        # Preserve graph metadata order. Collective-bearing relation
                        # order must be identical on every rank; never sort it by
                        # rank-local edge count.
                        for edge_type, edge_index in edge_index_dict.items():
                            if not isinstance(edge_type, tuple) or len(edge_type) != 3:
                                continue
                            src_type, _, dst_type = edge_type
                            if src_type not in x_dict or dst_type not in x_dict:
                                continue
                            if edge_index is None or not hasattr(edge_index, "shape"):
                                continue
                            conv = manager_self._get_conv_for_edge_type(layer, edge_type)
                            if conv is None:
                                continue

                            if edge_type in LABELED_EDGE_TYPES:
                                out = manager_self._distributed_sage_relation_forward(
                                    conv=conv,
                                    x_src=x_dict[src_type],
                                    x_dst=x_dict[dst_type],
                                    edge_index=edge_index,
                                    edge_type=edge_type,
                                    layer_idx=layer_index,
                                )
                            else:
                                # Non-labeled relations are fully replicated by the
                                # partitioner, so communication would duplicate them.
                                out = conv((x_dict[src_type], x_dict[dst_type]), edge_index)
                            out_by_dst.setdefault(dst_type, []).append(out)

                        out_dict = {}
                        for dst_type, outs in out_by_dst.items():
                            agg = manager_self._aggregate_relation_outputs(outs, hetero_aggr)
                            if agg is not None:
                                out_dict[dst_type] = agg
                        return out_dict
                    except Exception:
                        logger.error(
                            f"exact forward aggregation failed at layer {layer_index}; refusing silent semantic fallback",
                            exc_info=True,
                        )
                        raise
                return distributed_forward

            conv_layer._commopt_original_forward = original_forward
            conv_layer.forward = make_forward(conv_layer, layer_idx, original_forward, layer_aggr)
            conv_layer._commopt_forward_agg_overlap_patched = True

        model_core._commopt_forward_agg_overlap_installed = True
        logger.info(
            f"[Rank {getattr(self, 'rank', 0)}] installed exact forward aggregation: "
            f"commopt_patch={COMMOPT_PATCH_VERSION}, "
            f"mode={self._get_forward_aggregation_mode()}, "
            f"operator={self._get_forward_aggregation_operator()}, "
            f"chunks={self._get_forward_aggregation_num_chunks()}, "
            f"async={self._is_forward_aggregation_overlap_enabled()}, "
            f"partitioned_relations={[edge_type_to_str(et) for et in LABELED_EDGE_TYPES]}"
        )

    def _install_relation_scheduled_convs_if_needed(self):
        """
        Replace PyG HeteroConv.forward with an explicit relation-wise scheduler.

        This is an exact message-passing schedule for mode='all': it still runs one
        SAGEConv per relation and aggregates outputs by destination node type using
        the layer's aggregation rule.  It only changes relation execution order and
        optionally records relation/chunk timing.

        When relation_filter_mode != 'all', the experiment intentionally removes
        some relations from message passing. Use it only as a speed/accuracy ablation.
        """
        if not self._is_relation_scheduled_conv_enabled():
            return
        model_core = self._get_model_core()
        if not hasattr(model_core, "conv_layers"):
            logger.warning("model has no conv_layers; cannot install relation scheduler")
            return
        if getattr(model_core, "_commopt_relation_scheduler_installed", False):
            return

        manager_self = self
        for layer_idx, conv_layer in enumerate(model_core.conv_layers):
            if getattr(conv_layer, "_commopt_relation_scheduler_patched", False):
                continue
            original_forward = conv_layer.forward
            aggr = getattr(conv_layer, "aggr", "mean")

            def make_scheduled_forward(layer, layer_index: int, original_fwd, layer_aggr: str):
                def scheduled_forward(x_dict, edge_index_dict):
                    try:
                        # Build eligible relation list with edge counts.
                        relation_items = []
                        for pos, (et, ei) in enumerate(edge_index_dict.items()):
                            if not isinstance(et, tuple) or len(et) != 3:
                                continue
                            if ei is None or not hasattr(ei, "shape") or int(ei.shape[1]) <= 0:
                                continue
                            if not manager_self._relation_edge_type_allowed(et):
                                continue
                            if et[0] not in x_dict or et[2] not in x_dict:
                                continue
                            relation_items.append((et, ei, int(ei.shape[1]), pos))

                        relation_items = sorted(
                            relation_items,
                            key=lambda z: manager_self._relation_order_key(z[0], z[2], z[3])
                        )

                        group_size = manager_self._get_relation_group_size()
                        if group_size <= 0:
                            group_size = len(relation_items) if relation_items else 1

                        out_by_dst: Dict[str, List[torch.Tensor]] = {}
                        relation_order_names: List[str] = []

                        for group_id, group_start in enumerate(range(0, len(relation_items), group_size)):
                            group = relation_items[group_start:group_start + group_size]
                            group_t0 = time.perf_counter()
                            group_edge_count = 0
                            group_relation_names = []

                            for et, ei, num_edges, _pos in group:
                                et_str = edge_type_to_str(et)
                                group_relation_names.append(et_str)
                                relation_order_names.append(et_str)
                                group_edge_count += int(num_edges)
                                conv = manager_self._get_conv_for_edge_type(layer, et)
                                if conv is None:
                                    continue

                                rt0 = time.perf_counter()
                                src_type, _, dst_type = et
                                out = conv((x_dict[src_type], x_dict[dst_type]), ei)
                                if manager_self._is_relation_timing_enabled():
                                    manager_self._sync_device_for_timing()
                                rsec = float(time.perf_counter() - rt0)

                                out_by_dst.setdefault(dst_type, []).append(out)

                                if manager_self._is_relation_timing_enabled():
                                    manager_self._append_relation_forward_record({
                                        "epoch": int(getattr(manager_self, "_stage_epoch_counter", -1)),
                                        "rank": int(getattr(manager_self, "rank", 0)),
                                        "world_size": int(getattr(manager_self, "world_size", 1)),
                                        "mode": str(getattr(manager_self.config.training_config, "steady_experiment_mode", "unknown")),
                                        "layer_idx": int(layer_index),
                                        "edge_type": et_str,
                                        "relation_schedule_order": manager_self._get_relation_schedule_order(),
                                        "relation_filter_mode": manager_self._get_relation_filter_mode(),
                                        "relation_group_size": int(group_size),
                                        "num_edges": int(num_edges),
                                        "relation_forward_sec": rsec,
                                        "microsec_per_edge": float(rsec * 1e6 / max(1, int(num_edges))),
                                        "timestamp": time.time(),
                                    })

                            if manager_self._is_relation_timing_enabled():
                                manager_self._sync_device_for_timing()
                            group_sec = float(time.perf_counter() - group_t0)
                            manager_self._append_relation_chunk_record({
                                "epoch": int(getattr(manager_self, "_stage_epoch_counter", -1)),
                                "rank": int(getattr(manager_self, "rank", 0)),
                                "world_size": int(getattr(manager_self, "world_size", 1)),
                                "mode": str(getattr(manager_self.config.training_config, "steady_experiment_mode", "unknown")),
                                "layer_idx": int(layer_index),
                                "group_id": int(group_id),
                                "relation_schedule_order": manager_self._get_relation_schedule_order(),
                                "relation_filter_mode": manager_self._get_relation_filter_mode(),
                                "relation_group_size": int(group_size),
                                "num_relations": int(len(group)),
                                "num_edges": int(group_edge_count),
                                "edge_types_json": json.dumps(group_relation_names, ensure_ascii=False),
                                "relation_chunk_sec": group_sec,
                                "microsec_per_edge": float(group_sec * 1e6 / max(1, int(group_edge_count))),
                                "timestamp": time.time(),
                            })

                        out_dict = {}
                        for dst_type, outs in out_by_dst.items():
                            agg = manager_self._aggregate_relation_outputs(outs, layer_aggr)
                            if agg is not None:
                                out_dict[dst_type] = agg

                        # HeteroConv returns only dst types that received messages;
                        # the model wrapper adds residual types afterward.
                        return out_dict

                    except Exception as e:
                        logger.warning(
                            f"relation scheduled conv failed at layer {layer_index}: {e}; fallback to original HeteroConv",
                            exc_info=True,
                        )
                        return original_fwd(x_dict, edge_index_dict)
                return scheduled_forward

            conv_layer._commopt_original_forward = original_forward
            conv_layer.forward = make_scheduled_forward(conv_layer, layer_idx, original_forward, aggr)
            conv_layer._commopt_relation_scheduler_patched = True

        model_core._commopt_relation_scheduler_installed = True
        logger.info(
            f"[Rank {getattr(self, 'rank', 0)}] installed relation scheduler: "
            f"order={self._get_relation_schedule_order()}, "
            f"filter={self._get_relation_filter_mode()}, "
            f"group_size={self._get_relation_group_size()}, "
            f"timing={self._is_relation_timing_enabled()}"
        )

    def _sync_device_for_timing(self):
        """Synchronize the current device for stage-level timing only."""
        try:
            device = getattr(self, "device", None)
            if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
                return
        except Exception:
            pass
        try:
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.synchronize()
        except Exception:
            pass

    def _get_model_core(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _append_forward_loss_detail_record(self, record: Dict[str, Any]):
        try:
            out_dir = self._get_assignment_output_dir()
            path = out_dir / f"forward_loss_detail_rank{getattr(self, 'rank', 0)}.csv"
            df = pd.DataFrame([record])
            if path.exists():
                df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.debug(f"failed to append forward-loss detail record: {e}")

    def _append_edge_prediction_chunk_record(self, record: Dict[str, Any]):
        try:
            out_dir = self._get_assignment_output_dir()
            path = out_dir / f"edge_prediction_chunk_timing_rank{getattr(self, 'rank', 0)}.csv"
            df = pd.DataFrame([record])
            if path.exists():
                df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.debug(f"failed to append edge-prediction chunk record: {e}")

    def _append_loss_chunk_record(self, record: Dict[str, Any]):
        try:
            out_dir = self._get_assignment_output_dir()
            path = out_dir / f"loss_chunk_timing_rank{getattr(self, 'rank', 0)}.csv"
            df = pd.DataFrame([record])
            if path.exists():
                df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.debug(f"failed to append loss chunk record: {e}")

    def _install_chunked_edge_predictor_if_needed(self):
        """
        Install a chunked edge predictor on the underlying SupplyChainGCN module.

        The graph convolution/message-passing part is still executed once on the full
        rank-local graph shard. Only edge prediction is chunked by edge type and edge
        index range. This keeps training semantics identical while exposing whether
        the edge-prediction part of forward_loss benefits from chunk-level scheduling.
        """
        if not self._is_chunked_edge_prediction_enabled():
            return

        chunk_size = self._get_edge_prediction_chunk_size()
        if chunk_size <= 0:
            return

        model_core = self._get_model_core()
        if getattr(model_core, "_commopt_chunked_edge_predictor_installed", False):
            return

        if not hasattr(model_core, "_predict_edge"):
            logger.warning("model has no _predict_edge; cannot install chunked edge predictor")
            return

        original_predict_edge = model_core._predict_edge
        manager_self = self

        def chunked_predict_edge(edge_type, edge_key, data, x_dict, device):
            try:
                edge_data = data[edge_type]
                if not hasattr(edge_data, "y") or not hasattr(edge_data, "x"):
                    return None
                if edge_key not in model_core.edge_predictor:
                    return None
                if not hasattr(edge_data, "edge_index") or edge_data.edge_index.shape[1] == 0:
                    return None

                edge_features_all = edge_data.x.to(device)
                if not edge_features_all.is_contiguous():
                    edge_features_all = edge_features_all.contiguous()

                edge_index_all = edge_data.edge_index
                num_edges = int(edge_index_all.shape[1])
                if edge_features_all.shape[0] != num_edges:
                    logger.error(f"Edge features count mismatch for {edge_type}")
                    return None

                if num_edges <= chunk_size:
                    # Still time this as a single chunk for comparability.
                    t_chunk = time.perf_counter()
                    out = original_predict_edge(edge_type, edge_key, data, x_dict, device)
                    manager_self._sync_device_for_timing()
                    sec = float(time.perf_counter() - t_chunk)
                    manager_self._append_edge_prediction_chunk_record({
                        "epoch": int(getattr(manager_self, "_stage_epoch_counter", -1)),
                        "rank": int(getattr(manager_self, "rank", 0)),
                        "world_size": int(getattr(manager_self, "world_size", 1)),
                        "mode": str(getattr(manager_self.config.training_config, "steady_experiment_mode", "unknown")),
                        "edge_type": edge_type_to_str(edge_type),
                        "chunk_id": 0,
                        "chunk_start": 0,
                        "chunk_end": int(num_edges),
                        "chunk_num_edges": int(num_edges),
                        "edge_prediction_chunk_size": int(chunk_size),
                        "chunk_prediction_sec": sec,
                        "timestamp": time.time(),
                    })
                    return out

                outs = []
                edge_type_str = edge_type_to_str(edge_type)
                for chunk_id, start in enumerate(range(0, num_edges, chunk_size)):
                    end = min(start + chunk_size, num_edges)
                    t_chunk = time.perf_counter()

                    edge_features = edge_features_all[start:end]
                    edge_index = edge_index_all[:, start:end]

                    predictor_to_use = edge_key
                    features_to_use = edge_features

                    if len(edge_type) == 3:
                        src_type, _, dst_type = edge_type
                        if src_type in x_dict and dst_type in x_dict:
                            src_max_idx = x_dict[src_type].shape[0] - 1
                            dst_max_idx = x_dict[dst_type].shape[0] - 1
                            valid_src = edge_index[0] <= src_max_idx
                            valid_dst = edge_index[1] <= dst_max_idx
                            if bool(valid_src.all()) and bool(valid_dst.all()):
                                src_features = x_dict[src_type][edge_index[0]]
                                dst_features = x_dict[dst_type][edge_index[1]]
                                combined = torch.cat([edge_features, src_features, dst_features], dim=1)
                                if combined.shape[1] == model_core.edge_predictor[edge_key][0].in_features:
                                    features_to_use = combined
                                elif (
                                    edge_key + "_simple" in model_core.edge_predictor
                                    and edge_features.shape[1] == model_core.edge_predictor[edge_key + "_simple"][0].in_features
                                ):
                                    predictor_to_use = edge_key + "_simple"

                    if predictor_to_use not in model_core.edge_predictor:
                        predictor_to_use = edge_key

                    chunk_out = model_core.edge_predictor[predictor_to_use](features_to_use)
                    chunk_out = chunk_out.squeeze(-1)
                    chunk_out = torch.sigmoid(chunk_out)
                    outs.append(chunk_out)

                    manager_self._sync_device_for_timing()
                    sec = float(time.perf_counter() - t_chunk)
                    manager_self._append_edge_prediction_chunk_record({
                        "epoch": int(getattr(manager_self, "_stage_epoch_counter", -1)),
                        "rank": int(getattr(manager_self, "rank", 0)),
                        "world_size": int(getattr(manager_self, "world_size", 1)),
                        "mode": str(getattr(manager_self.config.training_config, "steady_experiment_mode", "unknown")),
                        "edge_type": edge_type_str,
                        "chunk_id": int(chunk_id),
                        "chunk_start": int(start),
                        "chunk_end": int(end),
                        "chunk_num_edges": int(end - start),
                        "edge_prediction_chunk_size": int(chunk_size),
                        "chunk_prediction_sec": sec,
                        "timestamp": time.time(),
                    })

                return torch.cat(outs, dim=0) if outs else None

            except Exception as e:
                logger.warning(f"Error in chunked edge prediction for {edge_type}: {e}; fallback to original")
                return original_predict_edge(edge_type, edge_key, data, x_dict, device)

        model_core._commopt_original_predict_edge = original_predict_edge
        model_core._predict_edge = chunked_predict_edge
        model_core._commopt_chunked_edge_predictor_installed = True
        logger.info(
            f"[Rank {getattr(self, 'rank', 0)}] installed chunked edge predictor, "
            f"chunk_size={chunk_size}"
        )

    def _compute_loss_with_optional_chunking(self, predictions, graph):
        """
        Compute the loss, optionally using chunked BCE over edge predictions.

        For the current experiment LOSS_MODE=edges_only. The chunked version is
        mathematically aligned with the original loss: it computes mean BCE per
        labeled edge type and then averages edge-type losses, plus the same L2 term.
        """
        if not self._is_chunked_loss_enabled():
            model_core = self._get_model_core()
            return model_core.compute_loss(predictions, graph)

        import torch.nn.functional as F
        model_core = self._get_model_core()
        device = getattr(model_core, "_current_device", lambda: self.device)()
        chunk_size = self._get_loss_chunk_size()
        if chunk_size <= 0:
            chunk_size = 250000

        loss_mode = getattr(model_core.config.training_config, "loss_mode", "both")
        if loss_mode not in ["edges_only", "both"]:
            return model_core.compute_loss(predictions, graph)

        t0 = time.perf_counter()
        edge_losses = []
        edge_types_with_labels = [
            ("inventory", "to", "virtual"),
            ("inventory", "to_transport_labeled", "inventory"),
        ]

        for edge_type in edge_types_with_labels:
            if edge_type not in predictions or not hasattr(graph[edge_type], "y"):
                continue
            edge_pred = predictions[edge_type]
            edge_true = graph[edge_type].y.to(device)
            if edge_pred.shape[0] != edge_true.shape[0]:
                min_len = min(edge_pred.shape[0], edge_true.shape[0])
                edge_pred = edge_pred[:min_len]
                edge_true = edge_true[:min_len]

            num_edges = int(edge_pred.shape[0])
            bce_sum = None
            valid_count = 0
            et_t0 = time.perf_counter()
            for chunk_id, start in enumerate(range(0, num_edges, chunk_size)):
                end = min(start + chunk_size, num_edges)
                c_t0 = time.perf_counter()
                p = edge_pred[start:end]
                y = edge_true[start:end]
                mask = ~torch.isnan(y)
                c_valid = int(mask.sum().item()) if mask.numel() > 0 else 0
                if c_valid > 0:
                    p = p[mask].float()
                    y = y[mask].float()
                    p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
                    c_loss_sum = F.binary_cross_entropy(p, y, reduction="sum")
                    bce_sum = c_loss_sum if bce_sum is None else bce_sum + c_loss_sum
                    valid_count += c_valid
                self._sync_device_for_timing()
                c_sec = float(time.perf_counter() - c_t0)
                self._append_loss_chunk_record({
                    "epoch": int(getattr(self, "_stage_epoch_counter", -1)),
                    "rank": int(getattr(self, "rank", 0)),
                    "world_size": int(getattr(self, "world_size", 1)),
                    "mode": str(getattr(self.config.training_config, "steady_experiment_mode", "unknown")),
                    "edge_type": edge_type_to_str(edge_type),
                    "chunk_id": int(chunk_id),
                    "chunk_start": int(start),
                    "chunk_end": int(end),
                    "chunk_num_edges": int(end - start),
                    "valid_count": int(c_valid),
                    "loss_chunk_size": int(chunk_size),
                    "loss_chunk_sec": c_sec,
                    "timestamp": time.time(),
                })

            if bce_sum is not None and valid_count > 0:
                edge_losses.append(bce_sum / float(valid_count))
            self._sync_device_for_timing()
            self._append_loss_chunk_record({
                "epoch": int(getattr(self, "_stage_epoch_counter", -1)),
                "rank": int(getattr(self, "rank", 0)),
                "world_size": int(getattr(self, "world_size", 1)),
                "mode": str(getattr(self.config.training_config, "steady_experiment_mode", "unknown")),
                "edge_type": edge_type_to_str(edge_type),
                "chunk_id": -1,
                "chunk_start": 0,
                "chunk_end": int(num_edges),
                "chunk_num_edges": int(num_edges),
                "valid_count": int(valid_count),
                "loss_chunk_size": int(chunk_size),
                "loss_chunk_sec": float(time.perf_counter() - et_t0),
                "timestamp": time.time(),
            })

        if not edge_losses:
            return model_core.compute_loss(predictions, graph)

        main_loss = sum(edge_losses) / len(edge_losses)
        l2_regularization = 0.0
        param_count = 0
        for param in model_core.parameters():
            if param.requires_grad:
                l2_regularization = l2_regularization + torch.norm(param, 2) ** 2
                param_count += param.numel()
        if param_count > 0:
            normalized_l2_reg = l2_regularization / param_count
            reg_term = getattr(model_core, "l2_lambda", 0.001) * normalized_l2_reg
        else:
            reg_term = 0.0
        total = main_loss + reg_term
        self._sync_device_for_timing()
        self._append_loss_chunk_record({
            "epoch": int(getattr(self, "_stage_epoch_counter", -1)),
            "rank": int(getattr(self, "rank", 0)),
            "world_size": int(getattr(self, "world_size", 1)),
            "mode": str(getattr(self.config.training_config, "steady_experiment_mode", "unknown")),
            "edge_type": "ALL",
            "chunk_id": -2,
            "chunk_start": 0,
            "chunk_end": 0,
            "chunk_num_edges": 0,
            "valid_count": 0,
            "loss_chunk_size": int(chunk_size),
            "loss_chunk_sec": float(time.perf_counter() - t0),
            "timestamp": time.time(),
        })
        return total

    def train_epoch(self, graphs: List[HeteroData]) -> float:
        """
        Stage-timed training epoch with optional persistent cache and forward-loss
        chunk diagnostics.

        New decomposition inside forward_loss:
          - model_forward_sec: model(graph), including GNN + edge prediction
          - loss_compute_sec: compute_loss(predictions, graph)
          - edge_prediction_chunk_timing_rank*.csv when chunked edge prediction is enabled
          - loss_chunk_timing_rank*.csv when chunked loss is enabled
        """
        import math

        self.model.train()
        model_device = next(self.model.parameters()).device

        # Install after DDP wrapping if needed. Exact forward aggregation and
        # the older relation-only scheduler are mutually exclusive.
        self._install_forward_aggregation_overlap_if_needed()
        if not self._is_forward_aggregation_enabled():
            self._install_relation_scheduled_convs_if_needed()
        self._install_chunked_edge_predictor_if_needed()

        epoch_idx = int(getattr(self, "_stage_epoch_counter", 0)) + 1
        self._stage_epoch_counter = epoch_idx
        timer = getattr(self, "stage_timer", None)
        if timer is not None:
            timer.start_epoch(epoch_idx)

        if self.use_distributed and self.world_size > 1:
            if timer is not None:
                with timer.stage("partition"):
                    local_graphs = self._distribute_graphs_balanced(graphs)
            else:
                local_graphs = self._distribute_graphs_balanced(graphs)
        else:
            local_graphs = graphs

        total_loss = 0.0
        successful_batches = 0
        has_data = len(local_graphs) > 0

        if self.use_distributed and dist.is_initialized():
            try:
                if timer is not None:
                    ctx = timer.stage("sync_data_state")
                else:
                    ctx = None
                if ctx is not None:
                    ctx.__enter__()
                data_count = torch.tensor([1.0 if has_data else 0.0], dtype=torch.float32, device=self.device)
                dist.all_reduce(data_count, op=dist.ReduceOp.SUM)
                ranks_with_data = int(data_count.item())
                if ctx is not None:
                    ctx.__exit__(None, None, None)
                if ranks_with_data == 0 and has_data:
                    ranks_with_data = 1
                if ranks_with_data == 0 and not has_data:
                    self._execute_dummy_step()
                    if timer is not None:
                        timer.end_epoch(train_loss=float("inf"), extra={"successful_batches": 0, "dummy_step": True})
                    return float("inf")
            except Exception as e:
                logger.warning(f"Data state collection failed: {e}")

        if self.use_distributed and not has_data:
            try:
                if timer is not None:
                    with timer.stage("dummy_step"):
                        self._execute_dummy_step()
                else:
                    self._execute_dummy_step()
            except Exception:
                pass

        for i, graph in enumerate(local_graphs):
            try:
                if timer is not None:
                    with timer.stage("data"):
                        graph = self._move_data_to_device(graph)
                else:
                    graph = self._move_data_to_device(graph)

                self.optimizer.zero_grad(set_to_none=True)

                predictions = None
                loss = None
                model_forward_sec = None
                loss_compute_sec = None
                pred_numel = 0
                pred_edges = {}

                if timer is not None:
                    forward_ctx = timer.stage("forward_loss")
                    forward_ctx.__enter__()
                else:
                    forward_ctx = None
                forward_loss_start = time.perf_counter()
                try:
                    # model forward
                    mf_t0 = time.perf_counter()
                    self._reset_forward_agg_backward_anchors()
                    if self.use_amp and self._amp_device_type is not None:
                        with torch.amp.autocast(device_type=self._amp_device_type, dtype=self.amp_dtype):
                            predictions = self.model(graph)
                    else:
                        predictions = self.model(graph)
                    self._sync_device_for_timing()
                    model_forward_sec = float(time.perf_counter() - mf_t0)

                    has_nan = False
                    if predictions:
                        for k, v in predictions.items():
                            if isinstance(v, torch.Tensor):
                                pred_numel += int(v.numel())
                                pred_edges[str(k)] = int(v.shape[0]) if v.dim() > 0 else int(v.numel())
                                if torch.isnan(v).any():
                                    has_nan = True
                                    break

                    if (has_nan or not predictions) and self.use_amp:
                        nan_msg = ", NaN" if has_nan else ", 空"
                        logger.warning(f"AMP 导致预测结果异常{nan_msg}, 自动禁用 AMP 并用 FP32 重试")
                        self.use_amp = False
                        self.amp_scaler = None
                        self._amp_device_type = None
                        mf_t0 = time.perf_counter()
                        self._reset_forward_agg_backward_anchors()
                        predictions = self.model(graph)
                        self._sync_device_for_timing()
                        model_forward_sec = float(time.perf_counter() - mf_t0)

                    if not predictions:
                        logger.warning(f"Empty predictions for graph {i}, skipping")
                        continue

                    # loss compute
                    lc_t0 = time.perf_counter()
                    if self.use_amp and self._amp_device_type is not None:
                        predictions_for_loss = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in predictions.items()}
                    else:
                        predictions_for_loss = predictions
                    loss = self._compute_loss_with_optional_chunking(predictions_for_loss, graph)
                    loss = self._attach_forward_agg_backward_anchors(loss)
                    self._sync_device_for_timing()
                    loss_compute_sec = float(time.perf_counter() - lc_t0)

                finally:
                    self._sync_device_for_timing()
                    forward_loss_total = float(time.perf_counter() - forward_loss_start)
                    if self._is_forward_detail_enabled():
                        self._append_forward_loss_detail_record({
                            "epoch": int(epoch_idx),
                            "rank": int(getattr(self, "rank", 0)),
                            "world_size": int(getattr(self, "world_size", 1)),
                            "mode": str(getattr(self.config.training_config, "steady_experiment_mode", "unknown")),
                            "enable_persistent_device_cache": bool(self._is_persistent_device_cache_enabled()),
                            "enable_chunked_edge_prediction": bool(self._is_chunked_edge_prediction_enabled()),
                            "edge_prediction_chunk_size": int(self._get_edge_prediction_chunk_size()),
                            "enable_chunked_loss": bool(self._is_chunked_loss_enabled()),
                            "loss_chunk_size": int(self._get_loss_chunk_size()),
                            "batch_idx": int(i),
                            "model_forward_sec": float(model_forward_sec) if model_forward_sec is not None else None,
                            "loss_compute_sec": float(loss_compute_sec) if loss_compute_sec is not None else None,
                            "forward_loss_measured_sec": float(forward_loss_total),
                            "prediction_numel": int(pred_numel),
                            "prediction_edges_json": json.dumps(pred_edges, ensure_ascii=False),
                            "timestamp": time.time(),
                        })
                    if forward_ctx is not None:
                        forward_ctx.__exit__(None, None, None)

                if loss is None:
                    logger.warning(f"No loss for graph {i}, skipping")
                    continue

                loss_value = loss.item()
                if math.isnan(loss_value) or math.isinf(loss_value):
                    logger.warning(f"Invalid loss for graph {i}: {loss_value}, skipping")
                    continue

                if timer is not None:
                    with timer.stage("backward"):
                        if self.use_amp and self.amp_scaler is not None:
                            self.amp_scaler.scale(loss).backward()
                            self.amp_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                else:
                    if self.use_amp and self.amp_scaler is not None:
                        self.amp_scaler.scale(loss).backward()
                        self.amp_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if timer is not None:
                    with timer.stage("optimizer"):
                        if self.use_amp and self.amp_scaler is not None:
                            self.amp_scaler.step(self.optimizer)
                            self.amp_scaler.update()
                        else:
                            self.optimizer.step()
                else:
                    if self.use_amp and self.amp_scaler is not None:
                        self.amp_scaler.step(self.optimizer)
                        self.amp_scaler.update()
                    else:
                        self.optimizer.step()

                total_loss += float(loss.item())
                successful_batches += 1

                amp_scale_info = ""
                if self.use_amp and self.amp_scaler is not None:
                    amp_scale_info = f", AMP scale: {self.amp_scaler.get_scale():.0f}"
                elif self.use_amp:
                    amp_scale_info = f", AMP: {self.amp_dtype}"
                if model_device.type == "npu" and i % 8 == 0:
                    try:
                        logger.info(
                            f"[Rank {self.rank}] Batch {i}: Loss {loss.item():.6f}, 显存: {torch.npu.memory_allocated(model_device) / 1024 ** 3:.2f}GB{amp_scale_info}"
                        )
                    except Exception:
                        pass
                elif model_device.type == "cuda" and i % 8 == 0:
                    try:
                        logger.info(
                            f"[Rank {self.rank}] Batch {i}: Loss {loss.item():.6f}, 显存: {torch.cuda.memory_allocated(model_device) / 1024 ** 3:.2f}GB{amp_scale_info}"
                        )
                    except Exception:
                        pass

            except Exception as e:
                if "out of memory" in str(e).lower() or "hip out of memory" in str(e).lower():
                    self._handle_oom_error(str(e))
                else:
                    logger.warning(f"Error processing graph {i}: {e}", exc_info=True)
                if self.use_distributed and dist.is_initialized():
                    # A rank-local exception may leave peers inside a different
                    # collective.  Starting a recovery all-reduce here creates a
                    # second ordering mismatch; fail the whole torchrun job so
                    # the launcher can record/retry it safely.
                    logger.error(
                        "Distributed batch failed; aborting this worker instead of continuing with a mismatched collective schedule."
                    )
                    raise
                continue

        if successful_batches == 0:
            logger.error("No successful batches in this epoch!")
            if timer is not None:
                timer.end_epoch(train_loss=float("inf"), extra={"successful_batches": 0})
            return float("inf")

        avg_loss = total_loss / max(1, successful_batches)

        if self.use_distributed and dist.is_initialized():
            try:
                if timer is not None:
                    sync_ctx = timer.stage("loss_sync")
                    sync_ctx.__enter__()
                else:
                    sync_ctx = None
                dist.barrier()
                if torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss)):
                    avg_loss = 0.0
                loss_tensor = torch.tensor(avg_loss, device=self.device, dtype=torch.float32)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = (loss_tensor / self.world_size).item()
                if sync_ctx is not None:
                    sync_ctx.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"DDP synchronization failed: {e}")

        if timer is not None:
            timer.end_epoch(
                train_loss=float(avg_loss),
                extra={
                    "successful_batches": int(successful_batches),
                    "partition_strategy": getattr(self.config.training_config, "partition_strategy", "unknown"),
                    "target_super_block_size": int(self._get_target_super_block_size()),
                },
            )

        self._append_runtime_record(
            record={
                "event": "train_epoch",
                "rank": int(getattr(self, "rank", 0)),
                "world_size": int(getattr(self, "world_size", 1)),
                "partition_strategy": getattr(self.config.training_config, "partition_strategy", "unknown"),
                "elapsed_sec": float(timer.records[-1]["epoch_total_sec"]) if timer is not None and timer.records else None,
                "loss": float(avg_loss) if np.isfinite(avg_loss) else avg_loss,
                "timestamp": time.time(),
            }
        )
        return avg_loss

    # ============================================================
    # SuperBlock construction
    # ============================================================

    def _get_target_super_block_size(self) -> int:
        return int(getattr(self.config.training_config, "target_super_block_size", 50000))

    def _get_profile_path(self) -> Optional[str]:
        path = getattr(self.config.training_config, "superblock_profile_path", None)
        if path is None:
            return None
        path = str(path).strip()
        return path if path else None

    def _get_assignment_output_dir(self) -> Path:
        out = getattr(self.config.training_config, "superblock_assignment_output_dir", None)
        if out:
            p = Path(str(out))
        else:
            p = Path(self.save_dir) / "communication_optimized_partition"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _build_target_groups(self, graph: HeteroData, edge_types: List[EdgeType]) -> List[TargetGroup]:
        groups: List[TargetGroup] = []
        group_id = 0

        for edge_type in edge_types:
            if edge_type not in graph.edge_types:
                logger.warning(f"[Rank {self.rank}] edge type not found: {edge_type}")
                continue

            edge_store = graph[edge_type]
            if not hasattr(edge_store, "edge_index") or edge_store.edge_index is None:
                continue

            edge_index = edge_store.edge_index.detach().cpu()
            if edge_index.numel() == 0 or edge_index.shape[1] == 0:
                continue

            _, _, dst_type = edge_type
            dst_nodes = edge_index[1].long()

            # 按 dst node 排序后分组，保证同一个目标节点 I(v) 不被拆开。
            order = torch.argsort(dst_nodes)
            sorted_dst = dst_nodes[order]
            unique_dst, counts = torch.unique_consecutive(sorted_dst, return_counts=True)

            start = 0
            for dst, cnt_tensor in zip(unique_dst, counts):
                cnt = int(cnt_tensor.item())
                edge_ids = order[start:start + cnt].clone()
                start += cnt

                groups.append(
                    TargetGroup(
                        group_id=group_id,
                        edge_type=edge_type,
                        dst_node_type=dst_type,
                        dst_node_id=int(dst.item()),
                        edge_ids=edge_ids,
                        num_edges=int(edge_ids.numel()),
                    )
                )
                group_id += 1

        return groups

    def _pack_target_groups_to_superblocks(
        self,
        target_groups: List[TargetGroup],
        target_super_block_size: int,
        profile_cost_map: Optional[Dict[Tuple[str, int], float]] = None,
        strategy: str = "superblock_edge_count",
    ) -> List[SuperBlock]:
        blocks: List[SuperBlock] = []
        block_id = 0

        groups_by_edge_type: Dict[EdgeType, List[TargetGroup]] = {}
        for g in target_groups:
            groups_by_edge_type.setdefault(g.edge_type, []).append(g)

        for edge_type, groups in groups_by_edge_type.items():
            groups = sorted(groups, key=lambda x: (x.dst_node_id, x.group_id))

            current_groups: List[TargetGroup] = []
            current_edges = 0

            def flush_current():
                nonlocal block_id, current_groups, current_edges

                if not current_groups:
                    return

                edge_ids = torch.cat([g.edge_ids for g in current_groups], dim=0)
                target_node_ids = [g.dst_node_id for g in current_groups]
                first = current_groups[0]
                et_str = edge_type_to_str(edge_type)

                # calibrated cost 优先来自 profiling 文件。
                # 注意：这里假设 profiling 文件和当前 block construction 使用相同 target_super_block_size 与同一图。
                profile_key = (et_str, block_id)
                if strategy == "superblock_calibrated_cost" and profile_cost_map and profile_key in profile_cost_map:
                    cost = float(profile_cost_map[profile_key])
                else:
                    cost = float(edge_ids.numel())

                blocks.append(
                    SuperBlock(
                        block_id=block_id,
                        edge_type=edge_type,
                        dst_node_type=first.dst_node_type,
                        target_node_ids=target_node_ids,
                        edge_ids=edge_ids,
                        num_edges=int(edge_ids.numel()),
                        num_target_groups=len(current_groups),
                        calibrated_cost=cost,
                    )
                )

                block_id += 1
                current_groups = []
                current_edges = 0

            for g in groups:
                if current_groups and current_edges + g.num_edges > target_super_block_size:
                    flush_current()

                current_groups.append(g)
                current_edges += g.num_edges

                if current_edges >= target_super_block_size:
                    flush_current()

            flush_current()

        return blocks

    def _load_profile_cost_map(self) -> Optional[Dict[Tuple[str, int], float]]:
        """
        读取 profiling 文件中的 cost。

        支持字段优先级：
        1. pred_time_ms_calibrated
        2. time_ms_median
        3. time_ms_mean

        key = (edge_type_str, block_id)
        """
        profile_path = self._get_profile_path()
        if not profile_path:
            return None

        p = Path(profile_path)
        if not p.exists():
            logger.warning(f"[Rank {self.rank}] profile file not found: {p}, fallback to edge count cost")
            return None

        try:
            df = pd.read_csv(p)

            if "block_id" not in df.columns or "edge_type" not in df.columns:
                logger.warning(f"[Rank {self.rank}] profile file missing block_id/edge_type: {p}")
                return None

            if "pred_time_ms_calibrated" in df.columns:
                cost_col = "pred_time_ms_calibrated"
            elif "time_ms_median" in df.columns:
                cost_col = "time_ms_median"
            elif "time_ms_mean" in df.columns:
                cost_col = "time_ms_mean"
            else:
                logger.warning(f"[Rank {self.rank}] profile file has no usable cost column: {p}")
                return None

            cost_map = {}
            for _, row in df.iterrows():
                if pd.isna(row[cost_col]):
                    continue
                key = (str(row["edge_type"]), int(row["block_id"]))
                cost_map[key] = float(row[cost_col])

            logger.info(f"[Rank {self.rank}] loaded {len(cost_map)} calibrated block costs from {p}, cost_col={cost_col}")
            return cost_map

        except Exception as e:
            logger.warning(f"[Rank {self.rank}] failed to load profile cost file: {p}, error={e}")
            return None

    # ============================================================
    # Assignment
    # ============================================================

    def _assign_superblocks_to_ranks(
            self,
            blocks,
            world_size,
            strategy,
    ):
        loads = {p: 0.0 for p in range(world_size)}
        edge_loads = {p: 0.0 for p in range(world_size)}
        block_counts = {p: 0 for p in range(world_size)}
        type_loads = {p: {} for p in range(world_size)}

        assignment = {}

        edge_balance_tol = float(
            getattr(self.config.training_config, "edge_balance_tolerance", 0.05)
        )

        lambda_edge = float(
            getattr(self.config.training_config, "assignment_lambda_edge_guard", 0.1)
        )

        lambda_type = float(
            getattr(self.config.training_config, "assignment_lambda_type", 0.2)
        )

        lambda_block = float(
            getattr(self.config.training_config, "assignment_lambda_block", 0.01)
        )

        def block_cost(b):
            return self._get_block_cost_for_strategy(b, strategy)

        def get_type_load(rank, edge_type_str):
            return float(type_loads[rank].get(edge_type_str, 0.0))

        total_edges = sum(float(b.num_edges) for b in blocks)
        avg_edges = total_edges / float(world_size)

        for b in sorted(blocks, key=block_cost, reverse=True):
            b_cost = block_cost(b)
            b_edges = float(b.num_edges)
            b_type = edge_type_to_str(b.edge_type)

            best_rank = None
            best_score = None

            for p in range(world_size):
                projected_loads = loads.copy()
                projected_edge_loads = edge_loads.copy()
                projected_block_counts = block_counts.copy()

                projected_loads[p] += b_cost
                projected_edge_loads[p] += b_edges
                projected_block_counts[p] += 1

                # 1. 主目标：cost makespan
                cost_makespan = max(projected_loads.values())

                # 2. 边数只作为 soft guard，不强求完全相等
                edge_excess = 0.0
                for r in range(world_size):
                    ratio = projected_edge_loads[r] / max(avg_edges, 1.0)
                    edge_excess += max(0.0, ratio - (1.0 + edge_balance_tol))

                # 3. 类型负载均衡作为次级项
                projected_type_loads = []
                for r in range(world_size):
                    val = get_type_load(r, b_type)
                    if r == p:
                        val += b_edges
                    projected_type_loads.append(val)

                avg_type = sum(projected_type_loads) / float(world_size)
                if avg_type > 0:
                    type_imbalance = max(projected_type_loads) / avg_type
                else:
                    type_imbalance = 1.0

                # 4. block 数量轻微约束，避免一个 rank block 太多
                avg_blocks = (sum(projected_block_counts.values()) / float(world_size))
                if avg_blocks > 0:
                    block_imbalance = max(projected_block_counts.values()) / avg_blocks
                else:
                    block_imbalance = 1.0

                score = (
                        cost_makespan
                        + lambda_edge * edge_excess * avg_edges
                        + lambda_type * (type_imbalance - 1.0) * avg_edges
                        + lambda_block * (block_imbalance - 1.0) * avg_edges
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best_rank = p

            assignment[b.block_id] = int(best_rank)
            loads[best_rank] += b_cost
            edge_loads[best_rank] += b_edges
            block_counts[best_rank] += 1
            type_loads[best_rank][b_type] = get_type_load(best_rank, b_type) + b_edges

        logger.info(
            f"[Rank {getattr(self, 'rank', 0)}] cost-aware assignment done: "
            f"cost_loads={loads}, edge_loads={edge_loads}, block_counts={block_counts}"
        )

        return assignment

    def estimate_block_cost(self, b):
        """
        Estimate SuperBlock execution cost.

        This is the lightweight time-oriented cost proxy used by
        superblock_cost_balance. It intentionally does not force exact edge-count
        equality. Edge count is only one term in the predicted cost; fixed block
        overhead and target-group/indexing overhead are also included.

        The coefficients are read from config.training_config so that the runner
        can sweep or replace them with profiling-calibrated values later.
        """
        edge_type_str = edge_type_to_str(b.edge_type)
        tc = self.config.training_config

        fixed_cost = float(getattr(tc, "cost_fixed_block", 500.0))
        target_group_cost = float(getattr(tc, "cost_per_target_group", 0.05))
        edge_cost = float(getattr(tc, "cost_per_edge", 1.0))

        type_factor_map = getattr(tc, "cost_type_factor_map", None)
        if type_factor_map is None:
            type_factor_map = {
                "inventory__to__virtual": 1.0,
                "inventory__to_transport_labeled__inventory": 1.0,
            }
        elif isinstance(type_factor_map, str):
            try:
                type_factor_map = json.loads(type_factor_map)
            except Exception:
                type_factor_map = {}

        type_factor = float(type_factor_map.get(edge_type_str, 1.0))

        cost = 0.0
        cost += edge_cost * type_factor * float(b.num_edges)
        cost += fixed_cost
        cost += target_group_cost * float(b.num_target_groups)
        return float(cost)

    def _partition_graph_by_superblocks(
        self,
        graph: HeteroData,
        world_size: int,
        rank: int,
        strategy: str,
    ) -> Tuple[HeteroData, Dict[str, Any]]:
        """
        节点全复制 + labeled edges 按 SuperBlock 分配。
        非 labeled edges 保持原图复制，和原始 balanced_edge 策略的语义保持一致。

        本版本额外记录 partition 子阶段耗时，用于后续拟合：
          - graph_clone_init_sec
          - partition_prepare_sec
          - target_group_build_sec
          - superblock_pack_sec
          - assignment_sec
          - rank_block_collect_sec
          - edge_select_assign_sec
          - save_plan_sec
        """
        partition_start = time.perf_counter()
        stage_times: Dict[str, float] = {}

        # 0. clone graph and initialize empty edge stores
        t = time.perf_counter()
        sub = graph.clone()
        self._init_empty_edge_stores(sub, graph)
        stage_times["graph_clone_init_sec"] = float(time.perf_counter() - t)

        # 1. prepare metadata
        t = time.perf_counter()
        edge_types = [et for et in LABELED_EDGE_TYPES if et in graph.edge_types]
        if not edge_types:
            logger.warning(f"[Rank {rank}] no labeled edge types found, fallback to balanced_edge")
            return self._partition_graph_by_edges_balanced(graph, world_size, rank)

        node_info = self._collect_node_info(graph)

        profile_cost_map = self._load_profile_cost_map() if strategy == "superblock_calibrated_cost" else None
        target_super_block_size = self._get_target_super_block_size()
        stage_times["partition_prepare_sec"] = float(time.perf_counter() - t)

        # 2. build target groups
        t = time.perf_counter()
        target_groups = self._build_target_groups(graph, edge_types)
        stage_times["target_group_build_sec"] = float(time.perf_counter() - t)

        # 3. pack target groups into SuperBlocks
        t = time.perf_counter()
        blocks = self._pack_target_groups_to_superblocks(
            target_groups=target_groups,
            target_super_block_size=target_super_block_size,
            profile_cost_map=profile_cost_map,
            strategy=strategy,
        )
        stage_times["superblock_pack_sec"] = float(time.perf_counter() - t)

        # 4. assign SuperBlocks to ranks
        t = time.perf_counter()
        assignment = self._assign_superblocks_to_ranks(
            blocks=blocks,
            world_size=world_size,
            strategy=strategy,
        )
        stage_times["assignment_sec"] = float(time.perf_counter() - t)

        # Record the exact list-scheduling quantities used by the SIGMOD
        # evaluation.  These scalars are cheap to keep even when the (very
        # large) per-SuperBlock assignment CSV is disabled.  For
        # ``superblock_edge_count`` the weight is the number of edges; for the
        # vector-aware strategy it is the same estimated work used by the
        # greedy assignment routine.
        block_weights = {
            int(b.block_id): max(0.0, float(self._get_block_cost_for_strategy(b, strategy)))
            for b in blocks
        }
        rank_work_loads = {p: 0.0 for p in range(world_size)}
        for b in blocks:
            assigned_rank = int(assignment[b.block_id])
            rank_work_loads[assigned_rank] += block_weights[int(b.block_id)]
        total_work_w = float(sum(block_weights.values()))
        w_max = float(max(block_weights.values())) if block_weights else 0.0
        w_over_g = total_work_w / max(1, world_size)
        lower_bound = max(w_over_g, w_max)
        greedy_upper_bound = w_over_g + w_max
        actual_makespan = float(max(rank_work_loads.values())) if rank_work_loads else 0.0

        # 5. collect blocks assigned to current rank
        t = time.perf_counter()
        selected_by_edge_type: Dict[EdgeType, List[torch.Tensor]] = {et: [] for et in edge_types}
        rank_selected_num_blocks = 0
        rank_selected_num_edges = 0
        rank_selected_num_target_groups = 0

        for b in blocks:
            if assignment.get(b.block_id) == rank:
                selected_by_edge_type[b.edge_type].append(b.edge_ids)
                rank_selected_num_blocks += 1
                rank_selected_num_edges += int(b.num_edges)
                rank_selected_num_target_groups += int(b.num_target_groups)

        stage_times["rank_block_collect_sec"] = float(time.perf_counter() - t)

        # 6. select and assign edge data for current rank
        t = time.perf_counter()
        edge_eids_mapping: Dict[EdgeType, List[int]] = {}
        edge_type_allocation: Dict[EdgeType, int] = {}
        edge_type_select_time: Dict[str, float] = {}

        for edge_type in edge_types:
            edge_type_t = time.perf_counter()
            edge_store = graph[edge_type]

            if edge_type not in selected_by_edge_type or len(selected_by_edge_type[edge_type]) == 0:
                self._set_empty_edge_type(sub, graph, edge_type)
                edge_eids_mapping[edge_type] = []
                edge_type_allocation[edge_type] = 0
                edge_type_select_time[edge_type_to_str(edge_type)] = float(time.perf_counter() - edge_type_t)
                continue

            selected_indices = torch.cat(selected_by_edge_type[edge_type], dim=0).long()
            selected_indices = torch.unique(selected_indices, sorted=True)

            src_type, _, dst_type = edge_type
            src_num_nodes = node_info.get(src_type, 0)
            dst_num_nodes = node_info.get(dst_type, 0)

            # 复用原始 validate 逻辑，但这里不是 start/end 连续段，所以单独验证。
            selected_indices = self._validate_selected_edge_indices(
                edge_store=edge_store,
                selected_indices=selected_indices,
                src_num_nodes=src_num_nodes,
                dst_num_nodes=dst_num_nodes,
            )

            if selected_indices.numel() == 0:
                self._set_empty_edge_type(sub, graph, edge_type)
                edge_eids_mapping[edge_type] = []
                edge_type_allocation[edge_type] = 0
                edge_type_select_time[edge_type_to_str(edge_type)] = float(time.perf_counter() - edge_type_t)
                continue

            if hasattr(edge_store, "eid"):
                eids = edge_store.eid[selected_indices].cpu().tolist()
            else:
                eids = selected_indices.cpu().tolist()

            self._assign_edge_data(sub, edge_type, edge_store, selected_indices.to(edge_store.edge_index.device), eids)
            edge_eids_mapping[edge_type] = eids
            edge_type_allocation[edge_type] = int(selected_indices.numel())
            edge_type_select_time[edge_type_to_str(edge_type)] = float(time.perf_counter() - edge_type_t)

        stage_times["edge_select_assign_sec"] = float(time.perf_counter() - t)

        partition_time_sec = float(time.perf_counter() - partition_start)

        # Also expose the per-edge-type select times as flattened scalar fields.
        for et_str, sec in edge_type_select_time.items():
            safe_et = et_str.replace("__", "_")
            stage_times[f"edge_select_assign_sec__{safe_et}"] = float(sec)

        mapping_info = {
            "rank": rank,
            "world_size": world_size,
            "partition_strategy": strategy,
            "target_super_block_size": target_super_block_size,
            "num_target_groups": len(target_groups),
            "num_superblocks": len(blocks),
            "rank_selected_num_blocks": int(rank_selected_num_blocks),
            "rank_selected_num_edges": int(rank_selected_num_edges),
            "rank_selected_num_target_groups": int(rank_selected_num_target_groups),
            "rank_selected_estimated_work": float(rank_work_loads.get(rank, 0.0)),
            "assignment_weight_kind": (
                "edge_count" if strategy == "superblock_edge_count" else "estimated_cost"
            ),
            "total_work_W": total_work_w,
            "W_over_G": float(w_over_g),
            "w_max": w_max,
            "lower_bound": float(lower_bound),
            "greedy_upper_bound": float(greedy_upper_bound),
            "actual_makespan": actual_makespan,
            "makespan_over_lower_bound": (
                float(actual_makespan / lower_bound) if lower_bound > 0 else 0.0
            ),
            "within_greedy_bound": bool(actual_makespan <= greedy_upper_bound + 1e-9),
            "edge_eids": edge_eids_mapping,
            "edge_type_allocation": {edge_type_to_str(k): int(v) for k, v in edge_type_allocation.items()},
            "node_info": node_info,
            "partition_time_sec": partition_time_sec,
            **stage_times,
        }

        # 7. save partition plan / diagnostics. This is timed separately because
        # rank0 writes full assignment files while nonzero ranks only write their
        # mapping_info and wait at the barrier.
        save_t = time.perf_counter()
        self._save_superblock_partition_plan(
            graph=graph,
            blocks=blocks,
            assignment=assignment,
            world_size=world_size,
            strategy=strategy,
            mapping_info=mapping_info,
        )
        save_plan_sec = float(time.perf_counter() - save_t)
        mapping_info["save_plan_sec"] = save_plan_sec
        mapping_info["partition_time_with_save_sec"] = float(partition_time_sec + save_plan_sec)

        # Append a compact per-rank CSV row after save_plan_sec is known.
        self._append_partition_stage_record(mapping_info)

        logger.info(
            f"[Rank {rank}] {strategy}: "
            f"superblocks={len(blocks)}, target_groups={len(target_groups)}, "
            f"rank_blocks={rank_selected_num_blocks}, rank_edges={rank_selected_num_edges}, "
            f"edge_alloc={mapping_info['edge_type_allocation']}, "
            f"partition_time={partition_time_sec:.3f}s, save_plan={save_plan_sec:.3f}s"
        )

        return sub, mapping_info

    def _validate_selected_edge_indices(
        self,
        edge_store,
        selected_indices: torch.Tensor,
        src_num_nodes: int,
        dst_num_nodes: int,
    ) -> torch.Tensor:
        if selected_indices.device != edge_store.edge_index.device:
            selected_indices = selected_indices.to(edge_store.edge_index.device)

        edge_index_selected = edge_store.edge_index[:, selected_indices]
        src_indices = edge_index_selected[0]
        dst_indices = edge_index_selected[1]

        valid_mask = (
            (src_indices >= 0)
            & (src_indices < src_num_nodes)
            & (dst_indices >= 0)
            & (dst_indices < dst_num_nodes)
        )

        if not bool(valid_mask.all()):
            invalid_count = int((~valid_mask).sum().item())
            logger.warning(f"[Rank {self.rank}] filtered invalid selected edges: {invalid_count}")
            selected_indices = selected_indices[valid_mask]

        return selected_indices

    # ============================================================
    # Diagnostics / output
    # ============================================================

    def _save_superblock_partition_plan(
        self,
        graph: HeteroData,
        blocks: List[SuperBlock],
        assignment: Dict[int, int],
        world_size: int,
        strategy: str,
        mapping_info: Dict[str, Any],
    ):
        save_partition_plan = bool(getattr(self.config.training_config, "save_partition_plan", True))
        if not save_partition_plan:
            return

        out_dir = self._get_assignment_output_dir()

        # 每个 rank 都写自己的 mapping_info，rank0 写完整 assignment。
        rank_info_path = out_dir / f"mapping_info_{strategy}_rank{self.rank}.json"
        with rank_info_path.open("w", encoding="utf-8") as f:
            json.dump(self._json_safe(mapping_info), f, ensure_ascii=False, indent=2)

        if self.use_distributed and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass

        if self.rank != 0:
            return

        rows = []
        rank_load = {p: 0.0 for p in range(world_size)}
        rank_edges = {p: 0 for p in range(world_size)}
        rank_blocks = {p: 0 for p in range(world_size)}
        target_to_ranks: Dict[Tuple[str, int], set] = {}

        for b in blocks:
            r = int(assignment[b.block_id])
            rank_load[r] += float(self._get_block_cost_for_strategy(b, strategy))
            rank_edges[r] += int(b.num_edges)
            rank_blocks[r] += 1

            for dst_id in b.target_node_ids:
                target_to_ranks.setdefault((b.dst_node_type, int(dst_id)), set()).add(r)

            rows.append(
                {
                    "strategy": strategy,
                    "block_id": b.block_id,
                    "edge_type": edge_type_to_str(b.edge_type),
                    "assigned_rank": r,
                    "num_edges": b.num_edges,
                    "num_target_groups": b.num_target_groups,
                    "calibrated_cost": b.calibrated_cost,
                    "target_id_min": min(b.target_node_ids) if b.target_node_ids else -1,
                    "target_id_max": max(b.target_node_ids) if b.target_node_ids else -1,
                    "estimated_cost": float(self._get_block_cost_for_strategy(b, strategy)),
                }
            )

        assignment_df = pd.DataFrame(rows)
        assignment_df.to_csv(out_dir / f"superblock_assignment_{strategy}.csv", index=False, encoding="utf-8-sig")

        rho_values = [len(v) for v in target_to_ranks.values()]
        conflict_nodes = sum(1 for x in rho_values if x > 1)
        aggregation_contention = sum(max(0, x - 1) for x in rho_values)

        avg_load = sum(rank_load.values()) / max(1, world_size)
        summary_rows = []
        for p in range(world_size):
            summary_rows.append(
                {
                    "strategy": strategy,
                    "rank": p,
                    "num_blocks": rank_blocks[p],
                    "num_edges": rank_edges[p],
                    # Backward-compatible name used by old summary scripts.
                    "calibrated_load": rank_load[p],
                    # Preferred name for the current time-oriented model.
                    "estimated_load": rank_load[p],
                    "load_imbalance_ratio": rank_load[p] / avg_load if avg_load > 0 else 0.0,
                    "block_count_imbalance_ratio": (
                        rank_blocks[p] / (sum(rank_blocks.values()) / max(1, world_size))
                        if sum(rank_blocks.values()) > 0 else 0.0
                    ),
                    "edge_count_imbalance_ratio": (
                        rank_edges[p] / (sum(rank_edges.values()) / max(1, world_size))
                        if sum(rank_edges.values()) > 0 else 0.0
                    ),
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / f"rank_load_summary_{strategy}.csv", index=False, encoding="utf-8-sig")

        # Edge-type load balance diagnostics. This is important for the scientific
        # part of the assignment model: ranks should be balanced not only in total
        # edge load, but also in heterogeneous edge-type composition.
        rank_type_edges = {}
        for p in range(world_size):
            rank_type_edges[p] = {}
        for b in blocks:
            r = int(assignment[b.block_id])
            et = edge_type_to_str(b.edge_type)
            rank_type_edges[r][et] = rank_type_edges[r].get(et, 0) + int(b.num_edges)

        type_rows = []
        type_imbalance_ratio = 1.0
        all_type_keys = sorted({edge_type_to_str(b.edge_type) for b in blocks})
        for et in all_type_keys:
            vals = [float(rank_type_edges[p].get(et, 0.0)) for p in range(world_size)]
            avg_val = sum(vals) / max(1, world_size)
            if avg_val > 0:
                type_imbalance_ratio = max(type_imbalance_ratio, max(vals) / avg_val)
            for p, val in enumerate(vals):
                type_rows.append({
                    "strategy": strategy,
                    "edge_type": et,
                    "rank": p,
                    "num_edges": int(val),
                    "avg_edges": float(avg_val),
                    "type_imbalance_to_avg": float(val / avg_val) if avg_val > 0 else 0.0,
                })
        pd.DataFrame(type_rows).to_csv(
            out_dir / f"rank_edge_type_load_summary_{strategy}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        avg_edges_global = sum(rank_edges.values()) / max(1, world_size)
        avg_blocks_global = sum(rank_blocks.values()) / max(1, world_size)

        global_summary = {
            "strategy": strategy,
            "world_size": world_size,
            "num_superblocks": len(blocks),
            "total_edges": int(sum(b.num_edges for b in blocks)),
            "total_targets": int(len(target_to_ranks)),
            "conflict_nodes_Na": int(conflict_nodes),
            "aggregation_contention_Ra": int(aggregation_contention),
            "avg_rho": float(sum(rho_values) / max(1, len(rho_values))) if rho_values else 0.0,
            "max_rho": int(max(rho_values)) if rho_values else 0,
            "max_rank_load": float(max(rank_load.values())) if rank_load else 0.0,
            "avg_rank_load": float(avg_load),
            # Preferred interpretation: estimated cost/load imbalance.
            "global_load_imbalance_ratio": float(max(rank_load.values()) / avg_load) if avg_load > 0 else 0.0,
            "estimated_cost_imbalance_ratio": float(max(rank_load.values()) / avg_load) if avg_load > 0 else 0.0,
            # These are diagnostics / feasibility guards rather than primary objectives.
            "edge_count_imbalance_ratio": float(max(rank_edges.values()) / avg_edges_global) if avg_edges_global > 0 else 0.0,
            "block_count_imbalance_ratio": float(max(rank_blocks.values()) / avg_blocks_global) if avg_blocks_global > 0 else 0.0,
            "edge_type_imbalance_ratio": float(type_imbalance_ratio),
            "rank_loads": {str(k): float(v) for k, v in rank_load.items()},
            "rank_edges": {str(k): int(v) for k, v in rank_edges.items()},
            "rank_blocks": {str(k): int(v) for k, v in rank_blocks.items()},
        }
        with (out_dir / f"partition_global_summary_{strategy}.json").open("w", encoding="utf-8") as f:
            json.dump(global_summary, f, ensure_ascii=False, indent=2)

        logger.info(f"[Rank 0] saved SuperBlock partition plan to {out_dir}")

    def _append_runtime_record(self, record: Dict[str, Any]):
        try:
            out_dir = self._get_assignment_output_dir()
            path = out_dir / f"runtime_rank{getattr(self, 'rank', 0)}.csv"
            df = pd.DataFrame([record])
            if path.exists():
                df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.debug(f"failed to append runtime record: {e}")


    def _append_partition_stage_record(self, mapping_info: Dict[str, Any]):
        """
        Append compact partition-stage timing diagnostics for later fitting.

        This file intentionally excludes huge fields such as edge_eids and node_info.
        One row is written per rank and per first-time partition construction.
        """
        try:
            out_dir = self._get_assignment_output_dir()
            path = out_dir / f"partition_stage_timing_rank{getattr(self, 'rank', 0)}.csv"

            scalar_keys = [
                "rank",
                "world_size",
                "partition_strategy",
                "target_super_block_size",
                "num_target_groups",
                "num_superblocks",
                "rank_selected_num_blocks",
                "rank_selected_num_edges",
                "rank_selected_num_target_groups",
                "rank_selected_estimated_work",
                "assignment_weight_kind",
                "total_work_W",
                "W_over_G",
                "w_max",
                "lower_bound",
                "greedy_upper_bound",
                "actual_makespan",
                "makespan_over_lower_bound",
                "within_greedy_bound",
                "partition_time_sec",
                "partition_time_with_save_sec",
                "save_plan_sec",
                "graph_clone_init_sec",
                "partition_prepare_sec",
                "target_group_build_sec",
                "superblock_pack_sec",
                "assignment_sec",
                "rank_block_collect_sec",
                "edge_select_assign_sec",
            ]

            row: Dict[str, Any] = {}
            for key in scalar_keys:
                if key in mapping_info:
                    row[key] = mapping_info[key]

            # Flatten edge allocation into scalar columns.
            edge_alloc = mapping_info.get("edge_type_allocation", {}) or {}
            for et, val in edge_alloc.items():
                row[f"edge_alloc__{et}"] = int(val)

            # Include per-edge-type select timing fields.
            for key, val in mapping_info.items():
                if key.startswith("edge_select_assign_sec__"):
                    row[key] = float(val)

            row["timestamp"] = time.time()

            df = pd.DataFrame([row])
            if path.exists():
                df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")

        except Exception as e:
            logger.debug(f"failed to append partition stage record: {e}")

    def _json_safe(self, obj):
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return obj

    def _get_block_cost_for_strategy(self, b, strategy: str) -> float:
        if strategy == "superblock_calibrated_cost":
            return float(b.calibrated_cost)
        if strategy == "superblock_cost_balance":
            return float(self.estimate_block_cost(b))
        return float(b.num_edges)

