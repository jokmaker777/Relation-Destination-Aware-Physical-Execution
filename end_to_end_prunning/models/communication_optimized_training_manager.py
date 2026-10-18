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
- 它不实现跨 rank 的精确 GNN 聚合 all_reduce，因此论文表述时应写作：
  “training runtime comparison under communication-aware edge partitioning strategy”
  而不是声称已经实现 exact distributed aggregation semantics。
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
        try:
            selected_edge_types = ";".join(edge_type_to_str(et) for et in self._get_partition_edge_types_for_graph(graph))
        except Exception:
            selected_edge_types = ""
        if strategy == "superblock_relation_cost_balance":
            try:
                tc = self.config.training_config
                relation_cost_signature = str(getattr(tc, "cost_relation_beta_map", "")) + "|" + str(getattr(tc, "cost_relation_alpha_map", ""))
            except Exception:
                relation_cost_signature = ""
        else:
            relation_cost_signature = ""
        return (
            str(graph_date),
            int(id(graph)),
            strategy,
            selected_edge_types,
            relation_cost_signature,
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
                            "superblock_relation_cost_balance",
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

    def train_epoch(self, graphs: List[HeteroData]) -> float:
        """
        Stage-timed training epoch.

        Compared with the parent TrainingManager, this method keeps the same training
        semantics but explicitly times:
          - partition / cached partition reuse
          - data movement
          - forward + loss
          - backward
          - optimizer
          - DDP loss synchronization

        The resulting stage_timing_rank*.csv files support overlap-aware runtime modeling.
        """
        import math

        self.model.train()
        model_device = next(self.model.parameters()).device

        epoch_idx = int(getattr(self, "_stage_epoch_counter", 0)) + 1
        self._stage_epoch_counter = epoch_idx
        timer = getattr(self, "stage_timer", None)
        if timer is not None:
            timer.start_epoch(epoch_idx)

        # Distributed graph allocation / SuperBlock partition.
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

        # DDP data-state synchronization.
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

                # forward + loss
                if timer is not None:
                    forward_ctx = timer.stage("forward_loss")
                    forward_ctx.__enter__()
                else:
                    forward_ctx = None
                try:
                    if self.use_amp and self._amp_device_type is not None:
                        with torch.amp.autocast(device_type=self._amp_device_type, dtype=self.amp_dtype):
                            predictions = self.model(graph)

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

                        predictions_fp32 = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in predictions.items()}
                        if hasattr(self.model, "module"):
                            loss = self.model.module.compute_loss(predictions_fp32, graph)
                        else:
                            loss = self.model.compute_loss(predictions_fp32, graph)
                    else:
                        predictions = self.model(graph)
                        if not predictions:
                            logger.warning(f"Empty predictions for graph {i}, skipping")
                            continue
                        if hasattr(self.model, "module"):
                            loss = self.model.module.compute_loss(predictions, graph)
                        else:
                            loss = self.model.compute_loss(predictions, graph)
                finally:
                    if forward_ctx is not None:
                        forward_ctx.__exit__(None, None, None)

                loss_value = loss.item()
                if math.isnan(loss_value) or math.isinf(loss_value):
                    logger.warning(f"Invalid loss for graph {i}: {loss_value}, skipping")
                    continue

                # backward
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

                # optimizer
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

                # Device status log.
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
            if timer is not None:
                timer.end_epoch(train_loss=float("inf"), extra={"successful_batches": 0})
            return float("inf")

        avg_loss = total_loss / max(1, successful_batches)

        # DDP loss synchronization.
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

        # Keep the original runtime record for backward compatibility with existing summary scripts.
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

    def _get_partition_edge_types(self) -> List[EdgeType]:
        """Return edge types that should be partitioned by SuperBlock.

        Default keeps the original labeled-only behavior.  For relation-aware
        cost ablation, the runner can override this through
        training_config.partition_edge_types or env PARTITION_EDGE_TYPES.

        Accepted formats:
          - JSON list of strings: ["src__rel__dst", ...]
          - JSON list of triples: [["src", "rel", "dst"], ...]
          - comma-separated strings: src__rel__dst,src__rel__dst
        """
        value = None
        try:
            value = getattr(self.config.training_config, "partition_edge_types", None)
        except Exception:
            value = None
        if value is None or value == "":
            value = os.environ.get("PARTITION_EDGE_TYPES", "")

        if isinstance(value, (list, tuple)):
            items = list(value)
        elif isinstance(value, str) and value.strip():
            raw = value.strip()
            try:
                items = json.loads(raw)
                if not isinstance(items, list):
                    items = []
            except Exception:
                items = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            items = []

        parsed: List[EdgeType] = []
        for item in items:
            try:
                if isinstance(item, str):
                    parsed.append(parse_edge_type(item))
                elif isinstance(item, (list, tuple)) and len(item) == 3:
                    parsed.append((str(item[0]), str(item[1]), str(item[2])))
            except Exception:
                continue

        if not parsed:
            parsed = list(LABELED_EDGE_TYPES)

        seen = set()
        out: List[EdgeType] = []
        for et in parsed:
            if et not in seen:
                seen.add(et)
                out.append(et)
        return out

    def _get_partition_edge_types_for_graph(self, graph: HeteroData) -> List[EdgeType]:
        available = set([et for et in graph.edge_types if isinstance(et, tuple) and len(et) == 3])
        return [et for et in self._get_partition_edge_types() if et in available]

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
        Relation-aware SuperBlock execution cost.

        The cost model follows the profiling-inspired alpha/beta form:

            C(B_k) = alpha_r * 1{n_{k,r}>0} + beta_r * n_{k,r}
                     + gamma * num_target_groups.

        In the current packer, each SuperBlock contains one relation, so
        n_{k,r}=b.num_edges.  Coefficients are relative cost units and are used
        only for greedy assignment; they are not claimed as absolute runtime.
        """
        edge_type_str = edge_type_to_str(b.edge_type)
        tc = self.config.training_config

        default_alpha = float(getattr(tc, "cost_fixed_block", 500.0))
        target_group_cost = float(getattr(tc, "cost_per_target_group", 0.05))
        default_beta = float(getattr(tc, "cost_per_edge", 1.0))

        default_beta_map = {
            "inventory__to__virtual": 1.00,
            "inventory__to_transport_labeled__inventory": 2.45,
            "production__to_2__inventory": 2.10,
            "inventory__to_remain__inventory": 0.81,
            "inventory__to_2__production": 0.44,
        }
        default_alpha_map = {
            "inventory__to__virtual": 350.0,
            "inventory__to_transport_labeled__inventory": 500.0,
            "production__to_2__inventory": 500.0,
            "inventory__to_remain__inventory": 450.0,
            "inventory__to_2__production": 400.0,
        }

        def _load_json_map(attr_name: str, fallback: Dict[str, float]) -> Dict[str, float]:
            value = getattr(tc, attr_name, None)
            if value is None or value == "":
                value = os.environ.get(attr_name.upper(), "")
            if value is None or value == "":
                return dict(fallback)
            if isinstance(value, dict):
                try:
                    return {str(k): float(v) for k, v in value.items()}
                except Exception:
                    return dict(fallback)
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        return {str(k): float(v) for k, v in parsed.items()}
                except Exception:
                    return dict(fallback)
            return dict(fallback)

        beta_map = _load_json_map("cost_relation_beta_map", default_beta_map)
        alpha_map = _load_json_map("cost_relation_alpha_map", default_alpha_map)

        type_factor_map = getattr(tc, "cost_type_factor_map", None)
        if isinstance(type_factor_map, str):
            try:
                type_factor_map = json.loads(type_factor_map)
            except Exception:
                type_factor_map = {}
        if not isinstance(type_factor_map, dict):
            type_factor_map = {}

        alpha = float(alpha_map.get(edge_type_str, default_alpha))
        beta = float(beta_map.get(edge_type_str, default_beta))
        beta *= float(type_factor_map.get(edge_type_str, 1.0))

        cost = 0.0
        if int(b.num_edges) > 0:
            cost += alpha
        cost += beta * float(b.num_edges)
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
        节点全复制 + configured relations 按 SuperBlock 分配。
        未被配置的 relations 保持原图复制，和原始 balanced_edge 策略的语义保持一致。

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
        edge_types = self._get_partition_edge_types_for_graph(graph)
        if not edge_types:
            logger.warning(f"[Rank {rank}] no configured partition edge types found, fallback to balanced_edge")
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
            "partition_edge_types": [edge_type_to_str(et) for et in edge_types],
            "target_super_block_size": target_super_block_size,
            "num_target_groups": len(target_groups),
            "num_superblocks": len(blocks),
            "rank_selected_num_blocks": int(rank_selected_num_blocks),
            "rank_selected_num_edges": int(rank_selected_num_edges),
            "rank_selected_num_target_groups": int(rank_selected_num_target_groups),
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
        if strategy in ("superblock_cost_balance", "superblock_relation_cost_balance"):
            return float(self.estimate_block_cost(b))
        return float(b.num_edges)

