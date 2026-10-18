# -*- coding: utf-8 -*-
"""
stage_timing_utils_no_parser.py

A lightweight timing utility for DDP GNN training.
Use this file by importing StageTimer in your TrainingManager / CommunicationOptimizedTrainingManager.

It records per-epoch stage times:
  - partition_sec          : graph partition or cached partition loading time
  - data_sec               : optional data transfer / preparation time
  - forward_loss_sec       : model forward + loss computation
  - backward_sec           : loss.backward(); DDP gradient communication is usually included here
  - optimizer_sec          : optimizer.step() / scheduler if included
  - epoch_total_sec        : full epoch wall time
  - rank wait approximate : max stage/epoch time across ranks minus local time

No argparse. All user-facing behavior is controlled by constructor arguments.
"""

import os
import time
import json
from contextlib import contextmanager
from typing import Dict, Any, Optional, List

import pandas as pd
import torch
import torch.distributed as dist


class StageTimer:
    def __init__(
        self,
        rank: int,
        world_size: int,
        device: Optional[torch.device] = None,
        output_dir: str = ".",
        enabled: bool = True,
        sync_cuda: bool = True,
        eta_comm: float = 0.5,
        eta_data: float = 0.3,
    ):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.sync_cuda = bool(sync_cuda)
        self.eta_comm = float(eta_comm)
        self.eta_data = float(eta_data)

        self.current_epoch: Optional[int] = None
        self.epoch_start_time: Optional[float] = None
        self.stage_times: Dict[str, float] = {}
        self.records: List[Dict[str, Any]] = []

        os.makedirs(self.output_dir, exist_ok=True)

    def _cuda_sync(self) -> None:
        if not self.enabled or not self.sync_cuda:
            return
        if torch.cuda.is_available():
            try:
                if self.device is not None:
                    torch.cuda.synchronize(self.device)
                else:
                    torch.cuda.synchronize()
            except Exception:
                pass

    def _dist_ready(self) -> bool:
        return dist.is_available() and dist.is_initialized() and self.world_size > 1

    def _all_reduce_scalar(self, value: float, op: str) -> float:
        if not self._dist_ready():
            return float(value)
        device = self.device if self.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.tensor([float(value)], dtype=torch.float64, device=device)
        if op == "max":
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        elif op == "min":
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
        elif op == "sum":
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        else:
            raise ValueError(f"Unsupported reduce op: {op}")
        return float(t.item())

    def _global_stats(self, value: float) -> Dict[str, float]:
        max_v = self._all_reduce_scalar(value, "max")
        min_v = self._all_reduce_scalar(value, "min")
        sum_v = self._all_reduce_scalar(value, "sum")
        avg_v = sum_v / max(1, self.world_size)
        return {"max": max_v, "min": min_v, "avg": avg_v, "wait_approx": max_v - float(value)}

    def start_epoch(self, epoch: int) -> None:
        if not self.enabled:
            return
        self._cuda_sync()
        self.current_epoch = int(epoch)
        self.epoch_start_time = time.perf_counter()
        self.stage_times = {}

    @contextmanager
    def stage(self, name: str):
        """Context manager for a training stage.

        Example:
            with timer.stage('forward_loss'):
                pred = model(data)
                loss = compute_loss(pred, data)
        """
        if not self.enabled:
            yield
            return
        self._cuda_sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._cuda_sync()
            elapsed = time.perf_counter() - start
            self.stage_times[name] = self.stage_times.get(name, 0.0) + float(elapsed)

    def add_time(self, name: str, seconds: float) -> None:
        if not self.enabled:
            return
        self.stage_times[name] = self.stage_times.get(name, 0.0) + float(seconds)

    def end_epoch(
        self,
        train_loss: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        save_each_epoch: bool = True,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        self._cuda_sync()
        now = time.perf_counter()
        epoch_total = float(now - self.epoch_start_time) if self.epoch_start_time is not None else sum(self.stage_times.values())

        record: Dict[str, Any] = {
            "epoch": self.current_epoch,
            "rank": self.rank,
            "world_size": self.world_size,
            "epoch_total_sec": epoch_total,
            "train_loss": train_loss,
        }

        # Local stage times.
        for k, v in sorted(self.stage_times.items()):
            record[f"{k}_sec"] = float(v)

        # Global epoch stats and rank wait approximation.
        epoch_stats = self._global_stats(epoch_total)
        record["epoch_global_max_sec"] = epoch_stats["max"]
        record["epoch_global_min_sec"] = epoch_stats["min"]
        record["epoch_global_avg_sec"] = epoch_stats["avg"]
        record["epoch_rank_wait_approx_sec"] = epoch_stats["wait_approx"]

        # Global stage wait approximations.
        # The most useful one is backward_rank_wait_approx_sec, because DDP all-reduce is usually triggered there.
        for k, v in sorted(self.stage_times.items()):
            st = self._global_stats(float(v))
            record[f"{k}_global_max_sec"] = st["max"]
            record[f"{k}_global_min_sec"] = st["min"]
            record[f"{k}_global_avg_sec"] = st["avg"]
            record[f"{k}_rank_wait_approx_sec"] = st["wait_approx"]

        # Peak memory.
        if torch.cuda.is_available():
            try:
                record["peak_memory_mb"] = float(torch.cuda.max_memory_allocated(self.device) / 1024.0 / 1024.0)
                record["reserved_memory_mb"] = float(torch.cuda.max_memory_reserved(self.device) / 1024.0 / 1024.0)
            except Exception:
                pass

        # Overlap-aware proxy.
        fwd = float(record.get("forward_loss_sec", 0.0) or 0.0)
        bwd = float(record.get("backward_sec", 0.0) or 0.0)
        opt = float(record.get("optimizer_sec", 0.0) or 0.0)
        data = float(record.get("data_sec", 0.0) or 0.0)
        part = float(record.get("partition_sec", 0.0) or 0.0)
        # Approx communication exposure by backward imbalance. If explicit comm_sec exists, use it.
        comm = float(record.get("comm_sec", 0.0) or 0.0)
        if comm <= 0:
            comm = float(record.get("backward_rank_wait_approx_sec", 0.0) or 0.0)
        data_like = data + part
        residual_comm = max(0.0, comm - self.eta_comm * bwd)
        residual_data = max(0.0, data_like - self.eta_data * fwd)
        overlap_epoch = fwd + bwd + opt + residual_comm + residual_data
        seq_epoch = fwd + bwd + opt + comm + data_like
        record["comm_exposed_proxy_sec"] = comm
        record["data_exposed_proxy_sec"] = data_like
        record["residual_comm_sec"] = residual_comm
        record["residual_data_sec"] = residual_data
        record["overlap_epoch_sec"] = overlap_epoch
        record["seq_epoch_proxy_sec"] = seq_epoch
        if seq_epoch > 0:
            record["overlap_gain_proxy"] = (seq_epoch - overlap_epoch) / seq_epoch
        else:
            record["overlap_gain_proxy"] = 0.0

        if extra:
            for k, v in extra.items():
                record[k] = v

        self.records.append(record)
        if save_each_epoch:
            self.save()
        return record

    def save(self) -> None:
        if not self.enabled:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, f"stage_timing_rank{self.rank}.csv")
        json_path = os.path.join(self.output_dir, f"stage_timing_rank{self.rank}.json")
        df = pd.DataFrame(self.records)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)


# ==============================================================================
# Minimal integration example for TrainingManager
# ==============================================================================

INTEGRATION_EXAMPLE = r'''
# 1) Add import near the top of communication_optimized_training_manager.py or training_manager.py
from stage_timing_utils_no_parser import StageTimer

# 2) After output_dir/save_dir and device/rank/world_size are known, initialize once:
self.stage_timer = StageTimer(
    rank=self.rank,
    world_size=self.world_size,
    device=self.device,
    output_dir=self.save_dir,   # or the q-sweep output_dir
    enabled=True,
    sync_cuda=True,
    eta_comm=0.5,
    eta_data=0.3,
)

# 3) In each epoch loop:
for epoch in range(1, num_epochs + 1):
    self.stage_timer.start_epoch(epoch)

    with self.stage_timer.stage("partition"):
        # Call your existing graph distribution / partition function here.
        # If the partition is cached, this stage will become near-zero.
        train_graphs_rank = self._distribute_graphs_balanced(train_graphs)

    with self.stage_timer.stage("data"):
        # Optional: move graph data to device, build batch, or leave empty if already done elsewhere.
        pass

    self.optimizer.zero_grad(set_to_none=True)

    with self.stage_timer.stage("forward_loss"):
        outputs = self.model(batch)
        loss = self.compute_loss(outputs, batch)

    with self.stage_timer.stage("backward"):
        loss.backward()

    with self.stage_timer.stage("optimizer"):
        self.optimizer.step()

    self.stage_timer.end_epoch(train_loss=float(loss.detach().item()))

# 4) The timer writes:
#    stage_timing_rank0.csv, stage_timing_rank1.csv
# These files can be read automatically by q_selection_summary_no_parser.py.
'''


if __name__ == "__main__":
    print("This file provides StageTimer. Import it in your training manager.")
    print(INTEGRATION_EXAMPLE)

