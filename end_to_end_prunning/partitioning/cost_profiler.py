from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch

from .edge_block import EdgeBlock


@dataclass
class BlockProfileSample:
    block_id: int
    edge_type: Tuple[str, str, str]
    num_edges: int
    edge_feat_dim: int
    num_labeled_edges: int
    forward_time_ms: float
    peak_memory_bytes: int


@dataclass
class ProfilingSummary:
    samples: List[BlockProfileSample] = field(default_factory=list)
    edge_type_stats: Dict[Tuple[str, str, str], Dict[str, float]] = field(default_factory=dict)

    def add_sample(self, sample: BlockProfileSample) -> None:
        self.samples.append(sample)
        stats = self.edge_type_stats.setdefault(sample.edge_type, {"count": 0, "time_ms": 0.0, "memory_bytes": 0})
        stats["count"] += 1
        stats["time_ms"] += float(sample.forward_time_ms)
        stats["memory_bytes"] += int(sample.peak_memory_bytes)

    def edge_type_summary(self) -> Dict[Tuple[str, str, str], Dict[str, float]]:
        result = {}
        for edge_type, stats in self.edge_type_stats.items():
            count = max(1, int(stats["count"]))
            result[edge_type] = {
                "count": count,
                "avg_forward_time_ms": stats["time_ms"] / count,
                "avg_peak_memory_bytes": stats["memory_bytes"] / count,
            }
        return result


def _edge_ids_to_index_tensor(edge_ids: torch.Tensor, original_edge_index: torch.Tensor) -> torch.Tensor:
    if original_edge_index is None:
        return torch.empty((2, edge_ids.numel()), dtype=torch.long, device=edge_ids.device)
    return original_edge_index[:, edge_ids].contiguous()


def profile_edge_block(
    block: EdgeBlock,
    data: torch.Tensor,
    *,
    device: Union[str, torch.device] = "cpu",
    warmup_iters: int = 1,
    profile_iters: int = 1,
    max_edges: int = 10000,
) -> BlockProfileSample:
    """对单个 EdgeBlock 做最小代表性 profiling。"""
    if not isinstance(device, torch.device):
        device = torch.device(device)
    torch.set_default_device(device)

    original_edge_index = getattr(data, "edge_index", None)
    edge_ids = block.edge_ids.detach().to(device)
    if edge_ids.numel() > max_edges:
        edge_ids = edge_ids[torch.randperm(edge_ids.numel(), device=device)[:max_edges]]

    edge_index = _edge_ids_to_index_tensor(edge_ids, original_edge_index)
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_ids.numel())
    hidden_dim = 64
    x_src = torch.randn((int(src.numel()), hidden_dim), device=device)
    x_dst = torch.randn((int(dst.unique().numel()), hidden_dim), device=device)

    model = torch.nn.Linear(hidden_dim * 2 + block.edge_feat_dim, 1, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(max(1, warmup_iters)):
        optimizer.zero_grad(set_to_none=True)
        h_src = x_src[src]
        h_dst = x_dst[dst]
        edge_attr = torch.randn((num_edges, max(1, block.edge_feat_dim)), device=device)
        out = model(torch.cat([h_src, h_dst, edge_attr], dim=-1))
        out.sum().backward()
        optimizer.step()

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(device)

    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    total_time_ms = 0.0
    peak_memory_bytes = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event.record()

    for _ in range(max(1, profile_iters)):
        optimizer.zero_grad(set_to_none=True)
        h_src = x_src[src]
        h_dst = x_dst[dst]
        edge_attr = torch.randn((num_edges, max(1, block.edge_feat_dim)), device=device)
        out = model(torch.cat([h_src, h_dst, edge_attr], dim=-1))
        loss = out.sum()
        loss.backward()
        optimizer.step()
        total_time_ms += float(loss.item())

    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize(device)
        total_time_ms = float(start_event.elapsed_time(end_event))
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        total_time_ms = float(total_time_ms)
        peak_memory_bytes = int(x_src.numel() * 4 + x_dst.numel() * 4 + num_edges * 4 * (hidden_dim * 2 + max(1, block.edge_feat_dim)))

    return BlockProfileSample(
        block_id=int(block.block_id),
        edge_type=tuple(block.edge_type),
        num_edges=int(block.num_edges),
        edge_feat_dim=int(block.edge_feat_dim),
        num_labeled_edges=int(block.num_labeled_edges),
        forward_time_ms=max(0.0, float(total_time_ms)),
        peak_memory_bytes=max(0, int(peak_memory_bytes)),
    )


def sample_blocks_for_profiling(
    blocks: List[EdgeBlock],
    *,
    samples_per_type: int = 5,
    max_samples: int = 40,
    seed: Optional[int] = 42,
) -> List[EdgeBlock]:
    if seed is not None:
        random.seed(seed)

    buckets: Dict[Tuple[str, str, str], List[EdgeBlock]] = {}
    for block in blocks:
        buckets.setdefault(block.edge_type, []).append(block)

    selected: List[EdgeBlock] = []
    for edge_type, bucket in buckets.items():
        bucket = sorted(bucket, key=lambda b: b.num_edges)
        step = max(1, len(bucket) // max(1, samples_per_type))
        sampled = bucket[::step][:samples_per_type]
        selected.extend(sampled)

    selected = sorted(selected, key=lambda b: b.block_id)
    if len(selected) > max_samples:
        selected = selected[:max_samples]
    return selected

