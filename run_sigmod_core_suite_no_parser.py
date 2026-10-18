#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SIGMOD core experiment suite: one direct-run, restartable entry point.

There are deliberately no command-line arguments.  Edit only RUN_PROFILE or
STAGES_TO_RUN below, upload the project, and run this file.  Worker processes
are launched by this same file through torchrun, so no second public-benchmark
driver is required.

The default ``two_day_initial`` profile prioritizes the shortest defensible
paper loop.  ``publication_full`` expands repetitions without changing the
protocol.  Every job writes an atomic summary and is skipped on a rerun.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


# ---------------------------------------------------------------------------
# The only settings normally edited by the user.
# ---------------------------------------------------------------------------
RUN_PROFILE = "two_day_initial"  # change to "publication_full" for final runs
STAGES_TO_RUN = (
    "preflight",
    "active_destination_stats",
    "public_dblp_end_to_end",
    "industrial_4gpu_main",
    "industrial_scaling",
    "industrial_c2_check",
    "planner_total_cost",
    "collect_report",
)

SERVER_DATA_ROOT = Path(
    os.environ.get("SIGMOD_ARTIFACT_ROOT", "/path/to/artifact_workspace")
)
INDUSTRIAL_ROOT = Path(
    os.environ.get(
        "SIGMOD_OUTPUT_ROOT",
        str(SERVER_DATA_ROOT / "outputs"),
    )
)
PUBLIC_HGB_ROOT = Path(
    os.environ.get(
        "SIGMOD_PUBLIC_DATA_ROOT",
        str(SERVER_DATA_ROOT / "public_datasets"),
    )
) / "hgb"
CORE_ROOT = INDUSTRIAL_ROOT / "sigmod_core_suite"
GPU_IDS = (0, 1, 2, 3)

ALL_SNAPSHOTS = (
    "20241124", "20241125", "20241126", "20241127", "20241128", "20241129",
    "20241213", "20241214", "20241215", "20241216", "20241217", "20241218",
)
REPRESENTATIVE_SNAPSHOTS = ("20241124", "20241129", "20241218")

PUBLIC_GPU_COUNTS = (1, 2, 4)
PUBLIC_CONFIGS = (
    "native_pyg",
    "direct_dense",
    "static_dense",
    "superblock_dense",
    "planner_selected",
)
PUBLIC_EPOCHS = 50
PUBLIC_HIDDEN_DIM = 64
PUBLIC_BLOCK_EDGES = 25_000
PUBLIC_LEARNING_RATE = 0.01
PUBLIC_WEIGHT_DECAY = 5e-4
PUBLIC_WARMUP_EPOCHS = 5
INDUSTRIAL_EXPECTED_EPOCHS = 10

if RUN_PROFILE == "two_day_initial":
    PUBLIC_REPEATS = (1, 2, 3)
    INDUSTRIAL_MAIN_REPEATS = (1,)       # normally already measured; mostly skips
    INDUSTRIAL_SCALING_REPEATS = (1,)    # first paper figure quickly
    INDUSTRIAL_C2_REPEATS = (1, 2, 3)    # the one remaining chunk experiment
    STATS_SNAPSHOTS = REPRESENTATIVE_SNAPSHOTS
elif RUN_PROFILE == "publication_full":
    PUBLIC_REPEATS = (1, 2, 3)
    INDUSTRIAL_MAIN_REPEATS = (1, 2, 3)
    INDUSTRIAL_SCALING_REPEATS = (1, 2, 3)
    INDUSTRIAL_C2_REPEATS = (1, 2, 3)
    STATS_SNAPSHOTS = ALL_SNAPSHOTS
else:
    raise ValueError(f"unknown RUN_PROFILE={RUN_PROFILE!r}")


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "sigmod_experiments"
SERVER_ROOT = EXPERIMENT_ROOT / "server_no_parser"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(SERVER_ROOT))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def complete_json(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("completed"))
    except Exception:
        return False


def sync_max(value: float, device) -> float:
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def sync_sum(value: float, device) -> float:
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def barrier(device=None) -> None:
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        if device is not None and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier()


def multiclass_metrics(labels, predictions, num_classes: int) -> Tuple[float, float]:
    import torch

    labels = labels.detach().cpu().long()
    predictions = predictions.detach().cpu().long()
    micro = float((labels == predictions).float().mean().item())
    scores: List[float] = []
    for class_id in range(num_classes):
        true_positive = int(((labels == class_id) & (predictions == class_id)).sum())
        false_positive = int(((labels != class_id) & (predictions == class_id)).sum())
        false_negative = int(((labels == class_id) & (predictions != class_id)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return micro, float(statistics.mean(scores))


# ---------------------------------------------------------------------------
# Exact public-graph end-to-end benchmark.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PublicPlan:
    name: str
    policy: str
    compact: bool
    block_edges: int = PUBLIC_BLOCK_EDGES


class _AllReduceExact:
    """Factory avoids importing torch before a worker selects its GPU."""

    @staticmethod
    def apply(value, process_group=None):
        import torch
        import torch.distributed as dist

        class Operation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor, group):
                ctx.process_group = group
                if (
                    not dist.is_initialized()
                    or dist.get_world_size(group=group) == 1
                ):
                    return tensor
                result = tensor.contiguous().clone()
                dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
                return result

            @staticmethod
            def backward(ctx, gradient):
                group = ctx.process_group
                if (
                    not dist.is_initialized()
                    or dist.get_world_size(group=group) == 1
                ):
                    return gradient, None
                result = gradient.contiguous().clone()
                dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
                return result, None

        return Operation.apply(value, process_group)


def prepare_dblp(device):
    import torch
    from sigmod_exp.datasets import DatasetSpec, load_graph

    data = load_graph(DatasetSpec(name="DBLP", kind="hgb-dblp", root=str(PUBLIC_HGB_ROOT)))
    node_types = list(data.node_types)
    offsets: Dict[str, int] = {}
    node_counts: Dict[str, int] = {}
    total_nodes = 0
    for node_type in node_types:
        store = data[node_type]
        count = int(store.num_nodes if store.num_nodes is not None else store.x.shape[0])
        offsets[node_type] = total_nodes
        node_counts[node_type] = count
        total_nodes += count

    edge_types = sorted(list(data.edge_types), key=lambda value: "__".join(value))
    relation_edges = []
    homogeneous_edges = []
    homogeneous_types = []
    for relation_id, edge_type in enumerate(edge_types):
        edge = data[edge_type].edge_index.long().cpu().contiguous()
        relation_edges.append(
            {
                "edge_type": edge_type,
                "src_offset": offsets[edge_type[0]],
                "dst_offset": offsets[edge_type[2]],
                "num_dst": node_counts[edge_type[2]],
                "edge_index": edge,
            }
        )
        global_edge = edge.clone()
        global_edge[0] += offsets[edge_type[0]]
        global_edge[1] += offsets[edge_type[2]]
        homogeneous_edges.append(global_edge)
        homogeneous_types.append(torch.full((edge.shape[1],), relation_id, dtype=torch.long))

    features = {}
    input_dims = {}
    for node_type in node_types:
        value = getattr(data[node_type], "x", None)
        if value is not None:
            features[node_type] = value.float().to(device)
            input_dims[node_type] = int(value.shape[1])
        else:
            features[node_type] = None
            input_dims[node_type] = 0
    labels = data["author"].y.long().to(device)
    train_ids = torch.nonzero(data["author"].train_mask, as_tuple=False).view(-1).long().to(device)
    test_ids = torch.nonzero(data["author"].test_mask, as_tuple=False).view(-1).long().to(device)
    return {
        "data": data,
        "node_types": node_types,
        "node_counts": node_counts,
        "offsets": offsets,
        "total_nodes": total_nodes,
        "edge_types": edge_types,
        "relation_edges": relation_edges,
        "edge_index": torch.cat(homogeneous_edges, dim=1).to(device),
        "edge_type": torch.cat(homogeneous_types).to(device),
        "features": features,
        "input_dims": input_dims,
        "labels": labels,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "num_classes": int(labels.max().item()) + 1,
        "total_edges": int(sum(item["edge_index"].shape[1] for item in relation_edges)),
    }


def build_partition_bundle(
    problem,
    plan: PublicPlan,
    rank: int,
    world_size: int,
    device,
    relation_process_groups=None,
):
    import torch
    from sigmod_exp.partition import PhysicalPlan, build_local_partition

    started = time.perf_counter()
    relations = []
    metadata_rows = []
    for relation_id, item in enumerate(problem["relation_edges"]):
        physical = PhysicalPlan(
            policy=plan.policy,
            block_edges=plan.block_edges,
            destination_ranges=1,
            collective_mode="blocking",
            aggregation="mean",
        )
        local_edge, metadata = build_local_partition(
            item["edge_index"], item["num_dst"], rank, world_size, physical
        )
        original_dst = item["edge_index"][1]
        degree = torch.bincount(original_dst, minlength=item["num_dst"]).float()
        active = torch.nonzero(degree > 0, as_tuple=False).view(-1)
        local_dst = local_edge[1]
        if plan.compact:
            compact_dst = torch.searchsorted(active, local_dst)
            buffer_rows = int(active.numel())
            denominator = degree.index_select(0, active).clamp_min(1.0)
        else:
            compact_dst = local_dst
            buffer_rows = int(item["num_dst"])
            denominator = degree.clamp_min(1.0)
        relations.append(
            {
                "relation_id": relation_id,
                "src": (local_edge[0] + item["src_offset"]).to(device),
                "dst": compact_dst.to(device),
                "active": active.to(device),
                "denominator": denominator.to(device),
                "buffer_rows": buffer_rows,
                "dst_offset": item["dst_offset"],
                "num_dst": item["num_dst"],
                "num_edges": int(item["edge_index"].shape[1]),
                "local_edges": int(local_edge.shape[1]),
                "process_group": (
                    None
                    if relation_process_groups is None
                    else relation_process_groups[relation_id]
                ),
            }
        )
        metadata_rows.append(
            {
                "relation": "__".join(item["edge_type"]),
                "active_destinations": int(active.numel()),
                "total_destinations": int(item["num_dst"]),
                "active_ratio": float(active.numel() / max(1, item["num_dst"])),
                **metadata,
            }
        )
    prep = sync_max(time.perf_counter() - started, device)
    return {"plan": plan, "relations": relations, "metadata": metadata_rows, "prep_sec": prep}


def make_custom_model(problem, hidden_dim: int):
    import torch

    class ExactRelationalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projectors = torch.nn.ModuleDict()
            self.embeddings = torch.nn.ParameterDict()
            for node_type in problem["node_types"]:
                input_dim = problem["input_dims"][node_type]
                if input_dim > 0:
                    self.projectors[node_type] = torch.nn.Linear(input_dim, hidden_dim)
                else:
                    parameter = torch.nn.Parameter(
                        torch.empty(problem["node_counts"][node_type], hidden_dim)
                    )
                    torch.nn.init.xavier_uniform_(parameter)
                    self.embeddings[node_type] = parameter
            relation_count = len(problem["edge_types"])
            self.weights1 = torch.nn.Parameter(torch.empty(relation_count, hidden_dim, hidden_dim))
            self.weights2 = torch.nn.Parameter(torch.empty(relation_count, hidden_dim, hidden_dim))
            self.root1 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.root2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, problem["num_classes"])
            torch.nn.init.xavier_uniform_(self.weights1)
            torch.nn.init.xavier_uniform_(self.weights2)

        def input_features(self):
            parts = []
            for node_type in problem["node_types"]:
                value = problem["features"][node_type]
                if value is None:
                    parts.append(self.embeddings[node_type])
                else:
                    parts.append(self.projectors[node_type](value))
            return torch.cat(parts, dim=0)

        def convolution(self, value, weights, root, bundle):
            output = root(value)
            for relation in bundle["relations"]:
                local = value.new_zeros((relation["buffer_rows"], value.shape[1]))
                if relation["src"].numel() > 0:
                    local.index_add_(0, relation["dst"], value.index_select(0, relation["src"]))
                else:
                    # A rank can legitimately own no edges for one relation
                    # under destination-aware partitioning.  Keep the empty
                    # local buffer in autograd so every rank executes that
                    # relation's backward collective in exactly the same run.
                    local.requires_grad_(value.requires_grad)
                reduced = _AllReduceExact.apply(local, relation["process_group"])
                reduced = reduced / relation["denominator"].unsqueeze(1)
                transformed = reduced @ weights[relation["relation_id"]]
                if bundle["plan"].compact:
                    indices = relation["active"] + relation["dst_offset"]
                else:
                    indices = torch.arange(
                        relation["num_dst"], device=value.device, dtype=torch.long
                    ) + relation["dst_offset"]
                output = output.index_add(0, indices, transformed)
            return output

        def forward(self, bundle):
            value = torch.relu(self.input_features())
            value = torch.relu(self.convolution(value, self.weights1, self.root1, bundle))
            value = self.convolution(value, self.weights2, self.root2, bundle)
            start = problem["offsets"]["author"]
            end = start + problem["node_counts"]["author"]
            return self.classifier(value[start:end])

    return ExactRelationalModel()


def make_native_model(problem, hidden_dim: int):
    import torch
    from torch_geometric.nn import RGCNConv

    class NativePyGRGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projectors = torch.nn.ModuleDict()
            self.embeddings = torch.nn.ParameterDict()
            for node_type in problem["node_types"]:
                input_dim = problem["input_dims"][node_type]
                if input_dim > 0:
                    self.projectors[node_type] = torch.nn.Linear(input_dim, hidden_dim)
                else:
                    parameter = torch.nn.Parameter(
                        torch.empty(problem["node_counts"][node_type], hidden_dim)
                    )
                    torch.nn.init.xavier_uniform_(parameter)
                    self.embeddings[node_type] = parameter
            count = len(problem["edge_types"])
            self.conv1 = RGCNConv(hidden_dim, hidden_dim, count, aggr="mean")
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, count, aggr="mean")
            self.classifier = torch.nn.Linear(hidden_dim, problem["num_classes"])

        def input_features(self):
            parts = []
            for node_type in problem["node_types"]:
                value = problem["features"][node_type]
                parts.append(
                    self.embeddings[node_type]
                    if value is None
                    else self.projectors[node_type](value)
                )
            return torch.cat(parts, dim=0)

        def forward(self, unused_bundle=None):
            value = torch.relu(self.input_features())
            value = torch.relu(self.conv1(value, problem["edge_index"], problem["edge_type"]))
            value = self.conv2(value, problem["edge_index"], problem["edge_type"])
            start = problem["offsets"]["author"]
            end = start + problem["node_counts"]["author"]
            return self.classifier(value[start:end])

    return NativePyGRGCN()


def initialize_aligned_rgcn(model, problem, hidden_dim: int, seed: int, native: bool) -> None:
    """Give the native PyG and custom exact models identical parameters.

    The two implementations are mathematically equivalent, but their default
    constructors consume random numbers in a different order.  Comparing the
    resulting accuracy therefore mixed framework semantics with initialization
    variance.  This explicit initialization keeps the model, optimizer and
    full-graph computation identical while leaving the implementations
    independent.
    """

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 7919)

    def xavier(shape):
        value = torch.empty(shape, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(value, generator=generator)
        return value

    with torch.no_grad():
        for node_type in problem["node_types"]:
            input_dim = int(problem["input_dims"][node_type])
            if input_dim > 0:
                layer = model.projectors[node_type]
                layer.weight.copy_(xavier(tuple(layer.weight.shape)))
                layer.bias.zero_()
            else:
                model.embeddings[node_type].copy_(
                    xavier(tuple(model.embeddings[node_type].shape))
                )

        relation_count = len(problem["edge_types"])
        relation1 = xavier((relation_count, hidden_dim, hidden_dim))
        relation2 = xavier((relation_count, hidden_dim, hidden_dim))
        # ``RGCNConv.root`` is multiplied as X @ root; nn.Linear stores the
        # transposed matrix and evaluates X @ weight.T.
        root1 = xavier((hidden_dim, hidden_dim))
        root2 = xavier((hidden_dim, hidden_dim))
        if native:
            model.conv1.weight.copy_(relation1)
            model.conv2.weight.copy_(relation2)
            model.conv1.root.copy_(root1)
            model.conv2.root.copy_(root2)
            if model.conv1.bias is not None:
                model.conv1.bias.zero_()
            if model.conv2.bias is not None:
                model.conv2.bias.zero_()
        else:
            model.weights1.copy_(relation1)
            model.weights2.copy_(relation2)
            model.root1.weight.copy_(root1.T)
            model.root2.weight.copy_(root2.T)
            model.root1.bias.zero_()
            model.root2.bias.zero_()

        model.classifier.weight.copy_(xavier(tuple(model.classifier.weight.shape)))
        model.classifier.bias.zero_()


def plan_candidates(config: str) -> List[PublicPlan]:
    if config in {"direct_dense", "direct_dense_aligned"}:
        return [PublicPlan(config, "contiguous", False)]
    if config == "static_dense":
        return [PublicPlan(config, "destination_hash", False)]
    if config == "superblock_dense":
        return [PublicPlan(config, "superblock_greedy", False)]
    if config == "planner_selected":
        values = []
        for block in (10_000, 25_000, 50_000, 100_000):
            values.append(PublicPlan(f"superblock_dense_b{block}", "superblock_greedy", False, block))
            values.append(PublicPlan(f"superblock_compact_b{block}", "superblock_greedy", True, block))
        values.extend(
            [
                PublicPlan("direct_dense", "contiguous", False),
                PublicPlan("static_dense", "destination_hash", False),
            ]
        )
        return values
    raise ValueError(config)


def public_worker() -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from torch.nn.parallel import DistributedDataParallel

    config_path = Path(os.environ["SIGMOD_CORE_JOB_CONFIG"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_path = Path(config["output_path"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("The public end-to-end worker requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")

    seed = int(config["seed"])
    random.seed(seed)
    # Some CUDA-compatible accelerator runtimes report devices correctly but
    # expose an empty torch.cuda.default_generators tuple.  torch.manual_seed()
    # implicitly calls torch.cuda.manual_seed_all() and then crashes before the
    # first kernel.  All random initialization in this benchmark happens on
    # CPU before model.to(device), so seeding the CPU default generator is both
    # sufficient for reproducibility and portable across those runtimes.
    torch.random.default_generator.manual_seed(seed)
    problem = prepare_dblp(device)
    native = config["config"] in {"native_pyg", "native_pyg_aligned"}
    aligned_semantics = config["config"] in {
        "native_pyg_aligned",
        "direct_dense_aligned",
    }
    if native and world_size != 1:
        raise ValueError("native_pyg is an external single-GPU full-graph baseline")
    relation_process_groups = None
    if world_size > 1 and not native:
        group_ranks = list(range(world_size))
        # Isolate every relation's exact aggregation from DDP bucket
        # collectives.  Separate relation communicators also make the code
        # robust to different autograd-ready ordering across ranks.
        relation_process_groups = [
            dist.new_group(ranks=group_ranks)
            for _ in problem["relation_edges"]
        ]
    model = (
        make_native_model(problem, int(config["hidden_dim"]))
        if native
        else make_custom_model(problem, int(config["hidden_dim"]))
    )
    if aligned_semantics:
        initialize_aligned_rgcn(
            model,
            problem,
            int(config["hidden_dim"]),
            seed,
            native,
        )
    model = model.to(device)
    with torch.no_grad():
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        parameter_sum = sum(
            float(parameter.detach().double().sum().item())
            for parameter in model.parameters()
        )
        parameter_l2 = math.sqrt(
            sum(
                float(parameter.detach().double().square().sum().item())
                for parameter in model.parameters()
            )
        )
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_ids = problem["train_ids"]
    local_train = train_ids[rank::world_size]
    labels = problem["labels"]
    profile_rows = []
    search_cost = 0.0

    def one_profile_step(bundle):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        barrier(device)
        started = time.perf_counter()
        logits = model(bundle)
        raw = functional.cross_entropy(
            logits.index_select(0, local_train), labels.index_select(0, local_train), reduction="sum"
        )
        loss = raw * world_size / max(1, int(train_ids.numel()))
        loss.backward()
        barrier(device)
        return sync_max(time.perf_counter() - started, device)

    if native:
        selected_bundle = None
        selected_plan_name = "native_pyg"
        prep_sec = 0.0
    else:
        bundles = []
        for candidate in plan_candidates(config["config"]):
            bundle = build_partition_bundle(
                problem,
                candidate,
                rank,
                world_size,
                device,
                relation_process_groups,
            )
            bundles.append(bundle)
            if config["config"] == "planner_selected":
                warm = one_profile_step(bundle)
                timed = one_profile_step(bundle)
                search_cost += bundle["prep_sec"] + warm + timed
                score = bundle["prep_sec"] + int(config["epochs"]) * timed
                profile_rows.append(
                    {
                        "plan": candidate.name,
                        "policy": candidate.policy,
                        "compact": candidate.compact,
                        "block_edges": candidate.block_edges,
                        "prep_sec": bundle["prep_sec"],
                        "profile_step_sec": timed,
                        "horizon_score_sec": score,
                    }
                )
        if profile_rows:
            best = min(range(len(profile_rows)), key=lambda index: profile_rows[index]["horizon_score_sec"])
            selected_bundle = bundles[best]
        else:
            selected_bundle = bundles[0]
        selected_plan_name = selected_bundle["plan"].name
        prep_sec = float(selected_bundle["prep_sec"])

    correctness: Dict[str, Any] = {}
    if not native and int(config["repeat"]) == 1:
        reference_bundle = (
            selected_bundle
            if selected_bundle["plan"].policy == "contiguous" and not selected_bundle["plan"].compact
            else build_partition_bundle(
                problem,
                PublicPlan("direct_dense_reference", "contiguous", False),
                rank,
                world_size,
                device,
                relation_process_groups,
            )
        )

        def output_and_gradient(bundle):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            value = model(bundle)
            raw = functional.cross_entropy(
                value.index_select(0, local_train),
                labels.index_select(0, local_train),
                reduction="sum",
            )
            (raw * world_size / max(1, int(train_ids.numel()))).backward()
            gradients = []
            for parameter in model.parameters():
                gradients.append(
                    torch.zeros_like(parameter).view(-1)
                    if parameter.grad is None
                    else parameter.grad.detach().clone().view(-1)
                )
            return value.detach(), torch.cat(gradients)

        selected_output, selected_gradient = output_and_gradient(selected_bundle)
        reference_output, reference_gradient = output_and_gradient(reference_bundle)
        output_delta = (selected_output - reference_output).abs()
        gradient_delta = (selected_gradient - reference_gradient).abs()
        correctness = {
            "reference_plan": reference_bundle["plan"].name,
            "forward_max_abs_error": float(output_delta.max().item()),
            "forward_relative_l2_error": float(
                output_delta.norm().item() / max(reference_output.norm().item(), 1e-12)
            ),
            "gradient_max_abs_error": float(gradient_delta.max().item()),
            "gradient_relative_l2_error": float(
                gradient_delta.norm().item() / max(reference_gradient.norm().item(), 1e-12)
            ),
        }
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats(device)
    epoch_rows = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        barrier(device)
        started = time.perf_counter()
        logits = model(selected_bundle)
        raw_loss = functional.cross_entropy(
            logits.index_select(0, local_train), labels.index_select(0, local_train), reduction="sum"
        )
        loss = raw_loss * world_size / max(1, int(train_ids.numel()))
        loss.backward()
        optimizer.step()
        barrier(device)
        train_sec = sync_max(time.perf_counter() - started, device)
        global_loss = sync_sum(float(raw_loss.detach().item()), device) / max(1, int(train_ids.numel()))

        model.eval()
        barrier(device)
        eval_started = time.perf_counter()
        with torch.no_grad():
            evaluated = model(selected_bundle)
        barrier(device)
        eval_sec = sync_max(time.perf_counter() - eval_started, device)
        if rank == 0:
            test_logits = evaluated.index_select(0, problem["test_ids"])
            test_labels = labels.index_select(0, problem["test_ids"])
            micro, macro = multiclass_metrics(
                test_labels, test_logits.argmax(dim=1), problem["num_classes"]
            )
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "train_sec": train_sec,
                    "eval_sec": eval_sec,
                    "train_loss": global_loss,
                    "test_micro_f1": micro,
                    "test_macro_f1": macro,
                }
            )

    peak_mb = sync_max(torch.cuda.max_memory_allocated(device) / (1024 ** 2), device)
    if rank == 0:
        steady = [row["train_sec"] for row in epoch_rows[int(config["warmup_epochs"]):]]
        if not steady:
            steady = [row["train_sec"] for row in epoch_rows]
        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        if native:
            forward_payload = 0
        else:
            forward_payload = sum(
                relation["buffer_rows"] * int(config["hidden_dim"]) * 4
                for relation in selected_bundle["relations"]
            ) * 2  # two graph-convolution layers
        logical_step_bytes = 2 * forward_payload + parameter_bytes
        ring_factor = 0.0 if world_size == 1 else 2.0 * (world_size - 1) / world_size
        wire_step_bytes_per_rank = logical_step_bytes * ring_factor
        summary = {
            "completed": True,
            "dataset": "HGB-DBLP",
            "model": "two-layer R-GCN",
            "framework": "PyG RGCNConv" if native else "exact edge-partitioned PyTorch/PyG",
            "semantic_protocol": (
                "aligned_exact_rgcn_v1" if aligned_semantics else "legacy_initialization"
            ),
            "parameter_count": parameter_count,
            "initial_parameter_sum": parameter_sum,
            "initial_parameter_l2": parameter_l2,
            "config": config["config"],
            "selected_plan": selected_plan_name,
            "world_size": world_size,
            "repeat": int(config["repeat"]),
            "seed": seed,
            "epochs": int(config["epochs"]),
            "prep_sec": prep_sec,
            "planner_search_sec": search_cost,
            "steady_epoch_sec_mean": statistics.mean(steady),
            "steady_epoch_sec_stdev": statistics.stdev(steady) if len(steady) > 1 else 0.0,
            "measured_50epoch_train_sec": sum(row["train_sec"] for row in epoch_rows),
            "total_50epoch_sec_including_search_prep": search_cost + prep_sec + sum(row["train_sec"] for row in epoch_rows),
            "throughput_edges_per_sec": problem["total_edges"] / statistics.mean(steady),
            "peak_gpu_memory_mb": peak_mb,
            "logical_communication_bytes_per_step": logical_step_bytes,
            "estimated_ring_wire_bytes_per_rank_step": wire_step_bytes_per_rank,
            "final_train_loss": epoch_rows[-1]["train_loss"],
            "final_test_micro_f1": epoch_rows[-1]["test_micro_f1"],
            "final_test_macro_f1": epoch_rows[-1]["test_macro_f1"],
            "profile_candidates": profile_rows,
            "correctness_against_direct_dense": correctness,
            "epoch_metrics": epoch_rows,
            "communication_definition": "forward+backward aggregation payload plus DDP parameter gradient; wire estimate uses ring 2(G-1)/G",
        }
        write_json_atomic(output_path, summary)
        write_csv(output_path.with_name("epoch_metrics.csv"), epoch_rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    barrier(device)
    if dist.is_initialized():
        dist.destroy_process_group()


def launch_public_configurations(
    names: Sequence[str],
    gpu_counts: Sequence[int],
    repeats: Sequence[int],
) -> None:
    output_root = CORE_ROOT / "public_dblp"
    config_root = output_root / "job_configs"
    log_root = output_root / "logs"
    config_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    for world_size in gpu_counts:
        for name in names:
            if name.startswith("native_pyg") and world_size != 1:
                continue
            for repeat in repeats:
                job_name = f"dblp__{name}__g{world_size}__r{repeat:02d}"
                result = output_root / job_name / "summary.json"
                if complete_json(result):
                    print(f"SKIP complete public job: {job_name}", flush=True)
                    continue
                payload = {
                    "config": name,
                    "world_size": world_size,
                    "repeat": repeat,
                    "seed": 20261004 + repeat - 1,
                    "epochs": PUBLIC_EPOCHS,
                    "hidden_dim": PUBLIC_HIDDEN_DIM,
                    "learning_rate": PUBLIC_LEARNING_RATE,
                    "weight_decay": PUBLIC_WEIGHT_DECAY,
                    "warmup_epochs": PUBLIC_WARMUP_EPOCHS,
                    "output_path": str(result),
                }
                config_path = config_root / f"{job_name}.json"
                write_json_atomic(config_path, payload)
                environment = os.environ.copy()
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in GPU_IDS[:world_size]),
                        "SIGMOD_CORE_WORKER": "public",
                        "SIGMOD_CORE_JOB_CONFIG": str(config_path),
                    }
                )
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={world_size}",
                    str(Path(__file__).resolve()),
                ]
                log_path = log_root / f"{job_name}.log"
                print("RUN PUBLIC:", job_name, flush=True)
                with log_path.open("w", encoding="utf-8") as stream:
                    completed = subprocess.run(
                        command,
                        cwd=str(PROJECT_ROOT),
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode != 0:
                    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
                    print(tail, flush=True)
                    raise RuntimeError(f"public job failed: {job_name}; log={log_path}")
                print(f"DONE PUBLIC: {job_name}", flush=True)


def launch_public_jobs() -> None:
    launch_public_configurations(PUBLIC_CONFIGS, PUBLIC_GPU_COUNTS, PUBLIC_REPEATS)


def launch_public_semantic_alignment() -> None:
    """Run a strict one-GPU PyG/custom comparison with identical parameters."""

    launch_public_configurations(
        ("native_pyg_aligned", "direct_dense_aligned"),
        (1,),
        (1, 2, 3),
    )


# ---------------------------------------------------------------------------
# Active-destination/destination-span evidence and industrial launch matrix.
# ---------------------------------------------------------------------------
def run_preflight() -> None:
    from settings_no_parser import INDUSTRIAL_RUNNER, MULTIDATE_METADATA_PATH

    checks = {
        "project_root": str(PROJECT_ROOT),
        "industrial_runner": str(INDUSTRIAL_RUNNER),
        "industrial_runner_exists": INDUSTRIAL_RUNNER.exists(),
        "multidate_metadata": str(MULTIDATE_METADATA_PATH),
        "multidate_metadata_exists": MULTIDATE_METADATA_PATH.exists(),
        "public_dblp_processed": str(PUBLIC_HGB_ROOT / "dblp" / "processed" / "data.pt"),
        "public_dblp_exists": (PUBLIC_HGB_ROOT / "dblp" / "processed" / "data.pt").exists(),
        "gpu_ids": list(GPU_IDS),
        "run_profile": RUN_PROFILE,
        "stages": list(STAGES_TO_RUN),
    }
    try:
        import torch

        checks["torch_version"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        checks["visible_gpu_count"] = torch.cuda.device_count()
        checks["gpu_names"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except Exception as exc:
        checks["torch_error"] = f"{type(exc).__name__}: {exc}"
    write_json_atomic(CORE_ROOT / "preflight.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    if not checks["industrial_runner_exists"] or not checks["multidate_metadata_exists"]:
        raise RuntimeError("工业图入口或多日期 metadata 缺失；先运行原来的 00/08 preflight")
    if "public_dblp_end_to_end" in STAGES_TO_RUN and not checks["public_dblp_exists"]:
        raise RuntimeError(
            f"缺少离线 DBLP: {PUBLIC_HGB_ROOT / 'dblp'}. "
            f"请把公开数据集解压到 {PUBLIC_HGB_ROOT}"
        )


def run_active_destination_stats() -> None:
    import torch
    from sigmod_exp.datasets import DatasetSpec, load_graph
    from settings_no_parser import MULTIDATE_METADATA_PATH

    output = CORE_ROOT / "active_destination" / "relation_stats.csv"
    done = output.with_suffix(".complete.json")
    if complete_json(done):
        print(f"SKIP complete active-destination stats: {output}", flush=True)
        return
    metadata = json.loads(MULTIDATE_METADATA_PATH.read_text(encoding="utf-8"))
    paths = {str(row["graph_tag"]): str(row["graph_path"]) for row in metadata}
    rows: List[Dict[str, Any]] = []
    for graph_tag in STATS_SNAPSHOTS:
        if graph_tag not in paths:
            raise KeyError(f"metadata 中没有 {graph_tag}")
        print(f"ACTIVE DESTINATION STATS: loading {graph_tag}", flush=True)
        graph = load_graph(
            DatasetSpec(name=graph_tag, kind="industrial_pickle", path=paths[graph_tag])
        )
        for edge_type in graph.edge_types:
            edge = graph[edge_type].edge_index.long().cpu().contiguous()
            if edge.numel() == 0:
                continue
            destination_store = graph[edge_type[2]]
            num_dst = int(
                destination_store.num_nodes
                if destination_store.num_nodes is not None
                else destination_store.x.shape[0]
            )
            active = int(torch.unique(edge[1]).numel())
            span_started = time.perf_counter()
            bitmask = torch.zeros((num_dst,), dtype=torch.uint8)
            edge_count = int(edge.shape[1])
            rank_edges = []
            for rank in range(4):
                start = edge_count * rank // 4
                end = edge_count * (rank + 1) // 4
                rank_edges.append(end - start)
                rank_dst = torch.unique(edge[1, start:end])
                bitmask[rank_dst] |= int(1 << rank)
            lookup = torch.tensor([value.bit_count() for value in range(16)], dtype=torch.uint8)
            spans = lookup[bitmask[bitmask > 0].long()]
            direct_mean_span = float(spans.float().mean().item()) if spans.numel() else 0.0
            direct_cross_groups = int((spans > 1).sum().item())
            direct_partition_sec = time.perf_counter() - span_started
            mean_edges = statistics.mean(rank_edges) if rank_edges else 0.0
            dense_bytes = num_dst * PUBLIC_HIDDEN_DIM * 4
            compact_bytes = active * PUBLIC_HIDDEN_DIM * 4
            rows.append(
                {
                    "graph": graph_tag,
                    "relation": "__".join(edge_type),
                    "edges": int(edge.shape[1]),
                    "destinations": num_dst,
                    "active_destinations": active,
                    "active_ratio": active / max(1, num_dst),
                    "dense_payload_bytes_per_layer": dense_bytes,
                    "compact_payload_bytes_per_layer": compact_bytes,
                    "payload_reduction_percent": 100.0 * (1.0 - compact_bytes / max(1, dense_bytes)),
                    "direct_mean_destination_span": direct_mean_span,
                    "direct_cross_gpu_destination_groups": direct_cross_groups,
                    "superblock_mean_destination_span": 1.0,
                    "superblock_cross_gpu_destination_groups": 0,
                    "direct_edge_imbalance": max(rank_edges) / max(mean_edges, 1e-12),
                    "superblock_edge_imbalance": None,
                    "direct_span_stat_sec": direct_partition_sec,
                    "superblock_span_source": "layout invariant: RD-groups are indivisible",
                }
            )
        del graph
    write_csv(output, rows)
    aggregate = {
        "completed": True,
        "graphs": sorted({row["graph"] for row in rows}),
        "relations": len(rows),
        "median_active_ratio": statistics.median(row["active_ratio"] for row in rows),
        "median_payload_reduction_percent": statistics.median(
            row["payload_reduction_percent"] for row in rows
        ),
        "mean_direct_destination_span": statistics.mean(
            row["direct_mean_destination_span"] for row in rows
        ),
        "mean_superblock_destination_span": statistics.mean(
            row["superblock_mean_destination_span"] for row in rows
        ),
        "important_note": "Payload numbers are exact structural byte counts; runtime benefit is evaluated separately.",
    }
    write_json_atomic(done, aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


DENSE_BLOCKING = {"name": "blocking_c1", "chunks": 1, "overlap": False, "operator": "dense"}
COMPACT_BLOCKING = {
    "name": "compact_blocking_c1",
    "chunks": 1,
    "overlap": False,
    "operator": "compact",
}
ASYNC_C2 = {"name": "async_chunk_2", "chunks": 2, "overlap": True, "operator": "dense"}


def launch_industrial_main() -> None:
    from common_no_parser import run_multidate_industrial_job

    for graph in ALL_SNAPSHOTS:
        for strategy in ("balanced_edge", "superblock_edge_count", "superblock_vector_balance"):
            for repeat in INDUSTRIAL_MAIN_REPEATS:
                run_multidate_industrial_job(
                    stage="core_industrial_4gpu_main",
                    graph_tag=graph,
                    world_size=4,
                    strategy=strategy,
                    block_edges=PUBLIC_BLOCK_EDGES,
                    mode=DENSE_BLOCKING,
                    repeat=repeat,
                )


def launch_industrial_scaling() -> None:
    from common_no_parser import run_multidate_industrial_job

    configurations = (
        ("balanced_edge", DENSE_BLOCKING),
        ("superblock_edge_count", DENSE_BLOCKING),
        ("superblock_edge_count", COMPACT_BLOCKING),
    )
    for graph in REPRESENTATIVE_SNAPSHOTS:
        for world_size in (1, 2, 3, 4):
            for strategy, mode in configurations:
                for repeat in INDUSTRIAL_SCALING_REPEATS:
                    run_multidate_industrial_job(
                        stage="core_industrial_scaling",
                        graph_tag=graph,
                        world_size=world_size,
                        strategy=strategy,
                        block_edges=PUBLIC_BLOCK_EDGES,
                        mode=mode,
                        repeat=repeat,
                    )


def launch_industrial_c2_check() -> None:
    from common_no_parser import run_multidate_industrial_job

    for graph in REPRESENTATIVE_SNAPSHOTS:
        for world_size in (2, 4):
            for mode in (DENSE_BLOCKING, ASYNC_C2):
                for repeat in INDUSTRIAL_C2_REPEATS:
                    run_multidate_industrial_job(
                        stage="core_industrial_c2_check",
                        graph_tag=graph,
                        world_size=world_size,
                        strategy="superblock_edge_count",
                        block_edges=PUBLIC_BLOCK_EDGES,
                        mode=mode,
                        repeat=repeat,
                    )


# ---------------------------------------------------------------------------
# Planner: charge profiling, include H=0, top-1/top-3, and horizon total cost.
# ---------------------------------------------------------------------------
def detect_memory_capacity_mb() -> float:
    # Explicit override is useful on login nodes where CUDA is intentionally
    # hidden from the analysis process.
    override = os.environ.get("SIGMOD_GPU_MEMORY_CAPACITY_MB", "").strip()
    if override:
        try:
            value = float(override)
            if value > 0:
                return value
        except ValueError:
            pass
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return min(
                float(torch.cuda.get_device_properties(index).total_memory) / (1024 ** 2)
                for index in range(torch.cuda.device_count())
            )
    except Exception:
        pass
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        values = [float(value.strip()) for value in completed.stdout.splitlines() if value.strip()]
        return min(values) if values else 0.0
    except Exception:
        pass

    # Operator-result JSON files record the physical device capacity as
    # ``gpu_memory_bytes``.  Reuse that hardware metadata instead of silently
    # writing capacity=0 when the report is collected on a CPU/login process.
    def recorded_capacity_bytes(value: Any) -> Iterable[float]:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if key in {"gpu_memory_bytes", "total_memory_bytes", "device_total_memory_bytes"}:
                    try:
                        answer = float(nested)
                        if answer >= 4 * (1024 ** 3):
                            yield answer
                    except (TypeError, ValueError):
                        pass
                yield from recorded_capacity_bytes(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from recorded_capacity_bytes(nested)

    capacities: List[float] = []
    search_roots = (CORE_ROOT, INDUSTRIAL_ROOT / "operator")
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                capacities.extend(recorded_capacity_bytes(payload))
            except Exception:
                continue
    return min(capacities) / (1024 ** 2) if capacities else 0.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        answer = float(value)
        return answer if math.isfinite(answer) else default
    except (TypeError, ValueError):
        return default


def collect_quality_filtered_industrial(
    horizon_epochs: int = 50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect industrial measurements and exclude unusable timing runs.

    A result file is not automatically a valid measurement.  In particular,
    earlier concurrent runs completed with ``peak_memory_mb=0`` and epoch times
    up to ten times slower than clean repeats.  This function records every
    inclusion/exclusion decision in ``industrial_quality_audit.csv`` and uses
    only accepted rows for both the planner and strong-scaling tables.
    """

    from collect_industrial_results import collect

    collected: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    for path in sorted(INDUSTRIAL_ROOT.rglob("train_result_rank0.json")):
        try:
            row = dict(collect(path, horizon_epochs))
            row["source_result"] = str(path)
            reasons: List[str] = []
            if _float(row.get("steady_epoch_sec_mean")) <= 0:
                reasons.append("steady_epoch_sec_mean<=0")
            if _float(row.get("peak_memory_mb")) <= 0:
                reasons.append("peak_memory_mb<=0")
            epochs_value = row.get("epochs_observed")
            if (
                epochs_value not in (None, "")
                and int(_float(epochs_value)) < INDUSTRIAL_EXPECTED_EPOCHS
            ):
                reasons.append(
                    f"epochs_observed<{INDUSTRIAL_EXPECTED_EPOCHS}"
                )
            row["_quality_reasons"] = reasons
            collected.append(row)
        except Exception as exc:
            failures.append(
                {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )

    group_fields = (
        "graph",
        "world_size",
        "strategy",
        "block_edges",
        "chunks",
        "mode",
        "overlap",
    )
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in collected:
        if row["_quality_reasons"]:
            continue
        key = tuple(str(row.get(field, "")) for field in group_fields)
        groups.setdefault(key, []).append(row)

    # A deliberately wide robust gate catches resource-contention failures
    # without trimming normal run-to-run variation.
    for values in groups.values():
        timings = sorted(_float(row["steady_epoch_sec_mean"]) for row in values)
        median = statistics.median(timings)
        if len(values) < 3 or median <= 0:
            continue
        for row in values:
            timing = _float(row["steady_epoch_sec_mean"])
            if not (0.5 * median <= timing <= 2.0 * median):
                row["_quality_reasons"].append(
                    f"runtime_outlier_vs_group_median={median:.6f}"
                )

    accepted: List[Dict[str, Any]] = []
    for row in collected:
        reasons = list(row.pop("_quality_reasons"))
        accepted_flag = not reasons
        audit.append(
            {
                "accepted": accepted_flag,
                "rejection_reason": "; ".join(reasons),
                "graph": row.get("graph"),
                "world_size": row.get("world_size"),
                "strategy": row.get("strategy"),
                "block_edges": row.get("block_edges"),
                "chunks": row.get("chunks"),
                "mode": row.get("mode"),
                "repeat": row.get("repeat"),
                "steady_epoch_sec_mean": row.get("steady_epoch_sec_mean"),
                "peak_memory_mb": row.get("peak_memory_mb"),
                "epochs_observed": row.get("epochs_observed"),
                "source_result": row.get("source_result"),
            }
        )
        if accepted_flag:
            accepted.append(row)
    write_csv(CORE_ROOT / "industrial_quality_audit.csv", audit)
    return accepted, failures, audit


def create_total_cost_tables(output_root: Path) -> None:
    plans = read_csv(output_root / "unified_plan_table.csv")
    replay = read_csv(output_root / "budgeted_replay_detail.csv")
    if not plans or not replay:
        return
    by_graph: Dict[str, List[Dict[str, Any]]] = {}
    by_graph_plan: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in plans:
        by_graph.setdefault(str(row["graph"]), []).append(row)
        by_graph_plan[(str(row["graph"]), str(row["plan_id"]))] = row

    horizons = (10, 25, 50, 100, 200)
    horizon_rows: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []
    for row in replay:
        graph = str(row["graph"])
        selected = by_graph_plan.get((graph, str(row.get("best_plan_id", ""))))
        if selected is None:
            continue
        budget_rows.append(
            {
                "graph": graph,
                "method": str(row["start_strategy"]),
                "budget": int(_float(row["requested_budget"])),
                "seed": int(_float(row["seed"])),
                "top1_regret_percent": 100.0 * _float(row["regret"]),
                "top3_regret_percent": 100.0 * _float(row.get("top3_regret")),
                "oracle_1pct_hit": int(str(row["within_1pct"]).lower() == "true"),
                "oracle_1pct_top3_hit": int(str(row.get("within_1pct_top3", "false")).lower() == "true"),
                "search_cost_sec": _float(row["planner_trial_cost_sec"]),
            }
        )
        if str(row["start_strategy"]) == "history_warm":
            joint_budget = dict(budget_rows[-1])
            joint_budget["method"] = "joint_search"
            budget_rows.append(joint_budget)
        for horizon in horizons:
            runtime = _float(selected["prep_sec_max"]) + horizon * _float(selected["steady_epoch_sec_mean"])
            horizon_rows.append(
                {
                    "graph": graph,
                    "method": str(row["start_strategy"]),
                    "budget": int(_float(row["requested_budget"])),
                    "seed": int(_float(row["seed"])),
                    "horizon_epochs": horizon,
                    "search_sec": _float(row["planner_trial_cost_sec"]),
                    "prep_sec": _float(selected["prep_sec_max"]),
                    "epoch_sec": _float(selected["steady_epoch_sec_mean"]),
                    "total_cost_sec": _float(row["planner_trial_cost_sec"]) + runtime,
                    "selected_plan_id": selected["plan_id"],
                }
            )
            if str(row["start_strategy"]) == "history_warm":
                joint_horizon = dict(horizon_rows[-1])
                joint_horizon["method"] = "joint_search"
                horizon_rows.append(joint_horizon)

    # H=0 fixed default and measured oracle.  These rows make short horizons honest.
    for graph, candidates in by_graph.items():
        def default_score(candidate):
            strategy_penalty = 0 if candidate["strategy"] == "superblock_vector_balance" else 1
            block_penalty = abs(math.log(max(1.0, _float(candidate["block_edges"])) / 25_000.0))
            chunk_penalty = abs(_float(candidate["chunks"], 1.0) - 1.0)
            gpu_penalty = abs(_float(candidate["world_size"], 4.0) - 4.0)
            overlap_penalty = 0 if str(candidate["overlap"]).lower() in {"false", "0"} else 1
            return strategy_penalty, block_penalty, chunk_penalty, gpu_penalty, overlap_penalty

        fixed = min(candidates, key=default_score)
        for horizon in horizons:
            oracle = min(
                candidates,
                key=lambda value: _float(value["prep_sec_max"])
                + horizon * _float(value["steady_epoch_sec_mean"]),
            )
            for method, candidate in (("fixed_default", fixed), ("measured_oracle", oracle)):
                horizon_rows.append(
                    {
                        "graph": graph,
                        "method": method,
                        "budget": 0,
                        "seed": 0,
                        "horizon_epochs": horizon,
                        "search_sec": 0.0,
                        "prep_sec": _float(candidate["prep_sec_max"]),
                        "epoch_sec": _float(candidate["steady_epoch_sec_mean"]),
                        "total_cost_sec": _float(candidate["prep_sec_max"])
                        + horizon * _float(candidate["steady_epoch_sec_mean"]),
                        "selected_plan_id": candidate["plan_id"],
                    }
                )
        oracle50 = min(
            candidates,
            key=lambda value: _float(value["prep_sec_max"]) + 50 * _float(value["steady_epoch_sec_mean"]),
        )
        fixed50 = _float(fixed["prep_sec_max"]) + 50 * _float(fixed["steady_epoch_sec_mean"])
        oracle50_cost = _float(oracle50["prep_sec_max"]) + 50 * _float(oracle50["steady_epoch_sec_mean"])
        budget_rows.append(
            {
                "graph": graph,
                "method": "fixed_default",
                "budget": 0,
                "seed": 0,
                "top1_regret_percent": 100.0 * (fixed50 / max(oracle50_cost, 1e-12) - 1.0),
                "top3_regret_percent": 100.0 * (fixed50 / max(oracle50_cost, 1e-12) - 1.0),
                "oracle_1pct_hit": int(fixed50 <= 1.01 * oracle50_cost),
                "oracle_1pct_top3_hit": int(fixed50 <= 1.01 * oracle50_cost),
                "search_cost_sec": 0.0,
            }
        )

        # Leakage-free historical two-stage baseline: first choose P/B/G from
        # blocking C1 history, then choose C/mode within that layout.  No
        # held-out runtime is used to make either choice.
        history = [row for other, values in by_graph.items() if other != graph for row in values]
        for horizon in horizons:
            historical_graphs = sorted({str(row["graph"]) for row in history})
            layout_scores: Dict[Tuple[str, str, str], List[float]] = {}
            for historical_graph in historical_graphs:
                rows_for_graph = [
                    row for row in history
                    if str(row["graph"]) == historical_graph
                    and int(_float(row["chunks"], 1.0)) == 1
                    and str(row["overlap"]).lower() in {"false", "0"}
                ]
                if not rows_for_graph:
                    continue
                costs = [
                    _float(row["prep_sec_max"]) + horizon * _float(row["steady_epoch_sec_mean"])
                    for row in rows_for_graph
                ]
                base = min(costs)
                for candidate, cost in zip(rows_for_graph, costs):
                    key = (
                        str(candidate["strategy"]),
                        str(candidate["block_edges"]),
                        str(candidate["world_size"]),
                    )
                    layout_scores.setdefault(key, []).append(cost / max(base, 1e-12))
            target_layouts = {
                (str(row["strategy"]), str(row["block_edges"]), str(row["world_size"]))
                for row in candidates
            }
            eligible_layouts = {
                key: values for key, values in layout_scores.items() if key in target_layouts
            }
            if eligible_layouts:
                selected_layout = min(
                    eligible_layouts,
                    key=lambda key: statistics.mean(eligible_layouts[key]),
                )
                communication_scores: Dict[Tuple[str, str, str], List[float]] = {}
                for historical_graph in historical_graphs:
                    rows_for_graph = [
                        row for row in history
                        if str(row["graph"]) == historical_graph
                        and (
                            str(row["strategy"]), str(row["block_edges"]), str(row["world_size"])
                        ) == selected_layout
                    ]
                    if not rows_for_graph:
                        continue
                    costs = [
                        _float(row["prep_sec_max"]) + horizon * _float(row["steady_epoch_sec_mean"])
                        for row in rows_for_graph
                    ]
                    base = min(costs)
                    for candidate, cost in zip(rows_for_graph, costs):
                        key = (
                            str(candidate["chunks"]),
                            str(candidate["mode"]),
                            str(candidate["overlap"]).lower(),
                        )
                        communication_scores.setdefault(key, []).append(cost / max(base, 1e-12))
                selected_communication = min(
                    communication_scores,
                    key=lambda key: statistics.mean(communication_scores[key]),
                ) if communication_scores else ("1", "blocking_c1", "false")
                exact = [
                    row for row in candidates
                    if (str(row["strategy"]), str(row["block_edges"]), str(row["world_size"])) == selected_layout
                    and (str(row["chunks"]), str(row["mode"]), str(row["overlap"]).lower()) == selected_communication
                ]
                if not exact:
                    exact = [
                        row for row in candidates
                        if (str(row["strategy"]), str(row["block_edges"]), str(row["world_size"])) == selected_layout
                        and int(_float(row["chunks"], 1.0)) == 1
                        and str(row["overlap"]).lower() in {"false", "0"}
                    ]
                two_stage = exact[0] if exact else fixed
            else:
                two_stage = fixed
            two_stage_total = _float(two_stage["prep_sec_max"]) + horizon * _float(two_stage["steady_epoch_sec_mean"])
            horizon_rows.append(
                {
                    "graph": graph,
                    "method": "two_stage_history",
                    "budget": 0,
                    "seed": 0,
                    "horizon_epochs": horizon,
                    "search_sec": 0.0,
                    "prep_sec": _float(two_stage["prep_sec_max"]),
                    "epoch_sec": _float(two_stage["steady_epoch_sec_mean"]),
                    "total_cost_sec": two_stage_total,
                    "selected_plan_id": two_stage["plan_id"],
                }
            )
            if horizon == 50:
                budget_rows.append(
                    {
                        "graph": graph,
                        "method": "two_stage_history",
                        "budget": 0,
                        "seed": 0,
                        "top1_regret_percent": 100.0 * (two_stage_total / max(oracle50_cost, 1e-12) - 1.0),
                        "top3_regret_percent": 100.0 * (two_stage_total / max(oracle50_cost, 1e-12) - 1.0),
                        "oracle_1pct_hit": int(two_stage_total <= 1.01 * oracle50_cost),
                        "oracle_1pct_top3_hit": int(two_stage_total <= 1.01 * oracle50_cost),
                        "search_cost_sec": 0.0,
                    }
                )

    write_csv(output_root / "budget_regret_extended.csv", budget_rows)
    write_csv(output_root / "horizon_total_cost.csv", horizon_rows)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for method in ("history_warm", "cold_start", "random"):
            grouped = {}
            for row in budget_rows:
                if row["method"] == method:
                    grouped.setdefault(row["budget"], []).append(row["top1_regret_percent"])
            axes[0].plot(
                sorted(grouped),
                [statistics.mean(grouped[value]) for value in sorted(grouped)],
                marker="o",
                label=method.replace("_", " "),
            )
        axes[0].set_xlabel("Trial budget H")
        axes[0].set_ylabel("Mean top-1 regret (%)")
        axes[0].grid(alpha=0.25)
        axes[0].legend(frameon=False)

        for method, budget in (
            ("fixed_default", 0),
            ("two_stage_history", 0),
            ("joint_search", 1),
            ("joint_search", 4),
            ("measured_oracle", 0),
        ):
            grouped = {}
            for row in horizon_rows:
                if row["method"] == method and row["budget"] == budget:
                    grouped.setdefault(row["horizon_epochs"], []).append(row["total_cost_sec"])
            if grouped:
                label = method.replace("_", " ") + (f" H={budget}" if budget else "")
                axes[1].plot(
                    sorted(grouped),
                    [statistics.mean(grouped[value]) for value in sorted(grouped)],
                    marker="o",
                    label=label,
                )
        axes[1].set_xlabel("Reuse horizon N (epochs)")
        axes[1].set_ylabel("Search + prep + training (s)")
        axes[1].grid(alpha=0.25)
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_root / "planner_budget_and_horizon.png", dpi=220)
        plt.close(fig)
    except Exception as exc:
        print(f"PLOT SKIPPED: {type(exc).__name__}: {exc}", flush=True)


def run_planner_total_cost() -> None:
    from budgeted_joint_planner import run_analysis

    output_root = CORE_ROOT / "planner_total_cost"
    measurement_csv = output_root / "all_industrial_measurements.csv"
    rows, failures, _ = collect_quality_filtered_industrial(50)
    if len(rows) < 4:
        raise RuntimeError("planner 至少需要四个有效工业计划")
    write_csv(measurement_csv, rows)
    write_json_atomic(output_root / "collection_failures.json", failures)
    capacity = detect_memory_capacity_mb()
    print(f"MEMORY CAPACITY: {capacity:.1f} MB", flush=True)
    run_analysis(
        input_csv=measurement_csv,
        output_dir=output_root,
        source_root=INDUSTRIAL_ROOT,
        horizon_epochs=50,
        budgets=(1, 2, 4, 8, 16),
        seeds=tuple(20261004 + value for value in range(20)),
        memory_capacity_mb=capacity,
        memory_safety_factor=1.10,
    )
    create_total_cost_tables(output_root)


# ---------------------------------------------------------------------------
# One compact collection pass for the draft tables.
# ---------------------------------------------------------------------------
def collect_public_results() -> List[Dict[str, Any]]:
    raw = []
    for path in sorted((CORE_ROOT / "public_dblp").rglob("summary.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("completed"):
                raw.append(row)
        except Exception:
            continue
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in raw:
        grouped.setdefault((str(row["config"]), int(row["world_size"])), []).append(row)
    summary = []
    for (config, world), values in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        epochs = [_float(row["steady_epoch_sec_mean"]) for row in values]
        totals = [_float(row["total_50epoch_sec_including_search_prep"]) for row in values]
        plan_totals_without_search = [
            _float(row["prep_sec"]) + _float(row["measured_50epoch_train_sec"])
            for row in values
        ]
        summary.append(
            {
                "dataset": "HGB-DBLP",
                "model": "R-GCN",
                "config": config,
                "world_size": world,
                "repeats": len(values),
                "selected_plans": "|".join(sorted({str(row["selected_plan"]) for row in values})),
                "prep_sec_mean": statistics.mean(_float(row["prep_sec"]) for row in values),
                "search_sec_mean": statistics.mean(_float(row["planner_search_sec"]) for row in values),
                "steady_epoch_sec_mean": statistics.mean(epochs),
                "steady_epoch_sec_stdev_across_runs": statistics.stdev(epochs) if len(epochs) > 1 else 0.0,
                "total_50epoch_sec_mean": statistics.mean(totals),
                "plan_50epoch_sec_without_search_mean": statistics.mean(plan_totals_without_search),
                "throughput_edges_per_sec_mean": statistics.mean(_float(row["throughput_edges_per_sec"]) for row in values),
                "peak_gpu_memory_mb_max": max(_float(row["peak_gpu_memory_mb"]) for row in values),
                "communication_bytes_per_step_mean": statistics.mean(
                    _float(row["logical_communication_bytes_per_step"]) for row in values
                ),
                "final_test_micro_f1_mean": statistics.mean(_float(row["final_test_micro_f1"]) for row in values),
                "final_test_macro_f1_mean": statistics.mean(_float(row["final_test_macro_f1"]) for row in values),
                "final_loss_mean": statistics.mean(_float(row["final_train_loss"]) for row in values),
                "forward_relative_l2_error_max": max(
                    _float(row.get("correctness_against_direct_dense", {}).get("forward_relative_l2_error"))
                    for row in values
                ),
                "gradient_relative_l2_error_max": max(
                    _float(row.get("correctness_against_direct_dense", {}).get("gradient_relative_l2_error"))
                    for row in values
                ),
            }
        )

    # Measured oracle is a post-hoc upper bound, never mislabeled as an executable system.
    for world in sorted({int(row["world_size"]) for row in summary}):
        candidates = [
            row for row in summary
            if int(row["world_size"]) == world
            and not str(row["config"]).startswith("native_pyg")
        ]
        if candidates:
            best = min(
                candidates,
                key=lambda row: _float(row["plan_50epoch_sec_without_search_mean"]),
            )
            oracle = dict(best)
            oracle["config"] = "measured_oracle"
            oracle["selected_plans"] = f"posthoc:{best['config']}:{best['selected_plans']}"
            oracle["search_sec_mean"] = 0.0
            oracle["total_50epoch_sec_mean"] = oracle["plan_50epoch_sec_without_search_mean"]
            summary.append(oracle)

    direct_by_world = {
        int(row["world_size"]): _float(row["steady_epoch_sec_mean"])
        for row in summary if row["config"] == "direct_dense"
    }
    single_gpu_by_config = {
        str(row["config"]): _float(row["steady_epoch_sec_mean"])
        for row in summary if int(row["world_size"]) == 1
    }
    for row in summary:
        baseline = direct_by_world.get(int(row["world_size"]))
        row["speedup_vs_direct_same_gpu"] = (
            baseline / max(_float(row["steady_epoch_sec_mean"]), 1e-12)
            if baseline is not None else None
        )
        one_gpu = single_gpu_by_config.get(str(row["config"]))
        row["strong_scaling_speedup"] = (
            one_gpu / max(_float(row["steady_epoch_sec_mean"]), 1e-12)
            if one_gpu is not None else None
        )
        row["strong_scaling_efficiency"] = (
            row["strong_scaling_speedup"] / int(row["world_size"])
            if row["strong_scaling_speedup"] is not None else None
        )
    write_csv(CORE_ROOT / "public_dblp_summary.csv", summary)
    return summary


def collect_public_semantic_alignment() -> Dict[str, Any]:
    """Compare aligned Native PyG and custom exact R-GCN runs by repeat."""

    output_root = CORE_ROOT / "public_dblp"
    rows: List[Dict[str, Any]] = []
    for repeat in (1, 2, 3):
        native_path = (
            output_root
            / f"dblp__native_pyg_aligned__g1__r{repeat:02d}"
            / "summary.json"
        )
        custom_path = (
            output_root
            / f"dblp__direct_dense_aligned__g1__r{repeat:02d}"
            / "summary.json"
        )
        if not (complete_json(native_path) and complete_json(custom_path)):
            continue
        native = json.loads(native_path.read_text(encoding="utf-8"))
        custom = json.loads(custom_path.read_text(encoding="utf-8"))
        native_epochs = native.get("epoch_metrics", [])
        custom_epochs = custom.get("epoch_metrics", [])
        count = min(len(native_epochs), len(custom_epochs))
        loss_diffs = [
            abs(
                _float(native_epochs[index].get("train_loss"))
                - _float(custom_epochs[index].get("train_loss"))
            )
            for index in range(count)
        ]
        micro_diffs = [
            abs(
                _float(native_epochs[index].get("test_micro_f1"))
                - _float(custom_epochs[index].get("test_micro_f1"))
            )
            for index in range(count)
        ]
        macro_diffs = [
            abs(
                _float(native_epochs[index].get("test_macro_f1"))
                - _float(custom_epochs[index].get("test_macro_f1"))
            )
            for index in range(count)
        ]
        parameter_count_equal = int(native.get("parameter_count", -1)) == int(
            custom.get("parameter_count", -2)
        )
        parameter_sum_diff = abs(
            _float(native.get("initial_parameter_sum"))
            - _float(custom.get("initial_parameter_sum"))
        )
        parameter_l2_diff = abs(
            _float(native.get("initial_parameter_l2"))
            - _float(custom.get("initial_parameter_l2"))
        )
        max_loss_diff = max(loss_diffs) if loss_diffs else math.inf
        max_micro_diff = max(micro_diffs) if micro_diffs else math.inf
        max_macro_diff = max(macro_diffs) if macro_diffs else math.inf
        rows.append(
            {
                "repeat": repeat,
                "epochs_compared": count,
                "parameter_count_equal": parameter_count_equal,
                "initial_parameter_sum_abs_diff": parameter_sum_diff,
                "initial_parameter_l2_abs_diff": parameter_l2_diff,
                "max_epoch_loss_abs_diff": max_loss_diff,
                "max_epoch_micro_f1_abs_diff": max_micro_diff,
                "max_epoch_macro_f1_abs_diff": max_macro_diff,
                "final_micro_f1_abs_diff": abs(
                    _float(native.get("final_test_micro_f1"))
                    - _float(custom.get("final_test_micro_f1"))
                ),
                "final_macro_f1_abs_diff": abs(
                    _float(native.get("final_test_macro_f1"))
                    - _float(custom.get("final_test_macro_f1"))
                ),
                "native_epoch_sec": _float(native.get("steady_epoch_sec_mean")),
                "custom_epoch_sec": _float(custom.get("steady_epoch_sec_mean")),
                "native_over_custom_epoch_time": _float(
                    native.get("steady_epoch_sec_mean")
                )
                / max(_float(custom.get("steady_epoch_sec_mean")), 1e-12),
                "semantic_match": (
                    parameter_count_equal
                    and parameter_sum_diff <= 1e-6
                    and parameter_l2_diff <= 1e-6
                    and count == PUBLIC_EPOCHS
                    and max_loss_diff <= 1e-4
                    and max_micro_diff <= 1e-4
                    and max_macro_diff <= 1e-4
                ),
            }
        )
    write_csv(CORE_ROOT / "public_dblp_semantic_alignment.csv", rows)
    summary = {
        "completed": len(rows) == 3,
        "pairs": len(rows),
        "all_semantic_match": bool(rows) and all(row["semantic_match"] for row in rows),
        "max_epoch_loss_abs_diff": max(
            (_float(row["max_epoch_loss_abs_diff"]) for row in rows),
            default=None,
        ),
        "max_epoch_micro_f1_abs_diff": max(
            (_float(row["max_epoch_micro_f1_abs_diff"]) for row in rows),
            default=None,
        ),
        "max_epoch_macro_f1_abs_diff": max(
            (_float(row["max_epoch_macro_f1_abs_diff"]) for row in rows),
            default=None,
        ),
    }
    write_json_atomic(CORE_ROOT / "public_dblp_semantic_alignment.json", summary)
    return summary


def collect_industrial_core() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw, _, _ = collect_quality_filtered_industrial(50)
    grouped: Dict[Tuple[str, int, str, str, int, int], List[Dict[str, Any]]] = {}
    for row in raw:
        key = (
            str(row["graph"]), int(row["world_size"]), str(row["strategy"]),
            str(row["mode"]), int(row["block_edges"]), int(row["chunks"]),
        )
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for key, values in grouped.items():
        epoch_values = [_float(row["steady_epoch_sec_mean"]) for row in values]
        aggregate.append(
            {
                "graph": key[0],
                "world_size": key[1],
                "strategy": key[2],
                "mode": key[3],
                "block_edges": key[4],
                "chunks": key[5],
                "repeats": len(values),
                "prep_sec_mean": statistics.mean(_float(row["prep_sec_max"]) for row in values),
                "epoch_sec_mean": statistics.mean(epoch_values),
                "epoch_sec_median": statistics.median(epoch_values),
                "epoch_sec_stdev": statistics.stdev(epoch_values) if len(epoch_values) > 1 else 0.0,
                "total_labeled_edges_mean": statistics.mean(
                    _float(row.get("total_labeled_edges")) for row in values
                ),
                "edges_per_gpu_mean": statistics.mean(
                    _float(row.get("total_labeled_edges")) / max(1, int(row["world_size"]))
                    for row in values
                ),
                "max_rank_edges_mean": statistics.mean(
                    _float(row.get("max_rank_edges")) for row in values
                ),
                "throughput_edges_per_sec": statistics.mean(
                    _float(row.get("total_labeled_edges")) for row in values
                ) / max(statistics.median(epoch_values), 1e-12),
                "peak_memory_mb_max": max(_float(row["peak_memory_mb"]) for row in values),
                "edge_imbalance_mean": statistics.mean(_float(row["edge_imbalance_to_mean"]) for row in values),
                "comm_total_sec_mean": statistics.mean(_float(row["comm_total_sec_mean"]) for row in values),
                "comm_exposed_sec_mean": statistics.mean(_float(row["comm_exposed_sec_mean"]) for row in values),
                "dense_payload_bytes_mean": statistics.mean(
                    _float(row.get("dense_payload_bytes_mean")) for row in values
                ),
                "compact_payload_bytes_mean": statistics.mean(
                    _float(row.get("compact_payload_bytes_mean")) for row in values
                ),
                "payload_reduction_ratio_mean": statistics.mean(
                    _float(row.get("payload_reduction_ratio_mean"), 1.0) for row in values
                ),
            }
        )
    for row in aggregate:
        matching_one = [
            value for value in aggregate
            if value["graph"] == row["graph"]
            and value["world_size"] == 1
            and value["strategy"] == row["strategy"]
            and value["mode"] == row["mode"]
            and value["block_edges"] == row["block_edges"]
        ]
        if matching_one:
            row["strong_scaling_speedup"] = (
                matching_one[0]["epoch_sec_median"]
                / row["epoch_sec_median"]
            )
            row["strong_scaling_efficiency"] = row["strong_scaling_speedup"] / row["world_size"]
        else:
            row["strong_scaling_speedup"] = None
            row["strong_scaling_efficiency"] = None
    aggregate.sort(key=lambda row: (row["graph"], row["world_size"], row["strategy"], row["mode"]))
    write_csv(CORE_ROOT / "industrial_core_summary.csv", aggregate)

    c2 = []
    for graph in REPRESENTATIVE_SNAPSHOTS:
        for world in (2, 4):
            blocking = [
                row for row in aggregate
                if row["graph"] == graph and row["world_size"] == world
                and row["strategy"] == "superblock_edge_count" and row["mode"] == "blocking_c1"
            ]
            async_rows = [
                row for row in aggregate
                if row["graph"] == graph and row["world_size"] == world
                and row["strategy"] == "superblock_edge_count" and row["mode"] == "async_chunk_2"
            ]
            if blocking and async_rows:
                speedup = (
                    blocking[0]["epoch_sec_median"]
                    / async_rows[0]["epoch_sec_median"]
                )
                c2.append(
                    {
                        "graph": graph,
                        "world_size": world,
                        "blocking_epoch_sec": blocking[0]["epoch_sec_median"],
                        "async_c2_epoch_sec": async_rows[0]["epoch_sec_median"],
                        "async_c2_speedup": speedup,
                        "async_c2_percent_change": 100.0 * (speedup - 1.0),
                        "blocking_repeats": blocking[0]["repeats"],
                        "async_repeats": async_rows[0]["repeats"],
                        "stable_positive_over_1pct": speedup > 1.01,
                    }
                )
    write_csv(CORE_ROOT / "industrial_c2_final_check.csv", c2)
    return aggregate, c2


def collect_report() -> None:
    public = collect_public_results()
    semantic = collect_public_semantic_alignment()
    industrial, c2 = collect_industrial_core()
    active_summary_path = CORE_ROOT / "active_destination" / "relation_stats.complete.json"
    active = (
        json.loads(active_summary_path.read_text(encoding="utf-8"))
        if active_summary_path.exists() else {}
    )
    planner_path = CORE_ROOT / "planner_total_cost" / "two_day_analysis_summary.json"
    planner = json.loads(planner_path.read_text(encoding="utf-8")) if planner_path.exists() else {}
    controlled_path = (
        CORE_ROOT
        / "planner_total_cost"
        / "controlled_memory_boundary_summary.json"
    )
    controlled = (
        json.loads(controlled_path.read_text(encoding="utf-8"))
        if controlled_path.exists()
        else {}
    )
    quality_rows = read_csv(CORE_ROOT / "industrial_quality_audit.csv")
    quality_rejected = sum(
        str(row.get("accepted", "")).lower() not in {"true", "1"}
        for row in quality_rows
    )

    public_best = [row for row in public if row["config"] == "measured_oracle"]
    compact_rows = [row for row in industrial if row["mode"] == "compact_blocking_c1"]
    lines = [
        "# SIGMOD core experiment checkpoint",
        "",
        f"- Profile: `{RUN_PROFILE}`",
        f"- Public DBLP aggregate configurations: {len(public)}",
        f"- Industrial aggregate configurations: {len(industrial)}",
        f"- C2 comparison cells: {len(c2)}",
        f"- Industrial measurements rejected by quality gates: {quality_rejected}/{len(quality_rows)}",
        "",
        "## Active-destination evidence",
        "",
        f"- Relations measured: {active.get('relations', 0)}",
        f"- Median active-destination ratio: {active.get('median_active_ratio', 'N/A')}",
        f"- Median structural payload reduction: {active.get('median_payload_reduction_percent', 'N/A')}%",
        f"- Direct vs SuperBlock mean destination span: {active.get('mean_direct_destination_span', 'N/A')} vs {active.get('mean_superblock_destination_span', 'N/A')}",
        "",
        "## Public end-to-end",
        "",
        f"- Aligned PyG/custom semantic pairs: {semantic.get('pairs', 0)}; "
        f"all matched: {semantic.get('all_semantic_match', False)}.",
    ]
    for row in sorted(public_best, key=lambda value: value["world_size"]):
        lines.append(
            f"- G={row['world_size']}: measured oracle `{row['selected_plans']}`, "
            f"epoch={row['steady_epoch_sec_mean']:.4f}s, Micro-F1={row['final_test_micro_f1_mean']:.4f}, "
            f"Macro-F1={row['final_test_macro_f1_mean']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Industrial compact candidate",
            "",
            f"- Completed aggregate compact cells: {len(compact_rows)}.",
            "- Treat compact payload reduction as a structural result until the measured runtime rows are complete.",
            "",
            "## Final C2 decision",
            "",
        ]
    )
    if c2:
        wins = sum(1 for row in c2 if row["stable_positive_over_1pct"])
        lines.append(f"- C2 is >1% faster in {wins}/{len(c2)} cells.")
        lines.append(
            "- Keep C2 in the main method only if this remains a repeat-level stable majority; otherwise move it to analysis/appendix."
        )
    else:
        lines.append("- C2 repeat matrix is not complete yet.")
    lines.extend(
        [
            "",
            "## Planner and memory",
            "",
            f"- Planner replay rows: {planner.get('planner_replay_rows', 0)}.",
            f"- Full joint P/B/C/G coverage: {planner.get('coverage', {}).get('is_full_joint_PBCG', False)}.",
            f"- Assignment-bound rows: {planner.get('assignment_bound_rows', 0)}.",
            f"- Memory-capacity sweep rows: {planner.get('memory_capacity_sweep_rows', 0)}.",
            f"- Memory summary: `{json.dumps(planner.get('memory', {}), ensure_ascii=False)}`",
            f"- Controlled memory boundary: `{json.dumps(controlled, ensure_ascii=False)}`",
            "- Counterfactual capacity sweeps and observed allocator-OOM boundaries are reported separately.",
            "",
            "## Output tables",
            "",
            "- `public_dblp_summary.csv`",
            "- `public_dblp_semantic_alignment.csv`",
            "- `industrial_core_summary.csv`",
            "- `industrial_quality_audit.csv`",
            "- `industrial_c2_final_check.csv`",
            "- `active_destination/relation_stats.csv`",
            "- `planner_total_cost/budget_regret_extended.csv`",
            "- `planner_total_cost/horizon_total_cost.csv`",
            "- `planner_total_cost/memory_capacity_sweep.csv`",
            "- `planner_total_cost/assignment_bound_evaluation.csv`",
            "- `planner_total_cost/controlled_memory_boundary_evaluation.csv`",
        ]
    )
    (CORE_ROOT / "CORE_RESULTS_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def coordinator_main() -> None:
    CORE_ROOT.mkdir(parents=True, exist_ok=True)
    stage_functions = {
        "preflight": run_preflight,
        "active_destination_stats": run_active_destination_stats,
        "public_dblp_end_to_end": launch_public_jobs,
        "industrial_4gpu_main": launch_industrial_main,
        "industrial_scaling": launch_industrial_scaling,
        "industrial_c2_check": launch_industrial_c2_check,
        "planner_total_cost": run_planner_total_cost,
        "collect_report": collect_report,
    }
    for stage in STAGES_TO_RUN:
        print(f"\n{'=' * 24} STAGE {stage} {'=' * 24}", flush=True)
        started = time.time()
        stage_functions[stage]()
        write_json_atomic(
            CORE_ROOT / "stage_status" / f"{stage}.json",
            {"completed": True, "stage": stage, "elapsed_sec": time.time() - started},
        )
    print(f"SIGMOD CORE SUITE COMPLETE: {CORE_ROOT}", flush=True)


if __name__ == "__main__":
    if os.environ.get("SIGMOD_CORE_WORKER") == "public":
        public_worker()
    else:
        coordinator_main()

