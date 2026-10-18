from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.nn import SAGEConv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import run_exactness_matrix_1to4gpu_no_parser as exactness
from end_to_end_prunning.models.communication_optimized_training_manager_forward_agg_overlap_no_parser import (
    CommunicationOptimizedTrainingManager,
)


def test_world_one_matches_reference_for_all_physical_orders():
    device = torch.device("cpu")
    edges = exactness.global_edges(device)
    generator = torch.Generator(device=device).manual_seed(exactness.SEED + 1)
    features = torch.randn(
        (exactness.NUM_SRC, exactness.HIDDEN_DIM), generator=generator, device=device
    )
    probe = torch.randn(
        (exactness.NUM_DST, exactness.HIDDEN_DIM), generator=generator, device=device
    )
    for config in exactness.CONFIGS:
        exactness.validate_ranges(int(config["chunks"]))
        expected_output, expected_gradient = exactness.reference(
            edges, features, probe, str(config["aggregation"])
        )
        output, gradient, _ = exactness.distributed_operator(
            edges,
            features,
            probe,
            int(config["chunks"]),
            bool(config["async"]),
            str(config["order"]),
            str(config["aggregation"]),
            rank=0,
            world=1,
            group=None,
        )
        assert torch.allclose(output, expected_output, rtol=1e-4, atol=1e-5), (
            config["name"], float((output - expected_output).abs().max())
        )
        assert torch.allclose(gradient, expected_gradient, rtol=1e-4, atol=1e-5), (
            config["name"], float((gradient - expected_gradient).abs().max())
        )


def test_active_destination_compact_matches_native_sage_world_one():
    torch.manual_seed(7)
    manager = CommunicationOptimizedTrainingManager.__new__(
        CommunicationOptimizedTrainingManager
    )
    manager.rank = 0
    manager.world_size = 1
    manager._forward_agg_group = None
    manager._forward_agg_groups = {}
    manager._forward_agg_compact_plan_cache = {}
    manager._forward_agg_chunk_plan_cache = {}
    manager._forward_agg_global_degree_cache = {}
    manager._forward_agg_graph_generation = 1
    manager._forward_agg_backward_anchors = []
    manager.config = SimpleNamespace(
        training_config=SimpleNamespace(
            enable_forward_aggregation=True,
            forward_aggregation_operator="compact",
            enable_forward_agg_timing=False,
        )
    )

    x_src = torch.randn(11, 8, requires_grad=True)
    x_dst = torch.randn(13, 8, requires_grad=True)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 6, 7, 8, 9], [1, 1, 3, 5, 5, 5, 9, 9, 11]],
        dtype=torch.long,
    )
    conv = SAGEConv((8, 8), 6, aggr="mean")
    expected = conv((x_src, x_dst), edge_index)
    actual = manager._distributed_sage_relation_forward(
        conv=conv,
        x_src=x_src,
        x_dst=x_dst,
        edge_index=edge_index,
        edge_type=("inventory", "to", "virtual"),
        layer_idx=0,
    )
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

