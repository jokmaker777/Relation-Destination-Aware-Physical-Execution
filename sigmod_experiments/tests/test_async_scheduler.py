from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from end_to_end_prunning.models.communication_optimized_training_manager_forward_agg_overlap_no_parser import (
    CommunicationOptimizedTrainingManager,
    _DetachedChunkedForwardAllReduceWithBackward,
)


def manager_for(policy: str, order: str):
    manager = CommunicationOptimizedTrainingManager.__new__(
        CommunicationOptimizedTrainingManager
    )
    manager.config = SimpleNamespace(
        training_config=SimpleNamespace(
            aggregation_chunk_policy=policy,
            aggregation_chunk_order=order,
        )
    )
    manager.world_size = 1
    manager._forward_agg_group = None
    manager._forward_agg_chunk_plan_cache = {}
    return manager


def test_global_edge_balanced_plan_is_exact_partition():
    manager = manager_for("global_edge_balanced", "natural")
    destinations = torch.tensor([0, 0, 0, 0, 0, 1, 3, 4, 5, 5, 6, 7])
    edge_index = torch.stack([torch.arange(destinations.numel()), destinations])
    plan = manager._get_destination_chunk_plan(
        ("src", "rel", "dst"), edge_index, num_dst_nodes=8, requested_chunks=3
    )

    assert plan["ranges"][0][0] == 0
    assert plan["ranges"][-1][1] == 8
    assert sum(plan["global_edge_counts"]) == destinations.numel()
    assert sorted(plan["order"].tolist()) == list(range(destinations.numel()))
    for chunk_id, (start, end) in enumerate(plan["ranges"]):
        p0, p1 = plan["offsets"][chunk_id : chunk_id + 2]
        selected = destinations.index_select(0, plan["order"][p0:p1])
        assert bool(torch.all(selected >= start))
        assert bool(torch.all(selected < end))


def test_all_chunk_orders_are_permutations():
    base_plan = {
        "num_chunks": 4,
        "ranges": [(0, 4), (4, 8), (8, 12), (12, 16)],
        "global_edge_counts": [100, 10, 80, 20],
    }
    for order in (
        "natural",
        "largest_first",
        "smallest_first",
        "zigzag",
        "overlap_greedy",
    ):
        manager = manager_for("destination_range", order)
        result = manager._get_chunk_execution_order(base_plan)
        assert sorted(result) == [0, 1, 2, 3]


def test_backward_anchor_keeps_otherwise_unused_tensor_in_autograd_graph():
    manager = CommunicationOptimizedTrainingManager.__new__(
        CommunicationOptimizedTrainingManager
    )
    manager.world_size = 2
    manager._forward_agg_backward_anchors = []

    reduced_output = torch.arange(8, dtype=torch.float32, requires_grad=True)
    unrelated = torch.tensor(3.0, requires_grad=True)
    manager._reset_forward_agg_backward_anchors()
    manager._register_forward_agg_backward_anchor(reduced_output)
    loss = manager._attach_forward_agg_backward_anchors(unrelated.square())
    loss.backward()

    assert reduced_output.grad is not None
    assert torch.equal(reduced_output.grad, torch.zeros_like(reduced_output))
    assert unrelated.grad.item() == 6.0


def test_plan_cache_key_does_not_use_tensor_identity():
    manager = manager_for("destination_range", "natural")
    manager._forward_agg_graph_generation = 1
    destinations = torch.tensor([0, 1, 1, 2, 3, 3, 3])
    first_edges = torch.stack([torch.arange(destinations.numel()), destinations])
    second_edges = first_edges.clone()

    first = manager._get_destination_chunk_plan(
        ("src", "rel", "dst"), first_edges, num_dst_nodes=4, requested_chunks=2
    )
    second = manager._get_destination_chunk_plan(
        ("src", "rel", "dst"), second_edges, num_dst_nodes=4, requested_chunks=2
    )

    assert first is second
    assert len(manager._forward_agg_chunk_plan_cache) == 1


def test_chunked_autograd_returns_gradients_in_input_order():
    local_chunks = [
        torch.tensor([1.0, 2.0], requires_grad=True),
        torch.tensor([3.0, 4.0], requires_grad=True),
        torch.tensor([5.0, 6.0], requires_grad=True),
    ]
    reduced_chunks = [value.detach().clone() for value in local_chunks]
    outputs = _DetachedChunkedForwardAllReduceWithBackward.apply(
        None, *local_chunks, *reduced_chunks
    )
    loss = outputs[0].sum() + 2.0 * outputs[2].sum()
    loss.backward()

    assert torch.equal(local_chunks[0].grad, torch.ones(2))
    assert torch.equal(local_chunks[1].grad, torch.zeros(2))
    assert torch.equal(local_chunks[2].grad, torch.full((2,), 2.0))

