from __future__ import annotations

from collections import Counter

import torch

from sigmod_experiments.sigmod_exp.partition import PhysicalPlan, build_local_partition


def encoded(edge_index: torch.Tensor, num_dst: int) -> Counter:
    values = (edge_index[0] * num_dst + edge_index[1]).tolist()
    return Counter(int(value) for value in values)


def test_all_policies_assign_every_edge_exactly_once() -> None:
    generator = torch.Generator().manual_seed(7)
    num_src, num_dst, num_edges = 31, 23, 2000
    edges = torch.stack(
        [
            torch.randint(num_src, (num_edges,), generator=generator),
            torch.randint(num_dst, (num_edges,), generator=generator),
        ]
    )
    for world_size in (1, 2, 3, 4):
        for policy in (
            "contiguous",
            "round_robin",
            "destination_hash",
            "destination_greedy",
            "superblock_greedy",
        ):
            plan = PhysicalPlan(policy, 127, 4, "async", "mean")
            local = [
                build_local_partition(edges, num_dst, rank, world_size, plan)[0]
                for rank in range(world_size)
            ]
            rebuilt = torch.cat(local, dim=1)
            assert rebuilt.shape[1] == edges.shape[1]
            assert encoded(rebuilt, num_dst) == encoded(edges, num_dst)


def test_destination_preserving_policies_do_not_split_targets() -> None:
    generator = torch.Generator().manual_seed(11)
    edges = torch.stack(
        [
            torch.randint(50, (4000,), generator=generator),
            torch.randint(29, (4000,), generator=generator),
        ]
    )
    for policy in ("destination_hash", "destination_greedy", "superblock_greedy"):
        plan = PhysicalPlan(policy, 300, 4, "async", "mean")
        _, metadata = build_local_partition(edges, 29, 0, 4, plan)
        assert metadata["cross_gpu_target_groups"] == 0
        assert metadata["mean_target_span"] == 1.0

