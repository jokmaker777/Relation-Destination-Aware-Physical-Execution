from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

import run_sigmod_core_suite_no_parser as core


def test_aligned_native_and_custom_rgcn_forward_match():
    edge_index = torch.tensor(
        [[0, 1, 2, 0], [1, 1, 2, 2]], dtype=torch.long
    )
    problem = {
        "node_types": ["author"],
        "node_counts": {"author": 3},
        "offsets": {"author": 0},
        "input_dims": {"author": 4},
        "features": {"author": torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10.0},
        "edge_types": [("author", "rel", "author")],
        "edge_index": edge_index,
        "edge_type": torch.zeros(edge_index.shape[1], dtype=torch.long),
        "num_classes": 2,
    }
    custom = core.make_custom_model(problem, 4)
    native = core.make_native_model(problem, 4)
    core.initialize_aligned_rgcn(custom, problem, 4, 20261004, native=False)
    core.initialize_aligned_rgcn(native, problem, 4, 20261004, native=True)

    degree = torch.bincount(edge_index[1], minlength=3).float().clamp_min(1.0)
    bundle = {
        "plan": core.PublicPlan("direct_dense_aligned", "contiguous", False),
        "relations": [
            {
                "relation_id": 0,
                "src": edge_index[0],
                "dst": edge_index[1],
                "active": torch.tensor([1, 2], dtype=torch.long),
                "denominator": degree,
                "buffer_rows": 3,
                "dst_offset": 0,
                "num_dst": 3,
                "num_edges": edge_index.shape[1],
                "local_edges": edge_index.shape[1],
                "process_group": None,
            }
        ],
    }
    custom_output = custom(bundle)
    native_output = native()
    assert torch.allclose(custom_output, native_output, atol=1e-6, rtol=1e-6)

    custom_output.square().sum().backward()
    native_output.square().sum().backward()
    assert torch.allclose(custom.weights1.grad, native.conv1.weight.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(custom.weights2.grad, native.conv2.weight.grad, atol=1e-6, rtol=1e-6)


def _write_industrial_fixture(
    root: Path,
    repeat: int,
    epoch_sec: float,
    peak_memory_mb: float,
) -> None:
    run_dir = root / f"fixture_R{repeat:02d}"
    run_dir.mkdir(parents=True)
    payload = {
        "graph_tag": "20241129",
        "world_size": 1,
        "strategy": "balanced_edge",
        "target_super_block_size": 25000,
        "target_num_blocks": 249,
        "aggregation_num_chunks": 1,
        "mode": "blocking_c1",
        "enable_forward_agg_overlap": False,
        "repeat_id": repeat,
        "run_seed": 20261004 + repeat,
        "total_labeled_edges": 1000,
        "peak_memory_mb": peak_memory_mb,
    }
    (run_dir / "train_result_rank0.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with (run_dir / "runtime_rank0.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["event", "elapsed_sec"])
        writer.writeheader()
        for _ in range(10):
            writer.writerow({"event": "train_epoch", "elapsed_sec": epoch_sec})


def test_industrial_quality_filter_rejects_zero_memory_and_runtime_outlier(
    tmp_path, monkeypatch
):
    industrial_root = tmp_path / "industrial"
    core_root = tmp_path / "core"
    for repeat, epoch_sec, peak_memory_mb in (
        (1, 23.0, 32000.0),
        (2, 24.0, 32100.0),
        (3, 25.0, 32200.0),
        (4, 24.0, 0.0),
        (5, 200.0, 32300.0),
    ):
        _write_industrial_fixture(
            industrial_root, repeat, epoch_sec, peak_memory_mb
        )
    monkeypatch.setattr(core, "INDUSTRIAL_ROOT", industrial_root)
    monkeypatch.setattr(core, "CORE_ROOT", core_root)

    accepted, failures, audit = core.collect_quality_filtered_industrial(50)

    assert not failures
    assert {int(row["repeat"]) for row in accepted} == {1, 2, 3}
    rejected = {
        int(row["repeat"]): str(row["rejection_reason"])
        for row in audit
        if not row["accepted"]
    }
    assert "peak_memory_mb<=0" in rejected[4]
    assert "runtime_outlier_vs_group_median" in rejected[5]
    assert (core_root / "industrial_quality_audit.csv").exists()

