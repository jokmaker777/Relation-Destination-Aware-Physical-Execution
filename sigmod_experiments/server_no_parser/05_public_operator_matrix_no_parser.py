from __future__ import annotations

import json
import sys
import time
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from common_no_parser import run_operator_job
from settings_no_parser import (
    BASE_OUTPUT_ROOT,
    GPU_COUNTS,
    OPERATOR_BLOCK_EDGES,
    OPERATOR_CHUNKS,
    OPERATOR_POLICIES,
    PUBLIC_DATASETS,
)
from sigmod_exp.datasets import DatasetSpec, load_graph, summarize_graph


ordered_datasets = sorted(
    PUBLIC_DATASETS,
    key=lambda value: (0 if value["kind"] == "synthetic" else 1, value["name"]),
)
skipped_datasets = []

for dataset in ordered_datasets:
    # 先在单进程下载/处理，避免 torchrun 的多个 rank 同时准备数据。
    try:
        summary = summarize_graph(load_graph(DatasetSpec.from_dict(dataset)))
    except Exception as exc:
        if dataset["kind"] == "synthetic":
            raise
        record = {
            "dataset": dataset["name"],
            "kind": dataset["kind"],
            "root": dataset.get("root"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "timestamp": time.time(),
            "action": (
                "Dataset is not available locally and the server cannot download it. "
                "Prepare it on an internet-connected machine and upload the complete "
                "dataset root to this path, then rerun this script."
            ),
        }
        skipped_datasets.append(record)
        print("=" * 80)
        print(f"SKIP PUBLIC DATASET: {dataset['name']}")
        print(f"root={dataset.get('root')}")
        print(f"reason={type(exc).__name__}: {exc}")
        print("Synthetic experiments will continue; rerun after manually uploading the dataset.")
        print("=" * 80)
        continue
    print(
        f"DATASET READY: {dataset['name']}, nodes={summary['num_nodes']:,}, "
        f"edges={summary['num_edges']:,}, relations={summary['num_edge_types']}"
    )

    for gpu_count in GPU_COUNTS:
        for policy in OPERATOR_POLICIES:
            block_choices = (
                OPERATOR_BLOCK_EDGES
                if policy == "superblock_greedy"
                else (OPERATOR_BLOCK_EDGES[0],)
            )
            for block_edges in block_choices:
                for chunks in OPERATOR_CHUNKS:
                    modes = ["blocking"] if chunks == 1 else ["async"]
                    if chunks == 4:
                        modes.append("blocking")
                    for collective_mode in modes:
                        run_operator_job(
                            stage="primary_matrix",
                            dataset=dataset,
                            world_size=gpu_count,
                            policy=policy,
                            block_edges=block_edges,
                            chunks=chunks,
                            collective_mode=collective_mode,
                            hidden_dim=32,
                            warmups=2,
                            repeats=5,
                            validate=bool(dataset.get("validate_correctness", False)),
                        )

skip_report = BASE_OUTPUT_ROOT / "missing_public_datasets.json"
skip_report.parent.mkdir(parents=True, exist_ok=True)
skip_report.write_text(
    json.dumps(skipped_datasets, ensure_ascii=False, indent=2), encoding="utf-8"
)
if skipped_datasets:
    print(f"PUBLIC DATASETS SKIPPED: {len(skipped_datasets)}, report={skip_report}")
    print("SYNTHETIC MATRIX COMPLETED; PUBLIC MATRIX IS NOT YET COMPLETE")
else:
    print("PUBLIC/SYNTHETIC OPERATOR MATRIX COMPLETED")


