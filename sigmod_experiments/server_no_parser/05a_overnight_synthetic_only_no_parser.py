from __future__ import annotations

import json

from common_no_parser import run_operator_job
from settings_no_parser import (
    BASE_OUTPUT_ROOT,
    BASE_SEED,
    GPU_COUNTS,
    OPERATOR_BLOCK_EDGES,
    OPERATOR_CHUNKS,
    OPERATOR_POLICIES,
    PUBLIC_DATASETS,
)


# 夜间任务先做一次短 correctness gate；任何卡数不精确就立即停止。
correctness_dataset = {
    "name": "synthetic-correctness",
    "kind": "synthetic",
    "relation_selector": "all",
    "options": {
        "num_src": 100_000,
        "num_dst": 100_000,
        "num_edges": 1_000_000,
        "num_relations": 3,
        "skew": 1.5,
        "seed": BASE_SEED,
    },
}

for gpu_count in GPU_COUNTS:
    run_operator_job(
        stage="correctness",
        dataset=correctness_dataset,
        world_size=gpu_count,
        policy="superblock_greedy",
        block_edges=25_000,
        chunks=4,
        collective_mode="async",
        hidden_dim=32,
        warmups=1,
        repeats=2,
        validate=True,
    )
    result = (
        BASE_OUTPUT_ROOT
        / "operator"
        / "correctness"
        / correctness_dataset["name"]
        / f"gpus-{gpu_count}"
        / "p-superblock_greedy__b-25000__c-4__m-async__a-mean__h-32"
        / "result.json"
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    failed = [
        row["relation"]
        for row in payload["records"]
        if not row.get("correctness", {}).get("allclose_rtol_1e-4_atol_1e-5", False)
    ]
    if failed:
        raise RuntimeError(f"{gpu_count}-GPU exactness failed: {failed}")
    print(f"CORRECTNESS GATE: {gpu_count}-GPU PASSED", flush=True)


synthetic_datasets = [
    dataset for dataset in PUBLIC_DATASETS if dataset["kind"] == "synthetic"
]
for dataset in synthetic_datasets:
    print(f"START SYNTHETIC DATASET: {dataset['name']}", flush=True)
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
                            validate=True,
                        )

print("OVERNIGHT SYNTHETIC MATRIX COMPLETED", flush=True)


