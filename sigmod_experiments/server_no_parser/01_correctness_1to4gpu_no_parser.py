import json

from settings_no_parser import BASE_OUTPUT_ROOT, BASE_SEED, GPU_COUNTS
from common_no_parser import run_operator_job


dataset = {
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
        dataset=dataset,
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
        / dataset["name"]
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
    print(f"{gpu_count}-GPU exactness PASSED")

print("ALL 1/2/3/4-GPU CORRECTNESS RUNS PASSED")


