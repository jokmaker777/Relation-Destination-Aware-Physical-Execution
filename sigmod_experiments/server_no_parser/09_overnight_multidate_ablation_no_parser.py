from __future__ import annotations

import json
import traceback

from common_no_parser import load_multidate_metadata, run_multidate_industrial_job
from settings_no_parser import (
    ASYNC_C4,
    BASE_OUTPUT_ROOT,
    BLOCKING_C1,
    MULTI_DATE_GRAPH_TAGS,
)


# 六个配置把布局和通信效果拆开，同时比较两类 SuperBlock assignment。
configurations = (
    ("balanced_edge", 25_000, BLOCKING_C1),
    ("balanced_edge", 25_000, ASYNC_C4),
    ("superblock_edge_count", 25_000, BLOCKING_C1),
    ("superblock_edge_count", 25_000, ASYNC_C4),
    ("superblock_vector_balance", 25_000, BLOCKING_C1),
    ("superblock_vector_balance", 25_000, ASYNC_C4),
)

metadata = load_multidate_metadata()
missing = [tag for tag in MULTI_DATE_GRAPH_TAGS if tag not in metadata]
if missing:
    raise RuntimeError(f"多日期 metadata 缺少: {missing}")

total_jobs = len(MULTI_DATE_GRAPH_TAGS) * len(configurations)
job_index = 0
failures = []
consecutive_failures = 0
failure_path = BASE_OUTPUT_ROOT / "multidate_ablation_failures.json"
print(f"START MULTIDATE ABLATION: {total_jobs} jobs, all on 4 GPUs")

for graph_tag in MULTI_DATE_GRAPH_TAGS:
    print("=" * 100)
    print(f"START DATE {graph_tag}")
    for strategy, block_edges, mode in configurations:
        job_index += 1
        print(
            f"[{job_index}/{total_jobs}] {graph_tag} | {strategy} | "
            f"B={block_edges} | {mode['name']}"
        )
        try:
            run_multidate_industrial_job(
                stage="multidate_ablation",
                graph_tag=graph_tag,
                world_size=4,
                strategy=strategy,
                block_edges=block_edges,
                mode=mode,
                repeat=1,
            )
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            failure = {
                "job_index": job_index,
                "graph_tag": graph_tag,
                "strategy": strategy,
                "block_edges": block_edges,
                "mode": mode["name"],
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps(failures, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"JOB FAILED BUT RECORDED: {failure_path}", flush=True)
            if consecutive_failures >= 3:
                raise RuntimeError(
                    "连续 3 个任务失败，停止批处理以避免整夜重复同一系统性错误。"
                    f"失败记录：{failure_path}"
                ) from exc
    print(f"DATE {graph_tag} COMPLETED")

if failures:
    print(
        f"MULTIDATE ABLATION FINISHED WITH {len(failures)} FAILED JOBS; "
        f"see {failure_path}"
    )
else:
    if failure_path.exists():
        failure_path.unlink()
    print("ALL MULTIDATE ABLATION JOBS COMPLETED")

