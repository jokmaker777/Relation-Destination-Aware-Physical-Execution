from __future__ import annotations

import csv

from common_no_parser import run_multidate_industrial_job
from settings_no_parser import (
    ASYNC_C2,
    ASYNC_C4,
    ASYNC_C8,
    BASE_OUTPUT_ROOT,
    BLOCKING_C1,
    BLOCKING_C4,
    REPRESENTATIVE_GRAPH_TAGS,
)


ranking_path = BASE_OUTPUT_ROOT / "multidate_4gpu_plan_ranking.csv"
if not ranking_path.exists():
    raise RuntimeError("请先运行 10_collect_multidate_no_parser.py")
with ranking_path.open("r", encoding="utf-8-sig", newline="") as stream:
    ranking = list(csv.DictReader(stream))
if not ranking:
    raise RuntimeError("多日期计划排名为空")

modes = {
    value["name"]: value
    for value in (BLOCKING_C1, BLOCKING_C4, ASYNC_C2, ASYNC_C4, ASYNC_C8)
}
selected = [
    (row["strategy"], int(row["block_edges"]), row["mode"])
    for row in ranking[:3]
]
selected.append(("balanced_edge", 25_000, BLOCKING_C1["name"]))
selected = list(dict.fromkeys(selected))

total_jobs = len(REPRESENTATIVE_GRAPH_TAGS) * len(selected) * 2
job_index = 0
print(f"START MULTIDATE REPEATS: {total_jobs} jobs")
for graph_tag in REPRESENTATIVE_GRAPH_TAGS:
    for strategy, block_edges, mode_name in selected:
        for repeat in (2, 3):
            job_index += 1
            print(
                f"[{job_index}/{total_jobs}] {graph_tag} | {strategy} | "
                f"B={block_edges} | {mode_name} | repeat={repeat}"
            )
            run_multidate_industrial_job(
                stage="multidate_selected_repeats",
                graph_tag=graph_tag,
                world_size=4,
                strategy=strategy,
                block_edges=block_edges,
                mode=modes[mode_name],
                repeat=repeat,
            )

print("MULTIDATE SELECTED REPEATS COMPLETED")
print("请再次运行 10_collect_multidate_no_parser.py")


