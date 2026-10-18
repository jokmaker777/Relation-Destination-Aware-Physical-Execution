from __future__ import annotations

import csv

from common_no_parser import run_industrial_job
from settings_no_parser import (
    ASYNC_C2,
    ASYNC_C4,
    ASYNC_C8,
    BASE_OUTPUT_ROOT,
    BLOCKING_C1,
    BLOCKING_C4,
)


ranking_path = BASE_OUTPUT_ROOT / "single_graph_4gpu_plan_ranking.csv"
if not ranking_path.exists():
    raise RuntimeError("请先运行 06_collect_and_rank_no_parser.py")

with ranking_path.open("r", encoding="utf-8-sig", newline="") as stream:
    ranking = list(csv.DictReader(stream))
if not ranking:
    raise RuntimeError("计划排名为空")

modes = {
    value["name"]: value
    for value in (BLOCKING_C1, BLOCKING_C4, ASYNC_C2, ASYNC_C4, ASYNC_C8)
}

selected = []
for row in ranking[:3]:
    selected.append(
        (
            row["strategy"],
            int(row["block_edges"]),
            modes[row["mode"]],
        )
    )
# 无论排名如何，都保留直接边均衡 blocking 基线。
selected.append(("balanced_edge", 25_000, BLOCKING_C1))
selected = list(dict.fromkeys((a, b, c["name"]) for a, b, c in selected))

for strategy, block_edges, mode_name in selected:
    mode = modes[mode_name]
    for repeat in (2, 3):
        run_industrial_job(
            stage="selected_repeats",
            world_size=4,
            strategy=strategy,
            block_edges=block_edges,
            mode=mode,
            repeat=repeat,
        )

print("SELECTED PLAN REPEATS COMPLETED. 请再次运行 06_collect_and_rank_no_parser.py")


