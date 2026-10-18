from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from collect_industrial_results import collect, write_csv
from settings_no_parser import BASE_OUTPUT_ROOT, REUSE_HORIZON_EPOCHS


source_roots = (
    BASE_OUTPUT_ROOT / "industrial" / "multidate_ablation",
    BASE_OUTPUT_ROOT / "industrial" / "multidate_selected_repeats",
)
paths = []
for root in source_roots:
    if root.exists():
        paths.extend(root.rglob("train_result_rank0.json"))
if not paths:
    raise RuntimeError("没有找到多日期结果；请先运行 09_overnight_multidate_ablation_no_parser.py")

raw_rows = [collect(path, REUSE_HORIZON_EPOCHS) for path in paths]
raw_output = BASE_OUTPUT_ROOT / "multidate_measurements.csv"
write_csv(raw_output, raw_rows)

# 同一图/计划先对 repeats 求均值，随后各日期等权，避免补重复的日期权重过大。
per_graph_plan = defaultdict(list)
for row in raw_rows:
    key = (
        row["graph"],
        row["strategy"],
        int(row["block_edges"]),
        int(row["chunks"]),
        row["mode"],
        bool(row["overlap"]),
    )
    per_graph_plan[key].append(row)

graph_plan_rows = []
for key, repeats in per_graph_plan.items():
    graph_plan_rows.append(
        {
            "graph": key[0],
            "strategy": key[1],
            "block_edges": key[2],
            "chunks": key[3],
            "mode": key[4],
            "overlap": key[5],
            "repeat_count": len(repeats),
            "prep_sec": statistics.mean(float(row["prep_sec_max"]) for row in repeats),
            "steady_epoch_sec": statistics.mean(
                float(row["steady_epoch_sec_mean"]) for row in repeats
            ),
            "total_horizon_sec": statistics.mean(
                float(row["estimated_total_sec_at_horizon"]) for row in repeats
            ),
        }
    )

by_plan = defaultdict(list)
for row in graph_plan_rows:
    key = (
        row["strategy"],
        row["block_edges"],
        row["chunks"],
        row["mode"],
        row["overlap"],
    )
    by_plan[key].append(row)

ranking = []
for key, graphs in by_plan.items():
    totals = [float(row["total_horizon_sec"]) for row in graphs]
    ranking.append(
        {
            "strategy": key[0],
            "block_edges": key[1],
            "chunks": key[2],
            "mode": key[3],
            "overlap": key[4],
            "graph_count": len(graphs),
            "total_horizon_sec_mean": statistics.mean(totals),
            "total_horizon_sec_median": statistics.median(totals),
            "steady_epoch_sec_mean": statistics.mean(
                float(row["steady_epoch_sec"]) for row in graphs
            ),
            "prep_sec_mean": statistics.mean(float(row["prep_sec"]) for row in graphs),
        }
    )
ranking.sort(key=lambda row: row["total_horizon_sec_mean"])
for rank, row in enumerate(ranking, 1):
    row["rank"] = rank

ranking_output = BASE_OUTPUT_ROOT / "multidate_4gpu_plan_ranking.csv"
columns = ["rank"] + [key for key in ranking[0] if key != "rank"]
with ranking_output.open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(ranking)

print(f"WROTE {len(raw_rows)} raw rows: {raw_output}")
print(f"WROTE {len(ranking)} plan rows: {ranking_output}")
print("TOP PLANS:")
for row in ranking:
    print(row)


