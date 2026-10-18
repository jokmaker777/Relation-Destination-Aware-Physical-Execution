from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from collect_industrial_results import collect, write_csv
from collect_operator_results import flatten
from settings_no_parser import BASE_OUTPUT_ROOT, REUSE_HORIZON_EPOCHS


industrial_root = BASE_OUTPUT_ROOT / "industrial"
industrial_output = BASE_OUTPUT_ROOT / "industrial_measurements.csv"
industrial_rows = [
    collect(path, REUSE_HORIZON_EPOCHS)
    for path in industrial_root.rglob("train_result_rank0.json")
]
industrial_rows.sort(
    key=lambda row: (
        int(row["world_size"]),
        str(row["strategy"]),
        int(row["block_edges"]),
        int(row["chunks"]),
        int(row["repeat"]),
    )
)
write_csv(industrial_output, industrial_rows)
print(f"WROTE {len(industrial_rows)} industrial rows: {industrial_output}")

operator_rows = []
for path in (BASE_OUTPUT_ROOT / "operator").rglob("result.json"):
    try:
        operator_rows.extend(flatten(path))
    except (OSError, ValueError, KeyError) as exc:
        print(f"SKIP incomplete operator result {path}: {exc}")
if operator_rows:
    operator_output = BASE_OUTPUT_ROOT / "operator_measurements.csv"
    operator_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["relation"],
            row["world_size"],
            row["policy"],
            row["block_edges"],
            row["chunks"],
        )
    )
    with operator_output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(operator_rows[0]))
        writer.writeheader()
        writer.writerows(operator_rows)
    print(f"WROTE {len(operator_rows)} operator rows: {operator_output}")

# 去除同一 seed/plan 在不同阶段可能出现的重复，再按 4-GPU 计划聚合。
unique = {}
for row in industrial_rows:
    if int(row["world_size"]) != 4:
        continue
    identity = (
        row["strategy"],
        int(row["block_edges"]),
        int(row["chunks"]),
        row["mode"],
        int(row["repeat"]),
        int(row["seed"]),
    )
    unique.setdefault(identity, row)

groups = defaultdict(list)
for row in unique.values():
    key = (
        row["strategy"],
        int(row["block_edges"]),
        int(row["chunks"]),
        row["mode"],
        bool(row["overlap"]),
    )
    groups[key].append(row)

ranking = []
for key, rows in groups.items():
    ranking.append(
        {
            "strategy": key[0],
            "block_edges": key[1],
            "chunks": key[2],
            "mode": key[3],
            "overlap": key[4],
            "repeat_count": len(rows),
            "prep_sec_mean": statistics.mean(float(row["prep_sec_max"]) for row in rows),
            "steady_epoch_sec_mean": statistics.mean(
                float(row["steady_epoch_sec_mean"]) for row in rows
            ),
            "total_horizon_sec_mean": statistics.mean(
                float(row["estimated_total_sec_at_horizon"]) for row in rows
            ),
        }
    )
ranking.sort(key=lambda row: row["total_horizon_sec_mean"])
for index, row in enumerate(ranking, 1):
    row["rank"] = index
ranking_output = BASE_OUTPUT_ROOT / "single_graph_4gpu_plan_ranking.csv"
if ranking:
    columns = ["rank"] + [key for key in ranking[0] if key != "rank"]
    with ranking_output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ranking)
    print(f"WROTE plan ranking: {ranking_output}")
    print("TOP 5 PLANS:")
    for row in ranking[:5]:
        print(json.dumps(row, ensure_ascii=False))


