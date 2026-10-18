from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    os.environ.get(
        "SIGMOD_ACTIVE_DESTINATION_RESULTS_ROOT",
        "/path/to/artifact_workspace/outputs/active_destination_overnight_4gpu",
    )
)


agg = pd.read_csv(ROOT / "overnight_aggregate.csv")
raw = pd.read_csv(ROOT / "overnight_all_completed_runs.csv")

numeric_agg = [
    "axis_value",
    "total_edges",
    "num_dst",
    "num_relations",
    "active_ratio",
    "skew",
    "hidden_dim",
    "independent_repeat_count",
    "step_sec_independent_median",
    "step_sec_independent_min",
    "step_sec_independent_max",
    "step_sec_independent_stdev",
    "speedup_vs_direct_dense_median",
    "speedup_vs_direct_dense_min",
    "speedup_vs_direct_dense_max",
    "compact_speedup_same_layout_median",
    "prep_sec_median",
    "peak_gpu_memory_mb_median",
    "peak_gpu_memory_mb_max",
    "payload_reduction_percent",
    "forward_relative_l2_error_max",
    "gradient_relative_l2_error_max",
]
for column in numeric_agg:
    agg[column] = pd.to_numeric(agg[column], errors="coerce")

numeric_raw = [
    "independent_repeat",
    "step_sec_median",
    "run_speedup_vs_direct_dense",
    "prep_sec_max",
    "peak_gpu_memory_mb",
    "forward_relative_l2_error",
    "gradient_relative_l2_error",
]
for column in numeric_raw:
    raw[column] = pd.to_numeric(raw[column], errors="coerce")


def wide_for_axis(axis: str) -> pd.DataFrame:
    selected = agg.loc[agg["axis"].astype(str).str.startswith(axis)].copy()
    index_columns = [
        "case_id",
        "total_edges",
        "num_relations",
        "active_ratio",
        "skew",
        "hidden_dim",
    ]
    wide = selected.pivot(index=index_columns, columns="method")
    wide.columns = [f"{measure}__{method}" for measure, method in wide.columns]
    return wide.reset_index()


cell_stability = (
    raw.groupby(["case_id", "method"], as_index=False)["step_sec_median"]
    .agg(["count", "mean", "std", "min", "max"])
    .reset_index()
)
cell_stability["cv"] = cell_stability["std"] / cell_stability["mean"]
cell_stability["relative_range"] = (
    cell_stability["max"] - cell_stability["min"]
) / cell_stability["mean"]

all_cases = sorted(agg["case_id"].unique())
layout_rows = []
for case_id in all_cases:
    block = agg.loc[agg["case_id"] == case_id].set_index("method")
    direct_compact = block.loc["direct_compact", "step_sec_independent_median"]
    superblock_compact = block.loc[
        "superblock_compact", "step_sec_independent_median"
    ]
    direct_dense = block.loc["direct_dense", "step_sec_independent_median"]
    superblock_dense = block.loc["superblock_dense", "step_sec_independent_median"]
    best_method = block["step_sec_independent_median"].idxmin()
    layout_rows.append(
        {
            "case_id": case_id,
            "axis": block.iloc[0]["axis"],
            "axis_value": block.iloc[0]["axis_value"],
            "direct_over_superblock_compact": direct_compact
            / superblock_compact,
            "direct_over_superblock_dense": direct_dense / superblock_dense,
            "best_method": best_method,
        }
    )
layout = pd.DataFrame(layout_rows)

active = wide_for_axis("active_ratio_fine")
active["direct_compact_speedup"] = (
    active["step_sec_independent_median__direct_dense"]
    / active["step_sec_independent_median__direct_compact"]
)
active["superblock_compact_speedup"] = (
    active["step_sec_independent_median__superblock_dense"]
    / active["step_sec_independent_median__superblock_compact"]
)
active["direct_memory_reduction_percent"] = 100 * (
    1
    - active["peak_gpu_memory_mb_median__direct_compact"]
    / active["peak_gpu_memory_mb_median__direct_dense"]
)
active["direct_prep_delta_sec"] = (
    active["prep_sec_median__direct_compact"]
    - active["prep_sec_median__direct_dense"]
)
step_saving = (
    active["step_sec_independent_median__direct_dense"]
    - active["step_sec_independent_median__direct_compact"]
)
active["break_even_steps"] = np.where(
    step_saving > 0,
    np.maximum(active["direct_prep_delta_sec"], 0) / step_saving,
    np.inf,
)
active = active.sort_values("active_ratio")

phase = wide_for_axis("phase_surface")
phase["direct_compact_speedup"] = (
    phase["step_sec_independent_median__direct_dense"]
    / phase["step_sec_independent_median__direct_compact"]
)
phase = phase.sort_values(["hidden_dim", "active_ratio"])

relations = wide_for_axis("relation_count_pressure")
relations["direct_compact_speedup"] = (
    relations["step_sec_independent_median__direct_dense"]
    / relations["step_sec_independent_median__direct_compact"]
)
relations = relations.sort_values("num_relations")

scale = wide_for_axis("scale_edges_pressure")
scale["direct_compact_speedup"] = (
    scale["step_sec_independent_median__direct_dense"]
    / scale["step_sec_independent_median__direct_compact"]
)
scale = scale.sort_values("total_edges")

skew = wide_for_axis("skew_boundary")
skew["direct_compact_speedup"] = (
    skew["step_sec_independent_median__direct_dense"]
    / skew["step_sec_independent_median__direct_compact"]
)
skew = skew.sort_values(["active_ratio", "skew"])


def records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    return json.loads(frame[columns].to_json(orient="records"))


summary = {
    "coverage": {
        "aggregate_rows": int(len(agg)),
        "raw_rows": int(len(raw)),
        "cases": int(agg["case_id"].nunique()),
        "methods": sorted(agg["method"].unique().tolist()),
        "repeats_per_cell_min": int(agg["independent_repeat_count"].min()),
        "repeats_per_cell_max": int(agg["independent_repeat_count"].max()),
        "missing_or_duplicate_cells": int(
            53 * 4 * 3 - len(raw)
        ),
    },
    "correctness": {
        "max_forward_relative_l2": float(
            agg["forward_relative_l2_error_max"].max()
        ),
        "max_gradient_relative_l2": float(
            agg["gradient_relative_l2_error_max"].max()
        ),
    },
    "stability": {
        "median_cell_cv": float(cell_stability["cv"].median()),
        "p95_cell_cv": float(cell_stability["cv"].quantile(0.95)),
        "max_cell_cv": float(cell_stability["cv"].max()),
        "median_relative_range": float(
            cell_stability["relative_range"].median()
        ),
        "p95_relative_range": float(
            cell_stability["relative_range"].quantile(0.95)
        ),
        "cells_cv_over_5_percent": int((cell_stability["cv"] > 0.05).sum()),
        "cells_cv_over_10_percent": int((cell_stability["cv"] > 0.10).sum()),
    },
    "plan_selection": {
        "best_method_counts": layout["best_method"].value_counts().to_dict(),
        "superblock_compact_wins": int(
            (layout["best_method"] == "superblock_compact").sum()
        ),
        "superblock_compact_winning_cases": layout.loc[
            layout["best_method"] == "superblock_compact", "case_id"
        ].tolist(),
        "median_superblock_vs_direct_compact_speedup": float(
            (1 / layout["direct_over_superblock_compact"]).median()
        ),
        "median_superblock_vs_direct_dense_speedup": float(
            (1 / layout["direct_over_superblock_dense"]).median()
        ),
    },
    "active_ratio_curve": records(
        active,
        [
            "active_ratio",
            "direct_compact_speedup",
            "superblock_compact_speedup",
            "payload_reduction_percent__direct_compact",
            "direct_memory_reduction_percent",
            "direct_prep_delta_sec",
            "break_even_steps",
            "speedup_vs_direct_dense_min__direct_compact",
            "speedup_vs_direct_dense_max__direct_compact",
        ],
    ),
    "phase_surface": records(
        phase,
        ["hidden_dim", "active_ratio", "direct_compact_speedup"],
    ),
    "relation_curve": records(
        relations,
        ["num_relations", "direct_compact_speedup"],
    ),
    "scale_curve": records(
        scale,
        ["total_edges", "direct_compact_speedup"],
    ),
    "skew_curve": records(
        skew,
        ["active_ratio", "skew", "direct_compact_speedup"],
    ),
}

print(json.dumps(summary, ensure_ascii=False, indent=2))

