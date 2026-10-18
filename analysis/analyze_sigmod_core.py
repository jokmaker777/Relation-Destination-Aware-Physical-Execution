from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    os.environ.get(
        "SIGMOD_CORE_RESULTS_ROOT",
        "/path/to/artifact_workspace/outputs/sigmod_core_suite",
    )
)


def nums(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        if column in {"graph", "strategy", "mode", "config", "selected_plans", "method", "start_strategy", "plan_id", "compact_plan", "best_plan", "oracle_plan", "result_dir"}:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() >= max(1, int(0.8 * len(frame))):
            frame[column] = converted
    return frame


def out(label: str, value) -> None:
    print(f"\n### {label}")
    if isinstance(value, pd.DataFrame):
        print(value.to_string(index=False))
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, default=float))


pub = nums(pd.read_csv(ROOT / "public_dblp_summary.csv"))
pub_cols = [
    "world_size", "config", "repeats", "steady_epoch_sec_mean",
    "steady_epoch_sec_stdev_across_runs", "total_50epoch_sec_mean",
    "plan_50epoch_sec_without_search_mean", "prep_sec_mean", "search_sec_mean",
    "peak_gpu_memory_mb_max", "communication_bytes_per_step_mean",
    "final_test_micro_f1_mean", "final_test_macro_f1_mean",
    "forward_relative_l2_error_max", "gradient_relative_l2_error_max",
]
out("PUBLIC", pub[pub_cols].sort_values(["world_size", "config"]))

pub_calc = []
for world in (1, 2, 4):
    direct = pub[(pub.world_size == world) & (pub.config == "direct_dense")].iloc[0]
    for config in ("static_dense", "superblock_dense", "planner_selected", "measured_oracle"):
        row = pub[(pub.world_size == world) & (pub.config == config)]
        if row.empty:
            continue
        row = row.iloc[0]
        pub_calc.append({
            "G": world,
            "config": config,
            "steady_speedup_vs_direct": direct.steady_epoch_sec_mean / row.steady_epoch_sec_mean,
            "50epoch_speedup_vs_direct_including_search": direct.total_50epoch_sec_mean / row.total_50epoch_sec_mean,
            "search_fraction": row.search_sec_mean / row.total_50epoch_sec_mean if row.total_50epoch_sec_mean else 0,
        })
out("PUBLIC_CALC", pd.DataFrame(pub_calc))

native = pub[(pub.world_size == 1) & (pub.config == "native_pyg")].iloc[0]
direct1 = pub[(pub.world_size == 1) & (pub.config == "direct_dense")].iloc[0]
out("PUBLIC_NATIVE_DIFF", {
    "custom_vs_native_epoch_speedup": native.steady_epoch_sec_mean / direct1.steady_epoch_sec_mean,
    "micro_f1_difference_custom_minus_native_points": 100 * (direct1.final_test_micro_f1_mean - native.final_test_micro_f1_mean),
    "macro_f1_difference_custom_minus_native_points": 100 * (direct1.final_test_macro_f1_mean - native.final_test_macro_f1_mean),
})

ind = nums(pd.read_csv(ROOT / "industrial_core_summary.csv"))
main = ind[(ind.world_size == 4) & (ind["mode"] == "blocking_c1") & ind.strategy.isin(["balanced_edge", "superblock_edge_count", "superblock_vector_balance"])]
pivot = main.pivot_table(index="graph", columns="strategy", values="epoch_sec_mean", aggfunc="first")
pivot["edge_speedup"] = pivot["balanced_edge"] / pivot["superblock_edge_count"]
pivot["vector_speedup"] = pivot["balanced_edge"] / pivot["superblock_vector_balance"]
out("INDUSTRIAL_4GPU_PER_GRAPH", pivot.reset_index())
out("INDUSTRIAL_4GPU_SPEEDUP_SUMMARY", {
    col: {
        "mean": float(pivot[col].mean()),
        "median": float(pivot[col].median()),
        "min": float(pivot[col].min()),
        "max": float(pivot[col].max()),
        "ge_2x": int((pivot[col] >= 2).sum()),
        "n": int(pivot[col].notna().sum()),
    }
    for col in ("edge_speedup", "vector_speedup")
})

representative = ind[
    ind.graph.isin([20241124, 20241129, 20241218])
    & ind.strategy.isin(["balanced_edge", "superblock_edge_count"])
    & ind["mode"].isin(["blocking_c1", "compact_blocking_c1"])
]
out("INDUSTRIAL_SCALING_AGGREGATED", representative[[
    "graph", "world_size", "strategy", "mode", "repeats", "prep_sec_mean",
    "epoch_sec_mean", "epoch_sec_stdev", "strong_scaling_speedup",
    "strong_scaling_efficiency", "peak_memory_mb_max", "edge_imbalance_mean",
]].sort_values(["graph", "strategy", "mode", "world_size"]))

raw = nums(pd.read_csv(ROOT / "planner_total_cost" / "all_industrial_measurements.csv"))
raw29 = raw[
    (raw.graph == 20241129)
    & raw.strategy.isin(["balanced_edge", "superblock_edge_count"])
    & raw["mode"].isin(["blocking_c1", "compact_blocking_c1"])
    & raw.world_size.isin([1, 2, 3, 4])
]
out("INDUSTRIAL_20241129_RAW_REPEATS", raw29[[
    "world_size", "strategy", "mode", "repeat", "steady_epoch_sec_mean",
    "steady_epoch_sec_stdev", "prep_sec_max", "peak_memory_mb", "result_dir",
]].sort_values(["world_size", "strategy", "mode", "repeat"]))

c2 = nums(pd.read_csv(ROOT / "industrial_c2_final_check.csv"))
out("C2", c2)
out("C2_SUMMARY", {
    "ge_1pct_cells": int(c2.stable_positive_over_1pct.astype(str).str.lower().eq("true").sum()),
    "cells": len(c2),
    "mean_percent_change": float(c2.async_c2_percent_change.mean()),
    "median_percent_change": float(c2.async_c2_percent_change.median()),
    "min_percent_change": float(c2.async_c2_percent_change.min()),
    "max_percent_change": float(c2.async_c2_percent_change.max()),
})

c2_raw = raw[
    raw.graph.isin([20241124, 20241129, 20241218])
    & raw.world_size.isin([2, 4])
    & (raw.strategy == "superblock_edge_count")
    & raw["mode"].isin(["blocking_c1", "async_chunk_2"])
]
c2_repeat = c2_raw.pivot_table(
    index=["graph", "world_size", "repeat"], columns="mode",
    values="steady_epoch_sec_mean", aggfunc="first"
).dropna().reset_index()
c2_repeat["async_change_percent"] = 100 * (
    c2_repeat["blocking_c1"] / c2_repeat["async_chunk_2"] - 1
)
out("C2_REPEAT_LEVEL", c2_repeat)

replay = nums(pd.read_csv(ROOT / "planner_total_cost" / "budgeted_replay_summary.csv"))
replay_view = replay[[
    "start_strategy", "budget", "mean_regret", "median_regret", "p90_regret",
    "max_regret", "within_1pct_rate", "oracle_selection_rate",
    "mean_planner_trial_cost_sec",
]].copy()
for col in ["mean_regret", "median_regret", "p90_regret", "max_regret", "within_1pct_rate", "oracle_selection_rate"]:
    replay_view[col] *= 100
out("PLANNER_REPLAY_PERCENT", replay_view.sort_values(["budget", "start_strategy"]))

budget_ext = nums(pd.read_csv(ROOT / "planner_total_cost" / "budget_regret_extended.csv"))
horizon = nums(pd.read_csv(ROOT / "planner_total_cost" / "horizon_total_cost.csv"))
out("PLANNER_EXTENDED_COVERAGE", {
    "budget_extended_methods": sorted(budget_ext.method.unique().tolist()),
    "budget_extended_rows": len(budget_ext),
    "horizon_methods": sorted(horizon.method.unique().tolist()),
    "horizon_rows": len(horizon),
})

hagg = horizon.groupby(["method", "horizon_epochs"], as_index=False).agg(
    mean_total=("total_cost_sec", "mean"), median_total=("total_cost_sec", "median"),
)
oracle = hagg[hagg.method == "measured_oracle"].set_index("horizon_epochs").mean_total
hagg["over_oracle_percent"] = hagg.apply(lambda r: 100 * (r.mean_total / oracle.loc[r.horizon_epochs] - 1), axis=1)
out("HORIZON_EXISTING", hagg.sort_values(["horizon_epochs", "method"]))

mem = nums(pd.read_csv(ROOT / "planner_total_cost" / "memory_model_evaluation.csv"))
out("MEMORY_RANGE", {
    "rows": len(mem),
    "actual_min_mb": float(mem.actual_peak_memory_mb.min()),
    "actual_max_mb": float(mem.actual_peak_memory_mb.max()),
    "zero_actual_rows": int((mem.actual_peak_memory_mb <= 0).sum()),
    "predicted_pruned": int((mem.predicted_feasible.astype(str).str.lower() == "false").sum()),
    "actual_infeasible": int((mem.actual_feasible.astype(str).str.lower() == "false").sum()),
    "median_ape_percent": float(100 * mem.absolute_percentage_error.median()),
    "p95_ape_percent": float(100 * mem.absolute_percentage_error.quantile(.95)),
    "fold_r2_median": float(mem.fold_r2.median()),
    "fold_r2_min": float(mem.fold_r2.min()),
    "fold_r2_max": float(mem.fold_r2.max()),
})

active = nums(pd.read_csv(ROOT / "active_destination" / "relation_stats.csv"))
out("ACTIVE", {
    "rows": len(active),
    "graphs": int(active.graph.nunique()) if "graph" in active else None,
    "median_active_ratio": float(active.active_ratio.median()),
    "q25_active_ratio": float(active.active_ratio.quantile(.25)),
    "q75_active_ratio": float(active.active_ratio.quantile(.75)),
    "median_payload_reduction_percent": float(active.payload_reduction_percent.median()),
    "min_payload_reduction_percent": float(active.payload_reduction_percent.min()),
    "max_payload_reduction_percent": float(active.payload_reduction_percent.max()),
})

summaries = []
for path in (ROOT / "public_dblp").glob("dblp__*/summary.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries.append(payload)
out("PUBLIC_COMPLETENESS", {
    "summary_json_count": len(summaries),
    "completed": sum(bool(x.get("completed")) for x in summaries),
    "epochs_min": min(int(x.get("epochs", 0)) for x in summaries),
    "epochs_max": max(int(x.get("epochs", 0)) for x in summaries),
    "max_forward_rel_l2": max(float((x.get("correctness_against_direct_dense") or {}).get("forward_relative_l2_error", 0)) for x in summaries),
    "max_gradient_rel_l2": max(float((x.get("correctness_against_direct_dense") or {}).get("gradient_relative_l2_error", 0)) for x in summaries),
})

