import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(
    os.environ.get(
        "SIGMOD_RESULTS_ROOT",
        "/path/to/artifact_workspace/outputs",
    )
)


def rows(rel):
    with (ROOT / rel).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return math.nan
    return float(value)


def pct(x):
    return 100.0 * x


def finite(values):
    return [x for x in values if math.isfinite(x)]


out = {}

# Unified industrial plan coverage and direct comparisons.
plans = rows("two_day_initial_results/unified_plan_table.csv")
out["coverage"] = {
    "rows": len(plans),
    "plans_per_graph": dict(sorted(Counter(r["graph"] for r in plans).items())),
    "world_size_counts": dict(sorted(Counter(r["world_size"] for r in plans).items())),
    "strategy_counts": dict(sorted(Counter(r["strategy"] for r in plans).items())),
    "mode_counts": dict(sorted(Counter(r["mode"] for r in plans).items())),
    "block_edges": sorted(set(r["block_edges"] for r in plans)),
}

by_graph = defaultdict(list)
for r in plans:
    by_graph[r["graph"]].append(r)

oracle_rows = []
for graph, rr in sorted(by_graph.items()):
    best = min(rr, key=lambda r: f(r, "actual_total_sec"))
    oracle_rows.append({
        "graph": graph,
        "world_size": int(best["world_size"]),
        "strategy": best["strategy"],
        "chunks": int(best["chunks"]),
        "mode": best["mode"],
        "actual_total_sec": f(best, "actual_total_sec"),
    })
out["oracles"] = oracle_rows
out["oracle_counts"] = {
    "world_size": dict(Counter(str(x["world_size"]) for x in oracle_rows)),
    "strategy": dict(Counter(x["strategy"] for x in oracle_rows)),
    "mode": dict(Counter(x["mode"] for x in oracle_rows)),
}

# Best async versus best blocking for each graph/world-size/strategy group.
async_pairs = []
groups = defaultdict(list)
for r in plans:
    groups[(r["graph"], r["world_size"], r["strategy"], r["block_edges"])].append(r)
for key, rr in sorted(groups.items()):
    blocking = [r for r in rr if r["overlap"].lower() == "false"]
    asyncs = [r for r in rr if r["overlap"].lower() == "true"]
    if blocking and asyncs:
        b = min(blocking, key=lambda r: f(r, "actual_total_sec"))
        a = min(asyncs, key=lambda r: f(r, "actual_total_sec"))
        async_pairs.append({
            "graph": key[0], "G": int(key[1]), "strategy": key[2],
            "blocking": f(b, "actual_total_sec"), "async": f(a, "actual_total_sec"),
            "speedup": f(b, "actual_total_sec") / f(a, "actual_total_sec"),
            "async_mode": a["mode"],
        })
out["best_async_vs_blocking"] = {
    "pairs": len(async_pairs),
    "mean_speedup": statistics.mean(x["speedup"] for x in async_pairs),
    "median_speedup": statistics.median(x["speedup"] for x in async_pairs),
    "min_speedup": min(x["speedup"] for x in async_pairs),
    "max_speedup": max(x["speedup"] for x in async_pairs),
    "async_wins": sum(x["speedup"] > 1 for x in async_pairs),
    "by_G": {
        str(g): {
            "n": len(v),
            "mean_speedup": statistics.mean(x["speedup"] for x in v),
            "async_wins": sum(x["speedup"] > 1 for x in v),
        }
        for g, v in sorted(((g, [x for x in async_pairs if x["G"] == g]) for g in sorted(set(x["G"] for x in async_pairs))))
    },
}

# Comparable 4-GPU C1 blocking layouts across dates.
layout = defaultdict(dict)
for r in plans:
    if r["world_size"] == "4" and r["chunks"] == "1" and r["overlap"].lower() == "false":
        layout[r["graph"]][r["strategy"]] = f(r, "actual_total_sec")
layout_rows = []
for graph, d in sorted(layout.items()):
    if "balanced_edge" in d:
        for strat in ("superblock_edge_count", "superblock_vector_balance"):
            if strat in d:
                layout_rows.append({
                    "graph": graph,
                    "strategy": strat,
                    "speedup_vs_balanced": d["balanced_edge"] / d[strat],
                })
out["layout_4gpu_blocking"] = {
    "comparisons": len(layout_rows),
    "by_strategy": {
        strat: {
            "n": len(v),
            "mean_speedup": statistics.mean(x["speedup_vs_balanced"] for x in v),
            "median_speedup": statistics.median(x["speedup_vs_balanced"] for x in v),
            "min_speedup": min(x["speedup_vs_balanced"] for x in v),
            "max_speedup": max(x["speedup_vs_balanced"] for x in v),
            "wins": sum(x["speedup_vs_balanced"] > 1 for x in v),
        }
        for strat, v in ((s, [x for x in layout_rows if x["strategy"] == s]) for s in sorted(set(x["strategy"] for x in layout_rows)))
    },
}

# Scaling where a graph/strategy/mode exists at 1, 2, and 4 GPUs.
scale_groups = defaultdict(dict)
for r in plans:
    k = (r["graph"], r["strategy"], r["chunks"], r["overlap"])
    scale_groups[k][int(r["world_size"])] = f(r, "actual_total_sec")
scale_rows = []
for k, d in scale_groups.items():
    if all(g in d for g in (1, 2, 4)):
        scale_rows.append({"key": k, "speedup_1to2": d[1]/d[2], "speedup_1to4": d[1]/d[4], "eff4": d[1]/d[4]/4})
out["scaling_comparable"] = {
    "n": len(scale_rows),
    "mean_1to2": statistics.mean(x["speedup_1to2"] for x in scale_rows) if scale_rows else None,
    "mean_1to4": statistics.mean(x["speedup_1to4"] for x in scale_rows) if scale_rows else None,
    "mean_eff4": statistics.mean(x["eff4"] for x in scale_rows) if scale_rows else None,
}

# Budgeted replay.
replay = rows("two_day_initial_results/budgeted_replay_summary.csv")
out["planner"] = [
    {
        "strategy": r["start_strategy"], "H": int(r["budget"]),
        "mean_regret_pct": pct(f(r, "mean_regret")),
        "p90_regret_pct": pct(f(r, "p90_regret")),
        "max_regret_pct": pct(f(r, "max_regret")),
        "within_1pct_pct": pct(f(r, "within_1pct_rate")),
        "oracle_selection_pct": pct(f(r, "oracle_selection_rate")),
        "trial_cost_sec": f(r, "mean_planner_trial_cost_sec"),
    }
    for r in replay
]

# Fallback and simple policy baselines.
fallback = rows("two_day_initial_results/blocking_fallback_evaluation.csv")
def policy_regret(r, policy):
    b, a = f(r, "blocking_cost_sec"), f(r, "async_cost_sec")
    oracle = min(b, a)
    chosen = b if policy == "blocking" else a
    return (chosen - oracle) / oracle
out["fallback"] = {
    "n": len(fallback),
    "model_accuracy": statistics.mean(r["selection_correct"].lower() == "true" for r in fallback),
    "model_mean_regret": statistics.mean(f(r, "regret") for r in fallback),
    "always_blocking_mean_regret": statistics.mean(policy_regret(r, "blocking") for r in fallback),
    "always_async_mean_regret": statistics.mean(policy_regret(r, "async") for r in fallback),
    "blocking_oracle_rate": statistics.mean(r["oracle_mode"] == "blocking" for r in fallback),
    "by_G": {},
}
for g in sorted(set(r["world_size"] for r in fallback), key=int):
    rr = [r for r in fallback if r["world_size"] == g]
    out["fallback"]["by_G"][g] = {
        "n": len(rr),
        "accuracy": statistics.mean(r["selection_correct"].lower() == "true" for r in rr),
        "mean_regret": statistics.mean(f(r, "regret") for r in rr),
        "oracle_async_rate": statistics.mean(r["oracle_mode"] == "async" for r in rr),
    }

# Memory model by GPU count.
mem = rows("two_day_initial_results/memory_model_evaluation.csv")
out["memory"] = {}
for g in sorted(set(r["plan"].split("|G", 1)[1].split("|", 1)[0] for r in mem), key=int):
    rr = [r for r in mem if f"|G{g}|" in r["plan"]]
    ape = [f(r, "absolute_percentage_error") for r in rr]
    out["memory"][g] = {
        "n": len(rr), "median_ape_pct": pct(statistics.median(ape)),
        "mean_ape_pct": pct(statistics.mean(ape)), "max_ape_pct": pct(max(ape)),
        "actual_peak_min_mb": min(f(r, "actual_peak_memory_mb") for r in rr),
        "actual_peak_max_mb": max(f(r, "actual_peak_memory_mb") for r in rr),
    }

# Exactness.
exact = rows("exactness_matrix_1to4gpu/exactness_1to4gpu_summary.csv")
for agg in sorted(set(r["aggregation"] for r in exact)):
    rr = [r for r in exact if r["aggregation"] == agg]
    out.setdefault("exactness", {})[agg] = {
        "n": len(rr), "passed": sum(r["passed"].lower() == "true" for r in rr),
        "edge_coverage_passed": sum(r["edge_coverage_pass"].lower() == "true" for r in rr),
        "range_coverage_passed": sum(r["destination_range_coverage_pass"].lower() == "true" for r in rr),
        "max_forward_abs": max(f(r, "forward_max_abs_error") for r in rr),
        "max_gradient_abs": max(f(r, "gradient_max_abs_error") for r in rr),
        "max_forward_rel": max(f(r, "forward_max_rel_error") for r in rr),
        "max_gradient_rel": max(f(r, "gradient_max_rel_error") for r in rr),
    }

# Pipeline recurrence and fallback.
pipe = rows("pipeline_recurrence_2to4gpu/pipeline_recurrence_2to4gpu_summary.csv")
out["pipeline"] = {
    "mean_ape_pct": pct(statistics.mean(f(r, "prediction_absolute_percentage_error") for r in pipe)),
    "median_ape_pct": pct(statistics.median(f(r, "prediction_absolute_percentage_error") for r in pipe)),
    "fallback_accuracy": statistics.mean(r["fallback_correct"].lower() == "true" for r in pipe),
    "fallback_mean_regret_pct": pct(statistics.mean(f(r, "fallback_regret") for r in pipe)),
    "by_chunks": {},
}
for c in sorted(set(r["chunks"] for r in pipe), key=int):
    rr = [r for r in pipe if r["chunks"] == c]
    out["pipeline"]["by_chunks"][c] = {
        "mean_ape_pct": pct(statistics.mean(f(r, "prediction_absolute_percentage_error") for r in rr)),
        "async_speedup_vs_blocking_c1": statistics.mean(
            (f(r, "measured_async_pipeline_sec") + f(r, "measured_net_gain_vs_blocking_c1_sec")) / f(r, "measured_async_pipeline_sec")
            for r in rr
        ),
        "mean_terminal_tail_us": 1e6 * statistics.mean(f(r, "terminal_tail_sec") for r in rr),
        "mean_signed_overhead_us": 1e6 * statistics.mean(f(r, "signed_launch_event_index_overhead_sec") for r in rr),
    }

# Alpha-beta fits.
ab = rows("collective_alpha_beta_1to4gpu/alpha_beta_1to4gpu_summary.csv")
out["alpha_beta"] = [
    {
        "G": int(r["world_size"]), "alpha_us": f(r, "alpha_us"),
        "bandwidth_gbps": f(r, "effective_bandwidth_gbps"), "r2": f(r, "fit_r2"),
    }
    for r in ab
]

# Confirmed industrial async methods.
confirm = rows("async_scheduler_search/confirm_4gpu_ranking.csv")
out["async_confirm_4gpu"] = [
    {
        "method": r["method"], "complete": r["complete_for_selection"],
        "graphs": int(r["graph_count"]), "speedup_vs_blocking": f(r, "speedup_vs_blocking"),
        "steady_epoch_sec": f(r, "steady_epoch_sec_mean"),
        "total_horizon_sec": f(r, "total_horizon_sec_mean"),
        "comm_exposed_sec": f(r, "comm_exposed_sec_mean"),
        "overlap_ratio": f(r, "overlap_ratio_mean"),
    }
    for r in confirm
]

# Public graph operator microbenchmarks.
pub = rows("overlap_other_graphs_summary.csv")
out["public_operator"] = {}
for ds in sorted(set(r["dataset"] for r in pub)):
    rr = [r for r in pub if r["dataset"] == ds]
    out["public_operator"][ds] = {
        "n": len(rr),
        "correct": sum(r["correct"].lower() == "true" for r in rr),
        "mean_async_c4_speedup_vs_blocking_c4": statistics.mean(f(r, "async_c4_speedup_vs_blocking_c4") for r in rr),
        "median_async_c4_speedup_vs_blocking_c4": statistics.median(f(r, "async_c4_speedup_vs_blocking_c4") for r in rr),
        "mean_async_c4_speedup_vs_blocking_c1": statistics.mean(f(r, "async_c4_speedup_vs_blocking_c1") for r in rr),
    }

print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=True))

