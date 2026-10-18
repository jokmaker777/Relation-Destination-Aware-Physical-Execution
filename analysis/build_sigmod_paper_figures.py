from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


WORKSPACE = Path(os.environ.get("SIGMOD_ARTIFACT_ROOT", "/path/to/artifact_workspace"))
SOURCE = Path(
    os.environ.get("SIGMOD_RESULTS_ROOT", str(WORKSPACE / "outputs"))
)
PACKAGE = Path(
    os.environ.get("SIGMOD_FIGURE_OUTPUT_ROOT", str(WORKSPACE / "paper_artifacts"))
)
FIGURES = PACKAGE / "figures"
TABLES = PACKAGE / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

CORE = SOURCE / "sigmod_core_suite"
PLANNER = CORE / "planner_total_cost"

# Okabe-Ito colorblind-safe palette.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#666666",
    "lightgray": "#BBBBBB",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

manifest = []


def save_figure(fig, stem):
    fig.tight_layout()
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png.name, pdf.name


def sem(series):
    x = pd.Series(series).dropna().astype(float)
    return x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0


def add_manifest(figure_id, files, title, sources, rule, rows, repetitions,
                 supports, does_not_support, status="READY"):
    def source_label(source):
        try:
            return str(source.relative_to(SOURCE))
        except ValueError:
            try:
                return "paper_package/" + str(source.relative_to(PACKAGE))
            except ValueError:
                return str(source)
    manifest.append({
        "figure_id": figure_id,
        "files": ";".join(files),
        "title": title,
        "source_files": ";".join(source_label(s) for s in sources),
        "filter_rule": rule,
        "rows_used": int(rows),
        "repetitions": repetitions,
        "supports": supports,
        "does_not_support": does_not_support,
        "status": status,
    })


industrial_path = CORE / "industrial_core_summary.csv"
industrial = pd.read_csv(industrial_path)

# Figure 1: 12-snapshot layout benefit.
q = industrial[
    (industrial["world_size"] == 4)
    & (industrial["mode"] == "blocking_c1")
    & (industrial["block_edges"] == 25000)
    & industrial["strategy"].isin([
        "balanced_edge", "superblock_edge_count", "superblock_vector_balance"
    ])
].copy()
rows = []
for graph, group in q.groupby("graph"):
    direct = group[group["strategy"] == "balanced_edge"].iloc[0]
    sb = group[group["strategy"] != "balanced_edge"].sort_values("epoch_sec_mean").iloc[0]
    rows.append({
        "graph": str(int(graph)),
        "direct_epoch_sec": direct["epoch_sec_mean"],
        "best_superblock_epoch_sec": sb["epoch_sec_mean"],
        "speedup_vs_direct": direct["epoch_sec_mean"] / sb["epoch_sec_mean"],
        "best_superblock_strategy": sb["strategy"],
        "direct_repeats": int(direct["repeats"]),
        "best_superblock_repeats": int(sb["repeats"]),
    })
snapshot = pd.DataFrame(rows).sort_values("graph")
snapshot.to_csv(TABLES / "table_01_industrial_12snapshot_speedup.csv", index=False)

fig, ax = plt.subplots(figsize=(7.0, 3.0))
bar_colors = snapshot["best_superblock_strategy"].map({
    "superblock_edge_count": COLORS["blue"],
    "superblock_vector_balance": COLORS["orange"],
})
x = np.arange(len(snapshot))
ax.bar(x, snapshot["speedup_vs_direct"], color=bar_colors, width=0.72,
       edgecolor="white", linewidth=0.5)
ax.axhline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1)
ax.set_ylabel("Speedup over Direct (×)")
ax.set_xlabel("Industrial snapshot")
ax.set_xticks(x)
ax.set_xticklabels([f"{g[4:6]}/{g[6:]}" for g in snapshot["graph"]], rotation=35, ha="right")
ax.set_ylim(0, max(3.0, snapshot["speedup_vs_direct"].max() * 1.13))
ax.text(0.99, 0.97,
        f"Mean {snapshot['speedup_vs_direct'].mean():.2f}×; range "
        f"{snapshot['speedup_vs_direct'].min():.2f}–{snapshot['speedup_vs_direct'].max():.2f}×",
        transform=ax.transAxes, ha="right", va="top")
handles = [
    mpl.patches.Patch(color=COLORS["blue"], label="Best: edge-count assignment"),
    mpl.patches.Patch(color=COLORS["orange"], label="Best: vector-aware assignment"),
]
ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2)
files = save_figure(fig, "fig01_industrial_12snapshot_speedup")
add_manifest(
    "F01", files, "SuperBlock speedup across 12 industrial snapshots",
    [industrial_path],
    "G=4, blocking C=1, B=25K; compare Direct with the faster of the two SuperBlock assignments",
    len(snapshot), "Mixed: most cells n=1; no error bars are shown",
    "SuperBlock is consistently faster than Direct across the 12 measured snapshots",
    "Statistical variance across snapshots or repeated-run confidence intervals",
)


# Figure 2a: repeated industrial scaling on the one fully repeated snapshot.
q = industrial[
    (industrial["graph"].astype(str) == "20241129")
    & (industrial["block_edges"] == 25000)
    & (industrial["mode"] == "blocking_c1")
    & industrial["strategy"].isin(["balanced_edge", "superblock_edge_count"])
].copy()
q["label"] = q["strategy"].map({
    "balanced_edge": "Direct",
    "superblock_edge_count": "SuperBlock",
})
scaling_rows = []
for label, group in q.groupby("label"):
    group = group.sort_values("world_size")
    base = group[group["world_size"] == 1].iloc[0]
    base_se = base["epoch_sec_stdev"] / math.sqrt(base["repeats"])
    for _, r in group.iterrows():
        speedup = base["epoch_sec_mean"] / r["epoch_sec_mean"]
        if r["world_size"] == 1:
            speedup_se = 0.0
        else:
            run_se = r["epoch_sec_stdev"] / math.sqrt(r["repeats"])
            speedup_se = speedup * math.sqrt(
                (base_se / base["epoch_sec_mean"]) ** 2
                + (run_se / r["epoch_sec_mean"]) ** 2
            )
        scaling_rows.append({
            "system": label,
            "world_size": int(r["world_size"]),
            "repeats": int(r["repeats"]),
            "epoch_sec_mean": r["epoch_sec_mean"],
            "epoch_sec_stdev": r["epoch_sec_stdev"],
            "speedup": speedup,
            "speedup_se_delta_method": speedup_se,
        })
industrial_scaling = pd.DataFrame(scaling_rows)
industrial_scaling.to_csv(TABLES / "table_02a_industrial_20241129_scaling.csv", index=False)

fig, ax = plt.subplots(figsize=(3.55, 3.0))
for label, color, marker in [
    ("Direct", COLORS["vermillion"], "o"),
    ("SuperBlock", COLORS["blue"], "s"),
]:
    d = industrial_scaling[industrial_scaling["system"] == label]
    ax.errorbar(d["world_size"], d["speedup"], yerr=d["speedup_se_delta_method"],
                label=label, color=color, marker=marker, linewidth=1.6,
                capsize=2.5, markersize=4.5)
ax.plot([1, 2, 3, 4], [1, 2, 3, 4], color=COLORS["lightgray"], linestyle="--",
        linewidth=1, label="Ideal")
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("Number of GPUs")
ax.set_ylabel("Strong-scaling speedup (×)")
ax.set_ylim(0, 4.2)
ax.legend(frameon=False, loc="upper left")
files = save_figure(fig, "fig02a_industrial_20241129_scaling")
add_manifest(
    "F02a", files, "Industrial strong scaling on snapshot 20241129",
    [industrial_path],
    "Snapshot 20241129, B=25K, blocking C=1, Direct and SuperBlock, G=1/2/3/4",
    len(industrial_scaling), "At least 3 independent runs per point; error bars are delta-method SE",
    "The industrial workload is latency/communication dominated and does not positively strong-scale",
    "Positive industrial strong scaling or extrapolation beyond four GPUs",
)


# Figure 2b: controlled positive scaling at 16M edges.
large_scaling_path = SOURCE / "active_destination_large_edge_scaling_1to4gpu" / "large_edge_strong_scaling.csv"
large = pd.read_csv(large_scaling_path)
controlled = large[
    (large["total_edges"] == 16_000_000)
    & large["active_ratio"].isin([0.1, 0.5, 0.9])
    & large["method"].isin(["direct_dense", "direct_compact"])
].copy()
controlled.to_csv(TABLES / "table_02b_controlled_16m_scaling.csv", index=False)

fig, ax = plt.subplots(figsize=(7.0, 3.15))
ratio_colors = {0.1: COLORS["blue"], 0.5: COLORS["orange"], 0.9: COLORS["green"]}
for ratio in [0.1, 0.5, 0.9]:
    for method, ls, marker, suffix in [
        ("direct_dense", "--", "o", "dense"),
        ("direct_compact", "-", "s", "compact"),
    ]:
        d = controlled[(controlled["active_ratio"] == ratio) & (controlled["method"] == method)]
        ax.plot(d["world_size"], d["strong_scaling_speedup"],
                label=f"{int(ratio*100)}% active, {suffix}", color=ratio_colors[ratio],
                linestyle=ls, marker=marker, linewidth=1.5, markersize=4.2)
ax.plot([1, 2, 3, 4], [1, 2, 3, 4], color=COLORS["lightgray"], linestyle=":",
        linewidth=1.2, label="Ideal")
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("Number of GPUs")
ax.set_ylabel("Strong-scaling speedup (×)")
ax.set_ylim(0, 4.15)
ax.legend(frameon=False, ncol=2, loc="upper left")
files = save_figure(fig, "fig02b_controlled_16m_scaling")
add_manifest(
    "F02b", files, "Controlled 16M-edge strong scaling",
    [large_scaling_path],
    "16M edges, active ratio in {10%, 50%, 90%}, dense and active-destination compact execution, G=1/2/3/4",
    len(controlled), "3 independent runs per point; plotted statistic is median",
    "Active-destination compaction enables positive scaling in the large controlled workload",
    "The same positive scaling on the smaller industrial workload",
)


# Figure 3: planner budget-regret and 1%-hit rate.
budget_path = PLANNER / "budget_regret_extended.csv"
budget = pd.read_csv(budget_path)
curves = budget[budget["method"].isin(["history_warm", "cold_start", "random"])].copy()
per_graph = curves.groupby(["graph", "method", "budget"], as_index=False).agg(
    top1_regret_percent=("top1_regret_percent", "mean"),
    oracle_1pct_hit=("oracle_1pct_hit", "mean"),
)
budget_summary = per_graph.groupby(["method", "budget"], as_index=False).agg(
    top1_regret_percent_mean=("top1_regret_percent", "mean"),
    top1_regret_percent_se=("top1_regret_percent", sem),
    oracle_1pct_hit_mean=("oracle_1pct_hit", "mean"),
    oracle_1pct_hit_se=("oracle_1pct_hit", sem),
)
zero = budget[budget["method"].isin(["fixed_default", "two_stage_history"])].copy()
zero_summary = zero.groupby(["method", "budget"], as_index=False).agg(
    top1_regret_percent_mean=("top1_regret_percent", "mean"),
    top1_regret_percent_se=("top1_regret_percent", sem),
    oracle_1pct_hit_mean=("oracle_1pct_hit", "mean"),
    oracle_1pct_hit_se=("oracle_1pct_hit", sem),
)
budget_plot_table = pd.concat([budget_summary, zero_summary], ignore_index=True)
budget_plot_table.to_csv(TABLES / "table_03_planner_budget_regret.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))
method_style = {
    "history_warm": ("History / joint", COLORS["blue"], "o"),
    "cold_start": ("Cold start", COLORS["orange"], "s"),
    "random": ("Random", COLORS["green"], "^"),
}
for method, (label, color, marker) in method_style.items():
    d = budget_summary[budget_summary["method"] == method].sort_values("budget")
    axes[0].errorbar(d["budget"], d["top1_regret_percent_mean"],
                     yerr=d["top1_regret_percent_se"], label=label, color=color,
                     marker=marker, linewidth=1.5, capsize=2)
    axes[1].errorbar(d["budget"], 100*d["oracle_1pct_hit_mean"],
                     yerr=100*d["oracle_1pct_hit_se"], label=label, color=color,
                     marker=marker, linewidth=1.5, capsize=2)
for method, label, color, marker in [
    ("fixed_default", "Fixed default", COLORS["gray"], "D"),
    ("two_stage_history", "Two-stage", COLORS["purple"], "P"),
]:
    d = zero_summary[zero_summary["method"] == method].iloc[0]
    axes[0].scatter([0], [d["top1_regret_percent_mean"]], color=color, marker=marker, s=30, label=label)
    axes[1].scatter([0], [100*d["oracle_1pct_hit_mean"]], color=color, marker=marker, s=30, label=label)
axes[0].set_ylabel("Top-1 regret (%)")
axes[1].set_ylabel("Within 1% of oracle (%)")
for ax in axes:
    ax.set_xlabel("Profiling budget H")
    ax.set_xticks([0, 1, 2, 4, 8, 16])
axes[0].set_yscale("log")
axes[1].set_ylim(0, 105)
axes[0].legend(frameon=False, ncol=2, loc="upper right")
files = save_figure(fig, "fig03_planner_budget_regret")
add_manifest(
    "F03", files, "Planner budget versus plan-selection quality",
    [budget_path],
    "Aggregate each seed within each snapshot, then report mean and SE across 12 snapshots",
    len(budget_plot_table), "20 replay seeds per snapshot for search methods; H=0 baselines have one value per snapshot",
    "Historical warm starts reach low regret with small profiling budgets",
    "That profiling is cost-effective for every training horizon; see the horizon-cost figure",
)


# Figure 4: horizon-total cost including search.
horizon_path = PLANNER / "horizon_total_cost.csv"
horizon = pd.read_csv(horizon_path)
selected = horizon[
    ((horizon["method"].isin(["history_warm", "cold_start", "random"])) & (horizon["budget"] == 4))
    | ((horizon["method"].isin(["fixed_default", "two_stage_history", "measured_oracle"])) & (horizon["budget"] == 0))
].copy()
per_graph_h = selected.groupby(["graph", "method", "budget", "horizon_epochs"], as_index=False).agg(
    total_cost_sec=("total_cost_sec", "mean"),
)
horizon_summary = per_graph_h.groupby(["method", "budget", "horizon_epochs"], as_index=False).agg(
    total_cost_sec_mean=("total_cost_sec", "mean"),
    total_cost_sec_se=("total_cost_sec", sem),
)
horizon_summary["total_cost_min_mean"] = horizon_summary["total_cost_sec_mean"] / 60.0
horizon_summary["total_cost_min_se"] = horizon_summary["total_cost_sec_se"] / 60.0
horizon_summary.to_csv(TABLES / "table_04_horizon_total_cost.csv", index=False)

fig, ax = plt.subplots(figsize=(7.0, 3.25))
hstyles = {
    "history_warm": ("History / joint, H=4", COLORS["blue"], "o", "-"),
    "cold_start": ("Cold start, H=4", COLORS["orange"], "s", "-"),
    "random": ("Random, H=4", COLORS["green"], "^", "-"),
    "fixed_default": ("Fixed default", COLORS["gray"], "D", "--"),
    "two_stage_history": ("Two-stage history", COLORS["purple"], "P", "--"),
    "measured_oracle": ("Measured oracle", COLORS["vermillion"], "X", ":"),
}
for method, (label, color, marker, ls) in hstyles.items():
    d = horizon_summary[horizon_summary["method"] == method].sort_values("horizon_epochs")
    ax.errorbar(d["horizon_epochs"], d["total_cost_min_mean"],
                yerr=d["total_cost_min_se"], label=label, color=color,
                marker=marker, linestyle=ls, linewidth=1.35, capsize=2, markersize=4)
ax.set_xscale("log")
ax.set_xticks([10, 25, 50, 100, 200])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xlabel("Training horizon N (epochs)")
ax.set_ylabel("Search + preparation + training (min)")
ax.legend(frameon=False, ncol=2, loc="upper left")
files = save_figure(fig, "fig04_horizon_total_cost")
add_manifest(
    "F04", files, "Planner total cost across training horizons",
    [horizon_path],
    "Use H=4 for history/cold/random and H=0 for fixed/two-stage/oracle; include measured search, preparation, and epoch cost",
    len(horizon_summary), "Means and SE across 12 snapshots after averaging replay seeds within snapshot",
    "Warm-start search is not automatically beneficial at short horizons; profiling cost changes the preferred method",
    "Plan-regret alone establishes end-to-end benefit",
)


# Figure 5: memory prediction accuracy and strictly reclassified OOM feasibility.
memory_path = PLANNER / "memory_model_evaluation.csv"
capacity_path = PLANNER / "memory_capacity_sweep.csv"
oom_audit_path = PACKAGE / "oom_audit" / "reclassified_memory_boundary.csv"
memory = pd.read_csv(memory_path)
capacity = pd.read_csv(capacity_path)
oom_audit = pd.read_csv(oom_audit_path)
memory.to_csv(TABLES / "table_05a_memory_prediction_rows.csv", index=False)
capacity.to_csv(TABLES / "table_05b_memory_capacity_sweep.csv", index=False)
oom_audit.to_csv(TABLES / "table_05c_reclassified_oom_boundary.csv", index=False)

median_ape = 100 * memory["absolute_percentage_error"].median()
p95_ape = 100 * memory["absolute_percentage_error"].quantile(0.95)
mae = memory["absolute_error_mb"].mean()
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
axes[0].scatter(memory["actual_peak_memory_mb"] / 1024,
                memory["predicted_peak_memory_mb"] / 1024,
                s=13, alpha=0.58, color=COLORS["blue"], edgecolors="none")
lim_lo = min(memory["actual_peak_memory_mb"].min(), memory["predicted_peak_memory_mb"].min()) / 1024
lim_hi = max(memory["actual_peak_memory_mb"].max(), memory["predicted_peak_memory_mb"].max()) / 1024
axes[0].plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=COLORS["gray"], linestyle="--", linewidth=1)
axes[0].set_xlabel("Measured peak memory (GiB)")
axes[0].set_ylabel("Predicted peak memory (GiB)")
axes[0].text(0.04, 0.96, f"Median APE: {median_ape:.2f}%\nP95 APE: {p95_ape:.2f}%\nMAE: {mae/1024:.2f} GiB",
             transform=axes[0].transAxes, va="top")

oom_audit["outcome"] = np.select(
    [oom_audit["fatal_oom"], oom_audit["completed_after_oom"]],
    ["Fatal OOM", "Recovered OOM"],
    default="No OOM",
)
oom_audit["strategy_label"] = oom_audit["strategy"].map({
    "balanced_edge": "Direct",
    "superblock_edge_count": "SuperBlock",
})
y_map = {"Direct": 1, "SuperBlock": 0}
outcome_style = {
    "No OOM": (COLORS["green"], "o"),
    "Recovered OOM": (COLORS["orange"], "^"),
    "Fatal OOM": (COLORS["vermillion"], "X"),
}
for outcome, (color, marker) in outcome_style.items():
    d = oom_audit[oom_audit["outcome"] == outcome]
    axes[1].scatter(100*d["memory_fraction"], d["strategy_label"].map(y_map),
                    color=color, marker=marker, s=46, label=outcome, zorder=3)
# A black square denotes that the conservative model predicted the run feasible.
d = oom_audit[oom_audit["predicted_feasible"]]
axes[1].scatter(100*d["memory_fraction"], d["strategy_label"].map(y_map),
                facecolors="none", edgecolors="black", marker="s", s=88,
                linewidths=0.9, label="Predicted feasible", zorder=4)
axes[1].set_xlabel("Effective memory limit (% of GPU)")
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["SuperBlock", "Direct"])
axes[1].set_ylim(-0.55, 1.55)
axes[1].set_xticks([35, 55, 75, 95])
axes[1].legend(frameon=False, loc="upper left", ncol=2)
unsafe_den = int(oom_audit["predicted_feasible"].sum())
safe_den = int(oom_audit["no_oom_feasible"].sum())
axes[1].text(0.98, 0.04,
             f"Strict audit: 0/{unsafe_den} unsafe accepts; "
             f"{int(oom_audit['false_reject_strict'].sum())}/{safe_den} false rejects",
             transform=axes[1].transAxes, fontsize=6.8, color=COLORS["gray"],
             ha="right")
files = save_figure(fig, "fig05_memory_prediction_and_capacity")
add_manifest(
    "F05", files, "Memory prediction and capacity feasibility sweep",
    [memory_path, oom_audit_path],
    "Use all 146 quality-filtered plans; controlled feasibility uses strict log-level OOM reclassification, including recovered OOMs",
    len(memory) + len(oom_audit), "146 plans and eight controlled memory-limit cells",
    "Memory predictions are reasonably accurate; strict controlled audit has zero unsafe accepts among predicted-feasible cells",
    "That all exit-code-zero runs are OOM-free, or that conservative pruning has no false rejects",
    status="READY_AFTER_STRICT_OOM_AUDIT",
)


# Figure 6: DBLP end-to-end runtime and semantics-aligned F1.
dblp_path = CORE / "public_dblp_summary.csv"
dblp = pd.read_csv(dblp_path)
runtime_configs = {
    "direct_dense": "Direct",
    "static_dense": "Static block",
    "superblock_dense": "SuperBlock",
    "planner_selected": "Planner-selected",
    "native_pyg": "Native PyG",
}
dr = dblp[dblp["config"].isin(runtime_configs)].copy()
dr["label"] = dr["config"].map(runtime_configs)
aligned = dblp[(dblp["world_size"] == 1) & dblp["config"].isin(["native_pyg_aligned", "direct_dense_aligned"])].copy()
dr.to_csv(TABLES / "table_06a_dblp_runtime.csv", index=False)
aligned.to_csv(TABLES / "table_06b_dblp_aligned_f1.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.25, 0.9]})
styles = {
    "Direct": (COLORS["vermillion"], "o"),
    "Static block": (COLORS["orange"], "s"),
    "SuperBlock": (COLORS["blue"], "^"),
    "Planner-selected": (COLORS["green"], "D"),
    "Native PyG": (COLORS["purple"], "P"),
}
for label, (color, marker) in styles.items():
    d = dr[dr["label"] == label].sort_values("world_size")
    axes[0].errorbar(d["world_size"], 1000*d["steady_epoch_sec_mean"],
                     yerr=1000*d["steady_epoch_sec_stdev_across_runs"],
                     label=label, color=color, marker=marker, linewidth=1.4,
                     capsize=2, markersize=4)
axes[0].set_xticks([1, 2, 4])
axes[0].set_xlabel("Number of GPUs")
axes[0].set_ylabel("Steady epoch time (ms)")
axes[0].legend(frameon=False, ncol=2, loc="upper left")

names = ["Native PyG\n(aligned)", "Direct\n(aligned)"]
aligned_order = aligned.set_index("config").loc[["native_pyg_aligned", "direct_dense_aligned"]]
xx = np.arange(2)
w = 0.33
axes[1].bar(xx-w/2, aligned_order["final_test_micro_f1_mean"], width=w, color=COLORS["blue"], label="Micro-F1")
axes[1].bar(xx+w/2, aligned_order["final_test_macro_f1_mean"], width=w, color=COLORS["orange"], label="Macro-F1")
axes[1].set_xticks(xx)
axes[1].set_xticklabels(names)
axes[1].set_ylabel("Test F1")
axes[1].set_ylim(0.84, 0.90)
axes[1].legend(frameon=False, loc="upper right")
files = save_figure(fig, "fig06_dblp_end_to_end")
add_manifest(
    "F06", files, "HGB-DBLP end-to-end runtime and aligned accuracy",
    [dblp_path],
    "Runtime: selected exact full-graph configurations at G=1/2/4; accuracy: semantics-aligned Native PyG and Direct at G=1",
    len(dr) + len(aligned), "3 independent runs per reported point",
    "The end-to-end pipeline preserves aligned predictive quality; DBLP is too small to positively strong-scale",
    "A multi-GPU Native PyG comparison or positive scaling on DBLP",
)


# Figure 7: incremental industrial ablation on 50-epoch plan cost, excluding search.
unified_path = PLANNER / "unified_plan_table.csv"
detail_path = PLANNER / "budgeted_replay_detail.csv"
plans = pd.read_csv(unified_path)
detail = pd.read_csv(detail_path)
abl_rows = []
for graph in sorted(plans["graph"].unique()):
    pg = plans[plans["graph"] == graph]
    def pick(strategy):
        z = pg[(pg["world_size"] == 4) & (pg["strategy"] == strategy)
               & (pg["block_edges"] == 25000) & (pg["mode"] == "blocking_c1")]
        return float(z.iloc[0]["actual_total_sec"])
    direct_cost = pick("balanced_edge")
    org_cost = pick("superblock_edge_count")
    vec_cost = pick("superblock_vector_balance")
    dg = detail[(detail["graph"] == graph) & (detail["start_strategy"] == "history_warm")
                & (detail["requested_budget"] == 4)]
    planner_cost = float(dg["best_cost_sec"].mean())
    oracle_cost = float(dg["oracle_cost_sec"].mean())
    for stage, cost in [
        ("Direct", direct_cost),
        ("Organization", org_cost),
        ("Vector-aware", vec_cost),
        ("Planner (H=4)", planner_cost),
        ("Oracle", oracle_cost),
    ]:
        abl_rows.append({"graph": str(int(graph)), "stage": stage, "cost_sec": cost,
                         "speedup_vs_direct": direct_cost / cost})
ablation = pd.DataFrame(abl_rows)
ablation_summary = ablation.groupby("stage", as_index=False).agg(
    speedup_mean=("speedup_vs_direct", "mean"),
    speedup_se=("speedup_vs_direct", sem),
    snapshots=("graph", "nunique"),
)
stage_order = ["Direct", "Organization", "Vector-aware", "Planner (H=4)", "Oracle"]
ablation_summary["stage"] = pd.Categorical(ablation_summary["stage"], stage_order, ordered=True)
ablation_summary = ablation_summary.sort_values("stage")
ablation.to_csv(TABLES / "table_07a_industrial_ablation_per_snapshot.csv", index=False)
ablation_summary.to_csv(TABLES / "table_07b_industrial_ablation_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(5.8, 3.0))
xx = np.arange(len(ablation_summary))
ax.bar(xx, ablation_summary["speedup_mean"], yerr=ablation_summary["speedup_se"],
       color=[COLORS["gray"], COLORS["orange"], COLORS["blue"], COLORS["green"], COLORS["vermillion"]],
       capsize=3, width=0.68, edgecolor="white")
ax.axhline(1, color=COLORS["gray"], linestyle="--", linewidth=1)
ax.set_xticks(xx)
ax.set_xticklabels(["Direct", "Organization", "Vector-aware", "Planner\n(H=4)", "Oracle"])
ax.set_ylabel("50-epoch speedup over Direct (×)")
ax.set_xlabel("Incremental system configuration")
files = save_figure(fig, "fig07_industrial_ablation")
add_manifest(
    "F07", files, "Industrial incremental ablation",
    [unified_path, detail_path],
    "Across 12 snapshots: fixed G=4/B=25K/C=1 for Direct and layout stages; history-warm H=4 best measured plan; measured oracle",
    len(ablation), "Mean and SE across 12 snapshots",
    "Most gain comes from organization/layout; the planner approaches the measured oracle",
    "Planner search overhead is excluded here; use the horizon-total-cost figure for end-to-end cost",
)


# Figure 8: active-destination phase boundary.
active_path = SOURCE / "active_destination_overnight_4gpu" / "overnight_aggregate.csv"
active = pd.read_csv(active_path)
phase = active[(active["axis"] == "active_ratio_fine")
               & active["method"].isin(["direct_compact", "superblock_compact"])].copy()
phase.to_csv(TABLES / "table_08_active_destination_phase_boundary.csv", index=False)

fig, ax = plt.subplots(figsize=(6.1, 3.0))
for method, label, color, marker in [
    ("direct_compact", "Direct + active compaction", COLORS["blue"], "o"),
    ("superblock_compact", "SuperBlock + active compaction", COLORS["orange"], "s"),
]:
    d = phase[phase["method"] == method].sort_values("active_ratio")
    xvals = 100*d["active_ratio"]
    y = d["speedup_vs_direct_dense_median"]
    lo = d["speedup_vs_direct_dense_min"]
    hi = d["speedup_vs_direct_dense_max"]
    ax.plot(xvals, y, color=color, marker=marker, linewidth=1.5, markersize=3.8, label=label)
    ax.fill_between(xvals, lo, hi, color=color, alpha=0.13, linewidth=0)
ax.axhline(1, color=COLORS["gray"], linestyle="--", linewidth=1, label="Dense Direct baseline")
ax.set_xlabel("Active destinations (%)")
ax.set_ylabel("Speedup over dense Direct (×)")
ax.set_yscale("log")
ax.legend(frameon=False, loc="upper right")
files = save_figure(fig, "fig08_active_destination_phase_boundary")
add_manifest(
    "F08", files, "Active-destination compaction phase boundary",
    [active_path],
    "4 GPUs, 1M edges, hidden dimension 64, four relations; active-ratio fine sweep; median with min-max band",
    len(phase), "3 independent process launches per point",
    "Compaction provides large gains when the active destination set is sparse and converges toward the dense baseline as activity rises",
    "The same phase boundary for arbitrary graph sizes, hidden dimensions, or industrial distributions",
)


manifest_df = pd.DataFrame(manifest)
manifest_df.to_csv(TABLES / "figure_manifest.csv", index=False)

notes = f"""# Draft figure notes

Generated from quality-filtered result summaries under `sigmod2027_20241129`. No source result was modified and no missing measurement was imputed.

## F01 — Industrial snapshots

- Uses blocking C=1, B=25K, and four GPUs.
- Each bar selects the faster of edge-count and vector-aware SuperBlock assignment for that snapshot.
- Mean speedup is {snapshot['speedup_vs_direct'].mean():.3f}x; observed range is {snapshot['speedup_vs_direct'].min():.3f}–{snapshot['speedup_vs_direct'].max():.3f}x.
- Most cells have one independent run. The figure intentionally has no pseudo-error bars.

## F02a — Industrial strong scaling

- Uses only snapshot 20241129 because it is the complete repeated 1/2/3/4-GPU industrial series.
- Every point has at least three independent runs.
- This is a negative systems result: the workload does not positively strong-scale. Do not describe the line as near-linear scaling.

## F02b — Controlled large-edge scaling

- Uses 16M edges and three active-destination ratios.
- The compact execution path reaches positive strong scaling, especially at 10% active destinations.
- Keep this separate from the industrial scaling result.

## F03 — Budget and regret

- Search seeds are averaged within each snapshot before uncertainty is calculated across 12 snapshots.
- `history_warm` and `joint_search` are numerically identical in the current replay output, so the plotted label is “History / joint.”
- H=4 history/joint mean regret is {budget_summary[(budget_summary.method=='history_warm') & (budget_summary.budget==4)].iloc[0].top1_regret_percent_mean:.3f}%.

## F04 — Horizon total cost

- Includes search, preparation, and training time.
- Uses H=4 for online search methods and H=0 for fixed, two-stage, and measured-oracle references.
- This is the figure to use for break-even claims. Do not infer total-cost benefit from F03 alone.

## F05 — Memory model

- Prediction panel: Median APE {median_ape:.2f}%, P95 APE {p95_ape:.2f}%, MAE {mae/1024:.2f} GiB over {len(memory)} plans.
- The controlled panel uses strict log-level reclassification: 4/8 cells encountered OOM, including 3 recovered-and-completed cells and 1 fatal cell; 4/8 cells completed without any OOM evidence.
- Under the strict no-OOM definition, the safe bound has 0/{unsafe_den} unsafe accepts and {int(oom_audit['false_reject_strict'].sum())}/{safe_den} false rejects.
- The old “7 feasible / 1 OOM” exit-code classification is deliberately not plotted.

## F06 — DBLP end-to-end

- Runtime points have three independent runs.
- Accuracy panel uses the semantics-aligned Native PyG and Direct pair only.
- DBLP is too small to positively strong-scale; this result supports portability and semantic alignment, not multi-GPU speedup.

## F07 — Ablation

- Direct, organization-only, and vector-aware stages use fixed four-GPU blocking plans.
- Planner H=4 and oracle use measured plan costs over a 50-epoch horizon.
- Search overhead is excluded from this ablation and is handled in F04.

## F08 — Active-destination phase boundary

- Shaded bands show min-to-max across three independent process launches.
- The log y-axis is intentional because the low-active-ratio speedups are an order of magnitude larger than the high-active-ratio gains.

## Global limitations

- No figure supports eight-GPU, multi-node, second-hardware, or HGT claims.
- F01 is robustness evidence across snapshots, not a repeated-run confidence analysis.
- F02a and F02b must remain separate: positive scaling appears in the large controlled workload, not in the current industrial workload.
- F05 uses the completed strict OOM audit and must not be replaced with the legacy exit-code summary.
"""
(TABLES / "draft_figure_notes.md").write_text(notes, encoding="utf-8")

print(json.dumps({
    "figures": len(manifest),
    "png_files": len(list(FIGURES.glob("*.png"))),
    "pdf_files": len(list(FIGURES.glob("*.pdf"))),
    "tables": len(list(TABLES.glob("*.csv"))),
    "output": str(PACKAGE),
}, indent=2))

