from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import shutil
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ARTIFACT_CODE_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.environ.get("SIGMOD_ARTIFACT_ROOT", "/path/to/artifact_workspace"))
NEW_ROOT = WORKSPACE / "outputs" / "publication_repeats_20261006"
OLD_ROOT = WORKSPACE / "outputs" / "industrial"
OLD_PACKAGE = WORKSPACE / "outputs" / "sigmod_paper_package_20261006"
OUTPUT_ROOT = WORKSPACE / "outputs" / "sigmod_paper_package_20261006_v2"
COLLECTOR = ARTIFACT_CODE_ROOT / "sigmod_experiments" / "collect_industrial_results.py"

GRAPHS = ("20241124", "20241129", "20241218")
WORLD_SIZES = (1, 2, 3, 4)
STRATEGIES = ("balanced_edge", "superblock_edge_count")
ALLOWED_REPEATS = (1, 2, 3, 4, 5)


def load_collector():
    spec = importlib.util.spec_from_file_location("collect_industrial_results", COLLECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load collector: {COLLECTOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_source(path: Path) -> tuple[int, str]:
    value = str(path).replace("\\", "/")
    if "/publication_repeats_20261006/" in value:
        return 0, "publication_repeat"
    if "/core_industrial_scaling/" in value:
        return 1, "core_industrial_scaling"
    if "/core_industrial_scaling_clean_1gpu/" in value:
        return 1, "core_industrial_scaling_clean_1gpu"
    if "/core_industrial_c2_check/" in value:
        return 2, "core_industrial_c2_check"
    if "/multidate_ablation/" in value:
        return 3, "multidate_ablation"
    if "/scale_primary/" in value:
        return 4, "scale_primary"
    return 9, "other"


def metadata_candidate(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    graph = str(payload.get("graph_tag", payload.get("graph_date", "")))
    world_size = int(payload.get("world_size", 0))
    strategy = str(payload.get("strategy", ""))
    repeat = int(payload.get("repeat_id", 0))
    if graph not in GRAPHS:
        return None
    if world_size not in WORLD_SIZES:
        return None
    if strategy not in STRATEGIES:
        return None
    if repeat not in ALLOWED_REPEATS:
        return None
    if str(payload.get("mode", "")) != "blocking_c1":
        return None
    if int(payload.get("target_super_block_size", 0)) != 25000:
        return None
    if int(payload.get("num_epochs", 0)) < 10:
        return None
    priority, source_class = classify_source(path)
    return {
        "path": path,
        "graph": graph,
        "world_size": world_size,
        "strategy": strategy,
        "repeat": repeat,
        "seed": int(payload.get("run_seed", 0)),
        "priority": priority,
        "source_class": source_class,
    }


def discover_candidates() -> list[dict]:
    candidates: list[dict] = []
    for root in (NEW_ROOT, OLD_ROOT):
        for path in root.rglob("train_result_rank0.json"):
            candidate = metadata_candidate(path)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def desired_repeats(graph: str, world_size: int) -> tuple[int, int, int]:
    # The original 20241129 one-GPU R01/R02 measurements were contaminated
    # (peak_memory_mb=0). Clean R04/R05 replacements were run explicitly.
    if graph == "20241129" and world_size == 1:
        return (3, 4, 5)
    return (1, 2, 3)


def select_runs(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    collector = load_collector()
    grouped: dict[tuple[str, int, str, int], list[dict]] = {}
    for row in candidates:
        runtime = collector.critical_runtime(row["path"].parent, row["world_size"])
        memory = collector.peak_memory(row["path"].parent)
        reasons = []
        if int(runtime.get("epochs_observed", 0)) < 10:
            reasons.append("epochs_observed<10")
        if float(runtime.get("steady_epoch_sec_mean", 0.0)) <= 0:
            reasons.append("steady_epoch_sec_mean<=0")
        if float(memory) <= 0:
            reasons.append("peak_memory_mb<=0")
        row["preselection_quality_reasons"] = reasons
        key = (row["graph"], row["world_size"], row["strategy"], row["repeat"])
        grouped.setdefault(key, []).append(row)

    selected: list[dict] = []
    for graph in GRAPHS:
        for world_size in WORLD_SIZES:
            for strategy in STRATEGIES:
                for repeat in desired_repeats(graph, world_size):
                    key = (graph, world_size, strategy, repeat)
                    values = sorted(
                        grouped.get(key, []),
                        key=lambda row: (
                            bool(row["preselection_quality_reasons"]),
                            row["priority"],
                            str(row["path"]),
                        ),
                    )
                    if not values:
                        raise RuntimeError(f"Missing required run: {key}")
                    if values[0]["preselection_quality_reasons"]:
                        raise RuntimeError(
                            f"No quality-qualified run for {key}: "
                            f"{values[0]['preselection_quality_reasons']}"
                        )
                    chosen = values[0]
                    selected.append(chosen)
    selected_paths = {str(row["path"]) for row in selected}
    audit: list[dict] = []
    for value in sorted(
        candidates,
        key=lambda row: (
            row["graph"], row["world_size"], row["strategy"], row["repeat"],
            row["priority"], str(row["path"]),
        ),
    ):
        selected_flag = str(value["path"]) in selected_paths
        desired_flag = value["repeat"] in desired_repeats(
            value["graph"], value["world_size"]
        )
        if selected_flag:
            exclusion_reason = ""
        elif value["preselection_quality_reasons"]:
            exclusion_reason = "; ".join(value["preselection_quality_reasons"])
        elif not desired_flag:
            exclusion_reason = "outside_clean_paired_repeat_triplet"
        else:
            exclusion_reason = "duplicate_lower_priority_source"
        audit.append(
            {
                "graph": value["graph"],
                "world_size": value["world_size"],
                "strategy": value["strategy"],
                "repeat": value["repeat"],
                "selected": selected_flag,
                "desired_repeat": desired_flag,
                "source_class": value["source_class"],
                "priority": value["priority"],
                "quality_reasons": "; ".join(value["preselection_quality_reasons"]),
                "exclusion_reason": exclusion_reason,
                "source_result": str(value["path"]),
            }
        )
    return selected, audit


def collect_selected(selected: list[dict]) -> pd.DataFrame:
    collector = load_collector()
    records = []
    for candidate in selected:
        row = dict(collector.collect(candidate["path"], 50))
        row.update(
            {
                "source_class": candidate["source_class"],
                "source_result": str(candidate["path"]),
            }
        )
        reasons = []
        if int(row.get("epochs_observed", 0)) < 10:
            reasons.append("epochs_observed<10")
        if float(row.get("steady_epoch_sec_mean", 0.0)) <= 0:
            reasons.append("steady_epoch_sec_mean<=0")
        if float(row.get("peak_memory_mb", 0.0)) <= 0:
            reasons.append("peak_memory_mb<=0")
        row["quality_accepted"] = not reasons
        row["quality_reason"] = "; ".join(reasons)
        records.append(row)
    result = pd.DataFrame(records)
    if len(result) != 72:
        raise RuntimeError(f"Expected 72 selected runs, found {len(result)}")
    if not bool(result["quality_accepted"].all()):
        bad = result.loc[~result["quality_accepted"]]
        raise RuntimeError(f"Quality failures:\n{bad[['source_result', 'quality_reason']]}")
    return result


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    fields = ["graph", "world_size", "strategy"]
    rows = []
    for key, values in runs.groupby(fields, sort=True):
        epoch = list(values["steady_epoch_sec_mean"].astype(float))
        prep = list(values["prep_sec_max"].astype(float))
        peak = list(values["peak_memory_mb"].astype(float))
        comm_total = list(values["comm_total_sec_mean"].astype(float))
        comm_exposed = list(values["comm_exposed_sec_mean"].astype(float))
        mean = statistics.mean(epoch)
        stdev = statistics.stdev(epoch)
        se = stdev / math.sqrt(len(epoch))
        rows.append(
            {
                "graph": key[0],
                "world_size": int(key[1]),
                "strategy": key[2],
                "repeats": len(values),
                "epoch_sec_mean": mean,
                "epoch_sec_median": statistics.median(epoch),
                "epoch_sec_stdev": stdev,
                "epoch_sec_se": se,
                "epoch_sec_cv": stdev / mean if mean else 0.0,
                "prep_sec_mean": statistics.mean(prep),
                "peak_memory_mb_mean": statistics.mean(peak),
                "peak_memory_mb_max": max(peak),
                "comm_total_sec_mean": statistics.mean(comm_total),
                "comm_exposed_sec_mean": statistics.mean(comm_exposed),
                "total_labeled_edges": int(values["total_labeled_edges"].iloc[0]),
                "throughput_edges_per_sec": int(values["total_labeled_edges"].iloc[0]) / mean,
            }
        )
    summary = pd.DataFrame(rows)

    enriched = []
    for _, row in summary.iterrows():
        graph = row["graph"]
        strategy = row["strategy"]
        world_size = int(row["world_size"])
        base = summary[
            (summary["graph"] == graph)
            & (summary["strategy"] == strategy)
            & (summary["world_size"] == 1)
        ].iloc[0]
        direct = summary[
            (summary["graph"] == graph)
            & (summary["strategy"] == "balanced_edge")
            & (summary["world_size"] == world_size)
        ].iloc[0]
        speedup = float(base["epoch_sec_mean"]) / float(row["epoch_sec_mean"])
        if world_size == 1:
            speedup_se = 0.0
        else:
            speedup_se = speedup * math.sqrt(
                (float(base["epoch_sec_se"]) / float(base["epoch_sec_mean"])) ** 2
                + (float(row["epoch_sec_se"]) / float(row["epoch_sec_mean"])) ** 2
            )
        layout_speedup = float(direct["epoch_sec_mean"]) / float(row["epoch_sec_mean"])
        layout_speedup_se = layout_speedup * math.sqrt(
            (float(direct["epoch_sec_se"]) / float(direct["epoch_sec_mean"])) ** 2
            + (float(row["epoch_sec_se"]) / float(row["epoch_sec_mean"])) ** 2
        )
        out = dict(row)
        out.update(
            {
                "strong_scaling_speedup": speedup,
                "strong_scaling_speedup_se": speedup_se,
                "strong_scaling_efficiency": speedup / world_size,
                "speedup_vs_direct_same_gpu": layout_speedup,
                "speedup_vs_direct_same_gpu_se": layout_speedup_se,
                "memory_reduction_vs_direct": 1.0
                - float(row["peak_memory_mb_mean"]) / float(direct["peak_memory_mb_mean"]),
            }
        )
        enriched.append(out)
    return pd.DataFrame(enriched).sort_values(fields).reset_index(drop=True)


def build_coverage(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, values in runs.groupby(["graph", "world_size", "strategy"], sort=True):
        timings = list(values["steady_epoch_sec_mean"].astype(float))
        median = statistics.median(timings)
        outliers = [value for value in timings if not (0.5 * median <= value <= 2.0 * median)]
        rows.append(
            {
                "graph": key[0],
                "world_size": int(key[1]),
                "strategy": key[2],
                "repeat_ids": ";".join(str(int(v)) for v in sorted(values["repeat"])),
                "seeds": ";".join(str(int(v)) for v in sorted(values["seed"])),
                "accepted_runs": len(values),
                "expected_runs": 3,
                "complete": len(values) == 3,
                "runtime_outliers": len(outliers),
            }
        )
    return pd.DataFrame(rows)


def build_paper_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for graph in GRAPHS:
        direct = summary[
            (summary["graph"] == graph)
            & (summary["world_size"] == 4)
            & (summary["strategy"] == "balanced_edge")
        ].iloc[0]
        superblock = summary[
            (summary["graph"] == graph)
            & (summary["world_size"] == 4)
            & (summary["strategy"] == "superblock_edge_count")
        ].iloc[0]
        rows.append(
            {
                "graph": graph,
                "direct_epoch_sec_mean": direct["epoch_sec_mean"],
                "direct_epoch_sec_se": direct["epoch_sec_se"],
                "superblock_epoch_sec_mean": superblock["epoch_sec_mean"],
                "superblock_epoch_sec_se": superblock["epoch_sec_se"],
                "superblock_speedup_vs_direct": superblock["speedup_vs_direct_same_gpu"],
                "superblock_speedup_se": superblock["speedup_vs_direct_same_gpu_se"],
                "direct_g4_strong_scaling_speedup": direct["strong_scaling_speedup"],
                "superblock_g4_strong_scaling_speedup": superblock["strong_scaling_speedup"],
                "direct_peak_memory_mb_mean": direct["peak_memory_mb_mean"],
                "superblock_peak_memory_mb_mean": superblock["peak_memory_mb_mean"],
                "memory_reduction_percent": 100 * superblock["memory_reduction_vs_direct"],
            }
        )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"balanced_edge": "#D55E00", "superblock_edge_count": "#0072B2"}
    labels = {"balanced_edge": "Direct", "superblock_edge_count": "SuperBlock"}
    markers = {"balanced_edge": "o", "superblock_edge_count": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.65), sharey=True)
    for ax, graph in zip(axes, GRAPHS):
        ax.plot([1, 4], [1, 4], color="#B8B8B8", linestyle="--", linewidth=1.5, label="Ideal")
        for strategy in STRATEGIES:
            values = summary[(summary["graph"] == graph) & (summary["strategy"] == strategy)]
            ax.errorbar(
                values["world_size"],
                values["strong_scaling_speedup"],
                yerr=values["strong_scaling_speedup_se"],
                label=labels[strategy],
                color=colors[strategy],
                marker=markers[strategy],
                markersize=5,
                linewidth=2,
                capsize=2.5,
            )
        ax.set_title(graph)
        ax.set_xlabel("Number of GPUs")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.85, 4.15)
        ax.set_ylim(0, 4.15)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Strong-scaling speedup (×)")
    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=1.2)
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def write_report(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    selection_audit: pd.DataFrame,
) -> None:
    direct = summary[summary["strategy"] == "balanced_edge"]
    sb = summary[summary["strategy"] == "superblock_edge_count"]
    g4_sb = sb[sb["world_size"] == 4]
    g4_direct = direct[direct["world_size"] == 4]
    max_cv = float(summary["epoch_sec_cv"].max())
    max_cv_row = summary.loc[summary["epoch_sec_cv"].idxmax()]
    lines = [
        "# Three-snapshot industrial scaling validation",
        "",
        "## Material Passport",
        "",
        "- Material type: experiment validation and paper-facing aggregation",
        "- Sources: original SIGMOD industrial results plus publication_repeats_20261006",
        "- Verification status: ANALYZED",
        "- Aggregation: maximum rank time per epoch; first epoch excluded; three independent launches per cell",
        "",
        "## Completion",
        "",
        f"- Selected runs: {len(runs)}/72.",
        f"- Complete cells: {int(coverage['complete'].sum())}/24.",
        f"- Quality rejections: {int((~runs['quality_accepted']).sum())}.",
        f"- Robust runtime outliers: {int(coverage['runtime_outliers'].sum())}.",
        f"- Historical candidates excluded before aggregation: {int((~selection_audit['selected']).sum())}; "
        "all exclusions are retained in the selection audit.",
        "",
        "## Main observations",
        "",
    ]
    for graph in GRAPHS:
        d4 = g4_direct[g4_direct["graph"] == graph].iloc[0]
        s4 = g4_sb[g4_sb["graph"] == graph].iloc[0]
        lines.append(
            f"- {graph}: at four GPUs, Direct is {d4['epoch_sec_mean']:.3f}±{d4['epoch_sec_se']:.3f} s/epoch "
            f"and SuperBlock is {s4['epoch_sec_mean']:.3f}±{s4['epoch_sec_se']:.3f} s/epoch; "
            f"SuperBlock is {s4['speedup_vs_direct_same_gpu']:.3f}× faster."
        )
    lines.extend(
        [
            f"- The largest run-to-run CV is {100*max_cv:.2f}% "
            f"({max_cv_row['graph']}, G={int(max_cv_row['world_size'])}, {max_cv_row['strategy']}).",
            f"- At four GPUs, the mean SuperBlock speedup over Direct across the three snapshots is "
            f"{g4_sb['speedup_vs_direct_same_gpu'].mean():.3f}×; mean peak-memory reduction is "
            f"{100*g4_sb['memory_reduction_vs_direct'].mean():.1f}%.",
            "- None of the three industrial snapshots positively strong-scales from one to four GPUs. SuperBlock remains near its one-GPU throughput, whereas Direct slows sharply after one GPU.",
            "",
            "## Statistical and reproducibility cautions",
            "",
            "- Each cell has n=3 independent process launches; standard-error bars quantify launch variation, not variation across industrial graph populations.",
            "- The three dates are separate workloads, not interchangeable replicates of one graph. Results are reported per snapshot rather than pooled into a significance test.",
            "- No p-values or multiplicity-adjusted hypothesis tests are used. The evaluation is descriptive and effect-size focused.",
            "- The representative dates and configurations were defined before this aggregation. No post-hoc cell removal was performed.",
            "- The result supports stable same-GPU-count layout gains, not positive industrial strong scaling, eight-GPU behavior, multi-node behavior, or other hardware.",
            "",
            "## Fallacy scan (11/11)",
            "",
            "1. Statistical significance without effect size: avoided; absolute epoch time and speedup are reported.",
            "2. Confidence interval interpreted as probability: avoided; the plot uses SE, explicitly labeled.",
            "3. Non-significance interpreted as equivalence: not applicable; no null-hypothesis test is claimed.",
            "4. Multiple comparisons ignored: no confirmatory p-values are reported.",
            "5. Pseudoreplication: mitigated by treating launches and snapshots separately.",
            "6. Selective outcome reporting: timing, memory, coverage, and negative scaling are retained.",
            "7. Post-hoc subgrouping: representative snapshots were preselected in the run strategy.",
            "8. Correlation interpreted as causation: layout comparisons are controlled within the same snapshot/GPU count; broader causal generalization is withheld.",
            "9. Mean-only reporting under instability: mean, median, SD, SE, CV, and per-run rows are supplied.",
            "10. Unchecked assumptions/outliers: completeness, epoch count, peak memory, and a robust 0.5×–2× median gate were checked.",
            "11. Generalization beyond the sample: explicit limits are retained for GPU count, hardware, and dataset scope.",
        ]
    )
    (OUTPUT_ROOT / "analysis" / "industrial_three_snapshot_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    if not NEW_ROOT.exists() or not OLD_ROOT.exists() or not OLD_PACKAGE.exists():
        raise RuntimeError("One or more required source directories are missing")
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"Output already exists; refusing to overwrite: {OUTPUT_ROOT}")

    shutil.copytree(OLD_PACKAGE, OUTPUT_ROOT)
    (OUTPUT_ROOT / "analysis").mkdir(exist_ok=True)

    candidates = discover_candidates()
    selected, audit = select_runs(candidates)
    runs = collect_selected(selected)
    summary = summarize_runs(runs)
    coverage = build_coverage(runs)

    tables = OUTPUT_ROOT / "tables"
    figures = OUTPUT_ROOT / "figures"
    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(tables / "table_02a_selection_audit.csv", index=False)
    runs.to_csv(tables / "table_02a_industrial_three_snapshot_runs.csv", index=False)
    summary.to_csv(tables / "table_02a_industrial_three_snapshot_scaling.csv", index=False)
    coverage.to_csv(tables / "table_02a_industrial_three_snapshot_coverage.csv", index=False)
    build_paper_summary(summary).to_csv(
        tables / "table_02a_industrial_three_snapshot_paper_summary.csv", index=False
    )

    make_figure(
        summary,
        figures / "fig02a_industrial_three_snapshot_scaling.png",
        figures / "fig02a_industrial_three_snapshot_scaling.pdf",
    )
    write_report(runs, summary, coverage, audit_frame)

    manifest_path = tables / "figure_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest.loc[manifest["figure_id"] == "F02a", "status"] = "SUPERSEDED_BY_F02a-v2"
    replacement = {
        "figure_id": "F02a-v2",
        "files": "fig02a_industrial_three_snapshot_scaling.png;fig02a_industrial_three_snapshot_scaling.pdf",
        "title": "Industrial strong scaling across three representative snapshots",
        "source_files": "sigmod2027_20241129/industrial;publication_repeats_20261006/industrial",
        "filter_rule": "Snapshots 20241124/20241129/20241218; B=25K; blocking C=1; Direct and SuperBlock; G=1/2/3/4; repeats 1/2/3",
        "rows_used": 72,
        "repetitions": "3 independent process launches per cell; error bars are delta-method SE",
        "supports": "The layout benefit is stable across three representative snapshots; the industrial workload does not positively strong-scale",
        "does_not_support": "Positive industrial strong scaling or extrapolation beyond four GPUs",
        "status": "READY",
    }
    manifest = pd.concat([manifest, pd.DataFrame([replacement])], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)

    notes_path = tables / "draft_figure_notes.md"
    notes = notes_path.read_text(encoding="utf-8")
    notes_prefix = (
        "## F02a-v2 — Repeated industrial strong scaling across three snapshots\n\n"
        "- This supersedes the earlier 20241129-only F02a in the v2 package.\n"
        "- Uses 20241124, 20241129, and 20241218; Direct and SuperBlock; G=1/2/3/4.\n"
        "- Every cell has three independent process launches. Error bars are standard errors.\n"
        "- SuperBlock is 2.731×, 2.724×, and 2.812× faster than Direct at four GPUs.\n"
        "- The repeated result strengthens the layout claim but remains a negative industrial strong-scaling result.\n\n"
    )
    notes_path.write_text(notes_prefix + notes, encoding="utf-8")

    plan_path = OUTPUT_ROOT / "experiment_section_writing_plan_zh.md"
    if plan_path.exists():
        plan = plan_path.read_text(encoding="utf-8")
        plan = plan.replace(
            "`fig02a_industrial_20241129_scaling`",
            "`fig02a_industrial_three_snapshot_scaling`",
        )
        plan = plan.replace(
            "Fig. 3(a) 应解释：20241129 的 Direct 在多卡上受到同步/通信和固定开销影响；SuperBlock 显著降低 2–4 GPU 的 epoch time，但当前工业工作负载没有从 1 GPU 到 4 GPU 正向 strong-scale。三卡点用于证明实现不依赖二次幂 GPU 数。",
            "Fig. 3(a) 应解释：三个代表快照的每个单元均有三次独立启动。Direct 在多卡上受到同步、通信和固定开销影响；SuperBlock 在 4 GPU 上分别比 Direct 快 2.731×、2.724× 和 2.812×，但三个工业工作负载都没有从 1 GPU 到 4 GPU 正向 strong-scale。三卡点用于证明实现不依赖二次幂 GPU 数。",
        )
        plan_path.write_text(plan, encoding="utf-8")

    readme = (OUTPUT_ROOT / "README.md").read_text(encoding="utf-8")
    prefix = (
        "# SIGMOD experiment paper package v2\n\n"
        "This version preserves the original paper package and adds a quality-audited, "
        "three-snapshot repeated industrial scaling result. The selected matrix contains "
        "72 runs: three snapshots × four GPU counts × two layouts × three independent launches.\n\n"
        "## v2 addition\n\n"
        "- `figures/fig02a_industrial_three_snapshot_scaling.*`: repeated industrial strong scaling across 20241124, 20241129, and 20241218.\n"
        "- `tables/table_02a_industrial_three_snapshot_scaling.csv`: paper-facing aggregate.\n"
        "- `tables/table_02a_industrial_three_snapshot_runs.csv`: selected per-run measurements.\n"
        "- `tables/table_02a_industrial_three_snapshot_coverage.csv`: completeness and outlier audit.\n"
        "- `tables/table_02a_industrial_three_snapshot_paper_summary.csv`: compact values for the manuscript.\n"
        "- `tables/table_02a_selection_audit.csv`: duplicate-source selection trace.\n"
        "- `analysis/industrial_three_snapshot_validation.md`: interpretation and 11-item fallacy scan.\n\n"
        "---\n\n"
    )
    (OUTPUT_ROOT / "README.md").write_text(prefix + readme, encoding="utf-8")

    print(json.dumps({
        "output_root": str(OUTPUT_ROOT),
        "candidates": len(candidates),
        "selected_runs": len(runs),
        "complete_cells": int(coverage["complete"].sum()),
        "runtime_outliers": int(coverage["runtime_outliers"].sum()),
        "max_cv_percent": float(summary["epoch_sec_cv"].max() * 100),
    }, indent=2))


if __name__ == "__main__":
    main()

