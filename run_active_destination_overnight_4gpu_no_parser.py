#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restartable overnight four-GPU active-destination experiment.

Direct run, no argparse.  The parent launches three independent torchrun
workers.  Each worker uses a fresh process group and seed, shuffles case and
method order, and writes one JSON per completed method.  Re-running the file
skips completed cells and resumes the unfinished repeat.

The experiment reuses the exact operator implementation in
``run_active_destination_phase_boundary_4gpu_no_parser.py``.  Results are
written to a new directory and never overwrite the original 104-cell pilot.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(
    "/path/to/artifact_workspace/sigmod2027_20241129/active_destination_overnight_4gpu"
)
GPU_IDS = (0, 1, 2, 3)
WORLD_SIZE = 4
INDEPENDENT_REPEATS = (1, 2, 3)
WARMUPS_PER_METHOD = 3
TIMED_STEPS_PER_METHOD = 9
BASE_SEED = 20261005
CANONICAL_METHODS = (
    "direct_dense",
    "direct_compact",
    "superblock_dense",
    "superblock_compact",
)


def safe_name(value: Any) -> str:
    return str(value).replace(".", "p")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def complete_json(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("completed"))
    except Exception:
        return False


def overnight_cases() -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    def add(axis, value, edges, nodes, relations, active, skew, hidden):
        output.append(
            {
                "axis": str(axis),
                "axis_value": float(value),
                "total_edges": int(edges),
                "num_dst": int(nodes),
                "num_relations": int(relations),
                "active_ratio": float(active),
                "skew": float(skew),
                "hidden_dim": int(hidden),
                "case_id": (
                    f"{axis}-{safe_name(value)}__e{int(edges)}__n{int(nodes)}"
                    f"__r{int(relations)}__a{safe_name(active)}"
                    f"__s{safe_name(skew)}__h{int(hidden)}"
                ),
            }
        )

    # Fine-grained fallback boundary at the paper's default d=64.
    for ratio in (
        0.01,
        0.05,
        0.10,
        0.25,
        0.40,
        0.50,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ):
        add("active_ratio_fine", ratio, 1_000_000, 250_000, 4, ratio, 1.0, 64)

    # Two-dimensional phase surface: active footprint x vector width.
    for hidden in (32, 128, 256):
        for ratio in (0.25, 0.50, 0.65, 0.75, 0.90, 1.00):
            add(
                f"phase_surface_h{hidden}",
                ratio,
                1_000_000,
                250_000,
                4,
                ratio,
                1.0,
                hidden,
            )

    # Stress empty-relation partitions and validate the memory fix.
    for relations in (8, 16, 24, 32, 48, 64):
        add(
            "relation_count_pressure",
            relations,
            1_000_000,
            50_000,
            relations,
            0.25,
            1.0,
            64,
        )

    # Larger operator sizes, using d=128 to expose the communication trend.
    for edges in (100_000, 500_000, 1_000_000, 2_000_000, 4_000_000):
        add(
            "scale_edges_pressure",
            edges,
            edges,
            max(10_000, edges // 8),
            4,
            0.25,
            1.0,
            128,
        )

    # Check whether the fallback boundary moves with degree skew.
    for ratio in (0.50, 0.75, 0.90):
        for skew in (0.0, 1.0, 2.0):
            add(
                f"skew_boundary_a{safe_name(ratio)}",
                skew,
                1_000_000,
                250_000,
                4,
                ratio,
                skew,
                64,
            )

    identifiers = [row["case_id"] for row in output]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Duplicate overnight case identifiers")
    return output


def repeat_root(repeat_id: int) -> Path:
    return OUTPUT_ROOT / f"repeat_{repeat_id:02d}"


def worker_main() -> None:
    import run_active_destination_phase_boundary_4gpu_no_parser as phase

    repeat_id = int(os.environ["SIGMOD_OVERNIGHT_REPEAT"])
    run_seed = BASE_SEED + repeat_id * 100_003
    method_order = list(CANONICAL_METHODS)
    random.Random(run_seed + 17).shuffle(method_order)
    case_rows = overnight_cases()
    random.Random(run_seed + 31).shuffle(case_rows)
    for row in case_rows:
        row["independent_repeat"] = repeat_id
        row["run_seed"] = run_seed
        row["method_order"] = ",".join(method_order)
        row["implementation_revision"] = "scalar_anchor_shared_relation_group_v3"

    def selected_cases():
        return [dict(row) for row in case_rows]

    phase.OUTPUT_ROOT = repeat_root(repeat_id)
    phase.BASE_SEED = run_seed
    phase.WARMUPS = WARMUPS_PER_METHOD
    phase.REPEATS = TIMED_STEPS_PER_METHOD
    phase.METHODS = tuple(method_order)
    phase.cases = selected_cases
    # The root coordinator produces cross-repeat plots and reports.
    phase.create_plots = lambda rows: None
    phase.create_report = lambda rows: None
    phase.run_worker()


def launch_repeat(repeat_id: int) -> None:
    destination = repeat_root(repeat_id)
    completion = destination / "complete.json"
    expected_rows = len(overnight_cases()) * len(CANONICAL_METHODS)
    if complete_json(completion):
        value = json.loads(completion.read_text(encoding="utf-8"))
        if int(value.get("result_rows", -1)) == expected_rows:
            print(f"SKIP complete independent repeat {repeat_id}: {destination}", flush=True)
            return

    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in GPU_IDS),
            "SIGMOD_OVERNIGHT_WORKER": "1",
            "SIGMOD_OVERNIGHT_REPEAT": str(repeat_id),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
    ]
    log_path = destination / "launcher.log"
    print(f"RUN independent repeat {repeat_id}: {' '.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n===== RESUME {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        returncode = process.wait()
    if returncode != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-180:]
        )
        print(tail, flush=True)
        raise RuntimeError(
            f"overnight repeat {repeat_id} failed; rerun the same file to resume: {log_path}"
        )


def load_completed_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for repeat_id in INDEPENDENT_REPEATS:
        root = repeat_root(repeat_id) / "cases"
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not value.get("completed") or not value.get("method"):
                continue
            value["independent_repeat"] = repeat_id
            value.pop("step_samples_sec", None)
            rows.append(value)
    return rows


def median(values: Sequence[float]) -> float:
    return float(statistics.median(float(value) for value in values))


def aggregate_results() -> List[Dict[str, Any]]:
    raw_rows = load_completed_rows()
    if not raw_rows:
        return []

    keyed = {
        (int(row["independent_repeat"]), str(row["case_id"]), str(row["method"])): row
        for row in raw_rows
    }
    for row in raw_rows:
        repeat_id = int(row["independent_repeat"])
        case_id = str(row["case_id"])
        method = str(row["method"])
        direct = keyed.get((repeat_id, case_id, "direct_dense"))
        if direct is not None:
            row["run_speedup_vs_direct_dense"] = float(direct["step_sec_median"]) / max(
                float(row["step_sec_median"]), 1e-12
            )
        if method == "direct_compact":
            dense = direct
        elif method == "superblock_compact":
            dense = keyed.get((repeat_id, case_id, "superblock_dense"))
        else:
            dense = None
        if dense is not None:
            row["run_compact_speedup_same_layout"] = float(
                dense["step_sec_median"]
            ) / max(float(row["step_sec_median"]), 1e-12)

    write_csv(OUTPUT_ROOT / "overnight_all_completed_runs.csv", raw_rows)
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        groups[(str(row["case_id"]), str(row["method"]))].append(row)

    aggregate: List[Dict[str, Any]] = []
    for (case_id, method), values in groups.items():
        first = values[0]
        step_values = [float(row["step_sec_median"]) for row in values]
        speed_values = [
            float(row["run_speedup_vs_direct_dense"])
            for row in values
            if "run_speedup_vs_direct_dense" in row
        ]
        compact_values = [
            float(row["run_compact_speedup_same_layout"])
            for row in values
            if "run_compact_speedup_same_layout" in row
        ]
        row = {
            key: first[key]
            for key in (
                "axis",
                "axis_value",
                "total_edges",
                "num_dst",
                "num_relations",
                "active_ratio",
                "skew",
                "hidden_dim",
                "case_id",
                "method",
                "world_size",
                "layout",
                "compact",
                "collective_payload_bytes_forward",
                "dense_reference_payload_bytes_forward",
                "payload_reduction_percent",
                "implementation_revision",
            )
            if key in first
        }
        row.update(
            {
                "independent_repeat_count": len(values),
                "step_sec_independent_median": median(step_values),
                "step_sec_independent_min": min(step_values),
                "step_sec_independent_max": max(step_values),
                "step_sec_independent_stdev": (
                    statistics.stdev(step_values) if len(step_values) > 1 else 0.0
                ),
                "speedup_vs_direct_dense_median": (
                    median(speed_values) if speed_values else ""
                ),
                "speedup_vs_direct_dense_min": min(speed_values) if speed_values else "",
                "speedup_vs_direct_dense_max": max(speed_values) if speed_values else "",
                "compact_speedup_same_layout_median": (
                    median(compact_values) if compact_values else ""
                ),
                "prep_sec_median": median(
                    [float(value["prep_sec_max"]) for value in values]
                ),
                "peak_gpu_memory_mb_median": median(
                    [float(value["peak_gpu_memory_mb"]) for value in values]
                ),
                "peak_gpu_memory_mb_max": max(
                    float(value["peak_gpu_memory_mb"]) for value in values
                ),
                "forward_relative_l2_error_max": max(
                    float(value["forward_relative_l2_error"]) for value in values
                ),
                "gradient_relative_l2_error_max": max(
                    float(value["gradient_relative_l2_error"]) for value in values
                ),
                "method_orders": ";".join(
                    sorted({str(value.get("method_order", "")) for value in values})
                ),
                "run_seeds": ";".join(
                    str(value)
                    for value in sorted({int(item.get("run_seed", 0)) for item in values})
                ),
            }
        )
        aggregate.append(row)

    aggregate.sort(
        key=lambda row: (
            str(row["axis"]),
            float(row["axis_value"]),
            CANONICAL_METHODS.index(str(row["method"])),
        )
    )
    write_csv(OUTPUT_ROOT / "overnight_aggregate.csv", aggregate)
    create_plots(aggregate)
    create_report(aggregate)
    return aggregate


def create_plots(rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"PLOTS SKIPPED: {type(exc).__name__}: {exc}", flush=True)
        return

    active = [row for row in rows if row["axis"] == "active_ratio_fine"]
    if active:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
        for method in CANONICAL_METHODS:
            values = sorted(
                [row for row in active if row["method"] == method],
                key=lambda row: float(row["axis_value"]),
            )
            x = [float(row["axis_value"]) for row in values]
            y = [float(row["speedup_vs_direct_dense_median"]) for row in values]
            lower = [float(row["speedup_vs_direct_dense_min"]) for row in values]
            upper = [float(row["speedup_vs_direct_dense_max"]) for row in values]
            axes[0].plot(x, y, marker="o", label=method.replace("_", " "))
            axes[0].fill_between(x, lower, upper, alpha=0.10)
            axes[1].plot(
                x,
                [float(row["peak_gpu_memory_mb_median"]) for row in values],
                marker="o",
                label=method.replace("_", " "),
            )
        axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Active-destination ratio")
        axes[0].set_ylabel("Median speedup vs Direct+dense")
        axes[1].set_xlabel("Active-destination ratio")
        axes[1].set_ylabel("Median peak allocated memory (MB)")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / "active_ratio_fine_4gpu.png", dpi=220)
        plt.close(fig)

    relation = [row for row in rows if row["axis"] == "relation_count_pressure"]
    if relation:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
        for method in CANONICAL_METHODS:
            values = sorted(
                [row for row in relation if row["method"] == method],
                key=lambda row: float(row["axis_value"]),
            )
            x = [float(row["axis_value"]) for row in values]
            axes[0].plot(
                x,
                [float(row["speedup_vs_direct_dense_median"]) for row in values],
                marker="o",
                label=method.replace("_", " "),
            )
            axes[1].plot(
                x,
                [float(row["peak_gpu_memory_mb_median"]) for row in values],
                marker="o",
                label=method.replace("_", " "),
            )
        axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Relation count")
        axes[0].set_ylabel("Median speedup vs Direct+dense")
        axes[1].set_xlabel("Relation count")
        axes[1].set_ylabel("Median peak allocated memory (MB)")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / "relation_count_pressure_4gpu.png", dpi=220)
        plt.close(fig)

    ratios = (0.25, 0.50, 0.65, 0.75, 0.90, 1.00)
    dimensions = (32, 64, 128, 256)
    speed_lookup: Dict[tuple, float] = {}
    for row in rows:
        if row["method"] != "direct_compact":
            continue
        hidden = int(row["hidden_dim"])
        ratio = float(row["active_ratio"])
        if hidden == 64 and row["axis"] == "active_ratio_fine":
            speed_lookup[(hidden, ratio)] = float(
                row["compact_speedup_same_layout_median"]
            )
        elif row["axis"] == f"phase_surface_h{hidden}":
            speed_lookup[(hidden, ratio)] = float(
                row["compact_speedup_same_layout_median"]
            )
    if all((hidden, ratio) in speed_lookup for hidden in dimensions for ratio in ratios):
        matrix = [
            [speed_lookup[(hidden, ratio)] for ratio in ratios]
            for hidden in dimensions
        ]
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ratios)), [f"{value:.2f}" for value in ratios])
        ax.set_yticks(range(len(dimensions)), [str(value) for value in dimensions])
        ax.set_xlabel("Active-destination ratio")
        ax.set_ylabel("Hidden dimension")
        for row_index, values in enumerate(matrix):
            for column_index, value in enumerate(values):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}x",
                    ha="center",
                    va="center",
                    color="white" if value < 3.0 else "black",
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax, label="Direct compact speedup")
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / "active_ratio_hidden_dim_phase_surface_4gpu.png", dpi=220)
        plt.close(fig)


def create_report(rows: Sequence[Mapping[str, Any]]) -> None:
    repeat_counts = [int(row["independent_repeat_count"]) for row in rows]
    max_forward = max(float(row["forward_relative_l2_error_max"]) for row in rows)
    max_gradient = max(float(row["gradient_relative_l2_error_max"]) for row in rows)
    by_case: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    best_counts: Dict[str, int] = defaultdict(int)
    for values in by_case.values():
        complete = [value for value in values if int(value["independent_repeat_count"]) == 3]
        if len(complete) == len(CANONICAL_METHODS):
            best = min(complete, key=lambda value: float(value["step_sec_independent_median"]))
            best_counts[str(best["method"])] += 1

    active = sorted(
        [row for row in rows if row["axis"] == "active_ratio_fine"],
        key=lambda row: (float(row["axis_value"]), CANONICAL_METHODS.index(str(row["method"]))),
    )
    active_by_ratio: Dict[float, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in active:
        active_by_ratio[float(row["axis_value"])][str(row["method"])] = row

    lines = [
        "# Four-GPU active-destination overnight experiment",
        "",
        f"- Structural cases: {len(by_case)} / {len(overnight_cases())}",
        f"- Aggregate method rows: {len(rows)}",
        f"- Independent repeats per completed cell: {min(repeat_counts)}–{max(repeat_counts)}",
        f"- Maximum forward relative L2 error: {max_forward:.3e}",
        f"- Maximum gradient relative L2 error: {max_gradient:.3e}",
        f"- Best-plan counts for fully repeated cases: {dict(best_counts)}",
        "",
        "## Fine active-ratio boundary",
        "",
        "| Active ratio | Direct compact / dense | SuperBlock compact / dense | Payload reduction |",
        "|---:|---:|---:|---:|",
    ]
    for ratio in sorted(active_by_ratio):
        methods = active_by_ratio[ratio]
        direct = methods.get("direct_compact")
        grouped = methods.get("superblock_compact")
        if direct is None or grouped is None:
            continue
        lines.append(
            f"| {ratio:.2f} | "
            f"{float(direct['compact_speedup_same_layout_median']):.3f}x | "
            f"{float(grouped['compact_speedup_same_layout_median']):.3f}x | "
            f"{float(direct['payload_reduction_percent']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Measurement protocol",
            "",
            f"Each cell uses {WARMUPS_PER_METHOD} warmups and {TIMED_STEPS_PER_METHOD} timed steps. "
            "The table aggregates the per-process-run medians across three fresh torchrun launches. "
            "Method order, case order, graph seed, and feature seed change across launches.",
            "",
            "## Main files",
            "",
            "- `overnight_aggregate.csv`",
            "- `overnight_all_completed_runs.csv`",
            "- `active_ratio_fine_4gpu.png`",
            "- `active_ratio_hidden_dim_phase_surface_4gpu.png`",
            "- `relation_count_pressure_4gpu.png`",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def coordinator_main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": "active_destination_overnight_4gpu",
        "cases_per_repeat": len(overnight_cases()),
        "methods_per_case": len(CANONICAL_METHODS),
        "independent_repeats": list(INDEPENDENT_REPEATS),
        "total_method_cells": (
            len(overnight_cases()) * len(CANONICAL_METHODS) * len(INDEPENDENT_REPEATS)
        ),
        "warmups_per_method": WARMUPS_PER_METHOD,
        "timed_steps_per_method": TIMED_STEPS_PER_METHOD,
        "output_root": str(OUTPUT_ROOT),
        "gpu_ids": list(GPU_IDS),
        "implementation_revision": "scalar_anchor_shared_relation_group_v3",
    }
    write_json_atomic(OUTPUT_ROOT / "suite_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

    for repeat_id in INDEPENDENT_REPEATS:
        launch_repeat(repeat_id)
        aggregate_results()

    rows = aggregate_results()
    expected_aggregate_rows = len(overnight_cases()) * len(CANONICAL_METHODS)
    if len(rows) != expected_aggregate_rows or any(
        int(row["independent_repeat_count"]) != len(INDEPENDENT_REPEATS)
        for row in rows
    ):
        raise RuntimeError(
            "Overnight workers returned without a complete three-repeat matrix; rerun this file to resume"
        )
    write_json_atomic(
        OUTPUT_ROOT / "complete.json",
        {
            "completed": True,
            **manifest,
            "aggregate_rows": len(rows),
        },
    )
    print(f"OVERNIGHT SUITE COMPLETE: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    if os.environ.get("SIGMOD_OVERNIGHT_WORKER") == "1":
        worker_main()
    else:
        coordinator_main()

