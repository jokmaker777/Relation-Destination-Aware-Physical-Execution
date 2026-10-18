#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restartable 1/2/3/4-GPU scaling study for active-destination compaction.

Direct run; no argparse.  This wrapper deliberately reuses the already
validated exact forward/backward operator in
``run_active_destination_phase_boundary_4gpu_no_parser.py`` instead of
maintaining a second implementation.

Experiment matrix
-----------------
* GPUs: 1, 2, 3, 4
* active destination ratios: 10%, 25%, 50%, 75%, 90%, 100%
* methods: Direct+dense, Direct+compact
* independent torchrun launches: 3
* each cell: 3 warmups + 9 timed forward/backward steps

The script is restartable.  Completed method JSON files and completed launch
markers are reused.  Final outputs include raw and aggregate CSV files, a
scaling CSV, two paper-oriented plots, and a concise README.
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(
    "/path/to/artifact_workspace/sigmod2027_20241129/"
    "active_destination_scaling_1to4gpu"
)
GPU_IDS = (0, 1, 2, 3)
GPU_COUNTS = (1, 2, 3, 4)
INDEPENDENT_REPEATS = (1, 2, 3)
ACTIVE_RATIOS = (0.10, 0.25, 0.50, 0.75, 0.90, 1.00)
WARMUPS_PER_METHOD = 3
TIMED_STEPS_PER_METHOD = 9
BASE_SEED = 20261006
METHODS = ("direct_dense", "direct_compact")


def safe_name(value: Any) -> str:
    return str(value).replace(".", "p")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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


def read_complete_json(path: Path) -> Dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if value.get("completed") else None
    except Exception:
        return None


def scaling_cases() -> List[Dict[str, Any]]:
    rows = []
    for ratio in ACTIVE_RATIOS:
        rows.append(
            {
                "axis": "active_ratio_scaling",
                "axis_value": float(ratio),
                "total_edges": 1_000_000,
                "num_dst": 250_000,
                "num_relations": 4,
                "active_ratio": float(ratio),
                "skew": 1.0,
                "hidden_dim": 64,
                "case_id": (
                    f"active_scaling-{safe_name(ratio)}"
                    "__e1000000__n250000__r4"
                    f"__a{safe_name(ratio)}__s1p0__h64"
                ),
            }
        )
    return rows


def launch_root(world_size: int, repeat_id: int) -> Path:
    return OUTPUT_ROOT / f"gpus-{world_size}" / f"repeat_{repeat_id:02d}"


def worker_main() -> None:
    import run_active_destination_phase_boundary_4gpu_no_parser as phase

    world_size = int(os.environ["SIGMOD_SCALING_WORLD_SIZE"])
    repeat_id = int(os.environ["SIGMOD_SCALING_REPEAT"])
    run_seed = BASE_SEED + repeat_id * 100_003

    selected_cases = scaling_cases()
    random.Random(run_seed + 31).shuffle(selected_cases)
    method_order = list(METHODS)
    random.Random(run_seed + 17).shuffle(method_order)
    for row in selected_cases:
        row["independent_repeat"] = repeat_id
        row["run_seed"] = run_seed
        row["method_order"] = ",".join(method_order)
        row["implementation_revision"] = (
            "active_destination_scaling_exact_autograd_v1"
        )

    phase.OUTPUT_ROOT = launch_root(world_size, repeat_id)
    phase.WORLD_SIZE = world_size
    phase.BASE_SEED = run_seed
    phase.WARMUPS = WARMUPS_PER_METHOD
    phase.REPEATS = TIMED_STEPS_PER_METHOD
    phase.METHODS = tuple(method_order)
    phase.cases = lambda: [dict(row) for row in selected_cases]
    # Cross-launch aggregation and plots are generated once by this wrapper.
    phase.create_plots = lambda rows: None
    phase.create_report = lambda rows: None
    phase.run_worker()


def expected_method_rows() -> int:
    return len(scaling_cases()) * len(METHODS)


def launch_one(world_size: int, repeat_id: int) -> None:
    destination = launch_root(world_size, repeat_id)
    completion = destination / "scaling_launch_complete.json"
    existing = read_complete_json(completion)
    if existing and int(existing.get("result_rows", -1)) == expected_method_rows():
        print(
            f"SKIP complete launch: GPUs={world_size}, repeat={repeat_id}",
            flush=True,
        )
        return

    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(value) for value in GPU_IDS[:world_size]
            ),
            "SIGMOD_ACTIVE_SCALING_WORKER": "1",
            "SIGMOD_SCALING_WORLD_SIZE": str(world_size),
            "SIGMOD_SCALING_REPEAT": str(repeat_id),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(Path(__file__).resolve()),
    ]
    log_path = destination / "launcher.log"
    print(
        f"RUN GPUs={world_size} repeat={repeat_id}: {' '.join(command)}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as stream:
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
            log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[-160:]
        )
        print(tail, flush=True)
        raise RuntimeError(
            f"Scaling launch failed: GPUs={world_size}, repeat={repeat_id}, "
            f"log={log_path}"
        )

    rows = list(destination.glob("cases/*/*.json"))
    if len(rows) != expected_method_rows():
        raise RuntimeError(
            f"Expected {expected_method_rows()} method results, found "
            f"{len(rows)} under {destination}"
        )
    write_json_atomic(
        completion,
        {
            "completed": True,
            "world_size": world_size,
            "independent_repeat": repeat_id,
            "result_rows": len(rows),
            "output_root": str(destination),
        },
    )


def load_raw_rows() -> List[Dict[str, Any]]:
    rows = []
    for world_size in GPU_COUNTS:
        for repeat_id in INDEPENDENT_REPEATS:
            root = launch_root(world_size, repeat_id)
            for path in sorted(root.glob("cases/*/*.json")):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not value.get("completed") or value.get("method") not in METHODS:
                    continue
                value["independent_repeat"] = repeat_id
                value["world_size"] = world_size
                value["source_json"] = str(path)
                value.pop("step_samples_sec", None)
                rows.append(value)
    return rows


def sample_stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(raw_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            int(row["world_size"]),
            str(row["case_id"]),
            str(row["method"]),
        )
        grouped[key].append(row)

    output = []
    for (world_size, case_id, method), values in sorted(grouped.items()):
        step = [float(value["step_sec_median"]) for value in values]
        prep = [float(value["prep_sec_max"]) for value in values]
        memory = [float(value["peak_gpu_memory_mb"]) for value in values]
        exemplar = values[0]
        output.append(
            {
                "world_size": world_size,
                "case_id": case_id,
                "active_ratio": float(exemplar["active_ratio"]),
                "method": method,
                "independent_repeat_count": len(values),
                "step_sec_independent_median": statistics.median(step),
                "step_sec_independent_min": min(step),
                "step_sec_independent_max": max(step),
                "step_sec_independent_stdev": sample_stdev(step),
                "step_cv": sample_stdev(step) / max(statistics.mean(step), 1e-12),
                "prep_sec_median": statistics.median(prep),
                "peak_gpu_memory_mb_median": statistics.median(memory),
                "peak_gpu_memory_mb_max": max(memory),
                "collective_payload_bytes_forward": int(
                    exemplar["collective_payload_bytes_forward"]
                ),
                "payload_reduction_percent": float(
                    exemplar["payload_reduction_percent"]
                ),
                "forward_relative_l2_error_max": max(
                    float(value["forward_relative_l2_error"]) for value in values
                ),
                "gradient_relative_l2_error_max": max(
                    float(value["gradient_relative_l2_error"]) for value in values
                ),
                "run_seeds": ",".join(
                    str(value.get("run_seed", "")) for value in values
                ),
                "method_orders": ";".join(
                    str(value.get("method_order", "")) for value in values
                ),
            }
        )

    by_cell = {
        (int(row["world_size"]), float(row["active_ratio"]), str(row["method"])): row
        for row in output
    }
    for row in output:
        baseline = by_cell.get(
            (int(row["world_size"]), float(row["active_ratio"]), "direct_dense")
        )
        if baseline:
            row["compact_speedup_same_gpu_count"] = (
                float(baseline["step_sec_independent_median"])
                / max(float(row["step_sec_independent_median"]), 1e-12)
            )
    return output


def make_scaling_rows(
    aggregate: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    lookup = {
        (int(row["world_size"]), float(row["active_ratio"]), str(row["method"])): row
        for row in aggregate
    }
    rows = []
    for ratio in ACTIVE_RATIOS:
        for method in METHODS:
            baseline = lookup.get((1, float(ratio), method))
            if baseline is None:
                continue
            one_gpu = float(baseline["step_sec_independent_median"])
            for world_size in GPU_COUNTS:
                current = lookup.get((world_size, float(ratio), method))
                if current is None:
                    continue
                current_time = float(current["step_sec_independent_median"])
                speedup = one_gpu / max(current_time, 1e-12)
                rows.append(
                    {
                        "active_ratio": ratio,
                        "method": method,
                        "world_size": world_size,
                        "step_sec_independent_median": current_time,
                        "strong_scaling_speedup": speedup,
                        "strong_scaling_efficiency": speedup / world_size,
                        "compact_speedup_same_gpu_count": current.get(
                            "compact_speedup_same_gpu_count"
                        ),
                        "peak_gpu_memory_mb_median": current[
                            "peak_gpu_memory_mb_median"
                        ],
                    }
                )
    return rows


def create_plots(
    aggregate: Sequence[Mapping[str, Any]],
    scaling: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compact = [row for row in aggregate if row["method"] == "direct_compact"]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for world_size in GPU_COUNTS:
        values = sorted(
            [row for row in compact if int(row["world_size"]) == world_size],
            key=lambda row: float(row["active_ratio"]),
        )
        axis.plot(
            [100 * float(row["active_ratio"]) for row in values],
            [float(row["compact_speedup_same_gpu_count"]) for row in values],
            marker="o",
            linewidth=1.8,
            label=f"{world_size} GPU" + ("s" if world_size > 1 else ""),
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Active destinations (%)")
    axis.set_ylabel("Compact speedup over dense")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "compact_speedup_by_gpu_count.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    colors = {0.10: "#0072B2", 0.50: "#009E73", 0.90: "#D55E00"}
    for ratio in (0.10, 0.50, 0.90):
        values = sorted(
            [
                row
                for row in scaling
                if row["method"] == "direct_compact"
                and math.isclose(float(row["active_ratio"]), ratio)
            ],
            key=lambda row: int(row["world_size"]),
        )
        label = f"active={int(100 * ratio)}%"
        axes[0].plot(
            [int(row["world_size"]) for row in values],
            [float(row["strong_scaling_speedup"]) for row in values],
            marker="o",
            color=colors[ratio],
            label=label,
        )
        axes[1].plot(
            [int(row["world_size"]) for row in values],
            [100 * float(row["strong_scaling_efficiency"]) for row in values],
            marker="o",
            color=colors[ratio],
            label=label,
        )
    axes[0].plot(GPU_COUNTS, GPU_COUNTS, color="black", linestyle="--", label="ideal")
    axes[0].set_ylabel("Strong-scaling speedup")
    axes[1].axhline(100.0, color="black", linestyle="--")
    axes[1].set_ylabel("Scaling efficiency (%)")
    for axis in axes:
        axis.set_xlabel("Number of GPUs")
        axis.set_xticks(GPU_COUNTS)
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "compact_strong_scaling.png", dpi=220)
    plt.close(fig)


def create_readme(
    raw: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    scaling: Sequence[Mapping[str, Any]],
) -> None:
    compact = [row for row in aggregate if row["method"] == "direct_compact"]
    max_forward = max(float(row["forward_relative_l2_error_max"]) for row in aggregate)
    max_gradient = max(float(row["gradient_relative_l2_error_max"]) for row in aggregate)
    max_cv = max(float(row["step_cv"]) for row in aggregate)
    median_cv = statistics.median(float(row["step_cv"]) for row in aggregate)

    lines = [
        "# Active-destination compaction: 1/2/3/4-GPU scaling",
        "",
        f"- Raw completed method cells: {len(raw)}",
        f"- Aggregate cells: {len(aggregate)}",
        f"- Independent launches per cell: {min(int(row['independent_repeat_count']) for row in aggregate)}–{max(int(row['independent_repeat_count']) for row in aggregate)}",
        f"- Maximum forward relative L2 error: {max_forward:.3e}",
        f"- Maximum gradient relative L2 error: {max_gradient:.3e}",
        f"- Median/max timing CV: {100 * median_cv:.2f}% / {100 * max_cv:.2f}%",
        "",
        "## Compact speedup over dense",
        "",
        "| Active ratio | 1 GPU | 2 GPUs | 3 GPUs | 4 GPUs |",
        "|---:|---:|---:|---:|---:|",
    ]
    lookup = {
        (int(row["world_size"]), float(row["active_ratio"])): row
        for row in compact
    }
    for ratio in ACTIVE_RATIOS:
        speeds = []
        for world_size in GPU_COUNTS:
            row = lookup.get((world_size, float(ratio)))
            speeds.append(
                f"{float(row['compact_speedup_same_gpu_count']):.3f}×"
                if row
                else "missing"
            )
        lines.append(
            f"| {100 * ratio:.0f}% | " + " | ".join(speeds) + " |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `scaling_all_completed_runs.csv`",
            "- `scaling_aggregate.csv`",
            "- `strong_scaling.csv`",
            "- `compact_speedup_by_gpu_count.png`",
            "- `compact_strong_scaling.png`",
            "",
            "Timing includes local gather, index_add, exact NCCL all-reduce, "
            "mean aggregation, and autograd all-reduce backward.",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def collect_all() -> None:
    raw = load_raw_rows()
    expected = (
        len(GPU_COUNTS)
        * len(INDEPENDENT_REPEATS)
        * len(scaling_cases())
        * len(METHODS)
    )
    if len(raw) != expected:
        raise RuntimeError(f"Expected {expected} raw method cells, found {len(raw)}")
    aggregate = aggregate_rows(raw)
    scaling = make_scaling_rows(aggregate)
    write_csv(OUTPUT_ROOT / "scaling_all_completed_runs.csv", raw)
    write_csv(OUTPUT_ROOT / "scaling_aggregate.csv", aggregate)
    write_csv(OUTPUT_ROOT / "strong_scaling.csv", scaling)
    create_plots(aggregate, scaling)
    create_readme(raw, aggregate, scaling)
    write_json_atomic(
        OUTPUT_ROOT / "complete.json",
        {
            "completed": True,
            "raw_method_cells": len(raw),
            "aggregate_cells": len(aggregate),
            "strong_scaling_rows": len(scaling),
            "gpu_counts": list(GPU_COUNTS),
            "independent_repeats": list(INDEPENDENT_REPEATS),
            "active_ratios": list(ACTIVE_RATIOS),
        },
    )


def coordinator_main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("ACTIVE-DESTINATION COMPACTION 1/2/3/4-GPU SCALING", flush=True)
    print(f"output={OUTPUT_ROOT}", flush=True)
    print(
        f"launches={len(GPU_COUNTS) * len(INDEPENDENT_REPEATS)}, "
        f"raw method cells={len(GPU_COUNTS) * len(INDEPENDENT_REPEATS) * expected_method_rows()}",
        flush=True,
    )
    print("=" * 78, flush=True)
    for repeat_id in INDEPENDENT_REPEATS:
        for world_size in GPU_COUNTS:
            launch_one(world_size, repeat_id)
    collect_all()
    print(f"COMPLETE: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    if os.environ.get("SIGMOD_ACTIVE_SCALING_WORKER") == "1":
        worker_main()
    else:
        coordinator_main()

