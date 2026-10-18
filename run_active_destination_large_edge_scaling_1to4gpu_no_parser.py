#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Large-edge 1/2/3/4-GPU strong-scaling experiment (no argparse).

This is the publication-oriented follow-up to the completed 1M-edge scaling
study.  It keeps destination count, relation count, hidden dimension, active
ratio, and collective payload fixed while increasing only the edge count.
Consequently, any improvement in strong scaling can be attributed to a higher
compute/communication ratio instead of a larger communication payload.

Matrix
------
* total edges: 4M, 8M, 16M
* active destinations: 10%, 50%, 90%
* GPUs: 1, 2, 3, 4
* methods: Direct+dense and Direct+compact
* independent torchrun launches: 3
* each method cell: 3 warmups + 9 timed forward/backward steps

The script is restartable and reuses the exact operator implementation from
``run_active_destination_phase_boundary_4gpu_no_parser.py`` through the
already-tested scaling launcher.  Run this file directly; there are no command
line arguments.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import run_active_destination_scaling_1to4gpu_no_parser as base


OUTPUT_ROOT = Path(
    "/path/to/artifact_workspace/sigmod2027_20241129/"
    "active_destination_large_edge_scaling_1to4gpu"
)
SMALL_SCALE_ROOT = Path(
    "/path/to/artifact_workspace/sigmod2027_20241129/"
    "active_destination_scaling_1to4gpu"
)
TOTAL_EDGES = (4_000_000, 8_000_000, 16_000_000)
ACTIVE_RATIOS = (0.10, 0.50, 0.90)
GPU_COUNTS = (1, 2, 3, 4)
INDEPENDENT_REPEATS = (1, 2, 3)
METHODS = ("direct_dense", "direct_compact")
BASE_SEED = 20261007


def safe_name(value: Any) -> str:
    return str(value).replace(".", "p")


def large_edge_cases() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for total_edges in TOTAL_EDGES:
        for ratio in ACTIVE_RATIOS:
            rows.append(
                {
                    "axis": "large_edge_strong_scaling",
                    "axis_value": int(total_edges),
                    "total_edges": int(total_edges),
                    "num_dst": 250_000,
                    "num_relations": 4,
                    "active_ratio": float(ratio),
                    "skew": 1.0,
                    "hidden_dim": 64,
                    "case_id": (
                        f"large_edges-{total_edges}"
                        f"__e{total_edges}__n250000__r4"
                        f"__a{safe_name(ratio)}__s1p0__h64"
                    ),
                }
            )
    return rows


def sample_stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(raw_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            int(row["world_size"]),
            int(row["total_edges"]),
            float(row["active_ratio"]),
            str(row["method"]),
        )
        grouped[key].append(row)

    output: List[Dict[str, Any]] = []
    for (world_size, total_edges, ratio, method), values in sorted(grouped.items()):
        step = [float(value["step_sec_median"]) for value in values]
        prep = [float(value["prep_sec_max"]) for value in values]
        memory = [float(value["peak_gpu_memory_mb"]) for value in values]
        exemplar = values[0]
        output.append(
            {
                "world_size": world_size,
                "total_edges": total_edges,
                "num_dst": int(exemplar["num_dst"]),
                "num_relations": int(exemplar["num_relations"]),
                "active_ratio": ratio,
                "hidden_dim": int(exemplar["hidden_dim"]),
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

    lookup = {
        (
            int(row["world_size"]),
            int(row["total_edges"]),
            float(row["active_ratio"]),
            str(row["method"]),
        ): row
        for row in output
    }
    for row in output:
        baseline = lookup.get(
            (
                int(row["world_size"]),
                int(row["total_edges"]),
                float(row["active_ratio"]),
                "direct_dense",
            )
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
        (
            int(row["world_size"]),
            int(row["total_edges"]),
            float(row["active_ratio"]),
            str(row["method"]),
        ): row
        for row in aggregate
    }
    rows: List[Dict[str, Any]] = []
    for total_edges in TOTAL_EDGES:
        for ratio in ACTIVE_RATIOS:
            for method in METHODS:
                baseline = lookup.get((1, total_edges, ratio, method))
                if baseline is None:
                    continue
                one_gpu = float(baseline["step_sec_independent_median"])
                for world_size in GPU_COUNTS:
                    current = lookup.get((world_size, total_edges, ratio, method))
                    if current is None:
                        continue
                    current_time = float(current["step_sec_independent_median"])
                    speedup = one_gpu / max(current_time, 1e-12)
                    rows.append(
                        {
                            "total_edges": total_edges,
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def combined_compact_scaling(
    new_scaling: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        dict(row) for row in new_scaling if row["method"] == "direct_compact"
    ]
    for row in read_csv(SMALL_SCALE_ROOT / "strong_scaling.csv"):
        try:
            ratio = float(row["active_ratio"])
            if row["method"] != "direct_compact" or ratio not in ACTIVE_RATIOS:
                continue
            rows.append(
                {
                    "total_edges": 1_000_000,
                    "active_ratio": ratio,
                    "method": "direct_compact",
                    "world_size": int(row["world_size"]),
                    "step_sec_independent_median": float(
                        row["step_sec_independent_median"]
                    ),
                    "strong_scaling_speedup": float(row["strong_scaling_speedup"]),
                    "strong_scaling_efficiency": float(
                        row["strong_scaling_efficiency"]
                    ),
                    "source": "previous_1m_edge_study",
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(
        rows,
        key=lambda row: (
            float(row["active_ratio"]),
            int(row["total_edges"]),
            int(row["world_size"]),
        ),
    )


def create_plots(
    aggregate: Sequence[Mapping[str, Any]],
    scaling: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = combined_compact_scaling(scaling)
    colors = {
        1_000_000: "#999999",
        4_000_000: "#0072B2",
        8_000_000: "#009E73",
        16_000_000: "#D55E00",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharex=True, sharey=True)
    for axis, ratio in zip(axes, ACTIVE_RATIOS):
        for total_edges in (1_000_000,) + TOTAL_EDGES:
            values = sorted(
                [
                    row
                    for row in combined
                    if int(row["total_edges"]) == total_edges
                    and math.isclose(float(row["active_ratio"]), ratio)
                ],
                key=lambda row: int(row["world_size"]),
            )
            if not values:
                continue
            axis.plot(
                [int(row["world_size"]) for row in values],
                [float(row["strong_scaling_speedup"]) for row in values],
                marker="o",
                linewidth=1.8,
                color=colors[total_edges],
                label=f"{total_edges // 1_000_000}M edges",
            )
        axis.plot(GPU_COUNTS, GPU_COUNTS, color="black", linestyle="--", label="ideal")
        axis.set_title(f"Active destinations: {100 * ratio:.0f}%")
        axis.set_xlabel("Number of GPUs")
        axis.set_xticks(GPU_COUNTS)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Strong-scaling speedup")
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "large_edge_strong_scaling.png", dpi=220)
    plt.close(fig)

    compact = [row for row in aggregate if row["method"] == "direct_compact"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharey=True)
    for axis, ratio in zip(axes, ACTIVE_RATIOS):
        for world_size in GPU_COUNTS:
            values = sorted(
                [
                    row
                    for row in compact
                    if int(row["world_size"]) == world_size
                    and math.isclose(float(row["active_ratio"]), ratio)
                ],
                key=lambda row: int(row["total_edges"]),
            )
            axis.plot(
                [int(row["total_edges"]) / 1_000_000 for row in values],
                [float(row["compact_speedup_same_gpu_count"]) for row in values],
                marker="o",
                label=f"{world_size} GPU" + ("s" if world_size > 1 else ""),
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(f"Active destinations: {100 * ratio:.0f}%")
        axis.set_xlabel("Total edges (millions)")
        axis.set_xticks([4, 8, 16])
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Compact speedup over dense")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "compact_gain_by_edge_scale.png", dpi=220)
    plt.close(fig)


def create_readme(
    raw: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    scaling: Sequence[Mapping[str, Any]],
) -> None:
    max_forward = max(float(row["forward_relative_l2_error_max"]) for row in aggregate)
    max_gradient = max(float(row["gradient_relative_l2_error_max"]) for row in aggregate)
    timing_cvs = [float(row["step_cv"]) for row in aggregate]
    four_gpu = [
        row
        for row in scaling
        if row["method"] == "direct_compact" and int(row["world_size"]) == 4
    ]
    lines = [
        "# Large-edge active-destination strong scaling",
        "",
        f"- Raw method cells: {len(raw)}",
        f"- Aggregate cells: {len(aggregate)}",
        f"- Maximum forward relative L2 error: {max_forward:.3e}",
        f"- Maximum gradient relative L2 error: {max_gradient:.3e}",
        f"- Median/max timing CV: {100 * statistics.median(timing_cvs):.2f}% / {100 * max(timing_cvs):.2f}%",
        "",
        "The design fixes destinations=250,000, relations=4, hidden=64 and",
        "changes only the edge count.  The exact compact collective payload is",
        "therefore constant within each active-ratio panel.",
        "",
        "## Four-GPU compact strong scaling",
        "",
        "| Edges | Active destinations | Speedup | Efficiency |",
        "|---:|---:|---:|---:|",
    ]
    for row in sorted(
        four_gpu,
        key=lambda value: (int(value["total_edges"]), float(value["active_ratio"])),
    ):
        lines.append(
            f"| {int(row['total_edges']) / 1_000_000:.0f}M | "
            f"{100 * float(row['active_ratio']):.0f}% | "
            f"{float(row['strong_scaling_speedup']):.3f}x | "
            f"{100 * float(row['strong_scaling_efficiency']):.1f}% |"
        )
    positive = sum(float(row["strong_scaling_speedup"]) > 1.0 for row in four_gpu)
    lines.extend(
        [
            "",
            f"Positive four-GPU scaling cells: {positive}/{len(four_gpu)}.",
            "",
            "## Files",
            "",
            "- `large_edge_all_completed_runs.csv`",
            "- `large_edge_aggregate.csv`",
            "- `large_edge_strong_scaling.csv`",
            "- `combined_1m_to_16m_compact_scaling.csv`",
            "- `large_edge_strong_scaling.png`",
            "- `compact_gain_by_edge_scale.png`",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_all() -> None:
    raw = base.load_raw_rows()
    expected = (
        len(GPU_COUNTS)
        * len(INDEPENDENT_REPEATS)
        * len(large_edge_cases())
        * len(METHODS)
    )
    if len(raw) != expected:
        raise RuntimeError(f"Expected {expected} raw method cells, found {len(raw)}")
    aggregate = aggregate_rows(raw)
    scaling = make_scaling_rows(aggregate)
    combined = combined_compact_scaling(scaling)
    base.write_csv(OUTPUT_ROOT / "large_edge_all_completed_runs.csv", raw)
    base.write_csv(OUTPUT_ROOT / "large_edge_aggregate.csv", aggregate)
    base.write_csv(OUTPUT_ROOT / "large_edge_strong_scaling.csv", scaling)
    base.write_csv(OUTPUT_ROOT / "combined_1m_to_16m_compact_scaling.csv", combined)
    create_plots(aggregate, scaling)
    create_readme(raw, aggregate, scaling)
    base.write_json_atomic(
        OUTPUT_ROOT / "complete.json",
        {
            "completed": True,
            "raw_method_cells": len(raw),
            "aggregate_cells": len(aggregate),
            "strong_scaling_rows": len(scaling),
            "gpu_counts": list(GPU_COUNTS),
            "independent_repeats": list(INDEPENDENT_REPEATS),
            "total_edges": list(TOTAL_EDGES),
            "active_ratios": list(ACTIVE_RATIOS),
            "fixed_num_dst": 250_000,
            "fixed_num_relations": 4,
            "fixed_hidden_dim": 64,
        },
    )


def configure_base() -> None:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.GPU_COUNTS = GPU_COUNTS
    base.INDEPENDENT_REPEATS = INDEPENDENT_REPEATS
    base.ACTIVE_RATIOS = ACTIVE_RATIOS
    base.BASE_SEED = BASE_SEED
    base.METHODS = METHODS
    base.scaling_cases = large_edge_cases
    base.aggregate_rows = aggregate_rows
    base.make_scaling_rows = make_scaling_rows
    base.create_plots = create_plots
    base.create_readme = create_readme
    base.collect_all = collect_all


def launch_one(world_size: int, repeat_id: int) -> None:
    """Launch this wrapper, not the imported 1M-edge launcher.

    The base launcher's ``Path(__file__)`` belongs to the imported module and
    would therefore start the old 1M-edge entry point in a fresh interpreter.
    Keeping the launch code here makes the child import this file, reapply the
    large-edge configuration, and enter ``base.worker_main()`` correctly.
    """
    destination = base.launch_root(world_size, repeat_id)
    completion = destination / "scaling_launch_complete.json"
    existing = base.read_complete_json(completion)
    if existing and int(existing.get("result_rows", -1)) == base.expected_method_rows():
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
                str(value) for value in base.GPU_IDS[:world_size]
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
            cwd=str(Path(__file__).resolve().parent),
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
            f"Large-edge launch failed: GPUs={world_size}, repeat={repeat_id}, "
            f"log={log_path}"
        )

    rows = list(destination.glob("cases/*/*.json"))
    if len(rows) != base.expected_method_rows():
        raise RuntimeError(
            f"Expected {base.expected_method_rows()} method results, found "
            f"{len(rows)} under {destination}"
        )
    base.write_json_atomic(
        completion,
        {
            "completed": True,
            "world_size": world_size,
            "independent_repeat": repeat_id,
            "result_rows": len(rows),
            "output_root": str(destination),
        },
    )


def coordinator_main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    launches = len(GPU_COUNTS) * len(INDEPENDENT_REPEATS)
    raw_cells = launches * len(large_edge_cases()) * len(METHODS)
    print("=" * 78, flush=True)
    print("LARGE-EDGE ACTIVE-DESTINATION 1/2/3/4-GPU STRONG SCALING", flush=True)
    print(f"output={OUTPUT_ROOT}", flush=True)
    print(f"launches={launches}, raw method cells={raw_cells}", flush=True)
    print(
        "fixed: destinations=250000, relations=4, hidden=64; "
        f"edges={TOTAL_EDGES}",
        flush=True,
    )
    print("=" * 78, flush=True)
    for repeat_id in INDEPENDENT_REPEATS:
        for world_size in GPU_COUNTS:
            launch_one(world_size, repeat_id)
    collect_all()
    print(f"COMPLETE: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    configure_base()
    if os.environ.get("SIGMOD_ACTIVE_SCALING_WORKER") == "1":
        base.worker_main()
    else:
        coordinator_main()

