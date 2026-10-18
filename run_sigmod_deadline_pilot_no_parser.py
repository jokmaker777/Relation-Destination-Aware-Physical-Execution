#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compressed, no-network SIGMOD evidence package.

This script is intentionally a pilot rather than a substitute for public data
or multiple complete GNN models.  It produces controlled synthetic evidence for
partition baselines, ablations, scale/skew/relation sensitivity, 1/2/4-GPU
scaling, exactness, system metrics, and (when the 12-snapshot CSV exists)
leave-one-snapshot-out planner evaluation.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "sigmod_experiments"
SERVER_ROOT = EXPERIMENT_ROOT / "server_no_parser"
sys.path.insert(0, str(SERVER_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from collect_operator_results import flatten
from common_no_parser import run_operator_job
from settings_no_parser import BASE_OUTPUT_ROOT, BASE_SEED


STAGE = "deadline_pilot"
OPERATOR_ROOT = BASE_OUTPUT_ROOT / "operator" / STAGE
SUMMARY_ROOT = BASE_OUTPUT_ROOT / "deadline_pilot_summary"
MULTIDATE_MEASUREMENTS = BASE_OUTPUT_ROOT / "multidate_measurements.csv"
GRAPH_COUNTS = (1, 2, 4)
WARMUPS = 1
REPEATS = 3


def synthetic_dataset(
    name: str,
    nodes: int,
    edges: int,
    relations: int,
    skew: float,
) -> Dict[str, Any]:
    return {
        "name": name,
        "kind": "synthetic",
        "relation_selector": "all",
        "validate_correctness": True,
        "options": {
            "num_src": nodes,
            "num_dst": nodes,
            "num_edges": edges,
            "num_relations": relations,
            "skew": skew,
            "seed": BASE_SEED,
        },
    }


DATASETS: Tuple[Dict[str, Any], ...] = (
    synthetic_dataset("pilot-base", 100_000, 1_000_000, 4, 1.0),
    synthetic_dataset("pilot-uniform", 100_000, 1_000_000, 4, 0.0),
    synthetic_dataset("pilot-skew-high", 100_000, 1_000_000, 4, 2.0),
    synthetic_dataset("pilot-scale-small", 50_000, 500_000, 4, 1.0),
    synthetic_dataset("pilot-scale-large", 200_000, 2_000_000, 4, 1.0),
    synthetic_dataset("pilot-relations-2", 100_000, 1_000_000, 2, 1.0),
    synthetic_dataset("pilot-relations-8", 100_000, 1_000_000, 8, 1.0),
)

DATASET_BY_NAME = {value["name"]: value for value in DATASETS}
BASELINE_DATASET_NAMES = (
    "pilot-base",
    "pilot-skew-high",
    "pilot-relations-8",
)


def plan(
    label: str,
    policy: str,
    chunks: int,
    collective_mode: str,
    block_edges: int = 25_000,
    aggregation: str = "mean",
    hidden_dim: int = 32,
) -> Dict[str, Any]:
    return {
        "label": label,
        "policy": policy,
        "chunks": chunks,
        "collective_mode": collective_mode,
        "block_edges": block_edges,
        "aggregation": aggregation,
        "hidden_dim": hidden_dim,
    }


DIRECT = plan("direct_blocking", "contiguous", 1, "blocking")
OVERLAP_ONLY = plan("overlap_only", "contiguous", 4, "async")
LAYOUT_ONLY = plan("layout_only", "superblock_greedy", 1, "blocking")
FULL = plan("full", "superblock_greedy", 4, "async")

BASELINE_PLANS: Tuple[Dict[str, Any], ...] = (
    DIRECT,
    OVERLAP_ONLY,
    LAYOUT_ONLY,
    FULL,
    plan("round_robin", "round_robin", 4, "async"),
    plan("destination_hash", "destination_hash", 4, "async"),
    plan("destination_greedy", "destination_greedy", 4, "async"),
)


def assert_no_existing_gpu_jobs() -> None:
    if os.environ.get("ALLOW_BUSY_GPUS", "0") == "1":
        return
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return
    active = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode == 0 and active:
        raise RuntimeError(
            "Other GPU jobs are active. Wait for the multidate/async search to "
            "finish before starting this performance benchmark:\n  "
            + "\n  ".join(active)
        )


def execute(
    dataset: Mapping[str, Any],
    world_size: int,
    plan_config: Mapping[str, Any],
) -> None:
    run_operator_job(
        stage=STAGE,
        dataset=dict(dataset),
        world_size=world_size,
        policy=str(plan_config["policy"]),
        block_edges=int(plan_config["block_edges"]),
        chunks=int(plan_config["chunks"]),
        collective_mode=str(plan_config["collective_mode"]),
        aggregation=str(plan_config["aggregation"]),
        hidden_dim=int(plan_config["hidden_dim"]),
        warmups=WARMUPS,
        repeats=REPEATS,
        validate=True,
    )


def job_key(
    dataset: Mapping[str, Any], world_size: int, config: Mapping[str, Any]
) -> Tuple[Any, ...]:
    return (
        dataset["name"],
        world_size,
        config["policy"],
        config["block_edges"],
        config["chunks"],
        config["collective_mode"],
        config["aggregation"],
        config["hidden_dim"],
    )


def build_jobs() -> List[Tuple[Mapping[str, Any], int, Mapping[str, Any]]]:
    jobs: List[Tuple[Mapping[str, Any], int, Mapping[str, Any]]] = []

    # Complete baseline/ablation comparison on three representative workloads.
    for dataset_name in BASELINE_DATASET_NAMES:
        dataset = DATASET_BY_NAME[dataset_name]
        for config in BASELINE_PLANS:
            jobs.append((dataset, 4, config))

    # Scaling and controlled graph-factor sensitivity with direct vs full.
    for dataset in DATASETS:
        for world_size in GRAPH_COUNTS:
            for config in (DIRECT, FULL):
                jobs.append((dataset, world_size, config))

    # Message-passing operator generality pilot: exact relation-wise SUM.
    for dataset_name in BASELINE_DATASET_NAMES:
        jobs.append(
            (
                DATASET_BY_NAME[dataset_name],
                4,
                plan(
                    "full_sum",
                    "superblock_greedy",
                    4,
                    "async",
                    aggregation="sum",
                ),
            )
        )

    # Feature-width sensitivity on the base graph.
    for hidden_dim in (16, 64):
        jobs.append(
            (
                DATASET_BY_NAME["pilot-base"],
                4,
                plan(
                    f"full_h{hidden_dim}",
                    "superblock_greedy",
                    4,
                    "async",
                    hidden_dim=hidden_dim,
                ),
            )
        )

    unique: Dict[Tuple[Any, ...], Tuple[Mapping[str, Any], int, Mapping[str, Any]]] = {}
    for dataset, world_size, config in jobs:
        unique[job_key(dataset, world_size, config)] = (
            dataset,
            world_size,
            config,
        )
    return list(unique.values())


def run_jobs() -> None:
    jobs = build_jobs()
    failures: List[Dict[str, Any]] = []
    consecutive_failures = 0
    print(f"START DEADLINE PILOT: {len(jobs)} unique jobs", flush=True)
    for index, (dataset, world_size, config) in enumerate(jobs, 1):
        print(
            f"[{index}/{len(jobs)}] dataset={dataset['name']} GPUs={world_size} "
            f"plan={config['label']} aggregation={config['aggregation']} "
            f"hidden={config['hidden_dim']}",
            flush=True,
        )
        try:
            execute(dataset, world_size, config)
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            failures.append(
                {
                    "dataset": dataset["name"],
                    "world_size": world_size,
                    "plan": dict(config),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
            (SUMMARY_ROOT / "failures.json").write_text(
                json.dumps(failures, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if consecutive_failures >= 3:
                raise RuntimeError(
                    "Three consecutive jobs failed; stopping to avoid wasting GPU time"
                ) from exc
    if failures:
        print(f"PILOT FINISHED WITH {len(failures)} FAILURES", flush=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def label_for(row: Mapping[str, Any]) -> str:
    key = (
        row["policy"],
        int(row["chunks"]),
        row["collective_mode"],
        row["aggregation"],
        int(row["hidden_dim"]),
    )
    known = {
        ("contiguous", 1, "blocking", "mean", 32): "direct_blocking",
        ("contiguous", 4, "async", "mean", 32): "overlap_only",
        ("superblock_greedy", 1, "blocking", "mean", 32): "layout_only",
        ("superblock_greedy", 4, "async", "mean", 32): "full",
        ("round_robin", 4, "async", "mean", 32): "round_robin",
        ("destination_hash", 4, "async", "mean", 32): "destination_hash",
        ("destination_greedy", 4, "async", "mean", 32): "destination_greedy",
        ("superblock_greedy", 4, "async", "sum", 32): "full_sum",
        ("superblock_greedy", 4, "async", "mean", 16): "full_h16",
        ("superblock_greedy", 4, "async", "mean", 64): "full_h64",
    }
    return known.get(key, "unlabeled")


def collect_and_summarize() -> None:
    relation_rows: List[Dict[str, Any]] = []
    for result in sorted(OPERATOR_ROOT.rglob("result.json")):
        try:
            relation_rows.extend(flatten(result))
        except (OSError, ValueError, KeyError) as exc:
            print(f"SKIP INVALID RESULT {result}: {exc}", flush=True)
    if not relation_rows:
        raise RuntimeError(f"No complete operator results under {OPERATOR_ROOT}")
    for row in relation_rows:
        row["plan_label"] = label_for(row)
    write_csv(SUMMARY_ROOT / "synthetic_relation_measurements.csv", relation_rows)

    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in relation_rows:
        key = (
            row["dataset"],
            int(row["world_size"]),
            row["plan_label"],
            row["policy"],
            int(row["block_edges"]),
            int(row["chunks"]),
            row["collective_mode"],
            row["aggregation"],
            int(row["hidden_dim"]),
        )
        grouped[key].append(row)

    plan_rows: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        total_edges = sum(int(row["num_edges"]) for row in rows)
        operator_sec = sum(float(row["operator_sec_median"]) for row in rows)
        prep_sec = sum(float(row["prep_sec_max"]) for row in rows)
        validated_rows = [row for row in rows if str(row["validated"]).lower() == "true"]
        correct = bool(validated_rows) and all(
            str(row["correct"]).lower() == "true" for row in validated_rows
        )
        plan_rows.append(
            {
                "dataset": key[0],
                "world_size": key[1],
                "plan_label": key[2],
                "policy": key[3],
                "block_edges": key[4],
                "chunks": key[5],
                "collective_mode": key[6],
                "aggregation": key[7],
                "hidden_dim": key[8],
                "relation_count": len(rows),
                "total_edges": total_edges,
                "prep_sec_sum": prep_sec,
                "operator_sec_sum": operator_sec,
                "throughput_edges_per_sec": total_edges / max(operator_sec, 1e-12),
                "peak_memory_bytes": max(
                    int(row["peak_memory_bytes_rank0"] or 0) for row in rows
                ),
                "edge_imbalance_max": max(
                    float(row["edge_imbalance_to_mean"]) for row in rows
                ),
                "cross_gpu_target_groups_sum": sum(
                    int(row["cross_gpu_target_groups"]) for row in rows
                ),
                "exactness_passed": correct,
            }
        )

    lookup = {
        (row["dataset"], row["world_size"], row["plan_label"], row["aggregation"], row["hidden_dim"]): row
        for row in plan_rows
    }
    for row in plan_rows:
        baseline = lookup.get(
            (
                row["dataset"],
                row["world_size"],
                "direct_blocking",
                "mean",
                32,
            )
        )
        row["speedup_vs_direct"] = (
            float(baseline["operator_sec_sum"]) / float(row["operator_sec_sum"])
            if baseline is not None
            and row["aggregation"] == "mean"
            and int(row["hidden_dim"]) == 32
            else ""
        )
        if baseline is not None:
            saved_per_iteration = float(baseline["operator_sec_sum"]) - float(
                row["operator_sec_sum"]
            )
            extra_prep = float(row["prep_sec_sum"]) - float(
                baseline["prep_sec_sum"]
            )
            row["break_even_iterations"] = (
                max(0.0, extra_prep) / saved_per_iteration
                if saved_per_iteration > 0
                else ""
            )
        else:
            row["break_even_iterations"] = ""
    plan_rows.sort(
        key=lambda row: (
            row["dataset"],
            int(row["world_size"]),
            float(row["operator_sec_sum"]),
        )
    )
    write_csv(SUMMARY_ROOT / "synthetic_plan_measurements.csv", plan_rows)

    scaling_rows: List[Dict[str, Any]] = []
    for dataset in DATASET_BY_NAME:
        for label in ("direct_blocking", "full"):
            one = lookup.get((dataset, 1, label, "mean", 32))
            if one is None:
                continue
            for world_size in GRAPH_COUNTS:
                row = lookup.get((dataset, world_size, label, "mean", 32))
                if row is None:
                    continue
                scaling_rows.append(
                    {
                        "dataset": dataset,
                        "plan_label": label,
                        "world_size": world_size,
                        "operator_sec_sum": row["operator_sec_sum"],
                        "throughput_edges_per_sec": row["throughput_edges_per_sec"],
                        "speedup_vs_1gpu": float(one["operator_sec_sum"])
                        / float(row["operator_sec_sum"]),
                        "parallel_efficiency": (
                            float(one["operator_sec_sum"])
                            / float(row["operator_sec_sum"])
                        )
                        / world_size,
                        "peak_memory_bytes": row["peak_memory_bytes"],
                        "exactness_passed": row["exactness_passed"],
                    }
                )
    write_csv(SUMMARY_ROOT / "synthetic_scalability.csv", scaling_rows)

    four_gpu_primary = [
        row
        for row in plan_rows
        if int(row["world_size"]) == 4
        and row["dataset"] in BASELINE_DATASET_NAMES
        and row["aggregation"] == "mean"
        and int(row["hidden_dim"]) == 32
    ]
    by_label: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in four_gpu_primary:
        by_label[str(row["plan_label"])].append(row)
    cross_dataset = []
    for label, rows in by_label.items():
        speedups = [float(row["speedup_vs_direct"]) for row in rows]
        cross_dataset.append(
            {
                "plan_label": label,
                "dataset_count": len(rows),
                "mean_speedup_vs_direct": mean(speedups),
                "min_speedup_vs_direct": min(speedups),
                "max_speedup_vs_direct": max(speedups),
                "mean_break_even_iterations": mean(
                    float(row["break_even_iterations"])
                    for row in rows
                    if row["break_even_iterations"] != ""
                )
                if any(row["break_even_iterations"] != "" for row in rows)
                else "",
            }
        )
    cross_dataset.sort(
        key=lambda row: float(row["mean_speedup_vs_direct"]), reverse=True
    )
    write_csv(SUMMARY_ROOT / "synthetic_cross_dataset_ranking.csv", cross_dataset)
    print(f"WROTE SYNTHETIC SUMMARIES: {SUMMARY_ROOT}", flush=True)


def run_planner_loso_if_ready() -> None:
    if not MULTIDATE_MEASUREMENTS.exists():
        print(
            f"SKIP PLANNER LOSO: {MULTIDATE_MEASUREMENTS} does not exist. "
            "Run 10_collect_multidate_no_parser.py after the 72 jobs finish, "
            "then rerun this script; completed operator jobs will be skipped.",
            flush=True,
        )
        return
    planner = EXPERIMENT_ROOT / "joint_planner.py"
    summary_rows = []
    for horizon in (10, 25, 50, 100):
        output_dir = SUMMARY_ROOT / f"planner_loso_h{horizon}"
        command = [
            sys.executable,
            str(planner),
            "--input",
            str(MULTIDATE_MEASUREMENTS),
            "--output-dir",
            str(output_dir),
            "--reuse-epochs",
            str(horizon),
            "--random-budget",
            "3",
            "--seed",
            str(BASE_SEED),
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=str(EXPERIMENT_ROOT), check=False)
        planner_sec = time.perf_counter() - started
        if completed.returncode != 0:
            print(f"PLANNER LOSO FAILED horizon={horizon}", flush=True)
            continue
        summary_path = output_dir / "planner_summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["planner_wall_sec"] = planner_sec
        summary_rows.append(payload)
    if summary_rows:
        write_csv(SUMMARY_ROOT / "planner_loso_horizon_summary.csv", summary_rows)
        print("WROTE PLANNER LOSO SUMMARIES", flush=True)


def main() -> None:
    assert_no_existing_gpu_jobs()
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "compressed deadline pilot; not a substitute for public datasets",
        "datasets": DATASETS,
        "baseline_plans": BASELINE_PLANS,
        "gpu_counts": GRAPH_COUNTS,
        "warmups": WARMUPS,
        "repeats": REPEATS,
        "job_count": len(build_jobs()),
    }
    (SUMMARY_ROOT / "pilot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_jobs()
    collect_and_summarize()
    run_planner_loso_if_ready()
    print("SIGMOD DEADLINE PILOT COMPLETED", flush=True)


if __name__ == "__main__":
    main()

