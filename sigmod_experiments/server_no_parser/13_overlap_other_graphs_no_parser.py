#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused no-argument communication/compute-overlap experiments.

Run on the GPU server after run_collective_safety_gate_no_parser.py passes.
The script first runs a small exactness gate, then two communication-heavy
synthetic graphs, and finally any uploaded HGB ACM/DBLP/IMDB graphs.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from collect_operator_results import flatten  # noqa: E402
from common_no_parser import run_operator_job  # noqa: E402
from settings_no_parser import BASE_OUTPUT_ROOT, BASE_SEED, PUBLIC_DATA_ROOT  # noqa: E402
from sigmod_exp.datasets import DatasetSpec, load_graph, summarize_graph  # noqa: E402


GPU_COUNTS = (2, 4)
POLICY = "superblock_greedy"
BLOCK_EDGES = 25_000
PLANS: Tuple[Tuple[int, str], ...] = (
    (1, "blocking"),
    (4, "blocking"),
    (2, "async"),
    (4, "async"),
)

CORRECTNESS_DATASET: Dict[str, Any] = {
    "name": "synthetic-overlap-correctness",
    "kind": "synthetic",
    "relation_selector": "all",
    "options": {
        "num_src": 50_000,
        "num_dst": 50_000,
        "num_edges": 500_000,
        "num_relations": 3,
        "skew": 1.5,
        "seed": BASE_SEED,
    },
}

SYNTHETIC_DATASETS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "synthetic-overlap-uniform-h128",
        "kind": "synthetic",
        "relation_selector": "all",
        "options": {
            "num_src": 250_000,
            "num_dst": 250_000,
            "num_edges": 4_000_000,
            "num_relations": 3,
            "skew": 0.0,
            "seed": BASE_SEED,
        },
    },
    {
        "name": "synthetic-overlap-skewed-h128",
        "kind": "synthetic",
        "relation_selector": "all",
        "options": {
            "num_src": 250_000,
            "num_dst": 250_000,
            "num_edges": 4_000_000,
            "num_relations": 3,
            "skew": 2.0,
            "seed": BASE_SEED,
        },
    },
)

PUBLIC_DATASETS: Tuple[Dict[str, Any], ...] = tuple(
    {
        "name": f"hgb-{name}",
        "kind": f"hgb-{name}",
        "root": str(PUBLIC_DATA_ROOT / "hgb"),
        "relation_selector": "topk:3",
    }
    for name in ("acm", "dblp", "imdb")
)


def result_path(
    stage: str,
    dataset: Mapping[str, Any],
    world_size: int,
    chunks: int,
    mode: str,
    hidden_dim: int,
) -> Path:
    slug = (
        f"p-{POLICY}__b-{BLOCK_EDGES}__c-{chunks}__m-{mode}"
        f"__a-mean__h-{hidden_dim}"
    )
    return (
        BASE_OUTPUT_ROOT
        / "operator"
        / stage
        / str(dataset["name"])
        / f"gpus-{world_size}"
        / slug
        / "result.json"
    )


def check_correctness(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failed = []
    for row in payload.get("records", []):
        correctness = row.get("correctness", {})
        if not correctness.get("validated", False):
            failed.append(f"{row.get('relation')}: not validated")
        elif not correctness.get("allclose_rtol_1e-4_atol_1e-5", False):
            failed.append(
                f"{row.get('relation')}: max_abs_error="
                f"{correctness.get('max_abs_error')}"
            )
    if failed:
        raise RuntimeError(f"Correctness gate failed for {path}: {failed}")


def run_plan_matrix(
    stage: str,
    datasets: Iterable[Mapping[str, Any]],
    hidden_dim: int,
    warmups: int,
    repeats: int,
    validate: bool,
) -> None:
    for dataset in datasets:
        for world_size in GPU_COUNTS:
            for chunks, mode in PLANS:
                run_operator_job(
                    stage=stage,
                    dataset=dict(dataset),
                    world_size=world_size,
                    policy=POLICY,
                    block_edges=BLOCK_EDGES,
                    chunks=chunks,
                    collective_mode=mode,
                    hidden_dim=hidden_dim,
                    warmups=warmups,
                    repeats=repeats,
                    validate=validate,
                )


def write_focused_summary(stages: Iterable[str]) -> Path:
    rows: List[Dict[str, Any]] = []
    for stage in stages:
        root = BASE_OUTPUT_ROOT / "operator" / stage
        for path in root.rglob("result.json") if root.exists() else []:
            rows.extend(flatten(path))

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["relation"],
            int(row["world_size"]),
            int(row["hidden_dim"]),
        )
        grouped[key].append(row)

    summary: List[Dict[str, Any]] = []
    for key, values in grouped.items():
        lookup = {
            (int(row["chunks"]), str(row["collective_mode"])): row
            for row in values
        }
        blocking_c1 = lookup.get((1, "blocking"))
        blocking_c4 = lookup.get((4, "blocking"))
        async_c2 = lookup.get((2, "async"))
        async_c4 = lookup.get((4, "async"))
        if not all((blocking_c1, blocking_c4, async_c2, async_c4)):
            continue

        def sec(row: Mapping[str, Any]) -> float:
            return float(row["operator_sec_median"])

        b1, b4, a2, a4 = map(sec, (blocking_c1, blocking_c4, async_c2, async_c4))
        summary.append(
            {
                "dataset": key[0],
                "relation": key[1],
                "world_size": key[2],
                "hidden_dim": key[3],
                "num_edges": int(blocking_c1["num_edges"]),
                "num_dst": int(blocking_c1["num_dst"]),
                "blocking_c1_sec": b1,
                "blocking_c4_sec": b4,
                "async_c2_sec": a2,
                "async_c4_sec": a4,
                # c4/c4 isolates async overlap from the cost of chunking.
                "async_c4_speedup_vs_blocking_c4": b4 / a4 if a4 > 0 else math.nan,
                "async_c4_speedup_vs_blocking_c1": b1 / a4 if a4 > 0 else math.nan,
                "async_c2_speedup_vs_blocking_c1": b1 / a2 if a2 > 0 else math.nan,
                "async_c4_stdev_sec": float(async_c4["operator_sec_stdev"]),
                "correct": async_c4.get("correct", ""),
                "gpu_name": async_c4.get("gpu_name", ""),
            }
        )

    summary.sort(
        key=lambda row: (
            str(row["dataset"]),
            int(row["world_size"]),
            str(row["relation"]),
        )
    )
    output = BASE_OUTPUT_ROOT / "overlap_other_graphs_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not summary:
        raise RuntimeError("No complete four-plan groups were found for the summary")
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return output


def main() -> None:
    print("STEP 1/3: exactness gate on 2 and 4 GPUs", flush=True)
    for world_size in GPU_COUNTS:
        run_operator_job(
            stage="overlap_focused_correctness",
            dataset=CORRECTNESS_DATASET,
            world_size=world_size,
            policy=POLICY,
            block_edges=BLOCK_EDGES,
            chunks=4,
            collective_mode="async",
            hidden_dim=32,
            warmups=1,
            repeats=3,
            validate=True,
        )
        check_correctness(
            result_path(
                "overlap_focused_correctness",
                CORRECTNESS_DATASET,
                world_size,
                4,
                "async",
                32,
            )
        )

    print("STEP 2/3: communication-heavy synthetic graphs", flush=True)
    run_plan_matrix(
        stage="overlap_focused_synthetic",
        datasets=SYNTHETIC_DATASETS,
        hidden_dim=128,
        warmups=3,
        repeats=10,
        validate=False,
    )

    print("STEP 3/3: uploaded HGB public graphs", flush=True)
    available_public: List[Mapping[str, Any]] = []
    skipped_public: List[Dict[str, str]] = []
    for dataset in PUBLIC_DATASETS:
        try:
            graph = load_graph(DatasetSpec.from_dict(dataset))
            graph_summary = summarize_graph(graph)
            print(
                f"READY {dataset['name']}: nodes={graph_summary['num_nodes']:,}, "
                f"edges={graph_summary['num_edges']:,}, "
                f"relations={graph_summary['num_edge_types']}",
                flush=True,
            )
            available_public.append(dataset)
        except Exception as exc:
            skipped_public.append(
                {
                    "dataset": str(dataset["name"]),
                    "root": str(dataset.get("root", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"SKIP {dataset['name']}: {type(exc).__name__}: {exc}", flush=True)

    if available_public:
        run_plan_matrix(
            stage="overlap_focused_public",
            datasets=available_public,
            hidden_dim=128,
            warmups=3,
            repeats=10,
            validate=True,
        )
    skip_path = BASE_OUTPUT_ROOT / "overlap_other_graphs_missing_public.json"
    skip_path.write_text(
        json.dumps(skipped_public, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = write_focused_summary(
        ("overlap_focused_synthetic", "overlap_focused_public")
    )
    print(f"FOCUSED OVERLAP EXPERIMENTS COMPLETED: {summary}", flush=True)


if __name__ == "__main__":
    main()

