from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def critical_runtime(directory: Path, world_size: int) -> Dict[str, float]:
    rank_values: Dict[int, List[float]] = {}
    for rank in range(world_size):
        rows = [
            row
            for row in read_csv(directory / f"runtime_rank{rank}.csv")
            if row.get("event") == "train_epoch"
        ]
        rank_values[rank] = [number(row.get("elapsed_sec")) for row in rows]
    lengths = [len(values) for values in rank_values.values() if values]
    if not lengths:
        return {
            "epochs_observed": 0,
            "first_epoch_sec": 0.0,
            "steady_epoch_sec_mean": 0.0,
            "steady_epoch_sec_median": 0.0,
            "steady_epoch_sec_stdev": 0.0,
        }
    count = min(lengths)
    critical = [max(rank_values[rank][epoch] for rank in rank_values) for epoch in range(count)]
    steady = critical[1:] if len(critical) > 1 else critical
    return {
        "epochs_observed": count,
        "first_epoch_sec": critical[0],
        "steady_epoch_sec_mean": statistics.mean(steady),
        "steady_epoch_sec_median": statistics.median(steady),
        "steady_epoch_sec_stdev": statistics.stdev(steady) if len(steady) > 1 else 0.0,
    }


def partition_metrics(directory: Path, world_size: int) -> Dict[str, float]:
    rows: List[Dict[str, str]] = []
    for rank in range(world_size):
        rank_rows = read_csv(directory / f"partition_stage_timing_rank{rank}.csv")
        if rank_rows:
            rows.append(rank_rows[-1])
    if not rows:
        return {
            "prep_sec_max": 0.0,
            "num_superblocks": 0.0,
            "max_rank_edges": 0.0,
            "edge_imbalance_to_mean": 0.0,
        }
    rank_edges = [number(row.get("rank_selected_num_edges")) for row in rows]
    total_rank_edges = sum(rank_edges)
    return {
        "prep_sec_max": max(number(row.get("partition_time_with_save_sec")) for row in rows),
        "num_superblocks": max(number(row.get("num_superblocks")) for row in rows),
        "max_rank_edges": max(rank_edges),
        "edge_imbalance_to_mean": (
            max(rank_edges) / (total_rank_edges / len(rank_edges)) if total_rank_edges else 0.0
        ),
    }


def communication_metrics(directory: Path, world_size: int) -> Dict[str, float]:
    by_epoch_rank: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    fields = {
        "aggregation_comm_total_sec": "comm_total_sec",
        "aggregation_comm_exposed_sec": "comm_exposed_sec",
        "aggregation_comm_hidden_sec": "comm_hidden_sec",
        "local_message_compute_sec": "message_compute_sec",
        "forward_pipeline_total_sec": "forward_pipeline_sec",
        "dense_payload_bytes": "dense_payload_bytes",
        "compact_payload_bytes": "compact_payload_bytes",
    }
    for rank in range(world_size):
        for row in read_csv(directory / f"forward_agg_overlap_timing_rank{rank}.csv"):
            epoch = int(number(row.get("epoch"), -1))
            if epoch < 0:
                continue
            for source, target in fields.items():
                by_epoch_rank[epoch][rank][target] += number(row.get(source))
    epochs = sorted(by_epoch_rank)
    steady_epochs = epochs[1:] if len(epochs) > 1 else epochs
    result: Dict[str, float] = {}
    for target in fields.values():
        critical = [
            max(by_epoch_rank[epoch][rank].get(target, 0.0) for rank in by_epoch_rank[epoch])
            for epoch in steady_epochs
        ]
        result[f"{target}_mean"] = statistics.mean(critical) if critical else 0.0
    total = result.get("comm_total_sec_mean", 0.0)
    hidden = result.get("comm_hidden_sec_mean", 0.0)
    result["overlap_ratio_mean"] = hidden / total if total > 0 else 0.0
    dense_bytes = result.get("dense_payload_bytes_mean", 0.0)
    compact_bytes = result.get("compact_payload_bytes_mean", 0.0)
    result["payload_reduction_ratio_mean"] = (
        compact_bytes / dense_bytes if dense_bytes > 0 else 1.0
    )
    return result


def peak_memory(directory: Path) -> float:
    values = []
    for path in directory.glob("train_result_rank*.json"):
        try:
            values.append(number(json.loads(path.read_text(encoding="utf-8"))["peak_memory_mb"]))
        except (OSError, ValueError, KeyError):
            continue
    return max(values) if values else 0.0


def collect(result_path: Path, reuse_epochs: int) -> Dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    directory = result_path.parent
    world_size = int(payload.get("world_size", 1))
    runtime = critical_runtime(directory, world_size)
    partition = partition_metrics(directory, world_size)
    communication = communication_metrics(directory, world_size)
    steady = runtime["steady_epoch_sec_mean"]
    prep = partition["prep_sec_max"]
    return {
        "result_dir": str(directory),
        "graph": payload.get("graph_tag", payload.get("graph_date", "unknown")),
        "world_size": world_size,
        "strategy": payload.get("strategy", "unknown"),
        "block_edges": int(payload.get("target_super_block_size", 0)),
        "num_blocks_target": int(payload.get("target_num_blocks", 0)),
        "chunks": int(payload.get("aggregation_num_chunks", 1)),
        "mode": payload.get("mode", "unknown"),
        "overlap": bool(payload.get("enable_forward_agg_overlap", False)),
        "repeat": int(payload.get("repeat_id", 0)),
        "seed": int(payload.get("run_seed", 0)),
        "total_labeled_edges": int(payload.get("total_labeled_edges", 0)),
        "peak_memory_mb": peak_memory(directory),
        "estimated_total_sec_at_horizon": prep + reuse_epochs * steady,
        "reuse_epochs": reuse_epochs,
        **runtime,
        **partition,
        **communication,
    }


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No completed train_result_rank0.json files were found")
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect industrial SIGMOD runs")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-epochs", type=int, default=50)
    args = parser.parse_args()
    root = Path(args.root)
    rows = [collect(path, args.reuse_epochs) for path in root.rglob("train_result_rank0.json")]
    rows.sort(
        key=lambda row: (
            str(row["graph"]),
            int(row["world_size"]),
            str(row["strategy"]),
            int(row["block_edges"]),
            int(row["chunks"]),
            int(row["repeat"]),
        )
    )
    write_csv(Path(args.output), rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

