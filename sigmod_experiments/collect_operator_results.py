from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def flatten(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for record in payload.get("records", []):
        plan = record["plan"]
        partition = record["partition"]
        correctness = record.get("correctness", {})
        hardware = record.get("hardware", {})
        rows.append(
            {
                "result": str(path),
                "dataset": record["dataset"],
                "dataset_kind": record["dataset_kind"],
                "relation": record["relation"],
                "world_size": record["world_size"],
                "policy": plan["policy"],
                "block_edges": plan["block_edges"],
                "chunks": plan["destination_ranges"],
                "collective_mode": plan["collective_mode"],
                "aggregation": plan["aggregation"],
                "hidden_dim": record["hidden_dim"],
                "num_src": record["num_src"],
                "num_dst": record["num_dst"],
                "num_edges": record["num_edges"],
                "prep_sec_max": record["prep_sec_max"],
                "operator_sec_median": record["operator_sec_median"],
                "operator_sec_mean": record["operator_sec_mean"],
                "operator_sec_stdev": record["operator_sec_stdev"],
                "throughput_edges_per_sec": record["throughput_edges_per_sec"],
                "collective_calls": record["collective_calls_per_repeat"],
                "collective_payload_bytes": record["collective_payload_bytes_per_repeat"],
                "peak_memory_bytes_rank0": record["peak_memory_bytes_rank0"],
                "max_rank_edges": partition["max_rank_edges"],
                "edge_imbalance_to_mean": partition["edge_imbalance_to_mean"],
                "num_blocks": partition["num_blocks"],
                "cross_gpu_target_groups": partition["cross_gpu_target_groups"],
                "mean_target_span": partition["mean_target_span"],
                "validated": correctness.get("validated", False),
                "correct": correctness.get("allclose_rtol_1e-4_atol_1e-5", ""),
                "max_abs_error": correctness.get("max_abs_error", ""),
                "gpu_name": hardware.get("gpu_name", "CPU"),
                "torch": hardware.get("torch", ""),
                "cuda_runtime": hardware.get("cuda_runtime", ""),
                "nccl": hardware.get("nccl", ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public/synthetic operator runs")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows: List[Dict[str, Any]] = []
    for path in Path(args.root).rglob("result.json"):
        try:
            rows.extend(flatten(path))
        except (OSError, ValueError, KeyError) as exc:
            print(f"skip {path}: {exc}")
    if not rows:
        raise RuntimeError("No complete operator result.json files were found")
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["relation"],
            row["world_size"],
            row["policy"],
            row["block_edges"],
            row["chunks"],
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

