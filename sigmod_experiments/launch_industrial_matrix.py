from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable industrial end-to-end launcher")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--sweep", action="append", help="Only run named sweep; repeatable")
    parser.add_argument("--graph", action="append", help="Only run named graph; repeatable")
    parser.add_argument("--gpu-count", action="append", type=int)
    parser.add_argument("--strategy", action="append")
    parser.add_argument("--mode", action="append")
    parser.add_argument("--block-edges", action="append", type=int)
    parser.add_argument("--repeat", action="append", type=int)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def experiment_name(
    mode: str,
    strategy: str,
    graph: str,
    blocks: int,
    block_edges: int,
    repeat: int,
    epochs: int,
    seed: int,
) -> str:
    return (
        f"forwardagg_{mode}_{strategy}_{safe_tag(graph)}_K{blocks}_Q{block_edges}_"
        f"R{repeat:02d}_ep{epochs}_seed{seed}"
    )


def job_iter(config: Dict[str, Any], selected: set[str]) -> Iterable[Dict[str, Any]]:
    graphs = {value["name"]: value for value in config["graphs"]}
    for sweep in config["sweeps"]:
        if selected and sweep["name"] not in selected:
            continue
        sweep_graphs = sweep.get("graphs", list(graphs))
        for graph_name in sweep_graphs:
            if graph_name not in graphs:
                raise KeyError(f"Unknown graph {graph_name!r} in sweep {sweep['name']!r}")
        product = itertools.product(
            sweep_graphs,
            sweep["gpu_counts"],
            sweep["strategies"],
            sweep["block_edges"],
            sweep["modes"],
            sweep["repeats"],
        )
        for graph_name, gpu_count, strategy, block_edges, mode, repeat in product:
            # Q has no effect for the edge baseline, so do not duplicate it.
            if strategy == "balanced_edge" and int(block_edges) != int(sweep["block_edges"][0]):
                continue
            yield {
                "sweep": sweep["name"],
                "graph": graphs[graph_name],
                "gpu_count": int(gpu_count),
                "strategy": strategy,
                "block_edges": int(block_edges),
                "mode": mode,
                "repeat": int(repeat),
                "epochs": int(sweep.get("epochs", config.get("epochs", 10))),
            }


def status_write(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    selected = set(args.sweep or [])
    output_root = Path(config["output_root"])
    source_script = Path(config.get("source_script", HERE.parent / "run_forward_aggregation_overlap_5day_seq_chunk10_repeat2_ep10_no_parser.py"))
    python = str(config.get("python", sys.executable))
    gpu_ids = [str(x) for x in config.get("gpu_ids", [0, 1, 2, 3])]
    base_seed = int(config.get("base_seed", 20261003))
    statuses: List[Dict[str, Any]] = []
    launched = 0

    if not source_script.exists():
        raise FileNotFoundError(source_script)

    for job in job_iter(config, selected):
        filters = (
            (args.graph, job["graph"]["name"]),
            (args.gpu_count, job["gpu_count"]),
            (args.strategy, job["strategy"]),
            (args.mode, job["mode"]["name"]),
            (args.block_edges, job["block_edges"]),
            (args.repeat, job["repeat"]),
        )
        if any(values and current not in values for values, current in filters):
            continue
        if args.max_jobs is not None and launched >= args.max_jobs:
            break
        graph = job["graph"]
        mode = job["mode"]
        total_edges = int(graph["total_labeled_edges"])
        blocks = max(1, math.ceil(total_edges / job["block_edges"]))
        seed = base_seed + job["repeat"] - 1
        job_root = output_root / job["sweep"] / f"gpus-{job['gpu_count']}"
        name = experiment_name(
            mode["name"],
            job["strategy"],
            graph["name"],
            blocks,
            job["block_edges"],
            job["repeat"],
            job["epochs"],
            seed,
        )
        result = job_root / name / "train_result_rank0.json"
        if result.exists() and not args.rerun:
            statuses.append({"result": str(result), "status": "skipped_complete", **job})
            continue

        command = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={job['gpu_count']}",
            str(source_script),
        ]
        record: Dict[str, Any] = {
            "result": str(result),
            "status": "dry_run" if args.dry_run else "running",
            "command": command,
            "sweep": job["sweep"],
            "graph": graph["name"],
            "gpu_count": job["gpu_count"],
            "strategy": job["strategy"],
            "block_edges": job["block_edges"],
            "mode": mode,
            "repeat": job["repeat"],
        }
        print(
            f"[{job['sweep']}] {graph['name']} {job['gpu_count']}GPU "
            f"{job['strategy']} Q={job['block_edges']} {mode['name']} R{job['repeat']}"
        )
        if args.dry_run:
            statuses.append(record)
            launched += 1
            continue

        job_root.mkdir(parents=True, exist_ok=True)
        log_path = job_root / f"{name}.launcher.log"
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids[: job["gpu_count"]]),
                "GRAPH_TAG": str(graph["name"]),
                "CACHE_PATH": str(graph["path"]),
                "TOTAL_LABELED_EDGES": str(total_edges),
                "OUTPUT_ROOT": str(job_root),
                "PARTITION_STRATEGY": str(job["strategy"]),
                "TARGET_SUPER_BLOCK_SIZE": str(job["block_edges"]),
                "TARGET_NUM_BLOCKS": str(blocks),
                "NUM_EPOCHS": str(job["epochs"]),
                "REPEAT_ID": str(job["repeat"]),
                "BASE_SEED": str(seed),
                "VARY_SEED_BY_REPEAT": "0",
                "FORWARD_AGG_EXPERIMENT_MODE": str(mode["name"]),
                "ENABLE_FORWARD_AGGREGATION": "1",
                "ENABLE_FORWARD_AGG_OVERLAP": "1" if mode["overlap"] else "0",
                "AGGREGATION_NUM_CHUNKS": str(mode["chunks"]),
                "AGGREGATION_ASYNC_OP": "1" if mode["overlap"] else "0",
                "AGGREGATION_DOUBLE_BUFFER": "1" if mode["overlap"] else "0",
                "AGGREGATION_CHUNK_POLICY": "destination_range",
                "ENABLE_FORWARD_AGG_TIMING": "1",
                "RUN_PARENT_EXPERIMENTS": "0",
                "COMMOPT_DDP_LAUNCHED": "1",
                "FORCE_SINGLE_PROCESS": "1",
            }
        )
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log_stream:
            completed = subprocess.run(
                command,
                cwd=str(source_script.parent),
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record.update(
            {
                "status": "complete" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "elapsed_sec": time.time() - started,
                "log": str(log_path),
            }
        )
        statuses.append(record)
        status_write(output_root / "launcher_status.json", statuses)
        launched += 1

    if not args.dry_run:
        status_write(output_root / "launcher_status.json", statuses)


if __name__ == "__main__":
    main()

