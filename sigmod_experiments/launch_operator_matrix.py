from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable SIGMOD operator experiment launcher")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--dataset", action="append", help="Only run named dataset; repeatable")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def result_is_complete(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(payload.get("records"))
    except (OSError, ValueError, TypeError):
        return False


def plans(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    matrix = config["matrix"]
    policies = list(matrix["policies"])
    block_edges = [int(x) for x in matrix["block_edges"]]
    chunks = [int(x) for x in matrix["chunks"]]
    blocking_chunks = {int(x) for x in matrix.get("blocking_chunks", [])}
    aggregations = list(matrix.get("aggregations", ["mean"]))
    hidden_dims = [int(x) for x in matrix.get("hidden_dims", [32])]
    for policy, aggregation, hidden_dim in itertools.product(
        policies, aggregations, hidden_dims
    ):
        policy_blocks = block_edges if policy == "superblock_greedy" else block_edges[:1]
        for block, chunk in itertools.product(policy_blocks, chunks):
            modes = ["blocking"] if chunk == 1 else ["async"]
            if chunk in blocking_chunks and "blocking" not in modes:
                modes.append("blocking")
            for mode in modes:
                yield {
                    "policy": policy,
                    "block_edges": block,
                    "chunks": chunk,
                    "collective_mode": mode,
                    "aggregation": aggregation,
                    "hidden_dim": hidden_dim,
                }


def slug(plan: Dict[str, Any]) -> str:
    return (
        f"p-{plan['policy']}__b-{plan['block_edges']}__c-{plan['chunks']}"
        f"__m-{plan['collective_mode']}__a-{plan['aggregation']}__h-{plan['hidden_dim']}"
    )


def write_status(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    output_root = Path(config["output_root"])
    selected = set(args.dataset or [])
    python = str(config.get("python", sys.executable))
    gpu_ids = [str(x) for x in config.get("gpu_ids", [0, 1, 2, 3])]
    port_base = int(config.get("master_port_base", 29610))
    common = config.get("benchmark", {})
    statuses: List[Dict[str, Any]] = []
    launched = 0

    for dataset in config["datasets"]:
        if selected and dataset["name"] not in selected:
            continue
        for world_size, plan in itertools.product(config["gpu_counts"], plans(config)):
            world_size = int(world_size)
            if world_size > len(gpu_ids):
                raise ValueError(
                    f"Requested {world_size} GPUs but gpu_ids contains only {gpu_ids}"
                )
            result = (
                output_root
                / dataset["name"]
                / f"gpus-{world_size}"
                / slug(plan)
                / "result.json"
            )
            if not args.rerun and result_is_complete(result):
                statuses.append({"result": str(result), "status": "skipped_complete"})
                continue
            if args.max_jobs is not None and launched >= args.max_jobs:
                if not args.dry_run:
                    write_status(output_root / "launcher_status.json", statuses)
                return

            command = [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={world_size}",
                str(HERE / "benchmark_operator.py"),
                "--dataset-json",
                json.dumps(dataset, ensure_ascii=False),
                "--output",
                str(result),
                "--policy",
                plan["policy"],
                "--block-edges",
                str(plan["block_edges"]),
                "--chunks",
                str(plan["chunks"]),
                "--collective-mode",
                plan["collective_mode"],
                "--aggregation",
                plan["aggregation"],
                "--hidden-dim",
                str(plan["hidden_dim"]),
                "--warmups",
                str(common.get("warmups", 2)),
                "--repeats",
                str(common.get("repeats", 5)),
                "--seed",
                str(common.get("seed", 20261003)),
            ]
            if bool(dataset.get("validate_correctness", False)):
                command.append("--validate-correctness")
            if "correctness_max_edges" in common:
                command.extend(
                    ["--correctness-max-edges", str(common["correctness_max_edges"])]
                )

            record: Dict[str, Any] = {
                "result": str(result),
                "dataset": dataset["name"],
                "world_size": world_size,
                "plan": plan,
                "status": "dry_run" if args.dry_run else "running",
                "command": command,
            }
            print(" ".join(command))
            if args.dry_run:
                statuses.append(record)
                launched += 1
                continue

            result.parent.mkdir(parents=True, exist_ok=True)
            log_path = result.parent / "run.log"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:world_size])
            environment["MASTER_PORT"] = str(port_base + (launched % 1000))
            started = time.time()
            with log_path.open("w", encoding="utf-8") as log_stream:
                completed = subprocess.run(
                    command,
                    cwd=str(HERE),
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
            write_status(output_root / "launcher_status.json", statuses)
            launched += 1

    if not args.dry_run:
        write_status(output_root / "launcher_status.json", statuses)


if __name__ == "__main__":
    main()

