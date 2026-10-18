from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from settings_no_parser import (
    BASE_OUTPUT_ROOT,
    BASE_SEED,
    EPOCHS,
    GPU_IDS,
    GRAPH_PATH,
    GRAPH_TAG,
    INDUSTRIAL_RUNNER,
    LOG_ROOT,
    METADATA_PATH,
    MULTIDATE_METADATA_PATH,
    OPERATOR_NO_PARSER,
    PROJECT_ROOT,
)


def load_metadata() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        raise RuntimeError(
            f"缺少 {METADATA_PATH}。请先运行 00_preflight_no_parser.py"
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_multidate_metadata() -> Dict[str, Dict[str, Any]]:
    if not MULTIDATE_METADATA_PATH.exists():
        raise RuntimeError(
            f"缺少 {MULTIDATE_METADATA_PATH}。请先运行 08_multidate_preflight_no_parser.py"
        )
    rows = json.loads(MULTIDATE_METADATA_PATH.read_text(encoding="utf-8"))
    return {str(row["graph_tag"]): row for row in rows}


def result_json_is_complete(path: Path, key: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(payload.get(key))
    except (OSError, ValueError, TypeError):
        return False


def run_logged(command, environment: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(str(value) for value in command), flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            [str(value) for value in command],
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    if completed.returncode != 0:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-120:])
        except OSError as exc:
            tail = f"无法读取失败日志: {exc}"
        print("\n========== FAILURE LOG TAIL (last 120 lines) ==========", flush=True)
        print(tail, flush=True)
        print("========== END FAILURE LOG TAIL ==========\n", flush=True)
        raise RuntimeError(
            f"任务失败，returncode={completed.returncode}，日志={log_path}"
        )
    print(f"DONE: {elapsed:.1f}s, log={log_path}", flush=True)


def industrial_experiment_name(
    graph_tag: str,
    mode_name: str,
    strategy: str,
    block_edges: int,
    repeat: int,
    seed: int,
    total_labeled_edges: int,
) -> str:
    blocks = max(1, math.ceil(total_labeled_edges / block_edges))
    return (
        f"forwardagg_{mode_name}_{strategy}_{graph_tag}_"
        f"K{blocks}_Q{block_edges}_R{repeat:02d}_ep{EPOCHS}_seed{seed}"
    )


def run_industrial_job(
    stage: str,
    world_size: int,
    strategy: str,
    block_edges: int,
    mode: Dict[str, Any],
    repeat: int,
) -> None:
    metadata = load_metadata()
    run_industrial_job_for_graph(
        stage=stage,
        graph_tag=GRAPH_TAG,
        graph_path=GRAPH_PATH,
        total_edges=int(metadata["observed_labeled_edges"]),
        world_size=world_size,
        strategy=strategy,
        block_edges=block_edges,
        mode=mode,
        repeat=repeat,
    )


def run_multidate_industrial_job(
    stage: str,
    graph_tag: str,
    world_size: int,
    strategy: str,
    block_edges: int,
    mode: Dict[str, Any],
    repeat: int,
) -> None:
    metadata = load_multidate_metadata()
    if graph_tag not in metadata:
        raise KeyError(f"多日期 metadata 中没有 {graph_tag}")
    row = metadata[graph_tag]
    run_industrial_job_for_graph(
        stage=stage,
        graph_tag=graph_tag,
        graph_path=Path(row["graph_path"]),
        total_edges=int(row["observed_labeled_edges"]),
        world_size=world_size,
        strategy=strategy,
        block_edges=block_edges,
        mode=mode,
        repeat=repeat,
    )


def run_industrial_job_for_graph(
    stage: str,
    graph_tag: str,
    graph_path: Path,
    total_edges: int,
    world_size: int,
    strategy: str,
    block_edges: int,
    mode: Dict[str, Any],
    repeat: int,
) -> None:
    if not INDUSTRIAL_RUNNER.exists():
        raise FileNotFoundError(
            f"缺少工业训练主脚本: {INDUSTRIAL_RUNNER}。"
            "请把原项目完整上传，或在 settings_no_parser.py 中设置 "
            "INDUSTRIAL_RUNNER_OVERRIDE。"
        )
    blocks = max(1, math.ceil(total_edges / block_edges))
    seed = BASE_SEED + repeat - 1
    job_root = BASE_OUTPUT_ROOT / "industrial" / stage / f"gpus-{world_size}"
    name = industrial_experiment_name(
        graph_tag, mode["name"], strategy, block_edges, repeat, seed, total_edges
    )
    result = job_root / name / "train_result_rank0.json"
    if result_json_is_complete(result, "mode"):
        print(f"SKIP complete: {result}", flush=True)
        return
    existing = next(
        (
            candidate
            for candidate in (BASE_OUTPUT_ROOT / "industrial").rglob(
                "train_result_rank0.json"
            )
            if candidate.parent.name == name
            and candidate.parent.parent.name == f"gpus-{world_size}"
            and result_json_is_complete(candidate, "mode")
        ),
        None,
    )
    if existing is not None:
        print(f"SKIP identical job already complete: {existing}", flush=True)
        return

    selected_gpu_ids = tuple(mode.get("gpu_ids", GPU_IDS[:world_size]))
    if len(selected_gpu_ids) != world_size:
        raise ValueError(
            f"mode.gpu_ids={selected_gpu_ids} 与 world_size={world_size} 不一致"
        )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(x) for x in selected_gpu_ids),
            # This experiment suite is GPU-only.  Bind every torchrun rank to
            # its assigned device before any lazy CUDA callback is executed,
            # and never silently fall back to CPU under an NCCL process group.
            "COMMOPT_FORCE_CUDA": "1",
            "COMMOPT_STRICT_DEVICE": "1",
            # CUDA_VISIBLE_DEVICES remaps the selected physical GPUs to the
            # process-local logical IDs 0..world_size-1.  Passing the physical
            # IDs here is wrong whenever the selected pool is not a prefix
            # such as (2, 3), and can also confuse CUDA-compatible runtimes
            # that expose logical devices only.
            "COMMOPT_CUDA_DEVICE_MAP": ",".join(
                str(x) for x in range(world_size)
            ),
            "COMMOPT_PHYSICAL_GPU_IDS": ",".join(
                str(x) for x in selected_gpu_ids
            ),
            "GRAPH_TAG": graph_tag,
            "CACHE_PATH": str(graph_path),
            "TOTAL_LABELED_EDGES": str(total_edges),
            "OUTPUT_ROOT": str(job_root),
            "PARTITION_STRATEGY": strategy,
            "TARGET_SUPER_BLOCK_SIZE": str(block_edges),
            "TARGET_NUM_BLOCKS": str(blocks),
            "NUM_EPOCHS": str(EPOCHS),
            "REPEAT_ID": str(repeat),
            "BASE_SEED": str(seed),
            "VARY_SEED_BY_REPEAT": "0",
            "FORWARD_AGG_EXPERIMENT_MODE": str(mode["name"]),
            "ENABLE_FORWARD_AGGREGATION": "1",
            "FORWARD_AGGREGATION_OPERATOR": str(mode.get("operator", "dense")),
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
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(INDUSTRIAL_RUNNER),
    ]
    log = LOG_ROOT / stage / f"g{world_size}__{name}.log"
    run_logged(command, environment, log)


def run_operator_job(
    stage: str,
    dataset: Dict[str, Any],
    world_size: int,
    policy: str,
    block_edges: int,
    chunks: int,
    collective_mode: str,
    hidden_dim: int = 32,
    warmups: int = 2,
    repeats: int = 5,
    validate: bool = False,
) -> None:
    slug = (
        f"p-{policy}__b-{block_edges}__c-{chunks}__m-{collective_mode}"
        f"__a-mean__h-{hidden_dim}"
    )
    result = (
        BASE_OUTPUT_ROOT
        / "operator"
        / stage
        / dataset["name"]
        / f"gpus-{world_size}"
        / slug
        / "result.json"
    )
    if result_json_is_complete(result, "records"):
        print(f"SKIP complete: {result}", flush=True)
        return

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(x) for x in GPU_IDS[:world_size]),
            "SIGMOD_DATASET_JSON": json.dumps(dataset, ensure_ascii=False),
            "SIGMOD_OUTPUT": str(result),
            "SIGMOD_POLICY": policy,
            "SIGMOD_BLOCK_EDGES": str(block_edges),
            "SIGMOD_CHUNKS": str(chunks),
            "SIGMOD_COLLECTIVE_MODE": collective_mode,
            "SIGMOD_AGGREGATION": "mean",
            "SIGMOD_HIDDEN_DIM": str(hidden_dim),
            "SIGMOD_WARMUPS": str(warmups),
            "SIGMOD_REPEATS": str(repeats),
            "SIGMOD_SEED": str(BASE_SEED),
            "SIGMOD_VALIDATE": "1" if validate else "0",
            "SIGMOD_CORRECTNESS_MAX_EDGES": "2000000",
        }
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(OPERATOR_NO_PARSER),
    ]
    log = LOG_ROOT / "operator" / stage / dataset["name"] / f"g{world_size}__{slug}.log"
    run_logged(command, environment, log)

