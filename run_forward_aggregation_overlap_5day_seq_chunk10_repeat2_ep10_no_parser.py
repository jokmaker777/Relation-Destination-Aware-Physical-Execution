#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_forward_aggregation_overlap_5day_seq_chunk10_repeat2_ep10_no_parser.py

目的：
对 5 个日期图依次执行完整的 forward aggregation 模式扫描：

  20241213, 20241214, 20241215, 20241217, 20241218

每个日期的执行顺序：
  repeat 1:
    sequential_aggregation
    async_chunk_2
    async_chunk_3
    ...
    async_chunk_10

  repeat 2:
    sequential_aggregation
    async_chunk_2
    async_chunk_3
    ...
    async_chunk_10

完成同一日期的两次完整扫描后，再进入下一日期。

实验规模：
  - 5 个日期；
  - 每个日期重复 2 次；
  - 每次重复包含 sequential + async chunk 2–10，共 10 种模式；
  - 每种模式训练 10 epochs；
  - 每个“日期 × repeat × mode”保存到独立结果目录；
  - 总任务数为 5 × 2 × 10 = 100。

说明：
  - sequential_aggregation 默认使用 4 个 blocking destination chunks，
    用于与 async_chunk_4 进行直接 overlap 对比；
  - 同一个 repeat 下的不同模式使用相同 seed，便于进行配对比较；
  - repeat 1 和 repeat 2 默认使用不同 seed；
  - parent_summary.json 和 parent_summary.csv 会持续记录全部任务状态。

运行前：
  cp communication_optimized_training_manager_forward_agg_overlap_no_parser.py \
     /path/to/artifact_code/end_to_end_prunning/models/communication_optimized_training_manager_forward_agg_overlap_no_parser.py

  cp run_forward_aggregation_overlap_5day_seq_chunk10_repeat2_ep10_no_parser.py \
     /path/to/artifact_code/end_to_end_prunning/run_forward_aggregation_overlap_5day_seq_chunk10_repeat2_ep10_no_parser.py

运行：
  python /path/to/artifact_code/run_forward_aggregation_overlap_5day_seq_chunk10_repeat2_ep10_no_parser.py
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch_geometric.data import HeteroData

from end_to_end_prunning.config.model_config import (
    ModelArchitectureConfig,
    TrainingConfig,
    create_preset_config,
)
from end_to_end_prunning.models.communication_optimized_training_manager_forward_agg_overlap_no_parser import (
    CommunicationOptimizedTrainingManager,
)
from end_to_end_prunning.models.gcn_model import SupplyChainGCN


# ============================================================
# Fixed experiment configuration
# ============================================================

# Child-process defaults. Parent runs override these values for every date/repeat.
GRAPH_TAG = os.environ.get("GRAPH_TAG", "20241218")
CACHE_PATH = os.environ.get(
    "CACHE_PATH",
    r"/path/to/artifact_workspace/cache/graph_20241218.pkl",
)
# Preserve the original 20241218 default from the uploaded script.
TOTAL_LABELED_EDGES = int(os.environ.get("TOTAL_LABELED_EDGES", "6409021"))

OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT",
    r"/path/to/artifact_workspace/result_forward_aggregation_overlap_q25k_5day_seq_chunk10_repeat2_ep10",
)

PARTITION_STRATEGY = os.environ.get("PARTITION_STRATEGY", "superblock_vector_balance")
TARGET_SUPER_BLOCK_SIZE = int(os.environ.get("TARGET_SUPER_BLOCK_SIZE", "25000"))
TARGET_NUM_BLOCKS = int(os.environ.get("TARGET_NUM_BLOCKS", "257"))

NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "10"))
REPEAT_ID = int(os.environ.get("REPEAT_ID", "1"))
BASE_SEED = int(os.environ.get("BASE_SEED", "20260228"))
VARY_SEED_BY_REPEAT = os.environ.get("VARY_SEED_BY_REPEAT", "1") == "1"

FORWARD_AGG_EXPERIMENT_MODE = os.environ.get(
    "FORWARD_AGG_EXPERIMENT_MODE", "sequential_aggregation"
)
STEADY_EXPERIMENT_MODE = FORWARD_AGG_EXPERIMENT_MODE

ENABLE_FORWARD_AGGREGATION = os.environ.get("ENABLE_FORWARD_AGGREGATION", "1") == "1"
FORWARD_AGGREGATION_OPERATOR = os.environ.get(
    "FORWARD_AGGREGATION_OPERATOR", "dense"
).strip().lower()
ENABLE_FORWARD_AGG_OVERLAP = os.environ.get("ENABLE_FORWARD_AGG_OVERLAP", "0") == "1"
AGGREGATION_NUM_CHUNKS = int(os.environ.get("AGGREGATION_NUM_CHUNKS", "4"))
AGGREGATION_ASYNC_OP = os.environ.get("AGGREGATION_ASYNC_OP", "0") == "1"
AGGREGATION_DOUBLE_BUFFER = os.environ.get("AGGREGATION_DOUBLE_BUFFER", "0") == "1"
AGGREGATION_CHUNK_POLICY = os.environ.get(
    "AGGREGATION_CHUNK_POLICY", "destination_range"
)
AGGREGATION_CHUNK_ORDER = os.environ.get("AGGREGATION_CHUNK_ORDER", "natural")
AGGREGATION_MAX_INFLIGHT = int(os.environ.get("AGGREGATION_MAX_INFLIGHT", "1"))
AGGREGATION_FUSE_DEGREE = os.environ.get("AGGREGATION_FUSE_DEGREE", "0") == "1"
ENABLE_FORWARD_AGG_TIMING = os.environ.get("ENABLE_FORWARD_AGG_TIMING", "1") == "1"

ENABLE_PERSISTENT_DEVICE_CACHE = True
ENABLE_STAGE_TIMING = True
SAVE_PARTITION_PLAN = os.environ.get("SAVE_PARTITION_PLAN", "0") == "1"
DISABLE_AMP_FOR_TIMING = os.environ.get("DISABLE_AMP_FOR_TIMING", "1") == "1"

PRESET_CONFIG_NAME = "medium"
GLOBAL_HIDDEN_DIM = 32
NUM_CONV_LAYERS = 2
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-5
PATIENCE = 50
LOSS_MODE = "edges_only"
USE_VALIDATION = False
VAL_CACHE_PATHS: List[str] = []

RUN_PARENT_EXPERIMENTS = os.environ.get("RUN_PARENT_EXPERIMENTS", "1") == "1"

# Each date is scanned twice. Within each repeat, modes run in the exact order
# sequential -> async_chunk_2 -> ... -> async_chunk_10.
REPEAT_LIST = [
    int(x.strip())
    for x in os.environ.get("REPEAT_LIST", "1,2").split(",")
    if x.strip()
]
ASYNC_CHUNK_LIST = [
    int(x.strip())
    for x in os.environ.get(
        "ASYNC_CHUNK_LIST", "2,3,4,5,6,7,8,9,10"
    ).split(",")
    if x.strip()
]
SEQUENTIAL_NUM_CHUNKS = int(
    os.environ.get("SEQUENTIAL_NUM_CHUNKS", "4")
)

# The five parent-level graph jobs. Cache paths can be overridden independently.
GRAPH_EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "graph_tag": "20241213",
        "cache_path": os.environ.get(
            "CACHE_PATH_20241213",
            r"/path/to/artifact_workspace/cache/graph_20241213.pkl",
        ),
        "total_labeled_edges": int(os.environ.get("TOTAL_LABELED_EDGES_20241213", "5421825")),
    },
    {
        "graph_tag": "20241214",
        "cache_path": os.environ.get(
            "CACHE_PATH_20241214",
            r"/path/to/artifact_workspace/cache/graph_20241214.pkl",
        ),
        "total_labeled_edges": int(os.environ.get("TOTAL_LABELED_EDGES_20241214", "5353896")),
    },
    {
        "graph_tag": "20241215",
        "cache_path": os.environ.get(
            "CACHE_PATH_20241215",
            r"/path/to/artifact_workspace/cache/graph_20241215.pkl",
        ),
        "total_labeled_edges": int(os.environ.get("TOTAL_LABELED_EDGES_20241215", "5061841")),
    },
    {
        "graph_tag": "20241217",
        "cache_path": os.environ.get(
            "CACHE_PATH_20241217",
            r"/path/to/artifact_workspace/cache/graph_20241217.pkl",
        ),
        "total_labeled_edges": int(os.environ.get("TOTAL_LABELED_EDGES_20241217", "5049681")),
    },
]

# Full ordered mode sweep used inside every date/repeat.
FORWARD_AGG_EXPERIMENTS: List[Dict[str, str]] = [
    {
        "mode": "sequential_aggregation",
        "enable_overlap": "0",
        "num_chunks": str(SEQUENTIAL_NUM_CHUNKS),
        "async_op": "0",
        "double_buffer": "0",
    },
    *[
        {
            "mode": f"async_chunk_{c}",
            "enable_overlap": "1",
            "num_chunks": str(c),
            "async_op": "1",
            "double_buffer": "1",
        }
        for c in ASYNC_CHUNK_LIST
    ],
]

SKIP_EXISTING_RESULTS = os.environ.get("SKIP_EXISTING_RESULTS", "1") == "1"
STOP_ON_FAILURE = os.environ.get("STOP_ON_FAILURE", "1") == "1"
AUTO_NUM_PROCESSES = int(os.environ.get("AUTO_NUM_PROCESSES", "2"))
AUTO_MASTER_PORT = int(os.environ.get("AUTO_MASTER_PORT", "29547"))
FORCE_SINGLE_PROCESS = os.environ.get("FORCE_SINGLE_PROCESS", "0") == "1"


# ============================================================
# Utilities
# ============================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    # torch.manual_seed() also calls torch.cuda.manual_seed_all().  On the
    # K500 CUDA-compatibility runtime that can enqueue a callback for every
    # visible device and undo/fail the rank-local device binding.  Seed the
    # CPU generator directly, then seed only the CUDA device already selected
    # by require_distributed_cuda().
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        # torchrun gives each process one local device.  Seeding every visible
        # GPU queues a lazy callback before rank-to-device binding on some CUDA
        # compatibility runtimes (including the K500 server) and can fail with
        # ``tuple index out of range`` during the later set_device call.  Seed
        # only the GPU already selected by require_distributed_cuda().
        torch.cuda.manual_seed(seed)
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.manual_seed(seed)
            if hasattr(torch.npu, "manual_seed_all"):
                torch.npu.manual_seed_all(seed)
    except Exception:
        pass


def get_visible_device_count() -> int:
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return int(torch.npu.device_count())
    except Exception:
        pass
    if torch.cuda.is_available():
        return int(torch.cuda.device_count())
    return 0


def make_torchrun_cmd(
    nproc: int,
    script_path: str,
    master_port: int,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc),
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(master_port),
        script_path,
    ]


def stream_subprocess_to_log(cmd: List[str], env: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("[COMMAND] " + " ".join(cmd) + "\n")
        for key in [
            "FORWARD_AGG_EXPERIMENT_MODE",
            "ENABLE_FORWARD_AGG_OVERLAP",
            "AGGREGATION_NUM_CHUNKS",
            "AGGREGATION_ASYNC_OP",
            "AGGREGATION_DOUBLE_BUFFER",
        ]:
            log_file.write(f"[{key}] {env.get(key, '')}\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        proc.wait()
        log_file.write(f"\n[RETURN_CODE] {proc.returncode}\n")
        return int(proc.returncode)


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def experiment_name(
    mode: str,
    graph_tag: str,
    repeat_id: int,
    epochs: int,
    seed: int,
    target_num_blocks: int,
    target_super_block_size: int,
) -> str:
    return (
        f"forwardagg_{mode}_{PARTITION_STRATEGY}_{_safe_tag(graph_tag)}_"
        f"K{target_num_blocks}_Q{target_super_block_size}_R{repeat_id:02d}_"
        f"ep{epochs}_seed{seed}"
    )


def _cache_exists(cache_path: str) -> bool:
    path = Path(cache_path)
    return path.exists() or Path(cache_path + ".tmp").exists()


def _write_parent_summary(parent_dir: Path, rows: List[Dict[str, Any]]) -> None:
    json_path = parent_dir / "parent_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Also save a CSV-like file without introducing pandas into the launcher.
    csv_path = parent_dir / "parent_summary.csv"
    columns = [
        "job_index",
        "graph_tag",
        "mode",
        "repeat_id",
        "run_seed",
        "num_epochs",
        "num_chunks",
        "enable_overlap",
        "return_code",
        "parent_elapsed_sec",
        "skipped_existing",
        "total_labeled_edges",
        "target_num_blocks",
        "target_super_block_size",
        "cache_path",
        "output_dir",
        "log_path",
    ]
    import csv
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_parent(nproc: int, script_path: str) -> None:
    output_root = Path(OUTPUT_ROOT)
    parent_dir = output_root / "forward_agg_parent"
    log_dir = parent_dir / "logs"
    parent_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        g
        for g in GRAPH_EXPERIMENTS
        if not _cache_exists(str(g["cache_path"]))
    ]
    if missing:
        details = "\n".join(
            f"  {g['graph_tag']}: {g['cache_path']}" for g in missing
        )
        raise FileNotFoundError(
            "The following graph caches were not found. "
            "Override them with CACHE_PATH_<DATE> if necessary:\n"
            + details
        )

    summary_rows: List[Dict[str, Any]] = []
    total_jobs = (
        len(GRAPH_EXPERIMENTS)
        * len(REPEAT_LIST)
        * len(FORWARD_AGG_EXPERIMENTS)
    )

    print("=" * 100, flush=True)
    print(
        "[FORWARD-AGG PARENT] Start five-date seq-to-chunk10 "
        "two-repeat sweep",
        flush=True,
    )
    print(
        f"[FORWARD-AGG PARENT] graphs="
        f"{[g['graph_tag'] for g in GRAPH_EXPERIMENTS]}",
        flush=True,
    )
    print(
        f"[FORWARD-AGG PARENT] repeats={REPEAT_LIST}, "
        f"modes={[x['mode'] for x in FORWARD_AGG_EXPERIMENTS]}",
        flush=True,
    )
    print(
        f"[FORWARD-AGG PARENT] order="
        "date -> repeat -> sequential/chunk2...chunk10",
        flush=True,
    )
    print(
        f"[FORWARD-AGG PARENT] epochs_per_job={NUM_EPOCHS}, "
        f"total_jobs={total_jobs}, nproc={nproc}",
        flush=True,
    )
    print(
        f"[FORWARD-AGG PARENT] Q={TARGET_SUPER_BLOCK_SIZE}, "
        f"strategy={PARTITION_STRATEGY}",
        flush=True,
    )
    print(f"[FORWARD-AGG PARENT] output_root={OUTPUT_ROOT}", flush=True)
    print("=" * 100, flush=True)

    failed = False
    job_index = 0

    # Required order:
    # date 1: repeat 1 [seq -> chunk10], repeat 2 [seq -> chunk10]
    # then date 2, and so on.
    for graph_cfg in GRAPH_EXPERIMENTS:
        graph_tag = str(graph_cfg["graph_tag"])
        cache_path = str(graph_cfg["cache_path"])
        total_labeled_edges = int(graph_cfg["total_labeled_edges"])
        target_num_blocks = int(
            math.ceil(
                total_labeled_edges / float(TARGET_SUPER_BLOCK_SIZE)
            )
        )

        print("\n" + "=" * 100, flush=True)
        print(
            f"[FORWARD-AGG PARENT] BEGIN graph={graph_tag}, "
            f"edges={total_labeled_edges}, "
            f"target_blocks={target_num_blocks}",
            flush=True,
        )
        print("=" * 100, flush=True)

        for repeat_id in REPEAT_LIST:
            run_seed = (
                BASE_SEED + repeat_id - 1
                if VARY_SEED_BY_REPEAT
                else BASE_SEED
            )

            print(
                f"\n[FORWARD-AGG PARENT] graph={graph_tag}, "
                f"BEGIN repeat={repeat_id}, seed={run_seed}",
                flush=True,
            )

            for exp in FORWARD_AGG_EXPERIMENTS:
                job_index += 1
                mode = exp["mode"]

                exp_name = experiment_name(
                    mode=mode,
                    graph_tag=graph_tag,
                    repeat_id=repeat_id,
                    epochs=NUM_EPOCHS,
                    seed=run_seed,
                    target_num_blocks=target_num_blocks,
                    target_super_block_size=TARGET_SUPER_BLOCK_SIZE,
                )
                exp_dir = output_root / exp_name
                expected_result = exp_dir / "train_result_rank0.json"
                log_path = (
                    log_dir
                    / graph_tag
                    / f"repeat_{repeat_id:02d}"
                    / f"{exp_name}.log"
                )

                base_summary = {
                    "job_index": job_index,
                    "graph_tag": graph_tag,
                    "mode": mode,
                    "repeat_id": repeat_id,
                    "run_seed": run_seed,
                    "num_epochs": NUM_EPOCHS,
                    "num_chunks": int(exp["num_chunks"]),
                    "enable_overlap": exp["enable_overlap"] == "1",
                    "total_labeled_edges": total_labeled_edges,
                    "target_num_blocks": target_num_blocks,
                    "target_super_block_size": TARGET_SUPER_BLOCK_SIZE,
                    "cache_path": cache_path,
                    "output_dir": str(exp_dir),
                    "log_path": str(log_path),
                }

                if SKIP_EXISTING_RESULTS and expected_result.exists():
                    print(
                        f"[FORWARD-AGG PARENT] "
                        f"[{job_index}/{total_jobs}] SKIP existing "
                        f"graph={graph_tag}, repeat={repeat_id}, "
                        f"mode={mode}: {expected_result}",
                        flush=True,
                    )
                    summary_rows.append(
                        {
                            **base_summary,
                            "return_code": 0,
                            "parent_elapsed_sec": 0.0,
                            "skipped_existing": True,
                        }
                    )
                    _write_parent_summary(parent_dir, summary_rows)
                    continue

                env = os.environ.copy()
                env["COMMOPT_DDP_LAUNCHED"] = "1"
                env["RUN_PARENT_EXPERIMENTS"] = "0"
                env["FORWARD_AGG_EXPERIMENT_MODE"] = mode
                env["STEADY_EXPERIMENT_MODE"] = mode
                env["ENABLE_FORWARD_AGGREGATION"] = "1"
                env["ENABLE_FORWARD_AGG_OVERLAP"] = exp[
                    "enable_overlap"
                ]
                env["AGGREGATION_NUM_CHUNKS"] = exp["num_chunks"]
                env["AGGREGATION_ASYNC_OP"] = exp["async_op"]
                env["AGGREGATION_DOUBLE_BUFFER"] = exp[
                    "double_buffer"
                ]
                env["AGGREGATION_CHUNK_POLICY"] = "destination_range"
                env["ENABLE_FORWARD_AGG_TIMING"] = "1"
                env["ENABLE_PERSISTENT_DEVICE_CACHE"] = "1"
                env["ENABLE_RELATION_SCHEDULED_CONV"] = "0"
                env["REPEAT_ID"] = str(repeat_id)
                env["GRAPH_TAG"] = graph_tag
                env["CACHE_PATH"] = cache_path
                env["TOTAL_LABELED_EDGES"] = str(
                    total_labeled_edges
                )
                env["TARGET_SUPER_BLOCK_SIZE"] = str(
                    TARGET_SUPER_BLOCK_SIZE
                )
                env["TARGET_NUM_BLOCKS"] = str(target_num_blocks)
                env["PARTITION_STRATEGY"] = PARTITION_STRATEGY
                env["NUM_EPOCHS"] = str(NUM_EPOCHS)
                env["OUTPUT_ROOT"] = OUTPUT_ROOT

                # The child reads BASE_SEED at import time. Pass the resolved
                # repeat seed so all modes in one repeat are paired.
                env["BASE_SEED"] = str(run_seed)

                master_port = AUTO_MASTER_PORT + (job_index % 200)
                cmd = make_torchrun_cmd(
                    nproc=nproc,
                    script_path=script_path,
                    master_port=master_port,
                )

                print("\n" + "#" * 100, flush=True)
                print(
                    f"[FORWARD-AGG PARENT] [{job_index}/{total_jobs}] "
                    f"graph={graph_tag}, repeat={repeat_id}, "
                    f"seed={run_seed}, mode={mode}, "
                    f"chunks={exp['num_chunks']}, "
                    f"overlap={exp['enable_overlap']}",
                    flush=True,
                )
                print(
                    f"[FORWARD-AGG PARENT] output_dir={exp_dir}",
                    flush=True,
                )
                print(
                    f"[FORWARD-AGG PARENT] log_path={log_path}",
                    flush=True,
                )
                print("#" * 100, flush=True)

                t0 = time.perf_counter()
                return_code = stream_subprocess_to_log(
                    cmd, env, log_path
                )
                elapsed = float(time.perf_counter() - t0)

                summary_rows.append(
                    {
                        **base_summary,
                        "return_code": return_code,
                        "parent_elapsed_sec": elapsed,
                        "skipped_existing": False,
                    }
                )
                _write_parent_summary(parent_dir, summary_rows)

                print(
                    f"[FORWARD-AGG PARENT] DONE "
                    f"[{job_index}/{total_jobs}] "
                    f"graph={graph_tag}, repeat={repeat_id}, "
                    f"mode={mode}, return_code={return_code}, "
                    f"elapsed={elapsed:.3f}s",
                    flush=True,
                )

                if return_code != 0 and STOP_ON_FAILURE:
                    failed = True
                    break

            if failed:
                break

            print(
                f"[FORWARD-AGG PARENT] graph={graph_tag}, "
                f"END repeat={repeat_id}",
                flush=True,
            )

        if failed:
            break

        print(
            f"[FORWARD-AGG PARENT] END graph={graph_tag}",
            flush=True,
        )

    _write_parent_summary(parent_dir, summary_rows)

    completed = sum(
        1 for row in summary_rows if row.get("return_code") == 0
    )
    print("=" * 100, flush=True)
    print(
        f"[FORWARD-AGG PARENT] Finished: "
        f"successful_or_skipped={completed}/{total_jobs}, "
        f"summary={parent_dir / 'parent_summary.json'}",
        flush=True,
    )
    print("=" * 100, flush=True)

    if failed:
        raise RuntimeError(
            "A child experiment failed and STOP_ON_FAILURE=1. "
            f"See {parent_dir / 'parent_summary.json'} "
            "and the corresponding log."
        )


# ============================================================
# Child process: one DDP experiment
# ============================================================

def get_rank_world() -> Tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_distributed = world_size > 1
    return rank, local_rank, world_size, use_distributed


def setup_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - rank{rank} - %(levelname)s - %(name)s - %(message)s",
        force=True,
    )


def load_graph_cache(cache_path: str):
    candidates = [Path(cache_path), Path(cache_path + ".tmp")]
    last_error: Optional[Exception] = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, HeteroData):
                return obj, None, None, obj
            if isinstance(obj, dict):
                graph = obj.get("graph") or obj.get("data") or obj.get("hetero_data")
                if graph is not None and hasattr(graph, "node_types"):
                    return graph, obj.get("node_manager"), obj.get("edge_manager"), obj
            if isinstance(obj, tuple):
                for item in obj:
                    if isinstance(item, HeteroData) or (
                        hasattr(item, "node_types") and hasattr(item, "edge_types")
                    ):
                        return item, None, None, obj
            if hasattr(obj, "node_types") and hasattr(obj, "edge_types"):
                return obj, None, None, obj
            raise TypeError(f"Unsupported cache format: {type(obj)}")
        except Exception as exc:
            last_error = exc
    raise FileNotFoundError(f"Cannot load graph cache {cache_path}; last_error={last_error}")


def infer_node_feature_dims(graph: HeteroData) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for node_type in graph.node_types:
        x = getattr(graph[node_type], "x", None)
        if isinstance(x, torch.Tensor):
            dims[node_type] = 1 if x.dim() == 1 else int(x.shape[1])
    return dims


class RuntimeDeviceConfig:
    def __init__(self, preferred_device: str = "auto"):
        self.preferred_device = preferred_device

    def get_device(self, local_rank: Optional[int] = None) -> torch.device:
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.preferred_device in ("auto", "npu"):
            try:
                if hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.set_device(local_rank)
                    return torch.device(f"npu:{local_rank}")
            except Exception:
                pass
        if self.preferred_device in ("auto", "cuda") and torch.cuda.is_available():
            device_id = mapped_cuda_device_id(local_rank)
            torch.cuda.set_device(device_id)
            return torch.device(f"cuda:{device_id}")
        return torch.device("cpu")


def mapped_cuda_device_id(local_rank: int) -> int:
    raw = os.environ.get("COMMOPT_CUDA_DEVICE_MAP", "").strip()
    if not raw:
        return int(local_rank)
    try:
        device_ids = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise RuntimeError(f"Invalid COMMOPT_CUDA_DEVICE_MAP={raw!r}") from exc
    if local_rank < 0 or local_rank >= len(device_ids):
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} has no entry in COMMOPT_CUDA_DEVICE_MAP={raw!r}"
        )
    return device_ids[local_rank]


def require_distributed_cuda(local_rank: int, world_size: int) -> int:
    """Validate CUDA before loading the graph or constructing the model."""
    if world_size <= 1 or os.environ.get("COMMOPT_FORCE_CUDA", "0") != "1":
        return mapped_cuda_device_id(local_rank)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    device_map = os.environ.get("COMMOPT_CUDA_DEVICE_MAP", "").strip()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "COMMOPT_FORCE_CUDA=1 but CUDA is unavailable; "
            f"CUDA_VISIBLE_DEVICES={visible}, torch={torch.__version__}, "
            f"torch.version.cuda={torch.version.cuda}"
        )
    count = int(torch.cuda.device_count())
    mapped_ids = (
        [int(value.strip()) for value in device_map.split(",") if value.strip()]
        if device_map
        else list(range(world_size))
    )
    if len(mapped_ids) != world_size:
        raise RuntimeError(
            f"Expected {world_size} entries in COMMOPT_CUDA_DEVICE_MAP, "
            f"found {mapped_ids}"
        )
    device_id = mapped_cuda_device_id(local_rank)
    if device_id < 0 or device_id >= count:
        raise RuntimeError(
            f"Mapped CUDA device {device_id} is outside the {count} devices seen by "
            f"PyTorch; map={device_map!r}, CUDA_VISIBLE_DEVICES={visible}"
        )
    try:
        torch.cuda.set_device(device_id)
        probe = torch.ones(1, device=f"cuda:{device_id}")
        torch.cuda.synchronize(device_id)
        del probe
    except Exception as exc:
        raise RuntimeError(
            f"Cannot use mapped CUDA device {device_id} for LOCAL_RANK={local_rank}; "
            f"map={device_map!r}, CUDA_VISIBLE_DEVICES={visible}: {exc}"
        ) from exc
    return device_id


def bind_manager_to_rank_cuda(
    manager: Any,
    local_rank: int,
    world_size: int,
    cuda_device_id: int,
) -> torch.device:
    """Keep NCCL models on the CUDA device assigned to this torchrun rank.

    Some older TrainingManager copies catch a device-initialization exception
    and silently fall back to CPU.  NCCL can initialize in that state, but DDP
    later fails while verifying CPU parameters.  Repair the manager before its
    lazy model initialization, then verify every parameter and buffer.
    """
    force_cuda = os.environ.get("COMMOPT_FORCE_CUDA", "0") == "1"
    if not force_cuda:
        return manager.device

    torch.cuda.set_device(cuda_device_id)
    expected = torch.device("cuda", cuda_device_id)
    manager.local_rank = int(local_rank)
    manager.world_size = int(world_size)
    manager.cuda_device_id = int(cuda_device_id)
    manager.device = expected
    manager.model = manager.model.to(expected)

    wrong_parameters = [
        name
        for name, value in manager.model.named_parameters()
        if value.device.type != "cuda" or value.device.index != cuda_device_id
    ]
    wrong_buffers = [
        name
        for name, value in manager.model.named_buffers()
        if value.device.type != "cuda" or value.device.index != cuda_device_id
    ]
    if wrong_parameters or wrong_buffers:
        raise RuntimeError(
            "Failed to bind model to rank-local CUDA device "
            f"{expected}; wrong_parameters={wrong_parameters[:8]}, "
            f"wrong_buffers={wrong_buffers[:8]}"
        )

    if world_size > 1:
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group was not initialized")
        backend = str(dist.get_backend()).lower()
        if "nccl" not in backend:
            raise RuntimeError(
                f"Expected an NCCL process group for CUDA DDP, found {backend!r}"
            )
    return expected


def build_config(graph: HeteroData, output_dir: str):
    config = create_preset_config(PRESET_CONFIG_NAME)
    config.training_config = TrainingConfig(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        batch_size=1,
        l2_lambda=5e-2,
        loss_mode=LOSS_MODE,
        partition_strategy=PARTITION_STRATEGY,
        enable_graph_partition=True,
    )
    config.architecture_config = ModelArchitectureConfig(
        num_conv_layers=NUM_CONV_LAYERS,
        global_hidden_dim=GLOBAL_HIDDEN_DIM,
    )
    preferred_device = (
        "cuda" if os.environ.get("COMMOPT_FORCE_CUDA", "0") == "1" else "auto"
    )
    config.device_config = RuntimeDeviceConfig(preferred_device)
    config.edge_types = [et for et in graph.edge_types if isinstance(et, tuple) and len(et) == 3]
    config._init_configs()

    for layer_config in config.layer_configs.values():
        layer_config.dropout_rate = 0.08
        layer_config.hidden_dim = GLOBAL_HIDDEN_DIM
    for edge_config in config.edge_configs.values():
        edge_config.dropout_rate = 0.08
        edge_config.hidden_dim = GLOBAL_HIDDEN_DIM

    tc = config.training_config
    tc.target_super_block_size = TARGET_SUPER_BLOCK_SIZE
    tc.superblock_assignment_output_dir = output_dir
    tc.save_partition_plan = SAVE_PARTITION_PLAN
    tc.enable_stage_timing = ENABLE_STAGE_TIMING
    tc.enable_persistent_device_cache = ENABLE_PERSISTENT_DEVICE_CACHE
    tc.steady_experiment_mode = STEADY_EXPERIMENT_MODE

    tc.superblock_assignment_objective = "vector_balance"
    tc.assignment_lambda_count = 1.0
    tc.assignment_lambda_type = 0.20
    tc.assignment_lambda_block = 0.02

    # Exact forward aggregation experiment knobs.
    tc.enable_forward_aggregation = ENABLE_FORWARD_AGGREGATION
    tc.forward_aggregation_operator = FORWARD_AGGREGATION_OPERATOR
    tc.enable_forward_agg_overlap = ENABLE_FORWARD_AGG_OVERLAP
    tc.aggregation_num_chunks = AGGREGATION_NUM_CHUNKS
    tc.aggregation_async_op = AGGREGATION_ASYNC_OP
    tc.aggregation_double_buffer = AGGREGATION_DOUBLE_BUFFER
    tc.aggregation_chunk_policy = AGGREGATION_CHUNK_POLICY
    tc.aggregation_chunk_order = AGGREGATION_CHUNK_ORDER
    tc.aggregation_max_inflight = AGGREGATION_MAX_INFLIGHT
    tc.aggregation_fuse_degree = AGGREGATION_FUSE_DEGREE
    tc.enable_forward_agg_timing = ENABLE_FORWARD_AGG_TIMING

    # Disable older relation reordering/chunk predictors in this experiment.
    tc.enable_relation_scheduled_conv = False
    tc.enable_relation_forward_timing = False
    tc.enable_chunked_edge_prediction = False
    tc.enable_chunked_loss = False
    tc.enable_forward_loss_detail_timing = True

    if DISABLE_AMP_FOR_TIMING:
        tc.use_amp = False
        tc.enable_amp = False
        tc.amp_enabled = False

    return config


def disable_amp_on_manager(manager: Any) -> None:
    if not DISABLE_AMP_FOR_TIMING:
        return
    for name, value in [
        ("use_amp", False),
        ("amp_enabled", False),
        ("enable_amp", False),
        ("scaler", None),
        ("amp_scaler", None),
        ("_amp_device_type", None),
        ("amp_dtype", torch.float32),
    ]:
        try:
            setattr(manager, name, value)
        except Exception:
            pass


def reset_peak_memory(local_rank: int) -> None:
    if torch.cuda.is_available():
        try:
            device_id = mapped_cuda_device_id(local_rank)
            torch.cuda.set_device(device_id)
            torch.cuda.reset_peak_memory_stats(device_id)
        except Exception:
            pass


def get_peak_memory_mb(local_rank: int) -> Optional[float]:
    if torch.cuda.is_available():
        try:
            device_id = mapped_cuda_device_id(local_rank)
            return float(torch.cuda.max_memory_allocated(device_id)) / (1024 ** 2)
        except Exception:
            return None
    return None


def extract_final_loss(history: Any) -> Optional[float]:
    if history is None:
        return None
    if isinstance(history, (int, float)):
        return float(history)
    if isinstance(history, dict):
        for key in ["final_loss", "best_loss", "train_loss", "loss", "last_loss"]:
            value = history.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for key in ["train_losses", "losses", "epoch_losses", "training_losses", "train_loss_history"]:
            values = history.get(key)
            if isinstance(values, list) and values:
                last = values[-1]
                if isinstance(last, (int, float)):
                    return float(last)
                if isinstance(last, dict):
                    for subkey in ["loss", "train_loss", "total_loss"]:
                        if isinstance(last.get(subkey), (int, float)):
                            return float(last[subkey])
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, (int, float)):
            return float(last)
        if isinstance(last, dict):
            for key in ["loss", "train_loss", "total_loss"]:
                if isinstance(last.get(key), (int, float)):
                    return float(last[key])
    return None


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def main_child() -> None:
    rank, local_rank, world_size, use_distributed = get_rank_world()
    cuda_device_id = require_distributed_cuda(local_rank, world_size)
    set_global_seed(BASE_SEED)
    setup_logging(rank)
    logger = logging.getLogger("forward_aggregation_overlap_runner")

    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    exp_name = experiment_name(
        mode=FORWARD_AGG_EXPERIMENT_MODE,
        graph_tag=GRAPH_TAG,
        repeat_id=REPEAT_ID,
        epochs=NUM_EPOCHS,
        seed=BASE_SEED,
        target_num_blocks=TARGET_NUM_BLOCKS,
        target_super_block_size=TARGET_SUPER_BLOCK_SIZE,
    )
    output_dir = str(output_root / exp_name)

    logger.info("=" * 80)
    logger.info(f"rank={rank}, local_rank={local_rank}, world_size={world_size}, distributed={use_distributed}")
    logger.info(
        f"cuda_device_id={cuda_device_id}, "
        f"cuda_device_map={os.environ.get('COMMOPT_CUDA_DEVICE_MAP', '<unset>')}"
    )
    logger.info(f"mode={FORWARD_AGG_EXPERIMENT_MODE}")
    logger.info(f"enable_forward_aggregation={ENABLE_FORWARD_AGGREGATION}")
    logger.info(f"forward_aggregation_operator={FORWARD_AGGREGATION_OPERATOR}")
    logger.info(f"enable_forward_agg_overlap={ENABLE_FORWARD_AGG_OVERLAP}")
    logger.info(f"aggregation_num_chunks={AGGREGATION_NUM_CHUNKS}")
    logger.info(f"aggregation_async_op={AGGREGATION_ASYNC_OP}")
    logger.info(f"aggregation_double_buffer={AGGREGATION_DOUBLE_BUFFER}")
    logger.info(f"chunk_policy={AGGREGATION_CHUNK_POLICY}")
    logger.info(f"chunk_order={AGGREGATION_CHUNK_ORDER}")
    logger.info(f"max_inflight={AGGREGATION_MAX_INFLIGHT}")
    logger.info(f"fuse_degree={AGGREGATION_FUSE_DEGREE}")
    logger.info(f"persistent_device_cache={ENABLE_PERSISTENT_DEVICE_CACHE}")
    logger.info(f"graph={GRAPH_TAG}, Q={TARGET_SUPER_BLOCK_SIZE}, strategy={PARTITION_STRATEGY}")
    logger.info(f"output_dir={output_dir}")
    logger.info("=" * 80)

    graph, _, _, _ = load_graph_cache(CACHE_PATH)
    graph_date = getattr(graph, "graph_date", None) or getattr(graph, "date", None) or Path(CACHE_PATH).stem
    logger.info(f"loaded graph_date={graph_date}, nodes={len(graph.node_types)}, edges={len(graph.edge_types)}")

    config = build_config(graph, output_dir)
    model = SupplyChainGCN(
        node_features_dim_dict=infer_node_feature_dims(graph),
        config=config,
        previous_days=getattr(config, "previous_days", 7),
    )
    manager = CommunicationOptimizedTrainingManager(
        model=model,
        config=config,
        experiment_name=exp_name,
        node_manager=None,
        graph_builder=None,
        use_distributed=use_distributed,
        local_rank=local_rank,
        world_size=world_size,
    )
    manager_device_before_rebind = str(manager.device)
    expected_device = bind_manager_to_rank_cuda(
        manager=manager,
        local_rank=local_rank,
        world_size=world_size,
        cuda_device_id=cuda_device_id,
    )
    logger.info(
        "rank-local device gate passed: before=%s, after=%s, "
        "local_rank=%s, logical_cuda_id=%s, physical_pool=%s",
        manager_device_before_rebind,
        expected_device,
        local_rank,
        cuda_device_id,
        os.environ.get("COMMOPT_PHYSICAL_GPU_IDS", "<unset>"),
    )
    disable_amp_on_manager(manager)
    manager.set_sample_data_for_initialization(graph)
    reset_peak_memory(local_rank)

    if use_distributed and dist.is_initialized():
        dist.barrier()
    t0 = time.perf_counter()
    history = manager.train([graph], None)
    if use_distributed and dist.is_initialized():
        dist.barrier()
    elapsed = float(time.perf_counter() - t0)

    result = {
        "mode": FORWARD_AGG_EXPERIMENT_MODE,
        "strategy": PARTITION_STRATEGY,
        "rank": rank,
        "local_rank": local_rank,
        "cuda_device_id": cuda_device_id,
        "world_size": world_size,
        "graph_tag": GRAPH_TAG,
        "graph_date": str(graph_date),
        "cache_path": CACHE_PATH,
        "total_labeled_edges": TOTAL_LABELED_EDGES,
        "repeat_id": REPEAT_ID,
        "run_seed": BASE_SEED,
        "target_num_blocks": TARGET_NUM_BLOCKS,
        "target_super_block_size": TARGET_SUPER_BLOCK_SIZE,
        "num_epochs": NUM_EPOCHS,
        "enable_forward_aggregation": ENABLE_FORWARD_AGGREGATION,
        "forward_aggregation_operator": FORWARD_AGGREGATION_OPERATOR,
        "enable_forward_agg_overlap": ENABLE_FORWARD_AGG_OVERLAP,
        "aggregation_num_chunks": AGGREGATION_NUM_CHUNKS,
        "aggregation_async_op": AGGREGATION_ASYNC_OP,
        "aggregation_double_buffer": AGGREGATION_DOUBLE_BUFFER,
        "aggregation_chunk_policy": AGGREGATION_CHUNK_POLICY,
        "aggregation_chunk_order": AGGREGATION_CHUNK_ORDER,
        "aggregation_max_inflight": AGGREGATION_MAX_INFLIGHT,
        "aggregation_fuse_degree": AGGREGATION_FUSE_DEGREE,
        "enable_forward_agg_timing": ENABLE_FORWARD_AGG_TIMING,
        "enable_persistent_device_cache": ENABLE_PERSISTENT_DEVICE_CACHE,
        "elapsed_sec": elapsed,
        "peak_memory_mb": get_peak_memory_mb(local_rank),
        "final_loss_extracted": extract_final_loss(history),
        "output_dir": output_dir,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"train_result_rank{rank}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(
        f"DONE mode={FORWARD_AGG_EXPERIMENT_MODE}, elapsed={elapsed:.3f}s, "
        f"peak_memory_mb={result['peak_memory_mb']}, final_loss={result['final_loss_extracted']}"
    )
    cleanup_distributed()
    gc.collect()


def main() -> None:
    launched = os.environ.get("COMMOPT_DDP_LAUNCHED", "0") == "1"
    if RUN_PARENT_EXPERIMENTS and not launched and not FORCE_SINGLE_PROCESS:
        visible = get_visible_device_count()
        if visible < AUTO_NUM_PROCESSES:
            raise RuntimeError(
                f"Need {AUTO_NUM_PROCESSES} visible devices for exact cross-rank aggregation, found {visible}"
            )
        run_parent(AUTO_NUM_PROCESSES, str(Path(__file__).resolve()))
        return
    main_child()


if __name__ == "__main__":
    main()

