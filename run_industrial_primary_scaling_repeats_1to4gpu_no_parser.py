#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finish publication repeats for the primary industrial strong-scaling plot.

Direct run; no argparse.  This intentionally measures only the two stable
physical-layout configurations supported by this project copy:

* Direct edge-balanced + exact blocking aggregation
* SuperBlock edge-count + exact blocking aggregation

The quick core suite already measures repeat 1.  This file adds independent
repeats 2 and 3 for the representative 2024-11-29 snapshot on 1/2/3/4 GPUs.
All jobs use the existing industrial runner and output naming, so completed
jobs are detected and skipped automatically after an interruption.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_ROOT = PROJECT_ROOT / "sigmod_experiments" / "server_no_parser"
sys.path.insert(0, str(SERVER_ROOT))

from common_no_parser import (  # noqa: E402
    load_multidate_metadata,
    run_multidate_industrial_job,
)
from settings_no_parser import BASE_OUTPUT_ROOT  # noqa: E402
from settings_no_parser import INDUSTRIAL_RUNNER  # noqa: E402


GRAPH_TAG = "20241129"
GPU_COUNTS = (1, 2, 3, 4)
INDEPENDENT_REPEATS = (2, 3)
BLOCK_EDGES = 25_000
STAGE = "core_industrial_scaling"

DENSE_BLOCKING = {
    "name": "blocking_c1",
    "chunks": 1,
    "overlap": False,
}

CONFIGURATIONS = (
    ("Direct edge-balanced", "balanced_edge", DENSE_BLOCKING),
    ("SuperBlock edge-count", "superblock_edge_count", DENSE_BLOCKING),
)


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def preflight() -> None:
    if not SERVER_ROOT.exists():
        raise FileNotFoundError(f"Missing experiment package: {SERVER_ROOT}")
    metadata = load_multidate_metadata()
    if GRAPH_TAG not in metadata:
        raise KeyError(
            f"Snapshot {GRAPH_TAG} is absent from multidate metadata. "
            "Run 08_multidate_preflight_no_parser.py first."
        )
    graph_path = Path(metadata[GRAPH_TAG]["graph_path"])
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)

    # PyCharm remote runs sometimes upload only this outer launcher and leave
    # imported project files stale under /path/to/deployment_copy.  Detect that
    # before spending several minutes loading the graph and constructing the
    # model.  The old seed path is known to fail on the K500 compatibility
    # runtime for world_size > 1.
    runner_source = INDUSTRIAL_RUNNER.read_text(encoding="utf-8")
    common_path = SERVER_ROOT / "common_no_parser.py"
    common_source = common_path.read_text(encoding="utf-8")
    manager_path = PROJECT_ROOT / "end_to_end_prunning" / "models" / "training_manager.py"
    manager_source = manager_path.read_text(encoding="utf-8")
    stale_reasons = []
    if "torch.random.default_generator.manual_seed(seed)" not in runner_source:
        stale_reasons.append(
            "industrial runner still uses the unsafe torch.manual_seed/manual_seed_all path"
        )
    if "COMMOPT_FORCE_CUDA" not in common_source:
        stale_reasons.append("common_no_parser.py does not force CUDA")
    if "COMMOPT_CUDA_DEVICE_MAP" not in common_source:
        stale_reasons.append("common_no_parser.py does not pass the logical GPU map")
    if (
        "COMMOPT_STRICT_DEVICE" not in manager_source
        or "Device initialization failed in strict mode" not in manager_source
    ):
        stale_reasons.append("training_manager.py can still silently fall back to CPU")
    if stale_reasons:
        raise RuntimeError(
            "STALE REMOTE PROJECT: PyCharm did not upload the patched dependencies. "
            + "; ".join(stale_reasons)
            + ". Upload the complete artifact directory (at minimum the industrial "
              "runner, common_no_parser.py, and training_manager.py), "
              "then rerun this launcher."
        )
    print(f"graph={graph_path}", flush=True)
    print(
        f"jobs={len(GPU_COUNTS) * len(INDEPENDENT_REPEATS) * len(CONFIGURATIONS)}; "
        f"GPUs={GPU_COUNTS}; repeats={INDEPENDENT_REPEATS}",
        flush=True,
    )


def main() -> None:
    preflight()
    started = time.time()
    for repeat in INDEPENDENT_REPEATS:
        for world_size in GPU_COUNTS:
            for label, strategy, mode in CONFIGURATIONS:
                print(
                    f"\nINDUSTRIAL REPEAT graph={GRAPH_TAG} GPUs={world_size} "
                    f"repeat={repeat} config={label}",
                    flush=True,
                )
                run_multidate_industrial_job(
                    stage=STAGE,
                    graph_tag=GRAPH_TAG,
                    world_size=world_size,
                    strategy=strategy,
                    block_edges=BLOCK_EDGES,
                    mode=mode,
                    repeat=repeat,
                )

    completion = (
        BASE_OUTPUT_ROOT
        / "sigmod_core_suite"
        / "stage_status"
        / "industrial_primary_scaling_repeats_20241129.json"
    )
    write_json_atomic(
        completion,
        {
            "completed": True,
            "graph_tag": GRAPH_TAG,
            "gpu_counts": list(GPU_COUNTS),
            "independent_repeats_added": list(INDEPENDENT_REPEATS),
            "configurations": [label for label, _, _ in CONFIGURATIONS],
            "jobs": len(GPU_COUNTS)
            * len(INDEPENDENT_REPEATS)
            * len(CONFIGURATIONS),
            "elapsed_sec": time.time() - started,
        },
    )
    print(f"COMPLETE: {completion}", flush=True)


if __name__ == "__main__":
    main()

