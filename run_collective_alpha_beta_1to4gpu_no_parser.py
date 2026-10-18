#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-argument 1/2/3/4-GPU AllReduce alpha-beta calibration.

Run once with ordinary Python.  The parent launches isolated torchrun jobs and
combines their measurements.  No graph is loaded, so this normally completes in
minutes rather than hours.
"""

from __future__ import annotations

import csv
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist


OUTPUT_ROOT = Path(
    os.environ.get(
        "SIGMOD_ALPHA_BETA_OUTPUT",
        "/path/to/artifact_workspace/sigmod2027_20241129/collective_alpha_beta_1to4gpu",
    )
)
GPU_COUNTS = (1, 2, 3, 4)
ELEMENT_COUNTS = (
    256,
    1_024,
    4_096,
    16_384,
    65_536,
    262_144,
    1_048_576,
    4_194_304,
    16_777_216,
)
WARMUPS = 4
REPEATS = 12
FIT_MIN_BYTES = 64 * 1024
WORKER_FLAG = "SIGMOD_ALPHA_BETA_WORKER"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)


def choose_device(local_rank: int):
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)
        return "hccl", torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return "nccl", torch.device(f"cuda:{local_rank}")
    raise RuntimeError("A CUDA/NPU distributed runtime is required")


def worker() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    backend, device = choose_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    dtype = torch.float32
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    rows: List[Dict[str, Any]] = []

    for element_count in ELEMENT_COUNTS:
        tensor = torch.ones(element_count, dtype=dtype, device=device)
        for _ in range(WARMUPS):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        sync(device)
        dist.barrier()

        local_values = []
        for _ in range(REPEATS):
            dist.barrier()
            sync(device)
            started = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            sync(device)
            local_values.append(time.perf_counter() - started)

        values = torch.tensor(local_values, dtype=torch.float64, device=device)
        gathered = [torch.empty_like(values) for _ in range(world)]
        dist.all_gather(gathered, values)
        if rank == 0:
            matrix = torch.stack([value.cpu() for value in gathered]).numpy()
            critical = matrix.max(axis=0)
            payload_bytes = int(element_count * dtype_bytes)
            row = {
                "host": socket.gethostname(),
                "backend": backend,
                "world_size": world,
                "element_count": element_count,
                "payload_bytes": payload_bytes,
                "payload_mib": payload_bytes / (1024 ** 2),
                "mean_sec": float(critical.mean()),
                "median_sec": float(np.median(critical)),
                "stdev_sec": float(critical.std()),
                "p90_sec": float(np.quantile(critical, 0.90)),
                "min_sec": float(critical.min()),
                "algorithmic_bandwidth_gbps": payload_bytes / max(float(critical.mean()), 1e-12) / 1e9,
            }
            rows.append(row)
            print(
                f"G={world} payload={row['payload_mib']:.3f} MiB "
                f"mean={row['mean_sec'] * 1e3:.3f} ms",
                flush=True,
            )
        del tensor

    if rank == 0:
        directory = OUTPUT_ROOT / f"gpus-{world}"
        write_csv(directory / "allreduce_measurements.csv", rows)
        fit_rows = [row for row in rows if int(row["payload_bytes"]) >= FIT_MIN_BYTES]
        x = np.asarray([row["payload_bytes"] for row in fit_rows], dtype=np.float64)
        y = np.asarray([row["mean_sec"] for row in fit_rows], dtype=np.float64)
        design = np.column_stack([np.ones_like(x), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ coefficients
        residual = float(((y - prediction) ** 2).sum())
        total = float(((y - y.mean()) ** 2).sum())
        alpha, beta = float(coefficients[0]), float(coefficients[1])
        fit = {
            "model": "T = alpha + beta * payload_bytes",
            "world_size": world,
            "backend": backend,
            "alpha_sec": alpha,
            "alpha_us": alpha * 1e6,
            "beta_sec_per_byte": beta,
            "effective_bandwidth_gbps": (1.0 / beta / 1e9) if beta > 0 else None,
            "fit_r2": 1.0 - residual / total if total > 0 else 1.0,
            "fit_min_bytes": FIT_MIN_BYTES,
        }
        (directory / "alpha_beta_fit.json").write_text(
            json.dumps(fit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    dist.barrier()
    dist.destroy_process_group()


def parent() -> None:
    if not torch.cuda.is_available() and not (
        hasattr(torch, "npu") and torch.npu.is_available()
    ):
        raise RuntimeError("No GPU/NPU runtime is available")
    available = (
        torch.cuda.device_count()
        if torch.cuda.is_available()
        else torch.npu.device_count()
    )
    if available < max(GPU_COUNTS):
        raise RuntimeError(f"Expected at least four accelerators, found {available}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for world in GPU_COUNTS:
        environment = os.environ.copy()
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment[WORKER_FLAG] = "1"
        log = OUTPUT_ROOT / f"gpus-{world}.log"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            str(Path(__file__).resolve()),
        ]
        print(f"RUN ALPHA-BETA G={world}: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parent),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-100:])
            raise RuntimeError(f"Alpha-beta G={world} failed:\n{tail}")

    combined = []
    fits = []
    for world in GPU_COUNTS:
        measurement = OUTPUT_ROOT / f"gpus-{world}" / "allreduce_measurements.csv"
        with measurement.open("r", encoding="utf-8-sig", newline="") as stream:
            combined.extend(csv.DictReader(stream))
        fits.append(
            json.loads(
                (OUTPUT_ROOT / f"gpus-{world}" / "alpha_beta_fit.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    write_csv(OUTPUT_ROOT / "allreduce_1to4gpu_measurements.csv", combined)
    write_csv(OUTPUT_ROOT / "alpha_beta_1to4gpu_summary.csv", fits)
    print(f"ALPHA-BETA 1/2/3/4-GPU COMPLETED: {OUTPUT_ROOT}")


if __name__ == "__main__":
    if os.environ.get(WORKER_FLAG) == "1":
        worker()
    else:
        parent()

