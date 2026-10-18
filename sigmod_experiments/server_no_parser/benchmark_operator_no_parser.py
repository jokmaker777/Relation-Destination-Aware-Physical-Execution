from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from benchmark_operator import run_benchmark


args = SimpleNamespace(
    dataset_json=os.environ["SIGMOD_DATASET_JSON"],
    output=os.environ["SIGMOD_OUTPUT"],
    policy=os.environ["SIGMOD_POLICY"],
    block_edges=int(os.environ["SIGMOD_BLOCK_EDGES"]),
    chunks=int(os.environ["SIGMOD_CHUNKS"]),
    collective_mode=os.environ["SIGMOD_COLLECTIVE_MODE"],
    aggregation=os.environ.get("SIGMOD_AGGREGATION", "mean"),
    hidden_dim=int(os.environ.get("SIGMOD_HIDDEN_DIM", "32")),
    warmups=int(os.environ.get("SIGMOD_WARMUPS", "2")),
    repeats=int(os.environ.get("SIGMOD_REPEATS", "5")),
    seed=int(os.environ.get("SIGMOD_SEED", "20261004")),
    validate_correctness=os.environ.get("SIGMOD_VALIDATE", "0") == "1",
    correctness_max_edges=int(
        os.environ.get("SIGMOD_CORRECTNESS_MAX_EDGES", "2000000")
    ),
)

json.loads(args.dataset_json)
run_benchmark(args)


