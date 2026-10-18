# SIGMOD experiment suite

This directory contains a clean, resumable experiment layer for the SIGMOD
version of the multi-GPU paper.  It does not modify the original experiment
outputs.

For the fixed SSH-server workflow using the 12 snapshots from 20241124--20241129
and 20241213--20241218, use the numbered scripts in `server_no_parser/`.  They
also retain the 20241129-only pilot path, require no command-line arguments, and are documented in
`server_no_parser/README_服务器直接运行.md`.

The suite has two complementary paths:

1. `launch_industrial_matrix.py` reuses the existing end-to-end supply-chain training
   implementation and runs the joint `(partition policy, block budget,
   destination ranges, GPU count)` matrix on the five anonymized snapshots.
2. `benchmark_operator.py` is a dataset-independent exact heterogeneous
   aggregation benchmark.  It supports the anonymized graphs, OGBN-MAG, HGB
   datasets, and controlled synthetic graphs on 1/2/3/4 GPUs.

The public-data benchmark is necessary because five adjacent private snapshots
cannot by themselves establish external validity or reproducibility.  The
recommended minimum for the submission is:

- five anonymized industrial snapshots for the full end-to-end case study;
- OGBN-MAG as the main public heterogeneous graph;
- HGB-DBLP as a smaller reproducible smoke/portability workload;
- controlled synthetic graphs for degree skew, relation count, and scale.

## Environment

Use the same PyTorch/PyG environment as the original project and install the
small extra dependency used by OGBN-MAG:

```bash
python -m pip install ogb
```

Record the exact environment before the final run:

```bash
python -m torch.utils.collect_env > environment.txt
nvidia-smi -q > nvidia-smi.txt
```

## 1. CPU smoke test

First validate that every industrial cache exists and that the configured
labeled-edge count matches the graph.  This check is mandatory because the
uploaded 20241218 script default and manuscript table use different totals.

```bash
python inspect_industrial_graphs.py \
  --config configs/industrial_matrix.sigmod.json \
  --output /path/to/artifact_workspace/sigmod2027/industrial_graph_inventory.json
```

Correct the JSON configuration if this command reports `MISMATCH`.  Then run
the CPU partition tests before consuming GPU time:

```bash
python -m pytest -q tests/test_cpu_smoke.py
```

## 2. Public/synthetic operator matrix

Edit paths in `configs/operator_matrix.sigmod.json`, download/validate each
dataset once in a single process, then preview the jobs:

```bash
python prepare_datasets.py --config configs/operator_matrix.sigmod.json
```

```bash
python launch_operator_matrix.py \
  --config configs/operator_matrix.sigmod.json \
  --dry-run
```

Run a small validation slice first:

```bash
python launch_operator_matrix.py \
  --config configs/operator_matrix.sigmod.json \
  --max-jobs 8
```

Resume the complete run with the same command but without `--max-jobs`.
Completed result files are skipped automatically.

Collect the completed public/synthetic runs with:

```bash
python collect_operator_results.py \
  --root /path/to/artifact_workspace/sigmod2027/operator \
  --output /path/to/artifact_workspace/sigmod2027/operator_measurements.csv
```

## 3. Industrial end-to-end matrix

The launcher calls the existing
`run_forward_aggregation_overlap_5day_seq_chunk10_repeat2_ep10_no_parser.py`
in child mode, so it does not use the incomplete hard-coded parent graph list.

```bash
python launch_industrial_matrix.py \
  --config configs/industrial_matrix.sigmod.json \
  --dry-run

python launch_industrial_matrix.py \
  --config configs/industrial_matrix.sigmod.json \
  --sweep scale_primary --repeat 1 --max-jobs 8
```

The industrial launcher also accepts repeatable filters: `--graph`,
`--gpu-count`, `--strategy`, `--mode`, `--block-edges`, and `--repeat`.  Use
`configs/operator_sensitivity.sigmod.json` after the primary matrix for the
controlled graph-size, skew, relation-count, and hidden-dimension study.

The full configuration covers 1/2/3/4 GPUs.  Do not report 8-GPU scalability
unless it is actually measured.

## 4. Collect measurements and evaluate the joint planner

```bash
python collect_industrial_results.py \
  --root /path/to/artifact_workspace/sigmod2027/industrial \
  --output /path/to/artifact_workspace/sigmod2027/plan_measurements.csv \
  --reuse-epochs 50

python joint_planner.py \
  --input /path/to/artifact_workspace/sigmod2027/plan_measurements.csv \
  --output-dir /path/to/artifact_workspace/sigmod2027/planner_eval \
  --reuse-epochs 50
```

`joint_planner.py` performs leave-one-graph-out evaluation and reports top-1
regret for the joint model, a two-stage selector, a fixed default, and the
measured oracle.

## Experimental order

1. Smoke test and correctness validation on synthetic data.
2. One industrial snapshot on 1 and 2 GPUs.
3. OGBN-MAG on 1 and 2 GPUs.
4. Extend successful plans to 3 and 4 GPUs.
5. Run all five industrial snapshots and three independent repeats.
6. Run HGB-DBLP and the synthetic skew/scale sweep.

Failures must remain in the launcher logs.  Do not silently drop failed plans
from the paper; report the feasibility boundary (usually memory) explicitly.

