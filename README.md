# Anonymous SIGMOD Code Artifact

This package contains the implementation, experiment launchers, validation
tests, planners, result collectors, and figure-generation code used by the
paper.  Author names, account names, host-specific deployment directories,
private graph files, generated results, IDE metadata, and local caches are not
included.

## Directory layout

- `end_to_end_prunning/`: training, relation--destination aggregation,
  SuperBlock placement, and communication scheduling implementation.
- `sigmod_experiments/`: reusable public/industrial benchmark layer,
  plan collectors, planner evaluation, and tests.
- `run_*.py`: direct-run GPU experiment launchers used for the paper.
- `analysis/`: aggregation and paper-figure generation scripts.
- `config/`: placeholder paths and environment notes.
- `ARTIFACT_MANIFEST.md`: paper-claim-to-script mapping.
- `DATA.md`: public and private data policy.

## Path configuration

No author or server path is embedded in this package.  Replace the placeholder
`/path/to/artifact_workspace`, or define these variables before running:

```text
SIGMOD_ARTIFACT_ROOT=/path/to/artifact_workspace
SIGMOD_GRAPH_CACHE_ROOT=/path/to/private_graphs
SIGMOD_PUBLIC_DATA_ROOT=/path/to/public_datasets
SIGMOD_OUTPUT_ROOT=/path/to/outputs
SIGMOD_RESULTS_ROOT=/path/to/collected_results
SIGMOD_FIGURE_OUTPUT_ROOT=/path/to/generated_figures
```

The no-command-line-argument workflow is configured in
`sigmod_experiments/server_no_parser/settings_no_parser.py`.  GPU identifiers,
snapshot identifiers, repetitions, and search grids are intentionally visible
there so that the effective configuration can be audited before a run.

Private graph files are expected to use the neutral names
`graph_<snapshot_id>.pkl`, for example `graph_20241129.pkl`.  Adapt the helper
`industrial_graph_path` if a different local naming convention is used.

## Environment

The reported environment used Python 3.10 and PyTorch 2.4.1.  Install a
PyTorch-Geometric build compatible with the local CUDA/accelerator backend;
binary extension wheels must match the installed PyTorch build.  The remaining
Python dependencies are listed in `requirements.txt`.

Record the effective environment before experiments:

```text
python -m torch.utils.collect_env > /path/to/outputs/environment.txt
```

## Minimal validation order

1. Edit `sigmod_experiments/server_no_parser/settings_no_parser.py` or set the
   environment variables above.
2. Run `python -m pytest -q sigmod_experiments/tests`.
3. Run `python sigmod_experiments/server_no_parser/00_preflight_no_parser.py`.
4. Run `python sigmod_experiments/server_no_parser/01_correctness_1to4gpu_no_parser.py`.
5. Run `python run_sigmod_core_suite_no_parser.py` after reviewing
   `RUN_PROFILE` and `STAGES_TO_RUN` at the top of that file.
6. Run the additional controlled and memory experiments listed in
   `ARTIFACT_MANIFEST.md`.
7. Set `SIGMOD_RESULTS_ROOT` and run
   `python analysis/build_sigmod_paper_figures.py` to regenerate tables and
   figures from collected result summaries.

Every long-running launcher is restartable and skips completed cells.  Logs of
failed configurations should be retained because OOM and feasibility outcomes
are part of the evaluation.

## Important scope notes

- The industrial dataset is private and is not distributed.
- Public HGB-DBLP and controlled synthetic paths support independent semantic
  and mechanism validation.
- The package contains code only; generated measurements and trained artifacts
  are intentionally excluded from this archive.
- A software license has not been asserted in this anonymous package.  Add the
  intended license before public release.

