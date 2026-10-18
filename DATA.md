# Data Availability

## Public data

The public evaluation uses HGB-DBLP through the PyTorch-Geometric
`HGBDataset` interface.  Dataset roots are configured through
`SIGMOD_PUBLIC_DATA_ROOT` or the JSON files under
`sigmod_experiments/configs/`.

The artifact also contains deterministic synthetic graph generators for
scale, relation-count, degree-skew, hidden-dimension, and active-destination
sensitivity experiments.  Their seeds and workload grids are stored in the
corresponding launchers.

## Private industrial data

The twelve industrial snapshots cannot be redistributed.  They are therefore
represented only by snapshot identifiers and placeholder locations:

```text
/path/to/private_graphs/graph_<snapshot_id>.pkl
```

The private files, raw business identifiers, labels, logs, and cached tensors
are absent from this archive.  The public and synthetic experiments provide a
redistributable path for checking aggregation semantics, controlled scaling,
active-destination compaction, collective models, and planner logic.

## Generated outputs

All launchers write to `SIGMOD_OUTPUT_ROOT`.  Do not place private datasets
inside the code directory.  Before sharing generated outputs, scan them for
absolute paths, hostnames, account names, and raw graph attributes.

