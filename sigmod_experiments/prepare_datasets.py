from __future__ import annotations

import argparse
import json
from pathlib import Path

from sigmod_exp.datasets import DatasetSpec, load_graph, summarize_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/validate datasets before torchrun")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    selected = set(args.dataset or [])
    for value in config["datasets"]:
        if selected and value["name"] not in selected:
            continue
        spec = DatasetSpec.from_dict(value)
        summary = summarize_graph(load_graph(spec))
        print(
            f"{spec.name}: nodes={summary['num_nodes']:,}, "
            f"edges={summary['num_edges']:,}, relations={summary['num_edge_types']}"
        )


if __name__ == "__main__":
    main()

