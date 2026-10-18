from __future__ import annotations

import argparse
import json
from pathlib import Path

from sigmod_exp.datasets import DatasetSpec, load_graph, summarize_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate industrial cache paths and configured labeled-edge counts"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--graph", action="append", help="Only inspect named graph")
    parser.add_argument("--output", help="Optional JSON report")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    selected = set(args.graph or [])
    labeled_relations = set(config["labeled_relations"])
    report = []
    mismatches = 0

    for graph in config["graphs"]:
        if selected and graph["name"] not in selected:
            continue
        spec = DatasetSpec(
            name=graph["name"],
            kind="industrial",
            path=graph["path"],
            relation_selector="all",
        )
        summary = summarize_graph(load_graph(spec))
        observed = sum(
            int(row["num_edges"])
            for row in summary["relations"]
            if row["edge_type"] in labeled_relations
        )
        configured = int(graph["total_labeled_edges"])
        missing_relations = sorted(
            labeled_relations
            - {row["edge_type"] for row in summary["relations"]}
        )
        matches = configured == observed and not missing_relations
        mismatches += int(not matches)
        row = {
            "graph": graph["name"],
            "path": graph["path"],
            "configured_labeled_edges": configured,
            "observed_labeled_edges": observed,
            "matches": matches,
            "missing_labeled_relations": missing_relations,
            "num_nodes": summary["num_nodes"],
            "num_edges_all_relations": summary["num_edges"],
            "num_edge_types": summary["num_edge_types"],
        }
        report.append(row)
        print(
            f"{graph['name']}: configured={configured:,}, observed={observed:,}, "
            f"all_edges={summary['num_edges']:,}, status={'OK' if matches else 'MISMATCH'}"
        )
        if missing_relations:
            print(f"  missing relations: {missing_relations}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if mismatches:
        raise SystemExit(
            f"Found {mismatches} mismatch(es). Correct the config before formal runs."
        )


if __name__ == "__main__":
    main()

