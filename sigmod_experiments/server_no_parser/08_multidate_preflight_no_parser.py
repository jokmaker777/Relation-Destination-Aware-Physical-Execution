from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from settings_no_parser import (
    LABELED_RELATIONS,
    MULTI_DATE_GRAPH_TAGS,
    MULTIDATE_METADATA_PATH,
    industrial_graph_path,
)
from sigmod_exp.datasets import DatasetSpec, load_graph, summarize_graph


missing_paths = []
for graph_tag in MULTI_DATE_GRAPH_TAGS:
    graph_path = industrial_graph_path(graph_tag)
    if not graph_path.exists() and not Path(str(graph_path) + ".tmp").exists():
        missing_paths.append(str(graph_path))
if missing_paths:
    print("以下日期图文件不存在：")
    for path in missing_paths:
        print(f"  {path}")
    raise FileNotFoundError("多日期预检查失败；请补齐文件或修改 settings_no_parser.py")

rows = []
for index, graph_tag in enumerate(MULTI_DATE_GRAPH_TAGS, 1):
    graph_path = industrial_graph_path(graph_tag)
    print(f"[{index}/{len(MULTI_DATE_GRAPH_TAGS)}] loading {graph_tag}: {graph_path}")
    started = time.time()
    data = load_graph(
        DatasetSpec(
            name=graph_tag,
            kind="industrial",
            path=str(graph_path),
            relation_selector="all",
        )
    )
    summary = summarize_graph(data)
    relation_counts = {
        row["edge_type"]: int(row["num_edges"]) for row in summary["relations"]
    }
    missing_relations = [
        relation for relation in LABELED_RELATIONS if relation not in relation_counts
    ]
    if missing_relations:
        raise RuntimeError(f"{graph_tag} 缺少 labeled relations: {missing_relations}")
    observed = sum(relation_counts[relation] for relation in LABELED_RELATIONS)
    row = {
        "graph_tag": graph_tag,
        "graph_path": str(graph_path),
        "observed_labeled_edges": observed,
        "num_nodes": summary["num_nodes"],
        "num_edges_all_relations": summary["num_edges"],
        "num_node_types": summary["num_node_types"],
        "num_edge_types": summary["num_edge_types"],
        "load_and_inspect_sec": time.time() - started,
    }
    rows.append(row)
    print(
        f"  labeled_edges={observed:,}, all_edges={summary['num_edges']:,}, "
        f"nodes={summary['num_nodes']:,}, elapsed={row['load_and_inspect_sec']:.1f}s"
    )
    del data
    gc.collect()

MULTIDATE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary = MULTIDATE_METADATA_PATH.with_suffix(".tmp")
temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
temporary.replace(MULTIDATE_METADATA_PATH)
print(f"WROTE {MULTIDATE_METADATA_PATH}")
print("ALL 12 INDUSTRIAL GRAPHS PASSED PREFLIGHT")


