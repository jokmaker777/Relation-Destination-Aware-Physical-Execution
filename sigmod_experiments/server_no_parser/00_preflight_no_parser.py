from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from settings_no_parser import (
    BASE_OUTPUT_ROOT,
    GPU_IDS,
    GRAPH_PATH,
    GRAPH_TAG,
    INDUSTRIAL_RUNNER,
    INDUSTRIAL_RUNNER_CANDIDATES,
    LABELED_RELATIONS,
    METADATA_PATH,
)
from sigmod_exp.datasets import DatasetSpec, load_graph, summarize_graph


print("=== SIGMOD server preflight ===")
print(f"graph={GRAPH_PATH}")
print(f"runner={INDUSTRIAL_RUNNER}")
print(f"output={BASE_OUTPUT_ROOT}")

if not GRAPH_PATH.exists() and not Path(str(GRAPH_PATH) + ".tmp").exists():
    raise FileNotFoundError(GRAPH_PATH)
runner_exists = INDUSTRIAL_RUNNER.exists()
if not runner_exists:
    print("WARNING: 未找到工业训练主脚本。已检查：")
    for candidate in INDUSTRIAL_RUNNER_CANDIDATES:
        print(f"  {candidate}")
    print("图检查仍会继续并生成 metadata；工业 pilot 前必须补齐该脚本。")
if len(GPU_IDS) < 4:
    raise RuntimeError("GPU_IDS 至少需要填写四张卡，才能运行 1/2/3/4 卡实验")

visible = torch.cuda.device_count()
print(f"visible_cuda_devices={visible}")
for index in range(visible):
    print(f"  cuda:{index} = {torch.cuda.get_device_name(index)}")
if visible < 4:
    raise RuntimeError("当前进程看不到四张 GPU，请检查登录节点/作业节点和 CUDA_VISIBLE_DEVICES")

spec = DatasetSpec(
    name=GRAPH_TAG,
    kind="industrial",
    path=str(GRAPH_PATH),
    relation_selector="all",
)
summary = summarize_graph(load_graph(spec))
relation_counts = {
    row["edge_type"]: int(row["num_edges"]) for row in summary["relations"]
}
missing = [value for value in LABELED_RELATIONS if value not in relation_counts]
if missing:
    raise RuntimeError(f"工业图缺少 labeled relations: {missing}")
observed = sum(relation_counts[value] for value in LABELED_RELATIONS)

metadata = {
    "graph_tag": GRAPH_TAG,
    "graph_path": str(GRAPH_PATH),
    "observed_labeled_edges": observed,
    "labeled_relations": list(LABELED_RELATIONS),
    "num_nodes": summary["num_nodes"],
    "num_edges_all_relations": summary["num_edges"],
    "num_node_types": summary["num_node_types"],
    "num_edge_types": summary["num_edge_types"],
    "relations": summary["relations"],
    "gpu_ids": GPU_IDS,
    "industrial_runner": str(INDUSTRIAL_RUNNER),
    "industrial_runner_exists": runner_exists,
}
METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"nodes={summary['num_nodes']:,}")
print(f"all_edges={summary['num_edges']:,}")
print(f"observed_labeled_edges={observed:,}")
print(f"metadata={METADATA_PATH}")
if runner_exists:
    print("PREFLIGHT PASSED")
else:
    print("GRAPH/GPU PREFLIGHT PASSED; INDUSTRIAL RUNNER STILL MISSING")

