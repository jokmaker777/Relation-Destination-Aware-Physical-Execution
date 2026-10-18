from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


EdgeType = Tuple[str, str, str]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    root: Optional[str] = None
    path: Optional[str] = None
    relation_selector: str = "topk:2"
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetSpec":
        return cls(
            name=str(value["name"]),
            kind=str(value["kind"]),
            root=value.get("root"),
            path=value.get("path"),
            relation_selector=str(value.get("relation_selector", "topk:2")),
            options=dict(value.get("options", {})),
        )


def edge_type_to_str(edge_type: EdgeType) -> str:
    return "__".join(str(x) for x in edge_type)


def _extract_heterodata(obj: Any):
    try:
        from torch_geometric.data import HeteroData
    except Exception as exc:  # pragma: no cover - actionable environment error
        raise RuntimeError("torch_geometric is required to load graph data") from exc

    if isinstance(obj, HeteroData):
        return obj
    if isinstance(obj, dict):
        for key in ("graph", "data", "hetero_data"):
            value = obj.get(key)
            if isinstance(value, HeteroData) or (
                hasattr(value, "node_types") and hasattr(value, "edge_types")
            ):
                return value
    if isinstance(obj, (tuple, list)):
        for value in obj:
            if isinstance(value, HeteroData) or (
                hasattr(value, "node_types") and hasattr(value, "edge_types")
            ):
                return value
    if hasattr(obj, "node_types") and hasattr(obj, "edge_types"):
        return obj
    raise TypeError(f"Unsupported graph container: {type(obj)!r}")


def _load_industrial(path: str):
    source = Path(path)
    candidates = [source, Path(str(source) + ".tmp")]
    errors: List[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("rb") as stream:
                return _extract_heterodata(pickle.load(stream))
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise FileNotFoundError(
        f"Could not load industrial graph {path!r}. "
        f"Checked {candidates}; errors={errors}"
    )


def _load_ogbn_mag(root: str):
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except Exception as exc:  # pragma: no cover - depends on server env
        raise RuntimeError("Install the public-data dependency with: pip install ogb") from exc
    dataset = PygNodePropPredDataset(name="ogbn-mag", root=root)
    return dataset[0]


def _load_hgb(name: str, root: str):
    try:
        from torch_geometric.datasets import HGBDataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("The installed torch_geometric does not provide HGBDataset") from exc
    dataset = HGBDataset(root=root, name=name)
    return dataset[0]


def _biased_destinations(
    num_edges: int,
    num_dst: int,
    skew: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if skew <= 0:
        return torch.randint(num_dst, (num_edges,), generator=generator)
    # A deterministic power transform gives a controlled head-heavy workload
    # without requiring scipy.  Larger skew concentrates more edges on small IDs.
    values = torch.rand((num_edges,), generator=generator, dtype=torch.float64)
    return torch.clamp((values.pow(1.0 + skew) * num_dst).long(), max=num_dst - 1)


def _load_synthetic(name: str, options: Mapping[str, Any]):
    try:
        from torch_geometric.data import HeteroData
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch_geometric is required for synthetic graphs") from exc

    num_src = int(options.get("num_src", 250_000))
    num_dst = int(options.get("num_dst", 250_000))
    num_edges = int(options.get("num_edges", 5_000_000))
    num_relations = int(options.get("num_relations", 4))
    skew = float(options.get("skew", 1.5))
    seed = int(options.get("seed", 20261003))
    if min(num_src, num_dst, num_edges, num_relations) <= 0:
        raise ValueError("Synthetic graph sizes must be positive")

    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["src"].num_nodes = num_src
    data["dst"].num_nodes = num_dst
    remaining = num_edges
    for rel_id in range(num_relations):
        rel_edges = remaining // (num_relations - rel_id)
        remaining -= rel_edges
        src = torch.randint(num_src, (rel_edges,), generator=generator)
        dst = _biased_destinations(rel_edges, num_dst, skew, generator)
        data[("src", f"rel_{rel_id}", "dst")].edge_index = torch.stack([src, dst])
    data.graph_name = name
    return data


def load_graph(spec: DatasetSpec):
    kind = spec.kind.lower()
    if kind in {"industrial", "industrial_pickle", "pickle"}:
        if not spec.path:
            raise ValueError(f"Dataset {spec.name} requires 'path'")
        return _load_industrial(spec.path)
    if kind in {"ogbn-mag", "ogbn_mag"}:
        return _load_ogbn_mag(spec.root or "data/ogb")
    if kind.startswith("hgb-"):
        return _load_hgb(kind.split("-", 1)[1], spec.root or "data/hgb")
    if kind == "synthetic":
        return _load_synthetic(spec.name, spec.options)
    raise ValueError(f"Unknown dataset kind: {spec.kind!r}")


def _num_nodes(data, node_type: str) -> int:
    store = data[node_type]
    value = getattr(store, "num_nodes", None)
    if value is not None:
        return int(value)
    x = getattr(store, "x", None)
    if isinstance(x, torch.Tensor):
        return int(x.shape[0])
    maximum = -1
    for edge_type in data.edge_types:
        edge_index = getattr(data[edge_type], "edge_index", None)
        if not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
            continue
        if edge_type[0] == node_type:
            maximum = max(maximum, int(edge_index[0].max().item()))
        if edge_type[2] == node_type:
            maximum = max(maximum, int(edge_index[1].max().item()))
    if maximum < 0:
        raise ValueError(f"Cannot infer num_nodes for node type {node_type!r}")
    return maximum + 1


def summarize_graph(data) -> Dict[str, Any]:
    node_counts = {node_type: _num_nodes(data, node_type) for node_type in data.node_types}
    relation_rows: List[Dict[str, Any]] = []
    total_edges = 0
    for edge_type in data.edge_types:
        edge_index = getattr(data[edge_type], "edge_index", None)
        count = int(edge_index.shape[1]) if isinstance(edge_index, torch.Tensor) else 0
        total_edges += count
        relation_rows.append(
            {
                "edge_type": edge_type_to_str(edge_type),
                "src_type": edge_type[0],
                "relation": edge_type[1],
                "dst_type": edge_type[2],
                "num_edges": count,
                "num_src": node_counts.get(edge_type[0], 0),
                "num_dst": node_counts.get(edge_type[2], 0),
            }
        )
    relation_rows.sort(key=lambda row: (-row["num_edges"], row["edge_type"]))
    return {
        "num_node_types": len(data.node_types),
        "num_edge_types": len(data.edge_types),
        "num_nodes": int(sum(node_counts.values())),
        "num_edges": int(total_edges),
        "node_counts": node_counts,
        "relations": relation_rows,
    }


def select_relations(data, selector: str) -> List[EdgeType]:
    relations = []
    for edge_type in data.edge_types:
        edge_index = getattr(data[edge_type], "edge_index", None)
        count = int(edge_index.shape[1]) if isinstance(edge_index, torch.Tensor) else 0
        if count > 0:
            relations.append((edge_type, count))
    relations.sort(key=lambda item: (-item[1], edge_type_to_str(item[0])))

    selector = str(selector or "topk:2").strip()
    if selector == "all":
        return [edge_type for edge_type, _ in relations]
    if selector.startswith("topk:"):
        k = max(1, int(selector.split(":", 1)[1]))
        return [edge_type for edge_type, _ in relations[:k]]

    requested = {value.strip() for value in selector.split(",") if value.strip()}
    selected = [edge_type for edge_type, _ in relations if edge_type_to_str(edge_type) in requested]
    missing = requested - {edge_type_to_str(edge_type) for edge_type in selected}
    if missing:
        raise KeyError(f"Relations not present in graph: {sorted(missing)}")
    return selected


def dataset_spec_json(spec: DatasetSpec) -> str:
    return json.dumps(
        {
            "name": spec.name,
            "kind": spec.kind,
            "root": spec.root,
            "path": spec.path,
            "relation_selector": spec.relation_selector,
            "options": dict(spec.options),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


