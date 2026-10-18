from __future__ import annotations

from pathlib import Path


# 仅在能够访问互联网的机器上运行。可改成空间充足的本地目录。
DOWNLOAD_ROOT = Path(__file__).resolve().parent / "public_datasets_offline"

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import HGBDataset


hgb_root = DOWNLOAD_ROOT / "hgb"
ogb_root = DOWNLOAD_ROOT / "ogb"

print(f"Downloading HGB-DBLP to {hgb_root}")
HGBDataset(root=str(hgb_root), name="DBLP")

print(f"Downloading ogbn-mag to {ogb_root}")
PygNodePropPredDataset(name="ogbn-mag", root=str(ogb_root))

print("DOWNLOAD COMPLETE")
print(f"Upload the complete directory {hgb_root} to:")
print("  /path/to/artifact_workspace/public_datasets/hgb")
print(f"Upload the complete directory {ogb_root} to:")
print("  /path/to/artifact_workspace/public_datasets/ogb")


