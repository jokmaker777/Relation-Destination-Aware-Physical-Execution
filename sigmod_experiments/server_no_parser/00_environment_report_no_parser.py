from __future__ import annotations

import importlib
import platform
import sys

import torch
import torch_geometric


print(f"python={sys.version}")
print(f"platform={platform.platform()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"torch_geometric={torch_geometric.__version__}")

for package in (
    "pyg_lib",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "torch_spline_conv",
):
    try:
        module = importlib.import_module(package)
        print(f"{package}=OK version={getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{package}=BROKEN {type(exc).__name__}: {exc}")


