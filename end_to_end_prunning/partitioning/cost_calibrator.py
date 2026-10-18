from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.exceptions import ConvergenceWarning

from .edge_block import EdgeBlock


def _sanitize_array(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("观测时间包含非有限值")
    return arr


def _build_design_matrix(blocks: List[EdgeBlock], base_feat: Optional[np.ndarray] = None) -> np.ndarray:
    rows = []
    for block in blocks:
        row = [
            1.0,
            float(block.num_edges),
            float(block.edge_feat_dim),
            float(block.num_edges * block.edge_feat_dim),
            float(block.num_labeled_edges),
        ]
        rows.append(row)
    design = np.asarray(rows, dtype=np.float64)
    if base_feat is not None:
        if base_feat.shape[0] != len(blocks):
            raise ValueError("base_feat 长度与 blocks 数量不一致")
        if base_feat.ndim == 1:
            base_feat = base_feat.reshape(-1, 1)
        design = np.concatenate([design, base_feat.astype(np.float64, copy=False)], axis=1)
    return design


def fit_edge_block_cost_model(
    blocks: List[EdgeBlock],
    observed_times: List[float],
    *,
    method: str = "ridge",
    positive: bool = True,
    alpha: float = 1.0,
    base_feat: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """拟合边块计算 cost model。"""
    if len(blocks) != len(observed_times):
        raise ValueError("blocks 与 observed_times 长度不一致")

    y = _sanitize_array(observed_times)
    X = _build_design_matrix(blocks, base_feat)
    if X.shape[0] < 2:
        raise ValueError("样本数量不足，至少需要 2 个边块样本")

    method = str(method).lower()
    if method not in {"ridge", "linear"}:
        raise ValueError("method 仅支持 ridge 或 linear")

    reg = Ridge(alpha=float(alpha), positive=bool(positive)) if method == "ridge" else LinearRegression(positive=bool(positive))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        reg.fit(X, y)

    preds = reg.predict(X)
    residuals = y - preds
    sse = float(np.sum(residuals ** 2))
    r2 = 1.0 - sse / float(np.sum((y - np.mean(y)) ** 2)) if float(np.sum((y - np.mean(y)) ** 2)) > 0 else 0.0
    mean_abs_error = float(np.mean(np.abs(residuals)))

    return reg.coef_, {
        "intercept": float(reg.intercept_),
        "r2": float(r2),
        "mean_abs_error": mean_abs_error,
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "method": method,
        "positive": positive,
        "alpha": float(alpha),
        "predicted": preds.tolist(),
        "observed": y.tolist(),
    }


def apply_cost_model(
    blocks: List[EdgeBlock],
    coef: np.ndarray,
    *,
    base_feat: Optional[np.ndarray] = None,
) -> None:
    """把拟合结果写回每个 EdgeBlock.compute_cost。"""
    X = _build_design_matrix(blocks, base_feat)
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    if coef.shape[0] != X.shape[1]:
        raise ValueError(f"coef 维度 {coef.shape[0]} 与特征维度 {X.shape[1]} 不一致")

    values = X @ coef
    for block, value in zip(blocks, values):
        block.compute_cost = float(value)

