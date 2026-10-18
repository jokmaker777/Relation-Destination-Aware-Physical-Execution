#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_forward_aggregation_overlap_5day_seq_chunk10_repeat2_tradeoff_model_no_parser.py

适配运行脚本：
  run_forward_aggregation_overlap_5day_seq_chunk10_repeat2_ep10_no_parser.py

目标实验：
  5 dates × 2 repeats ×
  [sequential_aggregation, async_chunk_2, ..., async_chunk_10] × 10 epochs。

默认预期日期：
  20241213, 20241214, 20241215, 20241217, 20241218

核心统计口径：
1. epoch 1 为 cold-start，epoch 2–10 为 steady epochs；
2. stage timing 在同一 epoch 内取各 rank 最大值，表示同步训练关键路径时间；
3. forward aggregation timing 先按“epoch × rank”汇总全部 layer/relation，
   再选取 forward pipeline 最慢的 critical rank；
4. sequential_aggregation 的 hidden communication 强制设为 0，
   exposed communication 等于 total communication；
5. 同一 graph × repeat 内，将各 async chunk 与 Sequential 配对；
6. 建立统一权衡分解：

       H(C) = hidden communication benefit
       J(C) = total communication inflation relative to Sequential
       D(C) = added local-message computation

       G_agg_no_overhead(C) = H(C) - J(C) - D(C)

   并同时保留 residual overhead、backward 和 rest 阶段，形成端到端分解；
7. 对 H(C)、D(C) 拟合饱和指数模型，对 backward 增量拟合线性模型；
8. J(C) 波动较大，不强制使用单一参数曲线，而采用：
   - 每个 chunk 的经验均值与方差；
   - graph/repeat 固定效应诊断；
   - 风险调整后的 chunk 选择；
9. 若风险调整收益不大于 0，则自动回退 Sequential。

主要输出：
  - experiment_completeness.csv
  - repeat_mode_steady_summary.csv
  - paired_repeat_comparison_vs_sequential.csv
  - tradeoff_observations.csv
  - tradeoff_chunk_summary.csv
  - tradeoff_model_parameters.csv
  - tradeoff_model_fit_metrics.csv
  - tradeoff_curve_predictions.csv
  - per_graph_chunk_selection.csv
  - per_run_best_chunk.csv
  - report.md
  - 若 matplotlib 可用，输出 3 张诊断图。

不使用 argparse。可通过环境变量覆盖：
  INPUT_ROOT
  INPUT_ROOTS
  OUTPUT_DIR
  STEADY_EPOCH_START
  EXPECTED_GRAPH_TAGS       逗号分隔
  EXPECTED_REPEATS          例如 1,2
  EXPECTED_ASYNC_CHUNKS     例如 2,3,4,5,6,7,8,9,10
  RISK_LAMBDA               风险惩罚系数，默认 1.0
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROFILE = "5day_seq_chunk10_repeat2_hidden_compute_tradeoff"
BASELINE_MODE = "sequential_aggregation"
STEADY_EPOCH_START = int(os.environ.get("STEADY_EPOCH_START", "2"))
DEFAULT_INPUT_ROOT = Path(r"/path/to/artifact_workspace/result_forward_aggregation_overlap_q25k_5day_seq_chunk10_repeat2_ep10")
INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", str(DEFAULT_INPUT_ROOT)))

_raw_roots = os.environ.get("INPUT_ROOTS", "").strip()
if _raw_roots:
    INPUT_ROOTS = [Path(x) for x in _raw_roots.split(os.pathsep) if x.strip()]
else:
    INPUT_ROOTS = [INPUT_ROOT]

DEFAULT_OUTPUT_NAME = "forward_aggregation_overlap_tradeoff_analysis_5day_repeat2"
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    str(INPUT_ROOT / DEFAULT_OUTPUT_NAME),
))

EXPECTED_GRAPH_TAG = os.environ.get("EXPECTED_GRAPH_TAG", "20241218").strip()
EXPECTED_GRAPH_TAGS = [
    x.strip()
    for x in os.environ.get(
        "EXPECTED_GRAPH_TAGS",
        "20241213,20241214,20241215,20241217,20241218",
    ).split(",")
    if x.strip()
]
RISK_LAMBDA = float(os.environ.get("RISK_LAMBDA", "1.0"))
EXPECTED_REPEATS = [
    int(x.strip())
    for x in os.environ.get("EXPECTED_REPEATS", "1,2").split(",")
    if x.strip()
]
EXPECTED_ASYNC_CHUNKS = [
    int(x.strip())
    for x in os.environ.get(
        "EXPECTED_ASYNC_CHUNKS", "2,3,4,5,6,7,8,9,10"
    ).split(",")
    if x.strip()
]
EXPECTED_MODES = [BASELINE_MODE] + [
    f"async_chunk_{c}" for c in EXPECTED_ASYNC_CHUNKS
]
EXPECTED_TOTAL_JOBS = len(EXPECTED_REPEATS) * len(EXPECTED_MODES)

KNOWN_STAGE_METRICS = [
    "epoch_total_sec",
    "data_sec",
    "forward_loss_sec",
    "backward_sec",
    "optimizer_sec",
    "loss_sync_sec",
    "sync_loss_sec",
    "partition_sec",
]
AGG_TIME_METRICS = [
    "local_message_compute_sec",
    "aggregation_comm_total_sec",
    "aggregation_comm_exposed_sec",
    "aggregation_comm_hidden_sec",
    "aggregation_wait_sec",
    "forward_pipeline_total_sec",
    "pipeline_overhead_sec",
    # Comparable across sequential and async modes:
    # pipeline total - local compute - exposed communication.
    "pipeline_residual_overhead_sec",
]
AGG_ALIAS_MAP = {
    "comm_total_sec": "aggregation_comm_total_sec",
    "allreduce_total_sec": "aggregation_comm_total_sec",
    "communication_total_sec": "aggregation_comm_total_sec",
    "comm_exposed_sec": "aggregation_comm_exposed_sec",
    "communication_exposed_sec": "aggregation_comm_exposed_sec",
    "comm_hidden_sec": "aggregation_comm_hidden_sec",
    "communication_hidden_sec": "aggregation_comm_hidden_sec",
    "message_compute_sec": "local_message_compute_sec",
    "local_compute_sec": "local_message_compute_sec",
    "pipeline_total_sec": "forward_pipeline_total_sec",
    "overhead_sec": "pipeline_overhead_sec",
}


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        frame = pd.read_csv(path)
        return frame if not frame.empty else None
    except Exception:
        return None


def mode_order(mode: Any) -> Tuple[int, int]:
    text = str(mode)
    if text == BASELINE_MODE:
        return (0, 0)
    match = re.search(r"async_chunk_(\d+)", text)
    if match:
        return (1, int(match.group(1)))
    return (2, 9999)


def chunk_count_from_mode(mode: Any, fallback: Any = np.nan) -> Any:
    match = re.search(r"async_chunk_(\d+)", str(mode))
    if match:
        return int(match.group(1))
    if str(mode) == BASELINE_MODE:
        try:
            return int(fallback)
        except Exception:
            return np.nan
    return fallback



def configuration_order(mode: Any) -> int:
    text = str(mode)
    if text == BASELINE_MODE:
        return 0
    match = re.search(r"async_chunk_(\d+)", text)
    return int(match.group(1)) if match else 9999


def configuration_label(mode: Any) -> str:
    text = str(mode)
    if text == BASELINE_MODE:
        return "Seq"
    match = re.search(r"async_chunk_(\d+)", text)
    return f"C={match.group(1)}" if match else text


def add_configuration_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "mode" not in frame.columns:
        return frame
    out = frame.copy()
    out["configuration_order"] = out["mode"].map(configuration_order)
    out["configuration_label"] = out["mode"].map(configuration_label)
    return out.sort_values(
        ["configuration_order", "mode"],
        kind="stable",
    ).reset_index(drop=True)


def parse_rank(path: Path) -> int:
    match = re.search(r"rank(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def parse_name_metadata(name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    date_match = re.search(r"(?<!\d)(20\d{6})(?!\d)", name)
    if date_match:
        out["graph_tag"] = date_match.group(1)

    mode_match = re.search(r"(sequential_aggregation|async_chunk_\d+)", name)
    if mode_match:
        out["mode"] = mode_match.group(1)

    patterns = {
        "repeat_id": r"(?:^|_)R0*(\d+)(?:_|$)",
        "target_num_blocks": r"(?:^|_)K(\d+)(?:_|$)",
        "target_super_block_size": r"(?:^|_)Q(\d+)(?:_|$)",
        "num_epochs": r"(?:^|_)ep(\d+)(?:_|$)",
        "run_seed": r"(?:^|_)seed(\d+)(?:_|$)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            out[key] = int(match.group(1))
    return out


def infer_metadata(exp_dir: Path, root: Path) -> Dict[str, Any]:
    meta = read_json(exp_dir / "train_result_rank0.json")
    parsed = parse_name_metadata(exp_dir.name)

    def first(*values: Any) -> Any:
        for value in values:
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return value
        return None

    mode = first(
        meta.get("mode"),
        meta.get("experiment_mode"),
        meta.get("steady_experiment_mode"),
        parsed.get("mode"),
        "unknown",
    )
    num_chunks = first(
        meta.get("aggregation_num_chunks"),
        meta.get("num_chunks"),
        chunk_count_from_mode(mode),
    )
    return {
        "input_root": str(root),
        "output_dir": str(exp_dir),
        "mode": str(mode),
        "num_chunks": num_chunks,
        "repeat_id": first(meta.get("repeat_id"), parsed.get("repeat_id"), 1),
        "graph_tag": str(first(meta.get("graph_tag"), meta.get("graph_date"), parsed.get("graph_tag"), "unknown")),
        "strategy": first(meta.get("strategy"), meta.get("partition_strategy")),
        "target_super_block_size": first(meta.get("target_super_block_size"), parsed.get("target_super_block_size")),
        "target_num_blocks": first(meta.get("target_num_blocks"), parsed.get("target_num_blocks")),
        "num_epochs": first(meta.get("num_epochs"), parsed.get("num_epochs")),
        "run_seed": first(meta.get("run_seed"), meta.get("base_seed"), parsed.get("run_seed")),
        "total_labeled_edges": meta.get("total_labeled_edges"),
        "peak_memory_mb": meta.get("peak_memory_mb"),
        "final_loss": meta.get("final_loss_extracted"),
        "elapsed_sec": meta.get("elapsed_sec"),
    }


def is_analysis_dir(path: Path) -> bool:
    lowered = path.name.lower()
    return (
        "analysis" in lowered
        or lowered in {"forward_agg_parent", "per_date", "logs"}
    )


def experiment_dirs() -> List[Tuple[Path, Path]]:
    found: List[Tuple[Path, Path]] = []
    seen: set[str] = set()
    for root in INPUT_ROOTS:
        if not root.exists():
            continue
        candidates: Iterable[Path] = root.iterdir()
        for path in candidates:
            if not path.is_dir() or is_analysis_dir(path):
                continue
            has_result = (path / "train_result_rank0.json").exists()
            has_stage = any(path.glob("stage_timing_rank*.csv"))
            has_agg = bool(find_agg_files(path))
            if has_result or has_stage or has_agg:
                resolved = str(path.resolve())
                if resolved not in seen:
                    found.append((root, path))
                    seen.add(resolved)
    return sorted(found, key=lambda item: str(item[1]))


def find_agg_files(exp_dir: Path) -> List[Path]:
    patterns = [
        "forward_agg_overlap_timing_rank*.csv",
        "forward_aggregation_overlap_timing_rank*.csv",
        "*forward*agg*timing*rank*.csv",
    ]
    files: List[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in exp_dir.glob(pattern):
            if "stage_timing" in path.name:
                continue
            key = str(path.resolve())
            if key not in seen:
                files.append(path)
                seen.add(key)
    return sorted(files)


def numeric_series(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce")


def summary_stats(values: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {
            "mean": np.nan, "std": np.nan, "median": np.nan,
            "min": np.nan, "max": np.nan, "cv_pct": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan,
        }
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    half = 1.96 * std / math.sqrt(len(x)) if len(x) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "median": float(x.median()),
        "min": float(x.min()),
        "max": float(x.max()),
        "cv_pct": float(std / mean * 100.0) if mean != 0 else np.nan,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def add_stats(record: Dict[str, Any], prefix: str, values: pd.Series) -> None:
    for key, value in summary_stats(values).items():
        record[f"{prefix}_{key}"] = value


def normalize_agg_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rename = {
        old: new
        for old, new in AGG_ALIAS_MAP.items()
        if old in out.columns and new not in out.columns
    }
    if rename:
        out = out.rename(columns=rename)
    for col in AGG_TIME_METRICS + ["aggregation_overlap_ratio"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_overlap_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = normalize_agg_columns(frame)
    if "mode" not in out.columns:
        return out

    seq = out["mode"].astype(str).eq(BASELINE_MODE)
    if "aggregation_comm_total_sec" in out.columns:
        total = out["aggregation_comm_total_sec"].clip(lower=0.0)
        out.loc[seq, "aggregation_comm_hidden_sec"] = 0.0
        out.loc[seq, "aggregation_comm_exposed_sec"] = total.loc[seq]
        out.loc[seq, "aggregation_overlap_ratio"] = 0.0

        async_mask = ~seq
        if "aggregation_comm_hidden_sec" in out.columns:
            hidden = out["aggregation_comm_hidden_sec"].clip(lower=0.0)
            out["aggregation_comm_hidden_sec"] = hidden
            valid = async_mask & total.gt(0.0)
            out.loc[valid, "aggregation_overlap_ratio"] = (
                hidden.loc[valid] / total.loc[valid]
            ).clip(0.0, 1.0)
            out.loc[async_mask & ~total.gt(0.0), "aggregation_overlap_ratio"] = 0.0
        if "aggregation_comm_exposed_sec" in out.columns:
            out["aggregation_comm_exposed_sec"] = out[
                "aggregation_comm_exposed_sec"
            ].clip(lower=0.0)
    out["overlap_metric_valid"] = ~seq
    return out


def collect_results() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage_rows: List[Dict[str, Any]] = []
    agg_frames: List[pd.DataFrame] = []
    run_rows: List[Dict[str, Any]] = []

    for root, exp_dir in experiment_dirs():
        meta = infer_metadata(exp_dir, root)
        run_rows.append(meta)

        rank_frames: List[pd.DataFrame] = []
        for timing_path in sorted(exp_dir.glob("stage_timing_rank*.csv")):
            frame = read_csv(timing_path)
            if frame is None:
                continue
            frame = frame.copy()
            frame["rank"] = parse_rank(timing_path)
            if "epoch" not in frame.columns:
                frame["epoch"] = frame.groupby("rank").cumcount() + 1
            frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
            frame = frame.dropna(subset=["epoch"])
            frame["epoch"] = frame["epoch"].astype(int)
            rank_frames.append(frame)

        if rank_frames:
            all_rank = pd.concat(rank_frames, ignore_index=True)
            timing_cols = [
                col for col in all_rank.columns
                if col == "epoch_total_sec" or col.endswith("_sec")
            ]
            for epoch, group in all_rank.groupby("epoch", sort=True):
                row: Dict[str, Any] = {
                    **meta,
                    "epoch": int(epoch),
                    "is_cold_start": int(int(epoch) == 1),
                    "is_steady": int(int(epoch) >= STEADY_EPOCH_START),
                    "world_size_observed": int(group["rank"].nunique()),
                }
                for col in timing_cols:
                    values = pd.to_numeric(group[col], errors="coerce").dropna()
                    if values.empty:
                        continue
                    row[col] = float(values.max())
                    row[f"{col}_rank_mean"] = float(values.mean())
                    row[f"{col}_rank_min"] = float(values.min())
                    if float(values.mean()) != 0:
                        row[f"{col}_rank_imbalance_pct"] = float(
                            (values.max() - values.mean()) / values.mean() * 100.0
                        )
                stage_rows.append(row)

        for agg_path in find_agg_files(exp_dir):
            frame = read_csv(agg_path)
            if frame is None:
                continue
            frame = normalize_agg_columns(frame)
            frame["rank"] = parse_rank(agg_path)
            frame["input_root"] = str(root)
            frame["output_dir"] = str(exp_dir)
            for key in [
                "mode", "num_chunks", "repeat_id", "graph_tag", "strategy",
                "target_super_block_size", "target_num_blocks", "num_epochs",
                "run_seed", "total_labeled_edges",
            ]:
                if key not in frame.columns or frame[key].isna().all():
                    frame[key] = meta.get(key)
            if "epoch" not in frame.columns:
                continue
            frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
            frame = frame.dropna(subset=["epoch"])
            frame["epoch"] = frame["epoch"].astype(int)
            agg_frames.append(frame)

    stage = pd.DataFrame(stage_rows)
    agg = pd.concat(agg_frames, ignore_index=True) if agg_frames else pd.DataFrame()
    agg = normalize_overlap_semantics(agg)
    runs = pd.DataFrame(run_rows).drop_duplicates(
        subset=["output_dir"], keep="last"
    ) if run_rows else pd.DataFrame()
    return stage, agg, runs


def build_agg_epoch_rank(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()
    data = agg.copy()
    available = [col for col in AGG_TIME_METRICS if col in data.columns]
    if not available:
        return pd.DataFrame()

    group_cols = [
        "input_root", "output_dir", "graph_tag", "mode", "num_chunks",
        "repeat_id", "rank", "epoch",
    ]
    group_cols = [col for col in group_cols if col in data.columns]
    grouped = data.groupby(group_cols, dropna=False)[available].sum(min_count=1).reset_index()

    if "aggregation_comm_total_sec" in grouped.columns:
        total = grouped["aggregation_comm_total_sec"]
        if "aggregation_comm_hidden_sec" not in grouped.columns:
            grouped["aggregation_comm_hidden_sec"] = np.nan
        if "aggregation_comm_exposed_sec" not in grouped.columns:
            grouped["aggregation_comm_exposed_sec"] = np.nan

        seq = grouped["mode"].astype(str).eq(BASELINE_MODE)
        grouped.loc[seq, "aggregation_comm_hidden_sec"] = 0.0
        grouped.loc[seq, "aggregation_comm_exposed_sec"] = total.loc[seq]

        grouped["aggregation_overlap_ratio"] = np.where(
            total > 0,
            grouped["aggregation_comm_hidden_sec"].fillna(0.0) / total,
            0.0,
        )
        grouped["aggregation_overlap_ratio"] = grouped[
            "aggregation_overlap_ratio"
        ].clip(0.0, 1.0)
        grouped.loc[seq, "aggregation_overlap_ratio"] = 0.0

    # Normalize the pipeline decomposition across execution modes.
    # The raw `pipeline_overhead_sec` field has different semantics in some
    # sequential and asynchronous logs. This residual uses the same identity
    # for both modes:
    #
    #   pipeline = local compute + exposed communication + residual overhead
    #
    required = {
        "forward_pipeline_total_sec",
        "local_message_compute_sec",
        "aggregation_comm_exposed_sec",
    }
    if required.issubset(grouped.columns):
        grouped["pipeline_residual_overhead_sec"] = (
            pd.to_numeric(
                grouped["forward_pipeline_total_sec"], errors="coerce"
            )
            - pd.to_numeric(
                grouped["local_message_compute_sec"], errors="coerce"
            )
            - pd.to_numeric(
                grouped["aggregation_comm_exposed_sec"], errors="coerce"
            )
        ).clip(lower=0.0)

    grouped["is_steady"] = (
        pd.to_numeric(grouped["epoch"], errors="coerce") >= STEADY_EPOCH_START
    ).astype(int)
    return grouped


def select_critical_rank(agg_epoch_rank: pd.DataFrame) -> pd.DataFrame:
    if agg_epoch_rank.empty:
        return pd.DataFrame()
    data = agg_epoch_rank.copy()
    candidates = [
        "forward_pipeline_total_sec",
        "aggregation_comm_exposed_sec",
        "aggregation_comm_total_sec",
    ]
    critical_metric = next((c for c in candidates if c in data.columns), None)
    group_cols = [
        "input_root", "output_dir", "graph_tag", "mode",
        "num_chunks", "repeat_id", "epoch",
    ]
    group_cols = [c for c in group_cols if c in data.columns]
    if critical_metric is None:
        return data.groupby(group_cols, dropna=False).first().reset_index()

    selected_rows: List[pd.Series] = []
    for _, group in data.groupby(group_cols, dropna=False, sort=False):
        values = pd.to_numeric(group[critical_metric], errors="coerce")
        if values.notna().any():
            idx = values.idxmax()
        else:
            idx = group.index[0]
        selected_rows.append(data.loc[idx])
    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    out["critical_rank_metric"] = critical_metric
    return out


def relation_layer_summary(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()
    data = agg[pd.to_numeric(agg["epoch"], errors="coerce") >= STEADY_EPOCH_START].copy()
    required = ["graph_tag", "mode", "layer_idx", "edge_type"]
    if any(col not in data.columns for col in required):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for keys, group in data.groupby(required, dropna=False):
        graph_tag, mode, layer_idx, edge_type = keys
        record: Dict[str, Any] = {
            "graph_tag": graph_tag,
            "mode": mode,
            "layer_idx": layer_idx,
            "edge_type": edge_type,
            "num_rows": int(len(group)),
            "num_repeats": int(group["repeat_id"].nunique()) if "repeat_id" in group.columns else np.nan,
        }
        for col in [
            "num_local_edges", "num_dst_nodes", "aggregation_num_chunks",
            *AGG_TIME_METRICS,
        ]:
            if col in group.columns:
                add_stats(record, col, group[col])

        if {"aggregation_comm_total_sec", "aggregation_comm_hidden_sec"}.issubset(group.columns):
            total = pd.to_numeric(group["aggregation_comm_total_sec"], errors="coerce").sum(min_count=1)
            hidden = pd.to_numeric(group["aggregation_comm_hidden_sec"], errors="coerce").sum(min_count=1)
            if str(mode) == BASELINE_MODE:
                record["aggregation_overlap_ratio_weighted"] = 0.0
            elif pd.notna(total) and total > 0 and pd.notna(hidden):
                record["aggregation_overlap_ratio_weighted"] = float(
                    np.clip(hidden / total, 0.0, 1.0)
                )
        rows.append(record)
    return pd.DataFrame(rows)


def stage_run_summary(stage: pd.DataFrame) -> pd.DataFrame:
    if stage.empty:
        return pd.DataFrame()
    steady = stage[stage["is_steady"] == 1].copy()
    group_cols = ["input_root", "output_dir", "graph_tag", "mode", "num_chunks", "repeat_id"]
    group_cols = [c for c in group_cols if c in steady.columns]
    rows: List[Dict[str, Any]] = []
    metric_cols = [col for col in KNOWN_STAGE_METRICS if col in steady.columns]
    imbalance_cols = [
        f"{col}_rank_imbalance_pct"
        for col in KNOWN_STAGE_METRICS
        if f"{col}_rank_imbalance_pct" in steady.columns
    ]
    for keys, group in steady.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        record["n_steady_epochs"] = int(group["epoch"].nunique())
        for col in metric_cols + imbalance_cols:
            add_stats(record, col, group[col])
        rows.append(record)
    return pd.DataFrame(rows)


def agg_run_summary(critical: pd.DataFrame) -> pd.DataFrame:
    if critical.empty:
        return pd.DataFrame()
    steady = critical[critical["is_steady"] == 1].copy()
    group_cols = ["input_root", "output_dir", "graph_tag", "mode", "num_chunks", "repeat_id"]
    group_cols = [c for c in group_cols if c in steady.columns]
    rows: List[Dict[str, Any]] = []
    for keys, group in steady.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        record["n_steady_epochs"] = int(group["epoch"].nunique())
        for col in AGG_TIME_METRICS + ["aggregation_overlap_ratio"]:
            if col in group.columns:
                add_stats(record, col, group[col])

        if {"aggregation_comm_total_sec", "aggregation_comm_hidden_sec"}.issubset(group.columns):
            total = pd.to_numeric(group["aggregation_comm_total_sec"], errors="coerce").sum(min_count=1)
            hidden = pd.to_numeric(group["aggregation_comm_hidden_sec"], errors="coerce").sum(min_count=1)
            if str(record.get("mode")) == BASELINE_MODE:
                record["aggregation_overlap_ratio_weighted"] = 0.0
            elif pd.notna(total) and total > 0 and pd.notna(hidden):
                record["aggregation_overlap_ratio_weighted"] = float(
                    np.clip(hidden / total, 0.0, 1.0)
                )
        rows.append(record)
    return pd.DataFrame(rows)


def merge_run_summaries(stage_run: pd.DataFrame, agg_run: pd.DataFrame) -> pd.DataFrame:
    if stage_run.empty:
        return agg_run.copy()
    if agg_run.empty:
        return stage_run.copy()
    keys = [
        c for c in ["input_root", "output_dir", "graph_tag", "mode", "num_chunks", "repeat_id"]
        if c in stage_run.columns and c in agg_run.columns
    ]
    drop_cols = [c for c in ["n_steady_epochs"] if c in agg_run.columns]
    return stage_run.merge(
        agg_run.drop(columns=drop_cols, errors="ignore"),
        on=keys,
        how="outer",
        suffixes=("", "_agg"),
    )


def add_graph_relative_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "graph_tag" not in frame.columns or "mode" not in frame.columns:
        return frame
    out = frame.copy()
    metric_pairs = {
        "epoch_total_sec_mean": "epoch_total_speedup_vs_sequential_pct",
        "forward_loss_sec_mean": "forward_loss_speedup_vs_sequential_pct",
        "forward_pipeline_total_sec_mean": "pipeline_speedup_vs_sequential_pct",
        "aggregation_comm_exposed_sec_mean": "exposed_comm_speedup_vs_sequential_pct",
    }
    for graph_tag, group in out.groupby("graph_tag"):
        baseline = group[group["mode"].astype(str) == BASELINE_MODE]
        if baseline.empty:
            continue
        for source, target in metric_pairs.items():
            if source not in out.columns:
                continue
            base_values = pd.to_numeric(baseline[source], errors="coerce").dropna()
            if base_values.empty or float(base_values.mean()) <= 0:
                continue
            base = float(base_values.mean())
            mask = out["graph_tag"].astype(str).eq(str(graph_tag))
            out.loc[mask, target] = (
                (base - pd.to_numeric(out.loc[mask, source], errors="coerce"))
                / base * 100.0
            )
        mask = out["graph_tag"].astype(str).eq(str(graph_tag))

        # Mechanism-level communication/computation trade-off.
        # Positive communication saving means less communication remains on
        # the critical path than in the sequential baseline.
        if "aggregation_comm_exposed_sec_mean" in out.columns:
            base_values = pd.to_numeric(
                baseline["aggregation_comm_exposed_sec_mean"],
                errors="coerce",
            ).dropna()
            if not base_values.empty:
                base = float(base_values.mean())
                out.loc[mask, "exposed_comm_saved_vs_sequential_sec"] = (
                    base
                    - pd.to_numeric(
                        out.loc[mask, "aggregation_comm_exposed_sec_mean"],
                        errors="coerce",
                    )
                )

        # Positive compute delta means chunking introduced more local message
        # computation than the sequential baseline.
        if "local_message_compute_sec_mean" in out.columns:
            base_values = pd.to_numeric(
                baseline["local_message_compute_sec_mean"],
                errors="coerce",
            ).dropna()
            if not base_values.empty:
                base = float(base_values.mean())
                out.loc[mask, "local_compute_delta_vs_sequential_sec"] = (
                    pd.to_numeric(
                        out.loc[mask, "local_message_compute_sec_mean"],
                        errors="coerce",
                    )
                    - base
                )

        # Compare only the mode-normalized residual overhead, not the raw
        # pipeline_overhead_sec whose semantics may differ across modes.
        if "pipeline_residual_overhead_sec_mean" in out.columns:
            base_values = pd.to_numeric(
                baseline["pipeline_residual_overhead_sec_mean"],
                errors="coerce",
            ).dropna()
            if not base_values.empty:
                base = float(base_values.mean())
                out.loc[
                    mask,
                    "pipeline_residual_overhead_delta_vs_sequential_sec",
                ] = (
                    pd.to_numeric(
                        out.loc[
                            mask,
                            "pipeline_residual_overhead_sec_mean",
                        ],
                        errors="coerce",
                    )
                    - base
                )

        if "forward_pipeline_total_sec_mean" in out.columns:
            base_values = pd.to_numeric(
                baseline["forward_pipeline_total_sec_mean"],
                errors="coerce",
            ).dropna()
            if not base_values.empty:
                base = float(base_values.mean())
                out.loc[mask, "forward_pipeline_saved_vs_sequential_sec"] = (
                    base
                    - pd.to_numeric(
                        out.loc[mask, "forward_pipeline_total_sec_mean"],
                        errors="coerce",
                    )
                )

        # The mechanistic balance should approximately agree with the directly
        # measured forward-pipeline saving:
        #
        # communication saving - added compute - added residual overhead.
        needed_balance_cols = {
            "exposed_comm_saved_vs_sequential_sec",
            "local_compute_delta_vs_sequential_sec",
            "pipeline_residual_overhead_delta_vs_sequential_sec",
        }
        if needed_balance_cols.issubset(out.columns):
            out.loc[mask, "aggregation_path_net_saving_sec"] = (
                pd.to_numeric(
                    out.loc[
                        mask,
                        "exposed_comm_saved_vs_sequential_sec",
                    ],
                    errors="coerce",
                )
                - pd.to_numeric(
                    out.loc[
                        mask,
                        "local_compute_delta_vs_sequential_sec",
                    ],
                    errors="coerce",
                )
                - pd.to_numeric(
                    out.loc[
                        mask,
                        "pipeline_residual_overhead_delta_vs_sequential_sec",
                    ],
                    errors="coerce",
                )
            )

        # Backward is retained as an end-to-end consequence, not as the
        # mechanism-level overlap trade-off term.
        if "backward_sec_mean" in out.columns:
            base_values = pd.to_numeric(
                baseline["backward_sec_mean"], errors="coerce"
            ).dropna()
            if not base_values.empty:
                out.loc[mask, "backward_delta_vs_sequential_sec"] = (
                    pd.to_numeric(
                        out.loc[mask, "backward_sec_mean"],
                        errors="coerce",
                    )
                    - float(base_values.mean())
                )

        if "epoch_total_sec_mean" in out.columns:
            base_values = pd.to_numeric(
                baseline["epoch_total_sec_mean"], errors="coerce"
            ).dropna()
            if not base_values.empty:
                base = float(base_values.mean())
                out.loc[mask, "epoch_total_saved_vs_sequential_sec"] = (
                    base
                    - pd.to_numeric(
                        out.loc[mask, "epoch_total_sec_mean"],
                        errors="coerce",
                    )
                )
    return out


def graph_mode_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()
    group_cols = ["graph_tag", "mode", "num_chunks"]
    rows: List[Dict[str, Any]] = []
    numeric_mean_cols = [
        c for c in run_summary.columns
        if c.endswith("_mean") and pd.api.types.is_numeric_dtype(run_summary[c])
    ]
    weighted_cols = [
        c for c in ["aggregation_overlap_ratio_weighted"]
        if c in run_summary.columns
    ]
    for keys, group in run_summary.groupby(group_cols, dropna=False):
        graph_tag, mode, num_chunks = keys
        record: Dict[str, Any] = {
            "graph_tag": graph_tag,
            "mode": mode,
            "num_chunks": num_chunks,
            "num_repeats": int(group["repeat_id"].nunique()) if "repeat_id" in group.columns else int(len(group)),
        }
        for col in numeric_mean_cols + weighted_cols:
            add_stats(record, col, group[col])
        rows.append(record)
    out = pd.DataFrame(rows)
    # Flatten the common "mean of run means" names for readability.
    rename = {}
    for col in out.columns:
        if col.endswith("_mean_mean"):
            rename[col] = col[:-5]
    out = out.rename(columns=rename)
    if "aggregation_overlap_ratio_weighted_mean" in out.columns:
        out = out.rename(
            columns={
                "aggregation_overlap_ratio_weighted_mean":
                    "aggregation_overlap_ratio_weighted"
            }
        )
    return add_graph_relative_metrics(out)


def overall_mode_summary(graph_summary: pd.DataFrame) -> pd.DataFrame:
    if graph_summary.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_cols = ["mode", "num_chunks"]
    numeric_cols = [
        c for c in graph_summary.columns
        if pd.api.types.is_numeric_dtype(graph_summary[c])
        and c not in {"num_chunks", "num_repeats"}
        and (
            c.endswith("_mean")
            or c.endswith("_pct")
            or c.endswith("_sec")
            or c == "aggregation_overlap_ratio_weighted"
        )
    ]
    for keys, group in graph_summary.groupby(group_cols, dropna=False):
        mode, num_chunks = keys
        record: Dict[str, Any] = {
            "mode": mode,
            "num_chunks": num_chunks,
            "num_graphs": int(group["graph_tag"].nunique()),
            "total_repeats": int(group["num_repeats"].sum()) if "num_repeats" in group.columns else np.nan,
        }
        for col in numeric_cols:
            add_stats(record, f"macro_{col}", group[col])
        rows.append(record)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_order"] = out["mode"].map(lambda x: mode_order(x))
        out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


def best_mode_by_graph(graph_summary: pd.DataFrame) -> pd.DataFrame:
    if graph_summary.empty or "epoch_total_sec_mean" not in graph_summary.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for graph_tag, group in graph_summary.groupby("graph_tag"):
        valid = group.dropna(subset=["epoch_total_sec_mean"])
        if valid.empty:
            continue
        best_time = valid.loc[valid["epoch_total_sec_mean"].idxmin()]
        async_rows = valid[valid["mode"].astype(str) != BASELINE_MODE]
        best_overlap = None
        if "aggregation_overlap_ratio_weighted" in valid.columns and not async_rows.empty:
            overlap_valid = async_rows.dropna(subset=["aggregation_overlap_ratio_weighted"])
            if not overlap_valid.empty:
                best_overlap = overlap_valid.loc[
                    overlap_valid["aggregation_overlap_ratio_weighted"].idxmax()
                ]
        record = {
            "graph_tag": graph_tag,
            "best_epoch_mode": best_time["mode"],
            "best_epoch_num_chunks": best_time.get("num_chunks"),
            "best_epoch_total_sec": best_time["epoch_total_sec_mean"],
            "best_epoch_speedup_vs_sequential_pct": best_time.get(
                "epoch_total_speedup_vs_sequential_pct", np.nan
            ),
        }
        if best_overlap is not None:
            record.update({
                "best_overlap_mode": best_overlap["mode"],
                "best_overlap_num_chunks": best_overlap.get("num_chunks"),
                "best_overlap_ratio": best_overlap.get(
                    "aggregation_overlap_ratio_weighted", np.nan
                ),
            })
        rows.append(record)
    return pd.DataFrame(rows)


def add_run_level_phase_residuals(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Add approximate non-pipeline residuals for phase interpretation.

    These residuals combine synchronized stage maxima and the selected critical
    aggregation rank, so they are descriptive rather than kernel-exact.
    """
    if run_summary.empty:
        return run_summary
    out = run_summary.copy()

    required = {
        "epoch_total_sec_mean",
        "forward_pipeline_total_sec_mean",
        "backward_sec_mean",
    }
    if required.issubset(out.columns):
        out["epoch_rest_approx_sec_mean"] = (
            pd.to_numeric(out["epoch_total_sec_mean"], errors="coerce")
            - pd.to_numeric(
                out["forward_pipeline_total_sec_mean"], errors="coerce"
            )
            - pd.to_numeric(out["backward_sec_mean"], errors="coerce")
        )

    required = {
        "forward_loss_sec_mean",
        "forward_pipeline_total_sec_mean",
    }
    if required.issubset(out.columns):
        out["forward_outside_pipeline_approx_sec_mean"] = (
            pd.to_numeric(out["forward_loss_sec_mean"], errors="coerce")
            - pd.to_numeric(
                out["forward_pipeline_total_sec_mean"], errors="coerce"
            )
        )
    return out


def paired_speedup(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Pair every async mode with sequential from the same repeat."""
    if run_summary.empty or "epoch_total_sec_mean" not in run_summary.columns:
        return pd.DataFrame()

    pair_keys = [
        c for c in ["input_root", "graph_tag", "repeat_id"]
        if c in run_summary.columns
    ]
    metric_cols = [
        c for c in [
            "epoch_total_sec_mean",
            "data_sec_mean",
            "forward_loss_sec_mean",
            "backward_sec_mean",
            "optimizer_sec_mean",
            "loss_sync_sec_mean",
            "sync_loss_sec_mean",
            "local_message_compute_sec_mean",
            "aggregation_comm_total_sec_mean",
            "aggregation_comm_exposed_sec_mean",
            "aggregation_comm_hidden_sec_mean",
            "aggregation_wait_sec_mean",
            "forward_pipeline_total_sec_mean",
            "pipeline_residual_overhead_sec_mean",
            "epoch_rest_approx_sec_mean",
            "forward_outside_pipeline_approx_sec_mean",
            "aggregation_overlap_ratio_weighted",
        ]
        if c in run_summary.columns
    ]

    baseline_cols = pair_keys + metric_cols
    baseline = run_summary[
        run_summary["mode"].astype(str).eq(BASELINE_MODE)
    ][baseline_cols].copy()
    if baseline.empty:
        return pd.DataFrame()

    baseline = baseline.rename(
        columns={
            c: f"sequential_{c}"
            for c in metric_cols
        }
    )

    async_rows = run_summary[
        ~run_summary["mode"].astype(str).eq(BASELINE_MODE)
    ].copy()
    if async_rows.empty:
        return pd.DataFrame()

    merged = async_rows.merge(
        baseline,
        on=pair_keys,
        how="inner",
        validate="many_to_one",
    )

    # Generic paired deltas and savings.
    for metric in metric_cols:
        seq_col = f"sequential_{metric}"
        if seq_col not in merged.columns:
            continue
        async_values = pd.to_numeric(merged[metric], errors="coerce")
        seq_values = pd.to_numeric(merged[seq_col], errors="coerce")
        stem = metric[:-5] if metric.endswith("_mean") else metric
        merged[f"paired_{stem}_delta_vs_seq"] = async_values - seq_values
        merged[f"paired_{stem}_saved_vs_seq"] = seq_values - async_values

    # Percentage speedups for the principal critical-path metrics.
    for metric, output_name in [
        ("epoch_total_sec_mean", "paired_epoch_speedup_pct"),
        ("forward_loss_sec_mean", "paired_forward_loss_speedup_pct"),
        (
            "forward_pipeline_total_sec_mean",
            "paired_forward_pipeline_speedup_pct",
        ),
        (
            "aggregation_comm_exposed_sec_mean",
            "paired_exposed_comm_reduction_pct",
        ),
    ]:
        seq_col = f"sequential_{metric}"
        if metric not in merged.columns or seq_col not in merged.columns:
            continue
        denom = pd.to_numeric(merged[seq_col], errors="coerce")
        numer = (
            denom
            - pd.to_numeric(merged[metric], errors="coerce")
        )
        merged[output_name] = np.where(
            denom > 0,
            numer / denom * 100.0,
            np.nan,
        )

    # Mechanism-level communication/computation balance for every repeat.
    needed = {
        "paired_aggregation_comm_exposed_sec_saved_vs_seq",
        "paired_local_message_compute_sec_delta_vs_seq",
        "paired_pipeline_residual_overhead_sec_delta_vs_seq",
    }
    if needed.issubset(merged.columns):
        merged["paired_aggregation_path_net_saving_sec"] = (
            merged[
                "paired_aggregation_comm_exposed_sec_saved_vs_seq"
            ]
            - merged[
                "paired_local_message_compute_sec_delta_vs_seq"
            ]
            - merged[
                "paired_pipeline_residual_overhead_sec_delta_vs_seq"
            ]
        )

    # End-to-end consequences.
    if "paired_epoch_total_sec_saved_vs_seq" in merged.columns:
        merged["paired_epoch_win"] = (
            merged["paired_epoch_total_sec_saved_vs_seq"] > 0
        ).astype(int)
    if "paired_forward_pipeline_total_sec_saved_vs_seq" in merged.columns:
        merged["paired_pipeline_win"] = (
            merged[
                "paired_forward_pipeline_total_sec_saved_vs_seq"
            ] > 0
        ).astype(int)
    if "paired_aggregation_comm_exposed_sec_saved_vs_seq" in merged.columns:
        merged["paired_exposed_comm_reduced"] = (
            merged[
                "paired_aggregation_comm_exposed_sec_saved_vs_seq"
            ] > 0
        ).astype(int)
    if "paired_aggregation_path_net_saving_sec" in merged.columns:
        merged["paired_aggregation_net_positive"] = (
            merged["paired_aggregation_path_net_saving_sec"] > 0
        ).astype(int)

    return add_configuration_columns(merged)


def paired_speedup_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired results across the five independent repeats."""
    if paired.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    group_cols = ["mode", "num_chunks"]
    summary_metrics = [
        c for c in [
            "paired_epoch_speedup_pct",
            "paired_epoch_total_sec_saved_vs_seq",
            "paired_forward_loss_speedup_pct",
            "paired_forward_loss_sec_saved_vs_seq",
            "paired_backward_sec_delta_vs_seq",
            "paired_optimizer_sec_delta_vs_seq",
            "paired_aggregation_comm_total_sec_delta_vs_seq",
            "paired_aggregation_comm_exposed_sec_saved_vs_seq",
            "paired_aggregation_comm_hidden_sec_delta_vs_seq",
            "paired_exposed_comm_reduction_pct",
            "paired_local_message_compute_sec_delta_vs_seq",
            "paired_pipeline_residual_overhead_sec_delta_vs_seq",
            "paired_aggregation_path_net_saving_sec",
            "paired_forward_pipeline_total_sec_saved_vs_seq",
            "paired_forward_pipeline_speedup_pct",
            "paired_epoch_rest_approx_sec_delta_vs_seq",
            "paired_forward_outside_pipeline_approx_sec_delta_vs_seq",
        ]
        if c in paired.columns
    ]

    for keys, group in paired.groupby(group_cols, dropna=False):
        mode, num_chunks = keys
        record: Dict[str, Any] = {
            "mode": mode,
            "num_chunks": num_chunks,
            "num_pairs": int(len(group)),
            "num_repeats": int(group["repeat_id"].nunique())
            if "repeat_id" in group.columns else int(len(group)),
        }
        binary_metrics = {
            "epoch_win_rate_pct": "paired_epoch_win",
            "pipeline_win_rate_pct": "paired_pipeline_win",
            "exposed_comm_reduction_rate_pct":
                "paired_exposed_comm_reduced",
            "aggregation_net_positive_rate_pct":
                "paired_aggregation_net_positive",
        }
        for output, source_col in binary_metrics.items():
            if source_col in group.columns:
                record[output] = float(
                    pd.to_numeric(
                        group[source_col], errors="coerce"
                    ).mean() * 100.0
                )

        for metric in summary_metrics:
            add_stats(record, metric, group[metric])
        rows.append(record)

    return add_configuration_columns(pd.DataFrame(rows))


def best_mode_by_repeat(
    run_summary: pd.DataFrame,
    paired: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for repeat_id, group in run_summary.groupby("repeat_id", dropna=False):
        record: Dict[str, Any] = {"repeat_id": repeat_id}
        criteria = [
            (
                "epoch",
                "epoch_total_sec_mean",
                "min",
            ),
            (
                "forward",
                "forward_loss_sec_mean",
                "min",
            ),
            (
                "backward",
                "backward_sec_mean",
                "min",
            ),
            (
                "pipeline",
                "forward_pipeline_total_sec_mean",
                "min",
            ),
            (
                "exposed_comm",
                "aggregation_comm_exposed_sec_mean",
                "min",
            ),
        ]
        for prefix, metric, direction in criteria:
            if metric not in group.columns:
                continue
            valid = group.dropna(subset=[metric])
            if valid.empty:
                continue
            idx = (
                valid[metric].idxmin()
                if direction == "min"
                else valid[metric].idxmax()
            )
            best = valid.loc[idx]
            record[f"best_{prefix}_mode"] = best["mode"]
            record[f"best_{prefix}_num_chunks"] = best.get(
                "num_chunks", np.nan
            )
            record[f"best_{prefix}_{metric}"] = best[metric]

        # Mechanism optimum is defined only over async modes.
        async_group = group[
            ~group["mode"].astype(str).eq(BASELINE_MODE)
        ]
        if not async_group.empty:
            metric = "aggregation_path_net_saving_sec"
            if metric in async_group.columns:
                valid = async_group.dropna(subset=[metric])
                if not valid.empty:
                    best = valid.loc[valid[metric].idxmax()]
                    record["best_aggregation_net_mode"] = best["mode"]
                    record["best_aggregation_net_num_chunks"] = best.get(
                        "num_chunks", np.nan
                    )
                    record["best_aggregation_net_saving_sec"] = best[metric]
        if paired is not None and not paired.empty:
            paired_repeat = paired[
                pd.to_numeric(
                    paired["repeat_id"], errors="coerce"
                ) == pd.to_numeric(
                    pd.Series([repeat_id]), errors="coerce"
                ).iloc[0]
            ]
            metric = "paired_aggregation_path_net_saving_sec"
            if metric in paired_repeat.columns:
                valid = paired_repeat.dropna(subset=[metric])
                if not valid.empty:
                    best = valid.loc[valid[metric].idxmax()]
                    record["best_aggregation_net_mode"] = best["mode"]
                    record["best_aggregation_net_num_chunks"] = best.get(
                        "num_chunks", np.nan
                    )
                    record["best_aggregation_net_saving_sec"] = best[metric]

        rows.append(record)
    return pd.DataFrame(rows).sort_values("repeat_id").reset_index(drop=True)


def best_overall_criteria(mode_summary: pd.DataFrame) -> pd.DataFrame:
    if mode_summary.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    criteria = [
        (
            "minimum steady epoch time",
            "epoch_total_sec_mean",
            "min",
            False,
        ),
        (
            "minimum forward/loss time",
            "forward_loss_sec_mean",
            "min",
            False,
        ),
        (
            "minimum backward time",
            "backward_sec_mean",
            "min",
            False,
        ),
        (
            "minimum exposed communication",
            "aggregation_comm_exposed_sec_mean",
            "min",
            False,
        ),
        (
            "minimum forward pipeline time",
            "forward_pipeline_total_sec_mean",
            "min",
            False,
        ),
        (
            "maximum weighted overlap ratio",
            "aggregation_overlap_ratio_weighted",
            "max",
            True,
        ),
        (
            "maximum aggregation-path net saving",
            "aggregation_path_net_saving_sec",
            "max",
            True,
        ),
        (
            "maximum epoch speedup",
            "epoch_total_speedup_vs_sequential_pct",
            "max",
            False,
        ),
    ]

    for criterion, metric, direction, async_only in criteria:
        if metric not in mode_summary.columns:
            continue
        candidates = mode_summary.copy()
        if async_only:
            candidates = candidates[
                ~candidates["mode"].astype(str).eq(BASELINE_MODE)
            ]
        candidates = candidates.dropna(subset=[metric])
        if candidates.empty:
            continue
        idx = (
            candidates[metric].idxmin()
            if direction == "min"
            else candidates[metric].idxmax()
        )
        best = candidates.loc[idx]
        rows.append(
            {
                "criterion": criterion,
                "metric": metric,
                "direction": direction,
                "best_mode": best["mode"],
                "best_num_chunks": best.get("num_chunks", np.nan),
                "best_value": best[metric],
                "num_repeats": best.get("num_repeats", np.nan),
            }
        )
    return pd.DataFrame(rows)


def experiment_completeness(
    runs: pd.DataFrame,
    stage: pd.DataFrame,
    agg_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Build the expected 5 × 10 job matrix and check discovered outputs."""
    rows: List[Dict[str, Any]] = []
    runs_data = runs.copy()
    if not runs_data.empty:
        runs_data["repeat_id"] = pd.to_numeric(
            runs_data["repeat_id"], errors="coerce"
        )
    for repeat_id in EXPECTED_REPEATS:
        for mode in EXPECTED_MODES:
            matched = runs_data[
                (runs_data["graph_tag"].astype(str) == EXPECTED_GRAPH_TAG)
                & (runs_data["repeat_id"] == repeat_id)
                & (runs_data["mode"].astype(str) == mode)
            ] if not runs_data.empty else pd.DataFrame()

            output_dirs = (
                matched["output_dir"].astype(str).tolist()
                if not matched.empty else []
            )
            stage_rows = pd.DataFrame()
            agg_rows = pd.DataFrame()
            if not stage.empty:
                stage_rows = stage[
                    (stage["graph_tag"].astype(str) == EXPECTED_GRAPH_TAG)
                    & (
                        pd.to_numeric(
                            stage["repeat_id"], errors="coerce"
                        ) == repeat_id
                    )
                    & (stage["mode"].astype(str) == mode)
                ]
            if not agg_raw.empty:
                agg_rows = agg_raw[
                    (agg_raw["graph_tag"].astype(str) == EXPECTED_GRAPH_TAG)
                    & (
                        pd.to_numeric(
                            agg_raw["repeat_id"], errors="coerce"
                        ) == repeat_id
                    )
                    & (agg_raw["mode"].astype(str) == mode)
                ]

            n_stage_epochs = (
                int(stage_rows["epoch"].nunique())
                if not stage_rows.empty and "epoch" in stage_rows.columns
                else 0
            )
            n_stage_ranks = 0
            if output_dirs:
                rank_set = set()
                for output_dir in output_dirs:
                    for path in Path(output_dir).glob(
                        "stage_timing_rank*.csv"
                    ):
                        rank_set.add(parse_rank(path))
                n_stage_ranks = len(rank_set)

            n_agg_ranks = (
                int(agg_rows["rank"].nunique())
                if not agg_rows.empty and "rank" in agg_rows.columns
                else 0
            )
            has_result_json = any(
                (Path(d) / "train_result_rank0.json").exists()
                for d in output_dirs
            )
            status = "complete"
            problems: List[str] = []
            if len(output_dirs) == 0:
                status = "missing"
                problems.append("no experiment directory")
            elif len(output_dirs) > 1:
                status = "duplicate"
                problems.append(f"{len(output_dirs)} experiment directories")
            if not has_result_json:
                problems.append("missing train_result_rank0.json")
            if n_stage_epochs < 10:
                problems.append(f"stage epochs={n_stage_epochs}/10")
            if n_stage_ranks == 0:
                problems.append("no stage timing ranks")
            if n_agg_ranks == 0:
                problems.append("no aggregation timing ranks")
            if status == "complete" and problems:
                status = "incomplete"

            rows.append(
                {
                    "graph_tag": EXPECTED_GRAPH_TAG,
                    "repeat_id": repeat_id,
                    "mode": mode,
                    "configuration_label": configuration_label(mode),
                    "configuration_order": configuration_order(mode),
                    "expected": 1,
                    "actual_experiment_dirs": len(output_dirs),
                    "has_train_result_rank0": int(has_result_json),
                    "stage_epochs_observed": n_stage_epochs,
                    "stage_ranks_observed": n_stage_ranks,
                    "aggregation_ranks_observed": n_agg_ranks,
                    "status": status,
                    "problems": "; ".join(problems),
                    "output_dirs": " | ".join(output_dirs),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["repeat_id", "configuration_order"]
    ).reset_index(drop=True)


def make_plot_mode_summary(mode_summary: pd.DataFrame) -> pd.DataFrame:
    if mode_summary.empty:
        return pd.DataFrame()
    cols = [
        "graph_tag",
        "mode",
        "configuration_label",
        "configuration_order",
        "num_chunks",
        "num_repeats",
        "epoch_total_sec_mean",
        "epoch_total_sec_mean_std",
        "epoch_total_sec_mean_ci95_low",
        "epoch_total_sec_mean_ci95_high",
        "forward_loss_sec_mean",
        "forward_loss_sec_mean_std",
        "backward_sec_mean",
        "backward_sec_mean_std",
        "local_message_compute_sec_mean",
        "local_message_compute_sec_mean_std",
        "aggregation_comm_total_sec_mean",
        "aggregation_comm_total_sec_mean_std",
        "aggregation_comm_exposed_sec_mean",
        "aggregation_comm_exposed_sec_mean_std",
        "aggregation_comm_hidden_sec_mean",
        "aggregation_comm_hidden_sec_mean_std",
        "aggregation_overlap_ratio_weighted",
        "aggregation_overlap_ratio_weighted_std",
        "forward_pipeline_total_sec_mean",
        "forward_pipeline_total_sec_mean_std",
        "exposed_comm_saved_vs_sequential_sec",
        "local_compute_delta_vs_sequential_sec",
        "pipeline_residual_overhead_delta_vs_sequential_sec",
        "aggregation_path_net_saving_sec",
        "backward_delta_vs_sequential_sec",
        "epoch_total_saved_vs_sequential_sec",
        "epoch_total_speedup_vs_sequential_pct",
    ]
    return mode_summary[[c for c in cols if c in mode_summary.columns]].copy()


def make_plot_repeat_paired(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    cols = [
        "repeat_id",
        "mode",
        "configuration_label",
        "configuration_order",
        "num_chunks",
        "epoch_total_sec_mean",
        "sequential_epoch_total_sec_mean",
        "paired_epoch_total_sec_saved_vs_seq",
        "paired_epoch_speedup_pct",
        "forward_loss_sec_mean",
        "sequential_forward_loss_sec_mean",
        "paired_forward_loss_sec_saved_vs_seq",
        "backward_sec_mean",
        "sequential_backward_sec_mean",
        "paired_backward_sec_delta_vs_seq",
        "aggregation_comm_exposed_sec_mean",
        "sequential_aggregation_comm_exposed_sec_mean",
        "paired_aggregation_comm_exposed_sec_saved_vs_seq",
        "local_message_compute_sec_mean",
        "sequential_local_message_compute_sec_mean",
        "paired_local_message_compute_sec_delta_vs_seq",
        "pipeline_residual_overhead_sec_mean",
        "sequential_pipeline_residual_overhead_sec_mean",
        "paired_pipeline_residual_overhead_sec_delta_vs_seq",
        "paired_aggregation_path_net_saving_sec",
        "forward_pipeline_total_sec_mean",
        "sequential_forward_pipeline_total_sec_mean",
        "paired_forward_pipeline_total_sec_saved_vs_seq",
        "paired_epoch_win",
    ]
    return paired[[c for c in cols if c in paired.columns]].copy()


def epoch_profile(stage: pd.DataFrame) -> pd.DataFrame:
    if stage.empty:
        return pd.DataFrame()
    metrics = [c for c in KNOWN_STAGE_METRICS if c in stage.columns]
    rows: List[Dict[str, Any]] = []
    for keys, group in stage.groupby(["graph_tag", "mode", "epoch"], dropna=False):
        graph_tag, mode, epoch = keys
        record = {
            "graph_tag": graph_tag,
            "mode": mode,
            "epoch": int(epoch),
            "num_runs": int(group["repeat_id"].nunique()) if "repeat_id" in group.columns else len(group),
        }
        for col in metrics:
            add_stats(record, col, group[col])
        rows.append(record)
    return pd.DataFrame(rows)


def safe_markdown(frame: pd.DataFrame, max_rows: int = 30) -> str:
    shown = frame.head(max_rows)
    try:
        return shown.to_markdown(index=False)
    except Exception:
        return "```\n" + shown.to_string(index=False) + "\n```"


def write_report(
    completeness: pd.DataFrame,
    mode_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    best_overall: pd.DataFrame,
    best_repeat: pd.DataFrame,
) -> None:
    complete_jobs = (
        int((completeness["status"] == "complete").sum())
        if not completeness.empty else 0
    )
    lines = [
        "# 20241218 forward aggregation overlap analysis",
        "",
        f"- Profile: `{PROFILE}`",
        f"- Input roots: `{[str(x) for x in INPUT_ROOTS]}`",
        f"- Expected jobs: {EXPECTED_TOTAL_JOBS}.",
        f"- Complete jobs: {complete_jobs}.",
        f"- Steady epochs start at epoch {STEADY_EPOCH_START}.",
        "- Stage time is synchronized by the maximum across ranks.",
        "- Sequential hidden communication is normalized to zero.",
        "- Async modes are paired with sequential from the same repeat.",
        "",
    ]

    if not completeness.empty:
        status = (
            completeness.groupby("status", dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        lines += [
            "## Experiment completeness",
            "",
            safe_markdown(status, 20),
            "",
        ]

    if not mode_summary.empty:
        key_cols = [
            c for c in [
                "mode", "configuration_label", "num_repeats",
                "epoch_total_sec_mean", "epoch_total_sec_mean_std",
                "forward_loss_sec_mean", "backward_sec_mean",
                "local_message_compute_sec_mean",
                "aggregation_comm_total_sec_mean",
                "aggregation_comm_exposed_sec_mean",
                "aggregation_comm_hidden_sec_mean",
                "aggregation_overlap_ratio_weighted",
                "forward_pipeline_total_sec_mean",
                "exposed_comm_saved_vs_sequential_sec",
                "local_compute_delta_vs_sequential_sec",
                "aggregation_path_net_saving_sec",
                "backward_delta_vs_sequential_sec",
                "epoch_total_speedup_vs_sequential_pct",
            ] if c in mode_summary.columns
        ]
        lines += [
            "## Five-repeat mode summary",
            "",
            safe_markdown(mode_summary[key_cols], 20),
            "",
        ]

    if not paired_summary.empty:
        key_cols = [
            c for c in [
                "mode", "configuration_label", "num_pairs",
                "epoch_win_rate_pct",
                "paired_epoch_speedup_pct_mean",
                "paired_epoch_speedup_pct_std",
                "paired_epoch_speedup_pct_ci95_low",
                "paired_epoch_speedup_pct_ci95_high",
                "paired_aggregation_comm_exposed_sec_saved_vs_seq_mean",
                "paired_local_message_compute_sec_delta_vs_seq_mean",
                "paired_aggregation_path_net_saving_sec_mean",
                "paired_backward_sec_delta_vs_seq_mean",
            ] if c in paired_summary.columns
        ]
        lines += [
            "## Paired comparison against sequential",
            "",
            safe_markdown(paired_summary[key_cols], 20),
            "",
        ]

    if not best_overall.empty:
        lines += [
            "## Overall best configurations",
            "",
            safe_markdown(best_overall, 20),
            "",
        ]

    if not best_repeat.empty:
        lines += [
            "## Best configuration in every repeat",
            "",
            safe_markdown(best_repeat, 20),
            "",
        ]

    if mode_summary.empty:
        lines += [
            "No valid timing data were found. Check each experiment directory "
            "for `stage_timing_rank*.csv` and forward aggregation timing files."
        ]

    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def save_frame(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(
        OUTPUT_DIR / filename,
        index=False,
        encoding="utf-8-sig",
    )



def experiment_completeness_multiday(
    runs: pd.DataFrame,
    stage: pd.DataFrame,
    agg_raw: pd.DataFrame,
) -> pd.DataFrame:
    """检查 date × repeat × mode 的完整性。"""
    graph_tags = list(EXPECTED_GRAPH_TAGS)
    if not graph_tags and not runs.empty and "graph_tag" in runs.columns:
        graph_tags = sorted(runs["graph_tag"].astype(str).dropna().unique())

    rows: List[Dict[str, Any]] = []
    runs_data = runs.copy()
    if not runs_data.empty:
        runs_data["repeat_id"] = pd.to_numeric(
            runs_data["repeat_id"], errors="coerce"
        )
        runs_data["graph_tag"] = runs_data["graph_tag"].astype(str)

    for graph_tag in graph_tags:
        for repeat_id in EXPECTED_REPEATS:
            for mode in EXPECTED_MODES:
                if runs_data.empty:
                    matched = pd.DataFrame()
                else:
                    matched = runs_data[
                        runs_data["graph_tag"].eq(str(graph_tag))
                        & runs_data["repeat_id"].eq(repeat_id)
                        & runs_data["mode"].astype(str).eq(mode)
                    ]

                output_dirs = (
                    matched["output_dir"].astype(str).tolist()
                    if not matched.empty else []
                )

                stage_rows = pd.DataFrame()
                if not stage.empty:
                    stage_rows = stage[
                        stage["graph_tag"].astype(str).eq(str(graph_tag))
                        & pd.to_numeric(
                            stage["repeat_id"], errors="coerce"
                        ).eq(repeat_id)
                        & stage["mode"].astype(str).eq(mode)
                    ]

                agg_rows = pd.DataFrame()
                if not agg_raw.empty:
                    agg_rows = agg_raw[
                        agg_raw["graph_tag"].astype(str).eq(str(graph_tag))
                        & pd.to_numeric(
                            agg_raw["repeat_id"], errors="coerce"
                        ).eq(repeat_id)
                        & agg_raw["mode"].astype(str).eq(mode)
                    ]

                n_stage_epochs = (
                    int(stage_rows["epoch"].nunique())
                    if not stage_rows.empty and "epoch" in stage_rows.columns
                    else 0
                )
                n_stage_ranks = 0
                for output_dir in output_dirs:
                    rank_set = {
                        parse_rank(path)
                        for path in Path(output_dir).glob(
                            "stage_timing_rank*.csv"
                        )
                    }
                    n_stage_ranks = max(n_stage_ranks, len(rank_set))

                n_agg_ranks = (
                    int(agg_rows["rank"].nunique())
                    if not agg_rows.empty and "rank" in agg_rows.columns
                    else 0
                )
                has_result = any(
                    (Path(d) / "train_result_rank0.json").exists()
                    for d in output_dirs
                )

                problems: List[str] = []
                status = "complete"
                if len(output_dirs) == 0:
                    status = "missing"
                    problems.append("no experiment directory")
                elif len(output_dirs) > 1:
                    status = "duplicate"
                    problems.append(
                        f"{len(output_dirs)} experiment directories"
                    )
                if not has_result:
                    problems.append("missing train_result_rank0.json")
                if n_stage_epochs < 10:
                    problems.append(f"stage epochs={n_stage_epochs}/10")
                if n_stage_ranks == 0:
                    problems.append("no stage timing ranks")
                if n_agg_ranks == 0:
                    problems.append("no aggregation timing ranks")
                if status == "complete" and problems:
                    status = "incomplete"

                rows.append(
                    {
                        "graph_tag": str(graph_tag),
                        "repeat_id": repeat_id,
                        "mode": mode,
                        "configuration_label": configuration_label(mode),
                        "configuration_order": configuration_order(mode),
                        "actual_experiment_dirs": len(output_dirs),
                        "has_train_result_rank0": int(has_result),
                        "stage_epochs_observed": n_stage_epochs,
                        "stage_ranks_observed": n_stage_ranks,
                        "aggregation_ranks_observed": n_agg_ranks,
                        "status": status,
                        "problems": "; ".join(problems),
                        "output_dirs": " | ".join(output_dirs),
                    }
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["graph_tag", "repeat_id", "configuration_order"],
        kind="stable",
    ).reset_index(drop=True)


def _finite_xy(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if frame.empty or x_col not in frame.columns or y_col not in frame.columns:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            pd.DataFrame(),
        )
    data = frame.copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[x_col, y_col])
    return (
        data[x_col].to_numpy(dtype=float),
        data[y_col].to_numpy(dtype=float),
        data,
    )


def _fit_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    if len(y) == 0:
        return {
            "n": 0,
            "r2": np.nan,
            "mae_sec": np.nan,
            "rmse_sec": np.nan,
            "max_abs_error_sec": np.nan,
        }
    error = pred - y
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "n": int(len(y)),
        "r2": float(r2),
        "mae_sec": float(np.mean(np.abs(error))),
        "rmse_sec": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_error_sec": float(np.max(np.abs(error))),
    }


def _saturating_value(c: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * (1.0 - np.exp(-b * (c - 1.0)))


def _fit_saturating_params(
    c: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """拟合 a * (1-exp(-b(C-1)))；无 scipy 时使用网格回退。"""
    if len(c) < 2:
        return np.nan, np.nan
    try:
        from scipy.optimize import curve_fit

        p0 = [max(float(np.nanmax(y)), 1e-5) * 1.10, 0.30]
        params, _ = curve_fit(
            _saturating_value,
            c,
            y,
            p0=p0,
            bounds=([0.0, 1e-6], [np.inf, 10.0]),
            maxfev=50000,
        )
        return float(params[0]), float(params[1])
    except Exception:
        best = (np.inf, np.nan, np.nan)
        for b in np.linspace(0.01, 2.0, 2000):
            phi = 1.0 - np.exp(-b * (c - 1.0))
            denom = float(np.dot(phi, phi))
            if denom <= 0:
                continue
            a = max(0.0, float(np.dot(phi, y) / denom))
            pred = a * phi
            sse = float(np.sum((pred - y) ** 2))
            if sse < best[0]:
                best = (sse, a, float(b))
        return best[1], best[2]


def fit_saturating_model(
    observations: pd.DataFrame,
    target_col: str,
    model_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    c, y, data = _finite_xy(observations, "num_chunks", target_col)
    if len(y) < 3:
        return pd.DataFrame(), pd.DataFrame()

    a, b = _fit_saturating_params(c, y)
    pred = _saturating_value(c, a, b)
    metrics = _fit_metrics(y, pred)

    loo_true: List[float] = []
    loo_pred: List[float] = []
    if {"graph_tag", "repeat_id"}.issubset(data.columns):
        group_id = (
            data["graph_tag"].astype(str)
            + "__R"
            + data["repeat_id"].astype(str)
        )
        for holdout in group_id.unique():
            train = data[group_id != holdout]
            test = data[group_id == holdout]
            if len(train) < 3 or test.empty:
                continue
            c_train = train["num_chunks"].to_numpy(dtype=float)
            y_train = train[target_col].to_numpy(dtype=float)
            a_i, b_i = _fit_saturating_params(c_train, y_train)
            if not np.isfinite(a_i) or not np.isfinite(b_i):
                continue
            c_test = test["num_chunks"].to_numpy(dtype=float)
            y_test = test[target_col].to_numpy(dtype=float)
            loo_true.extend(y_test.tolist())
            loo_pred.extend(_saturating_value(c_test, a_i, b_i).tolist())

    loo_metrics = _fit_metrics(
        np.asarray(loo_true, dtype=float),
        np.asarray(loo_pred, dtype=float),
    ) if loo_true else {
        "n": 0,
        "r2": np.nan,
        "mae_sec": np.nan,
        "rmse_sec": np.nan,
        "max_abs_error_sec": np.nan,
    }

    params = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "target": target_col,
                "model_form": "a*(1-exp(-b*(C-1)))",
                "intercept": 0.0,
                "a": a,
                "b": b,
                **metrics,
                "loo_n": loo_metrics["n"],
                "loo_r2": loo_metrics["r2"],
                "loo_mae_sec": loo_metrics["mae_sec"],
                "loo_rmse_sec": loo_metrics["rmse_sec"],
            }
        ]
    )

    candidate_c = np.asarray(EXPECTED_ASYNC_CHUNKS, dtype=float)
    curve = pd.DataFrame(
        {
            "num_chunks": candidate_c.astype(int),
            f"pred_{target_col}": _saturating_value(candidate_c, a, b),
        }
    )
    return params, curve


def fit_linear_model(
    observations: pd.DataFrame,
    target_col: str,
    model_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    c, y, _ = _finite_xy(observations, "num_chunks", target_col)
    if len(y) < 2:
        return pd.DataFrame(), pd.DataFrame()
    X = np.column_stack([np.ones_like(c), c])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    metrics = _fit_metrics(y, pred)
    params = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "target": target_col,
                "model_form": "beta0+beta1*C",
                "beta0": float(beta[0]),
                "beta1": float(beta[1]),
                **metrics,
            }
        ]
    )
    candidate_c = np.asarray(EXPECTED_ASYNC_CHUNKS, dtype=float)
    curve = pd.DataFrame(
        {
            "num_chunks": candidate_c.astype(int),
            f"pred_{target_col}": beta[0] + beta[1] * candidate_c,
        }
    )
    return params, curve


def _ols_r2(y: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    metrics = _fit_metrics(y, pred)
    metrics["num_parameters"] = int(X.shape[1])
    return metrics


def communication_inflation_diagnostics(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """比较 J(C) 的 chunk-only 与 run-fixed-effect 解释力。"""
    c, y, data = _finite_xy(
        observations,
        "num_chunks",
        "J_comm_inflation_sec",
    )
    if len(y) < 3:
        return pd.DataFrame()

    logc = np.log(c)
    X_chunk = np.column_stack([np.ones_like(c), logc, c])
    rows = [
        {
            "model_name": "J_chunk_only_logC_plus_C",
            "target": "J_comm_inflation_sec",
            "model_form": "beta0+beta1*log(C)+beta2*C",
            **_ols_r2(y, X_chunk),
        }
    ]

    run_id = (
        data["graph_tag"].astype(str)
        + "__R"
        + data["repeat_id"].astype(str)
    )
    dummies = pd.get_dummies(run_id, drop_first=True, dtype=float)
    X_run = np.column_stack(
        [np.ones(len(data)), dummies.to_numpy(dtype=float)]
    )
    rows.append(
        {
            "model_name": "J_run_fixed_effect_only",
            "target": "J_comm_inflation_sec",
            "model_form": "run fixed effects",
            **_ols_r2(y, X_run),
        }
    )
    X_both = np.column_stack(
        [np.ones(len(data)), logc, c, dummies.to_numpy(dtype=float)]
    )
    rows.append(
        {
            "model_name": "J_chunk_plus_run_fixed_effect",
            "target": "J_comm_inflation_sec",
            "model_form": "log(C)+C+run fixed effects",
            **_ols_r2(y, X_both),
        }
    )
    return pd.DataFrame(rows)


def build_tradeoff_observations(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()

    required = {
        "aggregation_comm_hidden_sec_mean",
        "paired_aggregation_comm_total_sec_delta_vs_seq",
        "paired_local_message_compute_sec_delta_vs_seq",
    }
    if not required.issubset(paired.columns):
        return pd.DataFrame()

    obs = paired.copy()
    obs["num_chunks"] = pd.to_numeric(obs["num_chunks"], errors="coerce")
    obs["run_id"] = (
        obs["graph_tag"].astype(str)
        + "__R"
        + obs["repeat_id"].astype(str)
    )

    obs["H_hidden_benefit_sec"] = pd.to_numeric(
        obs["aggregation_comm_hidden_sec_mean"], errors="coerce"
    )
    obs["J_comm_inflation_sec"] = pd.to_numeric(
        obs["paired_aggregation_comm_total_sec_delta_vs_seq"],
        errors="coerce",
    )
    obs["D_compute_delta_sec"] = pd.to_numeric(
        obs["paired_local_message_compute_sec_delta_vs_seq"],
        errors="coerce",
    )
    obs["effective_comm_saving_sec"] = (
        obs["H_hidden_benefit_sec"]
        - obs["J_comm_inflation_sec"]
    )
    obs["G_agg_no_overhead_sec"] = (
        obs["H_hidden_benefit_sec"]
        - obs["J_comm_inflation_sec"]
        - obs["D_compute_delta_sec"]
    )

    overhead_col = (
        "paired_pipeline_residual_overhead_sec_delta_vs_seq"
    )
    if overhead_col in obs.columns:
        obs["overhead_delta_sec"] = pd.to_numeric(
            obs[overhead_col], errors="coerce"
        )
    else:
        obs["overhead_delta_sec"] = 0.0

    obs["G_agg_with_overhead_sec"] = (
        obs["G_agg_no_overhead_sec"]
        - obs["overhead_delta_sec"].fillna(0.0)
    )

    measured_pipeline_col = (
        "paired_forward_pipeline_total_sec_saved_vs_seq"
    )
    if measured_pipeline_col in obs.columns:
        obs["G_pipeline_measured_sec"] = pd.to_numeric(
            obs[measured_pipeline_col], errors="coerce"
        )

    backward_col = "paired_backward_sec_delta_vs_seq"
    obs["backward_delta_sec"] = (
        pd.to_numeric(obs[backward_col], errors="coerce")
        if backward_col in obs.columns else np.nan
    )

    rest_col = "paired_epoch_rest_approx_sec_delta_vs_seq"
    obs["rest_delta_sec"] = (
        pd.to_numeric(obs[rest_col], errors="coerce")
        if rest_col in obs.columns else np.nan
    )

    epoch_gain_col = "paired_epoch_total_sec_saved_vs_seq"
    obs["G_epoch_measured_sec"] = (
        pd.to_numeric(obs[epoch_gain_col], errors="coerce")
        if epoch_gain_col in obs.columns else np.nan
    )
    obs["G_epoch_decomposed_sec"] = (
        obs["G_agg_with_overhead_sec"]
        - obs["backward_delta_sec"]
        - obs["rest_delta_sec"]
    )

    keep = [
        "input_root", "graph_tag", "repeat_id", "run_id",
        "mode", "configuration_label", "num_chunks",
        "H_hidden_benefit_sec", "J_comm_inflation_sec",
        "D_compute_delta_sec", "effective_comm_saving_sec",
        "overhead_delta_sec", "G_agg_no_overhead_sec",
        "G_agg_with_overhead_sec", "G_pipeline_measured_sec",
        "backward_delta_sec", "rest_delta_sec",
        "G_epoch_decomposed_sec", "G_epoch_measured_sec",
        "paired_epoch_speedup_pct",
    ]
    return obs[[c for c in keep if c in obs.columns]].sort_values(
        ["graph_tag", "repeat_id", "num_chunks"],
        kind="stable",
    ).reset_index(drop=True)


def _t_critical_975(n: int) -> float:
    if n <= 1:
        return np.nan
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df=n - 1))
    except Exception:
        return 1.96


def tradeoff_chunk_summary(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    metrics = [
        "H_hidden_benefit_sec",
        "J_comm_inflation_sec",
        "D_compute_delta_sec",
        "effective_comm_saving_sec",
        "G_agg_no_overhead_sec",
        "G_agg_with_overhead_sec",
        "backward_delta_sec",
        "rest_delta_sec",
        "G_epoch_measured_sec",
    ]
    rows: List[Dict[str, Any]] = []
    for c, group in observations.groupby("num_chunks", dropna=False):
        record: Dict[str, Any] = {
            "num_chunks": int(c),
            "num_runs": int(group["run_id"].nunique()),
            "num_graphs": int(group["graph_tag"].nunique()),
        }
        for metric in metrics:
            if metric in group.columns:
                add_stats(record, metric, group[metric])

        for source, target in [
            ("G_agg_no_overhead_sec", "aggregation_positive_rate_pct"),
            ("G_epoch_measured_sec", "epoch_positive_rate_pct"),
        ]:
            x = pd.to_numeric(group[source], errors="coerce").dropna()
            record[target] = (
                float((x > 0).mean() * 100.0) if not x.empty else np.nan
            )

        n = record["num_runs"]
        for metric in ["G_agg_no_overhead_sec", "G_epoch_measured_sec"]:
            mean = record.get(f"{metric}_mean", np.nan)
            std = record.get(f"{metric}_std", np.nan)
            se = std / math.sqrt(n) if n > 0 and pd.notna(std) else np.nan
            record[f"{metric}_risk_score"] = (
                mean - RISK_LAMBDA * se
                if pd.notna(mean) and pd.notna(se) else np.nan
            )
            tcrit = _t_critical_975(n)
            record[f"{metric}_t95_lower"] = (
                mean - tcrit * se
                if pd.notna(mean) and pd.notna(se) and pd.notna(tcrit)
                else np.nan
            )
            record[f"{metric}_t95_upper"] = (
                mean + tcrit * se
                if pd.notna(mean) and pd.notna(se) and pd.notna(tcrit)
                else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows).sort_values("num_chunks").reset_index(drop=True)


def build_tradeoff_curve(
    chunk_summary: pd.DataFrame,
    h_curve: pd.DataFrame,
    d_curve: pd.DataFrame,
    backward_curve: pd.DataFrame,
) -> pd.DataFrame:
    base = pd.DataFrame({"num_chunks": EXPECTED_ASYNC_CHUNKS})
    for frame in [h_curve, d_curve, backward_curve]:
        if not frame.empty:
            base = base.merge(frame, on="num_chunks", how="left")
    if not chunk_summary.empty:
        cols = [
            c for c in [
                "num_chunks", "num_runs",
                "J_comm_inflation_sec_mean",
                "J_comm_inflation_sec_std",
                "G_agg_no_overhead_sec_mean",
                "G_agg_no_overhead_sec_std",
                "aggregation_positive_rate_pct",
                "G_epoch_measured_sec_mean",
                "G_epoch_measured_sec_std",
                "epoch_positive_rate_pct",
            ] if c in chunk_summary.columns
        ]
        base = base.merge(chunk_summary[cols], on="num_chunks", how="left")

    h_col = "pred_H_hidden_benefit_sec"
    d_col = "pred_D_compute_delta_sec"
    if {h_col, d_col, "J_comm_inflation_sec_mean"}.issubset(base.columns):
        base["pred_structural_margin_H_minus_D_sec"] = (
            base[h_col] - base[d_col]
        )
        base["pred_G_agg_mean_sec"] = (
            base[h_col]
            - base[d_col]
            - base["J_comm_inflation_sec_mean"]
        )
        n = pd.to_numeric(base.get("num_runs"), errors="coerce")
        j_std = pd.to_numeric(
            base.get("J_comm_inflation_sec_std"), errors="coerce"
        )
        j_se = j_std / np.sqrt(n)
        base["pred_G_agg_risk_score_sec"] = (
            base["pred_G_agg_mean_sec"] - RISK_LAMBDA * j_se
        )

    if "G_epoch_measured_sec_mean" in base.columns:
        n = pd.to_numeric(base.get("num_runs"), errors="coerce")
        std = pd.to_numeric(
            base.get("G_epoch_measured_sec_std"), errors="coerce"
        )
        base["empirical_G_epoch_risk_score_sec"] = (
            base["G_epoch_measured_sec_mean"]
            - RISK_LAMBDA * std / np.sqrt(n)
        )
    return base.sort_values("num_chunks").reset_index(drop=True)


def _select_with_fallback(
    frame: pd.DataFrame,
    score_col: str,
    label: str,
) -> Dict[str, Any]:
    if frame.empty or score_col not in frame.columns:
        return {
            "selection_target": label,
            "selected_mode": "unavailable",
            "selected_num_chunks": np.nan,
            "selected_score_sec": np.nan,
        }
    valid = frame.dropna(subset=[score_col])
    if valid.empty:
        return {
            "selection_target": label,
            "selected_mode": "unavailable",
            "selected_num_chunks": np.nan,
            "selected_score_sec": np.nan,
        }
    best = valid.loc[valid[score_col].idxmax()]
    score = float(best[score_col])
    if score <= 0:
        return {
            "selection_target": label,
            "selected_mode": BASELINE_MODE,
            "selected_num_chunks": 4,
            "selected_score_sec": 0.0,
            "best_async_num_chunks": int(best["num_chunks"]),
            "best_async_score_sec": score,
        }
    return {
        "selection_target": label,
        "selected_mode": f"async_chunk_{int(best['num_chunks'])}",
        "selected_num_chunks": int(best["num_chunks"]),
        "selected_score_sec": score,
        "best_async_num_chunks": int(best["num_chunks"]),
        "best_async_score_sec": score,
    }


def per_graph_chunk_selection(
    observations: pd.DataFrame,
    h_curve: pd.DataFrame,
    d_curve: pd.DataFrame,
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    hd = h_curve.merge(d_curve, on="num_chunks", how="outer")
    rows: List[Dict[str, Any]] = []
    for graph_tag, graph in observations.groupby("graph_tag"):
        summary = tradeoff_chunk_summary(graph)
        curve = build_tradeoff_curve(
            summary,
            h_curve,
            d_curve,
            pd.DataFrame(),
        )
        for score_col, target in [
            ("pred_G_agg_risk_score_sec", "aggregation_risk_adjusted"),
            ("empirical_G_epoch_risk_score_sec", "epoch_risk_adjusted"),
            ("G_agg_no_overhead_sec_mean", "aggregation_empirical_mean"),
            ("G_epoch_measured_sec_mean", "epoch_empirical_mean"),
        ]:
            selected = _select_with_fallback(curve, score_col, target)
            selected["graph_tag"] = graph_tag
            selected["num_repeats"] = int(graph["repeat_id"].nunique())
            rows.append(selected)
    return pd.DataFrame(rows)


def per_run_best_chunk(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for keys, group in observations.groupby(
        ["graph_tag", "repeat_id"], dropna=False
    ):
        graph_tag, repeat_id = keys
        record: Dict[str, Any] = {
            "graph_tag": graph_tag,
            "repeat_id": repeat_id,
        }
        for metric, prefix in [
            ("G_agg_no_overhead_sec", "aggregation"),
            ("G_epoch_measured_sec", "epoch"),
        ]:
            valid = group.dropna(subset=[metric])
            if valid.empty:
                continue
            best = valid.loc[valid[metric].idxmax()]
            if float(best[metric]) > 0:
                record[f"best_{prefix}_mode"] = best["mode"]
                record[f"best_{prefix}_num_chunks"] = int(
                    best["num_chunks"]
                )
                record[f"best_{prefix}_gain_sec"] = float(best[metric])
            else:
                record[f"best_{prefix}_mode"] = BASELINE_MODE
                record[f"best_{prefix}_num_chunks"] = 4
                record[f"best_{prefix}_gain_sec"] = 0.0
                record[f"best_async_{prefix}_num_chunks"] = int(
                    best["num_chunks"]
                )
                record[f"best_async_{prefix}_gain_sec"] = float(
                    best[metric]
                )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["graph_tag", "repeat_id"]
    ).reset_index(drop=True)


def model_consistency_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    pairs = [
        (
            "pipeline_identity_with_overhead",
            "G_pipeline_measured_sec",
            "G_agg_with_overhead_sec",
        ),
        (
            "epoch_phase_decomposition",
            "G_epoch_measured_sec",
            "G_epoch_decomposed_sec",
        ),
    ]
    for name, truth_col, pred_col in pairs:
        if not {truth_col, pred_col}.issubset(observations.columns):
            continue
        data = observations[[truth_col, pred_col]].dropna()
        if data.empty:
            continue
        metrics = _fit_metrics(
            data[truth_col].to_numpy(dtype=float),
            data[pred_col].to_numpy(dtype=float),
        )
        rows.append(
            {
                "model_name": name,
                "target": truth_col,
                "model_form": pred_col,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def make_tradeoff_plots(
    observations: pd.DataFrame,
    curve: pd.DataFrame,
    chunk_summary: pd.DataFrame,
) -> List[str]:
    outputs: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return outputs

    if not observations.empty and not curve.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        grouped = observations.groupby("num_chunks", as_index=False).agg(
            H_mean=("H_hidden_benefit_sec", "mean"),
            D_mean=("D_compute_delta_sec", "mean"),
        )
        ax.scatter(grouped["num_chunks"], grouped["H_mean"], label="Observed H mean")
        ax.scatter(grouped["num_chunks"], grouped["D_mean"], label="Observed D mean")
        if "pred_H_hidden_benefit_sec" in curve.columns:
            ax.plot(curve["num_chunks"], curve["pred_H_hidden_benefit_sec"], label="Fitted H(C)")
        if "pred_D_compute_delta_sec" in curve.columns:
            ax.plot(curve["num_chunks"], curve["pred_D_compute_delta_sec"], label="Fitted D(C)")
        ax.set_xlabel("Chunk count C")
        ax.set_ylabel("Time (s)")
        ax.set_title("Hidden communication benefit and compute cost")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = OUTPUT_DIR / "01_hidden_compute_fitted_curves.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs.append(str(path))

    plot_cols = [
        c for c in [
            "J_comm_inflation_sec_mean",
            "pred_G_agg_mean_sec",
            "pred_G_agg_risk_score_sec",
        ] if c in curve.columns and pd.to_numeric(
            curve[c], errors="coerce"
        ).notna().any()
    ]
    if not curve.empty and plot_cols:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        if "J_comm_inflation_sec_mean" in plot_cols:
            ax.plot(curve["num_chunks"], curve["J_comm_inflation_sec_mean"], marker="o", label="Mean J(C)")
        if "pred_G_agg_mean_sec" in plot_cols:
            ax.plot(curve["num_chunks"], curve["pred_G_agg_mean_sec"], marker="o", label="Predicted mean Gagg")
        if "pred_G_agg_risk_score_sec" in plot_cols:
            ax.plot(curve["num_chunks"], curve["pred_G_agg_risk_score_sec"], marker="o", label="Risk-adjusted Gagg")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("Chunk count C")
        ax.set_ylabel("Time benefit (s)")
        ax.set_title("Communication inflation and aggregation gain")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = OUTPUT_DIR / "02_communication_inflation_and_aggregation_gain.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs.append(str(path))

    if not chunk_summary.empty and "G_epoch_measured_sec_mean" in chunk_summary.columns:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = chunk_summary["num_chunks"]
        y = chunk_summary["G_epoch_measured_sec_mean"]
        std = chunk_summary.get("G_epoch_measured_sec_std")
        n = chunk_summary.get("num_runs")
        if std is not None and n is not None:
            yerr = pd.to_numeric(std, errors="coerce") / np.sqrt(pd.to_numeric(n, errors="coerce"))
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
        else:
            ax.plot(x, y, marker="o")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("Chunk count C")
        ax.set_ylabel("Measured steady-epoch gain (s)")
        ax.set_title("End-to-end steady-epoch gain")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = OUTPUT_DIR / "03_steady_epoch_gain_by_chunk.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def write_tradeoff_report(
    completeness: pd.DataFrame,
    graph_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    observations: pd.DataFrame,
    chunk_summary: pd.DataFrame,
    model_parameters: pd.DataFrame,
    model_metrics: pd.DataFrame,
    curve: pd.DataFrame,
    selections: pd.DataFrame,
    per_graph_selection: pd.DataFrame,
    plot_paths: Sequence[str],
) -> None:
    complete_jobs = (
        int((completeness["status"] == "complete").sum())
        if not completeness.empty else 0
    )
    expected_jobs = (
        len(EXPECTED_GRAPH_TAGS)
        * len(EXPECTED_REPEATS)
        * len(EXPECTED_MODES)
    )
    discovered_graphs = (
        sorted(observations["graph_tag"].astype(str).unique())
        if not observations.empty else []
    )

    lines = [
        "# Five-day forward-aggregation trade-off analysis",
        "",
        f"- Profile: `{PROFILE}`",
        f"- Input roots: `{[str(x) for x in INPUT_ROOTS]}`",
        f"- Expected graph tags: `{EXPECTED_GRAPH_TAGS}`",
        f"- Discovered graph tags: `{discovered_graphs}`",
        f"- Complete jobs: {complete_jobs}/{expected_jobs}",
        f"- Steady epochs start at epoch {STEADY_EPOCH_START}.",
        f"- Risk penalty lambda: {RISK_LAMBDA}.",
        "",
        "## Trade-off definition",
        "",
        "- H(C): hidden communication time.",
        "- J(C): total communication inflation relative to Sequential.",
        "- D(C): additional local-message computation.",
        "- Gagg(C) = H(C) - J(C) - D(C).",
        "- End-to-end gain additionally subtracts residual overhead, backward delta, and the remaining-stage delta.",
        "",
    ]

    if not model_parameters.empty:
        lines += [
            "## Fitted stable cost/benefit components",
            "",
            safe_markdown(model_parameters, 20),
            "",
        ]
    if not model_metrics.empty:
        lines += [
            "## Model diagnostics",
            "",
            safe_markdown(model_metrics, 30),
            "",
        ]
    if not chunk_summary.empty:
        key_cols = [
            c for c in [
                "num_chunks", "num_runs",
                "H_hidden_benefit_sec_mean",
                "J_comm_inflation_sec_mean",
                "D_compute_delta_sec_mean",
                "G_agg_no_overhead_sec_mean",
                "aggregation_positive_rate_pct",
                "G_epoch_measured_sec_mean",
                "epoch_positive_rate_pct",
                "G_agg_no_overhead_sec_risk_score",
                "G_epoch_measured_sec_risk_score",
            ] if c in chunk_summary.columns
        ]
        lines += [
            "## Chunk-level observed trade-off",
            "",
            safe_markdown(chunk_summary[key_cols], 20),
            "",
        ]
    if not curve.empty:
        key_cols = [
            c for c in [
                "num_chunks",
                "pred_H_hidden_benefit_sec",
                "pred_D_compute_delta_sec",
                "pred_structural_margin_H_minus_D_sec",
                "J_comm_inflation_sec_mean",
                "pred_G_agg_mean_sec",
                "pred_G_agg_risk_score_sec",
                "G_epoch_measured_sec_mean",
                "empirical_G_epoch_risk_score_sec",
            ] if c in curve.columns
        ]
        lines += [
            "## Fitted curve and risk-adjusted scores",
            "",
            safe_markdown(curve[key_cols], 20),
            "",
        ]
    if not selections.empty:
        lines += [
            "## Overall selections with Sequential fallback",
            "",
            safe_markdown(selections, 20),
            "",
        ]
    if not per_graph_selection.empty:
        lines += [
            "## Per-graph selections",
            "",
            safe_markdown(per_graph_selection, 50),
            "",
        ]
    if plot_paths:
        lines += ["## Plots", ""] + [f"- `{p}`" for p in plot_paths] + [""]
    if not completeness.empty and complete_jobs < expected_jobs:
        bad = completeness[completeness["status"] != "complete"]
        lines += [
            "## Missing or incomplete jobs",
            "",
            safe_markdown(
                bad[[
                    "graph_tag", "repeat_id", "mode", "status", "problems"
                ]],
                120,
            ),
            "",
        ]

    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage, agg_raw, runs = collect_results()
    agg_epoch_rank = build_agg_epoch_rank(agg_raw)
    agg_critical = select_critical_rank(agg_epoch_rank)

    stage_run = stage_run_summary(stage)
    agg_run = agg_run_summary(agg_critical)
    run_summary = merge_run_summaries(stage_run, agg_run)
    run_summary = add_run_level_phase_residuals(run_summary)
    run_summary = add_configuration_columns(run_summary)

    graph_summary = graph_mode_summary(run_summary)
    graph_summary = add_configuration_columns(graph_summary)
    overall = overall_mode_summary(graph_summary)
    relation_summary = relation_layer_summary(agg_raw)
    epoch_index_profile = epoch_profile(stage)
    epoch_index_profile = add_configuration_columns(epoch_index_profile)

    paired = paired_speedup(run_summary)
    paired_summary = paired_speedup_summary(paired)
    best_repeat = best_mode_by_repeat(run_summary, paired)
    best_overall = best_overall_criteria(graph_summary)
    completeness = experiment_completeness_multiday(
        runs, stage, agg_raw
    )

    observations = build_tradeoff_observations(paired)
    chunk_summary = tradeoff_chunk_summary(observations)

    h_params, h_curve = fit_saturating_model(
        observations,
        "H_hidden_benefit_sec",
        "hidden_benefit_saturating",
    )
    d_params, d_curve = fit_saturating_model(
        observations,
        "D_compute_delta_sec",
        "compute_cost_saturating",
    )
    b_params, b_curve = fit_linear_model(
        observations,
        "backward_delta_sec",
        "backward_delta_linear",
    )

    model_parameters = pd.concat(
        [x for x in [h_params, d_params, b_params] if not x.empty],
        ignore_index=True,
    ) if any(not x.empty for x in [h_params, d_params, b_params]) else pd.DataFrame()

    inflation_diag = communication_inflation_diagnostics(observations)
    consistency = model_consistency_metrics(observations)
    model_metrics = pd.concat(
        [x for x in [model_parameters.copy(), inflation_diag, consistency] if not x.empty],
        ignore_index=True,
        sort=False,
    ) if any(not x.empty for x in [model_parameters, inflation_diag, consistency]) else pd.DataFrame()

    curve = build_tradeoff_curve(
        chunk_summary,
        h_curve,
        d_curve,
        b_curve,
    )

    selection_rows = [
        _select_with_fallback(
            curve,
            "pred_G_agg_risk_score_sec",
            "aggregation_model_risk_adjusted",
        ),
        _select_with_fallback(
            curve,
            "G_agg_no_overhead_sec_mean",
            "aggregation_empirical_mean",
        ),
        _select_with_fallback(
            curve,
            "empirical_G_epoch_risk_score_sec",
            "epoch_empirical_risk_adjusted",
        ),
        _select_with_fallback(
            curve,
            "G_epoch_measured_sec_mean",
            "epoch_empirical_mean",
        ),
    ]
    selections = pd.DataFrame(selection_rows)
    per_graph_selection = per_graph_chunk_selection(
        observations, h_curve, d_curve
    )
    per_run_best = per_run_best_chunk(observations)

    plot_paths = make_tradeoff_plots(
        observations, curve, chunk_summary
    )

    # 原始与中间表。
    save_frame(runs, "run_manifest.csv")
    save_frame(completeness, "experiment_completeness.csv")
    save_frame(stage, "stage_epoch_sync_long.csv")
    save_frame(agg_raw, "forward_agg_relation_raw.csv")
    save_frame(agg_epoch_rank, "forward_agg_epoch_rank_long.csv")
    save_frame(
        agg_critical,
        "forward_agg_epoch_critical_rank_long.csv",
    )
    save_frame(stage_run, "stage_run_summary.csv")
    save_frame(agg_run, "forward_agg_run_summary.csv")

    # 主结果表。
    save_frame(run_summary, "repeat_mode_steady_summary.csv")
    save_frame(graph_summary, "graph_mode_summary.csv")
    save_frame(overall, "overall_mode_macro_summary.csv")
    save_frame(paired, "paired_repeat_comparison_vs_sequential.csv")
    save_frame(paired_summary, "paired_mode_summary_vs_sequential.csv")
    save_frame(best_repeat, "best_mode_by_repeat.csv")
    save_frame(best_overall, "best_mode_overall.csv")
    save_frame(relation_summary, "relation_layer_summary.csv")
    save_frame(epoch_index_profile, "epoch_index_profile.csv")

    # 权衡模型表。
    save_frame(observations, "tradeoff_observations.csv")
    save_frame(chunk_summary, "tradeoff_chunk_summary.csv")
    save_frame(model_parameters, "tradeoff_model_parameters.csv")
    save_frame(model_metrics, "tradeoff_model_fit_metrics.csv")
    save_frame(curve, "tradeoff_curve_predictions.csv")
    save_frame(selections, "overall_chunk_selection.csv")
    save_frame(per_graph_selection, "per_graph_chunk_selection.csv")
    save_frame(per_run_best, "per_run_best_chunk.csv")

    write_tradeoff_report(
        completeness,
        graph_summary,
        paired_summary,
        observations,
        chunk_summary,
        model_parameters,
        model_metrics,
        curve,
        selections,
        per_graph_selection,
        plot_paths,
    )

    complete_jobs = (
        int((completeness["status"] == "complete").sum())
        if not completeness.empty else 0
    )
    expected_jobs = (
        len(EXPECTED_GRAPH_TAGS)
        * len(EXPECTED_REPEATS)
        * len(EXPECTED_MODES)
    )

    print("=" * 120)
    print("Five-day Seq-to-Chunk10 repeat-2 trade-off analysis completed")
    print(f"Profile: {PROFILE}")
    print(f"Input roots: {[str(x) for x in INPUT_ROOTS]}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Experiment directories discovered: {len(runs)}")
    print(f"Complete expected jobs: {complete_jobs}/{expected_jobs}")

    if not model_parameters.empty:
        print("\nFitted stable component models:")
        key_cols = [
            c for c in [
                "model_name", "model_form", "a", "b",
                "beta0", "beta1", "r2", "mae_sec", "rmse_sec",
                "loo_r2", "loo_rmse_sec",
            ] if c in model_parameters.columns
        ]
        print(model_parameters[key_cols].to_string(index=False))

    if not chunk_summary.empty:
        print("\nChunk-level trade-off summary:")
        key_cols = [
            c for c in [
                "num_chunks", "num_runs",
                "H_hidden_benefit_sec_mean",
                "J_comm_inflation_sec_mean",
                "D_compute_delta_sec_mean",
                "G_agg_no_overhead_sec_mean",
                "aggregation_positive_rate_pct",
                "G_epoch_measured_sec_mean",
                "epoch_positive_rate_pct",
                "G_agg_no_overhead_sec_risk_score",
                "G_epoch_measured_sec_risk_score",
            ] if c in chunk_summary.columns
        ]
        print(chunk_summary[key_cols].to_string(index=False))

    if not selections.empty:
        print("\nOverall selections with Sequential fallback:")
        print(selections.to_string(index=False))

    if not completeness.empty and complete_jobs < expected_jobs:
        print("\nIncomplete or missing jobs:")
        bad = completeness[completeness["status"] != "complete"]
        print(
            bad[[
                "graph_tag", "repeat_id", "mode", "status", "problems"
            ]].to_string(index=False)
        )

    print("\nMain outputs:")
    for filename in [
        "tradeoff_observations.csv",
        "tradeoff_chunk_summary.csv",
        "tradeoff_model_parameters.csv",
        "tradeoff_model_fit_metrics.csv",
        "tradeoff_curve_predictions.csv",
        "overall_chunk_selection.csv",
        "per_graph_chunk_selection.csv",
        "report.md",
    ]:
        print(f"  - {OUTPUT_DIR / filename}")
    print("=" * 120)


if __name__ == "__main__":
    main()

