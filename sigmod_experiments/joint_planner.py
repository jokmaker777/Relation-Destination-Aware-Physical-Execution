from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


PLAN_KEYS = ("graph", "world_size", "strategy", "block_edges", "chunks", "mode", "overlap")
MEASURE_KEYS = (
    "prep_sec_max",
    "steady_epoch_sec_mean",
    "num_superblocks",
    "max_rank_edges",
    "edge_imbalance_to_mean",
    "total_labeled_edges",
    "peak_memory_mb",
)


def f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_grouped(path: Path, horizon: int) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["graph"],
            int(f(row["world_size"])),
            row["strategy"],
            int(f(row["block_edges"])),
            int(f(row["chunks"])),
            row["mode"],
            truthy(row["overlap"]),
        )
        grouped[key].append(row)
    result: List[Dict[str, Any]] = []
    for key, repeats in grouped.items():
        value = dict(zip(PLAN_KEYS, key))
        for measure in MEASURE_KEYS:
            value[measure] = statistics.mean(f(row.get(measure)) for row in repeats)
        value["repeat_count"] = len(repeats)
        value["actual_total_sec"] = (
            value["prep_sec_max"] + horizon * value["steady_epoch_sec_mean"]
        )
        result.append(value)
    return result


class FeatureEncoder:
    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.strategies = sorted({str(row["strategy"]) for row in rows})
        self.modes = sorted({str(row["mode"]) for row in rows})
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def raw(self, row: Mapping[str, Any]) -> List[float]:
        chunks = max(1.0, f(row["chunks"]))
        numeric = [
            math.log1p(f(row["total_labeled_edges"])),
            math.log1p(f(row["block_edges"])),
            chunks,
            1.0 / chunks,
            f(row["world_size"]),
            math.log1p(f(row["num_superblocks"])),
            math.log1p(f(row["max_rank_edges"])),
            f(row["edge_imbalance_to_mean"]),
            1.0 if bool(row["overlap"]) else 0.0,
        ]
        numeric.extend(1.0 if row["strategy"] == value else 0.0 for value in self.strategies)
        numeric.extend(1.0 if row["mode"] == value else 0.0 for value in self.modes)
        return numeric

    def fit_transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = np.asarray([self.raw(row) for row in rows], dtype=np.float64)
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        return (matrix - self.mean) / self.scale

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("FeatureEncoder has not been fitted")
        matrix = np.asarray([self.raw(row) for row in rows], dtype=np.float64)
        return (matrix - self.mean) / self.scale


class LogRidge:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.coefficients: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: Sequence[float]) -> "LogRidge":
        target = np.log(np.maximum(np.asarray(y, dtype=np.float64), 1e-9))
        design = np.column_stack([np.ones((x.shape[0],)), x])
        penalty = np.eye(design.shape[1], dtype=np.float64) * self.alpha
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("LogRidge has not been fitted")
        design = np.column_stack([np.ones((x.shape[0],)), x])
        return np.exp(design @ self.coefficients)


def plan_id(row: Mapping[str, Any]) -> str:
    return (
        f"{row['strategy']}|B={int(f(row['block_edges']))}|C={int(f(row['chunks']))}"
        f"|{row['mode']}"
    )


def closest_heuristic(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    def score(row: Mapping[str, Any]) -> Tuple[float, ...]:
        return (
            0.0 if row["strategy"] == "superblock_vector_balance" else 1.0,
            abs(math.log(max(1.0, f(row["block_edges"])) / 25_000.0)),
            0.0 if bool(row["overlap"]) else 1.0,
            abs(f(row["chunks"]) - 4.0),
        )

    return min(rows, key=score)


def r2_score(actual: Sequence[float], predicted: Sequence[float]) -> float:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    denominator = float(((y - y.mean()) ** 2).sum())
    if denominator <= 1e-12:
        return 0.0
    return 1.0 - float(((y - p) ** 2).sum()) / denominator


def evaluate_fold(
    train: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    horizon: int,
    ridge: float,
    random_budget: int,
    seed: int,
) -> Dict[str, Any]:
    encoder = FeatureEncoder(train + candidates)
    x_train = encoder.fit_transform(train)
    x_test = encoder.transform(candidates)
    prep_model = LogRidge(ridge).fit(x_train, [row["prep_sec_max"] for row in train])
    epoch_model = LogRidge(ridge).fit(
        x_train, [row["steady_epoch_sec_mean"] for row in train]
    )
    predicted_prep = prep_model.predict(x_test)
    predicted_epoch = epoch_model.predict(x_test)
    predicted_total = predicted_prep + horizon * predicted_epoch
    actual = np.asarray([row["actual_total_sec"] for row in candidates])
    order = np.argsort(predicted_total)
    joint = candidates[int(order[0])]
    oracle = candidates[int(np.argmin(actual))]

    prep_order = np.argsort(predicted_prep)
    prep_choice = candidates[int(prep_order[0])]
    matching = [
        index
        for index, row in enumerate(candidates)
        if row["strategy"] == prep_choice["strategy"]
        and int(row["block_edges"]) == int(prep_choice["block_edges"])
    ]
    two_stage_index = min(matching, key=lambda index: predicted_epoch[index])
    two_stage = candidates[two_stage_index]
    heuristic = closest_heuristic(candidates)

    rng = random.Random(seed)
    sample_indexes = list(range(len(candidates)))
    rng.shuffle(sample_indexes)
    sample_indexes = sample_indexes[: min(random_budget, len(sample_indexes))]
    random_choice = min(sample_indexes, key=lambda index: candidates[index]["actual_total_sec"])
    random_plan = candidates[random_choice]

    oracle_cost = float(oracle["actual_total_sec"])
    joint_cost = float(joint["actual_total_sec"])
    top3_actual = min(float(candidates[int(index)]["actual_total_sec"]) for index in order[:3])
    return {
        "graph": candidates[0]["graph"],
        "world_size": int(candidates[0]["world_size"]),
        "candidate_count": len(candidates),
        "training_plan_count": len(train),
        "horizon_epochs": horizon,
        "oracle_plan": plan_id(oracle),
        "oracle_cost_sec": oracle_cost,
        "joint_plan": plan_id(joint),
        "joint_cost_sec": joint_cost,
        "joint_top1_regret": (joint_cost - oracle_cost) / max(oracle_cost, 1e-12),
        "joint_top3_regret": (top3_actual - oracle_cost) / max(oracle_cost, 1e-12),
        "two_stage_plan": plan_id(two_stage),
        "two_stage_cost_sec": float(two_stage["actual_total_sec"]),
        "two_stage_regret": (
            float(two_stage["actual_total_sec"]) - oracle_cost
        ) / max(oracle_cost, 1e-12),
        "heuristic_plan": plan_id(heuristic),
        "heuristic_cost_sec": float(heuristic["actual_total_sec"]),
        "heuristic_regret": (
            float(heuristic["actual_total_sec"]) - oracle_cost
        ) / max(oracle_cost, 1e-12),
        "random_budget": len(sample_indexes),
        "random_plan": plan_id(random_plan),
        "random_cost_sec": float(random_plan["actual_total_sec"]),
        "random_regret": (
            float(random_plan["actual_total_sec"]) - oracle_cost
        ) / max(oracle_cost, 1e-12),
        "heldout_total_cost_r2": r2_score(actual.tolist(), predicted_total.tolist()),
        "predicted_joint_cost_sec": float(predicted_total[int(order[0])]),
    }


def mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return statistics.mean(f(row[field]) for row in rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-graph-out joint plan evaluation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-epochs", type=int, default=50)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--random-budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    plans = load_grouped(Path(args.input), args.reuse_epochs)
    graphs = sorted({str(row["graph"]) for row in plans})
    evaluation: List[Dict[str, Any]] = []
    for graph in graphs:
        train = [row for row in plans if row["graph"] != graph]
        heldout = [row for row in plans if row["graph"] == graph]
        for world_size in sorted({int(row["world_size"]) for row in heldout}):
            candidates = [row for row in heldout if int(row["world_size"]) == world_size]
            if len(candidates) < 2 or len(train) < 2:
                continue
            evaluation.append(
                evaluate_fold(
                    train,
                    candidates,
                    args.reuse_epochs,
                    args.ridge,
                    args.random_budget,
                    args.seed + world_size + len(evaluation),
                )
            )
    if not evaluation:
        raise RuntimeError("Insufficient plans for leave-one-graph-out evaluation")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "planner_evaluation.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluation[0]))
        writer.writeheader()
        writer.writerows(evaluation)
    summary = {
        "folds": len(evaluation),
        "graphs": graphs,
        "horizon_epochs": args.reuse_epochs,
        "mean_joint_top1_regret": mean_field(evaluation, "joint_top1_regret"),
        "mean_joint_top3_regret": mean_field(evaluation, "joint_top3_regret"),
        "mean_two_stage_regret": mean_field(evaluation, "two_stage_regret"),
        "mean_heuristic_regret": mean_field(evaluation, "heuristic_regret"),
        "mean_random_regret": mean_field(evaluation, "random_regret"),
        "mean_heldout_total_cost_r2": mean_field(evaluation, "heldout_total_cost_r2"),
        "important_note": (
            "Plan-selection regret is primary. Report R2 as a diagnostic and do not "
            "hide negative held-out values."
        ),
    }
    (output_dir / "planner_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

