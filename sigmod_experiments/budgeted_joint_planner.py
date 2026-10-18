from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


PLAN_FIELDS = ("strategy", "block_edges", "chunks", "world_size", "mode", "overlap")
NUMERIC_FEATURES = (
    "total_labeled_edges",
    "block_edges",
    "chunks",
    "world_size",
    "num_superblocks",
    "max_rank_edges",
    "edge_imbalance_to_mean",
    "num_target_groups",
    "relation_skew",
    "degree_q50",
    "degree_q90",
    "degree_q99",
    "destination_footprint",
    "metadata_bytes",
    "bytes_per_range",
    "measured_bandwidth_gbps",
)


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def integer(value: Any, default: int = 0) -> int:
    return int(round(number(value, default)))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def structural_plan_id(row: Mapping[str, Any]) -> str:
    overlap = 1 if truthy(row.get("overlap")) else 0
    return (
        f"P={row.get('strategy', 'unknown')}|B={integer(row.get('block_edges'))}"
        f"|C={max(1, integer(row.get('chunks'), 1))}"
        f"|G={max(1, integer(row.get('world_size'), 1))}"
        f"|M={row.get('mode', 'unknown')}|A={overlap}"
    )


def compact_plan_id(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('strategy', 'unknown')}|B{integer(row.get('block_edges'))}"
        f"|C{max(1, integer(row.get('chunks'), 1))}"
        f"|G{max(1, integer(row.get('world_size'), 1))}"
        f"|{'async' if truthy(row.get('overlap')) else 'blocking'}"
    )


def aggregate_measurements(
    rows: Sequence[Mapping[str, Any]], horizon_epochs: int
) -> List[Dict[str, Any]]:
    keys = (
        "graph",
        "world_size",
        "strategy",
        "block_edges",
        "chunks",
        "mode",
        "overlap",
    )
    groups: MutableMapping[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("graph", "unknown")),
            max(1, integer(row.get("world_size"), 1)),
            str(row.get("strategy", "unknown")),
            integer(row.get("block_edges")),
            max(1, integer(row.get("chunks"), 1)),
            str(row.get("mode", "unknown")),
            truthy(row.get("overlap")),
        )
        if number(row.get("steady_epoch_sec_mean")) <= 0:
            continue
        groups[key].append(row)

    result: List[Dict[str, Any]] = []
    average_fields = {
        "prep_sec_max",
        "steady_epoch_sec_mean",
        "steady_epoch_sec_stdev",
        "peak_memory_mb",
        "num_superblocks",
        "max_rank_edges",
        "edge_imbalance_to_mean",
        "total_labeled_edges",
        "comm_total_sec_mean",
        "comm_exposed_sec_mean",
        "comm_hidden_sec_mean",
        "message_compute_sec_mean",
        "forward_pipeline_sec_mean",
        "overlap_ratio_mean",
        "dense_payload_bytes_mean",
        "compact_payload_bytes_mean",
        "payload_reduction_ratio_mean",
        "num_target_groups",
        "relation_skew",
        "degree_q50",
        "degree_q90",
        "degree_q99",
        "destination_footprint",
        "metadata_bytes",
        "bytes_per_range",
        "measured_bandwidth_gbps",
    }
    for key, repeats in groups.items():
        row: Dict[str, Any] = dict(zip(keys, key))
        for field in average_fields:
            values = [number(value.get(field)) for value in repeats]
            row[field] = statistics.mean(values) if values else 0.0
        row["repeat_count"] = len(repeats)
        row["horizon_epochs"] = int(horizon_epochs)
        row["actual_total_sec"] = row["prep_sec_max"] + horizon_epochs * row[
            "steady_epoch_sec_mean"
        ]
        row["trial_cost_sec"] = row["prep_sec_max"] + row["steady_epoch_sec_mean"]
        row["plan_id"] = structural_plan_id(row)
        row["compact_plan"] = compact_plan_id(row)
        result.append(row)
    result.sort(
        key=lambda row: (
            str(row["graph"]),
            integer(row["world_size"]),
            str(row["strategy"]),
            integer(row["block_edges"]),
            integer(row["chunks"]),
            str(row["mode"]),
        )
    )
    return result


class Encoder:
    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.strategies = sorted({str(row.get("strategy", "unknown")) for row in rows})
        self.modes = sorted({str(row.get("mode", "unknown")) for row in rows})
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def raw(self, row: Mapping[str, Any]) -> List[float]:
        chunks = max(1.0, number(row.get("chunks"), 1.0))
        block_edges = max(0.0, number(row.get("block_edges")))
        values = [
            math.log1p(max(0.0, number(row.get("total_labeled_edges")))),
            math.log1p(block_edges),
            chunks,
            1.0 / chunks,
            max(1.0, number(row.get("world_size"), 1.0)),
            math.log1p(max(0.0, number(row.get("num_superblocks")))),
            math.log1p(max(0.0, number(row.get("max_rank_edges")))),
            number(row.get("edge_imbalance_to_mean")),
            math.log1p(max(0.0, number(row.get("num_target_groups")))),
            number(row.get("relation_skew")),
            math.log1p(max(0.0, number(row.get("degree_q50")))),
            math.log1p(max(0.0, number(row.get("degree_q90")))),
            math.log1p(max(0.0, number(row.get("degree_q99")))),
            math.log1p(max(0.0, number(row.get("destination_footprint")))),
            math.log1p(max(0.0, number(row.get("metadata_bytes")))),
            math.log1p(max(0.0, number(row.get("bytes_per_range")))),
            math.log1p(max(0.0, number(row.get("measured_bandwidth_gbps")))),
            1.0 if truthy(row.get("overlap")) else 0.0,
        ]
        values.extend(
            1.0 if str(row.get("strategy")) == value else 0.0
            for value in self.strategies
        )
        values.extend(
            1.0 if str(row.get("mode")) == value else 0.0 for value in self.modes
        )
        return values

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = np.asarray([self.raw(row) for row in rows], dtype=np.float64)
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        return (matrix - self.mean) / self.scale

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("encoder is not fitted")
        matrix = np.asarray([self.raw(row) for row in rows], dtype=np.float64)
        return (matrix - self.mean) / self.scale


class WeightedLogRidge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coefficients: np.ndarray | None = None

    def fit(
        self, x: np.ndarray, y: Sequence[float], weights: Sequence[float] | None = None
    ) -> "WeightedLogRidge":
        target = np.log(np.maximum(np.asarray(y, dtype=np.float64), 1e-9))
        design = np.column_stack([np.ones((x.shape[0],)), x])
        sample_weights = (
            np.ones((x.shape[0],), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        root = np.sqrt(np.maximum(sample_weights, 1e-9))
        weighted_design = design * root[:, None]
        weighted_target = target * root
        penalty = np.eye(design.shape[1], dtype=np.float64) * self.alpha
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.solve(
            weighted_design.T @ weighted_design + penalty,
            weighted_design.T @ weighted_target,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("model is not fitted")
        design = np.column_stack([np.ones((x.shape[0],)), x])
        return np.exp(design @ self.coefficients)


def normalized(values: Sequence[float], reverse: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    lo, hi = float(array.min()), float(array.max())
    result = np.zeros_like(array) if hi - lo <= 1e-12 else (array - lo) / (hi - lo)
    return 1.0 - result if reverse else result


def plan_dimensions(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("strategy")),
        integer(row.get("block_edges")),
        integer(row.get("chunks")),
        integer(row.get("world_size")),
        truthy(row.get("overlap")),
    )


def one_hop(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return sum(a != b for a, b in zip(plan_dimensions(left), plan_dimensions(right))) == 1


def heuristic_index(rows: Sequence[Mapping[str, Any]]) -> int:
    def score(row: Mapping[str, Any]) -> Tuple[float, ...]:
        return (
            0.0 if str(row.get("strategy")) == "superblock_vector_balance" else 1.0,
            abs(math.log(max(1.0, number(row.get("block_edges"), 25_000)) / 25_000.0)),
            0.0 if not truthy(row.get("overlap")) else 1.0,
            abs(number(row.get("chunks"), 1.0) - 1.0),
            -number(row.get("world_size"), 1.0),
        )

    return min(range(len(rows)), key=lambda index: score(rows[index]))


def history_ranking(history: Sequence[Mapping[str, Any]]) -> List[str]:
    best_by_graph: Dict[str, float] = {}
    for row in history:
        graph = str(row["graph"])
        best_by_graph[graph] = min(
            best_by_graph.get(graph, float("inf")), number(row["actual_total_sec"])
        )
    scores: MutableMapping[str, List[float]] = defaultdict(list)
    for row in history:
        base = best_by_graph[str(row["graph"])]
        scores[str(row["plan_id"])].append(number(row["actual_total_sec"]) / base)
    return sorted(scores, key=lambda value: statistics.mean(scores[value]))


def pareto_frontier(predicted: np.ndarray, novelty: np.ndarray) -> List[int]:
    result: List[int] = []
    for index in range(len(predicted)):
        dominated = False
        for other in range(len(predicted)):
            if index == other:
                continue
            if (
                predicted[other] <= predicted[index]
                and novelty[other] >= novelty[index]
                and (
                    predicted[other] < predicted[index]
                    or novelty[other] > novelty[index]
                )
            ):
                dominated = True
                break
        if not dominated:
            result.append(index)
    return result or list(range(len(predicted)))


def replay_one(
    history: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    strategy: str,
    max_trials: int,
    budgets: Sequence[int],
    seed: int,
    ridge: float,
    exploration: float,
    feedback_weight: float,
    early_stop_gain: float,
    early_stop_patience: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    encoder = Encoder(list(history) + list(candidates))
    history_x = encoder.fit(history if history else candidates)
    candidate_x = encoder.transform(candidates)
    measured: List[int] = []
    unmeasured = set(range(len(candidates)))
    order: List[int] = []
    cumulative_planner_cost = 0.0
    last_improvement_trial = 0
    best_seen = float("inf")
    stopped = False
    warm_order = history_ranking(history) if history else []
    warm_indexes = [
        next(
            (i for i, row in enumerate(candidates) if str(row["plan_id"]) == plan),
            -1,
        )
        for plan in warm_order
    ]
    warm_indexes = [value for value in warm_indexes if value >= 0]
    random_order = list(range(len(candidates)))
    rng.shuffle(random_order)
    oracle_cost = min(number(row["actual_total_sec"]) for row in candidates)
    oracle_index = min(
        range(len(candidates)), key=lambda value: number(candidates[value]["actual_total_sec"])
    )

    while unmeasured and len(measured) < max_trials and not stopped:
        if strategy == "random":
            chosen = next(value for value in random_order if value in unmeasured)
        elif strategy == "history_warm" and not measured and warm_indexes:
            chosen = warm_indexes[0]
        elif strategy == "cold_start" and not measured:
            chosen = heuristic_index(candidates)
        else:
            training_rows: List[Mapping[str, Any]] = []
            targets: List[float] = []
            weights: List[float] = []
            if strategy == "history_warm":
                training_rows.extend(history)
                targets.extend(number(row["actual_total_sec"]) for row in history)
                weights.extend(1.0 for _ in history)
            for index in measured:
                training_rows.append(candidates[index])
                targets.append(number(candidates[index]["actual_total_sec"]))
                weights.append(feedback_weight)

            available = sorted(unmeasured)
            available_rows = [candidates[index] for index in available]
            if not training_rows:
                chosen = heuristic_index(candidates)
            else:
                train_x = encoder.transform(training_rows)
                model = WeightedLogRidge(ridge).fit(train_x, targets, weights)
                available_x = encoder.transform(available_rows)
                predicted = model.predict(available_x)
                distances = np.sqrt(
                    ((available_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
                ).min(axis=1)
                frontier = pareto_frontier(predicted, distances)
                candidate_positions = frontier
                if measured:
                    incumbent = min(
                        measured,
                        key=lambda index: number(candidates[index]["actual_total_sec"]),
                    )
                    neighbors = [
                        pos
                        for pos, original in enumerate(available)
                        if one_hop(candidates[incumbent], candidates[original])
                    ]
                    if neighbors:
                        candidate_positions = sorted(set(frontier + neighbors))
                cost_score = normalized(predicted)
                novelty_score = normalized(distances)
                acquisition = cost_score - exploration * novelty_score
                if measured:
                    incumbent = min(
                        measured,
                        key=lambda index: number(candidates[index]["actual_total_sec"]),
                    )
                    for pos, original in enumerate(available):
                        if one_hop(candidates[incumbent], candidates[original]):
                            acquisition[pos] -= 0.10
                best_position = min(candidate_positions, key=lambda pos: acquisition[pos])
                chosen = available[best_position]

        unmeasured.remove(chosen)
        measured.append(chosen)
        order.append(chosen)
        observed = number(candidates[chosen]["actual_total_sec"])
        cumulative_planner_cost += number(candidates[chosen]["trial_cost_sec"])
        gain = (best_seen - observed) / max(best_seen, 1e-12) if math.isfinite(best_seen) else 1.0
        if observed < best_seen:
            best_seen = observed
            if gain >= early_stop_gain:
                last_improvement_trial = len(measured)
        if (
            len(measured) >= 4
            and len(measured) - last_improvement_trial >= early_stop_patience
        ):
            stopped = True

    output: List[Dict[str, Any]] = []
    for budget in budgets:
        used = min(int(budget), len(order))
        if used <= 0:
            continue
        prefix = order[:used]
        best_index = min(
            prefix, key=lambda index: number(candidates[index]["actual_total_sec"])
        )
        best_cost = number(candidates[best_index]["actual_total_sec"])
        planner_cost = sum(number(candidates[index]["trial_cost_sec"]) for index in prefix)
        recommended = [best_index]
        remaining = [index for index in range(len(candidates)) if index not in prefix]
        if remaining:
            if strategy == "random":
                ranked_remaining = [index for index in random_order if index in remaining]
            else:
                training_rows: List[Mapping[str, Any]] = []
                training_targets: List[float] = []
                training_weights: List[float] = []
                if strategy == "history_warm":
                    training_rows.extend(history)
                    training_targets.extend(number(row["actual_total_sec"]) for row in history)
                    training_weights.extend(1.0 for _ in history)
                for index in prefix:
                    training_rows.append(candidates[index])
                    training_targets.append(number(candidates[index]["actual_total_sec"]))
                    training_weights.append(feedback_weight)
                if len(training_rows) >= 2:
                    recommendation_model = WeightedLogRidge(ridge).fit(
                        encoder.transform(training_rows),
                        training_targets,
                        training_weights,
                    )
                    remaining_predictions = recommendation_model.predict(
                        encoder.transform([candidates[index] for index in remaining])
                    )
                    ranked_remaining = [
                        remaining[position]
                        for position in np.argsort(remaining_predictions)
                    ]
                else:
                    ranked_remaining = remaining
            for index in ranked_remaining:
                if index not in recommended:
                    recommended.append(index)
                if len(recommended) >= 3:
                    break
        top3_cost = min(
            number(candidates[index]["actual_total_sec"])
            for index in recommended[:3]
        )
        output.append(
            {
                "graph": candidates[0]["graph"],
                "start_strategy": strategy,
                "seed": seed,
                "requested_budget": int(budget),
                "trials_used": used,
                "stopped_early": bool(stopped and used == len(order)),
                "candidate_count": len(candidates),
                "history_plan_count": len(history),
                "best_plan": candidates[best_index]["compact_plan"],
                "best_plan_id": candidates[best_index]["plan_id"],
                "best_cost_sec": best_cost,
                "oracle_plan": candidates[oracle_index]["compact_plan"],
                "oracle_plan_id": candidates[oracle_index]["plan_id"],
                "oracle_cost_sec": oracle_cost,
                "regret": (best_cost - oracle_cost) / max(oracle_cost, 1e-12),
                "top3_regret": (top3_cost - oracle_cost) / max(oracle_cost, 1e-12),
                "oracle_selected": best_index == oracle_index,
                "oracle_in_top3": oracle_index in recommended[:3],
                "within_1pct": best_cost <= oracle_cost * 1.01,
                "within_1pct_top3": top3_cost <= oracle_cost * 1.01,
                "top3_plan_ids": ";".join(
                    str(candidates[index]["plan_id"]) for index in recommended[:3]
                ),
                "planner_trial_cost_sec": planner_cost,
                "planner_cost_fraction_of_horizon": planner_cost / max(best_cost, 1e-12),
                "measurement_order": ";".join(candidates[index]["compact_plan"] for index in prefix),
                "measurement_plan_ids": ";".join(
                    str(candidates[index]["plan_id"]) for index in prefix
                ),
            }
        )
    return output


def replay_budgeted_planner(
    plans: Sequence[Mapping[str, Any]],
    budgets: Sequence[int],
    seeds: Sequence[int],
    ridge: float = 1.0,
    exploration: float = 0.15,
    feedback_weight: float = 6.0,
    early_stop_gain: float = 0.005,
    early_stop_patience: int = 3,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    graphs = sorted({str(row["graph"]) for row in plans})
    for graph in graphs:
        history = [row for row in plans if str(row["graph"]) != graph]
        candidates = [row for row in plans if str(row["graph"]) == graph]
        if len(history) < 2 or len(candidates) < 2:
            continue
        max_trials = min(max(budgets), len(candidates))
        for seed in seeds:
            for strategy in ("history_warm", "cold_start", "random"):
                output.extend(
                    replay_one(
                        history,
                        candidates,
                        strategy,
                        max_trials,
                        budgets,
                        seed,
                        ridge,
                        exploration,
                        feedback_weight,
                        early_stop_gain,
                        early_stop_patience,
                    )
                )
    return output


def summarize_replay(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: MutableMapping[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["start_strategy"]), integer(row["requested_budget"]))].append(row)
    result: List[Dict[str, Any]] = []
    for (strategy, budget), values in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        regrets = [number(row["regret"]) for row in values]
        top3_regrets = [number(row["top3_regret"]) for row in values]
        costs = [number(row["planner_trial_cost_sec"]) for row in values]
        result.append(
            {
                "start_strategy": strategy,
                "budget": budget,
                "fold_seed_count": len(values),
                "mean_regret": statistics.mean(regrets),
                "median_regret": statistics.median(regrets),
                "p90_regret": float(np.quantile(np.asarray(regrets), 0.90)),
                "max_regret": max(regrets),
                "mean_top3_regret": statistics.mean(top3_regrets),
                "median_top3_regret": statistics.median(top3_regrets),
                "p90_top3_regret": float(np.quantile(np.asarray(top3_regrets), 0.90)),
                "within_1pct_rate": statistics.mean(1.0 if truthy(row["within_1pct"]) else 0.0 for row in values),
                "within_1pct_top3_rate": statistics.mean(1.0 if truthy(row["within_1pct_top3"]) else 0.0 for row in values),
                "oracle_selection_rate": statistics.mean(1.0 if truthy(row["oracle_selected"]) else 0.0 for row in values),
                "oracle_top3_rate": statistics.mean(1.0 if truthy(row["oracle_in_top3"]) else 0.0 for row in values),
                "mean_planner_trial_cost_sec": statistics.mean(costs),
            }
        )
    return result


def r2_score(actual: Sequence[float], predicted: Sequence[float]) -> float:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    denominator = float(((y - y.mean()) ** 2).sum())
    return 0.0 if denominator <= 1e-12 else 1.0 - float(((y - p) ** 2).sum()) / denominator


def evaluate_memory_model(
    plans: Sequence[Mapping[str, Any]], capacity_mb: float, safety_factor: float
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    graphs = sorted({str(row["graph"]) for row in plans})
    for graph in graphs:
        train = [row for row in plans if str(row["graph"]) != graph and number(row.get("peak_memory_mb")) > 0]
        test = [row for row in plans if str(row["graph"]) == graph and number(row.get("peak_memory_mb")) > 0]
        if len(train) < 2 or not test:
            continue
        encoder = Encoder(train + test)
        x_train = encoder.fit(train)
        x_test = encoder.transform(test)
        model = WeightedLogRidge(1.0).fit(x_train, [number(row["peak_memory_mb"]) for row in train])
        predicted = model.predict(x_test)
        actual = np.asarray([number(row["peak_memory_mb"]) for row in test])
        training_actual = np.asarray([number(row["peak_memory_mb"]) for row in train])
        training_predicted = model.predict(x_train)
        residual_sigma_mb = float(np.std(training_actual - training_predicted))
        safety_gamma = 1.645  # one-sided 95% Gaussian upper bound
        for row, estimate, observed in zip(test, predicted, actual):
            safe_estimate = float(
                estimate * safety_factor + safety_gamma * residual_sigma_mb
            )
            predicted_feasible = None if capacity_mb <= 0 else bool(safe_estimate <= capacity_mb)
            actual_feasible = None if capacity_mb <= 0 else bool(observed <= capacity_mb)
            output.append(
                {
                    "graph": graph,
                    "plan": row["compact_plan"],
                    "plan_id": row["plan_id"],
                    "actual_peak_memory_mb": float(observed),
                    "predicted_peak_memory_mb": float(estimate),
                    "safe_predicted_peak_memory_mb": safe_estimate,
                    "residual_sigma_mb": residual_sigma_mb,
                    "safety_gamma": safety_gamma,
                    "absolute_error_mb": float(abs(estimate - observed)),
                    "absolute_percentage_error": float(abs(estimate - observed) / max(observed, 1e-12)),
                    "fold_r2": r2_score(actual.tolist(), predicted.tolist()),
                    "capacity_mb": float(capacity_mb),
                    "safety_factor": float(safety_factor),
                    "trial_cost_sec": number(row.get("trial_cost_sec")),
                    "predicted_feasible": predicted_feasible,
                    "actual_feasible": actual_feasible,
                    "false_prune": bool(actual_feasible and not predicted_feasible) if capacity_mb > 0 else None,
                    "unsafe_accept": bool(predicted_feasible and not actual_feasible) if capacity_mb > 0 else None,
                }
            )
    return output


def summarize_memory(rows: Sequence[Mapping[str, Any]], capacity_mb: float) -> Dict[str, Any]:
    errors = [number(row["absolute_error_mb"]) for row in rows]
    percentages = [number(row["absolute_percentage_error"]) for row in rows]
    actual_infeasible = [row for row in rows if row.get("actual_feasible") is False]
    actual_feasible = [row for row in rows if row.get("actual_feasible") is True]
    predicted_pruned = [row for row in rows if row.get("predicted_feasible") is False]
    return {
        "evaluated_plans": len(rows),
        "capacity_mb": capacity_mb,
        "mean_absolute_error_mb": statistics.mean(errors) if errors else None,
        "median_absolute_percentage_error": statistics.median(percentages) if percentages else None,
        "p95_absolute_percentage_error": float(np.quantile(np.asarray(percentages), 0.95)) if percentages else None,
        "predicted_pruned_plans": len(predicted_pruned),
        "predicted_prune_rate": len(predicted_pruned) / len(rows) if rows else None,
        "measured_trial_time_avoided_sec": sum(
            number(row.get("trial_cost_sec")) for row in predicted_pruned
        ),
        "false_prune_rate": (
            statistics.mean(1.0 if truthy(row.get("false_prune")) else 0.0 for row in actual_feasible)
            if actual_feasible
            else None
        ),
        "unsafe_accept_rate": (
            statistics.mean(1.0 if truthy(row.get("unsafe_accept")) else 0.0 for row in actual_infeasible)
            if actual_infeasible
            else None
        ),
        "important_note": (
            "unsafe_accept_rate is not identifiable until at least one measured plan is infeasible/OOM"
            if capacity_mb <= 0 or not actual_infeasible
            else "classification includes observed infeasible plans"
        ),
    }


def memory_capacity_sweep(
    rows: Sequence[Mapping[str, Any]], capacities_mb: Sequence[float]
) -> List[Dict[str, Any]]:
    """Evaluate pruning at explicit memory budgets using measured peaks.

    Budgets below the installed device capacity are counterfactual capacity
    limits, not observed CUDA OOM executions.  They are nevertheless useful
    for checking whether a conservative estimator ever admits a measured plan
    whose peak exceeds the stated budget.
    """
    output: List[Dict[str, Any]] = []
    for capacity in sorted({float(value) for value in capacities_mb if float(value) > 0}):
        evaluated = list(rows)
        actual_infeasible = [
            row for row in evaluated if number(row.get("actual_peak_memory_mb")) > capacity
        ]
        actual_feasible = [
            row for row in evaluated if number(row.get("actual_peak_memory_mb")) <= capacity
        ]
        predicted_pruned = [
            row for row in evaluated if number(row.get("safe_predicted_peak_memory_mb")) > capacity
        ]
        unsafe = [
            row
            for row in actual_infeasible
            if number(row.get("safe_predicted_peak_memory_mb")) <= capacity
        ]
        false_reject = [
            row
            for row in actual_feasible
            if number(row.get("safe_predicted_peak_memory_mb")) > capacity
        ]
        output.append(
            {
                "capacity_mb": capacity,
                "evaluation_kind": "measured_peak_capacity_limit",
                "evaluated_plans": len(evaluated),
                "actual_infeasible_plans": len(actual_infeasible),
                "predicted_pruned_plans": len(predicted_pruned),
                "predicted_prune_rate": len(predicted_pruned) / len(evaluated) if evaluated else None,
                "unsafe_accept_count": len(unsafe),
                "unsafe_accept_rate": len(unsafe) / len(actual_infeasible) if actual_infeasible else None,
                "false_reject_count": len(false_reject),
                "false_reject_rate": len(false_reject) / len(actual_feasible) if actual_feasible else None,
                "measured_trial_time_avoided_sec": sum(
                    number(row.get("trial_cost_sec")) for row in predicted_pruned
                ),
                "note": (
                    "A plan is infeasible at this budget when its measured peak exceeds the budget; "
                    "this is not an intentionally triggered CUDA OOM run."
                ),
            }
        )
    return output


def fallback_evaluation(plans: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: MutableMapping[Tuple[str, int, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in plans:
        grouped[(str(row["graph"]), integer(row["world_size"]), str(row["strategy"]), integer(row["block_edges"]))].append(row)
    output: List[Dict[str, Any]] = []
    graphs = sorted({str(row["graph"]) for row in plans})
    for heldout in graphs:
        historical_deltas: MutableMapping[Tuple[int, str, int], List[float]] = defaultdict(list)
        global_deltas: List[float] = []
        for key, values in grouped.items():
            graph, world, strategy, block = key
            if graph == heldout:
                continue
            blocking = [number(row["actual_total_sec"]) for row in values if not truthy(row["overlap"])]
            async_values = [number(row["actual_total_sec"]) for row in values if truthy(row["overlap"])]
            if blocking and async_values:
                delta = min(async_values) / min(blocking) - 1.0
                historical_deltas[(world, strategy, block)].append(delta)
                global_deltas.append(delta)
        for key, values in grouped.items():
            graph, world, strategy, block = key
            if graph != heldout:
                continue
            blocking_rows = [row for row in values if not truthy(row["overlap"])]
            async_rows = [row for row in values if truthy(row["overlap"])]
            if not blocking_rows or not async_rows:
                continue
            blocking = min(blocking_rows, key=lambda row: number(row["actual_total_sec"]))
            async_plan = min(async_rows, key=lambda row: number(row["actual_total_sec"]))
            deltas = historical_deltas.get((world, strategy, block), global_deltas)
            predicted_delta = statistics.median(deltas) if deltas else 0.0
            chosen = async_plan if predicted_delta < 0 else blocking
            oracle = min((blocking, async_plan), key=lambda row: number(row["actual_total_sec"]))
            chosen_cost = number(chosen["actual_total_sec"])
            oracle_cost = number(oracle["actual_total_sec"])
            output.append(
                {
                    "graph": graph,
                    "world_size": world,
                    "strategy": strategy,
                    "block_edges": block,
                    "predicted_async_delta": predicted_delta,
                    "chosen_mode": "async" if truthy(chosen["overlap"]) else "blocking",
                    "oracle_mode": "async" if truthy(oracle["overlap"]) else "blocking",
                    "selection_correct": truthy(chosen["overlap"]) == truthy(oracle["overlap"]),
                    "regret": (chosen_cost - oracle_cost) / max(oracle_cost, 1e-12),
                    "blocking_cost_sec": number(blocking["actual_total_sec"]),
                    "async_cost_sec": number(async_plan["actual_total_sec"]),
                }
            )
    return output


def summarize_fallback(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    regrets = [number(row["regret"]) for row in rows]
    return {
        "decisions": len(rows),
        "selection_accuracy": statistics.mean(1.0 if truthy(row["selection_correct"]) else 0.0 for row in rows) if rows else None,
        "mean_regret": statistics.mean(regrets) if regrets else None,
        "max_regret": max(regrets) if regrets else None,
        "async_selected_rate": statistics.mean(1.0 if row["chosen_mode"] == "async" else 0.0 for row in rows) if rows else None,
        "oracle_async_rate": statistics.mean(1.0 if row["oracle_mode"] == "async" else 0.0 for row in rows) if rows else None,
    }


def assignment_bounds(root: Path) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    covered: set[Tuple[str, str]] = set()
    for path in root.rglob("superblock_assignment_*.csv"):
        rows = read_csv(path)
        if not rows or "assigned_rank" not in rows[0]:
            continue
        loads: MutableMapping[int, float] = defaultdict(float)
        weights: List[float] = []
        for row in rows:
            weight = number(row.get("estimated_cost"), number(row.get("num_edges")))
            weights.append(weight)
            loads[integer(row.get("assigned_rank"))] += weight
        if not loads or not weights:
            continue
        world = len(loads)
        total = sum(weights)
        largest = max(weights)
        lower = max(total / world, largest)
        upper = total / world + largest
        makespan = max(loads.values())
        strategy = str(rows[0].get("strategy", "unknown"))
        covered.add((str(path.parent.resolve()), strategy))
        output.append(
            {
                "assignment_file": str(path),
                "source_kind": "per_block_assignment",
                "strategy": strategy,
                "world_size": world,
                "num_blocks": len(weights),
                "total_work_W": total,
                "W_over_G": total / world,
                "w_max": largest,
                "lower_bound": lower,
                "greedy_upper_bound": upper,
                "actual_makespan": makespan,
                "makespan_over_lower_bound": makespan / max(lower, 1e-12),
                "within_greedy_bound": makespan <= upper + 1e-9,
            }
        )

    # Normal experiment runs deliberately disable the per-block assignment
    # CSV because it can be very large.  Newer training runs place the exact
    # W/G, w_max and makespan scalars in the compact per-rank timing CSVs, so
    # evaluate those as a lossless fallback.
    grouped: MutableMapping[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for path in root.rglob("partition_stage_timing_rank*.csv"):
        rows = read_csv(path)
        if not rows:
            continue
        row = rows[-1]
        if number(row.get("total_work_W")) <= 0 or number(row.get("w_max")) <= 0:
            continue
        strategy = str(row.get("partition_strategy", "unknown"))
        grouped[(str(path.parent.resolve()), strategy)].append(row)

    for key, rows in grouped.items():
        if key in covered:
            continue
        parent, strategy = key
        exemplar = rows[0]
        world = max(1, integer(exemplar.get("world_size"), len(rows)))
        total = number(exemplar.get("total_work_W"))
        largest = number(exemplar.get("w_max"))
        lower = number(exemplar.get("lower_bound"), max(total / world, largest))
        upper = number(exemplar.get("greedy_upper_bound"), total / world + largest)
        makespan = number(exemplar.get("actual_makespan"))
        if makespan <= 0:
            makespan = max(number(row.get("rank_selected_estimated_work")) for row in rows)
        output.append(
            {
                "assignment_file": parent,
                "source_kind": "compact_partition_timing",
                "strategy": strategy,
                "weight_kind": str(exemplar.get("assignment_weight_kind", "unknown")),
                "world_size": world,
                "num_blocks": integer(exemplar.get("num_superblocks")),
                "total_work_W": total,
                "W_over_G": number(exemplar.get("W_over_G"), total / world),
                "w_max": largest,
                "lower_bound": lower,
                "greedy_upper_bound": upper,
                "actual_makespan": makespan,
                "makespan_over_lower_bound": makespan / max(lower, 1e-12),
                "within_greedy_bound": makespan <= upper + 1e-9,
            }
        )
    return output


def create_plots(output_dir: Path, replay_summary: Sequence[Mapping[str, Any]], memory_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if replay_summary:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for strategy in ("history_warm", "cold_start", "random"):
            rows = [row for row in replay_summary if row["start_strategy"] == strategy]
            rows.sort(key=lambda row: integer(row["budget"]))
            ax.plot(
                [integer(row["budget"]) for row in rows],
                [100.0 * number(row["mean_regret"]) for row in rows],
                marker="o",
                label=strategy.replace("_", " "),
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Trial budget H")
        ax.set_ylabel("Mean best-seen regret (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "budget_regret_curve.png", dpi=220)
        plt.close(fig)
    if memory_rows:
        actual = [number(row["actual_peak_memory_mb"]) for row in memory_rows]
        predicted = [number(row["predicted_peak_memory_mb"]) for row in memory_rows]
        lo, hi = min(actual + predicted), max(actual + predicted)
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        ax.scatter(actual, predicted, s=18, alpha=0.7)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
        ax.set_xlabel("Measured peak memory (MB)")
        ax.set_ylabel("Predicted peak memory (MB)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "memory_prediction_parity.png", dpi=220)
        plt.close(fig)


def run_analysis(
    input_csv: Path,
    output_dir: Path,
    source_root: Path | None,
    horizon_epochs: int,
    budgets: Sequence[int],
    seeds: Sequence[int],
    memory_capacity_mb: float,
    memory_safety_factor: float,
) -> Dict[str, Any]:
    raw = read_csv(input_csv)
    plans = aggregate_measurements(raw, horizon_epochs)
    if len(plans) < 4:
        raise RuntimeError("At least four completed plan measurements are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "unified_plan_table.csv", plans)

    replay_rows = replay_budgeted_planner(plans, budgets, seeds)
    replay_summary = summarize_replay(replay_rows)
    write_csv(output_dir / "budgeted_replay_detail.csv", replay_rows)
    write_csv(output_dir / "budgeted_replay_summary.csv", replay_summary)

    memory_rows = evaluate_memory_model(plans, memory_capacity_mb, memory_safety_factor)
    memory_summary = summarize_memory(memory_rows, memory_capacity_mb)
    write_csv(output_dir / "memory_model_evaluation.csv", memory_rows)
    sweep_capacities = [24576.0, 32768.0, 40960.0, 49152.0]
    if memory_capacity_mb > 0:
        sweep_capacities.append(float(memory_capacity_mb))
    memory_sweep_rows = memory_capacity_sweep(memory_rows, sweep_capacities)
    write_csv(output_dir / "memory_capacity_sweep.csv", memory_sweep_rows)
    (output_dir / "memory_model_summary.json").write_text(
        json.dumps(memory_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fallback_rows = fallback_evaluation(plans)
    fallback_summary = summarize_fallback(fallback_rows)
    write_csv(output_dir / "blocking_fallback_evaluation.csv", fallback_rows)
    (output_dir / "blocking_fallback_summary.json").write_text(
        json.dumps(fallback_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bound_rows = assignment_bounds(source_root) if source_root is not None else []
    write_csv(output_dir / "assignment_bound_evaluation.csv", bound_rows)
    create_plots(output_dir, replay_summary, memory_rows)

    coverage = {
        "raw_measurement_rows": len(raw),
        "unique_plans": len(plans),
        "graphs": sorted({str(row["graph"]) for row in plans}),
        "world_sizes": sorted({integer(row["world_size"]) for row in plans}),
        "strategies": sorted({str(row["strategy"]) for row in plans}),
        "block_edges": sorted({integer(row["block_edges"]) for row in plans}),
        "chunks": sorted({integer(row["chunks"]) for row in plans}),
        "modes": sorted({str(row["mode"]) for row in plans}),
        "is_full_joint_PBCG": (
            len({str(row["strategy"]) for row in plans}) >= 2
            and len({integer(row["block_edges"]) for row in plans}) >= 2
            and len({integer(row["chunks"]) for row in plans}) >= 2
            and len({integer(row["world_size"]) for row in plans}) >= 2
        ),
    }
    result = {
        "coverage": coverage,
        "planner_replay_rows": len(replay_rows),
        "planner_summary_rows": len(replay_summary),
        "memory": memory_summary,
        "memory_capacity_sweep_rows": len(memory_sweep_rows),
        "fallback": fallback_summary,
        "assignment_bound_rows": len(bound_rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "two_day_analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Budgeted feedback joint-planner replay")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--reuse-epochs", type=int, default=50)
    parser.add_argument("--budgets", default="1,2,4,8,16,32")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--memory-capacity-mb", type=float, default=0.0)
    parser.add_argument("--memory-safety-factor", type=float, default=1.10)
    args = parser.parse_args()
    budgets = sorted({max(1, int(value)) for value in args.budgets.split(",") if value.strip()})
    seeds = [args.seed + index for index in range(max(1, args.seeds))]
    run_analysis(
        Path(args.input),
        Path(args.output_dir),
        Path(args.source_root) if args.source_root else None,
        args.reuse_epochs,
        budgets,
        seeds,
        args.memory_capacity_mb,
        args.memory_safety_factor,
    )


if __name__ == "__main__":
    main()

