from __future__ import annotations

from sigmod_experiments.budgeted_joint_planner import (
    aggregate_measurements,
    assignment_bounds,
    fallback_evaluation,
    memory_capacity_sweep,
    replay_budgeted_planner,
    summarize_replay,
    write_csv,
)


def sample_rows():
    rows = []
    for graph, offset in (("d1", 0.0), ("d2", 1.0), ("d3", 2.0)):
        for chunks, epoch in ((1, 12.0 + offset), (2, 10.0 + offset), (4, 11.0 + offset)):
            rows.append(
                {
                    "graph": graph,
                    "world_size": 2,
                    "strategy": "superblock_vector_balance",
                    "block_edges": 25_000,
                    "chunks": chunks,
                    "mode": "blocking_c1" if chunks == 1 else f"async_chunk_{chunks}",
                    "overlap": chunks != 1,
                    "prep_sec_max": 2.0,
                    "steady_epoch_sec_mean": epoch,
                    "peak_memory_mb": 1000.0 + chunks,
                    "num_superblocks": 100,
                    "max_rank_edges": 500,
                    "edge_imbalance_to_mean": 1.02,
                    "total_labeled_edges": 1000,
                }
            )
    return rows


def test_budgeted_replay_never_beats_oracle_by_construction():
    plans = aggregate_measurements(sample_rows(), horizon_epochs=10)
    replay = replay_budgeted_planner(plans, budgets=(1, 2, 4), seeds=(7, 8))
    assert replay
    assert all(float(row["regret"]) >= -1e-12 for row in replay)
    assert all(int(row["trials_used"]) <= int(row["requested_budget"]) for row in replay)
    summary = summarize_replay(replay)
    assert {row["start_strategy"] for row in summary} == {
        "history_warm",
        "cold_start",
        "random",
    }


def test_fallback_uses_other_dates_only():
    plans = aggregate_measurements(sample_rows(), horizon_epochs=10)
    rows = fallback_evaluation(plans)
    assert len(rows) == 3
    assert all(row["oracle_mode"] == "async" for row in rows)
    assert all(float(row["regret"]) >= 0.0 for row in rows)


def test_compact_assignment_bound_fallback(tmp_path):
    run = tmp_path / "one_run"
    common = {
        "world_size": 2,
        "partition_strategy": "superblock_edge_count",
        "assignment_weight_kind": "edge_count",
        "num_superblocks": 3,
        "total_work_W": 30.0,
        "W_over_G": 15.0,
        "w_max": 12.0,
        "lower_bound": 15.0,
        "greedy_upper_bound": 27.0,
        "actual_makespan": 17.0,
    }
    write_csv(
        run / "partition_stage_timing_rank0.csv",
        [{**common, "rank": 0, "rank_selected_estimated_work": 17.0}],
    )
    write_csv(
        run / "partition_stage_timing_rank1.csv",
        [{**common, "rank": 1, "rank_selected_estimated_work": 13.0}],
    )
    rows = assignment_bounds(tmp_path)
    assert len(rows) == 1
    assert rows[0]["source_kind"] == "compact_partition_timing"
    assert rows[0]["actual_makespan"] == 17.0
    assert rows[0]["within_greedy_bound"] is True


def test_memory_capacity_sweep_reports_unsafe_and_false_reject():
    rows = [
        {
            "actual_peak_memory_mb": 20.0,
            "safe_predicted_peak_memory_mb": 25.0,
            "trial_cost_sec": 2.0,
        },
        {
            "actual_peak_memory_mb": 40.0,
            "safe_predicted_peak_memory_mb": 30.0,
            "trial_cost_sec": 3.0,
        },
    ]
    summary = memory_capacity_sweep(rows, (32.0,))[0]
    assert summary["unsafe_accept_count"] == 1
    assert summary["false_reject_count"] == 0
    assert summary["predicted_pruned_plans"] == 0

