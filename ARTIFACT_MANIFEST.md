# Artifact Manifest

| Paper evidence | Primary code | Purpose |
|---|---|---|
| Core RD/SuperBlock execution | `end_to_end_prunning/models/communication_optimized_training_manager_forward_agg_overlap_no_parser.py` | Relation--destination fragments, placement, blocking/async reduction, measurements |
| Industrial end-to-end runs | `run_forward_aggregation_overlap_5day_seq_chunk10_repeat2_ep10_no_parser.py` | Exact full-graph training entry point |
| Main evaluation suite | `run_sigmod_core_suite_no_parser.py` | Public DBLP, industrial layouts, scaling, C2 check, planner summaries |
| Figure 5 industrial results/ablation | `run_sigmod_core_suite_no_parser.py` and `sigmod_experiments/collect_industrial_results.py` | Direct, organization, vector-aware, Planner, and Oracle comparisons |
| Figure 6 strong-scaling boundary | `run_active_destination_large_edge_scaling_1to4gpu_no_parser.py`, `run_active_destination_scaling_1to4gpu_no_parser.py` | Controlled and industrial 1--4 GPU scaling |
| Figure 6 active-destination sensitivity | `run_active_destination_phase_boundary_4gpu_no_parser.py` and `run_active_destination_overnight_4gpu_no_parser.py` | Dense/compact phase boundary |
| Figure 7 planner quality/cost | `sigmod_experiments/budgeted_joint_planner.py` and `run_remaining_joint_constraints_no_parser.py` | Budgeted replay, regret, horizon cost, warm/cold/random baselines |
| Figure 8 memory feasibility | `run_sigmod_gpuA_supplements_no_parser.py` and `run_remaining_joint_constraints_no_parser.py` | Controlled capacity sweep and conservative feasibility checks |
| Figure 9 HGB-DBLP | `run_sigmod_core_suite_no_parser.py` | Native PyG and exact custom R-GCN semantic/runtime comparison |
| Forward/gradient exactness | `run_exactness_matrix_1to4gpu_no_parser.py` | 1--4 GPU forward and gradient checks |
| Collective calibration | `run_collective_alpha_beta_1to4gpu_no_parser.py` | Startup/bandwidth calibration by GPU count |
| Pipeline recurrence | `run_pipeline_recurrence_2to4gpu_no_parser.py` | Predicted versus measured exposed communication |
| Blocking fallback | `run_collective_safety_gate_no_parser.py` and `run_async_scheduler_search_no_parser.py` | Async-versus-blocking selection and regret |
| Paper figures/tables | `analysis/build_sigmod_paper_figures.py` | Rebuilds figures and source-data tables from collected summaries |

The numbered scripts in `sigmod_experiments/server_no_parser/` provide the
older step-by-step, argument-free workflow.  They are retained for auditability
and compatibility with the result directories used during evaluation.

