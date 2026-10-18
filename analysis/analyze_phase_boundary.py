import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    os.environ.get(
        "SIGMOD_PHASE_BOUNDARY_RESULTS_ROOT",
        "/path/to/artifact_workspace/outputs/active_destination_phase_boundary_4gpu",
    )
)
df = pd.read_csv(ROOT / "phase_boundary_4gpu_summary.csv")

print("SHAPE", df.shape)
print("METHOD_COUNTS", df.groupby("method").size().to_dict())
print("AXIS_COUNTS", df.groupby("axis")["case_id"].nunique().to_dict())
print("COMPLETED", df["completed"].value_counts(dropna=False).to_dict())
print("NULLS", {k: int(v) for k, v in df.isna().sum().items() if v})

for metric in [
    "forward_relative_l2_error",
    "gradient_relative_l2_error",
    "step_sec_stdev",
]:
    index = df[metric].idxmax()
    print("MAX", metric, float(df.loc[index, metric]), df.loc[index, "case_id"], df.loc[index, "method"])

df["step_cv_percent"] = 100 * df["step_sec_stdev"] / df["step_sec_mean"]
print(
    "STEP_CV_PERCENT",
    {
        "median": float(df["step_cv_percent"].median()),
        "p95": float(df["step_cv_percent"].quantile(0.95)),
        "max": float(df["step_cv_percent"].max()),
        "gt5_count": int((df["step_cv_percent"] > 5).sum()),
    },
)

methods = ["direct_dense", "direct_compact", "superblock_dense", "superblock_compact"]
for axis in ["active_ratio", "hidden_dim", "relation_count", "scale_edges", "degree_skew"]:
    sub = df[df["axis"] == axis].copy()
    time = sub.pivot(index="axis_value", columns="method", values="step_sec_mean")[methods]
    speed = time["direct_dense"].to_numpy()[:, None] / time.to_numpy()
    speed = pd.DataFrame(speed, index=time.index, columns=time.columns)
    best = time.idxmin(axis=1)
    payload = sub.pivot(index="axis_value", columns="method", values="payload_reduction_percent")[methods]
    memory = sub.pivot(index="axis_value", columns="method", values="peak_gpu_memory_mb")[methods]
    prep = sub.pivot(index="axis_value", columns="method", values="prep_sec_max")[methods]
    imbalance = sub.pivot(index="axis_value", columns="method", values="edge_imbalance_max_over_mean")[methods]
    print("\nAXIS", axis)
    print("TIME_SEC")
    print(time.round(6).to_string())
    print("SPEEDUP_VS_DIRECT_DENSE")
    print(speed.round(3).to_string())
    print("BEST", best.to_dict())
    print("PAYLOAD_REDUCTION_PERCENT")
    print(payload.round(2).to_string())
    print("MEMORY_MB")
    print(memory.round(1).to_string())
    print("PREP_SEC")
    print(prep.round(4).to_string())
    print("IMBALANCE")
    print(imbalance.round(4).to_string())

pivot = df.pivot(index="case_id", columns="method", values="step_sec_mean")
comparisons = pd.DataFrame(index=pivot.index)
comparisons["direct_compact_vs_dense"] = pivot["direct_dense"] / pivot["direct_compact"]
comparisons["superblock_compact_vs_dense"] = pivot["superblock_dense"] / pivot["superblock_compact"]
comparisons["superblock_vs_direct_dense"] = pivot["direct_dense"] / pivot["superblock_dense"]
comparisons["superblock_vs_direct_compact"] = pivot["direct_compact"] / pivot["superblock_compact"]
for col in comparisons:
    values = comparisons[col]
    print(
        "COMPARISON",
        col,
        {
            "geomean": float(np.exp(np.log(values).mean())),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
            "wins_gt1pct": int((values > 1.01).sum()),
            "losses_gt1pct": int((values < 0.99).sum()),
            "cases": int(values.size),
        },
    )

prep = df.pivot(index="case_id", columns="method", values="prep_sec_max")
for horizon in [1, 5, 10, 25, 50, 100, 200]:
    total = prep + horizon * pivot
    best = total.idxmin(axis=1)
    counts = best.value_counts().to_dict()
    regret = total.div(total.min(axis=1), axis=0) - 1
    print("HORIZON", horizon, "BEST_COUNTS", counts)
    print(
        "HORIZON_SUPERBLOCK_COMPACT_REGRET",
        horizon,
        {
            "median_percent": float(100 * regret["superblock_compact"].median()),
            "p95_percent": float(100 * regret["superblock_compact"].quantile(0.95)),
        },
    )

baseline_time = pivot["direct_dense"]
baseline_prep = prep["direct_dense"]
for method in methods[1:]:
    saving = baseline_time - pivot[method]
    extra_prep = prep[method] - baseline_prep
    break_even = extra_prep / saving
    break_even[(saving <= 0) | (extra_prep <= 0)] = np.where(
        (saving > 0) & (extra_prep <= 0), 0.0, np.inf
    )
    finite = break_even[np.isfinite(break_even)]
    print(
        "BREAK_EVEN_STEPS",
        method,
        {
            "finite_cases": int(finite.size),
            "median": None if finite.empty else float(finite.median()),
            "p95": None if finite.empty else float(finite.quantile(0.95)),
            "never": int(np.isinf(break_even).sum()),
        },
    )

best_step = pivot.idxmin(axis=1)
print("BEST_STEP_COUNTS", best_step.value_counts().to_dict())

for method in ["direct_compact", "superblock_compact"]:
    mem = df.pivot(index="case_id", columns="method", values="peak_gpu_memory_mb")
    reduction = 100 * (1 - mem[method] / mem["direct_dense"])
    print(
        "MEM_REDUCTION_VS_DIRECT_DENSE",
        method,
        {
            "median_percent": float(reduction.median()),
            "min_percent": float(reduction.min()),
            "max_percent": float(reduction.max()),
        },
    )


