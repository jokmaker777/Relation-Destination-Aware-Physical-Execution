from common_no_parser import run_industrial_job
from settings_no_parser import GPU_COUNTS, LAYOUT_STRATEGIES, SCALE_MODES


for gpu_count in GPU_COUNTS:
    for strategy in LAYOUT_STRATEGIES:
        for mode in SCALE_MODES:
            run_industrial_job(
                stage="scale_primary",
                world_size=gpu_count,
                strategy=strategy,
                block_edges=25_000,
                mode=mode,
                repeat=1,
            )

print("INDUSTRIAL 1/2/3/4-GPU SCALE RUNS COMPLETED")


