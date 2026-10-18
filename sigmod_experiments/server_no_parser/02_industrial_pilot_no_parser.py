from common_no_parser import run_industrial_job
from settings_no_parser import BLOCKING_C1, ASYNC_C4


for gpu_count in (1, 2):
    for mode in (BLOCKING_C1, ASYNC_C4):
        run_industrial_job(
            stage="scale_primary",
            world_size=gpu_count,
            strategy="balanced_edge",
            block_edges=25_000,
            mode=mode,
            repeat=1,
        )

print("INDUSTRIAL PILOT COMPLETED")


