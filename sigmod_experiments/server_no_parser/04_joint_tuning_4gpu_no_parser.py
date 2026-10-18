from common_no_parser import run_industrial_job
from settings_no_parser import BLOCK_EDGE_CHOICES, SUPERBLOCK_STRATEGIES, TUNING_MODES


for strategy in SUPERBLOCK_STRATEGIES:
    for block_edges in BLOCK_EDGE_CHOICES:
        for mode in TUNING_MODES:
            run_industrial_job(
                stage="joint_tuning",
                world_size=4,
                strategy=strategy,
                block_edges=block_edges,
                mode=mode,
                repeat=1,
            )

print("SINGLE-GRAPH 4-GPU JOINT TUNING COMPLETED")


