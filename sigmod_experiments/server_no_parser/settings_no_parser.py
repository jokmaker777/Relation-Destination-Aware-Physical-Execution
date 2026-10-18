import os
from pathlib import Path


# =========================
# 只需要检查这一段配置
# =========================

ARTIFACT_WORKSPACE_ROOT = Path(
    os.environ.get("SIGMOD_ARTIFACT_ROOT", "/path/to/artifact_workspace")
)
INDUSTRIAL_GRAPH_ROOT = Path(
    os.environ.get(
        "SIGMOD_GRAPH_CACHE_ROOT",
        str(ARTIFACT_WORKSPACE_ROOT / "cache"),
    )
)

GRAPH_TAG = "20241129"
GRAPH_PATH = INDUSTRIAL_GRAPH_ROOT / f"graph_{GRAPH_TAG}.pkl"

MULTI_DATE_GRAPH_TAGS = (
    "20241124",
    "20241125",
    "20241126",
    "20241127",
    "20241128",
    "20241129",
    "20241213",
    "20241214",
    "20241215",
    "20241216",
    "20241217",
    "20241218",
)
REPRESENTATIVE_GRAPH_TAGS = (
    "20241124",
    "20241129",
    "20241213",
    "20241218",
)
PLAN_GRID_GRAPH_TAGS = (
    "20241124",
    "20241127",
    "20241129",
    "20241213",
    "20241216",
    "20241218",
)


def industrial_graph_path(graph_tag: str) -> Path:
    return INDUSTRIAL_GRAPH_ROOT / f"graph_{graph_tag}.pkl"

# 按希望使用的物理 GPU 顺序填写。1/2/3/4 卡实验依次使用列表前 1/2/3/4 张。
GPU_IDS = [0, 1, 2, 3]

BASE_OUTPUT_ROOT = Path(
    os.environ.get(
        "SIGMOD_OUTPUT_ROOT",
        str(ARTIFACT_WORKSPACE_ROOT / "outputs"),
    )
)
PUBLIC_DATA_ROOT = Path(
    os.environ.get(
        "SIGMOD_PUBLIC_DATA_ROOT",
        str(ARTIFACT_WORKSPACE_ROOT / "public_datasets"),
    )
)

EPOCHS = 10
BASE_SEED = 20261004
REUSE_HORIZON_EPOCHS = 50

# =========================
# 以下通常不需要修改
# =========================

SERVER_NO_PARSER_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SERVER_NO_PARSER_ROOT.parent
RUNNER_FILE_NAME = (
    "run_forward_aggregation_overlap_5day_seq_chunk10_repeat2_ep10_no_parser.py"
)

# 如果自动搜索不到，可把 None 改成服务器上的绝对路径，例如：
# INDUSTRIAL_RUNNER_OVERRIDE = Path("/path/to/artifact_code/run_....py")
INDUSTRIAL_RUNNER_OVERRIDE = None

INDUSTRIAL_RUNNER_CANDIDATES = (
    EXPERIMENT_ROOT / RUNNER_FILE_NAME,
    EXPERIMENT_ROOT.parent / RUNNER_FILE_NAME,
    Path.cwd() / RUNNER_FILE_NAME,
)
if INDUSTRIAL_RUNNER_OVERRIDE is not None:
    INDUSTRIAL_RUNNER = Path(INDUSTRIAL_RUNNER_OVERRIDE)
else:
    INDUSTRIAL_RUNNER = next(
        (path for path in INDUSTRIAL_RUNNER_CANDIDATES if path.exists()),
        None,
    )
    if INDUSTRIAL_RUNNER is None:
        # PyCharm 可能改变部署层级；在当前项目和用户服务器目录中再自动搜索。
        for search_root in (EXPERIMENT_ROOT, ARTIFACT_WORKSPACE_ROOT):
            if not search_root.exists():
                continue
            INDUSTRIAL_RUNNER = next(search_root.rglob(RUNNER_FILE_NAME), None)
            if INDUSTRIAL_RUNNER is not None:
                break
    if INDUSTRIAL_RUNNER is None:
        INDUSTRIAL_RUNNER = INDUSTRIAL_RUNNER_CANDIDATES[0]

# 原训练脚本会从它所在目录导入 end_to_end_prunning；公开算子任务则从
# sigmod_experiments 导入。因此根据任务自动选择合适的 cwd。
PROJECT_ROOT = (
    INDUSTRIAL_RUNNER.parent if INDUSTRIAL_RUNNER.exists() else EXPERIMENT_ROOT
)
OPERATOR_NO_PARSER = SERVER_NO_PARSER_ROOT / "benchmark_operator_no_parser.py"
METADATA_PATH = BASE_OUTPUT_ROOT / "industrial_graph_metadata.json"
MULTIDATE_METADATA_PATH = BASE_OUTPUT_ROOT / "multidate_graph_metadata.json"
LOG_ROOT = BASE_OUTPUT_ROOT / "launcher_logs"

LABELED_RELATIONS = (
    "inventory__to__virtual",
    "inventory__to_transport_labeled__inventory",
)

GPU_COUNTS = (1, 2, 3, 4)
LAYOUT_STRATEGIES = (
    "balanced_edge",
    "superblock_edge_count",
    "superblock_vector_balance",
)
SUPERBLOCK_STRATEGIES = (
    "superblock_edge_count",
    "superblock_vector_balance",
)
BLOCK_EDGE_CHOICES = (10_000, 25_000, 50_000, 100_000)

BLOCKING_C1 = {"name": "blocking_c1", "chunks": 1, "overlap": False}
BLOCKING_C4 = {"name": "blocking_c4", "chunks": 4, "overlap": False}
ASYNC_C2 = {"name": "async_chunk_2", "chunks": 2, "overlap": True}
ASYNC_C4 = {"name": "async_chunk_4", "chunks": 4, "overlap": True}
ASYNC_C8 = {"name": "async_chunk_8", "chunks": 8, "overlap": True}

SCALE_MODES = (BLOCKING_C1, ASYNC_C4)
TUNING_MODES = (BLOCKING_C1, BLOCKING_C4, ASYNC_C2, ASYNC_C4, ASYNC_C8)

PUBLIC_DATASETS = (
    {
        "name": "hgb-dblp",
        "kind": "hgb-dblp",
        "root": str(PUBLIC_DATA_ROOT / "hgb"),
        "relation_selector": "topk:3",
    },
    {
        "name": "synthetic-uniform",
        "kind": "synthetic",
        "relation_selector": "all",
        "validate_correctness": True,
        "options": {
            "num_src": 100_000,
            "num_dst": 100_000,
            "num_edges": 1_000_000,
            "num_relations": 3,
            "skew": 0.0,
            "seed": BASE_SEED,
        },
    },
    {
        "name": "synthetic-skewed",
        "kind": "synthetic",
        "relation_selector": "all",
        "validate_correctness": True,
        "options": {
            "num_src": 100_000,
            "num_dst": 100_000,
            "num_edges": 1_000_000,
            "num_relations": 3,
            "skew": 2.0,
            "seed": BASE_SEED,
        },
    },
    {
        "name": "ogbn-mag",
        "kind": "ogbn-mag",
        "root": str(PUBLIC_DATA_ROOT / "ogb"),
        "relation_selector": "topk:3",
    },
)

OPERATOR_POLICIES = (
    "contiguous",
    "round_robin",
    "destination_hash",
    "destination_greedy",
    "superblock_greedy",
)
OPERATOR_CHUNKS = (1, 2, 4, 8)
OPERATOR_BLOCK_EDGES = (10_000, 25_000, 50_000, 100_000)

