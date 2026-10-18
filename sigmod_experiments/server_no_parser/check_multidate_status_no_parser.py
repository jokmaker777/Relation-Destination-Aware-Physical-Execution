from __future__ import annotations

import json
from collections import Counter

from common_no_parser import result_json_is_complete
from settings_no_parser import BASE_OUTPUT_ROOT, MULTI_DATE_GRAPH_TAGS


root = BASE_OUTPUT_ROOT / "industrial" / "multidate_ablation" / "gpus-4"
completed = []
if root.exists():
    for result in root.rglob("train_result_rank0.json"):
        if not result_json_is_complete(result, "mode"):
            continue
        try:
            payload = json.loads(result.read_text(encoding="utf-8"))
            completed.append(str(payload.get("graph_tag", "unknown")))
        except (OSError, ValueError, TypeError):
            continue

counts = Counter(completed)
print(f"MULTIDATE STATUS: {len(completed)}/72 jobs complete")
for graph_tag in MULTI_DATE_GRAPH_TAGS:
    print(f"  {graph_tag}: {counts[graph_tag]}/6")

failure_path = BASE_OUTPUT_ROOT / "multidate_ablation_failures.json"
if failure_path.exists():
    try:
        failures = json.loads(failure_path.read_text(encoding="utf-8"))
        print(f"RECORDED FAILURES: {len(failures)} ({failure_path})")
    except (OSError, ValueError, TypeError):
        print(f"FAILURE FILE EXISTS BUT CANNOT BE READ: {failure_path}")

