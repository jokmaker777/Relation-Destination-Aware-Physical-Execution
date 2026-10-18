from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
steps = (
    "08_multidate_preflight_no_parser.py",
    "09_overnight_multidate_ablation_no_parser.py",
)

for step_index, file_name in enumerate(steps, 1):
    script = ROOT / file_name
    print("=" * 100, flush=True)
    print(f"MULTIDATE OVERNIGHT STEP {step_index}/{len(steps)}: {script}", flush=True)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{file_name} failed with return code {completed.returncode}; "
            "please inspect the error printed above."
        )

print("MULTIDATE OVERNIGHT PIPELINE COMPLETED", flush=True)

