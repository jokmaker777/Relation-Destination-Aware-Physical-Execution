from __future__ import annotations

import os

from settings_no_parser import BASE_OUTPUT_ROOT


state_dir = BASE_OUTPUT_ROOT / "overnight"
log_path = state_dir / "synthetic_overnight.log"
pid_path = state_dir / "synthetic_overnight.pid"

if not pid_path.exists():
    raise RuntimeError("没有找到夜间任务 PID；请先运行 start_overnight_no_parser.py")

pid = int(pid_path.read_text(encoding="utf-8").strip())
try:
    os.kill(pid, 0)
    running = True
except ProcessLookupError:
    running = False

lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
completed = any("OVERNIGHT SYNTHETIC MATRIX COMPLETED" in line for line in lines[-200:])

print(f"PID={pid}")
print(f"running={running}")
print(f"completed={completed}")
print(f"log={log_path}")
print("\n===== LOG TAIL =====")
print("\n".join(lines[-100:]))


