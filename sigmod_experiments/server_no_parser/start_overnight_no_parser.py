from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from settings_no_parser import BASE_OUTPUT_ROOT


script = Path(__file__).resolve().parent / "05a_overnight_synthetic_only_no_parser.py"
state_dir = BASE_OUTPUT_ROOT / "overnight"
state_dir.mkdir(parents=True, exist_ok=True)
log_path = state_dir / "synthetic_overnight.log"
pid_path = state_dir / "synthetic_overnight.pid"

if pid_path.exists():
    try:
        old_pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(old_pid, 0)
        raise RuntimeError(f"夜间任务似乎仍在运行，PID={old_pid}")
    except ProcessLookupError:
        pass
    except ValueError:
        pass

environment = os.environ.copy()
with log_path.open("a", encoding="utf-8") as stream:
    stream.write("\n===== START NEW OVERNIGHT RUN =====\n")
    stream.flush()
    process = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(script.parent),
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )

pid_path.write_text(str(process.pid), encoding="utf-8")
print(f"夜间任务已在服务器后台启动，PID={process.pid}")
print(f"日志={log_path}")
print("现在可以关闭 PyCharm/SSH；明早运行 check_overnight_status_no_parser.py")


