from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from run_pipeline_recurrence_2to4gpu_no_parser import recurrence


def test_two_stream_recurrence_and_terminal_tail():
    total, tail = recurrence([2.0, 2.0, 2.0], [3.0, 3.0, 3.0])
    assert total == 11.0
    assert tail == 5.0


def test_compute_dominant_pipeline():
    total, tail = recurrence([5.0, 5.0], [1.0, 1.0])
    assert total == 11.0
    assert tail == 1.0

