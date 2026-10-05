"""A thermal governor for long benchmarks on a laptop: CPU and GPU share one cooling envelope.

``governor()`` is cheap to call every step; it only reads the sensors every ``every`` calls and, when either chip
is hot, sleeps (GPU idle) until both have cooled below the resume thresholds.
"""

from __future__ import annotations

import glob
import subprocess
import time

PAUSE_GPU, PAUSE_CPU = 76, 88
RESUME_GPU, RESUME_CPU = 64, 76
_calls = 0
paused_seconds = 0.0


def gpu_temp() -> int:
    try:
        return int(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], text=True, timeout=10
            ).split()[0]
        )
    except Exception:
        return 0


def cpu_temp() -> int:
    temps = []
    for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            if open(zone + "/type").read().strip() == "x86_pkg_temp":
                temps.append(int(open(zone + "/temp").read()) // 1000)
        except OSError:
            pass
    return max(temps) if temps else 0


def governor(every: int = 3) -> float:
    """Sleep while too hot. Returns the seconds spent paused (to be excluded from throughput numbers)."""
    global _calls, paused_seconds
    _calls += 1
    if _calls % every:
        return 0.0
    if gpu_temp() < PAUSE_GPU and cpu_temp() < PAUSE_CPU:
        return 0.0
    t0 = time.perf_counter()
    while gpu_temp() > RESUME_GPU or cpu_temp() > RESUME_CPU:
        time.sleep(3)
    dt = time.perf_counter() - t0
    paused_seconds += dt
    return dt
