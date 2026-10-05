"""Orchestrate the quality comparison: a learning-rate sweep per method (seed 0), then extra seeds at the best rate.

    python benchmarks/compare_peft.py sweep      # phase 1, resumable
    python benchmarks/compare_peft.py seeds      # phase 2: seeds 1,2 at each config's best learning rate
    python benchmarks/compare_peft.py report     # markdown + JSON summary

One process at a time, with a cool-down between runs: this machine's GPU also drives the display.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "results" / "quality"
PY = sys.executable
MAX_TEMP = 70

EIGEN_DIAG = [3e-3, 1e-2, 3e-2]
EIGEN_CORE = [1e-3, 3e-3, 1e-2]
LORA = [1e-4, 3e-4, 1e-3]
# (method, rank, learning-rate grid)
GRID = [
    ("eigentune_diag", 16, EIGEN_DIAG),
    ("eigentune_core", 8, EIGEN_CORE),
    ("eigentune_diag", 64, EIGEN_DIAG),
    ("eigentune_core", 16, EIGEN_CORE),
    ("lora", 1, LORA),
    ("lora", 8, LORA),
    ("dora", 8, LORA),
    ("pissa", 8, [3e-5, 1e-4, 3e-4]),
    ("rslora", 8, LORA),
    ("lora_plus", 8, [5e-5, 1.5e-4, 5e-4]),
    ("lora_fa", 8, LORA),
    ("olora", 8, [3e-5, 1e-4, 3e-4]),
    # ablations
    ("eigentune_diag_minor", 16, EIGEN_DIAG),
    ("eigentune_core_minor", 8, EIGEN_CORE),
    ("eigentune_diag_relative", 16, EIGEN_DIAG),
]


def temp() -> int:
    try:
        return int(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], text=True
            ).split()[0]
        )
    except Exception:
        return 0


def cooldown():
    while temp() > MAX_TEMP:
        time.sleep(15)


def path(method, rank, lr, seed) -> Path:
    return OUT / f"{method}_r{rank}_lr{lr:g}_s{seed}.json"


def run_one(method, rank, lr, seed, steps=250):
    p = path(method, rank, lr, seed)
    if p.exists():
        return json.loads(p.read_text())
    cooldown()
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONPATH=str(HERE.parent))
    cmd = [
        PY,
        str(HERE / "quality.py"),
        "--method",
        method,
        "--rank",
        str(rank),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--out",
        str(p),
    ]
    t = time.time()
    try:
        subprocess.run(cmd, env=env, timeout=1200, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"FAILED {p.name}: {exc.stderr[-300:]}", flush=True)
        return None
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT {p.name}", flush=True)
        return None
    r = json.loads(p.read_text())
    print(
        f"{p.name}: eval={r.get('final_eval_loss')} diverged={r.get('diverged')} ({time.time() - t:.0f}s, "
        f"gpu {temp()}C)",
        flush=True,
    )
    return r


def best_lr(method, rank, grid):
    runs = [(path(method, rank, lr, 0), lr) for lr in grid if path(method, rank, lr, 0).exists()]
    scored = []
    for p, lr in runs:
        r = json.loads(p.read_text())
        if not r.get("diverged"):
            scored.append((r["final_eval_loss"], lr))
    return min(scored)[1] if scored else None


def cmd_sweep():
    for method, rank, grid in GRID:
        for lr in grid:
            run_one(method, rank, lr, 0)


def cmd_seeds():
    for method, rank, grid in GRID:
        lr = best_lr(method, rank, grid)
        if lr is None:
            continue
        for seed in (1, 2):
            run_one(method, rank, lr, seed)


def summarize():
    rows = []
    for method, rank, grid in GRID:
        lr = best_lr(method, rank, grid)
        if lr is None:
            continue
        runs = [
            json.loads(path(method, rank, lr, s).read_text()) for s in (0, 1, 2) if path(method, rank, lr, s).exists()
        ]
        runs = [r for r in runs if not r.get("diverged")]
        if not runs:
            continue
        losses = [r["final_eval_loss"] for r in runs]
        sweep = {
            lr_: json.loads(path(method, rank, lr_, 0).read_text())
            for lr_ in grid
            if path(method, rank, lr_, 0).exists()
        }
        rows.append(
            {
                "method": method,
                "rank": rank,
                "best_lr": lr,
                "seeds": len(runs),
                "eval_loss_mean": statistics.mean(losses),
                "eval_loss_std": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
                "trainable_parameters": runs[0]["trainable_parameters"],
                "adapter_bytes": runs[0]["adapter_bytes"],
                "adapter_bytes_with_bases": runs[0]["adapter_bytes_with_bases"],
                "runtime_basis_bytes": runs[0]["runtime_basis_bytes"],
                "peak_vram_mib": statistics.mean(r["peak_vram_mib"] for r in runs),
                "steps_per_second": statistics.mean(r["steps_per_second"] for r in runs),
                "init_seconds": statistics.mean(r["init_seconds"] for r in runs),
                "base_eval_loss_128": runs[0]["base_eval_loss_128"],
                "sweep": {
                    str(k): (v.get("final_eval_loss") if not v.get("diverged") else None) for k, v in sweep.items()
                },
                "curves": [r["curve"] for r in runs],
            }
        )
    return rows


def cmd_report():
    rows = summarize()
    (HERE / "results" / "quality_summary.json").write_text(json.dumps(rows, indent=2))
    rows.sort(key=lambda r: r["eval_loss_mean"])
    head = (
        "| method | rank | trainable params | adapter bytes | adapter + bases | runtime bases | eval loss (mean ± sd) | "
        "steps/s | peak VRAM MiB | best lr | seeds |\n|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['rank']} | {r['trainable_parameters']:,} | {r['adapter_bytes']:,} | "
            f"{r['adapter_bytes_with_bases']:,} | {r['runtime_basis_bytes']:,} | "
            f"{r['eval_loss_mean']:.4f} ± {r['eval_loss_std']:.4f} | {r['steps_per_second']:.2f} | "
            f"{r['peak_vram_mib']:.0f} | {r['best_lr']:g} | {r['seeds']} |"
        )
    text = "\n".join(lines)
    (HERE / "results" / "quality_summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    {"sweep": cmd_sweep, "seeds": cmd_seeds, "report": cmd_report}[sys.argv[1]]()
