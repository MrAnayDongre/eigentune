"""Orchestrate the quality comparison.

    python benchmarks/compare_peft.py search     # adaptive learning-rate search per method, seed 0
    python benchmarks/compare_peft.py seeds      # extra seeds at each config's best learning rate
    python benchmarks/compare_peft.py ablate     # one-seed ablations (selection, update rule)
    python benchmarks/compare_peft.py report     # markdown + JSON summary

The learning-rate search starts from a prior triple and keeps extending by 3x toward whichever edge holds the best
value, up to ``MAX_EVALS`` runs, so no method is judged on a grid that stops short of its optimum. Resumable; one
process at a time (``quality.py`` also runs a thermal governor: this machine's CPU and GPU share one cooling envelope).
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "results" / "quality"
PY = sys.executable
STEPS = 150
MAX_EVALS = 4

# (method, rank, starting learning rates)
CONFIGS = [
    ("eigentune_diag", 16, [1e-1, 3e-1, 9e-1]),
    ("eigentune_core", 8, [1e-2, 3e-2, 1e-1]),
    ("lora", 8, [3e-4, 1e-3, 3e-3]),
    ("lora", 1, [1e-3, 3e-3, 1e-2]),
    ("eigentune_diag", 64, [3e-2, 1e-1, 3e-1]),
    ("dora", 8, [3e-4, 1e-3, 3e-3]),
    ("pissa", 8, [1e-4, 3e-4, 1e-3]),
    ("rslora", 8, [1e-4, 3e-4, 1e-3]),
    ("lora_plus", 8, [1.5e-4, 5e-4, 1.5e-3]),
]
ABLATIONS = [  # (method, rank, the configuration whose best learning rate to scale from, multipliers)
    ("eigentune_diag_minor", 16, ("eigentune_diag", 16), [1.0]),
    ("eigentune_core_minor", 8, ("eigentune_core", 8), [1.0]),
    ("eigentune_diag_relative", 16, ("eigentune_diag", 16), [0.3, 0.1]),
]


def path(method, rank, lr, seed) -> Path:
    return OUT / f"{method}_r{rank}_lr{lr:g}_s{seed}.json"


def run_one(method, rank, lr, seed):
    p = path(method, rank, lr, seed)
    if not p.exists():
        env = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            PYTHONPATH=str(HERE.parent),
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",
        )
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
            str(STEPS),
            "--out",
            str(p),
        ]
        t = time.time()
        try:
            subprocess.run(cmd, env=env, timeout=3600, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            print(f"FAILED {p.name}: {exc.stderr[-300:]}", flush=True)
            return None
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT {p.name}", flush=True)
            return None
        print(f"{p.name}: done in {time.time() - t:.0f}s", flush=True)
    return json.loads(p.read_text())


def score(r):
    return math.inf if r is None or r.get("diverged") else r["final_eval_loss"]


def seed0_runs(method, rank):
    found = {}
    for f in glob.glob(str(OUT / f"{method}_r{rank}_lr*_s0.json")):
        r = json.loads(Path(f).read_text())
        found[r["lr"]] = score(r)
    return found


def best_lr(method, rank):
    runs = {lr: s for lr, s in seed0_runs(method, rank).items() if s < math.inf}
    return min(runs, key=runs.get) if runs else None


def search(method, rank, start):
    done = seed0_runs(method, rank)
    queue = [lr for lr in start if lr not in done]
    while True:
        while queue:
            lr = queue.pop(0)
            r = run_one(method, rank, lr, 0)
            done[lr] = score(r)
        if len(done) >= MAX_EVALS:
            return
        lrs = sorted(done)
        best = min(done, key=done.get)
        if done[best] == math.inf:
            queue = [lrs[0] / 3]  # everything diverged: go lower
        elif best == lrs[-1]:
            queue = [best * 3]
        elif best == lrs[0]:
            queue = [best / 3]
        else:
            return


def cmd_search():
    for method, rank, start in CONFIGS:
        search(method, rank, start)


def cmd_seeds():
    for method, rank, _ in CONFIGS:
        lr = best_lr(method, rank)
        if lr is not None:
            run_one(method, rank, lr, 1)


def cmd_ablate():
    for method, rank, (ref_m, ref_r), mults in ABLATIONS:
        ref = best_lr(ref_m, ref_r)
        if ref is None:
            continue
        for m in mults:
            run_one(method, rank, float(f"{ref * m:.3g}"), 0)


def summarize():
    rows = []
    for method, rank, _ in CONFIGS + [(a[0], a[1], None) for a in ABLATIONS]:
        lr = best_lr(method, rank)
        if lr is None:
            continue
        runs = [json.loads(path(method, rank, lr, s).read_text()) for s in (0, 1) if path(method, rank, lr, s).exists()]
        runs = [r for r in runs if not r.get("diverged")]
        losses = [r["final_eval_loss"] for r in runs]
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
                    f"{lr_:g}": (None if s == math.inf else s) for lr_, s in sorted(seed0_runs(method, rank).items())
                },
                "curves": [r["curve"] for r in runs],
            }
        )
    return rows


def cmd_report():
    rows = summarize()
    (HERE / "results" / "quality_summary.json").write_text(json.dumps(rows, indent=2))
    for r in sorted(rows, key=lambda r: r["eval_loss_mean"]):
        print(
            f"{r['method']:26s} r={r['rank']:<3d} params={r['trainable_parameters']:>9,}  eval={r['eval_loss_mean']:.4f} "
            f"+- {r['eval_loss_std']:.4f}  lr={r['best_lr']:g}  seeds={r['seeds']}"
        )


if __name__ == "__main__":
    {"search": cmd_search, "seeds": cmd_seeds, "ablate": cmd_ablate, "report": cmd_report}[sys.argv[1]]()
