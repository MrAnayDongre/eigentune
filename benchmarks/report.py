"""Render docs/benchmarks.md from benchmarks/results/*.json (the numbers are never typed by hand).

python benchmarks/report.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"
OUT = HERE.parent / "docs" / "benchmarks.md"


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def kernel_table(files, backend, tokens, phase, mode):
    """Speedup over torch (min - max over layer shapes and ranks) per token count, plus win counts."""
    by = collections.defaultdict(dict)
    for f in files:
        d = load(f)
        if d:
            for r in d["rows"]:
                by[(f, r["in"], r["out"], r["rank"], r["tokens"])][r["backend"]] = r
    out = []
    for n in tokens:
        ratios, wins, total = [], 0, 0
        for key, v in by.items():
            if key[4] != n or backend not in v or "torch" not in v:
                continue
            t, b = v["torch"][f"{phase}_{mode}"]["median_us"], v[backend][f"{phase}_{mode}"]["median_us"]
            ratios.append(t / b)
            wins += b < t
            total += 1
        out.append((n, ratios, wins, total))
    return out


def fmt_kernels(files, backend, tokens, title):
    lines = [
        f"**{title}**",
        "",
        "| tokens | forward eager | forward graph | backward eager | backward graph |",
        "|---|---|---|---|---|",
    ]
    cols = {
        (p, m): {n: (r, w, t) for n, r, w, t in kernel_table(files, backend, tokens, p, m)}
        for p in ("fwd", "bwd")
        for m in ("eager", "graph")
    }
    for n in tokens:
        cells = []
        for p in ("fwd", "bwd"):
            for m in ("eager", "graph"):
                r, w, t = cols[(p, m)][n]
                cells.append(f"{min(r):.2f} - {max(r):.2f}x (wins {w}/{t})" if r else "n/a")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section_env():
    d = load("kernels_bf16") or {}
    m = d.get("meta", {})
    return (
        "## Environment\n\n"
        f"GPU: {m.get('gpu')}; PyTorch {m.get('torch')}; Triton {m.get('triton')}; CUDA {m.get('cuda')}; "
        f"Python {m.get('python')}; EigenTune {m.get('eigentune')} at `{m.get('git_sha')}`.\n\n"
        "One laptop GPU that also drives the display, run one process at a time under a thermal governor. "
        "Every table below is regenerated from `benchmarks/results/*.json` by `benchmarks/report.py`.\n"
    )


def section_kernels():
    files = ["kernels_bf16", "kernels_bf16_n1024"]
    parts = [
        "## Kernels\n",
        "Speedup over the `torch` backend (> 1 is faster), as a range over layer shapes "
        "(2048x2048 ... 8192x8192, 4096x11008) and ranks (8, 16, 64); *wins* counts shapes where the backend was faster. "
        "*Eager* includes Python/launch overhead (what an eager user sees); *graph* is the same calls replayed from a CUDA graph "
        "(GPU time only). bf16, diagonal. `benchmarks/kernels.py`.\n",
        fmt_kernels(files, "native", [1, 4, 16, 64, 128], "Native CUDA vs torch"),
        "",
        fmt_kernels(files, "triton", [1, 4, 16, 64, 128, 512, 1024, 2048], "Triton vs torch"),
        "",
    ]
    if load("kernels_fp16"):
        parts += [
            fmt_kernels(["kernels_fp16"], "native", [1, 4, 16], "Native CUDA vs torch, fp16"),
            "",
            fmt_kernels(["kernels_fp16"], "triton", [512, 2048], "Triton vs torch, fp16"),
            "",
        ]
    if load("kernels_bf16_core"):
        parts += [
            fmt_kernels(["kernels_bf16_core"], "native", [1, 16], "Native CUDA vs torch, core method, bf16"),
            "",
            fmt_kernels(["kernels_bf16_core"], "triton", [1, 16, 512, 2048], "Triton vs torch, core method, bf16"),
            "",
        ]
    parts.append(
        "Where a column shows a range that dips below 1, that backend is slower than the PyTorch path for some shapes "
        "in that regime; the dispatch policy in `eigentune/kernels/policy.py` only selects a backend where it won "
        "across the measured shapes.\n"
    )
    return "\n".join(parts)


def section_init():
    d = load("initialization")
    if not d:
        return ""
    lines = [
        "## Basis initialization\n",
        "Real weights from Llama-2-7B layer 16, fetched tensor by tensor. Error is `||W_r(approx) - W_r(exact)||_F / "
        "||W_r(exact)||_F` against the CPU exact SVD. `benchmarks/initialization.py`.\n",
        "| weight | rank | device | strategy | seconds | peak GPU MiB | reconstruction error | singular-value error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in d["rows"]:
        peak = "-" if r["peak_mib"] is None else f"{r['peak_mib']:.0f}"
        lines.append(
            f"| {r['weight']} | {r['rank']} | {r['device']} | {r['strategy']} | {r['seconds']:.3f} | {peak} | "
            f"{r['recon_rel_error']:.1e} | {r['sv_rel_error']:.1e} |"
        )
    return "\n".join(lines) + "\n"


def section_memory():
    d = load("memory")
    if not d:
        return ""
    kinds = ["frozen", "eigentune_diag", "eigentune_core", "lora", "lora_bf16", "dora"]
    lines = [
        "## Activation memory\n",
        "MiB of activations autograd keeps for the backward pass, per adapted 4096x4096 layer in bf16 (adapter and base; "
        "device-independent, counted with `saved_tensors_hooks`). `benchmarks/memory.py`.\n",
        "| tokens | rank | input X | " + " | ".join(kinds) + " |",
        "|---|---|---|" + "---|" * len(kinds),
    ]
    for r in d["rows"]:
        lines.append(
            f"| {r['tokens']} | {r['rank']} | {r['input_mib']:.0f} | " + " | ".join(f"{r[k]:.3f}" for k in kinds) + " |"
        )
    lines.append(
        "\n`lora` is PEFT's default (adapters in fp32, so the input is cast and a copy kept); `lora_bf16` keeps adapters "
        "in the base dtype. Plain autograd with frozen bases also keeps only `Q` (measured: 0.250 MiB at 8192 tokens, rank 16), "
        "so the custom autograd function is not what saves this memory.\n"
    )
    return "\n".join(lines)


def section_quality():
    rows = load("quality_summary")
    if not rows:
        return ""
    base = rows[0]["base_eval_loss_128"]
    rows = sorted(rows, key=lambda r: r["eval_loss_mean"])
    lines = [
        "## Quality against PEFT baselines\n",
        "Qwen3-0.6B (bf16) fine-tuned on 2,000 GSM8K training solutions for 150 optimizer steps (effective batch 8, sequence 192, "
        "cosine schedule), seven attention and MLP projections per layer adapted. Metric: cross-entropy on the answer tokens of "
        "500 held-out GSM8K test problems (lower is better). The pretrained model scores "
        f"{base:.3f} on the first 128 of those. Learning rate swept per method on seed 0 (three values), then the best rate rerun "
        "on seeds 1 and 2; ablations are single-seed. `benchmarks/compare_peft.py`.\n",
        "| method | rank | trainable params | adapter file (B) | adapter + bases (B) | runtime bases (B) | eval loss | steps/s | peak VRAM (MiB) | init (s) | lr | seeds |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['rank']} | {r['trainable_parameters']:,} | {r['adapter_bytes']:,} | "
            f"{r['adapter_bytes_with_bases']:,} | {r['runtime_basis_bytes']:,} | "
            f"{r['eval_loss_mean']:.4f} +- {r['eval_loss_std']:.4f} | {r['steps_per_second']:.2f} | "
            f"{r['peak_vram_mib']:.0f} | {r['init_seconds']:.1f} | {r['best_lr']:g} | {r['seeds']} |"
        )
    lines += [
        "",
        "### Learning-rate sweep (seed 0, final eval loss)",
        "",
        "| method | rank | learning rates tried (final eval loss) |",
        "|---|---|---|",
    ]
    for r in rows:
        cells = [f"{lr}: {('diverged' if v is None else f'{v:.4f}')}" for lr, v in r["sweep"].items()]
        lines.append(f"| {r['method']} | {r['rank']} | " + "; ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def section_stack():
    d = load("memory_stack")
    if not d:
        return ""
    kinds = ["frozen", "eigentune_diag", "eigentune_core", "lora", "lora_bf16", "dora"]
    lines = [
        "## Whole-model activation memory\n",
        "Peak GPU memory above the resident weights for one forward and backward through a 4-layer Qwen3-shaped stack "
        "(1024 hidden, 3072 MLP), bf16, rank 16, all seven projections adapted, cap 6 GiB. `frozen` backpropagates through the "
        "unadapted layers only. `benchmarks/memory.py --model`.\n",
        "| tokens | " + " | ".join(kinds) + " |",
        "|---|" + "---|" * len(kinds),
    ]
    for r in d["rows"]:
        lines.append(f"| {r['tokens']} | " + " | ".join("OOM" if r[k] is None else f"{r[k]:.0f}" for k in kinds) + " |")
    return "\n".join(lines) + "\n"


def section_training():
    d = load("training")
    if not d:
        return ""
    lines = [
        "## Training-step latency by backend\n",
        "One forward and backward through the same 4-layer stack, rank 16, median of 5 repetitions; the figure in brackets is the speedup over the "
        "`torch` backend. Differences under about 20% at small token counts are within run-to-run noise (launch-bound eager execution). "
        "`benchmarks/training.py`.\n",
        "| method | tokens | torch | triton | native | auto |",
        "|---|---|---|---|---|---|",
    ]
    for r in d["rows"]:
        base = r["torch"]["median_us"]
        cells = [
            f"{r[b]['median_us'] / 1000:.2f} ms (x{base / r[b]['median_us']:.2f})"
            for b in ("torch", "triton", "native", "auto")
        ]
        lines.append(f"| {r['method']} | {r['tokens']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    findings = (HERE / "findings.md").read_text() if (HERE / "findings.md").exists() else ""
    text = (
        "# Benchmarks\n\n"
        + findings
        + "\n"
        + "\n".join(
            s
            for s in (
                section_env(),
                section_kernels(),
                section_init(),
                section_memory(),
                section_stack(),
                section_quality(),
                section_training(),
            )
            if s
        )
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print("wrote", OUT, f"({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
