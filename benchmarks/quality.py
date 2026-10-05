"""One fine-tuning run: a method, a rank, a learning rate and a seed, on GSM8K solutions with Qwen3-0.6B.

    python benchmarks/quality.py --method eigentune_diag --rank 16 --lr 1e-2 --seed 0 --out run.json

Everything that is not the method is fixed: model, data, split, sequence length, batch, steps, schedule, clipping.
Primary metric: held-out loss on the answer tokens of 500 GSM8K test problems.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import metadata  # noqa: E402
from _methods import ALL, build  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
SEQ = 192
MICRO, ACCUM = 4, 2
N_TRAIN, N_EVAL = 2000, 500


def load_data(tok):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")

    def enc(ex):
        q = f"Question: {ex['question']}\nAnswer:"
        a = " " + ex["answer"] + tok.eos_token
        qi, ai = tok(q, add_special_tokens=False)["input_ids"], tok(a, add_special_tokens=False)["input_ids"]
        ids = (qi + ai)[:SEQ]
        labels = ([-100] * len(qi) + ai)[:SEQ]
        return ids, labels

    train = [enc(ds["train"][i]) for i in range(len(ds["train"]))]
    test = [enc(ds["test"][i]) for i in range(N_EVAL)]
    return train, test


def collate(batch, pad):
    n = max(len(x[0]) for x in batch)
    ids = torch.tensor([x[0] + [pad] * (n - len(x[0])) for x in batch])
    lab = torch.tensor([x[1] + [-100] * (n - len(x[1])) for x in batch])
    att = torch.tensor([[1] * len(x[0]) + [0] * (n - len(x[0])) for x in batch])
    return ids.cuda(), lab.cuda(), att.cuda()


@torch.no_grad()
def evaluate(model, data, pad, limit=None):
    model.eval()
    tot, cnt = 0.0, 0
    data = data[:limit] if limit else data
    for i in range(0, len(data), 8):
        ids, lab, att = collate(data[i : i + 8], pad)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = lab[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="sum"
        )
        tot += loss.item()
        cnt += (tgt != -100).sum().item()
    model.train()
    return tot / cnt


def run(method, rank, lr, seed, steps):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_per_process_memory_fraction(0.8)  # leave headroom: this GPU also drives the display
    random.seed(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train, test = load_data(tok)
    rng = random.Random(seed)
    order = rng.sample(range(len(train)), N_TRAIN)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    base_eval = evaluate(model, test, pad, 128)
    model, info, make_opt = build(method, model, rank)
    model.train()
    opt = make_opt(lr)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    )
    torch.cuda.reset_peak_memory_stats()
    curve, losses, tokens, t_train = [(0, evaluate(model, test, pad, 128))], [], 0, 0.0
    cursor = 0
    for step in range(1, steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ACCUM):
            batch = [train[order[(cursor + j) % N_TRAIN]] for j in range(MICRO)]
            cursor += MICRO
            ids, lab, att = collate(batch, pad)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            (out.loss / ACCUM).backward()
            tokens += int(att.sum())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        losses.append(out.loss.item())
        if not math.isfinite(losses[-1]):
            return {**info, "lr": lr, "seed": seed, "diverged": True, "steps": step}
        if step % 25 == 0:
            curve.append((step, evaluate(model, test, pad, 128)))
    final = evaluate(model, test, pad)
    return {
        **info,
        "lr": lr,
        "seed": seed,
        "steps": steps,
        "diverged": False,
        "base_eval_loss_128": base_eval,
        "final_eval_loss": final,
        "curve": curve,
        "train_loss_last10": sum(losses[-10:]) / 10,
        "steps_per_second": steps / t_train,
        "tokens_per_second": tokens / t_train,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=ALL)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(a.method, a.rank, a.lr, a.seed, a.steps)
    res["meta"] = metadata()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("curve", "meta")}))


if __name__ == "__main__":
    main()
