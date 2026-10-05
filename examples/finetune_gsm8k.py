"""Fine-tune Qwen3-0.6B on GSM8K solutions with EigenTune and the Hugging Face Trainer (needs one GPU, ~4 GB).

    python examples/finetune_gsm8k.py --method diagonal --rank 16 --lr 1e-2 --steps 100

Writes an adapter next to every checkpoint (``<output>/checkpoint-N/eigentune``); load it with
``eigentune.load_adapter(base_model, path)``.
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from eigentune import EigenTuneConfig, get_eigentune_model, print_trainable_parameters
from eigentune.integrations.transformers import EigenTuneCallback

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--method", default="diagonal", choices=["diagonal", "spectral_core"])
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--output", default="gsm8k-eigentune")
a = ap.parse_args()

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
model = get_eigentune_model(model, EigenTuneConfig(rank=a.rank, method=a.method, target_modules=TARGETS))
print_trainable_parameters(model)


def encode(ex):
    q = tok(f"Question: {ex['question']}\nAnswer:", add_special_tokens=False)["input_ids"]
    ans = tok(" " + ex["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (q + ans)[:192]
    return {"input_ids": ids, "labels": ([-100] * len(q) + ans)[:192]}


data = load_dataset("openai/gsm8k", "main", split="train").map(encode, remove_columns=["question", "answer"])


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch])
    lab = torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch])
    mask = torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch])
    return {"input_ids": ids, "labels": lab, "attention_mask": mask}


args = TrainingArguments(
    output_dir=a.output,
    max_steps=a.steps,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=a.lr,
    lr_scheduler_type="cosine",
    warmup_steps=max(1, a.steps // 10),
    save_steps=a.steps,
    logging_steps=10,
    bf16=True,
    report_to=[],
    save_only_model=True,
    remove_unused_columns=False,
)
Trainer(model=model, args=args, train_dataset=data, data_collator=collate, callbacks=[EigenTuneCallback()]).train()
