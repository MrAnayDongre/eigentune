"""Load a trained EigenTune adapter into a fresh base model, merge it, and generate.

python examples/inference.py gsm8k-eigentune/checkpoint-100/eigentune "Question: ..."
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eigentune import load_adapter, merge_adapter

adapter, prompt = sys.argv[1], sys.argv[2]
model_id = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda()
load_adapter(model, adapter)  # verifies the adapter was trained on exactly these base weights
merge_adapter(model)  # no adapter overhead at inference time
ids = tok(prompt, return_tensors="pt").to("cuda")
print(tok.decode(model.generate(**ids, max_new_tokens=128)[0], skip_special_tokens=True))
