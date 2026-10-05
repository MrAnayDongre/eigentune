"""Hugging Face ``Trainer`` integration.

An EigenTune model is an ordinary ``nn.Module``, so ``Trainer`` trains it as is. What the Trainer's own checkpoints
do not give you is an adapter-only artifact; this callback writes one next to every checkpoint.

    trainer = Trainer(model=model, args=args, train_dataset=ds, callbacks=[EigenTuneCallback()])
"""

from __future__ import annotations

import os

from transformers import TrainerCallback

from ..model import iter_eigentune_layers
from ..serialization import save_adapter


class EigenTuneCallback(TrainerCallback):
    """Save the EigenTune adapter (``<checkpoint>/eigentune``) whenever the Trainer saves a checkpoint."""

    def __init__(self, save_bases: bool = False):
        self.save_bases = save_bases

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is not None and any(True for _ in iter_eigentune_layers(model)):
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", "eigentune")
            save_adapter(model, ckpt, save_bases=self.save_bases)
        return control
