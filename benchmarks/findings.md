Everything here was measured on one laptop (RTX 5070 Laptop GPU, 8 GB), one small model (Qwen3-0.6B) and one task
(GSM8K solutions). It does not show how EigenTune behaves on larger models, other tasks or other hardware. The tables below the
summary are regenerated from the stored results and are the source of truth.

## Summary

**Where EigenTune is ahead, measured**

* **Trainable parameters.** 3,136 to 12,544, against 630,784 (LoRA rank 1) to 5,046,272 (LoRA rank 8): 50x to 1,600x fewer.
* **Adapter file size, if the bases are rebuilt on load.** 59,587 to 97,483 bytes against 2.6 MB (LoRA rank 1) to 20.2 MB
  (LoRA rank 8). This is the default `save_adapter`.
* **Activation memory.** In a 4-layer Qwen3-shaped stack at 8,192 tokens, EigenTune's peak is 1,422 MiB, the same as a frozen
  model that only backpropagates (1,463 MiB); LoRA with PEFT's default fp32 adapters adds 1,118 MiB, LoRA with bf16 adapters 279 MiB, DoRA
  3,097 MiB (and DoRA runs out of memory at 12,288 tokens under the 6 GiB cap).
* **Training speed.** The same steps per second as LoRA (3.3 - 3.4 against 3.4 - 3.5); DoRA is about 1.7x slower.

**Where EigenTune is behind, measured**

* **Quality.** Eval loss 0.612 (best EigenTune, diagonal rank 64, 12,544 parameters) against 0.534 for LoRA rank 8. LoRA rank 1, with
  630,784 parameters, reaches 0.547, ahead of every EigenTune configuration. The pretrained model scores 1.388 on the first 128 of the 500 evaluation problems.
* **The `r x r` core does not beat the diagonal per parameter.** At the same 12,544 trainable parameters, diagonal rank 64 gets 0.612 and the
  core at rank 8 gets 0.629. The core reaches that with 8x fewer basis bytes (10.3 MB against 80.9 MB with bases embedded), so it is a
  storage trade-off, not a quality win.
* **Minor directions are not better than principal ones** here (diagonal rank 16: 0.684 against 0.664, one seed), and the relative update
  makes no difference (0.666 against 0.664).
* **Storage with the bases.** The frozen bases are not free: 20 to 81 MB at runtime, and the same if embedded in the file for exact
  portability, which is as large as or larger than the LoRA adapter (20.2 MB).
* **Initialization.** 4.5 to 14.5 s for the SVDs of 196 layers, against 0.6 s for LoRA and 1.4 s for PiSSA.
* **Whole-model peak memory in the training runs** is the same as LoRA (3.54 - 3.61 GB against 3.59 GB), because at that batch size the adapter is not
  what dominates memory.
* **Kernels do not change a training step much.** The kernel-level speedups below are real (up to 5.2x for a 4-token forward) but inside a
  transformer step the adapter is a small part of the time: the accelerated backends are within 0 - 5% of the PyTorch path at 3,072 and
  8,192 tokens, and within noise at small token counts (`benchmarks/results/training.json`).

**State of the art?** In none of the dimensions measured except trainable-parameter count, adapter file size (with bases rebuilt)
and adapter activation memory. On downstream quality EigenTune is clearly behind LoRA, DoRA, rsLoRA, LoRA+ and PiSSA on this task.

## How the quality comparison was run

Qwen3-0.6B (bf16), seven attention and MLP projections per layer adapted, 2,000 GSM8K training solutions, 150 optimizer steps,
effective batch 8 (micro-batch 2 with 4 accumulation steps), sequence length 192, cosine schedule, AdamW, no weight decay. The metric is
cross-entropy on the answer tokens of 500 held-out GSM8K test problems. For every method the learning rate was searched on seed 0 starting from a
three-value prior and extended by 3x toward whichever edge held the best value (at most four runs), so no method was judged on a grid that stops short
of its optimum; the best rate was then rerun on a second seed. **Two seeds** only: the spread between them is about 0.001, which
shows the run-to-run noise from data order and initialization is small, not that the ranking is established beyond this setup.
Ablations (minor selection, relative update) are single-seed. LoRA-FA and OLoRA are supported by the harness but were not run, to fit the thermal
budget of the machine; EVA, CorDA and MiCA were not run.
