# EigenTune: Surgical Fine-Tuning via Singular Value Scaling

[![PyPI version](https://badge.fury.io/py/eigentune.svg)](https://badge.fury.io/py/eigentune)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/MrAnayDongre/eigentune/blob/main/LICENSE)

EigenTune is a novel Parameter-Efficient Fine-Tuning (PEFT) method inspired by the mathematical properties of model weights. Instead of adding new matrices like LoRA, EigenTune identifies the most important "feature directions" in existing weight matrices (via SVD) and only fine-tunes their magnitudes.

This approach is highly parameter-efficient and aims to preserve pre-trained knowledge by re-calibrating existing features rather than introducing new ones.

## How it Works

1.  A target `nn.Linear` layer's weight matrix `W` is decomposed using SVD: `W = UΣVᵀ`.
2.  The orthogonal matrices `U` and `V` (representing feature directions) are frozen.
3.  A tiny, trainable vector `δ` of size `r` (rank) is introduced.
4.  The fine-tuned weight `W'` is implicitly represented as `W' = U(Σ + diag(δ))Vᵀ`.
5.  The forward pass is efficiently calculated as `y = Wx + (U_r diag(δ) Vh_r)x`, avoiding the formation of the full `W'` matrix.

## Installation

```bash
pip install eigentune