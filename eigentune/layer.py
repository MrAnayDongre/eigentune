# Copyright 2024 MrAnayDongre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# eigentune/layer.py

import torch
import torch.nn as nn

class EigenTunedLayer(nn.Module):
    """
    An implementation of EigenTune designed to work with a quantized base layer.

    This layer replaces a standard linear or LoRA layer. It performs a forward
    pass through the frozen base layer and adds a low-rank update calculated
    by scaling the principal components of the original full-precision weight
    matrix. The SVD is performed on a provided full-precision weight for
    mathematical stability, while the forward pass uses the quantized layer for
    memory and compute efficiency.

    Args:
        original_layer (nn.Module): The base layer to be wrapped, typically a
            quantized linear layer (e.g., `bnb.nn.Linear4bit`).
        rank (int): The number of top singular values to fine-tune.
        full_precision_weight (torch.Tensor): The original, full-precision
            (e.g., bfloat16) weight tensor of the layer. This is used for the
            one-time SVD calculation.
    """
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int,
        full_precision_weight: torch.Tensor
    ):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.out_features, self.in_features = full_precision_weight.shape

        # Freeze the base layer; its weights will not be updated.
        self.original_layer.requires_grad_(False)

        # Perform SVD on the full-precision weight for mathematical stability.
        # The resulting U and Vh matrices are the frozen "skill" directions.
        W = full_precision_weight.to(torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Register U and Vh as non-trainable buffers.
        self.register_buffer('U', U)
        self.register_buffer('Vh', Vh)

        # The only trainable parameters: a vector of scalars to adjust the
        # top 'rank' singular values.
        self.delta_s = nn.Parameter(torch.zeros(self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass: y = W_quant(x) + ( (x @ Vh_r.T) * δ ) @ U_r.T
        """
        # 1. Compute the output of the frozen base layer.
        original_output = self.original_layer(x)

        # 2. Compute the EigenTune low-rank update.
        dtype = x.dtype
        device = x.device

        # Select the top 'r' singular vectors and move them to the correct device/dtype.
        U_r = self.U[:, :self.rank].to(device=device, dtype=dtype)
        Vh_r = self.Vh[:self.rank, :].to(device=device, dtype=dtype)

        # Flatten input if necessary for matrix multiplication.
        original_shape = x.shape
        if x.dim() > 2:
            x_flat = x.view(-1, original_shape[-1])
        else:
            x_flat = x

        # Project input onto principal directions, scale by our trainable deltas,
        # and project back into the output space.
        v_x = x_flat @ Vh_r.T
        scaled_v_x = v_x * self.delta_s.to(device=device, dtype=dtype)
        update = scaled_v_x @ U_r.T

        # Reshape update to match original output shape.
        if x.dim() > 2:
            update = update.view(*original_shape[:-1], self.out_features)

        return original_output + update

    def __repr__(self):
        return (
            f"EigenTunedLayer(rank={self.rank}, "
            f"original_layer={self.original_layer.__class__.__name__})"
        )