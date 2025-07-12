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

# eigentune/model.py

import torch.nn as nn
from peft import get_peft_model, LoraConfig
from peft.tuners.lora import LoraLayer

from .config import EigenTuneConfig
from .layer import EigenTunedLayer

def get_eigentune_model(
    model: nn.Module,
    config: EigenTuneConfig,
    full_precision_state_dict: dict,
) -> nn.Module:
    """
    Returns a model prepared for EigenTune fine-tuning.

    This function encapsulates the "hijack" method. It first creates a
    standard PEFT model with LoRA layers as scaffolding, then replaces those
    LoRA layers with EigenTunedLayer instances.

    Args:
        model (nn.Module): The base model to be fine-tuned (typically quantized).
        config (EigenTuneConfig): The configuration for EigenTune.
        full_precision_state_dict (dict): A state dictionary of the original,
            full-precision model, used to perform the SVD.

    Returns:
        nn.Module: A PeftModel ready for EigenTune training.
    """
    # Step 1: Create a temporary LoRA model as scaffolding.
    # This ensures compatibility with the PEFT ecosystem and Trainer.
    lora_config = LoraConfig(
        r=config.rank,  # Use rank from config, it will be used by our layer
        target_modules=config.target_modules,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    # Step 2: Replace the LoRA layers with our custom EigenTunedLayer.
    layers_to_replace = []
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer):
            layers_to_replace.append(name)

    for name in layers_to_replace:
        parent_name = ".".join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent_module = peft_model.get_submodule(parent_name)

        lora_layer = peft_model.get_submodule(name)
        base_layer = lora_layer.get_base_layer()

        original_key = name.replace('base_model.model.', '') + '.weight'
        if original_key not in full_precision_state_dict:
            print(f"Warning: Could not find key {original_key} in state_dict. Skipping layer {name}.")
            continue

        full_weight = full_precision_state_dict[original_key]

        new_layer = EigenTunedLayer(
            base_layer,
            rank=config.rank,
            full_precision_weight=full_weight
        )
        setattr(parent_module, child_name, new_layer)

    return peft_model