# examples/inference.py
"""
An example of how to load a pre-trained EigenTune adapter for inference.

This script demonstrates loading the 4-bit base model and applying the
trained EigenTune adapter on top for efficient inference.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig

import sys
sys.path.append('.')
from eigentune import replace_peft_with_eigentune

def main():
    # --- 1. Configuration ---
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "./dolly-eigentuned-adapter"  # Path to your trained adapter

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- 2. Model Loading ---
    print("Loading 4-bit quantized model to GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # --- 3. Reconstructing the EigenTune Model Architecture ---
    # To load the adapter, we must first reconstruct the model's custom architecture.
    print("\nCreating PEFT scaffolding...")
    # The LoraConfig is loaded from the adapter itself to ensure consistency.
    lora_config = LoraConfig.from_pretrained(adapter_path)
    model = PeftModel(model, lora_config)

    print("\nReplacing LoRA layers with EigenTune layers...")
    # We need the full state dict again to correctly initialize the layers' SVD components.
    full_precision_state_dict = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).state_dict()
    model = replace_peft_with_eigentune(
        model,
        rank=lora_config.r,  # Use rank from the saved config
        full_precision_state_dict=full_precision_state_dict
    )

    # --- 4. Loading the Trained Adapter Weights ---
    # The `load_adapter` method is part of the PeftModel wrapper.
    model.load_adapter(adapter_path, "default")
    print(f"\nSuccessfully loaded EigenTune adapter from {adapter_path}")

    # --- 5. Inference ---
    model.eval()
    print("\nModel ready for inference!")

    test_instruction = "List three advantages of using solar power."
    prompt_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
    prompt = prompt_template.format(instruction=test_instruction)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\n--- Generating Response ---")
    print(f"Prompt:\n{prompt}")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean up the response to only show the generated part
    response = response.split("### Response:")[1].strip()
    
    print(f"\nModel Response:\n{response}")

if __name__ == "__main__":
    main()