# examples/train_on_dolly.py
"""
An end-to-end example of fine-tuning a quantized model using the EigenTune library.
This script demonstrates the clean, high-level API.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import prepare_model_for_kbit_training

import sys
sys.path.append('.')
# Import the new, professional API
from eigentune import EigenTuneConfig, get_eigentune_model

def create_dolly_prompt(sample):
    """Formats a sample from the Dolly dataset into a standard instruction prompt."""
    template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
    if sample["context"]:
        template = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:"
        )
    return f"{template.format_map(sample)}{sample['response']}"

def main():
    # --- 1. Configuration ---
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_id = "databricks/databricks-dolly-15k"
    adapter_output_path = "./dolly-eigentuned-adapter"
    training_output_path = "./dolly-eigentuned-output"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- 2. Model Loading (Two-Load Method) ---
    print("Loading full-precision weights to CPU for SVD...")
    full_precision_state_dict = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).state_dict()

    print("\nLoading 4-bit quantized model to GPU for training...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 3. Applying EigenTune (The Clean API) ---
    model = prepare_model_for_kbit_training(model)

    print("\nApplying EigenTune to the model...")
    eigentune_config = EigenTuneConfig(
        rank=4,
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_eigentune_model(
        model,
        eigentune_config,
        full_precision_state_dict=full_precision_state_dict
    )
    
    print("\nEigenTune applied. Final trainable parameters:")
    model.print_trainable_parameters()

    # --- 4. Dataset and Training ---
    # (This section remains the same as before)
    print(f"\nLoading and preparing the {dataset_id} dataset...")
    dataset = load_dataset(dataset_id, split="train").shuffle().select(range(1000))
    text_data = [create_dolly_prompt(sample) + tokenizer.eos_token for sample in dataset]
    tokenized_data = tokenizer(text_data, truncation=True, padding="max_length", max_length=256)
    
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.input_ids = data['input_ids']
            self.attention_mask = data['attention_mask']
        def __len__(self): return len(self.input_ids)
        def __getitem__(self, i): return {'input_ids': self.input_ids[i], 'attention_mask': self.attention_mask[i]}
    
    final_dataset = DictDataset(tokenized_data)

    trainer = Trainer(
        model=model,
        train_dataset=final_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=TrainingArguments(
            output_dir=training_output_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            num_train_epochs=1,
            logging_steps=10,
            bf16=True,
            report_to="none",
            save_strategy="no",
        ),
    )

    print("\nStarting training...")
    trainer.train()
    print("Training finished!")

    # --- 5. Saving the Adapter ---
    model.save_pretrained(adapter_output_path)
    tokenizer.save_pretrained(adapter_output_path)
    print(f"\nEigenTune adapter saved to {adapter_output_path}")

if __name__ == "__main__":
    main()