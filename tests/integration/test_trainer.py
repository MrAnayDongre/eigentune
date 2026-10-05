import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("accelerate")
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments  # noqa: E402

from eigentune import EigenTuneConfig, get_eigentune_model, load_adapter  # noqa: E402
from eigentune.integrations.transformers import EigenTuneCallback  # noqa: E402


def tiny(seed=0):
    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )


class Data(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, i):
        ids = torch.randint(0, 64, (16,), generator=torch.Generator().manual_seed(i))
        return {"input_ids": ids, "labels": ids}


def test_trainer_trains_and_the_callback_writes_a_loadable_adapter(tmp_path):
    model = get_eigentune_model(tiny(), EigenTuneConfig(rank=4, target_modules=["q_proj", "v_proj"]))
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    args = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=4,
        per_device_train_batch_size=4,
        save_steps=4,
        learning_rate=1e-2,
        report_to=[],
        use_cpu=True,
        save_strategy="steps",
        logging_steps=100,
        save_only_model=True,
        seed=0,
    )
    Trainer(model=model, args=args, train_dataset=Data(), callbacks=[EigenTuneCallback()]).train()
    after = [p.detach() for p in model.parameters() if p.requires_grad]
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), "adapter parameters should have moved"
    adapter = tmp_path / "checkpoint-4" / "eigentune"
    assert (adapter / "adapter_model.safetensors").exists()
    restored = load_adapter(tiny(), str(adapter))
    ids = torch.randint(0, 64, (2, 16))
    assert torch.allclose(restored(input_ids=ids).logits, model(input_ids=ids).logits, atol=1e-5)
