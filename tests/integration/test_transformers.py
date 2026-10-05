import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from eigentune import EigenTuneConfig, adapter_report, get_eigentune_model, load_adapter, save_adapter  # noqa: E402
from eigentune.model import iter_eigentune_layers  # noqa: E402


def tiny_llama(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(cfg)


def batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 128, (4, 16), generator=g)
    return {"input_ids": ids, "labels": ids}


def test_targets_and_freezing():
    m = get_eigentune_model(tiny_llama(), EigenTuneConfig(rank=4, target_modules=["q_proj", "v_proj"]))
    assert len(list(iter_eigentune_layers(m))) == 4
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(".delta") for n in trainable)
    assert adapter_report(m)["trainable_parameters"] == 4 * 4


def test_untouched_model_output_is_unchanged():
    base = tiny_llama()
    x = batch()
    ref = base(**x).loss.item()
    m = get_eigentune_model(base, EigenTuneConfig(rank=4))
    assert m(**x).loss.item() == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("method", ["diagonal", "spectral_core"])
def test_training_reduces_loss_and_only_adapters_move(method):
    m = get_eigentune_model(tiny_llama(), EigenTuneConfig(rank=8, method=method))
    before = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x = batch()
    first = m(**x).loss.item()
    for _ in range(25):
        opt.zero_grad()
        m(**x).loss.backward()
        opt.step()
    assert m(**x).loss.item() < first
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters() if n in before)


def test_gradient_accumulation_equals_one_big_batch():
    cfg = EigenTuneConfig(rank=4)
    x = batch()
    a = get_eigentune_model(tiny_llama(), cfg)
    a(**x).loss.backward()
    b = get_eigentune_model(tiny_llama(), cfg)
    for half in (slice(0, 2), slice(2, 4)):
        (b(input_ids=x["input_ids"][half], labels=x["labels"][half]).loss / 2).backward()
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            assert torch.allclose(p.grad, q.grad, atol=1e-5), n


def test_gradient_checkpointing_gives_the_same_gradients():
    cfg = EigenTuneConfig(rank=4)
    x = batch()
    a = get_eigentune_model(tiny_llama(), cfg)
    a(**x).loss.backward()
    b = get_eigentune_model(tiny_llama(), cfg)
    b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    b.enable_input_require_grads()
    b.train()
    b(**x).loss.backward()
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            assert torch.allclose(p.grad, q.grad, atol=1e-5), n


def test_autocast_bf16_runs_and_matches_fp32_roughly():
    m = get_eigentune_model(tiny_llama(), EigenTuneConfig(rank=4))
    x = batch()
    ref = m(**x).loss.item()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = m(**x).loss
    got.backward()
    assert got.item() == pytest.approx(ref, abs=0.1)


def test_save_load_and_merge_for_a_real_model(tmp_path):
    cfg = EigenTuneConfig(rank=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    a = get_eigentune_model(tiny_llama(), cfg)
    with torch.no_grad():
        for _, layer in iter_eigentune_layers(a):
            layer.delta.normal_(std=0.05)
    save_adapter(a, tmp_path)
    b = load_adapter(tiny_llama(), tmp_path)
    x = batch()
    assert torch.equal(a(**x).logits, b(**x).logits)
    from eigentune import merge_adapter

    out = b(**x).logits
    merge_adapter(b)
    assert torch.allclose(b(**x).logits, out, atol=1e-4)


def test_tied_embeddings_are_not_adapted_by_default():
    m = get_eigentune_model(tiny_llama(), EigenTuneConfig(rank=4))
    names = [n for n, _ in iter_eigentune_layers(m)]
    assert not any("lm_head" in n for n in names)
