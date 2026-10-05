import pytest
import torch
import torch.nn as nn

from eigentune import EigenTuneConfig, adapter_report, get_eigentune_model, load_adapter, save_adapter
from eigentune.model import iter_eigentune_layers
from eigentune.svd import allocate_ranks


def spectrum_model():
    """Two layers: one with a few dominant directions, one with a flat spectrum."""
    torch.manual_seed(0)
    peaked, flat = nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False)
    U = torch.linalg.qr(torch.randn(32, 32))[0]
    V = torch.linalg.qr(torch.randn(32, 32))[0]
    peaked.weight.data = (U * torch.tensor([10.0, 8.0, 6.0] + [0.1] * 29)) @ V.T
    flat.weight.data = (U * torch.ones(32)) @ V.T
    return nn.Sequential(peaked, flat)


def test_allocation_respects_the_budget_and_gives_each_layer_one():
    spectra = {"a": torch.tensor([5.0, 4.0, 3.0, 0.1]), "b": torch.tensor([1.0, 0.9, 0.8, 0.7])}
    energy = {"a": 50.0, "b": 3.0}
    ranks = allocate_ranks(spectra, energy, budget=6)
    assert sum(ranks.values()) == 6 and min(ranks.values()) >= 1
    assert allocate_ranks(spectra, energy, budget=1) == {"a": 1, "b": 1}  # never below one per layer


def test_energy_goes_to_the_layer_that_has_it():
    m = get_eigentune_model(spectrum_model(), EigenTuneConfig(rank=16, rank_budget=12, svd_backend="exact"))
    ranks = {n: layer.rank for n, layer in iter_eigentune_layers(m)}
    assert sum(ranks.values()) == 12
    assert ranks["0"] > ranks["1"] or ranks["1"] > ranks["0"]  # not uniform


def test_budget_model_is_identity_at_init_and_counts_parameters():
    base = spectrum_model()
    x = torch.randn(4, 32)
    ref = base(x)
    m = get_eigentune_model(base, EigenTuneConfig(rank=16, rank_budget=10))
    assert torch.equal(m(x), ref)
    assert adapter_report(m)["trainable_parameters"] == 10


def test_adaptive_ranks_survive_a_save_load_roundtrip(tmp_path):
    cfg = EigenTuneConfig(rank=16, rank_budget=10, svd_backend="exact")
    a = get_eigentune_model(spectrum_model(), cfg)
    with torch.no_grad():
        for _, layer in iter_eigentune_layers(a):
            layer.delta.normal_(std=0.1)
    save_adapter(a, tmp_path)
    b = load_adapter(spectrum_model(), tmp_path)
    x = torch.randn(3, 32)
    assert torch.equal(a(x), b(x))
    assert {n: layer.rank for n, layer in iter_eigentune_layers(a)} == {
        n: layer.rank for n, layer in iter_eigentune_layers(b)
    }


def test_rank_budget_requires_principal_selection():
    with pytest.raises(ValueError):
        EigenTuneConfig(rank_budget=8, selection="minor")
