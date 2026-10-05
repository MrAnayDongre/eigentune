import pytest

from eigentune import EigenTuneConfig


def test_defaults_and_roundtrip():
    cfg = EigenTuneConfig(rank=4, target_modules=["q_proj"], method="spectral_core", core_bandwidth=2)
    assert EigenTuneConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "kw",
    [
        {"rank": 0},
        {"method": "x"},
        {"selection": "x"},
        {"backend": "cuda"},
        {"core_bandwidth": 1},
        {"method": "spectral_core", "core_bandwidth": -1},
    ],
)
def test_invalid_config_is_rejected(kw):
    with pytest.raises(ValueError):
        EigenTuneConfig(**kw)


def test_kind():
    assert EigenTuneConfig().kind == "diag"
    assert EigenTuneConfig(method="spectral_core").kind == "core"
    assert EigenTuneConfig(method="spectral_core", core_bandwidth=0).kind == "diag"


def test_unknown_keys_are_ignored_on_load():
    assert EigenTuneConfig.from_dict({"rank": 3, "future_option": 1}).rank == 3


def test_legacy_layer_still_works():
    import warnings

    import torch
    import torch.nn as nn

    from eigentune import EigenTunedLayer

    base = nn.Linear(16, 12, bias=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        layer = EigenTunedLayer(base, 4, base.weight.data)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    x = torch.randn(3, 16)
    assert torch.equal(layer(x), base(x))
    assert layer.delta_s is layer.delta and layer.original_layer is base
