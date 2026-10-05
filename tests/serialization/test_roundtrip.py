import json

import pytest
import torch
import torch.nn as nn

from eigentune import (
    AdapterMismatchError,
    EigenTuneConfig,
    adapter_report,
    get_eigentune_model,
    load_adapter,
    save_adapter,
)
from eigentune.model import iter_eigentune_layers


def tiny(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))


def trained(cfg, seed=0):
    m = get_eigentune_model(tiny(seed), cfg)
    with torch.no_grad():
        for _, layer in iter_eigentune_layers(m):
            layer.trainable().normal_(std=0.1)
    return m


@pytest.mark.parametrize(
    "cfg",
    [
        EigenTuneConfig(rank=4),
        EigenTuneConfig(rank=4, method="spectral_core"),
        EigenTuneConfig(rank=4, method="spectral_core", core_bandwidth=1, selection="mixed", update="relative"),
    ],
)
@pytest.mark.parametrize("save_bases", [False, True])
def test_roundtrip_reproduces_outputs(tmp_path, cfg, save_bases):
    a = trained(cfg)
    save_adapter(a, tmp_path, save_bases=save_bases)
    b = load_adapter(tiny(), tmp_path)
    x = torch.randn(5, 16)
    assert torch.equal(a(x), b(x))


def test_default_adapter_file_is_only_the_trainables(tmp_path):
    a = trained(EigenTuneConfig(rank=4))
    save_adapter(a, tmp_path)
    from safetensors.torch import load_file

    keys = set(load_file(tmp_path / "adapter_model.safetensors"))
    assert keys == {"0.delta", "2.delta"}
    r = adapter_report(a)
    assert r["trainable_parameters"] == 8 and r["adapter_bytes"] == 32
    assert r["runtime_basis_bytes"] > r["adapter_bytes"]


def test_state_dict_has_no_bases():
    m = get_eigentune_model(tiny(), EigenTuneConfig(rank=4))
    assert all(k.endswith(".delta") or ".base." in k for k in m.state_dict())


def test_wrong_base_weights_are_refused(tmp_path):
    save_adapter(trained(EigenTuneConfig(rank=4)), tmp_path)
    with pytest.raises(AdapterMismatchError, match="different base weights"):
        load_adapter(tiny(seed=1), tmp_path)


def test_different_layers_are_refused(tmp_path):
    save_adapter(trained(EigenTuneConfig(rank=4)), tmp_path)
    with pytest.raises(AdapterMismatchError):
        load_adapter(nn.Sequential(nn.Linear(16, 32)), tmp_path)


@pytest.mark.parametrize("payload", ["not json", "{}", '{"format_version": 99, "config": {}, "layers": {}}'])
def test_corrupt_or_future_config_is_a_clear_error(tmp_path, payload):
    save_adapter(trained(EigenTuneConfig(rank=4)), tmp_path)
    (tmp_path / "eigentune_config.json").write_text(payload)
    with pytest.raises(ValueError):
        load_adapter(tiny(), tmp_path)


def test_signature_catches_a_changed_basis(tmp_path):
    save_adapter(trained(EigenTuneConfig(rank=4)), tmp_path)
    header = json.loads((tmp_path / "eigentune_config.json").read_text())
    for meta in header["layers"].values():
        meta["signature"] = [v + 1.0 for v in meta["signature"]]
    (tmp_path / "eigentune_config.json").write_text(json.dumps(header))
    with pytest.raises(AdapterMismatchError, match="save_bases=True"):
        load_adapter(tiny(), tmp_path)


def test_format_version_and_package_version_are_recorded(tmp_path):
    save_adapter(trained(EigenTuneConfig(rank=4)), tmp_path)
    header = json.loads((tmp_path / "eigentune_config.json").read_text())
    assert header["format_version"] == 1 and header["eigentune_version"]
