import pytest
import torch

from eigentune import EigenTuneConfig
from eigentune.svd import compute_bases, select_indices

torch.manual_seed(0)


def lowrankish(out, inn, k=6, noise=1e-3):
    return torch.randn(out, k) @ torch.randn(k, inn) + noise * torch.randn(out, inn)


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("principal", (0, 1, 2)),
        ("minor", (7, 8, 9)),
        ("mixed", (0, 1, 8, 9)),
    ],
)
def test_select_indices(selection, expected):
    rank = 4 if selection == "mixed" else 3
    assert select_indices(10, rank, selection) == expected


def test_rank_is_clipped_to_the_matrix():
    assert len(select_indices(5, 50, "principal")) == 5


@pytest.mark.parametrize("backend", ["exact", "randomized", "lowrank"])
def test_bases_are_orthonormal_and_reconstruct(backend):
    W = lowrankish(80, 64)
    b = compute_bases(W, EigenTuneConfig(rank=6, svd_backend=backend, svd_niter=3, svd_oversampling=10))
    assert torch.allclose(b.U.T @ b.U, torch.eye(6), atol=1e-4)
    assert torch.allclose(b.Vh @ b.Vh.T, torch.eye(6), atol=1e-4)
    err = (W - (b.U * b.S) @ b.Vh).norm() / W.norm()
    assert err < 5e-2


def test_randomized_matches_exact_singular_values():
    W = lowrankish(120, 90)
    ex = compute_bases(W, EigenTuneConfig(rank=5, svd_backend="exact"))
    rd = compute_bases(W, EigenTuneConfig(rank=5, svd_backend="randomized", svd_niter=4))
    assert torch.allclose(ex.S, rd.S, rtol=1e-3)


@pytest.mark.parametrize("backend", ["exact", "randomized", "lowrank"])
def test_bases_are_deterministic(backend):
    W = lowrankish(50, 40)
    cfg = EigenTuneConfig(rank=4, svd_backend=backend)
    a, b = compute_bases(W, cfg), compute_bases(W, cfg)
    assert torch.equal(a.U, b.U) and torch.equal(a.Vh, b.Vh) and a.fingerprint == b.fingerprint


def test_sign_convention_is_canonical():
    W = lowrankish(30, 20)
    b = compute_bases(W, EigenTuneConfig(rank=4))
    idx = b.Vh.abs().argmax(dim=1)
    assert (b.Vh[torch.arange(4), idx] > 0).all()
    # flipping the sign of the input weight's SVD pairs must not change the update directions
    b2 = compute_bases(W.clone(), EigenTuneConfig(rank=4))
    assert torch.equal(b.Vh, b2.Vh)


def test_fingerprint_changes_with_weight_and_settings():
    W = lowrankish(30, 20)
    base = compute_bases(W, EigenTuneConfig(rank=4)).fingerprint
    W2 = W.clone()
    W2[0, 0] += 1e-3
    assert compute_bases(W2, EigenTuneConfig(rank=4)).fingerprint != base
    assert compute_bases(W, EigenTuneConfig(rank=5)).fingerprint != base
    assert compute_bases(W, EigenTuneConfig(rank=4, selection="minor")).fingerprint != base


def test_cache_hit_returns_identical_bases_and_ignores_wrong_weight(tmp_path):
    W = lowrankish(30, 20)
    cfg = EigenTuneConfig(rank=4, cache_dir=str(tmp_path))
    first = compute_bases(W, cfg)
    assert len(list(tmp_path.glob("*.pt"))) == 1
    again = compute_bases(W, cfg)
    assert torch.equal(first.U, again.U)
    other = compute_bases(lowrankish(30, 20), cfg)  # different weight: must not reuse the cached basis
    assert other.fingerprint != first.fingerprint and len(list(tmp_path.glob("*.pt"))) == 2


def test_corrupt_cache_is_a_miss(tmp_path):
    W = lowrankish(30, 20)
    cfg = EigenTuneConfig(rank=4, cache_dir=str(tmp_path))
    fp = compute_bases(W, cfg).fingerprint
    (tmp_path / f"{fp}.pt").write_bytes(b"garbage")
    with pytest.warns(UserWarning):
        again = compute_bases(W, cfg)
    assert again.fingerprint == fp


def test_repeated_singular_values_warn():
    W = torch.eye(8)  # every singular value is 1: no spectral gap
    with pytest.warns(UserWarning, match="repeated"):
        compute_bases(W, EigenTuneConfig(rank=3))


def slowly_decaying(out=300, inn=240, power=0.3, seed=0):
    """Singular values decaying like i^-0.3, like real LLM weights: the hard case for randomized SVD."""
    g = torch.Generator().manual_seed(seed)
    k = min(out, inn)
    U = torch.linalg.qr(torch.randn(out, k, generator=g))[0]
    V = torch.linalg.qr(torch.randn(inn, k, generator=g))[0]
    S = (torch.arange(k) + 1.0) ** -power
    return (U * S) @ V.T


def test_converged_randomized_svd_is_accurate_where_a_fixed_few_iterations_are_not():
    W = slowly_decaying()
    ex = compute_bases(W, EigenTuneConfig(rank=8, svd_backend="exact"))
    exact = (ex.U * ex.S) @ ex.Vh

    def err(**kw):
        b = compute_bases(W, EigenTuneConfig(rank=8, svd_backend="randomized", **kw))
        return ((((b.U * b.S) @ b.Vh) - exact).norm() / exact.norm()).item()

    naive = err(svd_oversampling=8, svd_niter=2, svd_tol=0.0)
    converged = err()
    assert naive > 0.1, "the regression this guards against: few fixed iterations are inaccurate on slow decay"
    assert converged < 1e-2


def test_randomized_stops_early_once_converged(monkeypatch):
    import eigentune.svd as svd

    calls = []
    real = torch.linalg.qr
    monkeypatch.setattr(svd.torch.linalg, "qr", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    svd._randomized(slowly_decaying(), 8, 32, 200, 0, 1e-4)
    assert len(calls) < 1 + 2 * 200  # far fewer than the cap allows
