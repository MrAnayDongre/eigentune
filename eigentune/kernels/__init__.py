"""EigenTune compute backends.

A backend implements two operations on 2-D inputs (``N`` tokens) and registers itself by name:

* ``forward(x, Vh, U, w, kind, base_out) -> (y, Q)`` with ``Q = x @ Vh.T`` (``[N, r]``),
  ``Z = Q * w`` (``kind="diag"``) or ``Q @ w.T`` (``kind="core"``), ``y = base_out + Z @ U.T``
* ``backward(g, Q, Vh, U, w, kind, need_x) -> (grad_x or None, grad_w)``

``torch`` is the reference; every other backend is tested against it. See ``docs/kernels.md``.
"""

from .autograd import eigentune_update
from .dispatch import Backend, available_backends, get_backend, register_backend, select_backend

__all__ = ["Backend", "available_backends", "get_backend", "register_backend", "select_backend", "eigentune_update"]

from .reference import TorchBackend as _TorchBackend

register_backend(_TorchBackend())

try:  # the Triton backend is optional: CPU-only installs never import it
    from .triton_backend import TritonBackend as _TritonBackend

    register_backend(_TritonBackend())
except ImportError:
    pass

try:  # the native backend only registers itself; the extension is built on first use
    from .native_backend import NativeBackend as _NativeBackend

    register_backend(_NativeBackend())
except ImportError:
    pass
