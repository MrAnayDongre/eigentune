"""One autograd function for every backend: it saves ``Q`` (``[N, r]``) and the frozen bases, never ``x``."""

from __future__ import annotations

import torch

from .dispatch import get_backend, select_backend


class _EigenTuneFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base_out, Vh, U, w, kind, backend):
        b = select_backend(backend, x, w.shape[0], kind)
        y, Q = b.forward(x, Vh, U, w, kind, base_out)
        ctx.save_for_backward(Q, Vh, U, w)
        ctx.kind, ctx.backend_name = kind, b.name
        return y

    @staticmethod
    def backward(ctx, g):
        Q, Vh, U, w = ctx.saved_tensors
        g = g.contiguous()
        gx, gw = get_backend(ctx.backend_name).backward(g, Q, Vh, U, w, ctx.kind, ctx.needs_input_grad[0])
        return gx, (g if ctx.needs_input_grad[1] else None), None, None, gw.to(w.dtype), None, None


def eigentune_update(x, base_out, Vh, U, w, kind, backend="auto"):
    """``base_out + (scale(x @ Vh.T)) @ U.T`` for 2-D ``x``/``base_out``; differentiable in ``x``, ``base_out``, ``w``."""
    return _EigenTuneFunction.apply(x, base_out, Vh, U, w, kind, backend)
