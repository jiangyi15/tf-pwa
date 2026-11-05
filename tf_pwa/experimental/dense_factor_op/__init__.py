"""Fused dense-factor contraction as a TensorFlow custom op.

``DenseFactorFwd`` / ``DenseFactorGrad`` compute, in two CUDA kernels::

    den[u,e] = sum_i g[off[u]+i] * bt[e, off[u]+i]
    A[e]     = sum_u F[e,u] * C[u] / den[u,e]
    P[e]     = |A[e]|**2

and the vector-Jacobian product for an arbitrary per-event cotangent
``dl_dp = dL/dP``::

    dL/dC   = sum_e dl_dp[e] * dP[e]/dC
    dL/dg   = sum_e dl_dp[e] * dP[e]/dg

:func:`dense_factor_p` wraps the pair in a ``tf.custom_gradient``, so any
TensorFlow loss built on ``P`` differentiates through it.  The basis and the coefficient
layout are opaque to the kernel: it only sees ``bt`` (event-major basis),
``g`` (coefficients) and ``(off, sz)`` describing where each chain's
coefficients live.  That makes it valid for BWR (3), GS (5), Flatte
(2 + n_gamma) or any other basis the particle model defines.

The point of the op is that the whole ``C/D -> F·W -> |A|^2`` chain,
which the dense model currently expresses as ``sparse_dense_matmul`` +
``divide`` + ``matvec`` + ``reduce_sum``, becomes one pass over the data
with analytic gradients instead of building a large autodiff graph.
"""

import numpy as np
import tensorflow as tf

from .build import build

_lib = None


def _load():
    """Build/load the op on first use (never at import time)."""
    global _lib
    if _lib is None:
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError(
                "dense_factor_op has GPU kernels only; no GPU is visible "
                "to TensorFlow"
            )
        _lib = tf.load_op_library(build())
    return _lib


def is_available():
    """Return whether the op can be built, loaded and run on this machine.

    Importing this module never compiles anything; call this (or
    :func:`dense_factor_p`) to opt in.  Returns ``False`` instead of
    raising when there is no visible GPU, or when nvcc, g++, the CUDA
    toolkit or the abseil headers are missing.
    """
    if not tf.config.list_physical_devices("GPU"):
        return False
    try:
        _load()
        return True
    except Exception:
        return False


_P_FNS = {}


def dense_factor_p(f, bt, c, g, off, sz, max_basis=None):
    """Per-event probability with an arbitrary upstream gradient.

    The backward receives whatever ``dL/dP`` the surrounding loss produces
    and passes it to the gradient kernel, so any TensorFlow loss built on
    the per-event probability works.
    """
    if max_basis is None:
        value = tf.get_static_value(sz)
        if value is None:
            raise ValueError(
                "dense_factor_p needs max_basis when sz is symbolic"
            )
        max_basis = int(np.max(np.asarray(value)))
    max_basis = int(max_basis)
    fn = _P_FNS.get(max_basis)
    if fn is None:
        fn = _make_p(max_basis)
        _P_FNS[max_basis] = fn
    return fn(f, bt, c, g, off, sz)


def _make_p(max_basis):
    @tf.custom_gradient
    def fn(f, bt, c, g, off, sz):
        if f.shape.rank == 2:
            f = tf.expand_dims(f, 1)   # [ne, nu] -> [ne, 1, nu]
        a, p = _load().dense_factor_fwd(
            f, bt, c, g, off, sz, max_basis=max_basis
        )

        def grad(dp):
            if dp is None:
                return [None] * 6
            # d(sum dp_e P_e)/dC, /dg is exactly the VJP of the loss.
            dc, dg = _load().dense_factor_grad(
                f, bt, c, g, off, sz, dp, a, max_basis=max_basis
            )
            return None, None, dc, dg, None, None

        return p, grad

    return fn


def basis_event_major(b):
    """Transpose the dense ``B`` ``[n_param, n_event]`` to event-major.

    The kernel needs ``bt[e, j]`` contiguous in ``j`` so that the inner
    basis loop is coalesced.
    """
    return tf.transpose(tf.cast(b, tf.complex128))


def chain_layout(structure):
    """Build ``(off, sz)`` for the used chains of a dense structure."""
    metas = structure["chain_meta"]
    off = []
    sz = []
    for k in structure["u_chain"]:
        meta = metas[k]
        off.append(meta["param_offset"][0])
        sz.append(meta["den_sizes"][0])
    return (
        tf.constant(np.asarray(off, dtype=np.int32)),
        tf.constant(np.asarray(sz, dtype=np.int32)),
    )


__all__ = [
    "dense_factor_p",
    "basis_event_major",
    "chain_layout",
    "is_available",
]
