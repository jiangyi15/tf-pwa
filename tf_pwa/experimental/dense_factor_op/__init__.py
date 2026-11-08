"""Fused dense-factor contraction as a TensorFlow custom op.

``DenseFactorFwd`` / ``DenseFactorGrad`` compute, in two CUDA kernels::

    den[u,e] = sum_i g[off[u]+i] * bt[e, off[u]+i]
    A[e, p]  = sum_u F[e,p,u] * C[u] / den[u,e]

The forward returns the amplitude matrix ``A``; the probability is the
density-matrix contraction the dense model already ends with
(``DecayGroup.sum_with_polarization``).  The backward takes an arbitrary
complex cotangent ``dL/dA`` and returns the vector-Jacobian product for the
coefficients ``C`` and ``g``.

:func:`dense_factor_a` wraps the pair in a ``tf.custom_gradient``, so any
TensorFlow loss built on ``A`` differentiates through it.  The basis and the coefficient
layout are opaque to the kernel: it only sees ``bt`` (event-major basis),
``g`` (coefficients) and ``(off, sz)`` describing where each chain's
coefficients live.  That makes it valid for BWR (3), GS (5), Flatte
(2 + n_gamma) or any other basis the particle model defines.

The point of the op is that the ``C/D -> F·W -> A`` chain, which the dense
model expresses as ``sparse_dense_matmul`` + ``divide`` + ``matvec``,
becomes one pass over the data with analytic gradients instead of building
a large autodiff graph; the probability is a density-matrix contraction on
top of ``A``, shared with the dense model.
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
    :func:`dense_factor_a`) to opt in.  Returns ``False`` instead of
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


_A_FNS = {}
_A_FNS_N = {}


def _resolve(sz, max_basis, who):
    if max_basis is None:
        value = tf.get_static_value(sz)
        if value is None:
            raise ValueError(
                "{} needs max_basis when sz is symbolic".format(who)
            )
        max_basis = int(np.max(np.asarray(value)))
    return int(max_basis)


def _check_g(g):
    if g.dtype.is_complex:
        raise TypeError("g must be real, e.g. tf.math.real(g)")


def _cached(cache, key, maker):
    fn = cache.get(key)
    if fn is None:
        fn = maker(key)
        cache[key] = fn
    return fn


def _event_shape(f):
    return tf.shape(f)[:2]


def dense_factor_a(f, bt, c, g, off, sz, max_basis=None):
    """Per-event amplitude matrix ``A[e, p]``, shape ``[ne, np]``.

    This is the fused contraction's natural output::

        W_u = C_u / den_u,    A_p = sum_u F[p,u] W_u

    The backward takes an arbitrary complex ``dL/dA``, so any loss built on
    the amplitudes works — including the polarization contraction
    ``DecayGroup.sum_with_polarization`` of the dense model.

    ``g`` is the denominator coefficient vector and is real: it must be
    float64.  Passing a complex tensor raises instead of silently dropping
    its imaginary part.
    """
    max_basis = _resolve(sz, max_basis, "dense_factor_a")
    _check_g(g)
    fn = _cached(_A_FNS, max_basis, _make_a)
    return fn(f, bt, c, g, off, sz)


def dense_factor_an(f, bt, c, g, off, sz, max_basis=None):
    """Same as :func:`dense_factor_a` for chains with several lines.

    ``off``/``sz`` hold ``n_chain * n_line`` entries (``n_line`` is inferred
    from their length); a line with ``sz == 0`` is absent and multiplies the
    denominator by one.  ``n_line`` is not bounded and may be zero.
    """
    max_basis = _resolve(sz, max_basis, "dense_factor_an")
    _check_g(g)
    fn = _cached(_A_FNS_N, max_basis, _make_an)
    return fn(f, bt, c, g, off, sz)


def _fwd(f, bt, c, g, off, sz, max_basis, general):
    lib = _load()
    if f.shape.rank == 2:
        f = tf.expand_dims(f, 1)  # [ne, nu] -> [ne, 1, nu]
    if general:
        a, d = lib.dense_factor_fwd_n(
            f, bt, c, g, off, sz, max_basis=max_basis
        )
    else:
        a = lib.dense_factor_fwd(f, bt, c, g, off, sz, max_basis=max_basis)
        d = None
    return f, tf.reshape(a, _event_shape(f)), d


def _grad(f, bt, c, g, off, sz, dla, d, max_basis, general):
    lib = _load()
    if general:
        return lib.dense_factor_grad_n(
            f, bt, c, g, off, sz, dla, d, max_basis=max_basis
        )
    return lib.dense_factor_grad(
        f, bt, c, g, off, sz, dla, max_basis=max_basis
    )


def _make_a(max_basis, general=False):
    @tf.custom_gradient
    def fn(f, bt, c, g, off, sz):
        f, a, d = _fwd(f, bt, c, g, off, sz, max_basis, general)

        def grad(dla):
            if dla is None:
                return [None] * 6
            dc, dg = _grad(f, bt, c, g, off, sz, dla, d, max_basis, general)
            return None, None, dc, dg, None, None

        return a, grad

    return fn


def _make_an(max_basis):
    return _make_a(max_basis, general=True)


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
    "dense_factor_a",
    "dense_factor_an",
    "basis_event_major",
    "chain_layout",
    "is_available",
]
