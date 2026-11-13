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
_A_FNS_M = {}
_A_FNS_NM = {}
_A_FNS_RI = {}
_A_FNS_BI = {}


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

    ``c`` and ``g`` are the full coefficient vectors and ``f``/``bt`` the
    full basis; :func:`dense_factor_merge_a` is the same for a basis stored
    with duplicate blocks merged.

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


def dense_factor_merge_a(f, bt, c, g, off, sz, map_u, map_i, max_basis=None):
    """Like :func:`dense_factor_a` for a merged basis.

    ``f`` and ``bt`` are the reduced tensors; ``map_u`` (length ``nu``) and
    ``map_i`` (length ``nparam``) give the stored column of every full
    index, so the kernel reads the reduced basis directly instead of an
    expanded copy.
    """
    max_basis = _resolve(sz, max_basis, "dense_factor_merge_a")
    _check_g(g)
    fn = _cached(_A_FNS_M, max_basis, _make_a_merge)
    return fn(f, bt, c, g, off, sz, map_u, map_i)


def dense_factor_an(f, bt, c, g, off, sz, max_basis=None):
    """Same as :func:`dense_factor_a` for chains with several lines.

    ``off``/``sz`` hold ``n_chain * n_line`` entries (``n_line`` is
    inferred from their length); a line with ``sz == 0`` is absent and
    multiplies the denominator by one.  ``n_line`` is not bounded and may be
    zero.
    """
    max_basis = _resolve(sz, max_basis, "dense_factor_an")
    _check_g(g)
    fn = _cached(_A_FNS_N, max_basis, _make_an)
    return fn(f, bt, c, g, off, sz)


def dense_factor_merge_an(f, bt, c, g, off, sz, map_u, map_i, max_basis=None):
    """Same as :func:`dense_factor_merge_a` for several lines."""
    max_basis = _resolve(sz, max_basis, "dense_factor_merge_an")
    _check_g(g)
    fn = _cached(_A_FNS_NM, max_basis, _make_an_merge)
    return fn(f, bt, c, g, off, sz, map_u, map_i)


def dense_factor_ri(
    fs,
    bs,
    c,
    g,
    off,
    sz,
    mu,
    mi,
    fre,
    fim,
    bre,
    bim,
    max_basis=None,
    nproj=None,
):
    """Like :func:`dense_factor_merge_an` on the stored real blocks.

    This is the fused contraction for ``method: real_index``: ``fs`` and
    ``bs`` are the blocks as they are stored (``[ne, nf]`` / ``[ne, nb]``
    real, the merged columns of ``[Re | Im]``), and ``(fre, fim)`` /
    ``(bre, bim)`` say which stored columns hold the real and imaginary part
    of every reduced column, so the complex basis is never rebuilt.  The
    index merge is the one of the Merge ops (``mu``/``mi``); any number of
    denominator lines is accepted, like :func:`dense_factor_an`.

    ``np`` (the projection axis of the flattened F block) is recovered from
    the map lengths, so the maps must be constants - they are when they come
    from a merge record.  ``nproj`` may be supplied by a record-derived
    prepare; the inference is the fallback for direct callers.
    """
    max_basis = _resolve(sz, max_basis, "dense_factor_ri")
    _check_g(g)
    if nproj is not None:
        nproj = int(nproj)
    else:
        nproj = _ri_nproj(fre, mu)
    fn = _cached(_A_FNS_RI, (max_basis, nproj), _make_ri)
    return fn(fs, bs, c, g, off, sz, mu, mi, fre, fim, bre, bim)


def dense_factor_bi(
    fs,
    bs,
    c,
    g,
    off,
    sz,
    mu,
    mi,
    fre,
    fim,
    bre,
    bim,
    f_ptr,
    f_col,
    f_val,
    b_ptr,
    b_col,
    b_val,
    terms,
    max_basis=None,
    nproj=None,
):
    """Like :func:`dense_factor_merge_an` on a sparse basis_index block.

    This is the fused contraction for ``method: basis_index``: ``fs`` and
    ``bs`` store only the *basis* columns (``[ne, nf]`` / ``[ne, nb]`` real,
    the ``[Re | Im]`` merge of those columns), ``(fre, fim)`` / ``(bre, bim)``
    say where the real and imaginary part of every basis column lives, and
    the per-reduced-column sparse transform is CSR:

        column j = sum_{i < ptr[j+1]-ptr[j]} val[ptr[j]+i] * basis[col[ptr[j]+i]]

    ``f_ptr``/``f_col``/``f_val`` are the F transform (``f_ptr`` has
    ``np * nu_red + 1`` entries) and ``b_ptr``/``b_col``/``b_val`` the B one
    (``nparam_red + 1``).  Exact row bounds, no padding.  The index merge is
    the one of the Merge ops (``mu``/``mi``); any number of denominator lines
    is accepted.

    ``np`` (the projection axis of the flattened F block) is recovered from
    ``f_ptr``, so the maps and arrays must be constants - they are when they
    come from a merge record.  ``nproj`` may be supplied by a record-derived
    prepare; the inference is the fallback for direct callers.
    """
    # ``nproj`` is a template bound, so resolve it before the casts/reshapes
    # below: inside a ``tf.function`` those build ordinary graph ops whose
    # static value is unavailable, while a record tensor is a captured
    # constant.  Resolving afterwards silently disabled the fused path
    # whenever the amplitude was traced.
    if nproj is not None:
        nproj = int(nproj)
    else:
        nproj = _bi_nproj(f_ptr, mu)
    f_ptr = tf.reshape(tf.cast(f_ptr, tf.int32), [-1])
    f_col = tf.reshape(tf.cast(f_col, tf.int32), [-1])
    f_val = tf.reshape(tf.cast(f_val, tf.complex128), [-1])
    b_ptr = tf.reshape(tf.cast(b_ptr, tf.int32), [-1])
    b_col = tf.reshape(tf.cast(b_col, tf.int32), [-1])
    b_val = tf.reshape(tf.cast(b_val, tf.complex128), [-1])
    max_basis = _resolve(sz, max_basis, "dense_factor_bi")
    terms = int(terms)
    if not 1 <= terms <= 4:
        raise ValueError(
            "terms must be 1, 2, 3 or 4 (the unrolled transform reader), "
            "got {}".format(terms)
        )
    # the reader is an unrolled template: terms is part of the kernel choice
    _check_g(g)
    fn = _cached(
        _A_FNS_BI,
        (max_basis, nproj, terms),
        _make_bi,
    )
    return fn(
        fs,
        bs,
        c,
        g,
        off,
        sz,
        mu,
        mi,
        fre,
        fim,
        bre,
        bim,
        f_ptr,
        f_col,
        f_val,
        b_ptr,
        b_col,
        b_val,
    )


def _bi_nproj(f_ptr, mu):
    """``np`` of the flattened F block, from a sparse-transform record.

    ``f_ptr`` has ``np * nu_red + 1`` entries, so ``ncol = len(f_ptr) - 1``.
    The reduced chain ids in ``mu`` show ``nu_red``: no reduced group can be
    empty, so ``nu_red`` is one more than their largest id.  All inputs are
    constants of a merge record, so this needs no device read.
    """
    m = tf.get_static_value(mu)
    p = tf.get_static_value(f_ptr)
    if m is None or p is None:
        raise ValueError(
            "dense_factor_bi needs constant mu/f_ptr arrays to infer np"
        )
    m = np.asarray(m)
    ncol = np.asarray(p).size - 1
    n_u = int(np.max(m)) + 1 if m.size else 0
    if n_u < 1 or ncol % n_u:
        raise ValueError(
            "the F transform has {} columns for {} reduced chains".format(
                ncol, n_u
            )
        )
    return ncol // n_u


def _ri_nproj(fre, mu):
    """``np`` of the flattened real F block, from the record maps.

    ``fre`` has one entry per reduced column and projection
    (``np * n_u``), while the reduced chain ids in ``mu`` show ``n_u``: no
    reduced group can be empty, so ``n_u`` is one more than their largest
    id.  Both are constants of a merge record, so this needs no device read.
    """
    m = tf.get_static_value(mu)
    f = tf.get_static_value(fre)
    if m is None or f is None:
        raise ValueError(
            "dense_factor_ri needs constant mu/fre maps to infer np"
        )
    m = np.asarray(m)
    total = np.asarray(f).size
    n_u = int(np.max(m)) + 1 if m.size else 0
    if n_u < 1 or total % n_u:
        raise ValueError(
            "fre has {} entries for {} reduced chains".format(total, n_u)
        )
    return total // n_u


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


def _fwd_merge(f, bt, c, g, off, sz, mu, mi, max_basis, general):
    lib = _load()
    if f.shape.rank == 2:
        f = tf.expand_dims(f, 1)
    if general:
        a, d = lib.dense_factor_fwd_n_merge(
            f, bt, c, g, off, sz, mu, mi, max_basis=max_basis
        )
    else:
        a = lib.dense_factor_fwd_merge(
            f, bt, c, g, off, sz, mu, mi, max_basis=max_basis
        )
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


def _grad_merge(f, bt, c, g, off, sz, dla, d, mu, mi, max_basis, general):
    lib = _load()
    if general:
        return lib.dense_factor_grad_n_merge(
            f, bt, c, g, off, sz, mu, mi, dla, d, max_basis=max_basis
        )
    return lib.dense_factor_grad_merge(
        f, bt, c, g, off, sz, mu, mi, dla, max_basis=max_basis
    )


def _fwd_ri(
    fs, bs, c, g, off, sz, mu, mi, fre, fim, bre, bim, nproj, max_basis
):
    lib = _load()
    # fs/bs keep their stored dtype (float64, float32 or fp16): the kernel
    # reads the real block in place and widens each value to double.
    a, d = lib.dense_factor_fwd_ri(
        fs,
        bs,
        c,
        g,
        off,
        sz,
        mu,
        mi,
        fre,
        fim,
        bre,
        bim,
        max_basis=max_basis,
        np=nproj,
    )
    return tf.reshape(a, tf.stack([tf.shape(fs)[0], nproj])), d


def _grad_ri(
    fs, bs, c, g, off, sz, mu, mi, fre, fim, bre, bim, dla, d, nproj, max_basis
):
    lib = _load()
    # same as _fwd_ri: the stored block keeps its dtype
    return lib.dense_factor_grad_ri(
        fs,
        bs,
        c,
        g,
        off,
        sz,
        mu,
        mi,
        fre,
        fim,
        bre,
        bim,
        dla,
        d,
        max_basis=max_basis,
        np=nproj,
    )


def _fwd_bi(
    fs,
    bs,
    c,
    g,
    off,
    sz,
    mu,
    mi,
    fre,
    fim,
    bre,
    bim,
    f_ptr,
    f_col,
    f_val,
    b_ptr,
    b_col,
    b_val,
    nproj,
    max_basis,
    terms,
):
    lib = _load()
    # fs/bs go through with their stored dtype (float64, float32 or fp16);
    # the kernel reads the stored block in place and widens each value to
    # double, so an upcast here would only copy the payload.
    a, d = lib.dense_factor_fwd_bi(
        fs,
        bs,
        c,
        g,
        tf.cast(off, tf.int32),
        tf.cast(sz, tf.int32),
        tf.cast(mu, tf.int32),
        tf.cast(mi, tf.int32),
        tf.cast(fre, tf.int32),
        tf.cast(fim, tf.int32),
        tf.cast(bre, tf.int32),
        tf.cast(bim, tf.int32),
        tf.cast(f_ptr, tf.int32),
        tf.cast(f_col, tf.int32),
        tf.cast(f_val, tf.complex128),
        tf.cast(b_ptr, tf.int32),
        tf.cast(b_col, tf.int32),
        tf.cast(b_val, tf.complex128),
        max_basis=max_basis,
        np=nproj,
        terms=terms,
    )
    return tf.reshape(a, tf.stack([tf.shape(fs)[0], nproj])), d


def _grad_bi(
    fs,
    bs,
    c,
    g,
    off,
    sz,
    mu,
    mi,
    fre,
    fim,
    bre,
    bim,
    f_ptr,
    f_col,
    f_val,
    b_ptr,
    b_col,
    b_val,
    dla,
    d,
    nproj,
    max_basis,
    terms,
):
    lib = _load()
    # same as _fwd_bi: the stored block keeps its dtype
    return lib.dense_factor_grad_bi(
        fs,
        bs,
        c,
        g,
        tf.cast(off, tf.int32),
        tf.cast(sz, tf.int32),
        tf.cast(mu, tf.int32),
        tf.cast(mi, tf.int32),
        tf.cast(fre, tf.int32),
        tf.cast(fim, tf.int32),
        tf.cast(bre, tf.int32),
        tf.cast(bim, tf.int32),
        tf.cast(f_ptr, tf.int32),
        tf.cast(f_col, tf.int32),
        tf.cast(f_val, tf.complex128),
        tf.cast(b_ptr, tf.int32),
        tf.cast(b_col, tf.int32),
        tf.cast(b_val, tf.complex128),
        dla,
        d,
        max_basis=max_basis,
        np=nproj,
        terms=terms,
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


def _make_a_merge(max_basis, general=False):
    @tf.custom_gradient
    def fn(f, bt, c, g, off, sz, mu, mi):
        f, a, d = _fwd_merge(f, bt, c, g, off, sz, mu, mi, max_basis, general)

        def grad(dla):
            if dla is None:
                return [None] * 8
            dc, dg = _grad_merge(
                f, bt, c, g, off, sz, dla, d, mu, mi, max_basis, general
            )
            return None, None, dc, dg, None, None, None, None

        return a, grad

    return fn


def _make_an(max_basis):
    return _make_a(max_basis, general=True)


def _make_an_merge(max_basis):
    return _make_a_merge(max_basis, general=True)


def _make_ri(key):
    max_basis, nproj = key

    @tf.custom_gradient
    def fn(fs, bs, c, g, off, sz, mu, mi, fre, fim, bre, bim):
        a, d = _fwd_ri(
            fs,
            bs,
            c,
            g,
            off,
            sz,
            mu,
            mi,
            fre,
            fim,
            bre,
            bim,
            nproj,
            max_basis,
        )

        def grad(dla):
            if dla is None:
                return [None] * 12
            dc, dg = _grad_ri(
                fs,
                bs,
                c,
                g,
                off,
                sz,
                mu,
                mi,
                fre,
                fim,
                bre,
                bim,
                dla,
                d,
                nproj,
                max_basis,
            )
            return [None, None, dc, dg] + [None] * 8

        return a, grad

    return fn


def _make_bi(key):
    max_basis, nproj, terms = key

    @tf.custom_gradient
    def fn(
        fs,
        bs,
        c,
        g,
        off,
        sz,
        mu,
        mi,
        fre,
        fim,
        bre,
        bim,
        f_ptr,
        f_col,
        f_val,
        b_ptr,
        b_col,
        b_val,
    ):
        a, d = _fwd_bi(
            fs,
            bs,
            c,
            g,
            off,
            sz,
            mu,
            mi,
            fre,
            fim,
            bre,
            bim,
            f_ptr,
            f_col,
            f_val,
            b_ptr,
            b_col,
            b_val,
            nproj,
            max_basis,
            terms,
        )

        def grad(dla):
            if dla is None:
                return [None] * 18
            dc, dg = _grad_bi(
                fs,
                bs,
                c,
                g,
                off,
                sz,
                mu,
                mi,
                fre,
                fim,
                bre,
                bim,
                f_ptr,
                f_col,
                f_val,
                b_ptr,
                b_col,
                b_val,
                dla,
                d,
                nproj,
                max_basis,
                terms,
            )
            return [None, None, dc, dg] + [None] * 14

        return a, grad

    return fn


def basis_event_major(b):
    """Transpose the dense ``B`` ``[n_param, n_event]`` to event-major.

    The kernel needs ``bt[e, j]`` contiguous in ``j`` so that the inner
    basis loop is coalesced.
    """
    return tf.transpose(tf.cast(b, tf.complex128))


def chain_layout(structure):
    """Build the ``(off, sz)`` layout of a dense structure.

    ``off``/``sz`` hold ``n_chain * n_line`` entries: one pair per line
    slot, ``max_internal_lines`` slots for every used chain, which is what
    ``dense_factor_an`` expects (it takes ``n_line`` from their length).  A
    chain with fewer lines is padded with ``sz == 0``, so the pairs of the
    following chains stay at ``k * n_line``.
    """
    den_sizes = structure["den_sizes"]
    param_offset = structure["param_offset"]
    off = []
    sz = []
    for k in structure["u_chain"]:
        sizes = den_sizes[k]
        offsets = param_offset[k]
        for l in range(structure["max_internal_lines"]):
            if l < len(sizes):
                off.append(offsets[l])
                sz.append(sizes[l])
            else:  # absent line: multiply the product by 1
                off.append(0)
                sz.append(0)
    return (
        tf.constant(np.asarray(off, dtype=np.int32)),
        tf.constant(np.asarray(sz, dtype=np.int32)),
    )


__all__ = [
    "dense_factor_a",
    "dense_factor_an",
    "dense_factor_merge_a",
    "dense_factor_merge_an",
    "dense_factor_ri",
    "dense_factor_bi",
    "basis_event_major",
    "chain_layout",
    "is_available",
]
