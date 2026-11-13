"""``dense_factor`` amplitude model backed by the fused custom op.

Registered as ``amp_model: dense_factor_op``.  It reuses the dense
preprocessor (event-major ``F``/``B``, ``structure``, coefficient build)
and replaces only the contraction

.. math::

    W_u = \\frac{C_u}{\\sum_j g_j B_{u,j}},
    \\qquad A = \\sum_u F_u W_u,
    \\qquad P = |A|^2

with :func:`tf_pwa.experimental.dense_factor_op.dense_factor_a`.

The kernels take the denominator coefficients as real, which every term
in this plugin satisfies; a model whose lines build complex ones is
rejected rather than silently truncated.

The kernel returns the amplitude matrix ``A[e, p]``, so the model ends
with the same ``DecayGroup.sum_with_polarization`` contraction as the dense
path, polarization included.  Chains with one denominator line use
:func:`dense_factor_a`; every other line count goes through
:func:`dense_factor_an`, which is not bounded by the number of lines.

The original ``dense_factor`` model is untouched.  Importing this module is
the opt-in to the *compute*: the op package is built and loaded here, once,
and the model prefers the fused backend.  If the build fails (no toolkit, no
headers, no GPU) the module warns once and the same models run the TF
reference contraction instead - same configuration, same numbers at full
precision, just slower.

Config::

    data:
      amp_model: dense_factor_op
      preprocessor:
        dense_factor:
          force_cpu: true
"""

import warnings as _warnings
from types import MappingProxyType

import numpy as np
import tensorflow as tf

from tf_pwa.amp.amp import register_amp_model

from .dense_factor import (
    _TERMS,
    DenseFactorModel,
    _unpack_stored,
    _warn_op_failure,
    basis_key,
    merge_record,
)
from .dense_factor.backend import (
    BACKENDS,
    PREPARE,
    get_op,
    register_backend,
    register_op,
    register_prepare,
)

# Importing the package is the opt-in to the compute: its ``__init__``
# builds and loads the library (so a GPU runner's ``pytest --doctest-modules``
# builds during collection).  A machine without the toolkit, the headers or a
# GPU is not an error: the backend is simply not registered, ``supports``
# declines and the models run the TF reference, which is the same contraction
# at full precision.  That case is reported when a fused model is first built,
# not at import, so a GPU-less process that never uses these models stays
# silent.
try:
    import tf_pwa.experimental.dense_factor_op as _op
    from tf_pwa.experimental.dense_factor_op import chain_layout
except Exception as exc:  # pragma: no cover - depends on the machine
    _op = None
    chain_layout = None
    _HAVE_CUDA = False
    _CUDA_ERROR = exc
else:
    _HAVE_CUDA = bool(_op.is_available())
    _CUDA_ERROR = None

_WARNED_NO_CUDA = False


def _warn_no_cuda():
    """Report a missing build once, when a fused model first needs it."""
    global _WARNED_NO_CUDA
    if _WARNED_NO_CUDA:
        return
    _WARNED_NO_CUDA = True
    _warnings.warn(
        "dense_factor_op models run on the dense backend: %s"
        % (
            _CUDA_ERROR
            if _CUDA_ERROR is not None
            else "the fused op is not available on this machine "
            "(no GPU, or the build failed)"
        )
    )


def _wrap(method, run, strategy, record, F, B, C, g, layout, prepared):
    """Shared entry: guards, ``use_op``, warn-once, reference fallback.

    ``run`` is the op call itself; it returns ``None`` when the record is not
    what this kernel reads, which falls through to the reference too.  The
    layout is the model-owned dict: ``off``/``sz`` are the fused keys, and
    the reference reads its own keys from the same dict.  ``prepared`` holds
    the backend's static facts for this record (``None`` when there are
    none), and is only ever host-side ints -- a ``None`` prepared is not a
    failure, it just leaves the op to infer what it needs.
    """
    # the kernels read real denominator coefficients and at least one line;
    # the dense reference has neither restriction
    if g.dtype.is_complex:
        raise NotImplementedError(
            "dense_factor_op needs real denominator coefficients; "
            "use the dense_factor model for this one"
        )
    if layout["n_line"] < 1:
        raise NotImplementedError(
            "dense_factor_op needs at least one denominator line"
        )
    if layout["off"] is None or layout["sz"] is None:
        # the dense models have no fused layout to pass: the kernels cannot
        # run, so say so once and fall through to the reference
        _warn_op_failure(
            method,
            NotImplementedError(
                "no fused layout for this model; "
                "using the reference contraction"
            ),
        )
    elif getattr(strategy, "use_op", True):
        try:
            amp = run(
                strategy,
                record,
                F,
                B,
                C,
                g,
                layout["off"],
                layout["sz"],
                layout["n_line"],
                layout["max_basis"],
                prepared,
            )
        except Exception as exc:  # environment or map mismatch
            # one block the op cannot take must not break the model: report
            # it once and use the reference for this call
            _warn_op_failure(method, exc)
        else:
            if amp is not None:
                return amp
    # declined (the block is not what the op reads) or failed: the TF
    # reference takes the *stored* tensors and expands them itself
    return get_op("tf", method)(strategy, record, F, B, C, g, layout, prepared)


# -- the op calls (the raw kernel invocation, no guards) --------------------
def _op_plain(
    strategy, record, F, B, C, g, off, sz, n_line, max_basis, prepared=None
):
    F = _unpack_stored(F, record, C.dtype)
    B = _unpack_stored(B, record, C.dtype)
    return _plain_op(F, B, C, g, off, sz, n_line, max_basis)


def _op_merge(
    strategy, record, F, B, C, g, off, sz, n_line, max_basis, prepared=None
):
    F = _unpack_stored(F, record, C.dtype)
    B = _unpack_stored(B, record, C.dtype)
    fn = _op.dense_factor_merge_a if n_line == 1 else _op.dense_factor_merge_an
    return fn(
        F,
        B,
        C,
        g,
        off,
        sz,
        record["map_u"],
        record["map_i"],
        max_basis=max_basis,
    )


def _op_basis_index(
    strategy, record, F, B, C, g, off, sz, n_line, max_basis, prepared=None
):
    if not _stored_pair(F, B):
        _warn_op_failure(
            "basis_index",
            ValueError(
                "the stored block is not the real pair the op reads; "
                "this call reconstructs the basis"
            ),
        )
        return None
    # ``terms`` is a template bound of the unrolled reader, so it comes from
    # the prepare; ``record`` is only the fallback when the prepare declined
    terms = (prepared or {}).get("terms")
    if terms is None:
        terms = int(record.get("terms", 0))
    terms = int(terms)
    if not 1 <= terms <= _TERMS:
        _warnings.warn(
            "basis_index has a {}-term transform; the unrolled reader "
            "supports 1 to {}, so this call rebuilds the basis".format(
                terms, _TERMS
            )
        )
        return None
    return _op.dense_factor_bi(
        F,
        B,
        C,
        g,
        off,
        sz,
        record["map_u"],
        record["map_i"],
        record["f_re_map"],
        record["f_im_map"],
        record["b_re_map"],
        record["b_im_map"],
        record["f_ptr"],
        record["f_col"],
        record["f_val"],
        record["b_ptr"],
        record["b_col"],
        record["b_val"],
        terms,
        max_basis=max_basis,
        nproj=(prepared or {}).get("nproj"),
    )


def _op_real_index(
    strategy, record, F, B, C, g, off, sz, n_line, max_basis, prepared=None
):
    if not _stored_pair(F, B):
        _warn_op_failure(
            "real_index",
            ValueError(
                "the stored block is not the real pair the op reads; "
                "this call rebuilds the basis"
            ),
        )
        return None
    return _op.dense_factor_ri(
        F,
        B,
        C,
        g,
        off,
        sz,
        record["map_u"],
        record["map_i"],
        record["f_re_map"],
        record["f_im_map"],
        record["b_re_map"],
        record["b_im_map"],
        max_basis=max_basis,
        nproj=(prepared or {}).get("nproj"),
    )


def _op_expand(
    strategy, record, F, B, C, g, off, sz, n_line, max_basis, prepared=None
):
    F, B = strategy.expand(record, F, B)
    return _plain_op(F, B, C, g, off, sz, n_line, max_basis)


# -- the static inputs of the fused ops, derived from the record ------------
# Only the two methods whose ops read a ``nproj`` template bound earn one:
# ``dense_factor_bi``/``ri`` would otherwise recover it with
# ``tf.get_static_value`` on every call (a device sync in eager, and ``None``
# inside a ``tf.function``).  The record stores the projection size as a plain
# int, so this is captured once instead.
def _checked_nproj(record, width_key, offset=0):
    """``record["np"]``, checked against the map widths it drives.

    ``np`` drives a reshape in the kernel, so a stale value would reshape
    fine and be silently wrong.  The record's own maps pin it down; when they
    are not static there is nothing to check against, so the prepare declines
    and the op infers the value from the same maps.
    """
    nproj = int(record["np"])
    m = tf.get_static_value(record["map_u"])
    w = tf.get_static_value(record[width_key])
    if m is None or w is None:
        return None
    m = np.asarray(m)
    n_u = int(m.max()) + 1 if m.size else 0
    width = int(np.asarray(w).size) - offset
    if n_u < 1 or nproj * n_u != width:
        return None
    return nproj


@register_prepare("cuda", "basis_index")
def _prepare_basis_index(strategy, record, layout):
    if record is None or "np" not in record:
        return None
    if layout["off"] is None:
        return None
    nproj = _checked_nproj(record, "f_ptr", offset=1)
    if nproj is None:
        return None
    return {"nproj": nproj, "terms": int(record.get("terms", 0))}


@register_prepare("cuda", "real_index")
def _prepare_real_index(strategy, record, layout):
    if record is None or "np" not in record:
        return None
    if layout["off"] is None:
        return None
    nproj = _checked_nproj(record, "f_re_map")
    if nproj is None:
        return None
    return {"nproj": nproj}


# -- the registered entries: the fused-op backend owns the execution --------
# The compression strategies only store and expand a basis; the op calls live
# here.  Each entry keeps the op family's limits (real ``g``, at least one
# line), honours the strategy's ``use_op`` knob, and on a decline or failure
# reports once and returns the TF reference for the same record - which
# expands the stored basis itself.
#
#   ``None``             the plain ``dense_factor_a``/``_an`` pair, full width
#   ``index``            ``dense_factor_merge_a``/``_an`` with the merge maps
#   ``basis_index``      ``dense_factor_bi``, on the stored sparse basis
#   ``real_index``       ``dense_factor_ri``, on the stored real block
#   ``svd``/``real_svd`` expand the low-rank coefficients, then the plain op
@register_op("cuda", None)
def _run_plain(strategy, record, F, B, C, g, layout, prepared):
    """A full-width basis: the plain op reads it as passed."""
    return _wrap(
        None, _op_plain, strategy, record, F, B, C, g, layout, prepared
    )


@register_op("cuda", "index")
def _run_merge(strategy, record, F, B, C, g, layout, prepared):
    """The index-merged basis: the merge op replays the stored index."""
    return _wrap(
        "index", _op_merge, strategy, record, F, B, C, g, layout, prepared
    )


@register_op("cuda", "basis_index")
def _run_basis_index(strategy, record, F, B, C, g, layout, prepared):
    """The stored sparse basis: the sparse-transform op reads it."""
    return _wrap(
        "basis_index",
        _op_basis_index,
        strategy,
        record,
        F,
        B,
        C,
        g,
        layout,
        prepared,
    )


@register_op("cuda", "real_index")
def _run_real_index(strategy, record, F, B, C, g, layout, prepared):
    """The stored real block: the split-real op reads it."""
    return _wrap(
        "real_index",
        _op_real_index,
        strategy,
        record,
        F,
        B,
        C,
        g,
        layout,
        prepared,
    )


@register_op("cuda", "svd")
@register_op("cuda", "real_svd")
def _run_expand(strategy, record, F, B, C, g, layout, prepared):
    """``svd``/``real_svd``: expand the store, then the plain op."""
    # the real method, so the warn-once key, the message and the reference
    # lookup are ``real_svd`` for a ``real_svd`` record
    method = record["method"] if record else "svd"
    return _wrap(
        method, _op_expand, strategy, record, F, B, C, g, layout, prepared
    )


def _unregister(name):
    """Drop a backend and its prepares (a failed build must leave none)."""
    BACKENDS.pop(name, None)
    for key in [key for key in PREPARE if key[0] == name]:
        PREPARE.pop(key)


if _HAVE_CUDA:
    register_backend("cuda")
else:
    # the decorators above created the slot; a build that failed must leave
    # no backend behind, so ``supports`` declines it and every method uses
    # the TF reference (the warning is reported when a fused model is built)
    _unregister("cuda")


# ``_plain_op``/``_stored_pair`` helpers used by the entries above.
def _plain_op(F, B, C, g, off, sz, n_line, max_basis):
    """The plain contraction: one op call, no basis transformation."""
    fn = _op.dense_factor_a if n_line == 1 else _op.dense_factor_an
    return fn(F, B, C, g, off, sz, max_basis=max_basis)


def _stored_pair(F, B):
    """Whether ``(F, B)`` are the stored real pair the bi/ri kernels read."""
    if F is None or B is None:
        return False
    if F.dtype.is_complex or B.dtype.is_complex:
        return False
    return F.shape.rank == 2 and B.shape.rank == 2


@register_amp_model("dense_factor_op")
class DenseFactorOpModel(DenseFactorModel):
    """Fused-contraction variant of :class:`DenseFactorModel`."""

    #: Backend order: the fused kernels first, the TF reference as the
    #: fallback (a machine without the build, or a block the op rejects).
    prefer = ("cuda", "tf")

    #: The coefficient build is 300 scalars and is much cheaper on CPU/XLA
    #: than as ~200 tiny GPU kernels; this pairing is the measured optimum.
    coeff_on_cpu = True

    def init_params(self, name=""):
        # the base init builds the group's basis, or raises for an
        # unsupported term model, so it exists from here on.  It also builds
        # ``self._layout`` with the dense keys and ``off``/``sz`` unset.
        super().init_params(name)
        # The fused backend needs the layout, which comes from the op
        # package.  Only its absence is tolerated: a model then runs the TF
        # reference (the entry declines on ``off is None``) and this is the
        # first moment the missing build matters, so the warning is reported
        # here rather than at import.  Errors from a package that *is*
        # importable are real bugs and propagate.
        if not _HAVE_CUDA:
            _warn_no_cuda()
        if _op is not None:
            # ``_layout`` is read-only on the base model, so the fused keys
            # extend it as a new mapping rather than updating it in place
            self._layout = MappingProxyType(
                {**self._layout, **self._op_layout()}
            )

    def _op_layout(self):
        """The fused keys of the model layout, built once in ``init_params``.

        The layout depends only on the structure, which is fixed there, so
        it is built eagerly: eager tensors are captured as constants by any
        graph that uses them, while a ``tf.constant`` created during a trace
        belongs to that ``FuncGraph`` and breaks the next trace or a later
        eager call with "tensor is out of scope".
        """
        structure = self.decay_group.get_dense_structure()
        off, sz = chain_layout(structure)
        # max_basis is the kernel template S, an unrolled loop bound, so it
        # is taken over the entries u_chain actually sends to the kernel: a
        # chain with no term combination contributes no column at all
        sizes = [
            size
            for k in structure["u_chain"]
            for size in structure["den_sizes"][k]
        ]
        return {
            "off": off,
            "sz": sz,
            "n_line": structure["max_internal_lines"],
            "max_basis": max(sizes) if sizes else 0,
        }


@register_amp_model("dense_factor_op_merge")
class DenseFactorOpMergeModel(DenseFactorOpModel):
    """Fused contraction whose basis blocks are stored merged.

    Config::

        data:
          amp_model: dense_factor_op_merge
          preprocessor:
            dense_factor_merge: {}

    Nothing is expanded: the reduced ``F``/``B`` go straight to the kernel
    together with the merge index, so the contraction never materialises a
    full-width copy of the basis.  ``merge_key`` names the shared record and
    must match the preprocessor's.
    """

    def __init__(self, *args, merge_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_id = merge_key or basis_key(self.decay_group)

    @property
    def merge(self):
        """The compression record of this basis, in the shared space."""
        return merge_record(self)
