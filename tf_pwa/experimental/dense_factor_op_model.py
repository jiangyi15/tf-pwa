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

import tensorflow as tf

from tf_pwa.amp.amp import register_amp_model

from .dense_factor import (
    DenseFactorModel,
    _compression_of,
    _unpack_stored,
    _warn_op_failure,
    basis_key,
    merge_record,
)
from .dense_factor_backend import Backend, register_backend

# Importing the package is the opt-in to the compute: its ``__init__``
# builds and loads the library (so a GPU runner's ``pytest --doctest-modules``
# builds during collection).  A machine without the toolkit, the headers or a
# GPU is not an error: the class below is still registered, ``supports``
# declines and the models run the TF reference, which is the same contraction
# at full precision.  That case is reported when a fused model first asks for
# the backend, not at import, so a GPU-less process that never uses these
# models stays silent.
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
            else "the fused op is not built on this machine"
        )
    )


class CudaBackend(Backend):
    """The fused-op backend: only the methods the op implements.

    Each entry runs the strategy's op path (``_contract_op``) on the stored
    tensors.  An op failure is already reported once by ``_warn_op_failure``
    inside ``_contract_op``; the entry then returns the TF reference for the
    same record, so a block the kernel cannot take costs precision-preserving
    time rather than a broken model.
    """

    name = "cuda"

    #: every op takes ``(off, sz)``, so a model without a layout cannot use it
    needs_layout = True

    @classmethod
    def supports(cls, method, record):
        if not _HAVE_CUDA:
            # a fused model is actually being built/evaluated: this is the
            # point where the missing build matters, so report it here
            _warn_no_cuda()
            return False
        return method in cls.functions

    @staticmethod
    def _contract(method):
        def fn(strategy, record, F, B, C, g, off, sz, *, n_line, max_basis):
            # the kernels read real denominator coefficients and at least one
            # line; the dense reference has neither restriction
            if g.dtype.is_complex:
                raise NotImplementedError(
                    "dense_factor_op needs real denominator coefficients; "
                    "use the dense_factor model for this one"
                )
            if n_line < 1:
                raise NotImplementedError(
                    "dense_factor_op needs at least one denominator line"
                )
            if getattr(strategy, "use_op", True):
                # the op wants complex blocks; a strategy that stores one real
                # block has an op that reads it in its stored dtype, so only
                # the others are unpacked here
                Fo, Bo = F, B
                if not strategy.stores_real_block:
                    Fo = _unpack_stored(F, record, C.dtype)
                    Bo = _unpack_stored(B, record, C.dtype)
                own = getattr(strategy, "_contract_op", None)
                if own is not None:
                    amp = own(record, Fo, Bo, C, g, off, sz, n_line, max_basis)
                    if amp is not None:
                        return amp
                else:
                    # this method's op *is* the strategy's contract (index);
                    # a raising call is an op failure, report it once
                    try:
                        return strategy.contract(
                            record,
                            Fo,
                            Bo,
                            C,
                            g,
                            off,
                            sz,
                            n_line=n_line,
                            max_basis=max_basis,
                        )
                    except Exception as exc:
                        _warn_op_failure(method, exc)
            # declined (op not built, block it cannot read) or failed: the
            # TF reference takes the *stored* tensors and expands them itself
            return Backend.functions[method](
                strategy,
                record,
                F,
                B,
                C,
                g,
                off,
                sz,
                n_line=n_line,
                max_basis=max_basis,
            )

        return fn


# ``None`` is the plain (unmerged) basis: ``_NoCompression.contract`` is the
# ``dense_factor_a``/``dense_factor_an`` pair, so the plain op model stays on
# the fused path.  The guards above hold for every entry: the op family takes
# real ``g`` in all of them, and the C++ shape checks reject ``nline < 1`` for
# the general, merge and basis-index ops alike.
CudaBackend.functions = {
    method: CudaBackend._contract(method)
    for method in (None, "index", "basis_index", "real_index")
}
register_backend(CudaBackend)


@register_amp_model("dense_factor_op")
class DenseFactorOpModel(DenseFactorModel):
    """Fused-contraction variant of :class:`DenseFactorModel`."""

    #: Backend order: the fused kernels first, the TF reference as the
    #: fallback (a machine without the build, or a block the op rejects).
    prefer = ("cuda", "tf")

    #: The coefficient build is 300 scalars and is much cheaper on CPU/XLA
    #: than as ~200 tiny GPU kernels; this pairing is the measured optimum.
    coeff_on_cpu = True

    #: ``(off, sz, n_line, max_basis)``, filled by ``init_params``
    _layout = None

    def init_params(self, name=""):
        # the base init builds the group's basis, or raises for an
        # unsupported term model, so it exists from here on
        super().init_params(name)
        # The fused backend needs the layout, which comes from the op
        # package.  Only its absence is tolerated: a model then runs the TF
        # reference (and ``needs_layout`` keeps the fused backend out of
        # ``prefer``) and this is the first moment the missing build matters,
        # so the warning is reported here rather than at import.  Errors from
        # a package that *is* importable are real bugs and propagate.
        if _op is None:
            _warn_no_cuda()
        self._layout = None if _op is None else self._op_layout()

    def _op_layout(self):
        """``(off, sz, n_line, max_basis)``, built once in ``init_params``.

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
        return (
            off,
            sz,
            structure["max_internal_lines"],
            max(sizes) if sizes else 0,
        )


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
