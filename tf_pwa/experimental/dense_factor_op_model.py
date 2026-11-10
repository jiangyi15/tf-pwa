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

The original ``dense_factor`` model is untouched; this module is opt-in
and only becomes usable when the custom op can be built on the machine.

Config::

    data:
      amp_model: dense_factor_op
      preprocessor:
        dense_factor:
          force_cpu: true
"""

import tensorflow as tf

from tf_pwa.amp.amp import register_amp_model

from .dense_factor import (
    DenseFactorModel,
    _compression_of,
    _unpack_stored,
    basis_key,
    merge_record,
)
from .dense_factor_op import chain_layout


@register_amp_model("dense_factor_op")
class DenseFactorOpModel(DenseFactorModel):
    """Fused-contraction variant of :class:`DenseFactorModel`."""

    #: The coefficient build is 300 scalars and is much cheaper on CPU/XLA
    #: than as ~200 tiny GPU kernels; this pairing is the measured optimum.
    coeff_on_cpu = True

    #: ``(off, sz, n_line, max_basis)``, filled by ``init_params``
    _layout = None

    def init_params(self, name=""):
        # the base init builds the group's basis, or raises for an
        # unsupported term model, so it exists from here on
        super().init_params(name)
        self._layout = self._op_layout()

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

    def _amplitude(self, F, B, C, g, dtype):
        """``A`` through the fused op; the rest of ``pdf`` is shared."""
        if g.dtype.is_complex:
            raise NotImplementedError(
                "dense_factor_op needs real denominator coefficients; "
                "use the dense_factor model for this one"
            )
        off, sz, n_line, max_basis = self._layout
        if n_line < 1:
            raise NotImplementedError(
                "dense_factor_op needs at least one denominator line"
            )
        # the strategy owns the stored representation and the call
        c = self.merge if hasattr(self, "merge_id") else None
        # the preprocessor may also store F and B at lower precision; a
        # split record keeps them real, which the dtype alone cannot say
        F = _unpack_stored(F, c, dtype)
        B = _unpack_stored(B, c, dtype)
        return _compression_of(c).contract(
            c, F, B, C, g, off, sz, n_line=n_line, max_basis=max_basis
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
