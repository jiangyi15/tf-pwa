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

from .dense_factor import DenseFactorModel
from .dense_factor_op import chain_layout, dense_factor_a, dense_factor_an


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
        # the preprocessor may store F and B at lower precision
        F = tf.cast(F, dtype)
        B = tf.cast(B, dtype)
        maps = self.op_maps()
        map_u, map_i = maps if maps is not None else (None, None)
        if n_line == 1:
            return dense_factor_a(
                F,
                B,
                C,
                g,
                off,
                sz,
                max_basis=max_basis,
                map_u=map_u,
                map_i=map_i,
            )
        return dense_factor_an(
            F,
            B,
            C,
            g,
            off,
            sz,
            max_basis=max_basis,
            map_u=map_u,
            map_i=map_i,
        )

    def op_maps(self):
        """Merge index ``(map_u, map_i)`` for the op, or ``None``."""
        return None


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
    full-width copy of the basis.
    """

    #: ``_find_merge`` record, set by the preprocessor
    merge = None

    def op_maps(self):
        merge = self.merge
        if isinstance(merge, dict):
            return merge["map_u"], merge["map_i"]
        return None
