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

from .dense_factor import DenseFactorModel, _expand_block
from .dense_factor_op import dense_factor_a, dense_factor_an


@register_amp_model("dense_factor_op")
class DenseFactorOpModel(DenseFactorModel):
    """Fused-contraction variant of :class:`DenseFactorModel`."""

    #: The coefficient build is 300 scalars and is much cheaper on CPU/XLA
    #: than as ~200 tiny GPU kernels; this pairing is the measured optimum.
    coeff_on_cpu = True

    def expand_block(self, F, B):
        """Hook for subclasses that store the basis merged."""
        return F, B

    def _op_layout(self):
        structure = self.structure
        metas = structure["chain_meta"]
        n_line = structure["max_internal_lines"]
        off, sz = [], []
        for k in structure["u_chain"]:
            meta = metas[k]
            sizes, offsets = meta["den_sizes"], meta["param_offset"]
            for l in range(n_line):
                if l < len(sizes):
                    off.append(offsets[l])
                    sz.append(sizes[l])
                else:  # absent line: multiply the product by 1
                    off.append(0)
                    sz.append(0)
        return (
            tf.constant(off, dtype=tf.int32),
            tf.constant(sz, dtype=tf.int32),
            n_line,
            int(max(sz)) if sz else 0,
        )

    def pdf(self, data):
        dg = self.decay_group
        structure = self.structure
        if structure is None:
            dg.get_particle_terms(data)
            raise NotImplementedError("dense_factor_op is not initialized")

        C, g = self._get_coeff()
        # F and B may be stored at lower precision; the kernels take the
        # coefficient dtype, and casting before the op keeps the gradients
        # with respect to C and g in that same precision.  C is complex by
        # construction while g only happens to be.
        if g.dtype.is_complex:
            raise NotImplementedError(
                "dense_factor_op needs real denominator coefficients; "
                "use the dense_factor model for this one"
            )
        dtype = C.dtype
        used = set(dg.chains_idx)
        mask = tf.constant(
            [1.0 if k in used else 0.0 for k in structure["u_chain"]],
            dtype=C.dtype,
        )
        C = C * mask

        off, sz, n_line, max_basis = self._op_layout()
        if n_line < 1:
            raise NotImplementedError(
                "dense_factor_op needs at least one denominator line"
            )

        # F is [ne, np, nu]; the op returns A[e, p] with one shared
        # denominator per chain and sum_with_polarization contracts it.
        def block(d):
            f_b, b_b = d.get("F", None), d.get("B", None)
            if f_b is None or b_b is None:
                f_b, b_b = self.build_dense_terms(d)
            # merged storage expands back here, so the op is unchanged
            f_b, b_b = self.expand_block(f_b, b_b)
            # the preprocessor may store F and B at lower precision
            f_b = tf.cast(f_b, dtype)
            b_b = tf.cast(b_b, dtype)
            if n_line == 1:
                return dense_factor_a(
                    f_b, b_b, C, g, off, sz, max_basis=max_basis
                )
            return dense_factor_an(
                f_b, b_b, C, g, off, sz, max_basis=max_basis
            )

        # the swap sum and the CP term are shared with the dense model
        return self.sum_amplitude(data, block)


@register_amp_model("dense_factor_op_merge")
class DenseFactorOpMergeModel(DenseFactorOpModel):
    """Fused contraction whose basis blocks are stored merged.

    Config::

        data:
          amp_model: dense_factor_op_merge
          preprocessor:
            dense_factor_merge: {}

    Only ``expand_block`` is overridden, so no multiple inheritance is
    involved: the fused ``pdf`` is the plain one, and the merged tensors are
    expanded just before the op call.
    """

    def expand_block(self, F, B):
        return _expand_block(self.structure, F, B)
