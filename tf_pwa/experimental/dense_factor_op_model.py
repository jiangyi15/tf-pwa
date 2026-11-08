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
from .dense_factor_op import dense_factor_a, dense_factor_an


@register_amp_model("dense_factor_op")
class DenseFactorOpModel(DenseFactorModel):
    """Fused-contraction variant of :class:`DenseFactorModel`."""

    #: The coefficient build is 300 scalars and is much cheaper on CPU/XLA
    #: than as ~200 tiny GPU kernels; this pairing is the measured optimum.
    coeff_on_cpu = True

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

        F = data.get("F", None)
        B = data.get("B", None)
        if F is None or B is None:
            F, B = self.build_dense_terms(data)

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
        F = tf.cast(F, dtype)
        B = tf.cast(B, dtype)
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
        if n_line == 1:
            amp = dense_factor_a(F, B, C, g, off, sz, max_basis=max_basis)
        else:
            amp = dense_factor_an(F, B, C, g, off, sz, max_basis=max_basis)
        # same contraction the dense path ends with, so a non-trivial
        # polarization density matrix is supported rather than rejected
        return dg.sum_with_polarization(amp)
