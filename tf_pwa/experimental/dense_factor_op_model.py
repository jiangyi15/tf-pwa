"""``dense_factor`` amplitude model backed by the fused custom op.

Registered as ``amp_model: dense_factor_op``.  It reuses the dense
preprocessor (event-major ``F``/``B``, ``structure``, coefficient build)
and replaces only the contraction

.. math::

    W_u = \\frac{C_u}{\\sum_j g_j B_{u,j}},
    \\qquad A = \\sum_u F_u W_u,
    \\qquad P = |A|^2

with :func:`tf_pwa.experimental.dense_factor_op.dense_factor_p`.

Chains with one denominator line use :func:`dense_factor_p`; every other
line count goes through :func:`dense_factor_pn`, which is not bounded by the
number of lines.  A non-trivial polarization density matrix is still
rejected, because it is a different contraction rather than a layout.

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
from .dense_factor_op import dense_factor_p, dense_factor_pn


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

        if dg.polarization != "none":
            raise NotImplementedError(
                "dense_factor_op supports polarization='none' only, got %r"
                % (dg.polarization,)
            )

        F = data.get("F", None)
        B = data.get("B", None)
        if F is None or B is None:
            F, B = self.build_dense_terms(data)

        C, g = self._get_coeff()
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
        # F is [ne, n_pol, n_u]; the kernels sum |A_p|^2 over the middle
        # axis with one shared denominator.
        f = tf.cast(F, g.dtype)
        b = tf.cast(B, g.dtype)
        if n_line == 1:
            return dense_factor_p(f, b, C, g, off, sz, max_basis=max_basis)
        return dense_factor_pn(f, b, C, g, off, sz, max_basis=max_basis)
