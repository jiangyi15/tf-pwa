"""``dense_factor`` amplitude model backed by the fused custom op.

Registered as ``amp_model: dense_factor_op``.  It reuses the dense
preprocessor (event-major ``F``/``B``, ``structure``, coefficient build)
and replaces only the contraction

    W = C / (g . B);  A = F . W;  P = |A|^2

with :func:`tf_pwa.experimental.dense_factor_op.dense_factor_p`.

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
from .dense_factor_op import dense_factor_p


@register_amp_model("dense_factor_op")
class DenseFactorOpModel(DenseFactorModel):
    """Fused-contraction variant of :class:`DenseFactorModel`."""

    #: The coefficient build is 300 scalars and is much cheaper on CPU/XLA
    #: than as ~200 tiny GPU kernels; this pairing is the measured optimum.
    coeff_on_cpu = True

    def _op_layout(self):
        structure = self.structure
        n_line = structure["max_internal_lines"]
        if n_line != 1:
            raise NotImplementedError(
                "dense_factor_op supports max_internal_lines == 1, "
                "got %d" % n_line
            )
        metas = structure["chain_meta"]
        off = [
            metas[k]["param_offset"][0] for k in structure["u_chain"]
        ]
        sz = [metas[k]["den_sizes"][0] for k in structure["u_chain"]]
        return (
            tf.constant(off, dtype=tf.int32),
            tf.constant(sz, dtype=tf.int32),
            int(max(sz)),
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
            [
                1.0 if k in used else 0.0
                for k in structure["u_chain"]
            ],
            dtype=C.dtype,
        )
        C = C * mask

        off, sz, max_basis = self._op_layout()
        # F is [ne, n_pol, n_u]; the kernel sums |A_p|^2 over the middle
        # axis with one shared denominator.
        return dense_factor_p(
            tf.cast(F, g.dtype), tf.cast(B, g.dtype), C, g, off, sz,
            max_basis=max_basis,
        )
