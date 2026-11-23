"""Amplitude models of the dense-factor plugin.

``DenseFactorModel`` is the reference contraction (``_dense_contract``);
``DenseFactorMergeModel`` adds the merge record so the stored basis can be
expanded on the way in.  Both register themselves on import.

The contraction is dispatched through :mod:`.backend` (the TF reference by
default); the strategies and their records come from :mod:`.compress`.
"""

from types import MappingProxyType

import numpy as np
import tensorflow as tf

from tf_pwa.amp.amp import BaseAmplitudeModel, register_amp_model

from .backend import (
    BACKENDS,
    get_op,
    get_prepare,
    pick_backend,
    register_backend,
    register_op,
)
from .compress import (
    _COMPRESSIONS,
    _compression_of,
    _unpack_terms,
    _warn_op_failure,
    basis_key,
    merge_record,
)

#: Every key ``get_layout`` provides; a backend reads the subset it needs.
#: ``off``/``sz`` are ``None`` on the dense models (no fused kernel layout).
LAYOUT_KEYS = (
    "off",
    "sz",
    "n_line",
    "max_basis",  # fused kernels
    "n_u",
    "sparse_index",
    "sparse_shape",  # TF reference
    "line_mask",
    "line_mask_all",
)


@register_amp_model("dense_factor")
class DenseFactorModel(BaseAmplitudeModel):
    #: backend order: the dense model is the reference and stays on TF
    prefer = ("tf",)

    def init_params(self, name=""):
        super().init_params(name)
        # the per-record static op inputs: a re-init drops stale entries (the
        # record identity check in ``_prepare_op`` covers a refit)
        self._prepared = {}
        dg = self.decay_group
        structure = dg.get_dense_structure()
        n_u = structure["n_u"]
        n_param = structure["n_param"]
        max_internal_lines = structure["max_internal_lines"]
        den_sizes = structure["den_sizes"]
        param_offset = structure["param_offset"]
        u_chain = structure["u_chain"]

        M1 = np.zeros((n_u, max_internal_lines, n_param), dtype=np.float64)
        line_mask = np.zeros((n_u, max_internal_lines), dtype=bool)

        for u, k in enumerate(u_chain):
            for l, (size, poff) in enumerate(
                zip(den_sizes[k], param_offset[k])
            ):
                M1[u, l, poff : poff + size] = 1.0
                line_mask[u, l] = True

        self.M1 = tf.constant(M1, dtype=tf.float64)
        coo = np.nonzero(M1.reshape(n_u * max_internal_lines, n_param))
        self.sparse_index = tf.constant(np.stack(coo, axis=1), dtype=tf.int64)
        self.sparse_shape = (n_u * max_internal_lines, n_param)
        self.M1T = tf.reshape(
            tf.transpose(self.M1, [2, 0, 1]),
            [n_param, n_u * max_internal_lines],
        )
        self.line_mask = tf.constant(line_mask)
        self.line_mask_all = bool(np.all(line_mask))

        self._layout = MappingProxyType(
            {
                "off": None,
                "sz": None,
                "n_line": max_internal_lines,
                "max_basis": 0,
                "n_u": n_u,
                "sparse_index": self.sparse_index,
                "sparse_shape": self.sparse_shape,
                "line_mask": self.line_mask,
                "line_mask_all": self.line_mask_all,
            }
        )

        self._coeff_fn = None
        if n_u > 0 and n_param > 0:

            def _build_coeff():
                return dg.build_dense_coeff(structure)

            try:
                self._coeff_fn = tf.function(_build_coeff)
            except Exception:
                self._coeff_fn = None

    def get_layout(self):
        """The layout this model's backends need, as a read-only mapping.

        Model-owned and built once in ``init_params`` -- eager tensors here are
        captured as constants by any graph that uses them, while a constant
        created during a trace belongs to that ``FuncGraph``.  Keys are
        ``LAYOUT_KEYS`` (``off``/``sz`` are ``None`` without a fused kernel
        layout).  It is a ``MappingProxyType``, so a backend that tries to
        write it raises instead of silently changing the model.
        """
        return self._layout

    def cp_transform(self):
        """``(frac, change)`` for the CP-conjugate amplitude, like get_amp3."""
        return self.decay_group.get_dense_structure()["cp"]

    def cp_amplitude(self, amp_cp):
        """Transpose and helicity-reverse the CP block, times ``frac``."""
        dg = self.decay_group
        structure = self.decay_group.get_dense_structure()
        f_shape = list(structure["f_shape"])
        n_axis = 1 + len(f_shape)
        frac, change = self.cp_transform()
        shape = tf.shape(amp_cp)
        out = tf.reshape(amp_cp, [shape[0]] + f_shape)
        out = tf.transpose(out, list(dg.get_swap_transpose(change, n_axis)))
        if dg.inverse_helicity_cp:
            out = tf.reverse(out, axis=list(range(1, n_axis)))
        out = tf.reshape(out, shape)
        return out * tf.cast(frac, amp_cp.dtype)

    def _get_coeff(self):
        """Return the numerator ``C`` and denominator ``g`` coefficients."""
        dg = self.decay_group
        structure = self.decay_group.get_dense_structure()
        C, g = None, None
        if getattr(self, "coeff_on_cpu", False):
            fn = self._coeff_cpu_fn()
            try:
                C, g = fn()
            except Exception:
                self.coeff_on_cpu = False
                C, g = None, None
        if C is None and self._coeff_fn is not None:
            try:
                C, g = self._coeff_fn()
            except Exception:
                self._coeff_fn = None
        if C is None:
            C, g = dg.build_dense_coeff(structure)
        return C, g

    def _coefficients(self, data):
        """Structure check, coefficient build and the used-chain mask.

        Returns ``(C, g, dtype)`` with ``C`` already masked.  The
        models that contract differently share this preamble: the product
        runs at the coefficient dtype, and ``_get_coeff`` does not read F or
        B, so the preprocessor may store them at lower precision.
        """
        dg = self.decay_group
        structure = self.decay_group.get_dense_structure()
        C, g = self._get_coeff()
        dtype = C.dtype
        used = set(dg.chains_idx)
        mask = tf.constant(
            [1.0 if k in used else 0.0 for k in structure["u_chain"]],
            dtype=dtype,
        )
        return C * mask, g, dtype

    def pdf(self, data):
        C, g, dtype = self._coefficients(data)

        def block(d):
            F, B = d.get("F", None), d.get("B", None)
            if F is None or B is None:
                F, B = self.decay_group.build_dense_terms(d)
            return self._amplitude(F, B, C, g, dtype)

        return self.sum_amplitude(data, block)

    def sum_amplitude(self, data, block):
        """Total amplitude: the swap sum plus the CP-conjugate block.

        ``block(d)`` contracts one cached dict; the identical-particle sum
        and the CP term are the same for the dense and the fused model.
        """
        dg = self.decay_group

        def sum_swaps(d):
            amp = block(d)
            for key, sub in (d.get("id_swap") or {}).items():
                if sub.get("F") is None or sub.get("B") is None:
                    raise ValueError(
                        "identical-particle block {} has no F/B; the "
                        "dense_factor preprocessor must build it".format(key)
                    )
                amp = amp + self.decay_group._swap_amplitude(block(sub), key)
            return amp

        amp = sum_swaps(data)
        cp = data.get("cp_swap", None)
        if cp is not None and dg.cp_particles:
            # the conjugated kinematics carry their own identical-particle
            # swaps, so it is the same sum on the cp_swap dict
            amp = amp + self.cp_amplitude(sum_swaps(cp))
        return dg.sum_with_polarization(amp)

    def expand_block(self, F, B):
        """Hook for models that store the basis merged.

        The contraction expands through the strategy (``strategy.expand``),
        so this hook is the public/test entry point and not part of
        ``_dense_contract``: overriding it does not change the fused path.
        """
        return F, B

    def _prepare_op(self, name, method, strategy, record, layout):
        """The static op inputs for (backend, method), cached per record.

        Keyed by the backend as well, so a prepared computed for one backend
        can never be served to another.  The record identity is the staleness
        check: a refit produces a new record and rebuilds.  A prepare that
        raises is reported once and treated as absent, so the op infers as
        before instead of failing the evaluation.
        """
        fn = get_prepare(name, method)
        if fn is None:
            return None
        hit = self._prepared.get((name, method))
        if hit is not None and hit[0] is record:
            return hit[1]
        try:
            prepared = fn(strategy, record, layout)
        except Exception as exc:
            _warn_op_failure("{}/{}".format(name, method), exc)
            prepared = None
        self._prepared[(name, method)] = (record, prepared)
        return prepared

    def _amplitude(self, F, B, C, g, dtype):
        """``A[e, p] = sum_u F[e, p, u] C[u] / den[u, e]`` for one block.

        The contraction itself is the backend table's: ``prefer`` names the
        order this model wants (``("tf",)`` here, ``("cuda", "tf")`` for the
        fused models), and the first backend that supports the method wins.
        See :mod:`tf_pwa.experimental.dense_factor.backend`.
        """
        record = self.merge if hasattr(self, "merge_id") else None
        method = record["method"] if record else None
        strategy = _compression_of(record)
        layout = self.get_layout()
        # every backend gets the same arguments; a fused entry that cannot
        # use this model's layout (``layout["off"] is None``) declines and
        # the reference runs for that call
        name = pick_backend(method, self.prefer)
        # the backend's static inputs are derived once per record, not per
        # call: the ops' template bounds are then plain ints
        prepared = self._prepare_op(name, method, strategy, record, layout)
        return get_op(name, method)(
            strategy, record, F, B, C, g, layout, prepared
        )

    def _coeff_cpu_fn(self):
        """Opt-in: build ``C``/``g`` and their VJP on CPU under XLA.

        The device scope has to wrap the **tape**.  Putting it only inside
        the coefficient builder leaves the generated backward on the GPU,
        which is why the first attempt saw forward-only gains.
        """
        cached = getattr(self, "_coeff_cpu_fn_cached", None)
        if cached is not None:
            return cached
        dg = self.decay_group
        structure = self.decay_group.get_dense_structure()

        @tf.function(jit_compile=True)
        def _fwd():
            with tf.device("/CPU:0"):
                return dg.build_dense_coeff(structure)

        @tf.function(jit_compile=True)
        def _vjp(dc, dgv, vs):
            with tf.device("/CPU:0"):
                with tf.GradientTape() as tape:
                    c, g = dg.build_dense_coeff(structure)
                return tape.gradient([c, g], vs, output_gradients=[dc, dgv])

        @tf.custom_gradient
        def fn():
            c, g = _fwd()

            def grad(dc, dgv, variables=None):
                dc = tf.zeros_like(c) if dc is None else dc
                dgv = tf.zeros_like(g) if dgv is None else dgv
                # ``variables`` is the list the tape used, and TF wants the
                # gradients back in that order: differentiate exactly it, so
                # no id map (and no assumption that the model's trainable set
                # never changes -- ``set_fix`` may unfix a parameter later)
                if variables is None:
                    variables = list(self.vm.trainable_variables)
                return (), _vjp(dc, dgv, list(variables))

            return (c, g), grad

        self._coeff_cpu_fn_cached = fn
        return fn


def _dense_contract(strategy, record, F, B, C, g, layout, prepared=None):
    """The TF reference contraction: expand the stored basis, then contract.

    ``A[e, p] = sum_u F[e, p, u] C[u] / den[u, e]`` over one block.  The
    strategy puts the stored basis back to full width (identity for a plain
    basis), the denominator is one sparse matmul driven by the model-owned
    ``sparse_index``/``line_mask`` in ``layout``, and the projection is a
    matvec.  ``off``/``sz`` are the fused kernels' keys and are ignored here:
    the dense body rebuilds the same denominator from ``sparse_index`` (a
    constant per model) instead of a per-call gather.  ``prepared`` is the
    fused backend's static-input dict; the reference derives everything it
    needs from the tensors, so it is ignored.
    """
    dtype = C.dtype
    F, B = strategy.expand(record, F, B)
    F = _unpack_terms(F, dtype)
    B = _unpack_terms(B, dtype)
    n_u = layout["n_u"]
    sparse_index = layout["sparse_index"]
    sparse_shape = layout["sparse_shape"]
    line_mask = layout["line_mask"]
    line_mask_all = layout["line_mask_all"]
    n_line = layout["n_line"]
    if n_line == 0:
        D = tf.ones((tf.shape(F)[0], n_u), dtype=dtype)
    else:
        values = tf.gather(tf.cast(g, B.dtype), sparse_index[:, 1])
        sp = tf.sparse.SparseTensor(sparse_index, values, sparse_shape)
        # B is stored event-first; the sparse matmul consumes it
        # param-major, so the fallback pays the transpose.
        D = tf.transpose(tf.sparse.sparse_dense_matmul(sp, tf.transpose(B)))
        if n_line == 1 and line_mask_all:
            D = tf.reshape(D, [tf.shape(F)[0], n_u])
        else:
            D = tf.reshape(D, [tf.shape(F)[0], n_u, n_line])
            D = tf.where(line_mask[None, :, :], D, tf.ones_like(D))
            D = tf.reduce_prod(D, axis=-1)

    W = tf.math.divide_no_nan(C[None, :], D)
    return tf.linalg.matvec(F, W)


#: The reference backend: the same contraction for every recorded method.
#: It is the fallback for any backend that declines or fails, so its table
#: must cover everything the preprocessor can record.
register_backend("tf")
for _method in _COMPRESSIONS:
    register_op("tf", _method)(_dense_contract)
del _method

_missing = set(_COMPRESSIONS) - set(BACKENDS["tf"]["functions"])
if _missing:
    raise RuntimeError(
        "the TF reference is missing contractions for {}".format(
            sorted(_missing)
        )
    )
if not all(callable(get_op("tf", method)) for method in _COMPRESSIONS):
    raise RuntimeError("the reference table must hold callables")
del _missing


@register_amp_model("dense_factor_merge")
class DenseFactorMergeModel(DenseFactorModel):
    """``dense_factor`` amplitude that expands the merged basis blocks.

    The preprocessor reduces what is stored; this model expands it back with
    its map record.  A model that does not expand (a plain ``dense_factor``)
    paired with the merge preprocessor is left with the reduced widths and
    fails at evaluation instead of being rejected here.

    ``merge_key`` names the shared record; it must match the preprocessor's.
    """

    def __init__(self, *args, merge_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_id = merge_key or basis_key(self.decay_group)

    @property
    def merge(self):
        """The compression record of this basis, in the shared space."""
        return merge_record(self)

    def expand_block(self, F, B):
        """Expand the merged blocks through this model's strategy.

        The contraction calls ``strategy.expand`` itself, so this hook is the
        public/test entry point; overriding it does not change
        ``_dense_contract``.
        """
        c = self.merge
        if not c:  # nothing was reduced
            return F, B
        return _compression_of(c).expand(c, F, B)
