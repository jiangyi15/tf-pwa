"""Dense factorized amplitude model plugin.

Importing this module attaches the dense term interface to the original
tf-pwa classes, so the upstream source files stay untouched::

    import tf_pwa.experimental.dense_factor

The amplitude is written as

.. math::

    A_{ij} = \\sum_u C_u \\frac{F_{u,ij}(x)}
             {\\prod_l \\sum_{m,n} M1_{u,l,m} g_m M2_{m,n} B_{l,n}(x)}

The decay group builds the data-only ``F`` and ``B`` tensors, together
with the structural ``M2``, in ``get_dense_structure`` on first use.  The amplitude
model builds its contraction layout ``M1`` in ``init_params``.  All
coefficients are rebuilt on every call.

This module must be imported before the amplitude is created, because the
``dense_factor`` preprocessor and amplitude model are registered on import.
"""

import contextlib
import itertools
import warnings
import weakref

import numpy as np
import tensorflow as tf

import tf_pwa.amp.interpolation  # noqa: F401  (register spline models)
from tf_pwa.amp.amp import BaseAmplitudeModel, register_amp_model
from tf_pwa.amp.preprocess import (
    BasePreProcessor,
    CachedAnglePreProcessor,
    register_preprocessor,
)
from tf_pwa.cal_angle import cal_angle_from_momentum
from tf_pwa.config import get_config, regist_config
from tf_pwa.data import data_merge, split_generator
from tf_pwa.phasespace import PhaseSpaceGenerator

from .backend import (
    BACKENDS,
    Backend,
    pick_backend,
)

# The compression strategies live in a sibling module.  It is a leaf (it never
# imports this one), and every name is re-exported so the existing import paths
# (`dense_factor_op_model`, the tests) keep working.
from .compress import (  # noqa: F401
    _COMPRESSION_DEFAULTS,
    _COMPRESSION_SOURCES,
    _COMPRESSIONS,
    _OP_FAILURES,
    _TERMS,
    _axis_groups,
    _BasisIndexCompression,
    _cast_like,
    _Compression,
    _compression_of,
    _expand_block,
    _expand_svd,
    _find_merge,
    _IndexCompression,
    _join_complex,
    _NoCompression,
    _pack_complex16,
    _pack_like,
    _project,
    _real_dtype,
    _RealIndexCompression,
    _RealSvdCompression,
    _reconstruct,
    _sparse_basis,
    _split_complex,
    _SvdCompression,
    _unpack_real,
    _unpack_stored,
    _unpack_terms,
    _warn_op_failure,
)
from .decays import (  # noqa: F401
    _AmpDecayBasis,
    _DecayChainBasis,
    _DecayGroupBasis,
    _HelicityBasis,
)

# The mixins live in sibling modules so this file keeps the preprocessor and
# the model; the strategies are in ``compress.py``.  Importing the mixins here
# runs their decorators, which
# is what attaches the dense term/basis interface on import of this module.
# They are re-exported so every existing import path keeps working.
from .interface import (  # noqa: F401
    _BASIS_TARGETS,
    AmpDecay,
    DecayChain,
    DecayGroup,
    HelicityDecay,
    _attach,
    _guard,
    basis,
    get_particle_model,
    rename_data_dict,
    terms,
)
from .particles import (  # noqa: F401
    _BWTerms,
    _DefaultTerms,
    _FlatteTerms,
    _GSTerms,
    _OneTerms,
    _SplineTerms,
)

# ---------------------------------------------------------------------------
# Amplitude model
# ---------------------------------------------------------------------------


@register_preprocessor("dense_factor")
class DenseFactorPreProcessor(CachedAnglePreProcessor):
    """Precompute the dense ``F`` and ``B`` tensors.

    Config::

        data:
          preprocessor:
            dense_factor:
              force_cpu: true
              drop_decay_particle: false
              store_dtype: complex128   # complex64 | complex32

    With ``force_cpu``, the angle pre-processing and the ``F``/``B``
    construction run on CPU.  Only the final ``F`` and ``B`` tensors are
    copied to GPU, using ``tf.identity``.

    With ``drop_decay_particle``, the ``"decay"`` and ``"particle"`` entries
    are removed from the returned dict once ``F`` and ``B`` exist.  ``pdf``
    reads only those two, and a lazy cache stores the whole returned dict,
    so dropping the angle tree makes the cache smaller.  It is opt-in
    because everything that indexes the angles directly
    (``partial_weight``, ``plot_partial_wave``, the fit fractions) needs
    them.

    The same rule applies to every block the preprocessor builds for
    ``id_swap`` / ``cp_swap``, so the option governs the whole returned
    dict.

    What it always produces is the angle data (``"decay"``, ``"particle"``),
    which comes from the decay group and is what every consumer reads: the
    AmplitudeModel, ``partial_weight``, plotting and the fit fractions.
    ``F``/``B`` are an additional representation of the same event, built
    only when a dense structure is available; without one the dict keeps
    the angles and is returned untouched, so a plotting-only or non-dense
    workflow is unaffected.

    ``build_batch`` builds the event axis in chunks: the input is sliced, the
    angle transform and the ``F``/``B`` build run on each slice, and the
    processed chunks are merged.  A merge preprocessor therefore never holds
    the full-width basis as one tensor (the peak is one chunk's terms plus the
    reduced result).  A chunk's angle leaves are released before the next
    slice only when ``drop_decay_particle`` lets the model give them up (and
    the input was the raw momentum dict, not an already-cal-angled one);
    otherwise they are kept and merged, because they are part of the result.
    ``None`` (the default) does one pass.  With ``lazy_call`` the outer
    batching already runs per batch, so the option is rejected there.  It must
    be at least ``merge_sample``, and the ``extra`` values must be event-major
    arrays so that they can be sliced.

    With ``method: svd`` the first chunk is the fit sample: at a real batch
    size its rank is ``np*nu``, the same as the whole set's, so the chunked
    and one-shot bases agree.  A deliberately truncated ``svd_rank`` or
    ``merge_rtol`` is fitted on that first chunk, so it can differ from a
    one-shot fit; a chunk whose rank only reaches its own row count raises
    instead of silently truncating the later ones.

    ``store_dtype`` picks the precision the two tensors are cached in:
    ``complex128`` (default), ``complex64`` (about 1e-7 relative) or
    ``complex32``.  For the fused ``basis_index``/``real_index`` ops the
    stored block reaches the kernel in its own dtype (float64, float32 or
    fp16) and is widened per value inside the kernel, so nothing is converted
    back before the op; the remaining strategies unpack to the coefficient
    dtype before any product.  Either way the arithmetic is complex128/float64.

    Accuracy of ``complex32`` is not one number: the *packed* complex form
    (two fp16 halves per value) sits at about 5e-4, while a *real* stored
    block in fp16 (``real_index``, and the sparse basis of ``basis_index``)
    is the plain fp16 floor amplified by whatever cancellation the basis
    does, measured from about 1e-3 to about 2e-2 depending on the model
    (4e-3 on a 20k CP sample, 1.8e-2 in the worst case seen there).  Treat
    ``store_dtype: complex32`` as an opt-in, not a 5e-4 guarantee.
    """

    def __init__(
        self,
        *args,
        force_cpu=False,
        drop_decay_particle=False,
        store_dtype="complex128",
        build_batch=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if store_dtype not in ("complex128", "complex64", "complex32"):
            raise ValueError(
                "store_dtype must be complex128, complex64 or complex32 "
                "(fp16 halves), got {!r}".format(store_dtype)
            )
        # the basis belongs to the group, so the data carries F/B even when
        # the amplitude model is not a dense one (the default model reads
        # only the angles, but plots, checks and the merge do use them)
        self.decay_group.get_dense_structure()
        self.force_cpu = force_cpu
        self.drop_decay_particle = drop_decay_particle
        self.store_dtype = store_dtype
        #: chunk the event axis while building F/B; None means one call
        self.build_batch = None if build_batch is None else int(build_batch)
        if self.build_batch is not None and self.build_batch < 1:
            raise ValueError("build_batch must be >= 1 or null")
        if self.build_batch is not None:
            config = getattr(self.root_config, "config", None) or {}
            if config.get("data", {}).get("lazy_call"):
                raise ValueError(
                    "build_batch cannot be used with lazy_call: the outer "
                    "batching already runs per batch and the batch axis is "
                    "dynamic"
                )
        self.gpu_device = None
        if tf.config.list_physical_devices("GPU"):
            self.gpu_device = "/GPU:0"

    def _build_terms(self, d):
        """Build ``F``/``B`` for one dict; the caller sets the chunking."""
        return self.decay_group.build_dense_terms(d)

    def call(self, x, **kwargs):
        if self.force_cpu:
            ctx = tf.device("/CPU:0")
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            if self.build_batch is None:
                x = BasePreProcessor.call(self, x, **kwargs)
                self._process(x)
            else:
                # stream the input: a chunk's angles are computed and used,
                # and with drop_decay_particle released before the next slice,
                # so neither the basis nor those angles are alive twice;
                # without it the angles are part of the result and are kept
                chunks = []
                for chunk in split_generator(x, self.build_batch):
                    chunk = BasePreProcessor.call(self, chunk, **kwargs)
                    self._process(chunk)
                    self._strip(chunk)
                    chunks.append(chunk)
                if len(chunks) == 1:
                    x = chunks[0]
                else:
                    x = data_merge(*chunks)
            blocks = self._blocks(x)
            # every block follows the same two rules as the base dict
            x["F"], x["B"] = self._cast(x["F"], x["B"])
            for d in blocks:
                d["F"], d["B"] = self._cast(d["F"], d["B"])

        x["F"], x["B"] = self._move(x["F"], x["B"])
        for d in blocks:
            d["F"], d["B"] = self._move(d["F"], d["B"])
        self._strip(x)
        return x

    def _blocks(self, d):
        """The ``cp_swap`` block and every nested ``id_swap`` block."""
        blocks = list((d.get("id_swap") or {}).values())
        cp = d.get("cp_swap", None)
        if cp is not None:
            blocks += [cp] + list((cp.get("id_swap") or {}).values())
        return blocks

    def _process(self, d):
        """Build ``F``/``B`` for one dict and compress it."""
        d["F"], d["B"] = self._build_terms(d)
        self._fill_blocks(d)
        cp = d.get("cp_swap", None)
        blocks = list((d.get("id_swap") or {}).values())
        if cp is not None:
            cp["F"], cp["B"] = self._build_terms(cp)
            self._fill_blocks(cp)
            self._strip(cp)
            blocks += [cp] + list((cp.get("id_swap") or {}).values())
        # a subclass may reduce the blocks here, while they are still on
        # the CPU at full precision
        self._after_blocks(d, blocks)

    def _after_blocks(self, x, blocks):
        """Hook for subclasses, called before the cast and the GPU move."""

    def _cast(self, F, B):
        """Lower the stored precision, inside the CPU scope.

        A split record stores real coefficients, so those are cast to the
        real dtype of the same width (float32 for complex64, float16 for
        complex32) instead of being turned complex.
        """
        if self.store_dtype == "complex64":
            return (
                _cast_like(F, tf.complex64, tf.float32),
                _cast_like(B, tf.complex64, tf.float32),
            )
        if self.store_dtype == "complex32":
            return _pack_like(F), _pack_like(B)
        return F, B

    def _move(self, F, B):
        """Move the final tensors to the GPU when ``force_cpu`` asks for it."""
        if self.force_cpu and self.gpu_device is not None:
            return tf.identity(F), tf.identity(B)
        return F, B

    def _strip(self, d):
        """Drop the angle tree when the option asks for it.

        Only the model that reads ``F``/``B`` can give the angles up; a
        wrapper around it keeps them for its other components.
        """
        if not self.drop_decay_particle:
            return
        if not isinstance(self.amp, DenseFactorModel):
            return
        d.pop("particle", None)
        d.pop("decay", None)

    def _fill_blocks(self, d):
        """Build ``F``/``B`` for every ``id_swap`` block of ``d``, in place.

        Applies the parity factor only: the particle-axis transpose belongs to
        the amplitude (``_swap_amplitude``), so every block keeps the shared
        column convention the compression record is fitted in.  ``cp_swap`` is handled
        by the caller, which calls this again on that dict.
        """
        for key, sub in (d.get("id_swap") or {}).items():
            Fs, Bs = self._build_terms({**d, **sub})
            sub["F"] = self.decay_group.swap_terms(Fs, key)
            sub["B"] = Bs
            self._strip(sub)


#: name of the shared record store in the config registry
MERGE_RECORDS = "dense_factor_merge_records"

if get_config(MERGE_RECORDS, None) is None:
    regist_config(MERGE_RECORDS, {})

_MERGE_KEY_COUNTER = itertools.count()


def basis_key(group):
    """Fallback key: one per decay group, shared by its preprocessor/models.

    It is stored on the group, so a copy of the group keeps it and every
    model built on that basis (including the sub-models of a ``ProdPDF``,
    which are not what ``get_amplitude`` returns) finds the same record.
    Pass ``merge_key`` to the preprocessor and the model to choose it.
    """
    key = getattr(group, "_dense_factor_merge_key", None)
    if key is None:
        key = "dense_factor_merge:{}".format(next(_MERGE_KEY_COUNTER))
        group._dense_factor_merge_key = key
        # nothing can look the record up once the basis is gone; an explicit
        # merge_key is the caller's to clear.  A copied group keeps the key
        # but not this finalizer, so it refits after the original dies.
        weakref.finalize(group, set_merge_record, key, None)
    return key


def set_merge_record(key, record):
    """Publish (or clear, with ``None``) the record under one key."""
    records = get_config(MERGE_RECORDS)
    if record is None:
        records.pop(key, None)
    else:
        records[key] = record


def get_merge_record(key):
    """The record for one key: ``None`` if none, ``False`` if disabled."""
    return get_config(MERGE_RECORDS).get(key, None)


def merge_record(amp):
    """The compression record of a model's basis, found by its key.

    Read from the shared space instead of an attribute set by the
    preprocessor, so a model reached through a composed PDF (which need not
    be the object ``get_amplitude`` returned) still finds it.
    """
    return get_merge_record(amp.merge_id)


@register_amp_model("dense_factor")
class DenseFactorModel(BaseAmplitudeModel):
    #: backend order: the dense model is the reference and stays on TF
    prefer = ("tf",)

    def init_params(self, name=""):
        super().init_params(name)
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

        self._coeff_fn = None
        if n_u > 0 and n_param > 0:

            def _build_coeff():
                return dg.build_dense_coeff(structure)

            try:
                self._coeff_fn = tf.function(_build_coeff)
            except Exception:
                self._coeff_fn = None

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
        # the TF reference reads the model's cached sparse layout from here
        strategy.model = self
        off, sz, n_line, max_basis = self._backend_layout()
        # a backend that needs the chain layout is skipped when there is none
        # (the normal case for the dense models); the reference rebuilds the
        # denominator from its own structure instead
        prefer = tuple(
            name
            for name in self.prefer
            if not (
                getattr(BACKENDS.get(name), "needs_layout", False)
                and off is None
            )
        )
        backend = pick_backend(method, record, prefer)
        return backend.functions[method](
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

    def _backend_layout(self):
        """``(off, sz, n_line, max_basis)`` for the fused backends.

        The op models fill ``_layout`` in ``init_params``; the dense models
        have none, and the TF reference ignores the two arrays (it rebuilds
        the denominator from ``sparse_index``).
        """
        layout = getattr(self, "_layout", None)
        if layout is not None:
            return layout
        structure = self.decay_group.get_dense_structure()
        return None, None, structure["max_internal_lines"], 0

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
        variables = list(self.vm.trainable_variables)

        @tf.function(jit_compile=True)
        def _fwd():
            with tf.device("/CPU:0"):
                return dg.build_dense_coeff(structure)

        @tf.function(jit_compile=True)
        def _vjp(dc, dgv):
            with tf.device("/CPU:0"):
                with tf.GradientTape() as tape:
                    c, g = dg.build_dense_coeff(structure)
                return tape.gradient(
                    [c, g], variables, output_gradients=[dc, dgv]
                )

        @tf.custom_gradient
        def fn():
            c, g = _fwd()

            def grad(dc, dgv, variables=None):
                dc = tf.zeros_like(c) if dc is None else dc
                dgv = tf.zeros_like(g) if dgv is None else dgv
                grads = _vjp(dc, dgv)
                if variables is None:
                    return (), grads
                by_id = {
                    id(v): x
                    for v, x in zip(self.vm.trainable_variables, grads)
                }
                return (), [by_id[id(v)] for v in variables]

            return (c, g), grad

        self._coeff_cpu_fn_cached = fn
        return fn


def _dense_contract(
    strategy, record, F, B, C, g, off, sz, *, n_line, max_basis
):
    """The TF reference contraction: expand the stored basis, then contract.

    ``A[e, p] = sum_u F[e, p, u] C[u] / den[u, e]`` over one block.  The
    strategy puts the stored basis back to full width (identity for a plain
    basis), the denominator is one sparse matmul driven by the model's cached
    ``sparse_index``/``line_mask``, and the projection is a matvec.

    ``off``/``sz``/``max_basis`` are the fused backends' layout; the dense
    body rebuilds the same denominator from ``sparse_index`` (a constant per
    model) instead of a per-call gather, so they are accepted and ignored
    here.  ``strategy.model`` carries the model whose layout this is (see
    ``_Compression.model``).
    """
    model = strategy.model
    dtype = C.dtype
    F, B = strategy.expand(record, F, B)
    F = _unpack_terms(F, dtype)
    B = _unpack_terms(B, dtype)
    n_u = model.decay_group.get_dense_structure()["n_u"]
    if n_line == 0:
        D = tf.ones((tf.shape(F)[0], n_u), dtype=dtype)
    else:
        values = tf.gather(tf.cast(g, B.dtype), model.sparse_index[:, 1])
        sp = tf.sparse.SparseTensor(
            model.sparse_index, values, model.sparse_shape
        )
        # B is stored event-first; the sparse matmul consumes it
        # param-major, so the fallback pays the transpose.
        D = tf.transpose(tf.sparse.sparse_dense_matmul(sp, tf.transpose(B)))
        if n_line == 1 and model.line_mask_all:
            D = tf.reshape(D, [tf.shape(F)[0], n_u])
        else:
            D = tf.reshape(D, [tf.shape(F)[0], n_u, n_line])
            D = tf.where(model.line_mask[None, :, :], D, tf.ones_like(D))
            D = tf.reduce_prod(D, axis=-1)

    W = tf.math.divide_no_nan(C[None, :], D)
    return tf.linalg.matvec(F, W)


#: Every method the preprocessor can record has a reference contraction, so a
#: backend that declines (or fails) always has somewhere to fall back to.
Backend.functions.update(dict.fromkeys(_COMPRESSIONS, _dense_contract))
assert all(
    callable(fn) for fn in Backend.functions.values()
), "the reference table must hold callables"


@register_preprocessor("dense_factor_merge")
class DenseFactorMergePreProcessor(DenseFactorPreProcessor):
    """``dense_factor`` preprocessor that merges identical basis blocks.

    Config::

        data:
          preprocessor:
            dense_factor_merge:
              merge_rtol: null   # automatic: 1e-8, or ~1e-6 with complex64
              merge_atol: 0.0
              merge_sample: 256
              merge_prepare: null
              store_dtype: complex128   # complex64 | complex32 (fp16 halves)
              method: index             # basis_index | real_index | svd | real_svd
              svd_rank: null
              merge_key: null   # shared record name; the model needs it too

    ``method`` picks what is stored: ``index`` keeps one representative per
    group of identical blocks (``merge_rtol``/``merge_atol``), ``svd`` keeps
    the low-rank coefficients ``U = F V^H`` with the rank taken from
    ``merge_rtol`` (relative singular value) or ``svd_rank``.  The SVD is exact
    at its numerical rank and is rebuilt before the contraction; a tolerance
    that drops real directions must be validated on the amplitude, because
    ``1/den`` amplifies the discarded ones.  ``real_svd`` fits ``F = C W^T``
    with the real SVD of ``[Re(F) | Im(F)]``: while ``F F^H`` is real - true
    for every basis in this plugin, where the complex part is a per-event
    factor times a real ``(p, u)`` structure - the singular values and the
    truncation error are those of ``svd``, while the stored coefficients are
    real (half the payload) and the basis is complex.  A basis that mixes
    real and imaginary parts inside a column has a different truncation under
    ``real_svd``; full rank is still exact, but validate a cut on the
    amplitude there.

    The record is fitted on the first real batch when the data is eager -
    the barrier factors depend on the event, so a generic sample need not
    span the data's basis for the SVD.  With ``lazy_call`` the widths are
    fixed when the map function is traced, so the record must exist first;
    it is then fitted on a phase-space sample generated by this preprocessor
    (``merge_sample`` events, default 256).  With
    ``lazy_call`` they are built in ``__init__``, because the tensor widths
    are fixed when the map function is traced; with eager data they are
    built from the first batch instead, so construction stays cheap and
    does not consume the RNG.  ``merge_prepare`` overrides that choice:
    ``true`` always builds at init, ``false`` always waits for the data.
    If the sample cannot be built the merge is disabled with a warning
    rather than failing.  The record is not attached to the configured
    amplitude: it is stored in the shared record space under ``merge_key``,
    and the amplitude models read it there, so a composed PDF - whose
    sub-model is not the object ``get_amplitude`` returns - still finds it.
    The key is only a name: this preprocessor cannot know which model will
    consume the blocks, and the model need not be reachable from its own
    ``amp``.  Give the same ``merge_key`` to the preprocessor and to
    ``amp_model``; without it each decay group gets its own key and both
    sides agree by construction.  A record already under the key is checked
    against this basis and method before it is reused.

    ``merge_rtol`` defaults to null, i.e. automatic and
    fixed at init from ``store_dtype``: ``1e-8``, or ``8*eps`` when the
    tensors are stored in complex64 (about ``1e-6``) or complex32 (about
    ``8e-3``), because the comparison can only be as fine as the stored
    values.  Give a number to override it, e.g. for a basis limited by
    cancellation; ``merge_atol`` adds a fixed floor to the index comparison
    (it does not affect the SVD cut).  ``store_dtype`` caches the merged
    blocks at that precision.  ``basis_index`` and ``real_index`` hand that
    block to the fused op unchanged (the kernel widens each stored value to
    double while reading); the other strategies convert it back before any
    product.  The accuracy of a real block stored in fp16 is not a fixed
    number: it is the fp16 floor times the basis' cancellation, measured from
    about 1e-3 to 2e-2 on the CP basis (see the ``store_dtype`` note in
    :class:`DenseFactorPreProcessor`).
    """

    def __init__(
        self,
        *args,
        merge_rtol=None,
        merge_atol=0.0,
        merge_sample=256,
        merge_prepare=None,
        method="index",
        svd_rank=None,
        merge_key=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if merge_rtol is None:
            # the comparison can only be as fine as what will be stored; the
            # flag, not the tensor, decides that, so this is fixed at init
            if self.store_dtype == "complex32":
                merge_rtol = 8 * float(np.finfo(np.float16).eps)
            elif self.store_dtype == "complex64":
                merge_rtol = 8 * float(np.finfo(np.complex64).eps)
            else:
                merge_rtol = 1e-8
        if method not in (
            "index",
            "basis_index",
            "real_index",
            "svd",
            "real_svd",
        ):
            raise ValueError(
                "method must be 'index', 'basis_index', 'real_index', "
                "'svd' or 'real_svd', got {!r}".format(method)
            )
        self.method = method
        if svd_rank is not None and int(svd_rank) < 1:
            raise ValueError("svd_rank must be >= 1")
        self.svd_rank = None if svd_rank is None else int(svd_rank)
        self.merge_rtol = float(merge_rtol)
        self.merge_atol = float(merge_atol)
        self.merge_sample = int(merge_sample)
        if merge_prepare is None:
            # only a lazy stream needs the maps before its map function is
            # traced; eager data can discover them from its first batch
            config = getattr(self.root_config, "config", None) or {}
            merge_prepare = bool(config.get("data", {}).get("lazy_call"))
        if (
            self.build_batch is not None
            and self.build_batch < self.merge_sample
        ):
            raise ValueError(
                "build_batch must be >= merge_sample: a chunk smaller than "
                "the fit sample makes the index probe unstable and is not a "
                "useful slice of the basis"
            )
        self.merge_prepare = bool(merge_prepare)
        #: names the shared record; must match the amplitude model's
        self.merge_id = merge_key or basis_key(self.decay_group)
        self.compression = self._make_compression()
        if self.merge_prepare:
            self._prepare_merge()

    def _make_compression(self):
        # the strategy declares the options it takes; read exactly those
        cls = _COMPRESSIONS[self.method]
        return cls(
            **{
                name: getattr(self, _COMPRESSION_SOURCES[name])
                for name in cls.options
            }
        )

    def _check_record(self, record):
        """A record reused under a key must describe this same basis."""
        structure = self.decay_group.get_dense_structure()
        method = record.get("method")
        if method != self.compression.name:
            raise ValueError(
                "merge_key {!r} already holds a {!r} record, but this "
                "preprocessor uses {!r}".format(
                    self.merge_id, method, self.compression.name
                )
            )
        if method in ("index", "basis_index", "real_index"):
            width, nparam = record["n_u"], record["n_param"]
        else:
            width, nparam = record["nu"], record["nparam"]
        if int(width) != int(structure["n_u"]):
            raise ValueError(
                "merge_key {!r} was fitted for a basis of width {}, not "
                "{}".format(self.merge_id, int(width), int(structure["n_u"]))
            )
        if int(nparam) != int(structure["n_param"]):
            raise ValueError(
                "merge_key {!r} was fitted for {} parameters, not {}".format(
                    self.merge_id, int(nparam), int(structure["n_param"])
                )
            )
        nproj = record.get("np", None)
        if nproj is not None and int(nproj) != int(
            np.prod(structure["f_shape"])
        ):
            raise ValueError(
                "merge_key {!r} was fitted for {} helicity projections, not "
                "{}".format(
                    self.merge_id,
                    int(nproj),
                    int(np.prod(structure["f_shape"])),
                )
            )
        return record

    def _existing_record(self):
        """The record already published under this key, if compatible."""
        record = get_merge_record(self.merge_id)
        if record:
            self._check_record(record)
        return record

    def _sample_terms(self, sample):
        """F/B of a generated phase-space sample."""
        dg = self.decay_group
        outs = list(dg.outs)
        masses = []
        for p in outs:
            m = p.get_mass() if hasattr(p, "get_mass") else p.mass
            if m is None:
                m = getattr(p, "data_mass", None)
            if m is None:
                raise ValueError("mass is required for {}".format(p))
            masses.append(float(m))
        top = dg.top
        top_mass = top.get_mass() if hasattr(top, "get_mass") else top.mass
        gen = PhaseSpaceGenerator(float(top_mass), masses)
        mom = gen.generate(sample)
        p4 = {str(p): m for p, m in zip(outs, mom)}
        # go through the base preprocessor so the sample follows the data's
        # angle conventions (center_mass, r_boost, ...) exactly; with
        # cp_trans and no per-event charges it leaves the momenta unchanged,
        # and the conjugated directions span the same row space here
        data = BasePreProcessor.call(self, {"p4": p4})
        return self._build_terms(data)

    def _prepare_merge(self, F=None, B=None):
        """Find the compression record, once.

        With real terms (``F``/``B`` of the first batch) it is fitted on
        data-like kinematics, which matters for the SVD: the barrier factors
        depend on the event, so a generic phase-space sample need not span
        the data's basis.  Without them - the lazy case, where the record
        must exist before the map function is traced - a sample is generated
        here.  It draws from the global TensorFlow RNG, so a later event
        generation with a fixed seed is shifted relative to a model without
        this preprocessor.
        """
        if self._existing_record() is not None:
            return
        sample = self.merge_sample
        if sample < 1:
            raise ValueError("the compression sample must be >= 1")
        try:
            if F is None or B is None:
                F, B = self._sample_terms(sample)
        except Exception as exc:
            # compression is an optimisation: leave it off and say why
            warnings.warn(
                "dense_factor_merge disabled for this model: {}".format(exc)
            )
            set_merge_record(self.merge_id, False)
            return
        record = self.compression.prepare(F, B)
        if (
            F is not None
            and self.build_batch is not None
            and record.get("method") in ("svd", "real_svd")
            and (
                record["rank_f"] >= int(F.shape[0])
                or record["rank_b"] >= int(B.shape[0])
            )
        ):
            # a chunk with rank == its own row count spans no more than the
            # events it holds; later chunks would hit unseen directions
            raise ValueError(
                "the first chunk spans the SVD basis only for its own "
                "events (rank == chunk size); later chunks may need unseen "
                "directions - increase build_batch or use method: 'index'"
            )
        set_merge_record(self.merge_id, record)

    def _check_blocks(self, merge, x, blocks):
        """Every block must be described by the shared record.

        The record is applied to each block, and a later ``build_batch`` chunk
        can fall outside the fitted span, so each is checked on a slice before
        the record is used; a block whose reconstruction is far worse than the
        main block's raises instead of silently changing the amplitude.
        """
        if not blocks or getattr(self.compression, "name", None) is None:
            return

        def error(d):
            F, B = d.get("F"), d.get("B")
            if F is None or B is None:
                # a block the model could not use is reported by pdf, not here
                return 0.0
            # spread the sample over the events: a later slice can reach
            # directions the first 512 do not
            step = max(1, int(F.shape[0]) // 512)
            F = F[::step][:512]
            B = B[::step][:512]
            out = self.compression.compress(merge, F, B)
            rec = self.compression.expand(merge, *out)
            err = 0.0
            for a, b in zip((F, B), rec):
                if a is None:
                    continue
                a = np.asarray(a)
                b = np.asarray(b)
                err = max(
                    err,
                    float(np.max(np.abs(a - b)))
                    / max(float(np.max(np.abs(a))), 1e-300),
                )
            return err

        ref = error(x)
        tolerated = max(5.0 * ref, 1e-6)
        for d in blocks:
            err = error(d)
            if err > tolerated:
                raise ValueError(
                    "the {} record does not describe an identical-particle "
                    "block: it was fitted on the main block's kinematics and "
                    "reconstructs this block to {:.1e} against {:.1e} on the "
                    "main block; use method: 'index'".format(
                        self.compression.name, err, ref
                    )
                )

    def _after_blocks(self, x, blocks):
        merge = self._existing_record()
        if merge is None:
            if not tf.executing_eagerly():
                raise RuntimeError(
                    "dense_factor_merge maps were not built before the graph "
                    "was traced; construct the preprocessor eagerly"
                )
            self._prepare_merge(x.get("F"), x.get("B"))
            merge = get_merge_record(self.merge_id)
        if not merge:
            return
        self._check_blocks(merge, x, blocks)

        def compress(d):
            F, B = self.compression.compress(merge, d.get("F"), d.get("B"))
            if F is not None:
                d["F"] = F
            if B is not None:
                d["B"] = B

        compress(x)
        for d in blocks:
            compress(d)


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


# ---------------------------------------------------------------------------
# The decorators in ``decays.py``/``particles.py`` run when this module imports
# them (above); nothing in this file defines an attached class.
# ---------------------------------------------------------------------------
