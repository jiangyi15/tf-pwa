"""Preprocessors of the dense-factor plugin.

``DenseFactorPreProcessor`` caches the data-only ``F``/``B`` basis, and
``DenseFactorMergePreProcessor`` adds the compression step that stores the
reduced basis and publishes its record.  Both register themselves on import.

They import the strategies and the record store from :mod:`.compress`; the one
place that needs the amplitude model (``_strip`` checks the model type) imports
it lazily, so this module does not depend on :mod:`.amp`.
"""

import contextlib
import warnings

import numpy as np
import tensorflow as tf

from tf_pwa.amp.preprocess import (
    BasePreProcessor,
    CachedAnglePreProcessor,
    register_preprocessor,
)
from tf_pwa.data import data_merge, split_generator
from tf_pwa.phasespace import PhaseSpaceGenerator

from .compress import (
    _COMPRESSION_SOURCES,
    _COMPRESSIONS,
    _cast_like,
    _pack_like,
    basis_key,
    get_merge_record,
    set_merge_record,
)


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
        # local import: this is the only place the preprocessor needs the
        # model, and importing .amp at module level would make this module
        # depend on the model for one isinstance check
        from .amp import DenseFactorModel

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
