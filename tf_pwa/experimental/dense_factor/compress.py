"""Compression strategies of the dense-factor plugin.

A strategy owns one ``method`` of the merge preprocessor: it stores a reduced
basis and writes the record (``method``, the maps, the transform) that the
amplitude model contracts through.  ``_COMPRESSIONS`` maps the ``method``
option to the class and ``_compression_of`` builds one for a record.

This module is a leaf on purpose: the strategies only consume the record dict,
so nothing here imports ``dense_factor`` and ``dense_factor`` can import this
module without a cycle.  The fused op package is imported lazily inside the
methods that call it, exactly as before.
"""

import itertools
import warnings
import weakref

import numpy as np
import tensorflow as tf

from tf_pwa.config import get_config, regist_config

# ---------------------------------------------------------------------------
# Merge identical basis blocks.  Many-wave models repeat the same kinematic
# basis once per wave: F depends only on the final-state kinematics and the
# quantum numbers, while m0, g0 and the couplings live in C and g.  The
# preprocessor finds the duplicates on the first batch, stores the maps in the
# structure and keeps one representative per group, so F and B are stored once
# per distinct basis block.  pdf expands them with the index and then runs the
# plain contraction, so the saving is the stored size (a lazy cache, the
# host-to-device copies), not the arithmetic.
# ---------------------------------------------------------------------------


def _axis_groups(X, axis, probe=2048, rtol=1e-8, atol=0.0):
    """Group equal slices of ``X`` along ``axis``, using ``probe`` rows.

    Returns ``(map, reps)``: the group of every slice and one representative
    index per group.  The test is ``max|a - b| <= atol + rtol * scale``, where
    ``scale`` is the larger of the two slices' peak magnitudes.  The
    default ``rtol`` is deliberately loose: roundoff between two computations
    of the same basis is at most ~1e-12 relative, genuinely different bases
    differ by 1e-4 or more even near a threshold, and the cost of the two
    mistakes is asymmetric - a missed merge only keeps the tensors large,
    while a wrong merge changes the amplitude.  A byte
    comparison is wrong for complex floats, and an element-wise relative one
    is too strict for a basis limited by cancellation: an entry that is the
    difference of large terms carries an absolute error set by the
    intermediate, not by its own small value, so the peak is the honest
    yardstick.  ``merge_atol`` adds a fixed floor; set ``merge_rtol: 0`` for a
    purely absolute test.  Non-finite slices are only merged when they agree
    exactly, since an ``Inf`` would make the tolerance itself infinite.
    """
    rows = np.asarray(X)
    step = max(1, rows.shape[0] // probe) if probe else 1
    A = np.moveaxis(rows[::step][:probe], axis, -1)
    if A.shape[-1] == 0:
        return [], []
    flat = np.ascontiguousarray(A).reshape(-1, A.shape[-1])
    scale = np.max(np.abs(flat), axis=0) if flat.size else np.zeros(0)
    reps, mapping = [], []
    for c in range(flat.shape[1]):
        col = flat[:, c]
        group = None
        for g, r in enumerate(reps):
            ref = flat[:, r]
            if np.array_equal(col, ref):
                group = g
                break
            if not (np.all(np.isfinite(col)) and np.all(np.isfinite(ref))):
                continue
            with np.errstate(invalid="ignore"):
                tol = atol + rtol * np.maximum(np.abs(col), np.abs(ref))
                same = np.all(np.abs(col - ref) <= tol)
            if same:
                group = g
                break
        if group is None:
            group = len(reps)
            reps.append(c)
        mapping.append(group)
    return mapping, reps


def _find_merge(F, B, probe=2048, rtol=1e-13, atol=0.0):
    """Maps of the duplicate basis slices of ``F`` and ``B``."""
    map_u, rep_u = _axis_groups(F, 2, probe, rtol, atol)
    map_i, rep_k = _axis_groups(B, 1, probe, rtol, atol)
    return {
        "map_u": tf.constant(map_u, tf.int32),
        "map_i": tf.constant(map_i, tf.int32),
        "rep_u": rep_u,
        "rep_k": rep_k,
        # the full widths, so the record says whether a block is reduced
        "n_u": int(F.shape[2]),
        "n_param": int(B.shape[1]),
    }


def _pack_complex16(x):
    """Store a complex tensor as a pair of float16 halves (complex32)."""
    return tf.cast(
        tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1), tf.float16
    )


def _cast_like(x, complex_dtype, real_dtype):
    """Lower a stored block to ``complex_dtype``, or to ``real_dtype`` if
    it is real coefficients."""
    return tf.cast(x, complex_dtype if x.dtype.is_complex else real_dtype)


def _pack_like(x):
    """``_cast_like`` for complex32.

    A complex block becomes ``[..., 2]`` float16 halves; a real block (the
    coefficients of a split method) is a plain float16 tensor.  The two are
    told apart by the record, not the dtype - see ``_unpack_stored``.
    """
    return _pack_complex16(x) if x.dtype.is_complex else tf.cast(x, tf.float16)


def _unpack_terms(x, dtype):
    """Read a stored ``F``/``B`` as complex, whatever ``store_dtype`` is.

    This assumes the packed complex32 form.  A *real* block can also be
    float16 (the coefficients of a split method at ``store_dtype:
    complex32``), and the dtype alone cannot tell the two apart: callers that
    may hold one must consult the record, as ``_unpack_stored`` does.
    """
    if x.dtype == tf.float16:  # complex32: last axis holds (real, imag)
        part = tf.float32 if dtype == tf.complex64 else tf.float64
        return tf.complex(tf.cast(x[..., 0], part), tf.cast(x[..., 1], part))
    if not x.dtype.is_complex:
        return x
    return tf.cast(x, dtype)


def _split_complex(M):
    """``[Re(M) | Im(M)]``: the real matrix with the same singular values.

    ``[A | B] = U S V^T`` gives ``M = A + iB = U S (V1 + i V2)^T``, so the
    coefficients are real and the basis is complex.  The singular values are
    those of the complex ``M`` whenever ``M M^H`` is real, which holds for
    this plugin's bases (``|Im(M M^H)| / |M M^H|`` is at the round-off).
    """
    if M.dtype.is_complex:
        return tf.concat([tf.math.real(M), tf.math.imag(M)], axis=1)
    return tf.concat([M, tf.zeros_like(M)], axis=1)


def _join_complex(Z, n, dtype):
    """Inverse of ``_split_complex`` on the reconstructed columns."""
    if dtype.is_complex:
        return tf.complex(Z[:, :n], Z[:, n:])
    return Z[:, :n]


def _real_dtype(dtype):
    """The real dtype that stores coefficients for this arithmetic dtype."""
    if dtype == tf.complex128:
        return tf.float64
    if dtype == tf.complex64:
        return tf.float32
    return dtype


def _unpack_real(x, dtype):
    """Read a stored real coefficient block at the arithmetic precision."""
    return tf.cast(x, _real_dtype(dtype))


def _unpack_stored(x, merge, dtype):
    """Read a stored tensor, knowing whether its record is real.

    A split record (``real_svd``, ``real_index``) keeps real blocks, possibly
    at float16; the dtype alone cannot separate that from the packed complex
    form, so the record decides.
    """
    if merge and merge.get("method") in (
        "real_svd",
        "real_index",
        "basis_index",
    ):
        return _unpack_real(x, merge["dtype"])
    return _unpack_terms(x, dtype)


def _project(M, Vt):
    """``M @ V`` for a basis stored as rows, ``Vt = V^H``."""
    V = tf.linalg.adjoint(Vt)
    if M.dtype.is_complex and not V.dtype.is_complex:
        return tf.complex(
            tf.matmul(tf.math.real(M), V), tf.matmul(tf.math.imag(M), V)
        )
    return tf.matmul(M, V)


def _reconstruct(C, Vt):
    """``C @ V^H`` for a basis stored as rows, ``Vt = V^H``."""
    if C.dtype.is_complex and not Vt.dtype.is_complex:
        return tf.complex(
            tf.matmul(tf.math.real(C), Vt), tf.matmul(tf.math.imag(C), Vt)
        )
    return tf.matmul(C, Vt)


def _expand_svd(c, F, B):
    """Rebuild the full basis from the stored low-rank coefficients."""
    if c.get("split"):
        F = _unpack_real(F, c["dtype"])
        B = _unpack_real(B, c["dtype_b"])
        n = int(c["np"]) * int(c["nu"])
        M = tf.matmul(F, c["Vf"])
        M = _join_complex(M, n, c["dtype"])
        M = tf.reshape(M, (tf.shape(M)[0], c["np"], c["nu"]))
        return M, _join_complex(
            tf.matmul(B, c["Vb"]), int(c["nparam"]), c["dtype_b"]
        )
    F = _unpack_terms(F, c["dtype"])
    B = _unpack_terms(B, c["dtype_b"])
    M = _reconstruct(F, c["Vf"])  # [ne,k] x [k,np*nu]
    M = tf.reshape(M, (tf.shape(M)[0], c["np"], c["nu"]))
    return M, _reconstruct(B, c["Vb"])


def _expand_block(merge, F, B):
    """Expand a stored block with the merge index.

    ``merge`` is the model's map record.  Group ids index the reduced
    tensors, so only a reduced block is gathered; a full-width one is passed
    through unchanged.
    """
    if not merge:
        # no maps: no merge preprocessor, or its sample could not be built,
        # so the blocks are stored at full width and are used as they are
        return F, B
    if int(F.shape[2]) != int(merge["n_u"]):
        F = tf.gather(F, merge["map_u"], axis=2)
    if int(B.shape[1]) != int(merge["n_param"]):
        B = tf.gather(B, merge["map_i"], axis=1)
    return F, B


#: Fit cap of the sparse transform.  This is only the matching-pursuit
#: budget: the record stores the *actual* longest row of the fitted
#: transform, which may be shorter (and is ``<= _TERMS``).
_TERMS = 4


def _sparse_basis(cols, rtol, probe=2048, kmax=_TERMS):
    """A sparse exact column basis: selected columns + per-column transform.

    Columns are added to the basis only when they are not already a short
    combination of the selected ones; every other column is written over the
    basis by matching pursuit, so the transform has at most ``kmax`` entries
    per column.  A column whose sparse residual is still above ``rtol`` is
    promoted to the basis, so the transform is exact to the same tolerance
    the column merge uses.
    """
    n, ncol = cols.shape
    step = max(1, n // probe) if probe else 1
    sample = cols[::step]
    sel, qvecs, rows = [], [], [None] * ncol

    def sparse_row(v):
        chosen, coef = [], None
        res = v.copy()
        nv = np.linalg.norm(v)
        for _ in range(kmax):
            if not sel:
                return None
            cors = np.array(
                [
                    abs(sample[:, j].conj() @ res)
                    / max(np.linalg.norm(sample[:, j]), 1e-300)
                    for j in sel
                ]
            )
            if chosen:
                cors[chosen] = -1.0
            k = int(np.argmax(cors))
            chosen.append(k)
            A = sample[:, [sel[t] for t in chosen]]
            coef = np.linalg.lstsq(A, v, rcond=None)[0]
            res = v - A @ coef
            if np.linalg.norm(res) <= rtol * nv:
                break
        if np.linalg.norm(res) > rtol * nv:
            return None
        return [(sel[t], coef[i]) for i, t in enumerate(chosen)]

    for c in range(ncol):
        v = sample[:, c]
        if np.linalg.norm(v) == 0.0:
            rows[c] = []
            continue
        r = v.copy()
        for q in qvecs:
            r = r - q * (q.conj() @ r)
        if np.linalg.norm(r) > rtol * np.linalg.norm(v):
            qvecs.append(r / np.linalg.norm(r))
            sel.append(c)
            rows[c] = None  # a basis column: unit row, set below
            continue
        row = sparse_row(v)
        if row is None:  # needs more terms than kmax: keep it exactly
            # its residual is below rtol, so it adds no new direction: do not
            # perturb the orthonormal set with a noise vector
            sel.append(c)
            rows[c] = None
        else:
            rows[c] = row
    pos = {c: i for i, c in enumerate(sel)}
    for c in range(ncol):
        if rows[c] is None:
            rows[c] = [(pos[c], 1.0 + 0.0j)]
        else:
            rows[c] = [(pos[j], w) for j, w in rows[c]]
    return sel, rows


#: default value of each compression option
_COMPRESSION_DEFAULTS = {"rtol": 0.0, "rank": None, "atol": 0.0, "sample": 0}

#: the preprocessor attribute that provides each compression option
_COMPRESSION_SOURCES = {
    "rtol": "merge_rtol",
    "rank": "svd_rank",
    "atol": "merge_atol",
    "sample": "merge_sample",
}


#: failures already reported, so a broken op warns once instead of per call
_OP_FAILURES = set()


def _warn_op_failure(method, exc):
    """Report a failed fused call without disabling the op for the process.

    Called by the fused backend (``dense_factor_op_model``), which owns the
    op execution; a single block the kernel cannot take (a degenerate record,
    a shape the wrapper rejects) must fall back there and leave every other
    record using the op.  Only a distinct failure is reported, so the log
    does not grow on every evaluation.
    """
    sig = (method, type(exc).__name__, str(exc)[:200])
    if sig in _OP_FAILURES:
        return
    _OP_FAILURES.add(sig)
    warnings.warn(
        "the dense_factor {} op failed ({}); this call falls back to the "
        "reference contraction".format(method or "plain", exc)
    )


class _Compression:
    """Base of a basis compression strategy.

    ``options`` is the exact set of constructor options this strategy takes:
    the preprocessor passes only those, and ``__init__`` rejects anything
    else, so a strategy cannot silently receive an option it ignores.
    """

    name = None
    options = ()
    #: True when the record stores one *real* block that the fused ops read
    #: in its stored dtype (float64/float32/fp16).  The op-model path then
    #: leaves the block alone instead of expanding it first; every other
    #: strategy stores a complex (or packed) block that has to be unpacked.
    stores_real_block = False

    def __init__(self, **opts):
        unknown = set(opts) - set(self.options)
        if unknown:
            raise TypeError(
                "{} does not take {}".format(
                    type(self).__name__, sorted(unknown)
                )
            )
        for name in self.options:
            setattr(self, name, opts.get(name, _COMPRESSION_DEFAULTS[name]))


class _IndexCompression(_Compression):
    """Store one representative per group of identical blocks."""

    name = "index"
    options = ("rtol", "atol", "sample")

    def prepare(self, F, B):
        record = _find_merge(
            F, B, probe=self.sample, rtol=self.rtol, atol=self.atol
        )
        record["method"] = self.name
        return record

    def compress(self, record, F, B):
        if F is not None:
            F = tf.gather(F, tf.constant(record["rep_u"], tf.int32), axis=2)
        if B is not None:
            B = tf.gather(B, tf.constant(record["rep_k"], tf.int32), axis=1)
        return F, B

    def expand(self, record, F, B):
        return _expand_block(record, F, B)


class _BasisIndexCompression(_IndexCompression):
    """``index`` with a sparse exact basis, stored as one real block.

    After the index merge, a sparse exact basis of the remaining columns is
    found (``_sparse_basis``): a column is stored only if it is not already a
    combination of at most ``_TERMS`` stored ones, and every dropped column
    keeps a transform row of at most that many entries.  The transform is
    stored as CSR (``ptr``/``col``/``val``, exact per-row bounds, no
    padding), fitted on the merge
    probe but checked on every event, and the store falls back to all columns
    (with a warning) if the probe misled it.  The stored basis columns are then written as
    ``[Re | Im]`` and merged like ``real_index``, so both factors apply: the
    rank reduction of the basis and the real storage of the columns.

    The complex basis and the dropped columns are rebuilt by grouping the
    columns by their row length and scattering each term's contribution, so
    the contraction stays a gather; neither an SVD nor a new kernel is
    involved.  ``merge_atol`` only affects the ``[Re | Im]`` column merge,
    not the basis fit: the fit keeps a column only for an exact relation, so
    an absolute floor has nothing to cut.
    """

    stores_real_block = True
    name = "basis_index"

    #: Whether the fused backend may contract this record; ``False`` forces
    #: the reference path (a test knob that also proves the two agree).  The
    #: backend reads it, the strategy itself never calls an op.
    use_op = True

    def prepare(self, F, B):
        record = super().prepare(F, B)
        record["dtype"] = F.dtype
        record["dtype_b"] = B.dtype
        record["np"] = int(F.shape[1])
        for key, X, axis, rep in (
            ("f", F, 2, record["rep_u"]),
            ("b", B, 1, record["rep_k"]),
        ):
            red = tf.gather(X, tf.constant(rep, tf.int32), axis=axis)
            flat = tf.reshape(red, (tf.shape(red)[0], -1))
            fnp = flat.numpy()
            sel, rows = _sparse_basis(fnp, self.rtol, self.sample, kmax=_TERMS)
            if not sel:
                # every column is zero: the basis is empty, so the store must
                # fall back to the identity instead of writing no column
                sel = list(range(fnp.shape[1]))
                rows = [[(c, 1.0 + 0.0j)] for c in range(fnp.shape[1])]
            elif len(sel) < fnp.shape[1]:
                # the transform was fitted on a probe; check it on every event
                # and fall back to storing all columns if the probe misled it
                bsel = fnp[:, sel]
                rec = np.zeros_like(fnp)
                for c, row in enumerate(rows):
                    for j, w in row:
                        rec[:, c] += w * bsel[:, j]
                err = np.linalg.norm(fnp - rec, axis=0)
                scale = np.maximum(np.linalg.norm(fnp, axis=0), 1e-300)
                if np.max(err / scale) > self.rtol:
                    warnings.warn(
                        "the sparse basis does not hold beyond the fit probe; "
                        "basis_index stores every column"
                    )
                    sel = list(range(fnp.shape[1]))
                    rows = [[(c, 1.0 + 0.0j)] for c in range(fnp.shape[1])]
            basis = tf.gather(flat, tf.constant(sel, tf.int32), axis=1)
            both = tf.concat(
                [tf.math.real(basis), tf.math.imag(basis)], axis=1
            )
            mapping, reps = _axis_groups(
                both, 1, self.sample, self.rtol, self.atol
            )
            width = len(sel)
            record[key + "_sel"] = sel
            record[key + "_reps"] = reps
            record[key + "_re_map"] = tf.constant(mapping[:width], tf.int32)
            record[key + "_im_map"] = tf.constant(mapping[width:], tf.int32)
            # the transform as CSR: exact per-column bounds, no padding
            lengths = [len(row) for row in rows]
            ptr = np.zeros(len(rows) + 1, dtype=np.int32)
            np.cumsum(lengths, out=ptr[1:])
            nnz = int(ptr[-1])
            col = np.zeros(nnz, dtype=np.int32)
            value = np.zeros(nnz, dtype=np.complex128)
            e = 0
            for row in rows:
                for p, w in row:
                    col[e] = p
                    value[e] = w
                    e += 1
            if (
                e != nnz
                or ptr[0] != 0
                or ptr[-1] != nnz
                or np.any(np.diff(ptr) < 0)
                or (nnz and int(col.max()) >= len(sel))
            ):
                raise ValueError("malformed CSR transform")
            record[key + "_ptr"] = tf.constant(ptr, tf.int32)
            record[key + "_col"] = tf.constant(col, tf.int32)
            record[key + "_val"] = tf.constant(value, tf.complex128)
            record[key + "_max"] = max(lengths, default=0)
            record[key + "_width"] = int(flat.shape[1])
        # one unrolled reader serves both tensors: the widest row of either
        record["terms"] = max(record["f_max"], record["b_max"], 1)
        return record

    def compress(self, record, F, B):
        F = super().compress(record, F, None)[0]
        B = super().compress(record, None, B)[1]
        out = []
        for X, key in ((F, "f"), (B, "b")):
            if X is None:
                out.append(None)
                continue
            flat = tf.reshape(X, (tf.shape(X)[0], -1))
            basis = tf.gather(
                flat, tf.constant(record[key + "_sel"], tf.int32), axis=1
            )
            both = tf.concat(
                [tf.math.real(basis), tf.math.imag(basis)], axis=1
            )
            out.append(
                tf.gather(
                    both,
                    tf.constant(record[key + "_reps"], tf.int32),
                    axis=1,
                )
            )
        return out[0], out[1]

    def _rebuild(self, record, X, key, np_):
        """Real block -> complex basis -> full reduced columns."""
        dtype = record["dtype" if key == "f" else "dtype_b"]
        part = tf.float32 if dtype == tf.complex64 else tf.float64
        store = tf.cast(X, part)
        n = tf.shape(store)[0]
        # the maps are per column: gather the stored block, no event offset
        basis = tf.complex(
            tf.gather(store, record[key + "_re_map"], axis=1),
            tf.gather(store, record[key + "_im_map"], axis=1),
        )
        ptr = record[key + "_ptr"]
        col = record[key + "_col"]
        val = record[key + "_val"]
        width = record[key + "_width"]
        lens = ptr[1:] - ptr[:-1]
        full_t = tf.zeros((width, n), dtype=basis.dtype)
        for t in range(record[key + "_max"]):
            js = tf.where(lens > t)[:, 0]
            e = tf.gather(ptr, js) + t
            upd = (
                tf.gather(basis, tf.gather(col, e), axis=1)
                * tf.cast(tf.gather(val, e), basis.dtype)[None, :]
            )
            full_t = tf.tensor_scatter_nd_add(
                full_t, js[:, None], tf.transpose(upd)
            )
        full = tf.transpose(full_t)
        if np_:
            return tf.reshape(full, (-1, np_, width // np_))
        return full

    def expand(self, record, F, B):
        out = []
        for X, key, np_ in ((F, "f", record["np"]), (B, "b", None)):
            out.append(
                None if X is None else self._rebuild(record, X, key, np_)
            )
        F, B = out
        return _expand_block(record, F, B)


class _RealIndexCompression(_IndexCompression):
    """``index`` with the real and imaginary parts stored as one real block.

    The gathered columns are split into ``[Re | Im]`` and the same column
    merge is applied to that block, so an all-zero column is one stored
    column like any other duplicate, and the two parts can share a column (a
    constant ``1``, a repeated term) even when the complex columns differ.
    The payload is ``len(reps)`` reals per event instead of ``2 * width``
    (measured: 17-20% less for ``F`` and 33% for ``B`` on J=1/J=2 bases, 50%
    to 94% for ``F`` when the basis is real, amplitude unchanged at 1e-16);
    the fused op model contracts the real block directly through the split
    ``DenseFactorFwdRi``/``GradRi`` pair (measured 2.7 ms against 11.9 ms
    for the rebuild and 3.0 ms for ``index`` at 20k events), so only the dense
    model, which needs a complex basis, pays for the two gathers per batch.
    """

    stores_real_block = True
    name = "real_index"

    #: Whether the fused backend may contract this record; ``False`` forces
    #: the reference path (a test knob that also proves the two agree).  The
    #: backend reads it, the strategy itself never calls an op.
    use_op = True

    def prepare(self, F, B):
        record = super().prepare(F, B)
        record["dtype"] = F.dtype
        record["dtype_b"] = B.dtype
        record["np"] = int(F.shape[1])
        for key, X, axis, rep in (
            ("f", F, 2, record["rep_u"]),
            ("b", B, 1, record["rep_k"]),
        ):
            red = tf.gather(X, tf.constant(rep, tf.int32), axis=axis)
            flat = tf.reshape(red, (tf.shape(red)[0], -1))
            width = int(flat.shape[1])
            both = tf.concat([tf.math.real(flat), tf.math.imag(flat)], axis=1)
            mapping, reps = _axis_groups(
                both, 1, self.sample, self.rtol, self.atol
            )
            record[key + "_reps"] = reps
            record[key + "_re_map"] = tf.constant(mapping[:width], tf.int32)
            record[key + "_im_map"] = tf.constant(mapping[width:], tf.int32)
            record[key + "_width"] = width
        return record

    def compress(self, record, F, B):
        F = super().compress(record, F, None)[0]
        B = super().compress(record, None, B)[1]
        out = []
        for X, key in ((F, "f"), (B, "b")):
            if X is None:
                out.append(None)
                continue
            flat = tf.reshape(X, (tf.shape(X)[0], -1))
            both = tf.concat([tf.math.real(flat), tf.math.imag(flat)], axis=1)
            out.append(
                tf.gather(
                    both,
                    tf.constant(record[key + "_reps"], tf.int32),
                    axis=1,
                )
            )
        return out[0], out[1]

    def _reconstruct(self, record, F, B):
        out = []
        for X, key, np_ in ((F, "f", record["np"]), (B, "b", None)):
            if X is None:
                out.append(None)
                continue
            dtype = record["dtype" if key == "f" else "dtype_b"]
            part = tf.float32 if dtype == tf.complex64 else tf.float64
            # the maps are per column: gather the stored block, no event
            # offset (a flat gather was slower here as well)
            X = tf.cast(X, part)
            width = record[key + "_width"]
            re = tf.gather(X, record[key + "_re_map"], axis=1)
            im = tf.gather(X, record[key + "_im_map"], axis=1)
            comp = tf.complex(re, im)
            if not dtype.is_complex:
                comp = tf.math.real(comp)
            shape = (tf.shape(comp)[0],) + (
                (np_, width // np_) if np_ else (width,)
            )
            out.append(tf.reshape(comp, shape))
        return out[0], out[1]

    def expand(self, record, F, B):
        F, B = self._reconstruct(record, F, B)
        return _expand_block(record, F, B)


class _SvdCompression(_Compression):
    """Store the low-rank coefficients ``U = F V^H`` (complex SVD).

    ``method: real_svd`` is ``M = C W^T`` fitted with the real SVD of
    ``[Re(M) | Im(M)]`` (see ``_split_complex``): the same singular values
    and truncation error for this plugin's bases, real coefficients (half
    the payload) and a complex basis rebuilt from the two halves.
    """

    name = "svd"
    options = ("rtol", "rank")

    def _cut(self, s):
        """How many directions ``rtol``/``rank`` keep."""
        if self.rank is not None:
            return min(self.rank, int(s.shape[0]))
        k = int(tf.reduce_sum(tf.cast(s > self.rtol * s[0], tf.int32)))
        return max(k, 1)

    def _basis(self, M):
        """Right singular vectors above ``rtol`` (or the first ``rank``).

        ``tf.linalg.svd`` returns ``v = V`` for complex (``M = u s v^H``);
        store ``V^H`` so "row j is the j-th right singular vector".
        """
        s, u, v = tf.linalg.svd(M, full_matrices=False)
        k = self._cut(s)
        return tf.linalg.adjoint(v)[:k], k

    def prepare(self, F, B):
        Vf, rf = self._basis(tf.reshape(F, (tf.shape(F)[0], -1)))
        Vb, rb = self._basis(B)
        return {
            "method": self.name,
            "Vf": Vf,
            "Vb": Vb,
            "dtype": F.dtype,
            "dtype_b": B.dtype,
            "np": int(F.shape[1]),
            "nu": int(F.shape[2]),
            "nparam": int(B.shape[1]),
            "rank_f": rf,
            "rank_b": rb,
        }

    def compress(self, record, F, B):
        if F is not None:
            M = tf.reshape(F, (tf.shape(F)[0], -1))
            F = _project(M, record["Vf"])
        if B is not None:
            B = _project(B, record["Vb"])
        return F, B

    def expand(self, record, F, B):
        return _expand_svd(record, F, B)


class _RealSvdCompression(_SvdCompression):
    """``svd`` with ``M = C W^T``: real coefficients, complex basis.

    The fit is the real SVD of ``[Re(M) | Im(M)]``.  Its singular values are
    the complex ones for this plugin's bases (``M M^H`` is real), so the
    truncation error is identical to ``svd`` while the stored coefficients
    are real - half the payload - and the projection is real arithmetic.
    ``W = V1 + i V2`` is rebuilt from the two halves of the basis on demand.

    The rebuild runs at the arithmetic dtype (``_real_dtype(record["dtype"])``),
    not at the storage dtype: float32 coefficients are upcast to float64 for
    a complex128 model (measured 5e-8), while a complex64 model rebuilds in
    float32 and inherits the GPU float32 matmul floor (about 5e-4, TF32).
    ``store_dtype`` squares the width: float32 coefficients under complex64,
    float16 under complex32 - half the packed complex form there, with
    float16's own precision (about 1e-3).  The record, not the dtype, says
    the block is real (``_unpack_stored``).

    A real tensor (``B`` in most models) contributes an exact zero imaginary
    half; with ``merge_rtol: 0`` those zero singular values count as
    directions, so keep the default tolerance there.
    """

    name = "real_svd"

    def _basis(self, M):
        s, u, v = tf.linalg.svd(_split_complex(M), full_matrices=False)
        k = self._cut(s)
        return tf.transpose(v[:, :k]), k

    def prepare(self, F, B):
        record = super().prepare(F, B)
        record["split"] = True
        return record

    def compress(self, record, F, B):
        if F is not None:
            M = tf.reshape(F, (tf.shape(F)[0], -1))
            F = tf.matmul(_split_complex(M), tf.transpose(record["Vf"]))
        if B is not None:
            B = tf.matmul(_split_complex(B), tf.transpose(record["Vb"]))
        return F, B


class _NoCompression:
    """Nothing stored reduced: the basis is already full width."""

    name = None
    #: a full-width block is complex, so it is always unpacked first
    stores_real_block = False

    def expand(self, record, F, B):
        """Nothing was reduced, so the stored basis is already full width."""
        return F, B


_COMPRESSIONS = {
    None: _NoCompression,
    "index": _IndexCompression,
    "basis_index": _BasisIndexCompression,
    "real_index": _RealIndexCompression,
    "svd": _SvdCompression,
    "real_svd": _RealSvdCompression,
}


def _compression_of(record):
    """The strategy that owns a record; ``None``/``False`` mean none."""
    if not record:  # no record, or the failure sentinel
        return _NoCompression()
    return _COMPRESSIONS[record["method"]]()


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
