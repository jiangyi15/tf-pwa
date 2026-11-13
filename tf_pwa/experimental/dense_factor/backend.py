"""Backend registry for the dense-factor contraction.

Keep this module a leaf: the package imports it first, so importing anything
from the package here would see a half-initialised ``__init__``.

A *backend* is a named slot holding a table of method names (the
``_Compression`` names) to contraction callables::

    register_backend("cuda")

    @register_op("cuda", "basis_index")
    def basis_index(strategy, record, F, B, C, g, layout, prepared):
        ...

The callable signature is fixed so the fused kernels and the TF reference are
interchangeable::

    fn(strategy, record, F, B, C, g, layout, prepared) -> A

``strategy`` is the ``_Compression`` instance that owns ``record`` (it can
expand the stored basis back to full width), ``record`` is that merge record
(``None`` for a plain, unmerged basis), ``F``/``B`` are the **stored**
tensors, ``C``/``g`` are the numerator and denominator coefficients and
``layout`` is the model-owned, read-only mapping of ``LAYOUT_KEYS``
(``off``/``sz`` are the per-chain fused layout, ``None`` for the TF reference,
which reads its own keys from the same mapping).  ``prepared`` is the
backend's ``PREPARE`` result for ``(name, method)`` -- host-side static facts
of the inputs, cached per model per ``(backend, method)`` and record -- or
``None`` when that method needs none.  A callable
carries its own gradient: the TF entries rely on autodiff, the fused ones use
the op's ``tf.custom_gradient``.

A backend registers only the methods it overrides.  A method it leaves out is
absent from its table, so ``supports`` declines it and ``pick_backend`` asks
the next backend in ``prefer`` instead; that is what keeps the reference as a
per-method fallback rather than an all-or-nothing one.

The module itself has no CUDA dependency and imports nothing heavy; note that
importing it *through the package* runs the package ``__init__``, which imports
TensorFlow and registers every model, so the standalone-import cost is the
package's, not this module's.
"""

#: name -> {"functions": {method: contraction}}
BACKENDS = {}


def register_backend(name):
    """Create (or update) the backend slot ``name`` and return it.

    Every backend is called with the same arguments; a fused one declines a
    model without the per-chain layout (``layout["off"]`` is ``None``) itself,
    so the registry does not need a capability flag.
    """
    slot = BACKENDS.setdefault(name, {"functions": {}})
    return slot


def register_op(name, method):
    """Decorator: register ``fn`` as backend ``name``'s op for ``method``.

    ``method`` is ``None`` for the plain (unmerged) basis.  A backend registers
    only what it overrides; anything else stays absent so ``supports``
    declines it and the reference runs.
    """

    def deco(fn):
        register_backend(name)["functions"][method] = fn
        return fn

    return deco


#: (name, method) -> fn(strategy, record, layout) -> prepared
#: The prepared dict holds host-side *static* facts of the op inputs (ints,
#: dtypes) -- never tensors built by the current trace -- so it can be cached
#: across evaluations and ``tf.function`` traces.
PREPARE = {}


def register_prepare(name, method):
    """Register the static-input precompute for one (backend, method)."""

    def deco(fn):
        PREPARE[(name, method)] = fn
        return fn

    return deco


def get_prepare(name, method):
    """The prepare for (backend, method), or None when it needs none."""
    return PREPARE.get((name, method))


def supports(name, method):
    """Whether backend ``name`` has an op for ``method``."""
    return method in BACKENDS.get(name, {}).get("functions", {})


def get_op(name, method):
    """The contraction backend ``name`` registered for ``method``."""
    return BACKENDS[name]["functions"][method]


def pick_backend(method, prefer):
    """The first backend name in ``prefer`` that has ``method``.

    Selection is per method: ``prefer`` is the model's order of choice, and a
    backend that cannot take a *particular record* is still selected -- it
    declines inside its entry (warn once, then the reference).  If none of
    them has the method, the reference table is broken (it must cover every
    recorded method), so this is a wiring error and is reported as one.
    """
    for name in prefer:
        if supports(name, method):
            return name
    raise RuntimeError(
        "no reference contraction for {!r}: import "
        "tf_pwa.experimental.dense_factor to fill the table".format(method)
    )
