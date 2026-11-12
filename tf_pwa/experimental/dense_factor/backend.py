"""Backend registry for the dense-factor contraction.

Keep this module a leaf: the package imports it first, so importing anything
from the package here would see a half-initialised ``__init__``.

A *backend* is a table of method names (the ``_Compression`` names) to
contraction callables.  The callable signature is fixed so the fused kernels
and the TF reference are interchangeable::

    fn(strategy, record, F, B, C, g, off, sz, *, n_line, max_basis) -> A

``strategy`` is the ``_Compression`` instance that owns ``record`` (it can
expand the stored basis back to full width and, on the fused path, invoke the
op), ``record`` is that merge record (``None`` for a plain, unmerged basis),
``F``/``B`` are the **stored** tensors, ``C``/``g`` are the numerator and
denominator coefficients and ``off``/``sz`` the per-chain layout the kernels
need (``None`` for the TF reference, which rebuilds the denominator from its
own structure).  A callable carries its own gradient: the TF entries rely on
autodiff, the fused ones use the op's ``tf.custom_gradient``.

The module itself has no CUDA dependency and imports nothing heavy; note that
importing it *through the package* runs the package ``__init__``, which imports
TensorFlow and registers every model, so the standalone-import cost is the
package's, not this module's.
"""


class Backend:
    """One contraction backend.

    The base class *is* the TF reference backend: ``dense_factor`` fills
    ``functions`` with one entry per ``_Compression`` method.  A fused backend
    subclasses it, sets :attr:`name`, fills the methods its op implements and
    overrides :meth:`supports` if it needs a runtime condition (a build that
    failed, a shape it cannot read).
    """

    #: registry name; a model lists the ones it wants in ``prefer``
    name = "tf"

    #: method name -> contraction callable (see the module docstring)
    functions = {}

    #: True when the contractions need the per-chain layout (``off``/``sz``).
    #: A model without one skips such a backend instead of calling it with
    #: ``None``; the reference backend rebuilds the denominator itself.
    needs_layout = False

    @classmethod
    def supports(cls, method, record):
        """Whether this backend can contract ``(method, record)``.

        ``method`` is ``None`` for a plain basis and ``record`` may be
        ``None``; a backend that only reads reduced records overrides this.
        """
        return method in cls.functions


#: name -> backend class
BACKENDS = {}


def register_backend(cls):
    """Add a backend class to the registry and return it."""
    BACKENDS[cls.name] = cls
    return cls


#: The reference backend registers itself, so ``pick_backend`` is total for a
#: caller that only imports this module (the method table is filled in by
#: ``dense_factor``, which owns the strategies it points at).
register_backend(Backend)


def pick_backend(method, record, prefer=("cuda", "tf")):
    """The first backend in ``prefer`` that supports ``(method, record)``.

    ``prefer`` is the model's order of choice.  ``Backend`` (the TF reference)
    is always registered, so the fallback is total even when ``prefer`` names
    a backend this process never registered.
    """
    for name in prefer:
        cls = BACKENDS.get(name)
        if cls is not None and cls.supports(method, record):
            return cls
    cls = BACKENDS.get("tf")
    if cls is None:
        raise RuntimeError("the TF reference backend is not registered")
    if method not in cls.functions:
        raise RuntimeError(
            "no reference contraction for {!r}: import "
            "tf_pwa.experimental.dense_factor to fill the table".format(method)
        )
    return cls
