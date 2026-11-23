"""What a dense-factor model has to provide.

The plugin attaches two implementations to classes from ``tf_pwa.amp``:

- ``@terms(name)`` gives a *particle* model ``get_term_basis`` /
  ``get_term_coeff`` -- the denominator terms ``sum_i g_i B_i``.  It requires
  the physics methods named in ``requires`` to be untouched (a particle model
  that overrides ``get_amp`` keeps its own amplitude and no terms are
  installed).
- ``@basis(name)`` gives a *decay* class ``get_amplitude_basis`` /
  ``get_amplitude_coeff`` -- the numerator basis of the decay.

``name`` is a particle model name for ``terms`` and one of ``AmpDecay``,
``HelicityDecay``, ``DecayChain``, ``DecayGroup`` for ``basis``; ``basis``
also accepts a particle-model name, in which case it falls back to that model.
The two name spaces are otherwise disjoint: a particle model has terms, a decay
class has a basis.

``particles.py`` implements the term side, ``decays.py`` the basis side, and
``decays.py`` imports ``particles`` so importing the decays registers both.
"""

import functools

import tensorflow as tf

from tf_pwa.amp.core import (
    AmpDecay,
    DecayChain,
    DecayGroup,
    HelicityDecay,
    get_particle_model,
    rename_data_dict,
)

# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

_BASIS_TARGETS = {
    "AmpDecay": AmpDecay,
    "HelicityDecay": HelicityDecay,
    "DecayChain": DecayChain,
    "DecayGroup": DecayGroup,
}


def _guard(method, refs):
    """Allow the method only when the physics methods are not overridden."""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        for attr, ref in refs.items():
            if getattr(type(self), attr) is not ref:
                raise NotImplementedError(
                    "{} ({}) does not implement {}".format(
                        getattr(self, "name", type(self).__name__),
                        type(self).__name__,
                        method.__name__,
                    )
                )
        return method(self, *args, **kwargs)

    return wrapped


def terms(name, requires=("get_amp",)):
    """Attach the decorated methods to particle model ``name``.

    Subclasses only inherit the implementation when they do not override
    any method listed in ``requires``.  A subclass that changes the
    lineshape must provide its own term methods.

    Usage::

        @terms("BW")
        class _BWTerms:
            def get_term_basis(self, data, data_c=None, **kwargs):
                ...

            def get_term_coeff(self):
                ...
    """

    def deco(cls):
        target = get_particle_model(name)
        if target is None:
            return cls
        refs = {attr: getattr(target, attr) for attr in requires}
        for attr, value in vars(cls).items():
            if attr.startswith("__"):
                continue
            if not callable(value):
                continue
            if attr.startswith("_"):
                setattr(target, attr, value)
            else:
                setattr(target, attr, _guard(value, refs))
        return cls

    return deco


def basis(name):
    """Attach the decorated methods to the named amplitude class.

    ``name`` is either one of ``AmpDecay``, ``HelicityDecay``,
    ``DecayChain``, ``DecayGroup`` or a registered particle model name.
    """

    def deco(cls):
        target = _BASIS_TARGETS.get(name)
        if target is None:
            target = get_particle_model(name)
        if target is not None:
            _attach(target, cls)
        return cls

    return deco


def _attach(target, source):
    for name, value in vars(source).items():
        if name.startswith("__") or not callable(value):
            continue
        setattr(target, name, value)
