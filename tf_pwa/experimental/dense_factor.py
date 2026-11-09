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
import functools
import itertools
import warnings

import numpy as np
import tensorflow as tf

import tf_pwa.amp.interpolation  # noqa: F401  (register spline models)
from tf_pwa.amp.amp import BaseAmplitudeModel, register_amp_model
from tf_pwa.amp.core import (
    AmpDecay,
    DecayChain,
    DecayGroup,
    HelicityDecay,
    get_particle_model,
)
from tf_pwa.amp.flatte import cal_monentum
from tf_pwa.amp.preprocess import (
    BasePreProcessor,
    CachedAnglePreProcessor,
    register_preprocessor,
)
from tf_pwa.breit_wigner import (
    Bprime_num,
    Bprime_polynomial,
    dFun,
    dh_dsFun,
    hFun,
    twoBodyCMmom,
)
from tf_pwa.cal_angle import cal_angle_from_momentum
from tf_pwa.config import get_config
from tf_pwa.particle import cp_charge_group
from tf_pwa.phasespace import PhaseSpaceGenerator

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


# ---------------------------------------------------------------------------
# Particle terms
# ---------------------------------------------------------------------------


@terms("default")
class _DefaultTerms:

    def _get_bw_l(self):
        L = self.bw_l
        if L is None:
            L = min(self.decay[0].get_l_list())
        return L

    def _get_q0(self):
        return self.decay[0].get_relative_momentum({}, False)

    def _F_shape(self, q, m, L, d=None):
        if d is None:
            d = self.d
        q = tf.cast(q, m.dtype)
        return q ** (2 * L + 1) / (m * Bprime_polynomial(L, (q * d) ** 2))

    def get_term_basis(self, data, data_c=None, **kwargs):
        """Return the data-only basis of the rational terms."""
        m = data["m"]
        g0 = self.get_width()
        if g0 is None:
            raise NotImplementedError(
                "{} ({}) has no width for get_term_basis".format(
                    self, type(self).__name__
                )
            )
        one = tf.ones_like(m)
        zero = tf.zeros_like(m)
        if self.running_width:
            if data_c is None or "|q|" not in data_c:
                raise NotImplementedError(
                    "{} ({}) needs |q| for get_term_basis".format(
                        self, type(self).__name__
                    )
                )
            L = self._get_bw_l()
            f_shape = self._F_shape(data_c["|q|"], m, L)
        else:
            f_shape = one
        one_c = tf.complex(one, zero)
        basis = [
            one_c,
            tf.complex(m * m, zero),
            tf.complex(zero, -f_shape),
        ]
        return {"num": [one_c], "den": basis}

    def get_term_coeff(self):
        """Return the current coefficients for the rational terms."""
        g0 = self.get_width()
        if g0 is None:
            raise NotImplementedError(
                "{} ({}) has no width for get_term_coeff".format(
                    self, type(self).__name__
                )
            )
        m0 = self.get_mass()
        dtype = get_config("complex_dtype")
        rtype = get_config("dtype")
        c_g0 = tf.cast(g0, dtype)
        if self.running_width:
            L = self._get_bw_l()
            q0 = self._get_q0()
            f0 = self._F_shape(q0, tf.cast(m0, q0.dtype), L)
            eps = tf.cast(1e-15, f0.dtype)
            f0 = tf.where(
                tf.abs(f0) > eps,
                f0,
                tf.where(f0 < 0, -eps, eps),
            )
            f0 = tf.cast(f0, rtype)
        else:
            f0 = tf.cast(1.0, rtype)
        r_m0 = tf.cast(m0, rtype)
        r_g0 = tf.cast(g0, rtype)
        c0 = r_m0 * r_m0
        c1 = tf.cast(-1.0, rtype)
        c2 = r_m0 * r_g0 / f0
        num = c_g0 if self.width_norm else tf.cast(1.0, dtype)
        return {
            "num": [num],
            "den": [c0, c1, c2],
        }


@terms("BW")
class _BWTerms:

    def get_term_basis(self, data, data_c=None, **kwargs):
        m = data["m"]
        one = tf.ones_like(m)
        zero = tf.zeros_like(m)
        one_c = tf.complex(one, zero)
        basis = [
            one_c,
            tf.complex(m * m, zero),
            tf.complex(zero, -one),
        ]
        return {"num": [one_c], "den": basis}

    def get_term_coeff(self):
        g0 = self.get_width()
        if g0 is None:
            raise NotImplementedError(
                "{} ({}) has no width for get_term_coeff".format(
                    self, type(self).__name__
                )
            )
        m0 = self.get_mass()
        rtype = get_config("dtype")
        r_m0 = tf.cast(m0, rtype)
        r_g0 = tf.cast(g0, rtype)
        # ``get_amp`` always uses ``BW`` and ignores ``width_norm``,
        # so the decomposition must ignore it as well.
        c0 = r_m0 * r_m0
        c1 = tf.cast(-1.0, rtype)
        c2 = r_m0 * r_g0
        return {
            "num": [tf.cast(1.0, get_config("complex_dtype"))],
            "den": [c0, c1, c2],
        }

    def get_sympy_dom(self, m, m0, g0, m1=None, m2=None, sheet=0):
        from tf_pwa.formula import BW_dom

        return BW_dom(m, m0, g0)


@terms("one")
class _OneTerms:

    def get_term_basis(self, data, data_c=None, **kwargs):
        m = data["m"]
        one = tf.ones_like(m)
        zero = tf.zeros_like(m)
        basis = [tf.complex(one, zero)]
        return {"num": [basis[0]], "den": basis}

    def get_term_coeff(self):
        dtype = get_config("complex_dtype")
        return {
            "num": [tf.cast(1.0, dtype)],
            "den": [tf.cast(1.0, get_config("dtype"))],
        }


@terms("GS_rho")
class _GSTerms:

    def get_term_basis(self, data, data_c=None, **kwargs):
        m = data["m"]
        g0 = self.get_width()
        if g0 is None:
            return _DefaultTerms.get_term_basis(self, data, data_c, **kwargs)
        one = tf.ones_like(m)
        zero = tf.zeros_like(m)
        if not self.running_width:
            # ``get_amp`` falls back to the constant-width ``BW``.
            one_c = tf.complex(one, zero)
            basis = [
                one_c,
                tf.complex(m * m, zero),
                tf.complex(zero, -one),
            ]
            return {"num": [one_c], "den": basis}
        if data_c is None:
            return _DefaultTerms.get_term_basis(self, data, data_c, **kwargs)
        L = self.bw_l
        if L is None:
            L = min(self.decay[0].get_l_list())
        c2 = self.c_daug2Mass
        c3 = self.c_daug3Mass
        k2 = twoBodyCMmom(m, c2, c3) ** 2
        h_m = hFun(m * m, c2, c3)
        f_m = self._F_shape(data_c["|q|"], m, L)
        one_c = tf.complex(one, zero)
        basis = [
            one_c,
            tf.complex(m * m, zero),
            tf.complex(k2 * h_m, zero),
            tf.complex(k2, zero),
            tf.complex(zero, -f_m),
        ]
        return {"num": [one_c], "den": basis}

    def get_term_coeff(self):
        g0 = self.get_width()
        if g0 is None:
            return _DefaultTerms.get_term_coeff(self)
        m0 = self.get_mass()
        dtype = get_config("complex_dtype")
        rtype = get_config("dtype")
        if not self.running_width:
            # ``get_amp`` falls back to the constant-width ``BW``.
            return {
                "num": [tf.cast(1.0, dtype)],
                "den": [
                    tf.cast(m0 * m0, rtype),
                    tf.cast(-1.0, rtype),
                    tf.cast(m0 * g0, rtype),
                ],
            }
        c_m0 = tf.cast(m0, dtype)
        c_g0 = tf.cast(g0, dtype)
        r_m0 = tf.cast(m0, rtype)
        r_g0 = tf.cast(g0, rtype)
        c2 = self.c_daug2Mass
        c3 = self.c_daug3Mass
        L = self.bw_l
        if L is None:
            L = min(self.decay[0].get_l_list())
        k0 = twoBodyCMmom(m0, c2, c3)
        h0 = hFun(m0 * m0, c2, c3)
        dh0 = dh_dsFun(m0 * m0, c2, c3)
        q0 = self._get_q0()
        f0 = self._F_shape(q0, tf.cast(m0, q0.dtype), L)
        f0 = tf.cast(f0, rtype)
        a = r_g0 * r_m0 * r_m0 / tf.cast(k0**3, rtype)
        b = -a * tf.cast(h0, rtype)
        c = r_g0 * r_m0**4 * tf.cast(dh0, rtype) / tf.cast(k0, rtype)
        d = -r_g0 * r_m0 * r_m0 * tf.cast(dh0, rtype) / tf.cast(k0, rtype)
        num = (
            1.0
            + tf.cast(
                dFun(
                    tf.cast(m0 * m0, tf.float64),
                    c2,
                    c3,
                ),
                dtype,
            )
            * c_g0
            / c_m0
        )
        coeff = [
            r_m0 * r_m0 + c,
            tf.cast(-1.0, rtype) + d,
            a,
            b,
            r_m0 * r_g0 / f0,
        ]
        return {"num": [num], "den": coeff}


@terms("Flatte")
class _FlatteTerms:

    def get_term_basis(self, data, data_c=None, **kwargs):
        m = data["m"]
        one = tf.ones_like(m)
        zero = tf.zeros_like(m)
        basis = [tf.complex(one, zero), tf.complex(m * m, zero)]
        sign = tf.complex(tf.cast(self.im_sign, m.dtype) * one, zero)
        i_c = tf.complex(zero, one)
        for ma, mb in self.mass_list:
            p = cal_monentum(m, ma, mb)
            basis.append(sign * i_c * p / tf.cast(m, p.dtype))
        return {
            "num": [basis[0]],
            "den": basis,
        }

    def get_term_coeff(self):
        dtype = get_config("complex_dtype")
        rtype = get_config("dtype")
        m0 = tf.cast(self.get_mass(), rtype)
        coeff = [m0 * m0, tf.cast(-1.0, rtype)]
        for i in range(len(self.mass_list)):
            coeff.append(m0 * tf.cast(self.g_value[i](), rtype))
        return {"num": [tf.cast(1.0, dtype)], "den": coeff}


@terms("spline_c_idx", requires=("get_amp", "interp"))
class _SplineTerms:

    def spline_basis(self, m):
        """Return the spline basis matrix ``phi_i(m)``."""
        idx = self.get_bin_index(m)
        idx = tf.clip_by_value(idx, 0, self.h_matrix.shape[1] - 1)
        coeff = tf.gather(self.h_matrix, idx, axis=1)
        a, b, c, d = tf.unstack(coeff, axis=0)
        m = tf.expand_dims(m, axis=-1)
        return a + m * (b + m * (c + d * m))

    def get_term_basis(self, data, *args, **kwargs):
        m = data["m"]
        phi = self.spline_basis(m)
        n_term = phi.shape[-1]
        one = tf.ones_like(m, dtype=phi.dtype)
        return {
            "num": [phi[..., i] for i in range(n_term)],
            "den": [one],
        }

    def get_term_coeff(self):
        p = self.point_value()
        n_term = p.shape[0]
        one = tf.ones((), dtype=get_config("dtype"))
        return {
            "num": [p[i] for i in range(n_term)],
            "den": [one],
        }


# ---------------------------------------------------------------------------
# Decay and chain bases
# ---------------------------------------------------------------------------


@basis("AmpDecay")
class _AmpDecayBasis:
    def get_amplitude_basis(self, *args, **kwargs):
        """Return the data-only basis of this decay."""
        raise NotImplementedError(
            "{} ({}) does not implement get_amplitude_basis".format(
                self, type(self).__name__
            )
        )

    def get_amplitude_coeff(self):
        """Return the current LS coefficients."""
        raise NotImplementedError(
            "{} ({}) does not implement get_amplitude_coeff".format(
                self, type(self).__name__
            )
        )


@basis("HelicityDecay")
class _HelicityBasis:
    def get_barrier_basis(self, q, mass, d=None):
        """Data-only part of q^l B_l'(q, q_0, d) = basis(q) * coeff(q_0)."""
        if d is None:
            d = self.d
        ls = self.get_l_list()
        l_min = min(ls) if self.force_min_l else None
        ret = []
        for l in ls:
            if l_min is not None:
                l = l_min
            b = tf.ones_like(q)
            if self.has_barrier_factor:
                if self.has_bprime:
                    b = b / Bprime_num(l, q, d)
                if self.has_ql:
                    b = b * q**l
            if self.add_covariant_term:
                if (
                    l - self.core.J + self.outs[0].J + self.outs[1].J
                ) % 2 == 1:
                    b = b * mass
            if self.barrier_factor_mass:
                b = b / mass**l
            ret.append(b)
        return tf.stack(ret, axis=-1)

    def get_barrier_coeff(self, q0, d=None, q02=None):
        """Parameter-only coefficient F_l(q_0) (and q_0^{-l} if norm)."""
        if d is None:
            d = self.d
        if q02 is None:
            q02 = q0 * q0
        if self.no_q0:
            q0 = tf.ones_like(q0)
            q02 = tf.ones_like(q02)
        ls = self.get_l_list()
        l_min = min(ls) if self.force_min_l else None
        ret = []
        for l in ls:
            if l_min is not None:
                l = l_min
            c = tf.cast(1.0, q0.dtype)
            if self.has_barrier_factor:
                if self.has_bprime:
                    c = Bprime_num(l, q0, d)
                if self.has_ql and self.barrier_factor_norm:
                    c = c / tf.cast(tf.abs(q02), c.dtype) ** (l / 2)
            ret.append(c)
        return tf.stack(ret)

    def get_amplitude_basis(self, data, data_p, **kwargs):
        if (
            type(self).get_amp is not HelicityDecay.get_amp
            or type(self).get_helicity_amp
            is not HelicityDecay.get_helicity_amp
            or type(self).get_ls_amp is not HelicityDecay.get_ls_amp
            or self.aligned
        ):
            return _AmpDecayBasis.get_amplitude_basis(
                self, data, data_p, **kwargs
            )
        basis = self.get_factor_angle_amp(data, data_p, **kwargs)
        if self.has_barrier_factor:
            q = self.cache_relative_p(data, data_p)
            bf = self.get_barrier_basis(q, data_p[self.core]["m"], self.d)
            bf = tf.cast(bf, basis.dtype)
            bf = tf.reshape(
                bf,
                [-1, bf.shape[1]] + [1] * (len(basis.shape) - 2),
            )
            basis = basis * bf
        return {"basis": basis, "transform": None}

    def get_amplitude_coeff(self):
        if (
            type(self).get_amp is not HelicityDecay.get_amp
            or type(self).get_helicity_amp
            is not HelicityDecay.get_helicity_amp
            or type(self).get_ls_amp is not HelicityDecay.get_ls_amp
            or self.aligned
        ):
            return _AmpDecayBasis.get_amplitude_coeff(self)
        if self.no_q0:
            q0 = tf.cast(1.0, get_config("dtype"))
            q02 = q0
        else:
            q0 = self.get_relative_momentum({}, False)
            q02 = self.get_relative_momentum2({}, False)
        c = self.get_barrier_coeff(q0, self.d, q02=q02)
        g = self.get_g_ls()
        coeff = g * tf.cast(c, g.dtype)
        return {"coeff": list(tf.unstack(coeff))}


@basis("DecayChain")
class _DecayChainBasis:
    def _check_dense_chain_support(self, method):
        if type(self).get_amp is not DecayChain.get_amp:
            raise NotImplementedError(
                "{} ({}) does not implement {}".format(
                    getattr(self, "chain", type(self).__name__),
                    type(self).__name__,
                    method,
                )
            )

    def get_amplitude_basis(
        self, data_c, data_p, all_data=None, base_map=None
    ):
        self._check_dense_chain_support("get_amplitude_basis")
        base_map = self.get_base_map(base_map)
        iter_idx = ["..."]
        amp_d = []
        indices = []
        next_map = "zyxwvutsr"
        used_idx = ""
        final_indices = self.amp_index(base_map)
        for i in self:
            tmp_idx = i.amp_index(base_map)
            tmp_idx = [next_map[0], *tmp_idx]
            indices.append(tmp_idx)
            used_idx += next_map[0]
            amp_d.append(
                i.get_amplitude_basis(data_c[i], data_p, all_data=all_data)[
                    "basis"
                ]
            )
            next_map = next_map[1:]
        final_indices = "".join(iter_idx + list(used_idx) + final_indices)

        if self.aligned:
            from tf_pwa.dfun import get_D_matrix_lambda

            for i in self:
                for idxj, j in enumerate(i.outs):
                    if j.J != 0:
                        ang = data_c[i][j].get("aligned_angle", None)
                        if ang is None and not getattr(
                            i, "helicity_inner_full", False
                        ):
                            continue
                        dt = get_D_matrix_lambda(
                            ang,
                            j.J,
                            i.list_helicity_inner()[idxj],
                            j.spins,
                        )
                        amp_d.append(tf.stop_gradient(dt))
                        idx = [base_map[j], base_map[j].upper()]
                        indices.append(idx)
                        final_indices = final_indices.replace(*idx)

        idxs = []
        for i in indices:
            tmp = "".join(iter_idx + i)
            idxs.append(tmp)
        idx = ",".join(idxs)
        idx_s = "{}->{}".format(idx, final_indices)
        return tf.einsum(idx_s, *amp_d)

    def get_amplitude_coeff(self):
        self._check_dense_chain_support("get_amplitude_coeff")
        return [i.get_amplitude_coeff()["coeff"] for i in self]

    def get_particle_terms(self, data_p, data_c, all_data=None):
        """Return ``[(particle, basis, coeff), ...]`` for internal lines."""
        self._check_dense_chain_support("get_particle_terms")
        ret = []
        if not self.inner:
            return ret
        for i in self.inner:
            if len(i.decay) >= 1:
                decay_i = i.decay[0]
                found = False
                for j in i.decay:
                    if j in self:
                        decay_i = j
                        found = True
                        break
                if not found:
                    raise IndexError(
                        "not found {} decay in {}".format(i, self)
                    )
                data_c_i = data_c[decay_i]
                if "|q|" not in data_c_i:
                    data_c_i["|q|"] = decay_i.get_relative_momentum(
                        data_p, True
                    )
                if "|q0|" not in data_c_i:
                    data_c_i["|q0|"] = decay_i.get_relative_momentum(
                        data_p, False
                    )
                if "|q|2" not in data_c_i:
                    data_c_i["|q|2"] = decay_i.get_relative_momentum2(
                        data_p, True
                    )
                if "|q0|2" not in data_c_i:
                    data_c_i["|q0|2"] = decay_i.get_relative_momentum2(
                        data_p, False
                    )
                basis = i.get_term_basis(
                    data_p[i], data_c_i, all_data=all_data
                )
            else:
                basis = i.get_term_basis(data_p[i], all_data=all_data)
            ret.append((i, basis, i.get_term_coeff()))
        return ret


@basis("DecayGroup")
class _DecayGroupBasis:
    def get_dense_structure(self):
        """Return the dense basis description, building it on first use.

        It follows from the chains and their term models only, so the data
        side can ask for it whatever amplitude model is loaded.  There is no
        separate init: the first caller builds it, everyone afterwards gets
        the same dict, and a failure propagates to that caller without
        leaving a half-built basis behind.
        """
        structure = getattr(self, "_dense_structure", None)
        if structure is None:
            structure = self._build_dense_structure()
            self._dense_structure = structure
        return structure

    def get_particle_terms(self, data):
        """Return per-chain particle term basis and coefficients."""
        from tf_pwa.amp.core import rename_data_dict

        data_particle = data["particle"]
        data_decay = data["decay"]

        used_chains = tuple([self.chains[i] for i in self.chains_idx])
        chain_maps = self.get_chains_map(used_chains)
        ret = []
        for decay_chain in used_chains:
            for chains in chain_maps:
                if str(decay_chain) in [str(i) for i in chains]:
                    maps = chains[decay_chain]
                    break
            chain_topo = decay_chain.standard_topology()
            found = False
            for i in data_decay.keys():
                if i == chain_topo:
                    data_decay_i = data_decay[i]
                    found = True
                    break
            if not found:
                raise KeyError("not found {}".format(chain_topo))
            data_c = rename_data_dict(data_decay_i, maps)
            data_p = rename_data_dict(data_particle, maps)
            ret.append(
                decay_chain.get_particle_terms(data_p, data_c, all_data=data)
            )
        return ret

    def get_amplitude_basis(self, data):
        from tf_pwa.amp.core import rename_data_dict

        data_particle = data["particle"]
        data_decay = data["decay"]

        used_chains = tuple([self.chains[i] for i in self.chains_idx])
        chain_maps = self.get_chains_map(used_chains)
        base_map = self.get_base_map()
        ret = []
        for decay_chain in used_chains:
            for chains in chain_maps:
                if str(decay_chain) in [str(i) for i in chains]:
                    maps = chains[decay_chain]
                    break
            chain_topo = decay_chain.standard_topology()
            found = False
            for i in data_decay.keys():
                if i == chain_topo:
                    data_decay_i = data_decay[i]
                    found = True
                    break
            if not found:
                raise KeyError("not found {}".format(chain_topo))
            data_c = rename_data_dict(data_decay_i, maps)
            data_p = rename_data_dict(data_particle, maps)
            amp = decay_chain.get_amplitude_basis(
                data_c, data_p, base_map=base_map, all_data=data
            )
            ret.append(amp)
        return ret

    def get_amplitude_coeff(self):
        return [self.chains[i].get_amplitude_coeff() for i in self.chains_idx]

    def _build_dense_structure(self):
        """Build the dense basis description; called once, from the getter.

        Everything here follows from the chains and their term models, so it
        is the group's data, shared by every model over this group.  Only the
        first dense model (or the preprocessor) pays for it, and an
        unsupported term model is rejected there rather than on behalf of a
        stock model that never asks.
        """
        chains = list(self.chains)

        u_chain = []
        u_ls = []
        u_term = []
        # the chains themselves stay on the decay group; only the layout of
        # the denominator-coefficient vector is stored
        den_sizes = []
        param_offset = []
        max_internal_lines = 0

        off = 0
        for k, chain in enumerate(chains):
            decays = list(chain)
            lines = list(chain.inner)
            n_ls = [len(d.get_ls_list()) for d in decays]
            # a line whose model has no rational basis makes the whole
            # model unsupported, so fail here with the term's own message
            # instead of leaving a husk that raises "not initialized" only
            # when the first batch arrives
            terms = [p.get_term_coeff() for p in lines]
            n_term = [len(t["num"]) for t in terms]
            sizes = [len(t["den"]) for t in terms]
            offsets = []
            for size in sizes:
                offsets.append(off)
                off += size
            den_sizes.append(sizes)
            param_offset.append(offsets)

            max_internal_lines = max(max_internal_lines, len(lines))

            ls_combos = (
                list(itertools.product(*[range(i) for i in n_ls]))
                if n_ls
                else [()]
            )
            term_combos = (
                list(itertools.product(*[range(i) for i in n_term]))
                if n_term
                else [()]
            )
            for ls in ls_combos:
                for term in term_combos:
                    u_chain.append(k)
                    u_ls.append(list(ls))
                    u_term.append(list(term))

        structure = {
            "den_sizes": den_sizes,
            "param_offset": param_offset,
            "u_chain": u_chain,
            "u_ls": u_ls,
            "u_term": u_term,
            "n_param": off,
            "n_u": len(u_chain),
            "max_internal_lines": max_internal_lines,
            # the helicity axes are a particle property, one per top/final,
            # so they are known here rather than when the first F is built
            "f_shape": (len(self.top.spins),)
            + tuple(len(p.spins) for p in self.outs),
            # likewise the CP transform, which only walks the group
            "cp": self._cp_transform(),
        }
        # the basis transform of B; kept next to the terms that apply it
        self.M2 = tf.eye(off, dtype=tf.float64)
        return structure

    def swap_terms(self, F, key):
        """Sign and particle-axis transpose of one swap's ``F`` block.

        ``get_amp2`` adds ``factor * transpose(amp_swapped)``, permuting the
        particle (helicity) axes.  ``F`` is ``[ne, prod(f_shape), nu]``, so
        the permutation is applied to ``np`` with the chain axis kept last;
        transposing the basis transposes the contraction.  The preprocessor
        stores the result, so ``pdf`` only sums prepared blocks.
        """
        f_shape = list(self.get_dense_structure()["f_shape"])
        n_axis = 1 + len(f_shape)
        perm = list(self.get_id_swap_transpose(key, n_axis)) + [n_axis]
        shape = tf.shape(F)
        out = tf.reshape(F, [shape[0]] + f_shape + [shape[2]])
        out = tf.transpose(out, perm)
        out = tf.reshape(out, shape)
        factor = tf.cast(float(self.get_swap_factor(key)), F.dtype)
        return out * factor

    def build_dense_terms(self, data):
        structure = self.get_dense_structure()
        old_chains_idx = self.chains_idx
        self.chains_idx = list(range(len(self.chains)))
        try:
            F_chains = self.get_amplitude_basis(data)
            particle_terms = self.get_particle_terms(data)
        finally:
            self.chains_idx = old_chains_idx

        F_list = []
        B_blocks = []
        n_data = tf.shape(F_chains[0])[0]

        for idx, chain in enumerate(self.chains):
            F_chain = F_chains[idx]
            decays = list(chain)
            lines = list(chain.inner)
            n_ls = [len(d.get_ls_list()) for d in decays]
            n_term = [len(term[1]["num"]) for term in particle_terms[idx]]

            n_ls_combo = int(np.prod(n_ls)) if n_ls else 1
            n_decay = len(decays)
            f_shape = list(F_chain.shape[1 + n_decay :])
            if any(i is None for i in f_shape):
                raise NotImplementedError(
                    "dense_factor needs a static helicity shape"
                )
            shape_t = tuple(int(i) for i in f_shape)
            if structure["f_shape"] != shape_t:
                raise NotImplementedError(
                    "dense_factor needs one helicity shape for every chain"
                )
            if len(shape_t) != 1 + len(self.outs):
                raise NotImplementedError(
                    "dense_factor needs one helicity axis per particle "
                    "({} for {})".format(1 + len(self.outs), shape_t)
                )
            F_ls = tf.reshape(F_chain, [n_data, n_ls_combo] + f_shape)

            term_combos = (
                list(itertools.product(*[range(i) for i in n_term]))
                if n_term
                else [()]
            )
            F_num = []
            for combo in term_combos:
                f = tf.ones((n_data,), dtype=F_chain.dtype)
                for l, t in enumerate(combo):
                    f = f * tf.cast(
                        particle_terms[idx][l][1]["num"][t],
                        F_chain.dtype,
                    )
                F_num.append(f)

            for ls_idx in range(n_ls_combo):
                base = F_ls[:, ls_idx]
                shape = [n_data] + [1] * (len(base.shape) - 1)
                for f in F_num:
                    F_list.append(base * tf.reshape(f, shape))

            for l in range(len(lines)):
                B_blocks.append(particle_terms[idx][l][1]["den"])

        if F_list:
            F = tf.stack(F_list, axis=1)
            F = tf.reshape(F, [n_data, len(F_list), -1])
            F = tf.transpose(F, [0, 2, 1])
        else:
            F = tf.zeros((n_data, 0, 0), dtype=F_chains[0].dtype)
        B_cols = [
            tf.cast(col, F_chains[0].dtype)
            for block in B_blocks
            for col in block
        ]
        if B_cols:
            B = tf.stack(B_cols, axis=-1)
            B = tf.matmul(B, tf.cast(self.M2, B.dtype), transpose_b=True)
        else:
            B = tf.zeros((n_data, 0), dtype=F_chains[0].dtype)
        dtype = get_config("complex_dtype")
        # Event-first: (n_data, n_param).  Batching slices the event axis
        # contiguously and the fused kernel reads one event's basis block
        # without a transpose.  The sparse fallback below transposes it;
        # that cost belongs to the fallback, not to the stored layout.
        return tf.cast(F, dtype), tf.cast(B, dtype)

    def _cp_transform(self):
        """``(frac, change)`` for the CP-conjugate amplitude, like get_amp3.

        ``frac`` is the product of the C parities of the particles that the
        CP map leaves in place; ``change`` is the particle permutation of the
        ones that move.
        """
        if not getattr(self, "cp_particles", None):
            return 1.0, ()
        name_map = {str(i): i for i in self.outs}
        frac = 1.0
        change = []
        for a, b in cp_charge_group(
            [str(i) for i in self.outs],
            self.identical_particles,
            self.cp_particles,
        ):
            for i, j in zip(a, b):
                if i == j:
                    frac = frac * getattr(name_map[i], "C", -1)
                else:
                    change.append((i, j))
        return float(frac), tuple(change)

    def build_dense_coeff(self, structure):
        """Build numerator and denominator coefficients for a term cache."""
        C = []

        chain_c = []
        ls_vecs = []
        num_vecs = []
        den_vecs = []
        for chain in self.chains:
            decays = list(chain)
            lines = list(chain.inner)
            chain_c.append(tf.reshape(chain.get_amp_total(), (-1,))[0])
            ls_vecs.append([d.get_amplitude_coeff()["coeff"] for d in decays])
            term_coeffs = [p.get_term_coeff() for p in lines]
            num_vecs.append([c["num"] for c in term_coeffs])
            den_vecs.append([c["den"] for c in term_coeffs])

        dtype = get_config("complex_dtype")
        for u, k in enumerate(structure["u_chain"]):
            c = tf.cast(chain_c[k], dtype)
            for d, ls_idx in enumerate(structure["u_ls"][u]):
                c = c * tf.cast(ls_vecs[k][d][ls_idx], dtype)
            for l, term_idx in enumerate(structure["u_term"][u]):
                c = c * tf.cast(num_vecs[k][l][term_idx], dtype)
            C.append(tf.cast(c, dtype))

        g_blocks = [g for row in den_vecs for block in row for g in block]
        if C:
            C = tf.stack(C)
        else:
            C = tf.zeros((0,), dtype=dtype)
        # the denominator coefficients are real; only C is complex
        rtype = get_config("dtype")
        if g_blocks:
            g = tf.cast(tf.stack(g_blocks), rtype)
        else:
            g = tf.zeros((0,), dtype=rtype)
        return C, g


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

    ``store_dtype`` picks the precision the two tensors are cached in:
    ``complex128`` (default), ``complex64`` (about 1e-7 relative) or
    ``complex32`` (two fp16 halves, about 5e-4).  They are converted back to
    the coefficient dtype as soon as ``pdf`` reads them, before any product,
    so the graph the gradients traverse stays at full precision and the
    arithmetic is always complex128.
    """

    def __init__(
        self,
        *args,
        force_cpu=False,
        drop_decay_particle=False,
        store_dtype="complex128",
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
        self.gpu_device = None
        if tf.config.list_physical_devices("GPU"):
            self.gpu_device = "/GPU:0"

    def call(self, x, **kwargs):
        if self.force_cpu:
            ctx = tf.device("/CPU:0")
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            x = BasePreProcessor.call(self, x, **kwargs)
            x["F"], x["B"] = self.decay_group.build_dense_terms(x)
            self._fill_blocks(x)
            cp = x.get("cp_swap", None)
            blocks = list((x.get("id_swap") or {}).values())
            if cp is not None:
                cp["F"], cp["B"] = self.decay_group.build_dense_terms(cp)
                self._fill_blocks(cp)
                self._strip(cp)
                blocks += [cp] + list((cp.get("id_swap") or {}).values())
            # a subclass may reduce the blocks here, while they are still
            # on the CPU at full precision
            self._after_blocks(x, blocks)
            # every block follows the same two rules as the base dict
            x["F"], x["B"] = self._cast(x["F"], x["B"])
            for d in blocks:
                d["F"], d["B"] = self._cast(d["F"], d["B"])

        x["F"], x["B"] = self._move(x["F"], x["B"])
        for d in blocks:
            d["F"], d["B"] = self._move(d["F"], d["B"])
        self._strip(x)
        return x

    def _after_blocks(self, x, blocks):
        """Hook for subclasses, called before the cast and the GPU move."""

    def _cast(self, F, B):
        """Lower the stored precision, inside the CPU scope."""
        if self.store_dtype == "complex64":
            return tf.cast(F, tf.complex64), tf.cast(B, tf.complex64)
        if self.store_dtype == "complex32":
            return _pack_complex16(F), _pack_complex16(B)
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

        Applies the parity factor and the particle-axis transpose, so the
        model only has to sum the prepared blocks.  ``cp_swap`` is handled
        by the caller, which calls this again on that dict.
        """
        for key, sub in (d.get("id_swap") or {}).items():
            Fs, Bs = self.decay_group.build_dense_terms({**d, **sub})
            sub["F"] = self.decay_group.swap_terms(Fs, key)
            sub["B"] = Bs
            self._strip(sub)


@register_amp_model("dense_factor")
class DenseFactorModel(BaseAmplitudeModel):
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
                amp = amp + block(sub)
            return amp

        amp = sum_swaps(data)
        cp = data.get("cp_swap", None)
        if cp is not None and dg.cp_particles:
            # the conjugated kinematics carry their own identical-particle
            # swaps, so it is the same sum on the cp_swap dict
            amp = amp + self.cp_amplitude(sum_swaps(cp))
        return dg.sum_with_polarization(amp)

    def expand_block(self, F, B):
        """Hook for models that store the basis merged."""
        return F, B

    def _amplitude(self, F, B, C, g, dtype):
        """``A[e, p] = sum_u F[e, p, u] C[u] / den[u, e]`` for one block."""
        structure = self.decay_group.get_dense_structure()
        F, B = self.expand_block(F, B)
        F = _unpack_terms(F, dtype)
        B = _unpack_terms(B, dtype)
        n_u = structure["n_u"]
        n_line = structure["max_internal_lines"]
        if n_line == 0:
            D = tf.ones((tf.shape(F)[0], n_u), dtype=dtype)
        else:
            values = tf.gather(tf.cast(g, B.dtype), self.sparse_index[:, 1])
            sp = tf.sparse.SparseTensor(
                self.sparse_index, values, self.sparse_shape
            )
            # B is stored event-first; the sparse matmul consumes it
            # param-major, so the fallback pays the transpose.
            D = tf.transpose(
                tf.sparse.sparse_dense_matmul(sp, tf.transpose(B))
            )
            if n_line == 1 and self.line_mask_all:
                D = tf.reshape(D, [tf.shape(F)[0], n_u])
            else:
                D = tf.reshape(D, [tf.shape(F)[0], n_u, n_line])
                D = tf.where(self.line_mask[None, :, :], D, tf.ones_like(D))
                D = tf.reduce_prod(D, axis=-1)

        W = tf.math.divide_no_nan(C[None, :], D)
        return tf.linalg.matvec(F, W)

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


def _unpack_terms(x, dtype):
    """Read a stored ``F``/``B`` as complex, whatever ``store_dtype`` is."""
    if x.dtype == tf.float16:  # complex32: last axis holds (real, imag)
        part = tf.float32 if dtype == tf.complex64 else tf.float64
        return tf.complex(tf.cast(x[..., 0], part), tf.cast(x[..., 1], part))
    return tf.cast(x, dtype)


def _expand_svd(c, F, B):
    """Rebuild the full basis from the stored low-rank coefficients."""
    F = _unpack_terms(F, c["Vf"].dtype)
    B = _unpack_terms(B, c["Vb"].dtype)
    M = tf.matmul(F, c["Vf"])  # [ne,k] x [k,np*nu]
    M = tf.reshape(M, (tf.shape(M)[0], c["np"], c["nu"]))
    return M, tf.matmul(B, c["Vb"])


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


class _IndexCompression:
    """Store one representative per group of identical blocks."""

    name = "index"

    def __init__(self, rtol=0.0, atol=0.0, sample=0):
        self.rtol = rtol
        self.atol = atol
        self.sample = sample

    def prepare(self, F, B):
        record = _find_merge(
            F, B, probe=self.sample, rtol=self.rtol, atol=self.atol
        )
        record["method"] = self.name
        return record

    def compress(self, record, F, B):
        if F is not None:
            F = tf.gather(F, record["rep_u"], axis=2)
        if B is not None:
            B = tf.gather(B, record["rep_k"], axis=1)
        return F, B

    def expand(self, record, F, B):
        return _expand_block(record, F, B)

    def contract(self, record, F, B, C, g, off, sz, *, n_line, max_basis):
        from .dense_factor_op import (
            dense_factor_merge_a,
            dense_factor_merge_an,
        )

        fn = dense_factor_merge_a if n_line == 1 else dense_factor_merge_an
        return fn(
            F,
            B,
            C,
            g,
            off,
            sz,
            record["map_u"],
            record["map_i"],
            max_basis=max_basis,
        )


class _SvdCompression:
    """Store the low-rank coefficients ``U = F V^H``."""

    name = "svd"

    def __init__(self, rtol=0.0, rank=None):
        self.rtol = rtol
        self.rank = rank

    def _basis(self, M):
        """Right singular vectors above ``rtol`` (or the first ``rank``)."""
        s, u, v = tf.linalg.svd(M, full_matrices=False)
        if self.rank is not None:
            k = min(self.rank, int(s.shape[0]))
        else:
            k = int(tf.reduce_sum(tf.cast(s > self.rtol * s[0], tf.int32)))
            k = max(k, 1)
        # tf.linalg.svd returns v = V for complex (M = u s v^H); store V^H so
        # "row j is the j-th right singular vector" everywhere else
        return tf.linalg.adjoint(v)[:k], k

    def prepare(self, F, B):
        Vf, rf = self._basis(tf.reshape(F, (tf.shape(F)[0], -1)))
        Vb, rb = self._basis(B)
        return {
            "method": self.name,
            "Vf": Vf,
            "Vb": Vb,
            "np": int(F.shape[1]),
            "nu": int(F.shape[2]),
            "nparam": int(B.shape[1]),
            "rank_f": rf,
            "rank_b": rb,
        }

    def compress(self, record, F, B):
        if F is not None:
            M = tf.reshape(F, (tf.shape(F)[0], -1))
            F = tf.matmul(M, tf.linalg.adjoint(record["Vf"]))
        if B is not None:
            B = tf.matmul(B, tf.linalg.adjoint(record["Vb"]))
        return F, B

    def expand(self, record, F, B):
        return _expand_svd(record, F, B)

    def contract(self, record, F, B, C, g, off, sz, *, n_line, max_basis):
        # the stored coefficients are rebuilt here, then the plain op runs
        F, B = _expand_svd(record, F, B)
        from .dense_factor_op import dense_factor_a, dense_factor_an

        fn = dense_factor_a if n_line == 1 else dense_factor_an
        return fn(F, B, C, g, off, sz, max_basis=max_basis)


class _NoCompression:
    """Nothing stored reduced: the basis is already full width."""

    name = None

    def contract(self, record, F, B, C, g, off, sz, *, n_line, max_basis):
        from .dense_factor_op import dense_factor_a, dense_factor_an

        fn = dense_factor_a if n_line == 1 else dense_factor_an
        return fn(F, B, C, g, off, sz, max_basis=max_basis)


_COMPRESSIONS = {
    None: _NoCompression,
    "index": _IndexCompression,
    "svd": _SvdCompression,
}


def _compression_of(record):
    """The strategy that owns a record; ``None``/``False`` mean none."""
    if not record:  # no record, or the failure sentinel
        return _NoCompression()
    return _COMPRESSIONS[record["method"]]()


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
              method: index             # or svd
              svd_rank: null

    ``method`` picks what is stored: ``index`` keeps one representative per
    group of identical blocks (``merge_rtol``/``merge_atol``), ``svd`` keeps
    the low-rank coefficients ``U = F V^H`` with the rank taken from
    ``merge_rtol`` (relative singular value) or ``svd_rank``.  The SVD is exact
    at its numerical rank and is rebuilt before the contraction; a tolerance
    that drops real directions must be validated on the amplitude, because
    ``1/den`` amplifies the discarded ones.

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
    rather than failing.  ``merge_rtol`` defaults to null, i.e. automatic and
    fixed at init from ``store_dtype``: ``1e-8``, or ``8*eps`` when the
    tensors are stored in complex64 (about ``1e-6``) or complex32 (about
    ``8e-3``), because the comparison can only be as fine as the stored
    values.  Give a number to override it, e.g. for a basis limited by
    cancellation; ``merge_atol`` adds a fixed floor to the index comparison
    (it does not affect the SVD cut).  ``store_dtype`` caches the merged
    blocks at that precision and converts them back before any product.
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
        if method not in ("index", "svd"):
            raise ValueError(
                "method must be 'index' or 'svd', got {!r}".format(method)
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
        self.compression = self._make_compression()
        if merge_prepare:
            self._prepare_merge()

    def _make_compression(self):
        if self.method == "index":
            return _IndexCompression(
                self.merge_rtol, self.merge_atol, self.merge_sample
            )
        return _SvdCompression(self.merge_rtol, self.svd_rank)

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
        data = cal_angle_from_momentum(p4, self.decay_struct)
        return self.decay_group.build_dense_terms(data)

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
        if getattr(self.amp, "merge", None) is not None:
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
            self.amp.merge = False
            return
        self.amp.merge = self.compression.prepare(F, B)

    def _after_blocks(self, x, blocks):
        merge = getattr(self.amp, "merge", None)
        if merge is None:
            if not tf.executing_eagerly():
                raise RuntimeError(
                    "dense_factor_merge maps were not built before the graph "
                    "was traced; construct the preprocessor eagerly"
                )
            self._prepare_merge(x.get("F"), x.get("B"))
            merge = self.amp.merge
        if not merge:
            return

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
    """

    #: compression record, set by the preprocessor
    merge = None

    def expand_block(self, F, B):
        c = self.merge
        if not c:  # nothing was reduced
            return F, B
        return _compression_of(c).expand(c, F, B)


# ---------------------------------------------------------------------------
# The methods above attach themselves when the decorated classes are defined.
# ---------------------------------------------------------------------------
