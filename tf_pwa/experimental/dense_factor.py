"""Dense factorized amplitude model plugin.

Importing this module attaches the dense term interface to the original
tf-pwa classes, so the upstream source files stay untouched::

    import tf_pwa.experimental.dense_factor

The amplitude is written as

.. math::

    A_{ij} = \\sum_u C_u \\frac{F_{u,ij}(x)}
             {\\prod_l \\sum_{m,n} M1_{u,l,m} g_m M2_{m,n} B_{l,n}(x)}

The preprocessor produces the data-only ``F`` and ``B`` tensors.  The
amplitude model builds the structural ``M1`` and ``M2`` matrices in
``init_params``.  All coefficients are rebuilt on every call.

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

    def build_dense_coeff(self, structure):
        """Build numerator and denominator coefficients for a term cache."""
        C = []

        chain_c = []
        ls_vecs = []
        num_vecs = []
        den_vecs = []
        for meta in structure["chain_meta"]:
            chain_c.append(tf.reshape(meta["chain"].get_amp_total(), (-1,))[0])
            ls_vecs.append(
                [d.get_amplitude_coeff()["coeff"] for d in meta["decays"]]
            )
            term_coeffs = [p.get_term_coeff() for p in meta["lines"]]
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
              store_complex64: false

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

    With ``store_complex64``, ``F`` and ``B`` are stored as complex64.  They
    are cast back to the coefficient dtype as soon as ``pdf`` reads them,
    before any product, so the graph the gradients traverse stays at full
    precision.
    That halves the cached payload and the host-to-device copy; the price is
    complex64 rounding of the two tensors (about 1e-7 relative), not of the
    arithmetic, which still runs in complex128.
    """

    def __init__(
        self,
        *args,
        force_cpu=False,
        drop_decay_particle=False,
        store_complex64=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.force_cpu = force_cpu
        self.drop_decay_particle = drop_decay_particle
        self.store_complex64 = store_complex64
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
            structure = getattr(self.amp, "structure", None)
            if structure is None:
                # Trigger the particle-specific NotImplementedError.
                self.decay_group.get_particle_terms(x)
                return x
            x["F"], x["B"] = self.amp.build_dense_terms(x)
            self._fill_blocks(x)
            cp = x.get("cp_swap", None)
            blocks = list((x.get("id_swap") or {}).values())
            if cp is not None:
                cp["F"], cp["B"] = self.amp.build_dense_terms(cp)
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
        if self.store_complex64:
            return tf.cast(F, tf.complex64), tf.cast(B, tf.complex64)
        return F, B

    def _move(self, F, B):
        """Move the final tensors to the GPU when ``force_cpu`` asks for it."""
        if self.force_cpu and self.gpu_device is not None:
            return tf.identity(F), tf.identity(B)
        return F, B

    def _strip(self, d):
        """Drop the angle tree when the option asks for it."""
        if self.drop_decay_particle:
            d.pop("particle", None)
            d.pop("decay", None)

    def _fill_blocks(self, d):
        """Build ``F``/``B`` for every ``id_swap`` block of ``d``, in place.

        Applies the parity factor and the particle-axis transpose, so the
        model only has to sum the prepared blocks.  ``cp_swap`` is handled
        by the caller, which calls this again on that dict.
        """
        for key, sub in (d.get("id_swap") or {}).items():
            Fs, Bs = self.amp.build_dense_terms({**d, **sub})
            sub["F"] = self.amp.swap_terms(Fs, key)
            sub["B"] = Bs
            self._strip(sub)


@register_amp_model("dense_factor")
class DenseFactorModel(BaseAmplitudeModel):
    def init_params(self, name=""):
        super().init_params(name)
        self.structure = None
        dg = self.decay_group
        chains = list(dg.chains)

        chain_meta = []
        u_chain = []
        u_ls = []
        u_term = []
        max_internal_lines = 0

        for k, chain in enumerate(chains):
            decays = list(chain)
            lines = list(chain.inner)
            n_ls = [len(d.get_ls_list()) for d in decays]
            try:
                terms = [p.get_term_coeff() for p in lines]
            except NotImplementedError:
                return
            n_term = [len(t["num"]) for t in terms]
            den_sizes = [len(t["den"]) for t in terms]

            n_internal_lines = len(lines)
            max_internal_lines = max(max_internal_lines, n_internal_lines)

            chain_meta.append(
                {
                    "chain": chain,
                    "decays": decays,
                    "lines": lines,
                    "n_ls": n_ls,
                    "n_term": n_term,
                    "den_sizes": den_sizes,
                }
            )

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

        n_param = sum(sum(meta["den_sizes"]) for meta in chain_meta)
        n_u = len(u_chain)

        off = 0
        for meta in chain_meta:
            offsets = []
            for size in meta["den_sizes"]:
                offsets.append(off)
                off += size
            meta["param_offset"] = offsets

        M1 = np.zeros((n_u, max_internal_lines, n_param), dtype=np.float64)
        line_mask = np.zeros((n_u, max_internal_lines), dtype=bool)

        for u, k in enumerate(u_chain):
            meta = chain_meta[k]
            for l, (size, poff) in enumerate(
                zip(meta["den_sizes"], meta["param_offset"])
            ):
                M1[u, l, poff : poff + size] = 1.0
                line_mask[u, l] = True

        self.structure = {
            "chain_meta": chain_meta,
            "u_chain": u_chain,
            "u_ls": u_ls,
            "u_term": u_term,
            "n_param": n_param,
            "n_u": n_u,
            "max_internal_lines": max_internal_lines,
        }
        self.M1 = tf.constant(M1, dtype=tf.float64)
        coo = np.nonzero(M1.reshape(n_u * max_internal_lines, n_param))
        self.sparse_index = tf.constant(np.stack(coo, axis=1), dtype=tf.int64)
        self.sparse_shape = (n_u * max_internal_lines, n_param)
        self.M1T = tf.reshape(
            tf.transpose(self.M1, [2, 0, 1]),
            [n_param, n_u * max_internal_lines],
        )
        self.M2 = tf.eye(n_param, dtype=tf.float64)
        self.line_mask = tf.constant(line_mask)
        self.line_mask_all = bool(np.all(line_mask))

        self._coeff_fn = None
        if n_u > 0 and n_param > 0:
            dg = self.decay_group
            structure = self.structure

            def _build_coeff():
                return dg.build_dense_coeff(structure)

            try:
                self._coeff_fn = tf.function(_build_coeff)
            except Exception:
                self._coeff_fn = None

    def cp_transform(self):
        """``(frac, change)`` for the CP-conjugate amplitude, like get_amp3.

        ``frac`` is the product of the C parities of the particles that the
        CP map leaves in place; ``change`` is the particle permutation of the
        ones that move.
        """
        structure = self.structure
        dg = self.decay_group
        if not getattr(dg, "cp_particles", None):
            return 1.0, ()
        if "cp" not in structure:
            name_map = {str(i): i for i in dg.outs}
            frac = 1.0
            change = []
            for a, b in cp_charge_group(
                [str(i) for i in dg.outs],
                dg.identical_particles,
                dg.cp_particles,
            ):
                for i, j in zip(a, b):
                    if i == j:
                        frac = frac * getattr(name_map[i], "C", -1)
                    else:
                        change.append((i, j))
            structure["cp"] = (float(frac), tuple(change))
        return structure["cp"]

    def cp_amplitude(self, amp_cp):
        """Transpose and helicity-reverse the CP block, times ``frac``."""
        dg = self.decay_group
        structure = self.structure
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

    def swap_terms(self, F, key):
        """Sign and particle-axis transpose of one swap's ``F`` block.

        ``get_amp2`` adds ``factor * transpose(amp_swapped)``, permuting the
        particle (helicity) axes.  ``F`` is ``[ne, prod(f_shape), nu]``, so
        the permutation is applied to ``np`` with the chain axis kept last;
        transposing the basis transposes the contraction.  The preprocessor
        stores the result, so ``pdf`` only sums prepared blocks.
        """
        dg = self.decay_group
        f_shape = list(self.structure["f_shape"])
        n_axis = 1 + len(f_shape)
        perm = list(dg.get_id_swap_transpose(key, n_axis)) + [n_axis]
        shape = tf.shape(F)
        out = tf.reshape(F, [shape[0]] + f_shape + [shape[2]])
        out = tf.transpose(out, perm)
        out = tf.reshape(out, shape)
        factor = tf.cast(float(dg.get_swap_factor(key)), F.dtype)
        return out * factor

    def build_dense_terms(self, data):
        structure = self.structure
        dg = self.decay_group

        old_chains_idx = dg.chains_idx
        dg.chains_idx = list(range(len(dg.chains)))
        try:
            F_chains = dg.get_amplitude_basis(data)
            particle_terms = dg.get_particle_terms(data)
        finally:
            dg.chains_idx = old_chains_idx

        F_list = []
        B_blocks = []
        n_data = tf.shape(F_chains[0])[0]

        for idx, meta in enumerate(structure["chain_meta"]):
            F_chain = F_chains[idx]
            decays = meta["decays"]
            lines = meta["lines"]
            n_ls = meta["n_ls"]
            n_term = meta["n_term"]

            n_ls_combo = int(np.prod(n_ls)) if n_ls else 1
            n_decay = len(decays)
            f_shape = list(F_chain.shape[1 + n_decay :])
            if any(i is None for i in f_shape):
                raise NotImplementedError(
                    "dense_factor needs a static helicity shape"
                )
            shape_t = tuple(int(i) for i in f_shape)
            if "f_shape" not in structure:
                structure["f_shape"] = shape_t
            elif structure["f_shape"] != shape_t:
                raise NotImplementedError(
                    "dense_factor needs one helicity shape for every chain"
                )
            if len(shape_t) != 1 + len(dg.outs):
                raise NotImplementedError(
                    "dense_factor needs one helicity axis per particle "
                    "({} for {})".format(1 + len(dg.outs), shape_t)
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

    def _get_coeff(self):
        """Return the numerator ``C`` and denominator ``g`` coefficients."""
        dg = self.decay_group
        structure = self.structure
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

    def pdf(self, data):
        dg = self.decay_group
        structure = self.structure
        if structure is None:
            dg.get_particle_terms(data)
            raise NotImplementedError("dense_factor is not initialized")

        C, g = self._get_coeff()
        # The preprocessor may store F and B at lower precision.  Cast before
        # any product, so the graph the gradients traverse stays at full
        # precision; _get_coeff does not read F or B, so the order is free.
        # C is complex by construction while g only happens to be, so the
        # numerator decides the dtype.
        dtype = C.dtype
        used = set(dg.chains_idx)
        mask = tf.constant(
            [1.0 if k in used else 0.0 for k in structure["u_chain"]],
            dtype=dtype,
        )
        C = C * mask

        def block(d):
            F, B = d.get("F", None), d.get("B", None)
            if F is None or B is None:
                F, B = self.build_dense_terms(d)
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

    def _amplitude(self, F, B, C, g, dtype):
        """``A[e, p] = sum_u F[e, p, u] C[u] / den[u, e]`` for one block."""
        structure = self.structure
        F = tf.cast(F, dtype)
        B = tf.cast(B, dtype)
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
        structure = self.structure
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
    }


def _expand_block(structure, F, B):
    """Expand a stored block with the merge index.

    Group ids index the reduced tensors, so only a reduced block is gathered;
    a full-width one is passed through unchanged.
    """
    merge = structure.get("merge")
    if merge is None:
        raise ValueError(
            "dense_factor_merge needs the dense_factor_merge preprocessor "
            "to build the basis maps"
        )
    if int(F.shape[2]) != int(structure["n_u"]):
        F = tf.gather(F, merge["map_u"], axis=2)
    if int(B.shape[1]) != int(structure["n_param"]):
        B = tf.gather(B, merge["map_i"], axis=1)
    return F, B


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

    The maps are found from a phase-space sample generated by this
    preprocessor (``merge_sample`` events, default 256).  With
    ``lazy_call`` they are built in ``__init__``, because the tensor widths
    are fixed when the map function is traced; with eager data they are
    built from the first batch instead, so construction stays cheap and
    does not consume the RNG.  ``merge_prepare`` overrides that choice:
    ``true`` always builds at init, ``false`` always waits for the data.
    If the sample cannot be built the merge is disabled with a warning
    rather than failing.  ``merge_rtol`` defaults to null, i.e. automatic and
    fixed at init from ``store_complex64``: ``1e-8``, or ``8*eps`` (about
    ``1e-6``) when the tensors will be stored as complex64, because the
    comparison can only be as fine as the stored values.  Give a number to
    override it, e.g. for a basis limited by cancellation; ``merge_atol``
    adds a fixed floor.
    """

    def __init__(
        self,
        *args,
        merge_rtol=None,
        merge_atol=0.0,
        merge_sample=256,
        merge_prepare=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if merge_rtol is None:
            # the comparison can only be as fine as what will be stored; the
            # flag, not the tensor, decides that, so this is fixed at init
            if self.store_complex64:
                merge_rtol = 8 * float(np.finfo(np.complex64).eps)
            else:
                merge_rtol = 1e-8
        self.merge_rtol = float(merge_rtol)
        self.merge_atol = float(merge_atol)
        self.merge_sample = int(merge_sample)
        if merge_prepare is None:
            # only a lazy stream needs the maps before its map function is
            # traced; eager data can discover them from its first batch
            config = getattr(self.root_config, "config", None) or {}
            merge_prepare = bool(config.get("data", {}).get("lazy_call"))
        if merge_prepare:
            self._prepare_merge()

    def _prepare_merge(self):
        """Discover the maps from a small generated sample, once.

        A generated phase-space sample is eager and well spread, unlike the
        head of the data, and it needs no real events - which is what makes
        the maps available before ``lazy_call`` traces its map function.  It
        draws from the global TensorFlow RNG, so a later event generation
        with a fixed seed is shifted relative to a model without this
        preprocessor.
        """
        amp = self.amp
        structure = getattr(amp, "structure", None)
        if not isinstance(structure, dict):
            # init_params bailed on an unsupported model; that is reported
            # by the model itself at use time
            return
        if structure.get("merge") is not None:
            return
        if self.merge_sample < 1:
            raise ValueError("merge_sample must be >= 1")
        dg = self.decay_group
        try:
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
            mom = gen.generate(self.merge_sample)
            p4 = {str(p): m for p, m in zip(outs, mom)}
            data = cal_angle_from_momentum(p4, self.decay_struct)
        except Exception as exc:
            # the merge is an optimisation: leave it off and say why
            warnings.warn(
                "dense_factor_merge disabled for this model: {}".format(exc)
            )
            structure["merge"] = False
            return
        F, B = amp.build_dense_terms(data)
        # the comparison can only be as fine as the stored precision: with
        # complex64 the rounding floor is ~1e-7, so the automatic tolerance
        # follows the dtype; an explicit merge_rtol is used as given
        if self.merge_rtol is None:
            eps = np.finfo(tf.as_dtype(F.dtype).as_numpy_dtype).eps
            rtol = max(1e-8, 8 * eps)  # 1e-8 / ~1e-6
        else:
            rtol = self.merge_rtol
        structure["merge"] = _find_merge(
            F,
            B,
            probe=self.merge_sample,
            rtol=rtol,
            atol=self.merge_atol,
        )

    def _after_blocks(self, x, blocks):
        structure = self.amp.structure
        if not isinstance(structure, dict):
            return
        merge = structure.get("merge")
        if merge is None:
            if not tf.executing_eagerly():
                raise RuntimeError(
                    "dense_factor_merge maps were not built before the graph "
                    "was traced; construct the preprocessor eagerly"
                )
            self._prepare_merge()
            merge = structure.get("merge")
        if not merge:
            return

        def reduce(d):
            if d.get("F") is not None:
                d["F"] = tf.gather(d["F"], merge["rep_u"], axis=2)
            if d.get("B") is not None:
                d["B"] = tf.gather(d["B"], merge["rep_k"], axis=1)

        reduce(x)
        for d in blocks:
            reduce(d)


@register_amp_model("dense_factor_merge")
class DenseFactorMergeModel(DenseFactorModel):
    """``dense_factor`` amplitude whose basis blocks are stored merged."""

    def expand_block(self, F, B):
        return _expand_block(self.structure, F, B)

    def _amplitude(self, F, B, C, g, dtype):
        F, B = self.expand_block(F, B)
        return super()._amplitude(F, B, C, g, dtype)


# ---------------------------------------------------------------------------
# The methods above attach themselves when the decorated classes are defined.
# ---------------------------------------------------------------------------
