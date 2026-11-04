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

import functools
import itertools

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
from tf_pwa.config import get_config

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
            if attr.startswith("__") or not callable(value):
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
        return q ** (2 * L + 1) / (
            m * Bprime_polynomial(L, (q * d) ** 2)
        )

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
        basis = tf.stack(
            [
                tf.complex(one, zero),
                tf.complex(m * m, zero),
                tf.complex(zero, -f_shape),
            ],
            axis=-1,
        )
        return {"num": [tf.cast(one, basis.dtype)], "den": [basis]}

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
        c_m0 = tf.cast(m0, dtype)
        c_g0 = tf.cast(g0, dtype)
        if self.running_width:
            L = self._get_bw_l()
            q0 = self._get_q0()
            f0 = self._F_shape(q0, tf.cast(m0, q0.dtype), L)
            f0 = tf.cast(f0, dtype)
        else:
            f0 = tf.cast(1.0, dtype)
        if self.width_norm:
            c0 = c_m0 * c_m0 / c_g0
            c1 = -1.0 / c_g0
            c2 = c_m0 / f0
        else:
            c0 = c_m0 * c_m0
            c1 = tf.cast(-1.0, dtype)
            c2 = c_m0 * c_g0 / f0
        return {
            "num": [tf.cast(1.0, dtype)],
            "den": [[c0, c1, c2]],
        }


@terms("BW")
class _BWTerms:
    def get_term_basis(self, data, data_c=None, **kwargs):
        m = data["m"]
        one = tf.ones_like(m)
        zero = tf.zeros_like(m)
        basis = tf.stack(
            [
                tf.complex(one, zero),
                tf.complex(m * m, zero),
                tf.complex(zero, -one),
            ],
            axis=-1,
        )
        return {"num": [tf.cast(one, basis.dtype)], "den": [basis]}

    def get_term_coeff(self):
        g0 = self.get_width()
        if g0 is None:
            raise NotImplementedError(
                "{} ({}) has no width for get_term_coeff".format(
                    self, type(self).__name__
                )
            )
        m0 = self.get_mass()
        dtype = get_config("complex_dtype")
        c_m0 = tf.cast(m0, dtype)
        c_g0 = tf.cast(g0, dtype)
        # ``get_amp`` always uses ``BW`` and ignores ``width_norm``,
        # so the decomposition must ignore it as well.
        c0 = c_m0 * c_m0
        c1 = tf.cast(-1.0, dtype)
        c2 = c_m0 * c_g0
        return {
            "num": [tf.cast(1.0, dtype)],
            "den": [[c0, c1, c2]],
        }

    def get_sympy_dom(self, m, m0, g0, m1=None, m2=None, sheet=0):
        from tf_pwa.formula import BW_dom

        return BW_dom(m, m0, g0)


@terms("GS_rho")
class _GSTerms:
    def get_term_basis(self, data, data_c=None, **kwargs):
        m = data["m"]
        g0 = self.get_width()
        if g0 is None:
            return _DefaultTerms.get_term_basis(
                self, data, data_c, **kwargs
            )
        one = tf.ones_like(m)
        zero = tf.zeros_like(m)
        if not self.running_width:
            # ``get_amp`` falls back to the constant-width ``BW``.
            basis = tf.stack(
                [
                    tf.complex(one, zero),
                    tf.complex(m * m, zero),
                    tf.complex(zero, -one),
                ],
                axis=-1,
            )
            return {"num": [tf.cast(one, basis.dtype)], "den": [basis]}
        if data_c is None:
            return _DefaultTerms.get_term_basis(
                self, data, data_c, **kwargs
            )
        L = self.bw_l
        if L is None:
            L = min(self.decay[0].get_l_list())
        c2 = self.c_daug2Mass
        c3 = self.c_daug3Mass
        k2 = twoBodyCMmom(m, c2, c3) ** 2
        h_m = hFun(m * m, c2, c3)
        f_m = self._F_shape(data_c["|q|"], m, L)
        basis = tf.stack(
            [
                tf.complex(one, zero),
                tf.complex(m * m, zero),
                tf.complex(k2 * h_m, zero),
                tf.complex(k2, zero),
                tf.complex(zero, -f_m),
            ],
            axis=-1,
        )
        return {"num": [tf.cast(one, basis.dtype)], "den": [basis]}

    def get_term_coeff(self):
        g0 = self.get_width()
        if g0 is None:
            return _DefaultTerms.get_term_coeff(self)
        m0 = self.get_mass()
        dtype = get_config("complex_dtype")
        if not self.running_width:
            # ``get_amp`` falls back to the constant-width ``BW``.
            return {
                "num": [tf.cast(1.0, dtype)],
                "den": [
                    [
                        tf.cast(m0 * m0, dtype),
                        tf.cast(-1.0, dtype),
                        tf.cast(m0 * g0, dtype),
                    ]
                ],
            }
        c_m0 = tf.cast(m0, dtype)
        c_g0 = tf.cast(g0, dtype)
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
        f0 = tf.cast(f0, dtype)
        a = c_g0 * c_m0 * c_m0 / tf.cast(k0**3, dtype)
        b = -a * tf.cast(h0, dtype)
        c = c_g0 * c_m0**4 * tf.cast(dh0, dtype) / tf.cast(k0, dtype)
        d = -c_g0 * c_m0 * c_m0 * tf.cast(dh0, dtype) / tf.cast(k0, dtype)
        num = 1.0 + tf.cast(
            dFun(
                tf.cast(m0 * m0, tf.float64),
                c2,
                c3,
            ),
            dtype,
        ) * c_g0 / c_m0
        coeff = [
            c_m0 * c_m0 + c,
            tf.cast(-1.0, dtype) + d,
            a,
            b,
            c_m0 * c_g0 / f0,
        ]
        return {"num": [num], "den": [coeff]}


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
            "num": [tf.cast(one, basis[0].dtype)],
            "den": [tf.stack(basis, axis=-1)],
        }

    def get_term_coeff(self):
        dtype = get_config("complex_dtype")
        m0 = tf.cast(self.get_mass(), dtype)
        coeff = [m0 * m0, tf.cast(-1.0, dtype)]
        for i in range(len(self.mass_list)):
            coeff.append(m0 * tf.cast(self.g_value[i](), dtype))
        return {"num": [tf.cast(1.0, dtype)], "den": [coeff]}


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
        one = tf.ones((tf.shape(m)[0], 1), dtype=phi.dtype)
        return {
            "num": [phi[..., i] for i in range(n_term)],
            "den": [one for _ in range(n_term)],
        }

    def get_term_coeff(self):
        p = self.point_value()
        n_term = p.shape[0]
        one = tf.ones((), dtype=p.dtype)
        return {
            "num": [p[i] for i in range(n_term)],
            "den": [[one] for _ in range(n_term)],
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

    def get_barrier_coeff(self, q0, d=None):
        """Parameter-only coefficient F_l(q_0) (and q_0^{-l} if norm)."""
        if d is None:
            d = self.d
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
                    c = c / q0**l
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
            bf = self.get_barrier_basis(
                q, data_p[self.core]["m"], self.d
            )
            bf = tf.cast(bf, basis.dtype)
            bf = tf.reshape(
                bf,
                [bf.shape[0], bf.shape[1]]
                + [1] * (len(basis.shape) - 2),
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
        q0 = self.get_relative_momentum({}, False)
        c = self.get_barrier_coeff(q0, self.d)
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
                i.get_amplitude_basis(
                    data_c[i], data_p, all_data=all_data
                )["basis"]
            )
            next_map = next_map[1:]
        final_indices = "".join(
            iter_idx + list(used_idx) + final_indices
        )

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
                decay_chain.get_particle_terms(
                    data_p, data_c, all_data=data
                )
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
        return [
            self.chains[i].get_amplitude_coeff() for i in self.chains_idx
        ]

    def build_dense_coeff(self, structure):
        """Build numerator and denominator coefficients for a term cache."""
        C = []
        used_chains = set(self.chains_idx)

        chain_c = []
        ls_vecs = []
        num_vecs = []
        den_vecs = []
        for meta in structure["chain_meta"]:
            chain_c.append(
                tf.reshape(meta["chain"].get_amp_total(), (-1,))[0]
            )
            ls_vecs.append(
                [
                    d.get_amplitude_coeff()["coeff"]
                    for d in meta["decays"]
                ]
            )
            num_vecs.append(
                [p.get_term_coeff()["num"] for p in meta["lines"]]
            )
            den_vecs.append(
                [p.get_term_coeff()["den"][0] for p in meta["lines"]]
            )

        for u, k in enumerate(structure["u_chain"]):
            if k not in used_chains:
                C.append(
                    tf.zeros((), dtype=get_config("complex_dtype"))
                )
                continue
            c = chain_c[k]
            for d, ls_idx in enumerate(structure["u_ls"][u]):
                c = c * ls_vecs[k][d][ls_idx]
            for l, term_idx in enumerate(structure["u_term"][u]):
                c = c * num_vecs[k][l][term_idx]
            C.append(tf.cast(c, get_config("complex_dtype")))

        g_blocks = [g for row in den_vecs for block in row for g in block]
        if C:
            C = tf.stack(C)
        else:
            C = tf.zeros((0,), dtype=get_config("complex_dtype"))
        if g_blocks:
            g = tf.cast(tf.stack(g_blocks), get_config("complex_dtype"))
        else:
            g = tf.zeros((0,), dtype=get_config("complex_dtype"))
        return C, g


# ---------------------------------------------------------------------------
# Amplitude model
# ---------------------------------------------------------------------------


@register_preprocessor("dense_factor")
class DenseFactorPreProcessor(CachedAnglePreProcessor):
    def call(self, x, **kwargs):
        x = BasePreProcessor.call(self, x, **kwargs)
        structure = getattr(self.amp, "structure", None)
        if structure is None:
            # Trigger the particle-specific NotImplementedError.
            self.decay_group.get_particle_terms(x)
            return x
        F, B, spin_shape = self.amp.build_dense_terms(x)
        x["F"] = F
        x["B"] = B
        x["F_spin_shape"] = spin_shape
        return x


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
                n_term = [len(p.get_term_coeff()["num"]) for p in lines]
                den_list = [p.get_term_coeff()["den"] for p in lines]
            except NotImplementedError:
                return
            den_sizes = []
            for d in den_list:
                if len(d) == 1:
                    den_sizes.append(len(d[0]))
                elif all(len(i) == 1 for i in d):
                    # multiple numerator terms, denominator is 1
                    den_sizes.append(1)
                else:
                    return

            n_internal_lines = len(lines)
            max_internal_lines = max(
                max_internal_lines, n_internal_lines
            )

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

        n_param = sum(
            sum(meta["den_sizes"]) for meta in chain_meta
        )
        n_u = len(u_chain)

        param_offset = []
        off = 0
        for meta in chain_meta:
            offsets = []
            for size in meta["den_sizes"]:
                offsets.append(off)
                off += size
            meta["param_offset"] = offsets

        M1 = np.zeros(
            (n_u, max_internal_lines, n_param), dtype=np.float64
        )
        line_mask = np.zeros(
            (n_u, max_internal_lines), dtype=bool
        )

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
        self.M1T = tf.reshape(
            tf.transpose(self.M1, [2, 0, 1]),
            [n_param, n_u * max_internal_lines],
        )
        self.M2 = tf.eye(n_param, dtype=tf.float64)
        self.line_mask = tf.constant(line_mask)
        self.line_mask_all = bool(np.all(line_mask))

    def build_dense_terms(self, data):
        structure = self.structure
        dg = self.decay_group
        if data.get("id_swap") or getattr(dg, "identical_particles", None):
            raise NotImplementedError(
                "dense_factor does not support identical-particle "
                "symmetrization yet"
            )
        if data.get("cp_swap") is not None or getattr(
            dg, "cp_particles", None
        ):
            raise NotImplementedError(
                "dense_factor does not support CP conjugation yet"
            )

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
            spin_shape = list(F_chain.shape[1 + n_decay :])
            F_ls = tf.reshape(
                F_chain, [n_data, n_ls_combo] + spin_shape
            )

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
                B_blocks.append(
                    tf.cast(
                        particle_terms[idx][l][1]["den"][0],
                        F_chain.dtype,
                    )
                )

        if F_list:
            F = tf.stack(F_list, axis=1)
            spin_shape = tuple(F.shape[2:])
            F = tf.reshape(F, [n_data, len(F_list), -1])
            F = tf.transpose(F, [0, 2, 1])
        else:
            spin_shape = ()
            F = tf.zeros((n_data, 0, 0), dtype=F_chains[0].dtype)
        if B_blocks:
            B = tf.concat(B_blocks, axis=-1)
            B = tf.matmul(
                B, tf.cast(self.M2, B.dtype), transpose_b=True
            )
        else:
            B = tf.zeros((n_data, 0), dtype=F_chains[0].dtype)
        return F, B, spin_shape

    def pdf(self, data):
        dg = self.decay_group
        structure = self.structure
        if structure is None:
            dg.get_particle_terms(data)
            raise NotImplementedError("dense_factor is not initialized")

        F = data.get("F", None)
        B = data.get("B", None)
        spin_shape = data.get("F_spin_shape", None)
        if F is None or B is None:
            F, B, spin_shape = self.build_dense_terms(data)

        C, g = dg.build_dense_coeff(structure)
        dtype = g.dtype
        B = tf.cast(B, dtype)
        M1T = tf.cast(self.M1T, dtype)
        n_u = structure["n_u"]
        n_line = structure["max_internal_lines"]
        if n_line == 0:
            D = tf.ones((tf.shape(F)[0], n_u), dtype=g.dtype)
        else:
            M1gT = M1T * g[:, None]
            D = tf.matmul(B, M1gT)
            if n_line == 1 and self.line_mask_all:
                D = tf.reshape(D, [tf.shape(F)[0], n_u])
            else:
                D = tf.reshape(D, [tf.shape(F)[0], n_u, n_line])
                D = tf.where(
                    self.line_mask[None, :, :], D, tf.ones_like(D)
                )
                D = tf.reduce_prod(D, axis=-1)

        e = tf.shape(F)[0]
        W = C[None, :] / D
        amp = tf.linalg.matvec(F, W)
        amp = tf.reshape(amp, [e] + list(spin_shape))
        return dg.sum_with_polarization(amp)


# ---------------------------------------------------------------------------
# The methods above attach themselves when the decorated classes are defined.
# ---------------------------------------------------------------------------
