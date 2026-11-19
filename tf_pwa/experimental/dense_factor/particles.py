"""Denominator terms per particle model.

Implements ``@terms`` from :mod:`.interface` for the particle
models the plugin supports: ``default``, ``BW``, ``one``, ``GS_rho``,
``Flatte``, ``FlatteGen``, ``Flatte2``, ``BWR_LS``, ``width_linear_*``,
``spline_c_idx``.  Importing this module installs them.

The one exception to "this module is the term side and ``decays.py`` the
basis side" is ``BWR_LS``: its LS numerator carries ``γ_i/q0^l_i`` on top of
the shared denominator, so the narrow ``ParticleDecayLS.get_amplitude_coeff``
hook below lives here, next to the terms whose assumptions it shares.
"""

import numpy as np
import tensorflow as tf

import tf_pwa.amp.interpolation  # noqa: F401  (register the spline models)
import tf_pwa.amp.split_ls  # noqa: F401  (register BWR_LS)
from tf_pwa.amp.core import get_particle_model
from tf_pwa.amp.flatte import cal_monentum
from tf_pwa.amp.split_ls import ParticleBWRLS, ParticleDecayLS
from tf_pwa.breit_wigner import (
    Bprime_num,
    Bprime_polynomial,
    dFun,
    dh_dsFun,
    hFun,
    twoBodyCMmom,
)
from tf_pwa.config import get_config

from .interface import terms

# ---------------------------------------------------------------------------
# Term basis (denominator) per particle model
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


@terms("FlatteGen")
class _FlatteGenTerms:
    """Terms for ``FlatteGen`` (and its ``Flatte2`` coupling variant).

    The denominator is

        m0² - m² + im_sign Σ_i g_i (m0/m) q_i (m0/|q_i0|)
                            (|q_i|/|q_i0|)^(2 l_i) B_l'²

    Split at the event boundary: the numerator of ``Bprime_q2`` is a
    constant of the event, so it belongs to the coefficient, and only
    ``1 / P_l((q d)²)`` stays in the basis.
    """

    def get_term_basis(self, data, data_c=None, **kwargs):
        m = data["m"]
        one = tf.ones_like(m)
        zero = tf.zeros_like(m)
        one_c = tf.complex(one, zero)
        sign = tf.complex(tf.cast(self.im_sign, m.dtype) * one, zero)
        i_c = tf.complex(zero, one)
        basis = [one_c, tf.complex(m * m, zero)]
        for (ma, mb), l in zip(self.mass_list, self.l_list):
            p = cal_monentum(m, ma, mb)
            absq = tf.abs(p)
            ratio = absq ** (2 * l)
            if self.has_bprime:
                ratio = ratio / Bprime_polynomial(l, (absq * self.d) ** 2)
            term = (
                sign * i_c * p / tf.cast(m, p.dtype) * tf.cast(ratio, p.dtype)
            )
            if self.cut_phsp:
                term = tf.where(m < ma + mb, tf.zeros_like(term), term)
            basis.append(term)
        return {"num": [one_c], "den": basis}

    def get_term_coeff(self):
        dtype = get_config("complex_dtype")
        rtype = get_config("dtype")
        m0 = tf.cast(self.get_mass(), rtype)
        coeff = [m0 * m0, tf.cast(-1.0, rtype)]
        g_values = self.get_coeff()
        for i, ((ma, mb), l) in enumerate(zip(self.mass_list, self.l_list)):
            if self.no_q0:
                q0 = tf.cast(1.0, rtype)
            else:
                p0 = cal_monentum(m0, ma, mb)
                q0 = tf.abs(p0)
            c = tf.cast(g_values[i], rtype)
            if not self.no_m0:
                c = c * m0
            if not self.no_q0:
                c = c * m0 / q0
            if l != 0:
                c = c / q0 ** (2 * l)
            if self.has_bprime:
                c = c * tf.cast(
                    Bprime_polynomial(l, (q0 * self.d) ** 2), rtype
                )
            coeff.append(c)
        return {"num": [tf.cast(1.0, dtype)], "den": coeff}


@terms("BWR_LS", requires=("get_ls_amp", "get_ls_amp_frac"))
class _BWRLSTerms:
    """Shared denominator of the LS-split ``BWR_LS`` amplitude.

    ``get_ls_amp_frac`` uses one denominator for every LS,

        dom = m0² - m² - i m0 g0 sqrt(q²/q0²) Σ_i γ_i² bf_i²,

    with ``bf_i = (q/q0)^l_i B_l'(q, q0, d)``.  With
    ``B_l'² = P_l((q0 d)²) / P_l((q d)²)`` the event part is the basis and
    the parameter part (``m0``, ``g0``, ``q0``, ``γ_i``) the coefficient.
    The extra ``m`` (or ``1/m`` for ``fix_bug1``) comes from the
    ``m/m0`` factor applied to the width in ``get_ls_amp_frac``.

    Below threshold the plugin reads the clamped ``|q|`` (finite, zero
    width) while the core's unclamped ``sqrt(q2/q02)`` is NaN; that state is
    unreachable for physical data, where an invariant mass is at least the
    sum of the daughter masses.
    """

    def _get_decay(self):
        decays = getattr(self, "decay", None)
        if not decays:
            raise NotImplementedError(
                "{} (BWR_LS) has no decay for get_term_basis".format(self)
            )
        return decays[0]

    def _check_supported(self):
        decay = self._get_decay()
        bad = []
        if (
            type(decay).get_barrier_factor2
            is not ParticleDecayLS.get_barrier_factor2
        ):
            bad.append("LS-decay")
        for name in ("has_barrier_factor", "has_bprime", "has_ql"):
            if not getattr(decay, name, False):
                bad.append(name + "=False")
        for name in (
            "force_min_l",
            "add_covariant_term",
            "barrier_factor_mass",
            "no_q0",
        ):
            if getattr(decay, name, False):
                bad.append(name + "=True")
        if getattr(decay, "ls_index", None) is not None:
            bad.append("ls_index")
        if bad:
            raise NotImplementedError(
                "{} (BWR_LS) cannot express {} as dense_factor "
                "terms".format(self, ", ".join(bad))
            )
        if ParticleDecayLS.get_amplitude_coeff is not _bwr_ls_amplitude_coeff:
            # a reload of tf_pwa.amp.split_ls uninstalls the numerator hook
            raise NotImplementedError(
                "{} (BWR_LS) lost its numerator hook; re-import "
                "tf_pwa.experimental.dense_factor".format(self)
            )
        return decay

    def _l_list(self):
        if getattr(self, "ls_list", None) is None:
            raise NotImplementedError(
                "{} (BWR_LS) needs ls_list for get_term_basis".format(self)
            )
        return [l for l, s in self.ls_list]

    def get_term_basis(self, data, data_c=None, **kwargs):
        self._check_supported()
        if data_c is None or "|q|" not in data_c:
            raise NotImplementedError(
                "{} (BWR_LS) needs |q| for get_term_basis".format(self)
            )
        m = data["m"]
        q = data_c["|q|"]
        one = tf.ones_like(m)
        zero = tf.zeros_like(m)
        basis = [tf.complex(one, zero), tf.complex(m * m, zero)]
        for l in self._l_list():
            val = q ** (2 * l + 1) / tf.cast(
                Bprime_polynomial(l, (q * self.d) ** 2), q.dtype
            )
            if self.fix_bug1:
                val = val / tf.cast(m, q.dtype)
            else:
                val = val * tf.cast(m, q.dtype)
            basis.append(tf.complex(tf.cast(zero, val.dtype), -val))
        return {"num": [basis[0]], "den": basis}

    def get_term_coeff(self):
        self._check_supported()
        g0 = self.get_width()
        if g0 is None:
            raise NotImplementedError(
                "{} (BWR_LS) has no width for get_term_coeff".format(self)
            )
        dtype = get_config("complex_dtype")
        rtype = get_config("dtype")
        m0 = tf.cast(self.get_mass(), rtype)
        g0 = tf.cast(g0, rtype)
        q0 = tf.cast(self.decay[0].get_relative_momentum({}, False), rtype)
        gamma = self.factor_gamma(self._l_list())
        coeff = [m0 * m0, tf.cast(-1.0, rtype)]
        for l, g in zip(self._l_list(), gamma):
            c = g0 * tf.cast(g, rtype) ** 2 / q0 ** (2 * l + 1)
            if self.fix_bug1:
                c = c * m0 * m0
            c = c * tf.cast(Bprime_polynomial(l, (q0 * self.d) ** 2), rtype)
            coeff.append(c)
        return {"num": [tf.cast(1.0, dtype)], "den": coeff}


def _bwr_ls_amplitude_coeff(self):
    """``ParticleDecayLS`` coefficient with the LS numerator factors.

    The dense basis of ``HelicityDecay`` already holds the event part of
    ``bf_i``.  ``ParticleDecayLS`` multiplies every LS amplitude by
    ``γ_i (q/q0)^l_i`` (``get_ls_amp_frac``), so the parameter part
    ``γ_i / q0^l_i`` belongs in this coefficient.  The shared denominator
    comes from ``_BWRLSTerms``; the ``BWR_LS2`` family returns one
    denominator per LS and is deliberately not handled here.
    """
    ret = super(ParticleDecayLS, self).get_amplitude_coeff()
    core = self.core
    if not isinstance(core, ParticleBWRLS):
        return ret
    q0 = tf.cast(self.get_relative_momentum({}, False), get_config("dtype"))
    gamma = core.factor_gamma([l for l, s in core.ls_list])
    coeff = ret["coeff"]
    if len(coeff) != len(gamma):
        raise NotImplementedError(
            "{} (BWR_LS) cannot align its LS coefficients with the "
            "dense basis".format(core)
        )
    out = []
    for c, l, g in zip(coeff, [l for l, s in core.ls_list], gamma):
        f = tf.cast(g, c.dtype)
        if not self.barrier_factor_norm:
            f = f / tf.cast(q0, c.dtype) ** l
        out.append(c * f)
    return {**ret, "coeff": out}


_bwr_ls_amplitude_coeff.__name__ = "get_amplitude_coeff"
_bwr_ls_amplitude_coeff.__doc__ = (
    "``ParticleDecayLS`` coefficient with the BWR_LS numerator factors."
)
ParticleDecayLS.get_amplitude_coeff = _bwr_ls_amplitude_coeff


@terms("width_linear_txt", requires=("get_amp", "interp", "convert_to_amp"))
@terms("width_linear_npy", requires=("get_amp", "interp", "convert_to_amp"))
class _WidthLinearTerms:
    """Denominator terms for the interpolated-width linear models.

    ``convert_to_amp`` builds ``1 / (m0² - m² - m0 g0 (fm - Re fm(m0)))``
    with ``fm`` the interpolated ``Π(m)`` (and ``g0 / Im fm(m0)`` when
    ``width_scale`` is set).  ``fm`` is event-only and ``fm(m0)`` is a
    parameter, so the terms are

        basis  ``[1, m², -fm]``
        coeff  ``[m0² + m0 g0 Re fm(m0), -1, m0 g0]``
    """

    def get_term_basis(self, data, data_c=None, **kwargs):
        m = data["m"]
        one = tf.ones_like(m)
        zero = tf.zeros_like(m)
        one_c = tf.complex(one, zero)
        fm = self.interp(m)
        return {
            "num": [one_c],
            "den": [one_c, tf.complex(m * m, zero), -fm],
        }

    def get_term_coeff(self):
        dtype = get_config("complex_dtype")
        rtype = get_config("dtype")
        g0 = self.get_width()
        if g0 is None:
            raise NotImplementedError(
                "{} ({}) has no width for get_term_coeff".format(
                    self, type(self).__name__
                )
            )
        m0 = tf.cast(self.get_mass(), rtype)
        g0 = tf.cast(g0, rtype)
        fm0 = self.interp(tf.stack([m0]))[0]
        refm0 = tf.math.real(fm0)
        if getattr(self, "width_scale", False):
            g0 = g0 / tf.math.imag(fm0)
        coeff = [
            m0 * m0 + m0 * g0 * tf.cast(refm0, rtype),
            tf.cast(-1.0, rtype),
            m0 * g0,
        ]
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
