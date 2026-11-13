"""Denominator terms per particle model.

Implements ``@terms`` from :mod:`.interface` for the particle
models the plugin supports: ``default``, ``BW``, ``one``, ``GS_rho``,
``Flatte`` and ``spline_c_idx``.  Importing this module installs them.
"""

import functools

import numpy as np
import tensorflow as tf

import tf_pwa.amp.interpolation  # noqa: F401  (register the spline models)
from tf_pwa.amp.core import get_particle_model
from tf_pwa.amp.flatte import cal_monentum
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
