"""
Basic amplitude model
"""


import numpy as np

from tf_pwa.breit_wigner import BW, BWR, BWR2, GS, Bprime, BWR_normal, Gamma
from tf_pwa.breit_wigner import barrier_factor2 as barrier_factor
from tf_pwa.dec_parser import load_dec_file
from tf_pwa.dfun import get_D_matrix_lambda
from tf_pwa.particle import _spin_int, _spin_range
from tf_pwa.tensorflow_wrapper import tf

from .core import (
    AmpBase,
    AmpDecay,
    HelicityDecay,
    Particle,
    get_relative_p,
    regist_decay,
    regist_particle,
)


@regist_particle("BWR2")
class ParticleBWR2(Particle):
    """
    .. math::
        R(m) = \\frac{1}{m_0^2 - m^2 - i m_0 \\Gamma(m)}

    """

    def get_amp(self, data, data_c, **kwargs):
        mass = self.get_mass()
        width = self.get_width()
        if width is None:
            return tf.ones_like(data["m"])
        if not self.running_width:
            ret = BW(data["m"], mass, width)
        else:
            q2 = data_c["|q|2"]
            q02 = data_c["|q0|2"]
            if self.bw_l is None:
                decay = self.decay[0]
                self.bw_l = min(decay.get_l_list())
            ret = BWR2(data["m"], mass, width, q2, q02, self.bw_l, self.d)
        return ret


@regist_particle("BWR_normal")
class ParticleBWR_normal(Particle):
    """
    .. math::
        R(m) = \\frac{\\sqrt{m_0 \\Gamma(m)}}{m_0^2 - m^2 - i m_0 \\Gamma(m)}

    """

    def get_amp(self, data, data_c, **kwargs):
        mass = self.get_mass()
        width = self.get_width()
        if width is None:
            return tf.ones_like(data["m"])
        if not self.running_width:
            ret = BW(data["m"], mass, width)
        else:
            q2 = data_c["|q|2"]
            q02 = data_c["|q0|2"]
            if self.bw_l is None:
                decay = self.decay[0]
                self.bw_l = min(decay.get_l_list())
            ret = BWR_normal(
                data["m"], mass, width, q2, q02, self.bw_l, self.d
            )
        return ret


# added by xiexh for GS model rho
@regist_particle("GS_rho")
class ParticleGS(Particle):
    r"""
    Gounaris G.J., Sakurai J.J., Phys. Rev. Lett., 21 (1968), pp. 244-247

    `c_daug2Mass`: mass for daughter particle 2 (:math:`\pi^{+}`) 0.13957039

    `c_daug3Mass`: mass for daughter particle 3 (:math:`\pi^{0}`) 0.1349768

    .. math::
      R(m) = \frac{1 + D \Gamma_0 / m_0}{(m_0^2 -m^2) + f(m) - i m_0 \Gamma(m)}

    .. math::
      f(m) = \Gamma_0 \frac{m_0 ^2 }{q_0^3} \left[q^2 [h(m)-h(m_0)] + (m_0^2 - m^2) q_0^2 \frac{d h}{d m}|_{m0} \right]

    .. math::
      h(m) = \frac{2}{\pi} \frac{q}{m} \ln \left(\frac{m+q}{2m_{\pi}} \right)

    .. math::
      \frac{d h}{d m}|_{m0} = h(m_0) [(8q_0^2)^{-1} - (2m_0^2)^{-1}] + (2\pi m_0^2)^{-1}

    .. math::
      D = \frac{f(0)}{\Gamma_0 m_0} = \frac{3}{\pi}\frac{m_\pi^2}{q_0^2} \ln \left(\frac{m_0 + 2q_0}{2 m_\pi }\right)
        + \frac{m_0}{2\pi q_0} - \frac{m_\pi^2 m_0}{\pi q_0^3}
    """

    def __init__(self, *args, **kwargs):
        self.c_daug2Mass = 0.13957039
        self.c_daug3Mass = 0.1349768
        super().__init__(*args, **kwargs)

    def get_amp(self, data, data_c, **kwargs):
        mass = self.get_mass()
        width = self.get_width()
        if width is None:
            return tf.ones_like(data["m"])
        if not self.running_width:
            ret = BW(data["m"], mass, width)
        else:
            q = data_c["|q|"]
            q0 = data_c["|q0|"]
            if self.bw_l is None:
                decay = self.decay[0]
                self.bw_l = min(decay.get_l_list())
            ret = GS(
                data["m"],
                mass,
                width,
                q,
                q0,
                self.bw_l,
                self.d,
                self.c_daug2Mass,
                self.c_daug3Mass,
            )
        return ret


# added by xiexh end


@regist_particle("BW")
class ParticleBW(Particle):
    """
    .. math::
        R(m) = \\frac{1}{m_0^2 - m^2 - i m_0 \\Gamma_0}

    """

    def get_amp(self, data, _data_c=None, **kwargs):
        mass = self.get_mass()
        width = self.get_width()
        ret = BW(data["m"], mass, width)
        return ret


@regist_particle("Kmatrix")
class ParticleKmatrix(Particle):
    def init_params(self):
        self.d = 3.0
        self.mass1 = self.add_var("mass1", fix=True)
        self.mass2 = self.add_var("mass2", fix=True)
        self.width1 = self.add_var("width1", fix=True)
        self.width2 = self.add_var("width2", fix=True)
        self.KNR = self.add_var("KNR", is_complex=True)
        self.alpha = self.add_var("alpha")
        self.beta0 = self.add_var("beta0", is_complex=True)
        self.beta1 = self.add_var("beta1", is_complex=True, fix=True)
        self.beta2 = self.add_var("beta2", is_complex=True)
        if self.bw_l is None:
            decay = self.decay[0]
            self.bw_l = min(decay.get_l_list())

    def get_amp(self, data, data_c=None, **kwargs):
        m = data["m"]
        mass1 = self.mass1()
        mass2 = self.mass2()
        width1 = self.width1()
        width2 = self.width2()
        q = data_c["|q|"]
        mdaughter1 = kwargs["all_data"]["particle"][self.decay[0].outs[0]]["m"]
        mdaughter2 = kwargs["all_data"]["particle"][self.decay[0].outs[1]]["m"]
        q1 = get_relative_p(mass1, mdaughter1, mdaughter2)
        q2 = get_relative_p(mass2, mdaughter1, mdaughter2)
        mlist = tf.stack([mass1, mass2])
        wlist = tf.stack([width1, width2])
        qlist = tf.stack([q1, q2])
        Klist = []
        for mi, wi, qi in zip(mlist, wlist, qlist):
            rw = Gamma(m, wi, q, qi, self.bw_l, mi, self.d)
            Klist.append(mi * rw / (mi ** 2 - m ** 2))
        KK = tf.reduce_sum(Klist, axis=0)
        KK += self.alpha()
        beta_term = self.get_beta(
            m=m,
            mlist=mlist,
            wlist=wlist,
            q=q,
            qlist=qlist,
            Klist=Klist,
            **kwargs,
        )
        MM = tf.complex(np.float64(1), -KK)
        MM = beta_term / MM
        return MM + self.KNR()

    def get_beta(self, m, **kwargs):
        m1, m2 = kwargs["mlist"]
        w1, w2 = kwargs["wlist"]
        q1, q2 = kwargs["qlist"]
        q = kwargs["q"]
        z = (q * self.d) ** 2
        z1 = (q1 * self.d) ** 2
        z2 = (q2 * self.d) ** 2
        Klist = kwargs["Klist"]
        beta1 = self.beta1()
        beta1 = beta1 * tf.cast(Klist[0] * m / m1 * q1 / q, beta1.dtype)
        beta1 = beta1 / tf.cast(
            (z / z1) ** self.bw_l * Bprime(self.bw_l, q, q1, self.d) ** 2,
            beta1.dtype,
        )
        beta2 = self.beta2()
        beta2 = beta2 * tf.cast(Klist[1] * m / m2 * q2 / q, beta2.dtype)
        beta2 = beta2 / tf.cast(
            (z / z2) ** self.bw_l * Bprime(self.bw_l, q, q2, self.d) ** 2,
            beta2.dtype,
        )
        beta0 = self.beta0()  # * tf.cast(2 * z / (z + 1), beta1.dtype)
        return beta0 + beta1 + beta2


@regist_particle("LASS")
class ParticleLass(Particle):
    def init_params(self):
        super(ParticleLass, self).init_params()
        self.a = self.add_var("a")
        self.r = self.add_var("r")

    def get_amp(self, data, data_c=None, **kwargs):
        r"""
        .. math::
          R(m) = \frac{m}{q cot \delta_B - i q}
            + e^{2i \delta_B}\frac{m_0 \Gamma_0 \frac{m_0}{q_0}}
                                  {(m_0^2 - m^2) - i m_0\Gamma_0 \frac{q}{m}\frac{m_0}{q_0}}

        .. math::
          cot \delta_B = \frac{1}{a q} + \frac{1}{2} r q

        .. math::
          e^{2i\delta_B} = \cos 2 \delta_B + i \sin 2\delta_B
                         = \frac{cot^2\delta_B -1 }{cot^2 \delta_B +1} + i \frac{2 cot \delta_B }{cot^2 \delta_B +1 }

        """
        m = data["m"]
        q = data_c["|q|"]
        q0 = data_c["|q0|"]
        mass = self.get_mass()
        width = self.get_width()
        a, r = tf.abs(self.a()), tf.abs(self.r())
        cot_delta_B = (1.0 / a) / q + 0.5 * r * q
        cot2_delta_B = cot_delta_B * cot_delta_B
        expi_2delta_B = tf.complex(cot2_delta_B - 1, 2 * cot_delta_B)
        expi_2delta_B /= tf.cast(cot2_delta_B + 1, expi_2delta_B.dtype)
        ret = 1.0 / tf.complex(q * cot_delta_B, -q)
        ret = tf.cast(m, ret.dtype) * ret
        ret += (
            expi_2delta_B
            * BWR(m, mass, width, q, q0, 0, 1.0)
            * tf.cast(mass * width * mass / q0, ret.dtype)
        )
        return ret


@regist_particle("one")
class ParticleOne(Particle):
    """
    .. math::
        R(m) = 1

    """

    def init_params(self):
        pass

    def get_amp(self, data, _data_c=None, **kwargs):
        mass = data["m"]
        zeros = tf.zeros_like(mass)
        ones = tf.ones_like(mass)
        return tf.complex(ones, zeros)


@regist_particle("exp")
class ParticleExp(Particle):
    """
    .. math::
        R(m) = e^{-|a| m}

    """

    def init_params(self):
        self.a = self.add_var("a")

    def get_amp(self, data, _data_c=None, **kwargs):
        mass = data["m"]
        zeros = tf.zeros_like(mass)
        a = tf.abs(self.a())
        return tf.complex(tf.exp(-a * mass), zeros)


@regist_particle("exp_com")
class ParticleExp(Particle):
    """
    .. math::
        R(m) = e^{-|a+ib| m^2}

    """

    def init_params(self):
        self.a = self.add_var("a")
        self.b = self.add_var("b")

    def get_amp(self, data, _data_c=None, **kwargs):
        mass = data["m"]
        zeros = tf.zeros_like(mass)
        a = tf.abs(self.a())
        b = self.b()
        r = -tf.complex(a, b) * tf.complex(mass * mass, zeros)
        return tf.exp(r)


@regist_decay("particle-decay")
class ParticleDecay(HelicityDecay):
    def get_ls_amp(self, data, data_p, **kwargs):
        amp = super(ParticleDecay, self).get_ls_amp(data, data_p, **kwargs)
        a = self.core
        b = self.outs[0]
        c = self.outs[1]
        mass = a.get_mass()
        width = a.get_width()
        m = data_p[a]["m"]
        if width is None:
            ret = tf.zeros_like(m)
            ret = tf.complex(ret, ret)
        elif not a.running_width:
            ret = tf.reshape(BW(m, mass, width), (-1, 1))
        else:
            q = data["|q|"]
            q0 = data["|q0|"]
            ret = []
            for i in self.get_l_list():
                bw = BWR(m, mass, width, q, q0, i, self.d)
                ret.append(tf.reshape(bw, (-1, 1)))
            ret = tf.concat(ret, axis=-1)
        return ret * amp


@regist_decay("helicity_full")
class HelicityDecayNP(HelicityDecay):
    def init_params(self):
        a = self.outs[0].spins
        b = self.outs[1].spins
        self.H = self.add_var("H", is_complex=True, shape=(len(a), len(b)))

    def get_helicity_amp(self, data, data_p, **kwargs):
        return tf.stack(self.H())


@regist_decay("helicity_full-bf")
class HelicityDecayNPbf(HelicityDecay):
    def init_params(self):
        self.d = 3.0
        a = self.outs[0].spins
        b = self.outs[1].spins
        self.H = self.add_var("H", is_complex=True, shape=(len(a), len(b)))

    def get_helicity_amp(self, data, data_p, **kwargs):
        q0 = self.get_relative_momentum(data_p, False)
        data["|q0|"] = q0
        if "|q|" in data:
            q = data["|q|"]
        else:
            q = self.get_relative_momentum(data_p, True)
            data["|q|"] = q
        bf = barrier_factor([min(self.get_l_list())], q, q0, self.d)
        H = tf.stack(self.H())
        bf = tf.cast(tf.reshape(bf, (-1, 1, 1)), H.dtype)
        return H * bf


def get_parity_term(j1, p1, j2, p2, j3, p3):
    p = p1 * p2 * p3 * (-1) ** (j1 - j2 - j3)
    return p


@regist_decay("helicity_parity")
class HelicityDecayP(HelicityDecay):
    def init_params(self):
        a = self.core
        b = self.outs[0]
        c = self.outs[1]
        n_b = len(b.spins)
        n_c = len(c.spins)
        self.parity_term = get_parity_term(a.J, a.P, b.J, b.P, c.J, c.P)
        if n_b > n_c:
            self.H = self.add_var(
                "H", is_complex=True, shape=((n_b + 1) // 2, n_c)
            )
            self.part_H = 0
        else:
            self.H = self.add_var(
                "H", is_complex=True, shape=(n_b, (n_c + 1) // 2)
            )
            self.part_H = 1

    def get_helicity_amp(self, data, data_p, **kwargs):
        n_b = len(self.outs[0].spins)
        n_c = len(self.outs[1].spins)
        H_part = tf.stack(self.H())
        if self.part_H == 0:
            H = tf.concat(
                [H_part, self.parity_term * H_part[(n_b - 2) // 2 :: -1]],
                axis=0,
            )
        else:
            H = tf.concat(
                [H_part, self.parity_term * H_part[:, (n_c - 2) // 2 :: -1]],
                axis=1,
            )
        return H


@regist_decay("gls-cpv")
class HelicityDecay(HelicityDecay):
    def init_params(self):
        self.d = 3.0
        ls = self.get_ls_list()
        self.g_ls = self.add_var(
            "g_ls", is_complex=True, shape=(len(ls),), is_cp=True
        )
        try:
            self.g_ls.set_fix_idx(fix_idx=0, fix_vals=(1.0, 0.0))
        except Exception as e:
            print(e, self, self.get_ls_list())

    def get_g_ls(self, charge=1):
        gls = self.g_ls(charge)
        if self.ls_index is None:
            return tf.stack(gls)
        # print(self, gls, self.ls_index)
        return tf.stack([gls[k] for k in self.ls_index])

    def get_ls_amp(self, data, data_p, **kwargs):
        charge = kwargs.get("all_data", {}).get("charge_conjugation", None)
        g_ls_p = self.get_g_ls(1)
        if charge is None:
            g_ls = g_ls_p
        else:
            g_ls_m = self.get_g_ls(-1)
            g_ls = tf.where((charge > 0)[:, None], g_ls_p, g_ls_m)
        # print(g_ls)
        q0 = self.get_relative_momentum2(data_p, False)
        data["|q0|2"] = q0
        if "|q|2" in data:
            q = data["|q|2"]
        else:
            q = self.get_relative_momentum2(data_p, True)
            data["|q|2"] = q
        if self.has_barrier_factor:
            bf = self.get_barrier_factor2(
                data_p[self.core]["m"], q, q0, self.d
            )
            mag = g_ls
            m_dep = mag * tf.cast(bf, mag.dtype)
        else:
            m_dep = g_ls
        return m_dep
