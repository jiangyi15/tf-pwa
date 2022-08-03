import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tf_pwa.amp import HelicityDecay, Particle, register_particle
from tf_pwa.amp.Kmatrix import KmatrixSplitLSParticle
from tf_pwa.breit_wigner import Bprime_polynomial


def get_relative_p(m, m1, m2):
    p2 = (m * m - (m1 + m2) ** 2) * (m * m - (m1 - m2) ** 2) / 4 / m / m
    return tf.where(m > m1 + m2, tf.sqrt(tf.abs(p2)), tf.zeros_like(p2))


def barrier_factor(m, m1, m2, l, d=3.0):
    if l == 0:
        return tf.ones_like(m)
    q = get_relative_p(m, m1, m2)
    z = (q * d) ** 2
    return tf.sqrt(Bprime_polynomial(l, tf.ones_like(z)) * z**l) / tf.sqrt(
        Bprime_polynomial(l, z)
    )


@register_particle("KmatrixSimple")
class KmatrixSimple(KmatrixSplitLSParticle):
    """

        simple Kmatrix formula.

    K-matrix

    .. math::

        K_{i,j} = \\sum_{a} \\frac{g_{i,a} g_{j,a}}{m_a^2 - m^2+i\\epsilon}

    P-vector

    .. math::

        P_{i} = \\sum_{a} \\frac{\\beta_{a} g_{i,a}}{m_a^2 - m^2 +i\\epsilon} + f_{bkg,i}

    total amplitude
    .. math::

        R(m) = n (1 - K i \\rho n^2)^{-1} P

    barrief factor
    .. math::

        n_{ii} = q_i^l B'_l(q_i, 1/d, d)

    phase space factor

    .. math::

        \\rho_{ii} = q_i/m

    :math:`q_i` is 0 when below threshold

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mass_list = kwargs["mass_list"]
        self.mass_list = mass_list

        self.decay_list = kwargs.get("decay_list", None)
        self.l_list = kwargs.get("l_list", None)
        self.index_list = kwargs.get("index_list", None)
        self.extra_decay_list = kwargs.get("extra_decay_list", [])
        self.extra_l_list = kwargs.get("extra_l_list", [])

        self.n_pole = len(mass_list)

    def init_params(self):
        if self.decay_list is None:
            ls_list = self.decay[0].get_ls_list()
            self.decay_list = [
                [
                    self.decay[0].outs[0].get_mass(),
                    self.decay[0].outs[1].get_mass(),
                ]
            ] * len(
                ls_list
            )  # kwargs["decay_list"]
            self.decay_list = self.decay_list + self.extra_decay_list
            if self.l_list is None:
                self.l_list = [i[0] for i in ls_list]  # kwargs["l_list"]
                self.l_list = self.l_list + self.extra_l_list
            if self.index_list is None:
                self.index_list = list(range(len(ls_list)))
        assert len(self.decay_list) == len(self.l_list)
        self.n_channel = len(self.decay_list)
        # print(decay_list, self.n_channel)
        self._epsilon = 1e-10
        self.coeffs = self.add_var(
            "gij", shape=(self.n_channel, self.n_pole)
        )  # np.random.random((self.n_channel, self.n_pole)) *2 # * 10
        self.beta = self.add_var(
            "beta", is_complex=True, shape=(self.n_pole)
        )  # np.random.random(self.n_pole) + np.random.random(self.n_pole) * 1.0j
        self.bkg = self.add_var(
            "bkg", is_complex=True, shape=(self.n_channel)
        )  # np.random.random(self.n_channel)
        # print(self.bkg())
        self.beta.set_fix_idx(0, [1, 0])
        self.mi = [
            self.add_var(f"mass{i}", value=mi, fix=True)
            for i, mi in enumerate(self.mass_list)
        ]

    def __call__(self, m):
        return self.get_ls_amp(m)

    def get_ls_amp(self, m):
        s = m * m
        qi = tf.stack(
            [get_relative_p(m, m1, m2) for m1, m2 in self.decay_list], axis=-1
        )
        rho = qi / m[:, None]
        n2 = self.build_barrier_factor(s)
        K = self.build_k_matrix(s)
        P = self.build_p_vector(s)
        K_i_rho_n2 = K * tf.cast(
            (rho * n2**2)[..., None, :], K.dtype
        )  # np.einsum("...ij,...j->...ij", K, rho * n2**2)
        dom = (
            tf.cast(tf.eye(self.n_channel), K.dtype) - 1.0j * K_i_rho_n2
        )  # .einsum("...ij,...jk->...ik", np.eye(self.n_channel) * rho[:,:,None], K)
        k_inv = tf.linalg.inv(dom)
        ret = tf.reduce_sum(k_inv * P[..., None, :], axis=-1) * tf.cast(
            n2, P.dtype
        )
        return tf.stack([ret[..., i] for i in self.index_list], axis=-1)

    def build_barrier_factor(self, s):
        ret = []
        for (m1, m2), l in zip(self.decay_list, self.l_list):
            ret.append(barrier_factor(np.sqrt(s), m1, m2, l))
        return tf.stack(ret, axis=-1)

    def build_k_matrix(self, s):
        K = []
        for i in range(self.n_channel):
            for j in range(self.n_channel):
                tmp = 0
                for k in range(self.n_pole):  # self.mass_list:
                    mi = self.mi[k]()  # self.mass_list[k]
                    tmp = tmp + tf.cast(
                        self.coeffs()[i][k] * self.coeffs()[j][k],
                        tf.complex128,
                    ) / (
                        tf.cast(mi**2 - s, tf.complex128)
                        - 1.0j * self._epsilon
                    )
                K.append(tmp)
        ret = tf.reshape(
            tf.stack(K, axis=-1), (-1, self.n_channel, self.n_channel)
        )
        return ret

    def build_p_vector(self, s):
        P = []
        for i in range(self.n_channel):
            tmp = 0
            for k in range(self.n_pole):  # self.mass_list:
                mi = self.mi[k]()  # self.mass_list[k]
                tmp = tmp + self.beta()[k] * tf.cast(
                    self.coeffs()[i][k], tf.complex128
                ) / (
                    tf.cast(mi**2 - s, tf.complex128) - 1.0j * self._epsilon
                )
            P.append(tmp)
        # print(self.bkg.vm.variables)
        return tf.reshape(
            tf.stack(P, axis=-1), (-1, self.n_channel)
        ) + tf.stack(self.bkg())

    def phsp_fractor(self, m, m1, m2):
        q = get_relative_p(m, m1, m2)
        return q / m
