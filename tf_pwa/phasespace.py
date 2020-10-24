from math import pi

import numpy as np
import tensorflow as tf

from .angle import LorentzVector


def get_p(M, ma, mb):
    m2 = M * M
    m_p = (ma + mb) ** 2
    m_m = (ma - mb) ** 2
    p2 = (m2 - m_p) * (m2 - m_m)
    p = (p2 + tf.abs(p2)) / 2
    ret = tf.sqrt(p) / (2.0 * M)
    return tf.cast(ret, "float64")


class PhaseSpaceGenerator(object):
    """Phase Space Generator for n-body decay"""

    def __init__(self, m0, mass):
        self.m_mass = []
        self.set_decay(m0, mass)
        self.sum_mass = sum(self.m_mass)

    def generate_mass(self, n_iter):
        """generate possible inner mass."""
        sm = self.sum_mass - self.m_mass[-1] - self.m_mass[-2]
        m_n = self.m_mass[-1]
        ret = []
        for i in range(self.m_nt - 2):
            b = self.m0 - sm
            a = m_n + self.m_mass[-i - 2]
            random = tf.random.uniform([n_iter], dtype="float64")
            ms = (b - a) * random + a
            m_n = ms
            sm = sm - self.m_mass[-i - 3]
            ret.append(ms)
        return ret

    def generate(self, n_iter: int, force=True, flatten=True) -> list:
        """generate `n_iter` events

        :param n_iter: number of events
        :param force: switch for cutting generated data to required size
        :param flatten: switch for sampling with weights

        :return:   daughters 4-momentum, list of ndarray with shape (n_iter, 4)
        """
        n_gen = 0
        n_total = n_iter

        mass = self.generate_mass(n_iter)
        if not flatten or self.m_nt == 2:
            pi = self.generate_momentum(mass, n_iter)
            if flatten:
                return pi
            weight = self.get_weight(mass)
            return weight, pi

        mass_f = self.flatten_mass(mass)
        n_gen += mass_f[0].shape[0]

        # loop until number of generated events above required
        while force and n_gen < n_iter:
            n_iter2 = int(
                1.01 * (n_total - n_gen) / (n_gen + 1) * n_iter
            )  # guess the total events required
            n_iter2 = min(n_iter2, 4000000)
            mass2 = self.generate_mass(n_iter2)
            mass_f2 = self.flatten_mass(mass2)
            n_gen += mass_f2[0].shape[0]
            n_total += n_iter2
            mass_f = [tf.concat([i, j], 0) for i, j in zip(mass_f, mass_f2)]

        if force:
            mass_f = [i[:n_iter] for i in mass_f]
        return self.generate_momentum(mass_f)

    def generate_momentum(self, mass, n_iter=None):
        """generate random momentum from mass, boost them to a same rest frame"""
        if n_iter is None:
            n_iter = mass[0].shape[0]
        mass_t = [self.m_mass[-1]]
        for i in mass:
            mass_t.append(i)
        mass_t.append(self.m0)
        zeros = tf.zeros([n_iter], dtype="float64")
        p_list = [
            tf.stack([zeros + self.m_mass[-1], zeros, zeros, zeros], axis=-1)
        ]
        for i in range(0, self.m_nt - 1):
            p_list = self.generate_momentum_i(
                mass_t[i + 1], mass_t[i], self.m_mass[-i - 2], n_iter, p_list
            )

        return p_list

    def generate_momentum_i(self, m0, m1, m2, n_iter, p_list=[]):
        """
        :math:`|p|` =  m0,m1,m2 in m0 rest frame
        :param p_list: extra list for momentum need to boost
        """
        # random angle
        cos_theta = 2 * tf.random.uniform([n_iter], dtype="float64") - 1
        sin_theta = tf.sqrt(1 - cos_theta * cos_theta)
        phi = 2 * pi * tf.random.uniform([n_iter], dtype="float64")
        # 4-momentum
        q = tf.broadcast_to(get_p(m0, m1, m2), phi.shape)
        p_0 = tf.sqrt(q * q + m2 * m2)
        p_x = q * sin_theta * tf.cos(phi)
        p_y = q * sin_theta * tf.sin(phi)
        p_z = q * cos_theta

        p = tf.stack([p_0, p_x, p_y, p_z], axis=-1)
        ret = [p]
        # recoil momentum
        p_boost = tf.stack([tf.sqrt(q * q + m1 * m1), p_x, p_y, p_z], axis=-1)
        for i in p_list:
            ret.append(LorentzVector.rest_vector(p_boost, i))
        return ret

    def flatten_mass(self, ms):
        """sampling from mass with weight"""
        weight = self.get_weight(ms)
        rnd = tf.random.uniform(weight.shape, dtype="float64")
        select = weight > rnd
        return [tf.boolean_mask(i, select) for i in ms]

    def get_weight(self, ms):
        r"""calculate weight of mass

        .. math::
            w = \frac{1}{w_{max}} \frac{1}{M}\prod_{i=0}^{n-2} q(M_i,M_{i+1},m_{i+1})

        """
        mass_t = [self.m_mass[-1]]
        for i in ms:
            mass_t.append(i)
        mass_t.append(self.m0)
        R = []
        for i in range(self.m_nt - 1):
            p = get_p(mass_t[i + 1], mass_t[i], self.m_mass[-i - 2])
            R.append(p)
        wt = tf.math.reduce_prod(tf.stack(R), 0)
        return wt / self.m_wtMax

    def set_decay(self, m0, mass):
        r"""set decay mass, calculate max weight

        .. math::
            w_{max} = \frac{1}{M}\prod_{i=0}^{n-2} q(max(M_i),min(M_{i+1}),m_{i+1})

        .. math::
            max(M_i) = M_0 - \sum_{j=1}^{i} (m_j)

        .. math::
            min(M_i) = \sum_{j=i}^{n} (m_j)

        """
        self.m0 = m0
        self.m_nt = len(mass)
        self.m_teCmTm = m0
        for i in mass:
            self.m_mass.append(i)
            self.m_teCmTm = self.m_teCmTm - i

        if self.m_teCmTm <= 0:
            raise ValueError("mass is not validated.")
        emmax = self.m_teCmTm + self.m_mass[-1]
        emmin = 0
        wtmax = 1
        for n in range(1, self.m_nt):
            emmin += self.m_mass[-n]
            emmax += self.m_mass[-n - 1]
            p = get_p(emmax, emmin, self.m_mass[-n - 1])
            wtmax *= p
        self.m_wtMax = tf.convert_to_tensor(wtmax, dtype="float64")
