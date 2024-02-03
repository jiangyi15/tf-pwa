from math import pi

import numpy as np
import tensorflow as tf

from .angle import LorentzVector


def get_p(M, ma, mb):
    m2 = M * M
    m_p = (ma + mb) ** 2
    m_m = (ma - mb) ** 2
    p2 = (m2 - m_p) * (m2 - m_m)
    p = tf.where(p2 <= 0, tf.zeros_like(p2), p2)
    p = tf.cast(p, tf.float64)
    ret = tf.sqrt(p) / (2.0 * tf.cast(M, p.dtype))
    return ret


class UniformGenerator:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def generate(self, N):
        random = tf.random.uniform([N], dtype="float64")
        ms = (self.b - self.a) * random + self.a
        return ms


class PhaseSpaceGenerator(object):
    """Phase Space Generator for n-body decay"""

    def __init__(self, m0, mass):
        self.m_mass = []
        self.set_decay(m0, mass)
        self.sum_mass = sum(self.m_mass)
        self.mass_range = self.get_mass_range()
        self.mass_generator = [None for i in self.mass_range]

    def get_mass_range(self):
        sm = self.sum_mass - self.m_mass[-1] - self.m_mass[-2]
        m_n = self.m_mass[-1]
        ret = []
        for i in range(self.m_nt - 2):
            b = self.m0 - sm
            a = m_n + self.m_mass[-i - 2]
            ret.append((a, b))
            # random = tf.random.uniform([n_iter], dtype="float64")
            # ms = (b - a) * random + a
            m_n = a  # ms
            sm = sm - self.m_mass[-i - 3]
            # ret.append(ms)
        return ret

    def generate_mass(self, n_iter):
        """generate possible inner mass."""
        sm = self.sum_mass - self.m_mass[-1] - self.m_mass[-2]
        m_n = self.m_mass[-1]
        ret = []
        for i in range(self.m_nt - 2):
            b = self.m0 - sm
            a = m_n + self.m_mass[-i - 2]
            if self.mass_generator[i] is None:
                random = tf.random.uniform([n_iter], dtype="float64")
                ms = (b - a) * random + a
            else:
                ms = self.mass_generator[i].generate(n_iter)
            # print("a", n_iter, a, b, tf.reduce_min(ms),tf.reduce_max(ms))
            m_n = ms
            sm = sm - self.m_mass[-i - 3]
            ret.append(ms)
        return ret

    def mass_importances(self, mass):
        """generate possible inner mass."""
        sm = self.sum_mass - self.m_mass[-1] - self.m_mass[-2]
        m_n = self.m_mass[-1]
        w = 1.0
        for i in range(self.m_nt - 2):
            b = self.m0 - sm
            a = m_n + self.m_mass[-i - 2]
            ms = mass[i]
            if i >= 1 and self.mass_generator[i] is None:
                w = w * (b - a) / (b - self.mass_range[i][0])
            else:
                pass  # ms = self.mass_generator[i].generate(n_iter)
            # print("a", n_iter, a, b, tf.reduce_min(ms),tf.reduce_max(ms))
            m_n = ms
            sm = sm - self.m_mass[-i - 3]
        return w

    def generate(
        self, n_iter: int, force=True, flatten=True, importances=True
    ) -> list:
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
            weight = self.get_weight(mass, importances=importances)
            return weight, pi

        mass_f = self.flatten_mass(mass, importances=importances)
        n_gen += int(mass_f[0].shape[0])

        # loop until number of generated events above required
        while force and n_gen < n_iter:
            # guess the total events required
            n_iter2 = int(1.01 * (n_total - n_gen) / (n_gen + 1) * n_iter)
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
        p_list = []
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
        if len(p_list) == 0:
            ret.append(LorentzVector.neg(p_boost))
        for i in p_list:
            ret.append(LorentzVector.rest_vector(p_boost, i))
        return ret

    def flatten_mass(self, ms, importances=True):
        """sampling from mass with weight"""
        weight = self.get_weight(ms, importances=importances)
        rnd = tf.random.uniform(weight.shape, dtype="float64")
        select = weight > rnd
        return [tf.boolean_mask(i, select) for i in ms]

    def get_weight(self, ms, importances=True):
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
        ret = wt / self.m_wtMax
        if importances:
            return self.mass_importances(ms) * ret
        return ret

    def cal_max_weight(self):
        if len(self.mass_range) == 0:
            pass

        def f(x):
            return float(-self.get_weight(x))

        old_gen = self.mass_generator
        self.mass_generator = [None for i in old_gen]
        x0 = self.generate_mass(1)
        x0 = np.stack([i.numpy()[0] for i in x0])

        self.mass_generator = old_gen
        from scipy.optimize import minimize

        ret = minimize(f, np.array(x0), bounds=self.mass_range)
        self.m_wtMax *= (-ret.fun) * 1.001
        return self.m_wtMax

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


class ChainGenerator:
    """

    struct = m0 -> [m1, m2, m3]  # (m0, [m1, m2, m3])
    m0 -> float
    mi -> float | struct

    """

    def __init__(self, m0, mi):
        struct = (m0, mi)
        self.struct = struct
        self.idxs, self.gen = _get_generator(struct)
        self.unpack_map = {}

    def generate(self, N):
        pi = [i.generate(N) for i in self.gen]
        return _restruct_pi(self.struct, self.idxs, pi)

    def get_gen(self, idx_gen):
        for i, d in enumerate(self.idxs):
            if d == idx_gen:
                return self.gen[i]

    def cal_max_weight(self):
        for i in self.gen:
            i.cal_max_weight()


def _get_generator(struct):
    ret = []
    idxs = []

    # depth first flatten
    def _flatten(s, idx=()):
        m0, mi = s
        mi_real = []
        for i, m_i in enumerate(mi):
            if isinstance(m_i, (tuple, list)):
                tmp = m_i[0]
                _flatten(m_i, idx=(*idx, i))
            else:
                tmp = m_i
            mi_real.append(tmp)
        ret.append((m0, mi_real))
        idxs.append(idx)

    _flatten(struct)

    gen = [PhaseSpaceGenerator(m0, mi) for m0, mi in ret]
    return idxs, gen


def _restruct_pi(struct, idxs, pi_s):
    def empty_pi(struct):
        m0, mi = struct
        mi_real = []
        for i, m_i in enumerate(mi):
            if isinstance(m_i, (tuple, list)):
                tmp = empty_pi(m_i)
            else:
                tmp = [None]
            mi_real.append(tmp)
        return [[None], mi_real]

    ret = empty_pi(struct)

    def loop_index(tree, idx):
        for i in idx:
            tree = tree[1][i]
        return tree

    iters = list(zip(idxs, pi_s))
    for idx, pi in iters[::-1]:
        head, tree = loop_index(ret, idx)
        for i, j in zip(tree, pi):
            if isinstance(i[0], list):
                i[0][0] = j
            else:
                i[0] = j

    def tree_boost(p0, tree):
        if isinstance(tree, list):
            return [tree_boost(p0, i) for i in tree]
        if p0 is not None:
            return LorentzVector.rest_vector(LorentzVector.neg(p0), tree)
        return tree

    for idx in idxs:
        all_tree = loop_index(ret, idx)
        head, tree = all_tree
        all_tree[1] = tree_boost(head[0], tree)
        # print(all_tree[0], sum(i[0] if not isinstance(i[0],list) else i[0][0] for i in all_tree[1]))

    def strip_tree(tree):
        if len(tree) == 1:
            return tree[0]
        m1, mi = tree
        return [strip_tree(i) for i in mi]

    return strip_tree(ret)


def generate_phsp(m0, mi, N=1000):
    """general method to generate decay chain phase sapce
    >>> (a, b), c = generate_phsp(1.0, (
    ...                                 (0.3, (0.1, 0.1)),
    ...                                  0.2),
    ...                            N = 10)
    >>> assert np.allclose(LorentzVector.M(a+b+c), 1.0)

    """
    return ChainGenerator(m0, mi).generate(N)


def square_dalitz_cut(p):
    """Copy from EvtGen old version"""
    p1, p2, p3 = p

    m0 = LorentzVector.M(p1 + p2 + p3)
    m1 = LorentzVector.M(p1)
    m2 = LorentzVector.M(p2)
    m3 = LorentzVector.M(p3)

    m12 = LorentzVector.M(p1 + p2)
    m23 = LorentzVector.M(p2 + p3)
    m13 = LorentzVector.M(p1 + p3)

    m12norm = 2 * ((m12 - (m1 + m2)) / (m0 - (m1 + m2 + m3))) - 1
    mPrime = tf.math.acos(m12norm) / np.pi
    thPrime = (
        tf.math.acos(
            (
                m12 * m12 * (m23 * m23 - m13 * m13)
                - (m2 * m2 - m1 * m1) * (m0 * m0 - m3 * m3)
            )
            / (
                tf.sqrt(
                    (m12 * m12 + m1 * m1 - m2 * m2) ** 2
                    - 4 * m12 * m12 * m1 * m1
                )
                * tf.sqrt(
                    (-m12 * m12 + m0 * m0 - m3 * m3) ** 2
                    - 4 * m12 * m12 * m3 * m3
                )
            )
        )
        / np.pi
    )
    p1st = tf.sqrt(
        (-m12 * m12 - m1 * m1 + m2 * m2) ** 2 - 4 * m12 * m12 * m1 * m1
    ) / (2 * m12)
    p3st = tf.sqrt(
        (-m12 * m12 + m0 * m0 - m3 * m3) ** 2 - 4 * m12 * m12 * m3 * m3
    ) / (2 * m12)
    jacobian = (
        2
        * np.pi**2
        * tf.sin(np.pi * thPrime)
        * tf.sin(np.pi * thPrime)
        * p1st
        * p3st
        * m12
        * (m0 - (m1 + m2 + m3))
    )

    prob = 1 / jacobian

    return tf.where(prob < 1.0, prob, tf.ones_like(prob))


def generate_square_dalitz12(m0, mi, N=1000):
    gen = PhaseSpaceGenerator(m0, mi)
    from tf_pwa.generator.generator import multi_sampling

    return multi_sampling(gen.generate, square_dalitz_cut, N=N, max_weight=1)[
        0
    ]
